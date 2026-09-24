"""ORIC PDXC Piezo Stage Controller — serial, server, and client.

Serial settings: 115200 baud, 8N1, no flow control (PDXC manual Ch. 6).

Wire protocol (TCP, newline-terminated JSON):
  -> {"method": "move_in", "args": {}}
  <- {"ok": true, "result": "done"}

Server-level methods (answered without taking the device lock, so they never
queue behind a move):
  -> {"method": "get_snapshot"}   <- {"ok": true, "result": {"com": {...}, ...}}
  -> {"method": "shutdown"}       <- {"ok": true, "result": "shutting down"}
     then the server closes the COM port, stops its beacon and exits 0.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Optional

import serial

from beacon.discovery.client import NetClient
from beacon.discovery.server import NetServer

logger = logging.getLogger(__name__)

SERVER_ID = "pdxc"
_BAUD = 115200
_TIMEOUT_S = 2.0
_MOVE_MARGIN_S = 0.3        # settle margin added to each move's computed duration
_MOVE_TCP_TIMEOUT = 290.0   # client socket timeout for a full move
_DEVICE_TCP_TIMEOUT = 10.0  # client socket timeout for other device methods
                            # (they wait on the device lock, i.e. behind a move)
_CLIENT_TIMEOUT_S = 2.0     # connect + server-level methods (get_snapshot, shutdown)
_MOVE_METHODS = ("move_in", "move_out", "move_to")   # long-running: move lock
_SERVER_METHODS = ("get_snapshot", "shutdown")       # answered by the server itself

# Grace period between answering "shutdown" and tearing down, so the reply is
# on the wire before the listening socket closes.
_SHUTDOWN_REPLY_GRACE_S = 0.2
# If start() has not returned this long after stop(), force the exit.
_SHUTDOWN_FORCE_EXIT_S = 1.5

MAX_PULSES = 65535          # MOVF/MOVB pulse-count ceiling (manual 6.3.24)
MIN_PULSES = 1
_FRQ_MIN_HZ = 800           # SMC drive-frequency range (manual 6.3.6)
_FRQ_MAX_HZ = 20000

# Named stage positions.  The stage is open-loop with no encoder, so "where
# it is" can only ever be the last position it was *commanded* to.  Jogs
# (move_in / move_out / move_forward / move_backward) invalidate that back to
# UNKNOWN; only move_to(), which drives the full throw into the end stop,
# establishes a known position.
POSITION_IN = "in"            # beamsplitter inserted: light to APD, camera blocked
POSITION_OUT = "out"          # beamsplitter retracted: camera clear
POSITION_UNKNOWN = "unknown"
_POSITIONS = (POSITION_IN, POSITION_OUT)

# move_to() drives a fixed number of fixed-length moves into the end stop.
# The throw is asymmetric: "in" takes three moves, "out" one.  Overdriving is
# harmless (the stage slips at the stop), underdriving leaves it short.
_DEFAULT_THROW_PULSES = 40000     # pulses per move
_DEFAULT_MOVES = {POSITION_IN: 3, POSITION_OUT: 1}
_MAX_THROW_MOVES = 10

# Step size, throw length and last commanded position are persisted on the
# *server* host so they survive server restarts and greet every client.
_DEFAULTS_FILE = os.path.join(
    os.path.expanduser("~"), ".waxx", "pdxc_server_defaults.json"
)

# ERR? codes (manual 6.3.20).  The register clears on each query.
_ERROR_MESSAGES = {
    0: "no error",
    1: "command not defined",
    2: "data out-of-range",
    3: "failed to execute last command",
    4: "no waveform data loaded",
    5: "need home first",
    6: "device works in wrong mode",
    7: "stage move abnormal",
    8: "over current",
    9: "over temperature",
    10: "wrong stage detected",
}


# ---------------------------------------------------------------------------
# Hardware class
# ---------------------------------------------------------------------------

class PDXC:
    """Direct RS-232/USB control of the ORIC PDXC Piezo Stage Controller.

    All commands are CR-terminated per the PDXC manual.  After each command
    the controller emits a '>' prompt; after each query it emits the value
    followed by CR then '>'.
    """

    def __init__(self, port: str = "COM26", baudrate: int = _BAUD,
                 timeout: float = _TIMEOUT_S) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self._last_rx_monotonic: Optional[float] = None   # last non-empty reply
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )
        self._freq_hz: int = _FRQ_MAX_HZ   # refreshed by initialize()
        _defaults = self._load_defaults()
        self._step_size: int = self._sane(
            _defaults.get("step_size"), MIN_PULSES, MAX_PULSES, MAX_PULSES, "step_size")
        self._throw_pulses: int = self._sane(
            _defaults.get("throw_pulses"), MIN_PULSES, MAX_PULSES,
            _DEFAULT_THROW_PULSES, "throw_pulses")
        _moves = _defaults.get("throw_moves")
        if not isinstance(_moves, dict):
            _moves = {}
        self._throw_moves: dict = {
            state: self._sane(_moves.get(state), 1, _MAX_THROW_MOVES, default,
                              f"throw_moves[{state}]")
            for state, default in _DEFAULT_MOVES.items()
        }
        _pos = _defaults.get("position")
        self._position: str = _pos if _pos in _POSITIONS else POSITION_UNKNOWN
        logger.info("PDXC connected on %s at %d baud", port, baudrate)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sane(value, lo: int, hi: int, fallback: int, name: str) -> int:
        """Coerce a persisted setting, falling back if absent or out of range."""
        try:
            value = int(value)
        except (TypeError, ValueError):
            return fallback
        if lo <= value <= hi:
            return value
        logger.warning("[PDXC] Ignoring out-of-range stored %s: %s", name, value)
        return fallback

    def close(self) -> None:
        if self._ser.is_open:
            self._ser.close()

    @property
    def is_open(self) -> bool:
        try:
            return bool(self._ser.is_open)
        except Exception:
            return False

    @property
    def last_rx_seconds_ago(self) -> Optional[float]:
        t = self._last_rx_monotonic
        return None if t is None else time.monotonic() - t

    def _write(self, cmd: str) -> str:
        """Send a command; return the reply text up to the '>' ready-prompt.

        The controller reports failures in-band as ``CMD_NOT_DEFINED`` or
        ``Data Out-Of-Range!`` (manual 6.1), so those are printed and raised
        here rather than discarded.
        """
        self._ser.reset_input_buffer()
        if not cmd.endswith("\r"):
            cmd += "\r"
        self._ser.write(cmd.encode("ascii"))
        resp = self._ser.read_until(b">").decode("ascii", errors="ignore")
        if resp:
            self._last_rx_monotonic = time.monotonic()
        if "CMD_NOT_DEFINED" in resp or "Out-Of-Range" in resp:
            msg = f"PDXC rejected {cmd.strip()!r}: {resp.strip()}"
            print(msg)
            logger.error(msg)
            raise RuntimeError(msg)
        return resp.strip()

    def _query(self, cmd: str) -> str:
        """Send a query; return the value string.

        Reads to the '>' ready-prompt and strips it plus any CR/LF framing or
        command echo.  Some firmware revisions terminate the value with CR
        before the prompt, others emit the prompt on the same line ("1>"), so
        keying off the prompt alone handles both without a read timeout.

        Decodes with errors='ignore' to discard any non-ASCII framing bytes
        (e.g. 0xff) that some PDXC firmware revisions echo before the response.

        Single-line replies only; not usable for POSP? (8192-row dump).
        """
        self._ser.reset_input_buffer()
        if not cmd.endswith("\r"):
            cmd += "\r"
        self._ser.write(cmd.encode("ascii"))
        raw = self._ser.read_until(b">").decode("ascii", errors="ignore")
        if raw:
            self._last_rx_monotonic = time.monotonic()
        body = raw.split(">")[0].replace("\r", "\n")
        lines = [ln.strip() for ln in body.split("\n")]
        lines = [ln for ln in lines if ln and ln != cmd.strip()]
        return lines[-1] if lines else ""

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_smc_mode(self) -> str:
        """Switch output to the SMC connector (SW=1).

        Takes at least 1 s to execute (manual 6.3.23); callers must wait
        before issuing further commands.
        """
        self._write("SW=1")
        return "ok"

    def get_output_mode(self) -> str:
        """Query the output mode (SW?): '0' = D-Sub, '1' = SMC."""
        return self._query("SW?")

    def set_frequency(self, hz: int, channel: int = 0) -> str:
        """Set the SMC drive frequency in steps/s (FRQ= CH1, FRQ2= CH2).

        This, not SPD, is the speed control in SMC mode.  SPD is the D-Sub
        speed in mm/s and accepts only 2-20 (manual 6.3.5 vs 6.3.6/6.3.7).
        """
        if not _FRQ_MIN_HZ <= hz <= _FRQ_MAX_HZ:
            raise ValueError(
                f"SMC frequency must be {_FRQ_MIN_HZ}-{_FRQ_MAX_HZ} Hz, got {hz}"
            )
        self._write(f"FRQ2={hz}" if channel else f"FRQ={hz}")
        return "ok"

    def get_frequency(self, channel: int = 0) -> int:
        """Query the SMC drive frequency in steps/s (FRQ? / FRQ2?)."""
        raw = self._query("FRQ2?" if channel else "FRQ?")
        try:
            return int(raw.strip())
        except ValueError as exc:
            raise RuntimeError(f"unexpected FRQ? reply: {raw!r}") from exc

    def check_error(self) -> int:
        """Query the error register (ERR?) and print any fault to the terminal.

        Returns the numeric code (0 = no error, -1 = unparseable reply).  The
        register clears on every query (manual 6.3.20).
        """
        raw = self._query("ERR?")
        try:
            code = int(raw.strip())
        except ValueError:
            msg = f"PDXC ERR? returned unparseable reply: {raw!r}"
            print(msg)
            logger.warning(msg)
            return -1
        if code:
            msg = f"PDXC error {code}: {_ERROR_MESSAGES.get(code, 'unknown code')}"
            print(msg)
            logger.error(msg)
        return code

    def initialize(self, freq_hz: int = _FRQ_MAX_HZ, channel: int = 0) -> str:
        """Switch to SMC output and set the drive frequency for *channel*.

        No loop-mode command is sent: SMC stages carry no encoder, so the
        controller is open-loop by construction (manual 4.4 / 5.3) and LP=
        applies to D-Sub operation only.
        """
        self.check_error()                     # clear any stale code
        self.set_smc_mode()
        time.sleep(1.5)                        # SW= needs >=1 s (manual 6.3.23)
        mode = self.get_output_mode().strip()
        if mode != "1":
            raise RuntimeError(f"PDXC failed to switch to SMC output (SW? -> {mode!r})")
        self.set_frequency(freq_hz, channel)
        readback = self.get_frequency(channel)
        if not _FRQ_MIN_HZ <= readback <= _FRQ_MAX_HZ:
            raise RuntimeError(f"implausible FRQ? readback: {readback} Hz")
        if readback != freq_hz:
            print(f"PDXC frequency readback {readback} Hz != requested {freq_hz} Hz")
        self._freq_hz = readback
        print(f"PDXC ready: SMC output, CH{channel + 1} at {self._freq_hz} steps/s")
        self.check_error()
        return "ok"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self) -> float:
        """Return current position in mm, or NaN on parse failure."""
        try:
            return float(self._query("POS?"))
        except ValueError:
            return float("nan")

    def get_serial(self) -> str:
        """Return the PDXC controller serial number."""
        return self._query("SN?")

    def get_stage_serial(self) -> str:
        """Return the connected stage serial number (SN2?)."""
        return self._query("SN2?")

    def get_firmware(self) -> str:
        """Return firmware and hardware version string (FV?)."""
        return self._query("FV?")

    # ------------------------------------------------------------------
    # Persisted settings (server-side)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_defaults() -> dict:
        """Read the persisted settings file; {} if missing or unreadable."""
        try:
            with open(_DEFAULTS_FILE) as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("[PDXC] Could not read defaults file: %s", exc)
        return {}

    def _save_defaults(self) -> None:
        """Persist the current settings to disk (best effort)."""
        try:
            os.makedirs(os.path.dirname(_DEFAULTS_FILE), exist_ok=True)
            with open(_DEFAULTS_FILE, "w") as fh:
                json.dump({"step_size": self._step_size,
                           "throw_pulses": self._throw_pulses,
                           "throw_moves": self._throw_moves,
                           "position": self._position}, fh, indent=2)
        except Exception as exc:
            logger.warning("[PDXC] Could not write defaults file: %s", exc)

    def get_step_size(self) -> int:
        """Return the default pulse count used by move_in / move_out."""
        return self._step_size

    def set_step_size(self, steps: int) -> int:
        """Set and persist the default pulse count for move_in / move_out."""
        steps = int(steps)
        if not MIN_PULSES <= steps <= MAX_PULSES:
            raise ValueError(f"steps must be {MIN_PULSES}-{MAX_PULSES}, got {steps}")
        self._step_size = steps
        self._save_defaults()
        return self._step_size

    def get_throw_pulses(self) -> int:
        """Return the pulse count of each move issued by move_to()."""
        return self._throw_pulses

    def set_throw_pulses(self, pulses: int) -> int:
        """Set and persist the per-move pulse count used by move_to()."""
        pulses = int(pulses)
        if not MIN_PULSES <= pulses <= MAX_PULSES:
            raise ValueError(f"throw must be {MIN_PULSES}-{MAX_PULSES}, got {pulses}")
        self._throw_pulses = pulses
        self._save_defaults()
        return self._throw_pulses

    def get_throw_moves(self, state: str) -> int:
        """Return how many moves move_to() issues to reach *state*."""
        return self._throw_moves[self._valid_state(state)]

    def set_throw_moves(self, state: str, moves: int) -> int:
        """Set and persist how many moves move_to() issues to reach *state*."""
        state = self._valid_state(state)
        moves = int(moves)
        if not 1 <= moves <= _MAX_THROW_MOVES:
            raise ValueError(f"moves must be 1-{_MAX_THROW_MOVES}, got {moves}")
        self._throw_moves[state] = moves
        self._save_defaults()
        return self._throw_moves[state]

    @staticmethod
    def _valid_state(state: str) -> str:
        state = str(state).lower()
        if state not in _POSITIONS:
            raise ValueError(f"position must be one of {_POSITIONS}, got {state!r}")
        return state

    def get_position_state(self) -> str:
        """Return the last commanded position: 'in', 'out' or 'unknown'."""
        return self._position

    def set_position_state(self, state: str) -> str:
        """Declare the stage position without moving it.

        For recovering the tracked state after the stage has been moved by
        hand, or for asserting a known starting position.
        """
        state = str(state).lower()
        if state not in _POSITIONS + (POSITION_UNKNOWN,):
            raise ValueError(f"position must be one of {_POSITIONS}, got {state!r}")
        self._mark_position(state)
        return self._position

    def _mark_position(self, state: str) -> None:
        if state != self._position:
            self._position = state
            self._save_defaults()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _move_one(self, direction: str, steps: int, channel: int) -> None:
        """Issue a single MOVF or MOVB command (non-blocking on device)."""
        if not 1 <= steps <= MAX_PULSES:
            raise ValueError(f"steps must be 1-{MAX_PULSES}, got {steps}")
        self._write(f"{direction}={steps}.{channel}")

    def _wait_stopped(self, steps: int) -> bool:
        """Block for the expected duration of a *steps*-pulse move.

        SMC stages have no encoder, so POS? returns nothing usable and
        completion cannot be polled.  The wait is steps / drive-frequency
        plus a settle margin.
        """
        time.sleep(steps / self._freq_hz + _MOVE_MARGIN_S)
        return True

    def move_forward(self, steps: int, channel: int = 0) -> str:
        """Single MOVF command (non-blocking, fire-and-forget)."""
        self._mark_position(POSITION_UNKNOWN)
        self._move_one("MOVF", steps, channel)
        return "ok"

    def move_backward(self, steps: int, channel: int = 0) -> str:
        """Single MOVB command (non-blocking, fire-and-forget)."""
        self._mark_position(POSITION_UNKNOWN)
        self._move_one("MOVB", steps, channel)
        return "ok"

    def move_in(self, steps: Optional[int] = None, channel: int = 0) -> str:
        """Move the beamsplitter in. Blocks until the move should be done.

        *steps* defaults to the persisted step size.  "In" drives MOVB: the
        stage travel direction is inverted relative to the controller sense.
        """
        steps = self._step_size if steps is None else int(steps)
        self._mark_position(POSITION_UNKNOWN)   # a jog of unknown extent
        self._move_one("MOVB", steps, channel)
        self._wait_stopped(steps)
        self.check_error()
        return "done"

    def move_out(self, steps: Optional[int] = None, channel: int = 0) -> str:
        """Move the beamsplitter out. Blocks until the move should be done.

        *steps* defaults to the persisted step size.  "Out" drives MOVF; see
        move_in for the direction inversion.
        """
        steps = self._step_size if steps is None else int(steps)
        self._mark_position(POSITION_UNKNOWN)   # a jog of unknown extent
        self._move_one("MOVF", steps, channel)
        self._wait_stopped(steps)
        self.check_error()
        return "done"

    def move_to(self, state: str, channel: int = 0, force: bool = False) -> str:
        """Drive the stage to the named end stop and remember it got there.

        Unlike the move_in / move_out jogs this drives the full throw so the
        stage lands against its mechanical stop regardless of where it
        started: ``get_throw_moves(state)`` moves of ``get_throw_pulses()``
        pulses each.  The throw is asymmetric -- "in" takes three moves,
        "out" one -- so the two directions are configured separately.

        Returns immediately with "already <state>" when the stage was last
        commanded there, so calling this at the top of every experiment costs
        nothing after the first run.  Pass ``force=True`` to move anyway.
        """
        state = self._valid_state(state)
        if self._position == state and not force:
            return f"already {state}"

        direction = "MOVB" if state == POSITION_IN else "MOVF"
        steps = self._throw_pulses
        self._mark_position(POSITION_UNKNOWN)   # mid-throw: neither end
        for _ in range(self._throw_moves[state]):
            self._move_one(direction, steps, channel)
            self._wait_stopped(steps)
        self.check_error()
        self._mark_position(state)
        return f"moved {state}"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class PDXC_Server(NetServer):
    """TCP server exposing PDXC serial control over LAN.

    Initialises the device (SMC mode + max speed) on ``start()``.
    Only one move runs at a time (_move_lock); all other methods are
    serialised by _device_lock.  Each accepted TCP connection is handled
    on its own daemon thread.
    """

    def __init__(self, com_port: str = "COM26") -> None:
        NetServer.__init__(self, SERVER_ID, port=0)
        self._com_port = com_port
        self._device: Optional[PDXC] = None
        self._running = False
        self._device_lock = threading.Lock()
        self._move_lock = threading.Lock()
        self._srv: Optional[socket.socket] = None
        self._com_status = "disconnected"          # for the "com" snapshot
        self._com_last_error: Optional[str] = None
        self._shutdown_requested = False

    def start(self) -> None:
        self._com_status = "connecting"
        try:
            self._device = PDXC(self._com_port)
            self._device.initialize()
        except Exception as exc:
            self._com_status = "error"
            self._com_last_error = f"{type(exc).__name__}: {exc}"
            raise
        self._com_status = "connected"
        self._com_last_error = None
        self._running = True

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", 0))          # OS assigns a free port
        srv.listen(5)
        srv.settimeout(1.0)
        self._srv = srv
        self._waxx_port = srv.getsockname()[1]   # tell beacon the real port
        self._start_beacon()
        print(f"PDXC server listening on port {self._waxx_port}")

        try:
            while self._running:
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        raise
                    break   # stop() closed the listening socket on purpose
                threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True,
                ).start()
        except KeyboardInterrupt:
            print("\nShutting down PDXC server...")
        finally:
            self._running = False
            self._stop_beacon()
            srv.close()
            self._srv = None
            self._close_device()

    def _close_device(self) -> None:
        """Release the COM port.  Idempotent."""
        device = self._device
        if device is None:
            return
        try:
            device.close()
        except Exception as exc:
            logger.warning("PDXC close raised: %s", exc)
        self._com_status = "disconnected"

    def stop(self) -> None:
        """Stop accepting; start()'s finally then releases beacon and COM port."""
        self._running = False
        # Unblock start()'s accept() right away instead of waiting for its
        # 1 s timeout.
        srv = self._srv
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass

    def request_shutdown(self) -> None:
        """Graceful process shutdown requested over TCP (dashboard Stop).

        Returns immediately so the reply can be sent; a daemon thread then
        waits a short grace period and calls :meth:`stop`, which unblocks the
        accept loop so :meth:`start` runs its ``finally`` (beacon off, COM
        port closed) and returns to the launcher, which exits with code 0.
        If the process is still alive shortly after that, the thread closes
        the device itself and forces ``os._exit(0)``.
        """
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        logger.warning("Process shutdown requested over TCP; stopping PDXC server")
        print("PDXC server: shutdown requested over TCP")

        def _worker() -> None:
            time.sleep(_SHUTDOWN_REPLY_GRACE_S)
            try:
                self.stop()
            except Exception:
                logger.exception("stop() raised during requested shutdown")
            time.sleep(_SHUTDOWN_FORCE_EXIT_S)
            logger.warning("start() did not return after stop(); forcing process exit")
            self._stop_beacon()
            self._close_device()
            logging.shutdown()
            os._exit(0)

        threading.Thread(target=_worker, name="pdxc-shutdown", daemon=True).start()

    def _com_snapshot(self) -> dict:
        """Serial-link summary in ``SerialSnapshot.as_dict()`` shape.

        The PDXC server opens its port once in ``start()`` and has no
        reconnect logic, so ``status`` is: open failed -> error (the process
        then exits anyway); device present and its port open -> connected;
        otherwise disconnected.  Read lock-free so it never queues behind a
        move holding the device lock.
        """
        device = self._device
        if self._com_status == "connected" and (device is None or not device.is_open):
            status = "disconnected"
        else:
            status = self._com_status
        last_rx = device.last_rx_seconds_ago if device is not None else None
        return {
            "port": self._com_port,
            "baud": int(device.baudrate) if device is not None else _BAUD,
            "status": status,
            "last_error": self._com_last_error,
            "last_rx_seconds_ago": None if last_rx is None else round(last_rx, 3),
            "reconnect_attempts": 0,   # no auto-reconnect on this server
            "config_valid": True,
        }

    def get_snapshot(self) -> dict:
        device = self._device
        return {
            "com": self._com_snapshot(),
            "com_port": self._com_port,
            "server_port": self._waxx_port,
            "position": device.get_position_state() if device is not None else None,
            "move_in_progress": self._move_lock.locked(),
        }

    def _handle_client(self, sock: socket.socket, addr) -> None:
        logger.debug("PDXC client connected: %s", addr)
        try:
            data = b""
            while b"\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    return
                data += chunk
            cmd = json.loads(data.split(b"\n")[0])
            resp = self._dispatch(cmd)
            sock.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        except Exception as exc:
            logger.exception("PDXC handler error: %s", exc)
            try:
                sock.sendall(
                    (json.dumps({"ok": False, "error": str(exc)}) + "\n").encode("utf-8")
                )
            except Exception:
                pass
        finally:
            sock.close()

    def _dispatch(self, cmd: dict) -> dict:
        method = cmd.get("method", "")
        args = cmd.get("args", {})
        # Server-level methods: no device lock, so they answer during a move.
        if method == "get_snapshot":
            try:
                return {"ok": True, "result": self.get_snapshot()}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
        if method == "shutdown":
            self.request_shutdown()
            return {"ok": True, "result": "shutting down"}
        if method in _MOVE_METHODS:
            acquired = self._move_lock.acquire(timeout=5.0)
            if not acquired:
                return {"ok": False, "error": "another move is in progress"}
            try:
                with self._device_lock:
                    result = getattr(self._device, method)(**args)
                return {"ok": True, "result": result}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            finally:
                self._move_lock.release()
        else:
            with self._device_lock:
                try:
                    result = getattr(self._device, method)(**args)
                    return {"ok": True, "result": result}
                except Exception as exc:
                    return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PDXC_Client(NetClient):
    """TCP client mirroring the PDXC motion API via ``PDXC_Server``.

    Discovered automatically via UDP broadcast beacon (server_id ``"pdxc"``).
    """

    def __init__(self, discovery_timeout: float = 5.0,
                 timeout: float = _CLIENT_TIMEOUT_S) -> None:
        super().__init__(SERVER_ID, discovery_timeout=discovery_timeout)
        # TCP connect timeout, and the reply timeout for server-level methods
        # (get_snapshot / shutdown).  Device methods keep their own longer
        # reply timeouts because they queue behind moves on the server.
        self.timeout = float(timeout)

    def _send(self, method: str, **kwargs) -> dict:
        payload = (json.dumps({"method": method, "args": kwargs}) + "\n").encode("utf-8")
        if method in _MOVE_METHODS:
            tcp_timeout = _MOVE_TCP_TIMEOUT
        elif method in _SERVER_METHODS:
            tcp_timeout = self.timeout
        else:
            tcp_timeout = _DEVICE_TCP_TIMEOUT
        for attempt in range(2):
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=self.timeout
                ) as sock:
                    sock.settimeout(tcp_timeout)
                    sock.sendall(payload)
                    buf = b""
                    while b"\n" not in buf:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                    return json.loads(buf.split(b"\n")[0])
            except (ConnectionRefusedError, ConnectionResetError,
                    OSError, socket.timeout) as exc:
                if attempt == 0 and self._rediscover(timeout=2.0):
                    continue
                raise RuntimeError(f"PDXC server unreachable: {exc}") from exc

    def _call(self, method: str, **kwargs):
        """Send *method*, raising on a server-reported failure."""
        resp = self._send(method, **kwargs)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "unknown error"))
        return resp["result"]

    def get_snapshot(self) -> dict:
        """Server snapshot dict (``com`` link summary, position, ...).

        Answered by the server without the device lock, so it works during a
        move.  Raises on network failure or a server-reported error.
        """
        result = self._call("get_snapshot")
        return result if isinstance(result, dict) else {"raw": result}

    def request_shutdown(self) -> bool:
        """Ask the server process to exit cleanly (close COM, stop beacon, exit 0)."""
        return bool(self._send("shutdown").get("ok"))

    def move_in(self, steps: Optional[int] = None, channel: int = 0) -> str:
        """Move the beamsplitter in (server step size when *steps* is None)."""
        return self._call("move_in", steps=steps, channel=channel)

    def move_out(self, steps: Optional[int] = None, channel: int = 0) -> str:
        """Move the beamsplitter out (server step size when *steps* is None)."""
        return self._call("move_out", steps=steps, channel=channel)

    def move_to(self, state: str, channel: int = 0, force: bool = False) -> str:
        """Drive the stage to POSITION_IN or POSITION_OUT.

        A no-op when the server already has the stage recorded at *state*.
        """
        return self._call("move_to", state=state, channel=channel, force=force)

    def get_position_state(self) -> str:
        """Return the last commanded position: 'in', 'out' or 'unknown'."""
        return self._call("get_position_state")

    def set_position_state(self, state: str) -> str:
        """Declare the stage position on the server without moving it."""
        return self._call("set_position_state", state=state)

    def get_step_size(self) -> int:
        """Return the step size the server currently has stored."""
        return int(self._call("get_step_size"))

    def set_step_size(self, steps: int) -> int:
        """Set and persist the server-side step size; returns the value set."""
        return int(self._call("set_step_size", steps=int(steps)))

    def get_throw_pulses(self) -> int:
        """Return the full-throw pulse count move_to() uses."""
        return int(self._call("get_throw_pulses"))

    def set_throw_pulses(self, pulses: int) -> int:
        """Set and persist the per-move pulse count used by move_to()."""
        return int(self._call("set_throw_pulses", pulses=int(pulses)))

    def get_throw_moves(self, state: str) -> int:
        """Return how many moves move_to() issues to reach *state*."""
        return int(self._call("get_throw_moves", state=state))

    def set_throw_moves(self, state: str, moves: int) -> int:
        """Set and persist how many moves move_to() issues to reach *state*."""
        return int(self._call("set_throw_moves", state=state, moves=int(moves)))


__all__ = ["PDXC", "PDXC_Server", "PDXC_Client", "SERVER_ID",
           "MIN_PULSES", "MAX_PULSES",
           "POSITION_IN", "POSITION_OUT", "POSITION_UNKNOWN"]
