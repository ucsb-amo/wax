"""ORIC PDXC Piezo Stage Controller — serial, server, and client.

Serial settings: 115200 baud, 8N1, no flow control (PDXC manual Ch. 6).

Wire protocol (TCP, newline-terminated JSON):
  -> {"method": "move_in", "args": {}}
  <- {"ok": true, "result": "done"}
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Optional

import serial

from waxx.util.comms_server.waxx_client import WaxxClient
from waxx.util.comms_server.waxx_server import WaxxServer

logger = logging.getLogger(__name__)

SERVER_ID = "pdxc"
_BAUD = 115200
_TIMEOUT_S = 2.0
_MOVE_WAIT_S = 3.0           # fixed inter-move gap (open-loop; POS? returns NaN)
_MOVE_TCP_TIMEOUT = 290.0   # client socket timeout for a full move_in/move_out


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
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )
        logger.info("PDXC connected on %s at %d baud", port, baudrate)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._ser.is_open:
            self._ser.close()

    def _write(self, cmd: str) -> None:
        """Send a command and consume the trailing '>' ready-prompt."""
        self._ser.reset_input_buffer()
        if not cmd.endswith("\r"):
            cmd += "\r"
        self._ser.write(cmd.encode("ascii"))
        self._ser.read_until(b">")

    def _query(self, cmd: str) -> str:
        """Send a query; return the value string (strips CR and '>' prompt).

        Decodes with errors='ignore' to discard any non-ASCII framing bytes
        (e.g. 0xff) that some PDXC firmware revisions echo before the response.
        """
        self._ser.reset_input_buffer()
        if not cmd.endswith("\r"):
            cmd += "\r"
        self._ser.write(cmd.encode("ascii"))
        raw = self._ser.read_until(b"\r").decode("ascii", errors="ignore").strip()
        self._ser.read_until(b">")   # consume prompt
        return raw

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_smc_mode(self) -> str:
        """Switch output to SMC connector (SW=1)."""
        self._write("SW=1")
        return "ok"

    def get_output_mode(self) -> str:
        """Query current output mode (SW?)."""
        return self._query("SW?")

    def set_open_loop(self) -> str:
        """Set controller to open-loop mode (LP=1)."""
        self._write("LP=1")
        return "ok"

    def get_loop_mode(self) -> str:
        """Query current loop mode (LP?)."""
        return self._query("LP?")

    def set_speed(self, speed: int) -> str:
        """Set stage travel speed (SPD=x)."""
        self._write(f"SPD={speed}")
        return "ok"

    def get_speed(self) -> str:
        """Query current speed setting (SPD?)."""
        return self._query("SPD?")

    def initialize(self, speed: int = 20000) -> str:
        """Switch to SMC mode, open-loop mode, and set speed."""
        self.set_smc_mode()
        time.sleep(0.15)
        self.set_open_loop()
        time.sleep(0.15)
        self.set_speed(speed)
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
    # Motion
    # ------------------------------------------------------------------

    def _move_one(self, direction: str, steps: int, channel: int) -> None:
        """Issue a single MOVF or MOVB command (non-blocking on device)."""
        self._write(f"{direction}={steps}.{channel}")

    def _wait_stopped(self) -> bool:
        """Wait for a move to complete.

        In open-loop SMC mode POS? returns NaN, so position polling is not
        usable.  A fixed sleep is the reliable alternative.
        """
        time.sleep(_MOVE_WAIT_S)
        return True

    def move_forward(self, steps: int, channel: int = 0) -> str:
        """Single MOVF command (non-blocking, fire-and-forget)."""
        self._move_one("MOVF", steps, channel)
        return "ok"

    def move_backward(self, steps: int, channel: int = 0) -> str:
        """Single MOVB command (non-blocking, fire-and-forget)."""
        self._move_one("MOVB", steps, channel)
        return "ok"

    def move_in(self, steps: int = 60000, channel: int = 0) -> str:
        """Two sequential MOVF moves of *steps* each. Blocks until done."""
        self._move_one("MOVF", steps, channel)
        self._wait_stopped()
        self._move_one("MOVF", steps, channel)
        self._wait_stopped()
        return "done"

    def move_out(self, steps: int = 60000, channel: int = 0) -> str:
        """Two sequential MOVB moves of *steps* each. Blocks until done."""
        self._move_one("MOVB", steps, channel)
        self._wait_stopped()
        self._move_one("MOVB", steps, channel)
        self._wait_stopped()
        return "done"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class PDXC_Server(WaxxServer):
    """TCP server exposing PDXC serial control over LAN.

    Initialises the device (SMC mode + max speed) on ``start()``.
    Only one move runs at a time (_move_lock); all other methods are
    serialised by _device_lock.  Each accepted TCP connection is handled
    on its own daemon thread.
    """

    def __init__(self, com_port: str = "COM26") -> None:
        WaxxServer.__init__(self, SERVER_ID, port=0)
        self._com_port = com_port
        self._device: Optional[PDXC] = None
        self._running = False
        self._device_lock = threading.Lock()
        self._move_lock = threading.Lock()

    def start(self) -> None:
        self._device = PDXC(self._com_port)
        self._device.initialize()
        self._running = True

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", 0))          # OS assigns a free port
        srv.listen(5)
        srv.settimeout(1.0)
        self._waxx_port = srv.getsockname()[1]   # tell beacon the real port
        self._start_beacon()
        print(f"PDXC server listening on port {self._waxx_port}")

        try:
            while self._running:
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True,
                ).start()
        except KeyboardInterrupt:
            print("\nShutting down PDXC server...")
        finally:
            self._stop_beacon()
            srv.close()
            if self._device is not None:
                self._device.close()
            self._running = False

    def stop(self) -> None:
        self._running = False

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
        if method in ("move_in", "move_out"):
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

class PDXC_Client(WaxxClient):
    """TCP client mirroring the PDXC motion API via ``PDXC_Server``.

    Discovered automatically via UDP broadcast beacon (server_id ``"pdxc"``).
    """

    def __init__(self, discovery_timeout: float = 5.0) -> None:
        super().__init__(SERVER_ID, discovery_timeout=discovery_timeout)

    def _send(self, method: str, **kwargs) -> dict:
        payload = (json.dumps({"method": method, "args": kwargs}) + "\n").encode("utf-8")
        tcp_timeout = _MOVE_TCP_TIMEOUT if method in ("move_in", "move_out") else 10.0
        for attempt in range(2):
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=10.0
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

    def move_in(self, steps: int = 60000, channel: int = 0) -> str:
        resp = self._send("move_in", steps=steps, channel=channel)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "unknown error"))
        return resp["result"]

    def move_out(self, steps: int = 60000, channel: int = 0) -> str:
        resp = self._send("move_out", steps=steps, channel=channel)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "unknown error"))
        return resp["result"]


__all__ = ["PDXC", "PDXC_Server", "PDXC_Client", "SERVER_ID"]
