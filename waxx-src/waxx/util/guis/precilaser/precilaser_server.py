from __future__ import annotations

import atexit
import inspect
import json
import logging
import logging.handlers
import os
import signal
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

from waxx.util.comms_server.com_link import ComLink, ComLinkRefused
from waxx.util.guis.precilaser.precilaser_controller import (
    PrecilaserController,
    PrecilaserStartupController,
)
from beacon.discovery.server import NetServer


LOGGER = logging.getLogger("precilaser_server")
LOGGER.setLevel(logging.INFO)
CONTROLLER_LOGGER = logging.getLogger("precilaser_controller")
CONTROLLER_LOGGER.setLevel(logging.INFO)

# Baud rate the controller opens the port with (its constructor default); the
# server never overrides it, so this is what the "com" snapshot reports before
# a controller exists.
_DEFAULT_BAUD = int(inspect.signature(PrecilaserController.__init__).parameters["baudrate"].default)

# Grace period between answering "OK" to SHUTDOWN and starting to tear down,
# so the reply is on the wire before the listening socket closes.
_SHUTDOWN_REPLY_GRACE_S = 0.2
# If main() has not exited this long after stop() finished, force the exit.
_SHUTDOWN_FORCE_EXIT_S = 1.5
# How long a sequence waits, per reconnect attempt, for the link to reopen the
# port after a serial error (the link's backoff opens at +1 s, +3 s, +8 s).
_SEQUENCE_RECONNECT_WAIT_S = 5.0
# Settle time before reopening the port after a checksum mismatch.
_CHECKSUM_RESYNC_DELAY_S = 0.5


class ConnectionState(Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


class SequenceState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"


@dataclass
class LaserStatus:
    connected: bool = False
    connection_state: ConnectionState = ConnectionState.DISCONNECTED
    pd_ok: bool = False
    temperature_ok: bool = False
    laser_enabled: bool = False
    power_stability_enabled: bool = False
    working_current_a: float = 0.0
    pd_values: list[float] = field(default_factory=lambda: [0.0] * 5)
    temperatures_c: list[float] = field(default_factory=lambda: [0.0] * 4)
    stage_currents_a: list[float] = field(default_factory=list)


class LogBufferHandler(logging.Handler):
    def __init__(self, server: "PrecilaserLaserServer"):
        super().__init__()
        self.server = server

    def emit(self, record: logging.LogRecord) -> None:
        self.server.append_log_entry(self.format(record))


DEFAULT_RAMP_UP_ENDPOINT_A = 9.5


class PrecilaserLink(ComLink):
    """The server's COM port (``ComLink``): opened in the background after the
    server is up, reopened on its backoff after a failure, held closed by a
    person's Disconnect.

    Opening only opens the port and clears its buffers
    (``PrecilaserController.connect``): nothing is sent to the laser, so a
    connect or reconnect never changes its output.
    """

    def __init__(self, owner: "PrecilaserLaserServer"):
        super().__init__(
            owner.serial_port,
            baud=_DEFAULT_BAUD,
            server=owner,
            name=f"precilaser {owner.serial_port}",
            io_lock=owner._laser_lock,
            logger=LOGGER,
        )
        self._owner = owner

    def _open(self) -> PrecilaserController:
        laser = PrecilaserController(port=self.port)
        laser.connect()
        return laser

    def _close(self, laser: PrecilaserController) -> None:
        laser.close()

    def _on_connected(self, laser: PrecilaserController) -> None:
        self._owner._on_link_connected(laser)

    def _disconnect_refusal(self) -> Optional[str]:
        return self._owner._disconnect_refusal()


class PrecilaserLaserServer(NetServer):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        serial_port: str = "COM6",
        poll_interval_s: float = 1.0,
        max_log_entries: int = 2000,
        auto_connect: bool = True,
        config_path: Optional[str] = None,
    ):
        NetServer.__init__(self, "precilaser", port)
        self.host = host
        self.port = int(port)
        self.serial_port = serial_port
        self.poll_interval_s = float(poll_interval_s)
        self.max_log_entries = int(max_log_entries)
        self.auto_connect = auto_connect
        self.reconnect_delay_s = 0.3

        # Server-side persisted settings (JSON).  Currently just the current
        # that the turn-on ramp finishes at.  config_path is provided by the
        # kexp launcher; None disables persistence (value stays in memory).
        self.config_path = config_path
        self._config_lock = threading.Lock()
        self.ramp_up_endpoint_a = DEFAULT_RAMP_UP_ENDPOINT_A
        self._load_config()

        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.accept_thread: Optional[threading.Thread] = None
        self.poll_thread: Optional[threading.Thread] = None
        self.sequence_thread: Optional[threading.Thread] = None

        self.status = LaserStatus()
        self.sequence_state = SequenceState.IDLE
        self.sequence_type: Optional[str] = None
        self._interrupt_requested = False

        self._state_lock = threading.Lock()
        # Re-entrant: it is also the link's io_lock, and report_error() is
        # called with it held.
        self._laser_lock = threading.RLock()
        self._log_lock = threading.Lock()
        self._log_entries: list[str] = []
        self._log_offset = 0
        self._stopped = False

        # The COM port (opened by start(); the "com" snapshot and beacon state).
        self.link = PrecilaserLink(self)

        # Set once a graceful SHUTDOWN has been fully processed (stop() done);
        # main() waits on it and exits 0.
        self.shutdown_event = threading.Event()
        self._shutdown_requested = False

        self.log_handler = LogBufferHandler(self)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )

    @property
    def laser(self) -> Optional[PrecilaserController]:
        """The open controller (the link's device), or None while not connected."""
        return self.link.device

    def start(self) -> None:
        if self.running:
            return
        # Pre-bind and start listening so the TCP server is ready before the
        # beacon fires and before the link opens the COM port.  This lets the
        # GUI connect and display server status while the port is still being
        # opened (or keeps failing).
        _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _sock.bind((self.host, 0))
        _sock.listen(16)
        self.server_socket = _sock
        self.port = _sock.getsockname()[1]
        self._waxx_port = self.port
        self._start_beacon()
        self.running = True
        LOGGER.addHandler(self.log_handler)
        CONTROLLER_LOGGER.addHandler(self.log_handler)
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()
        if self.auto_connect:
            # Opens the port on the link's own thread, retrying with backoff
            # while it fails; never blocks or stops the server.
            self.link.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stop_beacon()
        self._stopped = True
        self.running = False
        self.interrupt_sequence()
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None
        if self.accept_thread is not None:
            self.accept_thread.join(timeout=2.0)
        if self.poll_thread is not None:
            self.poll_thread.join(timeout=2.0)
        if self.sequence_thread is not None and self.sequence_thread.is_alive():
            self.sequence_thread.join(timeout=2.0)
        self._release_serial_on_stop()
        LOGGER.removeHandler(self.log_handler)
        CONTROLLER_LOGGER.removeHandler(self.log_handler)
        self.log_handler.close()

    def _release_serial_on_stop(self) -> None:
        """Close the COM port at process stop, even if a sequence is still flagged RUNNING.

        ``disconnect_laser()`` refuses while a sequence runs (correct for an
        operator request).  At stop the process is going away regardless, so
        the port must be released anyway: ``link.stop()`` (bounded) stops the
        link's thread and only closes the serial handle; it does not send
        anything to the laser.
        """
        was_open = self.laser is not None
        if was_open and self.sequence_state == SequenceState.RUNNING:
            LOGGER.warning(
                "Sequence still running at stop; force-closing serial port %s", self.serial_port
            )
        self.link.stop()
        if was_open:
            LOGGER.info("Serial connection closed on %s", self.serial_port)
        with self._state_lock:
            self.status = LaserStatus(connected=False, connection_state=ConnectionState.DISCONNECTED)

    def request_shutdown(self) -> None:
        """Graceful process shutdown requested over TCP (dashboard Stop).

        Returns immediately so the caller can send its reply; a daemon thread
        then waits a short grace period, runs :meth:`stop` (which closes the
        serial port and stops the beacon) and sets :attr:`shutdown_event` so
        ``main()`` returns and the process exits 0.  If ``main()`` has not
        exited shortly after that, the thread forces ``os._exit(0)``.
        """
        with self._state_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
        LOGGER.warning("Process shutdown requested over TCP; stopping Precilaser server")

        def _worker() -> None:
            time.sleep(_SHUTDOWN_REPLY_GRACE_S)
            try:
                self.stop()
            except Exception:
                LOGGER.exception("stop() raised during requested shutdown")
            self.shutdown_event.set()
            time.sleep(_SHUTDOWN_FORCE_EXIT_S)
            LOGGER.warning("main loop did not exit after stop(); forcing process exit")
            logging.shutdown()
            os._exit(0)

        threading.Thread(target=_worker, name="precilaser-shutdown", daemon=True).start()

    def append_log_entry(self, message: str) -> None:
        with self._log_lock:
            self._log_entries.append(message)
            overflow = len(self._log_entries) - self.max_log_entries
            if overflow > 0:
                self._log_entries = self._log_entries[overflow:]
                self._log_offset += overflow

    def get_logs_since(self, start_index: int) -> dict:
        with self._log_lock:
            normalized_start = max(int(start_index), self._log_offset)
            relative_start = normalized_start - self._log_offset
            messages = self._log_entries[relative_start:]
            next_index = self._log_offset + len(self._log_entries)
        return {
            "messages": messages,
            "next_index": next_index,
        }

    @staticmethod
    def _connection_state_from_com(com: dict) -> ConnectionState:
        """``LaserStatus.connection_state`` as the link sees it now.

        connected -> CONNECTED; error (and a retry open after an error) ->
        ERROR; not started / stopped / held by a person -> DISCONNECTED.
        """
        status = com.get("status")
        if status == "connected":
            return ConnectionState.CONNECTED
        if status == "error" or (status == "connecting" and com.get("last_error")):
            return ConnectionState.ERROR
        return ConnectionState.DISCONNECTED

    def get_snapshot(self) -> dict:
        # The link's own lock is never taken under _state_lock.
        com = self.link.snapshot()
        connection_state = self._connection_state_from_com(com)
        with self._state_lock:
            status_dict = asdict(self.status)
            status_dict["connected"] = connection_state == ConnectionState.CONNECTED
            status_dict["connection_state"] = connection_state.value
            return {
                "status": status_dict,
                "serial_port": self.serial_port,
                "com": com,
                "sequence": {
                    "state": self.sequence_state.value,
                    "type": self.sequence_type,
                },
                "log_count": self._log_offset + len(self._log_entries),
                "ramp_up_endpoint_a": self.ramp_up_endpoint_a,
            }

    def _load_config(self) -> None:
        if not self.config_path or not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            value = float(data.get("ramp_up_endpoint_a", self.ramp_up_endpoint_a))
            if value >= 0:
                self.ramp_up_endpoint_a = value
            LOGGER.info("Loaded ramp-up endpoint %.2f A from %s", self.ramp_up_endpoint_a, self.config_path)
        except Exception as exc:
            LOGGER.warning("Failed to load precilaser config from %s: %s", self.config_path, exc)

    def _save_config(self) -> None:
        if not self.config_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as fh:
                json.dump({"ramp_up_endpoint_a": self.ramp_up_endpoint_a}, fh, indent=2)
        except Exception as exc:
            LOGGER.warning("Failed to save precilaser config to %s: %s", self.config_path, exc)

    def get_ramp_up_endpoint(self) -> float:
        with self._config_lock:
            return self.ramp_up_endpoint_a

    def set_ramp_up_endpoint(self, value_a: float) -> None:
        value_a = float(value_a)
        if value_a < 0:
            raise ValueError("ramp_up_endpoint_a must be >= 0")
        with self._config_lock:
            self.ramp_up_endpoint_a = value_a
            self._save_config()
        LOGGER.info("Ramp-up endpoint set to %.2f A", value_a)

    def _accept_loop(self) -> None:
        try:
            if self.server_socket is None:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(16)
            self.server_socket.settimeout(1.0)
            LOGGER.info("Precilaser server listening on %s:%s", self.host, self.port)
            while self.running:
                try:
                    client_socket, _ = self.server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.running:
                        LOGGER.exception("Server accept failed")
                    break
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True,
                ).start()
        except Exception as exc:
            LOGGER.exception("Server failed to start: %s", exc)
        finally:
            if self.server_socket is not None:
                try:
                    self.server_socket.close()
                except OSError:
                    pass
                self.server_socket = None

    def _handle_client(self, client_socket: socket.socket) -> None:
        try:
            data = client_socket.recv(4096).decode("utf-8", errors="replace").strip()
            if not data:
                return
            response = self._dispatch_command(data)
            client_socket.sendall(f"{response}\n".encode("utf-8"))
        except Exception as exc:
            LOGGER.exception("Client error: %s", exc)
        finally:
            client_socket.close()

    def _dispatch_command(self, raw_command: str) -> str:
        parts = raw_command.strip().split(maxsplit=1)
        command = parts[0].upper()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command not in {"GET_SNAPSHOT", "GET_LOGS"}:
            LOGGER.info("TCP command received: %s", raw_command)

        try:
            if command == "GET_SNAPSHOT":
                return json.dumps(self.get_snapshot())
            if command == "GET_LOGS":
                start_index = int(argument) if argument else 0
                return json.dumps(self.get_logs_since(start_index))
            # COM_CONNECT / COM_DISCONNECT are the generic link commands; the
            # old CONNECT_SERIAL / DISCONNECT_SERIAL do the same.  Connect
            # returns at once (the link opens the port in the background);
            # disconnect holds the port closed until the next connect.
            if command in {"COM_CONNECT", "CONNECT_SERIAL"}:
                self.connect_laser()
                return "OK"
            if command in {"COM_DISCONNECT", "DISCONNECT_SERIAL"}:
                self.disconnect_laser()
                return "OK"
            if command == "SHUTDOWN":
                # Process-level stop (dashboard).  Distinct from START_SHUTDOWN,
                # which is the laser's power-down *sequence*.
                self.request_shutdown()
                return "OK"
            if command == "START_STARTUP":
                return "OK" if self.start_startup_sequence() else "ERROR: startup request rejected"
            if command == "START_SHUTDOWN":
                return "OK" if self.start_shutdown_sequence() else "ERROR: shutdown request rejected"
            if command == "INTERRUPT":
                self.interrupt_sequence()
                return "OK"
            if command == "SET_WORKING_CURRENT":
                self.set_working_current(float(argument))
                return "OK"
            if command == "SET_LASER_ENABLE":
                self.set_laser_enable(argument.lower() in {"1", "true", "on"})
                return "OK"
            if command == "SET_STABILITY_MODE":
                self.set_stability_mode(argument.lower() in {"1", "true", "on"})
                return "OK"
            if command == "GET_RAMP_ENDPOINT":
                return json.dumps({"ramp_up_endpoint_a": self.get_ramp_up_endpoint()})
            if command == "SET_RAMP_ENDPOINT":
                self.set_ramp_up_endpoint(float(argument))
                return "OK"
        except ComLinkRefused as exc:
            LOGGER.warning("Command refused: %s (%s)", command, exc)
            return f"ERROR: {exc}"
        except Exception as exc:
            LOGGER.exception("Command failed: %s", exc)
            return f"ERROR: {exc}"

        return f"ERROR: unknown command {command}"

    def _poll_loop(self) -> None:
        while self.running:
            try:
                self.poll_status()
            except Exception as exc:
                LOGGER.exception("Background polling failed: %s", exc)
            time.sleep(self.poll_interval_s)

    def poll_status(self) -> None:
        with self._laser_lock:
            self._poll_status_locked()

    def _poll_status_locked(self) -> None:
        laser = self.laser
        if laser is None:
            return
        try:
            status = laser.query_working_status()
            updated_status = LaserStatus(
                connected=True,
                connection_state=ConnectionState.CONNECTED,
                pd_ok=status.pd_ok,
                temperature_ok=status.temperature_ok,
                laser_enabled=status.laser_enabled,
                power_stability_enabled=status.power_stability_enabled,
                working_current_a=status.working_current_a,
                pd_values=status.pd_values[:5],
                temperatures_c=status.temperatures_c[:4],
                stage_currents_a=list(status.stage_currents_a),
            )
            with self._state_lock:
                self.status = updated_status
            self.link.note_rx()
        except Exception as exc:
            exc_str = str(exc)
            if isinstance(exc, ValueError) and "Invalid frame tail" in exc_str:
                LOGGER.debug("Invalid frame tail detected (%s); reconnecting serial", exc)
                self._resync_serial_locked(laser, self.reconnect_delay_s, "invalid tail")
                return
            if isinstance(exc, ValueError) and "checksum mismatch" in exc_str.lower():
                LOGGER.info("Checksum error, restarting COM connection")
                self._resync_serial_locked(laser, _CHECKSUM_RESYNC_DELAY_S, "checksum error")
                return
            self._handle_serial_error(exc, where="status poll")

    def _resync_serial_locked(
        self, laser: PrecilaserController, settle_s: float, reason: str
    ) -> None:
        """Close and reopen the port in place after a garbled frame.

        A quick resync of the link's own controller (same handle), kept out
        of the link's error path because it is routine (thousands in the
        log) and the settle time matters: an error report would log a
        warning, show the COM pill red and wait the link's 1 s backoff.
        Caller holds ``_laser_lock`` (the link's io_lock), so the link cannot
        close or swap the device meanwhile.  A person's Disconnect (the hold)
        or a stop seen before the reopen leaves the port closed; a failed
        reopen goes to the link (``report_error``), which reconnects on its
        backoff unless held.
        """
        try:
            laser.close()
        except Exception:
            pass
        time.sleep(settle_s)
        if self.link.held or not self.running or self.link.device is not laser:
            LOGGER.info("Serial port left closed after %s (disconnected or stopping)", reason)
            return
        try:
            laser.connect()
            LOGGER.debug("Serial reconnected on %s after %s", self.serial_port, reason)
        except Exception as reconnect_exc:
            LOGGER.exception("Serial reconnect after %s failed: %s", reason, reconnect_exc)
            self._handle_serial_error(reconnect_exc, where=f"reconnect after {reason}")

    def _handle_serial_error(self, exc: Exception, where: str = "") -> None:
        """The server's I/O on the port failed: clear the telemetry and let
        the link close the port and reopen it on its backoff (not while a
        person holds it closed)."""
        LOGGER.exception("Serial communication failed: %s", exc)
        with self._state_lock:
            self.status = LaserStatus(connected=False, connection_state=ConnectionState.ERROR)
        self.link.report_error(exc, where=where)

    def _on_link_connected(self, laser: PrecilaserController) -> None:
        """After every successful open (the link's thread).  Sends nothing to
        the laser: background polling populates telemetry on the next cycle."""
        LOGGER.info("Serial connection opened on %s", self.serial_port)
        with self._state_lock:
            self.status.connected = True
            self.status.connection_state = ConnectionState.CONNECTED

    def _disconnect_refusal(self) -> Optional[str]:
        """Why a person's Disconnect is refused now (None: allowed)."""
        with self._state_lock:
            if self.sequence_state == SequenceState.RUNNING:
                return "a sequence is running"
        return None

    def connect_laser(self) -> None:
        """Release a hold and open the port as soon as possible (the link's
        thread opens it; returns at once).  Harmless while a sequence runs:
        the link never replaces an open port."""
        self.link.connect()

    def disconnect_laser(self) -> None:
        """Close the port and hold it closed until the next connect.

        Raises ``ComLinkRefused`` while a sequence is running.
        """
        self.link.disconnect(hold=True)
        LOGGER.info("Serial connection closed on %s", self.serial_port)
        with self._state_lock:
            self.status = LaserStatus(connected=False, connection_state=ConnectionState.DISCONNECTED)

    def set_working_current(self, working_current_a: float) -> None:
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.set_working_current(working_current_a)
            LOGGER.info("Working current changed to %.2f A", working_current_a)
            self._poll_status_locked()

    def set_laser_enable(self, enabled: bool) -> None:
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.set_laser_enable(enabled)
            LOGGER.info("Laser enable set to %s", enabled)
            self._poll_status_locked()

    def set_stability_mode(self, enabled: bool) -> None:
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.set_power_stability_mode(enabled)
            LOGGER.info("Power stability mode set to %s", enabled)
            self._poll_status_locked()

    def start_startup_sequence(self) -> bool:
        return self._start_sequence("STARTUP")

    def start_shutdown_sequence(self) -> bool:
        return self._start_sequence("SHUTDOWN")

    def _start_sequence(self, sequence_type: str) -> bool:
        with self._state_lock:
            if self.laser is None or self.link.state != "connected":
                LOGGER.warning("Cannot start %s sequence while laser is disconnected", sequence_type.lower())
                return False
            if self.sequence_thread is not None and self.sequence_thread.is_alive():
                LOGGER.warning("A sequence is already running")
                return False
            self.sequence_state = SequenceState.RUNNING
            self.sequence_type = sequence_type
            self._interrupt_requested = False

        target = self._run_startup_sequence if sequence_type == "STARTUP" else self._run_shutdown_sequence
        self.sequence_thread = threading.Thread(target=target, daemon=True)
        self.sequence_thread.start()
        LOGGER.info("%s sequence requested", sequence_type.title())
        return True

    def interrupt_sequence(self) -> None:
        with self._state_lock:
            self._interrupt_requested = True
        LOGGER.warning("Sequence interrupt requested")

    def _should_continue_sequence(self) -> bool:
        with self._state_lock:
            return self.running and (not self._interrupt_requested)

    def _run_startup_sequence(self) -> None:
        self._run_sequence("STARTUP")

    def _run_shutdown_sequence(self) -> None:
        self._run_sequence("SHUTDOWN")

    def _on_ramp_step(self, current_a: float) -> None:
        """Called after each ramp step to push working_current_a into the snapshot without a serial round-trip."""
        with self._state_lock:
            self.status.working_current_a = current_a

    def _wait_for_link(self, timeout_s: float) -> bool:
        """Wait (a sequence's thread) for the link to have the port open.

        False when it did not open in time, the sequence was interrupted, or
        a person holds the port closed: the hold is never overridden here.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.link.held or not self._should_continue_sequence():
                return False
            if self.laser is not None and self.link.state == "connected":
                return True
            time.sleep(0.05)
        return self.laser is not None

    def _run_sequence(self, sequence_type: str) -> None:
        max_reconnect_attempts = 3
        reconnect_attempt = 0
        while True:
            laser_ref: Optional[PrecilaserController] = None
            try:
                with self._laser_lock:
                    if self.laser is None:
                        raise RuntimeError("Laser not connected")
                    laser_ref = self.laser
                startup_controller = PrecilaserStartupController(
                    laser_ref,
                    target_working_current_a=self.get_ramp_up_endpoint(),
                )
                startup_controller.on_step = self._on_ramp_step
                if sequence_type == "STARTUP":
                    startup_controller.run_turn_on_procedure(should_continue=self._should_continue_sequence)
                else:
                    startup_controller.run_turn_off_procedure(should_continue=self._should_continue_sequence)

                # Pull a final full snapshot at sequence end (PD, temp, stage currents, flags).
                self.poll_status()

                with self._state_lock:
                    self.sequence_state = SequenceState.COMPLETED
                LOGGER.info("%s sequence completed", sequence_type.title())
                return
            except Exception as exc:
                # User-requested interrupt: stop without retrying.
                if not self._should_continue_sequence():
                    LOGGER.warning("%s sequence interrupted", sequence_type.title())
                    with self._state_lock:
                        self.sequence_state = SequenceState.INTERRUPTED
                    return
                if isinstance(exc, RuntimeError) and "interrupted" in str(exc).lower():
                    LOGGER.warning("%s sequence interrupted", sequence_type.title())
                    with self._state_lock:
                        self.sequence_state = SequenceState.INTERRUPTED
                    return

                # A person's Disconnect (forced) holds the port closed: stop
                # here rather than wait for a reconnect that will not come.
                if self.link.held:
                    LOGGER.warning(
                        "%s sequence stopped: serial port %s was disconnected (%s)",
                        sequence_type.title(), self.serial_port, exc,
                    )
                    with self._state_lock:
                        self.sequence_state = SequenceState.INTERRUPTED
                    return

                # COM / serial error: try to reconnect and resume from current laser state.
                reconnect_attempt += 1
                if reconnect_attempt > max_reconnect_attempts:
                    LOGGER.exception(
                        "%s sequence failed after %d reconnect attempt(s): %s",
                        sequence_type.title(), reconnect_attempt - 1, exc,
                    )
                    with self._state_lock:
                        self.sequence_state = SequenceState.INTERRUPTED
                    return

                LOGGER.warning(
                    "COM error during %s sequence (reconnect attempt %d/%d): %s"
                    " — reconnecting and resuming...",
                    sequence_type.lower(), reconnect_attempt, max_reconnect_attempts, exc,
                )
                # The link closes the port and reopens it on its backoff; only
                # the handle this attempt used is reported, so a port the link
                # has already reopened is not closed again.
                with self._laser_lock:
                    if laser_ref is not None and self.laser is laser_ref:
                        self._handle_serial_error(exc, where=f"{sequence_type.lower()} sequence")
                if self._wait_for_link(_SEQUENCE_RECONNECT_WAIT_S):
                    LOGGER.info(
                        "Reconnected on %s; resuming %s sequence",
                        self.serial_port, sequence_type.lower(),
                    )
                else:
                    LOGGER.error(
                        "Serial port %s not reopened during %s sequence (%s)",
                        self.serial_port, sequence_type.lower(), self.link.state,
                    )



def main(
    host: str = "0.0.0.0",
    port: int = 0,
    serial_port: str = "COM6",
    log_path: str | None = None,
    config_path: str | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        _fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(_fh)
        LOGGER.info("Logging to file: %s", log_path)
    server = PrecilaserLaserServer(
        host=host, port=port, serial_port=serial_port, config_path=config_path
    )

    # Ensure COM port is released on any exit path (crash, SIGTERM, atexit).
    atexit.register(server.stop)

    def _handle_sigterm(signum, frame):
        LOGGER.info("SIGTERM received, stopping server")
        server.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    server.start()
    try:
        # Wake promptly when a TCP SHUTDOWN has finished stop(); fall through
        # to the finally (idempotent stop) and return -> exit code 0.
        while not server.shutdown_event.wait(0.25):
            pass
        LOGGER.info("Precilaser server exiting after requested shutdown")
    except KeyboardInterrupt:
        LOGGER.info("Stopping Precilaser server")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
