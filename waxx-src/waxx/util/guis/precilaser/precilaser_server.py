from __future__ import annotations

import atexit
import json
import logging
import signal
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

from waxx.util.guis.precilaser.precilaser_controller import (
    PrecilaserController,
    PrecilaserStartupController,
)


LOGGER = logging.getLogger("precilaser_server")
LOGGER.setLevel(logging.INFO)


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


class PrecilaserLaserServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5560,
        serial_port: str = "COM6",
        poll_interval_s: float = 1.0,
        max_log_entries: int = 2000,
        auto_connect: bool = True,
    ):
        self.host = host
        self.port = int(port)
        self.serial_port = serial_port
        self.poll_interval_s = float(poll_interval_s)
        self.max_log_entries = int(max_log_entries)
        self.auto_connect = auto_connect
        self.reconnect_delay_s = 0.3

        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.accept_thread: Optional[threading.Thread] = None
        self.poll_thread: Optional[threading.Thread] = None
        self.sequence_thread: Optional[threading.Thread] = None

        self.laser: Optional[PrecilaserController] = None
        self.status = LaserStatus()
        self.sequence_state = SequenceState.IDLE
        self.sequence_type: Optional[str] = None
        self._interrupt_requested = False

        self._state_lock = threading.Lock()
        self._laser_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._log_entries: list[str] = []
        self._log_offset = 0
        self._stopped = False

        self.log_handler = LogBufferHandler(self)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        LOGGER.addHandler(self.log_handler)
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()
        if self.auto_connect:
            try:
                self.connect_laser()
            except Exception as exc:
                LOGGER.exception("Initial serial connection failed: %s", exc)

    def stop(self) -> None:
        if self._stopped:
            return
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
        self.disconnect_laser()
        LOGGER.removeHandler(self.log_handler)
        self.log_handler.close()

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

    def get_snapshot(self) -> dict:
        with self._state_lock:
            status_dict = asdict(self.status)
            status_dict["connection_state"] = self.status.connection_state.value
            return {
                "status": status_dict,
                "sequence": {
                    "state": self.sequence_state.value,
                    "type": self.sequence_type,
                },
                "log_count": self._log_offset + len(self._log_entries),
            }

    def _accept_loop(self) -> None:
        try:
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
            if command == "CONNECT_SERIAL":
                self.connect_laser()
                return "OK"
            if command == "DISCONNECT_SERIAL":
                self.disconnect_laser()
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
        except Exception as exc:
            LOGGER.exception("Command failed: %s", exc)
            return f"ERROR: {exc}"

        return f"ERROR: unknown command {command}"

    def _poll_loop(self) -> None:
        while self.running:
            try:
                if self.sequence_state != SequenceState.RUNNING:
                    self.poll_status()
            except Exception as exc:
                LOGGER.exception("Background polling failed: %s", exc)
            time.sleep(self.poll_interval_s)

    def poll_status(self) -> None:
        with self._laser_lock:
            self._poll_status_locked()

    def _poll_status_locked(self) -> None:
        if self.laser is None:
            return
        try:
            status = self.laser.query_working_status()
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
        except Exception as exc:
            if isinstance(exc, ValueError) and "Invalid frame tail" in str(exc):
                self._recover_serial_after_invalid_tail_locked(exc)
                return
            self._handle_serial_error(exc)

    def _recover_serial_after_invalid_tail_locked(self, exc: Exception) -> None:
        LOGGER.debug("Invalid frame tail detected (%s); reconnecting serial", exc)
        try:
            if self.laser is not None:
                try:
                    self.laser.close()
                except Exception:
                    pass
            time.sleep(self.reconnect_delay_s)
            self.laser = PrecilaserController(port=self.serial_port)
            self.laser.connect()
            LOGGER.debug("Serial reconnected on %s after invalid frame tail", self.serial_port)
            with self._state_lock:
                self.status.connected = True
                self.status.connection_state = ConnectionState.CONNECTED
        except Exception as reconnect_exc:
            LOGGER.exception("Serial reconnect after invalid tail failed: %s", reconnect_exc)
            self._handle_serial_error(reconnect_exc)

    def _handle_serial_error(self, exc: Exception) -> None:
        LOGGER.exception("Serial communication failed: %s", exc)
        try:
            if self.laser is not None:
                self.laser.close()
        except Exception:
            pass
        self.laser = None
        with self._state_lock:
            self.status = LaserStatus(connected=False, connection_state=ConnectionState.ERROR)

    def connect_laser(self) -> None:
        with self._laser_lock:
            if self.sequence_state == SequenceState.RUNNING:
                LOGGER.warning("Cannot connect laser while a sequence is running")
                return
            try:
                if self.laser is None:
                    self.laser = PrecilaserController(port=self.serial_port)
                self.laser.connect()
                LOGGER.info("Serial connection opened on %s", self.serial_port)
                # Do not block CONNECT_SERIAL on an immediate status query.
                # Background polling will populate telemetry on the next cycle.
                with self._state_lock:
                    self.status.connected = True
                    self.status.connection_state = ConnectionState.CONNECTED
            except Exception as exc:
                self._handle_serial_error(exc)
                raise

    def disconnect_laser(self) -> None:
        with self._laser_lock:
            if self.sequence_state == SequenceState.RUNNING:
                LOGGER.warning("Cannot disconnect laser while a sequence is running")
                return
            if self.laser is not None:
                try:
                    self.laser.close()
                    LOGGER.info("Serial connection closed on %s", self.serial_port)
                except Exception:
                    pass
                self.laser = None
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
            if self.laser is None or self.status.connection_state != ConnectionState.CONNECTED:
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

    def _run_sequence(self, sequence_type: str) -> None:
        try:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            startup_controller = PrecilaserStartupController(self.laser)
            if sequence_type == "STARTUP":
                with self._laser_lock:
                    startup_controller.run_turn_on_procedure(should_continue=self._should_continue_sequence)
                    self._poll_status_locked()
            else:
                with self._laser_lock:
                    startup_controller.run_turn_off_procedure(should_continue=self._should_continue_sequence)
                    self._poll_status_locked()

            with self._state_lock:
                self.sequence_state = SequenceState.COMPLETED
            LOGGER.info("%s sequence completed", sequence_type.title())
        except Exception as exc:
            LOGGER.exception("%s sequence failed: %s", sequence_type.title(), exc)
            with self._state_lock:
                self.sequence_state = SequenceState.INTERRUPTED
        finally:
            with self._state_lock:
                if self.sequence_state != SequenceState.RUNNING:
                    pass



def main(host: str = "0.0.0.0", port: int = 5560, serial_port: str = "COM6") -> None:
    logging.basicConfig(level=logging.INFO)
    server = PrecilaserLaserServer(host=host, port=port, serial_port=serial_port)

    # Ensure COM port is released on any exit path (crash, SIGTERM, atexit).
    atexit.register(server.stop)

    def _handle_sigterm(signum, frame):
        LOGGER.info("SIGTERM received, stopping server")
        server.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    server.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        LOGGER.info("Stopping Precilaser server")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
