from __future__ import annotations

import atexit
import json
import logging
import signal
import socket
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

from waxx.util.guis.als.als_fiber_amplifier import ALSLaserController, ALSLaserStartupController


LOGGER = logging.getLogger("als_laser_server")
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
    power_enabled: bool = False
    interlock_enabled: bool = False
    second_stage_enabled: bool = False
    power_setpoint_percent: float = 0.0
    temperature_act_p: float = 0.0
    temperature_set_p: float = 0.0
    imon_pa: float = 0.0
    lmon: float = 0.0
    pmon_w: float = 0.0
    connected: bool = False
    connection_state: ConnectionState = ConnectionState.DISCONNECTED


class LogBufferHandler(logging.Handler):
    def __init__(self, server: "ALSLaserServer"):
        super().__init__()
        self.server = server

    def emit(self, record: logging.LogRecord) -> None:
        self.server.append_log_entry(self.format(record))


class ALSLaserServer:
    STARTUP_STEPS = [
        (1, "Turn Power On", "step_1_turn_laser_power_on"),
        (2, "Turn Interlock On", "step_2_turn_interlock_on"),
        (3, "Wait for IMON-PA", "step_3_wait_for_imon_pa"),
        (4, "Turn On Second Stage", "step_4_turn_on_second_stage"),
        (5, "Ramp to 80%", "step_5_ramp_to_80_percent"),
        (6, "Warm Up at 80%", "step_6_warm_up_at_80_percent"),
        (7, "Turn to 100%", "step_7_turn_to_100_percent"),
    ]
    SHUTDOWN_STEPS = [
        (1, "Ramp Down to 0%", "step_1_ramp_down_to_zero_percent"),
        (2, "Turn Off Second Stage", "step_2_turn_off_second_stage"),
        (3, "Turn Off Interlock", "step_3_turn_off_interlock"),
        (4, "Turn Off Power", "step_4_turn_off_laser_power"),
    ]

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5557,
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

        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.accept_thread: Optional[threading.Thread] = None
        self.poll_thread: Optional[threading.Thread] = None
        self.sequence_thread: Optional[threading.Thread] = None

        self.laser: Optional[ALSLaserController] = None
        self.status = LaserStatus()
        self.sequence_state = SequenceState.IDLE
        self.sequence_type: Optional[str] = None
        self.sequence_started_epoch: Optional[float] = None
        self.current_step_number: Optional[int] = None
        self.current_step_started_epoch: Optional[float] = None
        self.startup_step_states = ["not_done"] * len(self.STARTUP_STEPS)
        self.shutdown_step_states = ["not_done"] * len(self.SHUTDOWN_STEPS)
        self.startup_step_notes = [""] * len(self.STARTUP_STEPS)
        self.shutdown_step_notes = [""] * len(self.SHUTDOWN_STEPS)
        self._current_step_already_done = False
        self._interrupt_requested = False

        self._state_lock = threading.Lock()
        self._laser_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._log_entries: list[str] = []
        self._log_offset = 0
        self._cleanup_registered = False

        self.log_handler = LogBufferHandler(self)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
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
        if not self._cleanup_registered:
            atexit.register(self._cleanup_on_exit)
            self._cleanup_registered = True
        if self.auto_connect:
            try:
                self.connect_laser()
            except Exception as exc:
                LOGGER.exception("Initial serial connection failed: %s", exc)

    def stop(self) -> None:
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

    def _cleanup_on_exit(self) -> None:
        """Best-effort process-exit cleanup that releases the serial port."""
        try:
            if self.running:
                self.stop()
            else:
                self.disconnect_laser()
        except Exception:
            pass

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
                "serial_port": self.serial_port,
                "sequence": {
                    "state": self.sequence_state.value,
                    "type": self.sequence_type,
                    "started_epoch": self.sequence_started_epoch,
                    "current_step_number": self.current_step_number,
                    "current_step_started_epoch": self.current_step_started_epoch,
                    "startup_steps": list(self.startup_step_states),
                    "shutdown_steps": list(self.shutdown_step_states),
                    "startup_step_notes": list(self.startup_step_notes),
                    "shutdown_step_notes": list(self.shutdown_step_notes),
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
            LOGGER.info("ALS server listening on %s:%s", self.host, self.port)
            while self.running:
                try:
                    client_socket, _ = self.server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.running:
                        LOGGER.exception("ALS server accept failed")
                    break
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True,
                ).start()
        except Exception as exc:
            LOGGER.exception("ALS server failed to start: %s", exc)
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

        if command not in {"GET_SNAPSHOT", "GET_LOGS", "GET_POWER_SETPOINT", "GET_POWER_SETPOINT_PERCENT"}:
            LOGGER.info("TCP command received: %s", raw_command)

        try:
            if command == "GET_SNAPSHOT":
                return json.dumps(self.get_snapshot())
            if command == "GET_LOGS":
                start_index = int(argument) if argument else 0
                return json.dumps(self.get_logs_since(start_index))
            if command in {"GET_POWER_SETPOINT", "GET_POWER_SETPOINT_PERCENT"}:
                return f"POWER_SETPOINT: {self.status.power_setpoint_percent:.3f}"
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
            if command == "SET_POWER_PERCENT":
                self.set_power_percent(float(argument))
                return "OK"
            if command == "SET_POWER_SUPPLY_ON":
                return "OK" if self.set_power_supply_on() else "ERROR: power-on request rejected"
            if command == "SET_POWER_SUPPLY_OFF":
                return "OK" if self.set_power_supply_off() else "ERROR: power-off request rejected"
            if command == "SET_INTERLOCK_ON":
                return "OK" if self.set_interlock_on() else "ERROR: interlock-on request rejected"
            if command == "SET_INTERLOCK_OFF":
                return "OK" if self.set_interlock_off() else "ERROR: interlock-off request rejected"
            if command == "SET_SECOND_STAGE_ON":
                return "OK" if self.set_second_stage_on() else "ERROR: 2nd-stage-on request rejected"
            if command == "SET_SECOND_STAGE_OFF":
                return "OK" if self.set_second_stage_off() else "ERROR: 2nd-stage-off request rejected"
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
            frame = self.laser.stop_and_read()
            if frame is None:
                return
            converted = self.laser.convert_frame(frame)
            power_raw = self.laser.cmd_ask_power_consign()
            power_enabled = bool(frame.statuses.get("STS_RELAY_PSU", 0)) or bool(
                frame.statuses.get("STS_RACK_PSU", 0)
            )
            updated_status = LaserStatus(
                power_enabled=power_enabled,
                interlock_enabled=bool(self.laser.cmd_ask_interlock_sts()),
                second_stage_enabled=bool(self.laser.cmd_ask_secondstage_sts()),
                power_setpoint_percent=power_raw / 65535.0 * 100.0,
                temperature_act_p=converted.TACT_P,
                temperature_set_p=converted.TSET_P,
                imon_pa=converted.IMON_PA,
                lmon=converted.LMON,
                pmon_w=converted.PMON_W,
                connected=True,
                connection_state=ConnectionState.CONNECTED,
            )
            with self._state_lock:
                self.status = updated_status
        except Exception as exc:
            self._handle_serial_error(exc)

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
                    self.laser = ALSLaserController(port=self.serial_port)
                self.laser.connect()
                self.laser.handshake()
                LOGGER.info("Serial connection opened on %s", self.serial_port)
                self._poll_status_locked()
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

    def set_power_percent(self, power_percent: float) -> None:
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.set_power_percent(power_percent)
            LOGGER.info("Power setpoint changed to %.1f%%", power_percent)
            self._poll_status_locked()

    def set_power_supply_on(self) -> bool:
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.cmd_set_power_supply_on()
            LOGGER.info("Power supply turned on")
            self._poll_status_locked()
        return True

    def set_power_supply_off(self) -> bool:
        with self._state_lock:
            if self.status.interlock_enabled:
                LOGGER.warning("cannot turn off power if interlock is on")
                return False
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.cmd_set_power_supply_off()
            LOGGER.info("Power supply turned off")
            self._poll_status_locked()
        return True

    def set_interlock_on(self) -> bool:
        with self._state_lock:
            if not self.status.power_enabled:
                LOGGER.warning("cannot turn on interlock if power is off")
                return False
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.cmd_set_interlock_on()
            LOGGER.info("Interlock turned on")
            self._poll_status_locked()
        return True

    def set_interlock_off(self) -> bool:
        with self._state_lock:
            if self.status.second_stage_enabled:
                LOGGER.warning("cannot turn off interlock if 2nd stage is on")
                return False
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.cmd_set_interlock_off()
            LOGGER.info("Interlock turned off")
            self._poll_status_locked()
        return True

    def set_second_stage_on(self) -> bool:
        with self._state_lock:
            if not self.status.interlock_enabled:
                LOGGER.warning("cannot turn on 2nd stage if interlock is off")
                return False
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.cmd_set_second_stage_on()
            LOGGER.info("2nd stage turned on")
            self._poll_status_locked()
        return True

    def set_second_stage_off(self) -> bool:
        with self._state_lock:
            if abs(self.status.power_setpoint_percent) > 1e-6:
                LOGGER.warning("cannot turn off 2nd stage if laser power setpoint is not 0")
                return False
        with self._laser_lock:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            self.laser.cmd_set_second_stage_off()
            LOGGER.info("2nd stage turned off")
            self._poll_status_locked()
        return True

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
            self.sequence_started_epoch = time.time()
            self.current_step_number = None
            self.current_step_started_epoch = None
            self._interrupt_requested = False
            self.startup_step_states = ["not_done"] * len(self.STARTUP_STEPS)
            self.shutdown_step_states = ["not_done"] * len(self.SHUTDOWN_STEPS)
            self.startup_step_notes = [""] * len(self.STARTUP_STEPS)
            self.shutdown_step_notes = [""] * len(self.SHUTDOWN_STEPS)

        target = self._run_startup_sequence if sequence_type == "STARTUP" else self._run_shutdown_sequence
        self.sequence_thread = threading.Thread(target=target, daemon=True)
        self.sequence_thread.start()
        LOGGER.info("%s sequence requested", sequence_type.title())
        return True

    def interrupt_sequence(self) -> None:
        with self._state_lock:
            self._interrupt_requested = True
        LOGGER.warning("Sequence interrupt requested")

    def _run_startup_sequence(self) -> None:
        self._run_sequence(sequence_type="STARTUP", steps=self.STARTUP_STEPS)

    def _run_shutdown_sequence(self) -> None:
        self._run_sequence(sequence_type="SHUTDOWN", steps=self.SHUTDOWN_STEPS)

    def _run_sequence(self, sequence_type: str, steps: list[tuple[int, str, str]]) -> None:
        try:
            if self.laser is None:
                raise RuntimeError("Laser not connected")
            startup_controller = ALSLaserStartupController(
                self.laser,
                interrogate_callback=self._on_sequence_interrogated_state,
            )
            for step_num, step_name, method_name in steps:
                with self._state_lock:
                    if self._interrupt_requested:
                        LOGGER.warning("%s sequence interrupted", sequence_type.title())
                        self.sequence_state = SequenceState.INTERRUPTED
                        self.current_step_number = None
                        self.current_step_started_epoch = None
                        return
                    self.current_step_number = step_num
                    self.current_step_started_epoch = time.time()
                    self._current_step_already_done = False
                    self._set_step_state(sequence_type, step_num - 1, "doing")
                LOGGER.info("%s step %s: %s", sequence_type.title(), step_num, step_name)
                with self._laser_lock:
                    getattr(startup_controller, method_name)()
                    self._poll_status_locked()
                with self._state_lock:
                    self._set_step_state(sequence_type, step_num - 1, "done")
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    if self._current_step_already_done:
                        self._set_step_note(sequence_type, step_num - 1, f"already done at {timestamp}")
                    else:
                        self._set_step_note(sequence_type, step_num - 1, f"completed at {timestamp}")
                    self._current_step_already_done = False
            with self._state_lock:
                self.sequence_state = SequenceState.COMPLETED
                self.current_step_number = None
                self.current_step_started_epoch = None
            LOGGER.info("%s sequence completed", sequence_type.title())
        except Exception as exc:
            LOGGER.exception("%s sequence failed: %s", sequence_type.title(), exc)
            with self._state_lock:
                self.sequence_state = SequenceState.INTERRUPTED
                self.current_step_number = None
                self.current_step_started_epoch = None
        finally:
            if self.sequence_state != SequenceState.RUNNING:
                pass

    def _on_sequence_interrogated_state(self, state: dict) -> None:
        """Receive controller interrogation snapshots and publish them to GUI clients."""
        updated_status = LaserStatus(
            power_enabled=bool(state.get("power_enabled", False)),
            interlock_enabled=bool(state.get("interlock_enabled", False)),
            second_stage_enabled=bool(state.get("second_stage_enabled", False)),
            power_setpoint_percent=float(state.get("power_setpoint_percent", 0.0)),
            temperature_act_p=float(state.get("temperature_act_p", 0.0)),
            temperature_set_p=float(state.get("temperature_set_p", 0.0)),
            imon_pa=float(state.get("imon_pa", 0.0)),
            lmon=float(state.get("lmon", 0.0)),
            pmon_w=float(state.get("pmon_w", 0.0)),
            connected=bool(state.get("connected", True)),
            connection_state=ConnectionState.CONNECTED,
        )
        context = str(state.get("context", "")).lower()
        with self._state_lock:
            self.status = updated_status
            if "skipped" in context or "already" in context:
                self._current_step_already_done = True

    def _set_step_state(self, sequence_type: str, index: int, state: str) -> None:
        if sequence_type == "STARTUP":
            self.startup_step_states[index] = state
        else:
            self.shutdown_step_states[index] = state

    def _set_step_note(self, sequence_type: str, index: int, note: str) -> None:
        if sequence_type == "STARTUP":
            self.startup_step_notes[index] = note
        else:
            self.shutdown_step_notes[index] = note


def main(host: str = "0.0.0.0", port: int = 5557, serial_port: str = "COM6") -> None:
    logging.basicConfig(level=logging.INFO)
    server = ALSLaserServer(host=host, port=port, serial_port=serial_port)
    server.start()

    def _handle_termination_signal(signum, _frame) -> None:
        LOGGER.warning("Received signal %s; stopping ALS server", signum)
        server.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_termination_signal)
    signal.signal(signal.SIGTERM, _handle_termination_signal)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        LOGGER.info("Stopping ALS server")
    except BaseException:
        LOGGER.exception("ALS server exiting due to unexpected error")
        raise
    finally:
        server.stop()


if __name__ == "__main__":
    main()
