from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading
import time
from typing import Callable, Optional

import serial


LOGGER = logging.getLogger("precilaser_controller")


@dataclass(frozen=True)
class PrecilaserStatus:
    """Decoded snapshot of laser state from the controller status frames."""

    pd_ok: bool
    temperature_ok: bool
    laser_enabled: bool
    power_stability_enabled: bool
    working_current_a: float
    stage_currents_a: list[float] = field(default_factory=list)
    pd_values: list[float] = field(default_factory=list)
    temperatures_c: list[float] = field(default_factory=list)
    raw_system_flags: int = 0
    raw_drive_unlock: int = 0
    raw_payload: bytes = b""


class PrecilaserController:
    """Serial controller for the Precilaser UART protocol.

        Notes:
        - Implements the frame format and register map from
            serial_communication_Amplifier_Protocol-EN.pdf.
        - Protocol requires at least 200 ms between commands.
    """

    FRAME_HEAD = 0x50
    FRAME_TAIL = b"\r\n"
    DEFAULT_HOST = 0x00
    DEFAULT_ADDR = 0x00

    CMD_QUERY_WORKING_STATUS = 0x04
    CMD_QUERY_TEC_TEMPS = 0x05
    CMD_SET_LASER_ENABLE = 0x30
    CMD_SET_WORKING_CURRENT = 0xA1
    CMD_SET_STABILITY_MODE = 0x47

    def __init__(
        self,
        port: str = "COM20",
        baudrate: int = 115200,
        timeout_s: float = 1.0,
        host: int = DEFAULT_HOST,
        address: int = DEFAULT_ADDR,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self.host = host & 0xFF
        self.address = address & 0xFF
        self._ser: Optional[serial.Serial] = None
        self._io_lock = threading.Lock()
        self._last_command_time = 0.0
        self.min_command_interval_s = 0.21

    def connect(self) -> None:
        if self._ser is not None and self._ser.is_open:
            return
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> "PrecilaserController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _xor_checksum(data: bytes) -> int:
        value = 0
        for byte in data:
            value ^= byte
        return value & 0xFF

    @staticmethod
    def _sum_checksum(data: bytes) -> int:
        return sum(data) & 0xFF

    def _require_connection(self) -> serial.Serial:
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("Serial port is not open. Call connect() first.")
        return self._ser

    def _build_frame(self, command: int, payload: bytes = b"") -> bytes:
        body = bytes([
            self.host,
            self.address,
            command & 0xFF,
            len(payload) & 0xFF,
        ]) + payload
        sum_check = self._sum_checksum(body)
        xor_check = self._xor_checksum(body)
        return bytes([self.FRAME_HEAD]) + body + bytes([sum_check, xor_check]) + self.FRAME_TAIL

    def _read_exact(self, nbytes: int) -> bytes:
        ser = self._require_connection()
        data = ser.read(nbytes)
        if len(data) != nbytes:
            raise TimeoutError(f"Expected {nbytes} bytes, received {len(data)} bytes")
        return data

    def _read_frame(self) -> tuple[int, int, int, bytes]:
        while True:
            head = self._read_exact(1)[0]
            if head == self.FRAME_HEAD:
                break
        header = self._read_exact(4)
        host = header[0]
        address = header[1]
        command = header[2]
        payload_len = header[3]
        payload = self._read_exact(payload_len)
        sum_check = self._read_exact(1)[0]
        xor_check = self._read_exact(1)[0]
        tail = self._read_exact(2)
        if tail != self.FRAME_TAIL:
            raise ValueError(f"Invalid frame tail: {tail!r}")
        check_data = header + payload
        expected_sum = self._sum_checksum(check_data)
        expected_xor = self._xor_checksum(check_data)
        if sum_check != expected_sum:
            raise ValueError(
                f"SUM checksum mismatch: expected 0x{expected_sum:02X}, got 0x{sum_check:02X}"
            )
        if xor_check != expected_xor:
            raise ValueError(
                f"XOR checksum mismatch: expected 0x{expected_xor:02X}, got 0x{xor_check:02X}"
            )
        return host, address, command, payload

    @staticmethod
    def _expected_response_commands(command: int) -> set[int]:
        if command == 0x30:
            return {0x40}
        if command == 0xA1:
            return {0x41}
        if command == 0x04:
            return {0x44}
        if command == 0x05:
            return {0x45}
        if command == 0x47:
            return {0x49}
        return set()

    def _send_command(
        self,
        command: int,
        payload: bytes = b"",
        expected_commands: Optional[set[int]] = None,
        max_frames: int = 16,
    ) -> bytes:
        ser = self._require_connection()
        frame = self._build_frame(command, payload)
        with self._io_lock:
            now = time.monotonic()
            wait_time = self.min_command_interval_s - (now - self._last_command_time)
            if wait_time > 0:
                time.sleep(wait_time)
            ser.reset_input_buffer()
            ser.write(frame)
            self._last_command_time = time.monotonic()
            expected = self._expected_response_commands(command) if expected_commands is None else expected_commands
            for _ in range(max_frames):
                _, _, response_cmd, response_payload = self._read_frame()
                if not expected or response_cmd in expected:
                    return response_payload
            expected_text = "any" if not expected else ", ".join(f"0x{x:02X}" for x in sorted(expected))
            raise TimeoutError(
                f"No expected response for command 0x{command:02X}; expected {expected_text}"
            )

    @staticmethod
    def _u16_be(payload: bytes, offset: int) -> Optional[int]:
        if offset + 2 > len(payload):
            return None
        return (payload[offset] << 8) | payload[offset + 1]

    @staticmethod
    def _decode_temperatures_from_payload(payload: bytes) -> list[float]:
        # Response 0x45 payload: spare(1) + TACT[6] + spare(2) + spare(2)
        if len(payload) < 13:
            return []
        payload = payload[1:13]
        temps: list[float] = []
        for i in range(0, len(payload) - 1, 2):
            raw = (payload[i] << 8) | payload[i + 1]
            temps.append(raw / 100.0)
        return temps

    def query_working_status(self) -> PrecilaserStatus:
        payload = self._send_command(self.CMD_QUERY_WORKING_STATUS)

        stable_flag = payload[0] if len(payload) >= 1 else 0
        system_flags = self._u16_be(payload, 2) or 0
        drive_unlock = payload[4] if len(payload) >= 5 else 0

        stage_currents: list[float] = []
        for offset in (7, 14, 21):
            raw_current = self._u16_be(payload, offset)
            if raw_current is not None:
                stage_currents.append(raw_current / 100.0)

        pd_values: list[float] = []
        for offset in (28, 30, 32, 34):
            raw_pd = self._u16_be(payload, offset)
            if raw_pd is None:
                pd_values.append(float("nan"))
            else:
                pd_values.append(float(raw_pd))

        # Keep 5 channels for GUI compatibility; protocol currently defines PD1..PD4.
        pd_values.append(float("nan"))

        temperatures_c: list[float] = []
        for offset in (42, 44, 46, 48):
            raw_temp = self._u16_be(payload, offset)
            if raw_temp is None:
                temperatures_c.append(float("nan"))
            else:
                temperatures_c.append(raw_temp / 100.0)

        if stage_currents:
            working_current_a = max(stage_currents)
        else:
            working_current_a = 0.0

        # Per protocol table: Dn == 0 means normal (no alarm/fault).
        temperature_ok_bits = [(system_flags >> bit) & 0x1 for bit in (8, 9, 10, 11)]
        pd_ok_bits = [(system_flags >> bit) & 0x1 for bit in (4, 5, 6, 7)]

        return PrecilaserStatus(
            pd_ok=all(bit == 0 for bit in pd_ok_bits),
            temperature_ok=all(bit == 0 for bit in temperature_ok_bits),
            laser_enabled=bool(drive_unlock & 0x07),
            power_stability_enabled=bool(stable_flag),
            working_current_a=working_current_a,
            stage_currents_a=stage_currents,
            pd_values=pd_values,
            temperatures_c=temperatures_c,
            raw_system_flags=system_flags,
            raw_drive_unlock=drive_unlock,
            raw_payload=payload,
        )

    def query_tec_temperatures(self) -> list[float]:
        payload = self._send_command(self.CMD_QUERY_TEC_TEMPS)
        return self._decode_temperatures_from_payload(payload)

    def set_laser_enable(self, enabled: bool) -> None:
        if enabled:
            # Manual startup guidance: enable Driver1, then Driver2, then Driver3.
            for drive_unlock_value in (0x01, 0x03, 0x07):
                self._send_command(self.CMD_SET_LASER_ENABLE, payload=bytes([drive_unlock_value]))
        else:
            # Manual shutdown guidance: disable Driver3, then Driver2, then Driver1.
            for drive_unlock_value in (0x03, 0x01, 0x00):
                self._send_command(self.CMD_SET_LASER_ENABLE, payload=bytes([drive_unlock_value]))

    def set_power_stability_mode(self, enabled: bool) -> None:
        stable_flag = 0x01 if enabled else 0x00
        self._send_command(self.CMD_SET_STABILITY_MODE, payload=bytes([stable_flag]))

    def set_working_current(self, current_amps: float) -> None:
        if current_amps < 0:
            raise ValueError("current_amps must be >= 0")
        raw_value = int(round(current_amps * 100.0))
        if raw_value > 0xFFFF:
            raise ValueError("current_amps exceeds protocol range (max 655.35 A)")
        payload = bytes([(raw_value >> 8) & 0xFF, raw_value & 0xFF])
        self._send_command(self.CMD_SET_WORKING_CURRENT, payload=payload)


class PrecilaserStartupController:
    """High-level startup and shutdown procedures from the manual guidance."""

    def __init__(
        self,
        laser: PrecilaserController,
        seed_stabilize_s: float = 5.0,
        laser_stabilize_s: float = 5.0,
        ramp_step_a: float = 1.0,
        ramp_wait_s: float = 5.0,
        target_working_current_a: float = 10.0,
        sleep_fn: Optional[Callable[[float], None]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        if seed_stabilize_s < 0 or laser_stabilize_s < 0:
            raise ValueError("stabilization delays must be >= 0")
        if ramp_step_a <= 0 or ramp_wait_s < 0:
            raise ValueError("ramp_step_a must be > 0 and ramp_wait_s must be >= 0")
        if target_working_current_a < 0:
            raise ValueError("target_working_current_a must be >= 0")

        self.laser = laser
        self.seed_stabilize_s = float(seed_stabilize_s)
        self.laser_stabilize_s = float(laser_stabilize_s)
        self.ramp_step_a = float(ramp_step_a)
        self.ramp_wait_s = float(ramp_wait_s)
        self.target_working_current_a = float(target_working_current_a)
        self.sleep_fn = time.sleep if sleep_fn is None else sleep_fn
        self.log_fn = LOGGER.info if log_fn is None else log_fn
        self.on_step: Optional[Callable[[float], None]] = None

    def _log(self, message: str) -> None:
        now = time.time()
        local = time.localtime(now)
        millis = int((now - int(now)) * 1000)
        stamp = time.strftime("%H:%M:%S", local)
        self.log_fn(f"[{stamp}.{millis:03d}] {message}")

    def _sleep_with_interrupt_check(self, seconds: float, should_continue: Callable[[], bool]) -> None:
        if seconds <= 0:
            return
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            if not should_continue():
                raise RuntimeError("Startup/shutdown interrupted")
            self.sleep_fn(min(0.2, max(0.0, end_time - time.monotonic())))

    @staticmethod
    def _total_current(status: PrecilaserStatus) -> float:
        if status.stage_currents_a:
            return max(sum(status.stage_currents_a), 0.0)
        if math.isfinite(status.working_current_a):
            return max(status.working_current_a, 0.0)
        return 0.0

    def _estimate_current(self) -> float:
        return self._total_current(self.laser.query_working_status())

    def _read_status(self) -> PrecilaserStatus:
        return self.laser.query_working_status()

    def _ramp_to_current(
        self,
        target_current_a: float,
        should_continue: Callable[[], bool],
        on_step: Optional[Callable[[float], None]] = None,
    ) -> list[float]:
        current = self._estimate_current()
        applied: list[float] = []

        if abs(current - target_current_a) <= 1e-9:
            return applied

        step = abs(self.ramp_step_a)
        direction = 1.0 if target_current_a > current else -1.0
        direction_text = "up" if direction > 0 else "down"
        self._log(
            f"Ramping {direction_text} from {current:.2f} A to {target_current_a:.2f} A "
            f"in {step:.2f} A steps with {self.ramp_wait_s:.2f} s wait."
        )
        next_value = current
        step_index = 0
        while True:
            if not should_continue():
                raise RuntimeError("Startup/shutdown interrupted")
            if direction > 0:
                next_value = min(next_value + step, target_current_a)
            else:
                next_value = max(next_value - step, target_current_a)

            self.laser.set_working_current(next_value)
            applied.append(next_value)
            if on_step is not None:
                try:
                    on_step(next_value)
                except Exception:
                    pass
            step_index += 1
            self._log(f"Ramp step {step_index}: working current set to {next_value:.2f} A")

            if abs(next_value - target_current_a) <= 1e-9:
                break
            self._log(f"Waiting {self.ramp_wait_s:.2f} s before next ramp step")
            self._sleep_with_interrupt_check(self.ramp_wait_s, should_continue)

        return applied

    def run_turn_on_procedure(self, should_continue: Optional[Callable[[], bool]] = None) -> dict[str, object]:
        should_continue = (lambda: True) if should_continue is None else should_continue
        initial_status = self._read_status()

        if initial_status.laser_enabled:
            self._log(
                f"Turn-on: laser already enabled at {self._total_current(initial_status):.2f} A; "
                "continuing from existing current."
            )
        else:
            self._log("Turn-on: setting working current to 0 A.")
            self.laser.set_working_current(0.0)

            self._log("Turn-on: wait for seed-laser thermal stabilization.")
            self._sleep_with_interrupt_check(self.seed_stabilize_s, should_continue)

            self._log("Turn-on: wait for amplifier/laser current stabilization.")
            self._sleep_with_interrupt_check(self.laser_stabilize_s, should_continue)

            self._log("Turn-on: enabling laser drivers.")
            self.laser.set_laser_enable(True)

        self._log(f"Turn-on: ramping current to {self.target_working_current_a:.2f} A.")
        ramp_steps = self._ramp_to_current(self.target_working_current_a, should_continue, self.on_step)

        if initial_status.power_stability_enabled:
            self._log("Turn-on: power-stability mode already enabled.")
        else:
            self._log("Turn-on: enabling power-stability mode.")
            self.laser.set_power_stability_mode(True)

        final_status = self._read_status()
        return {
            "ramp_steps_a": ramp_steps,
            "final_status": final_status,
        }

    def run_turn_off_procedure(self, should_continue: Optional[Callable[[], bool]] = None) -> dict[str, object]:
        should_continue = (lambda: True) if should_continue is None else should_continue
        initial_status = self._read_status()

        if initial_status.power_stability_enabled:
            self._log("Turn-off: disabling power-stability mode.")
            self.laser.set_power_stability_mode(False)
        else:
            self._log("Turn-off: power-stability mode already disabled.")

        self._log("Turn-off: ramping working current to 0 A.")
        ramp_steps = self._ramp_to_current(0.0, should_continue, self.on_step)
        if ramp_steps:
            self._log(
                f"Turn-off: hold at 0 A for {self.ramp_wait_s:.2f} s before disabling drivers."
            )
            self._sleep_with_interrupt_check(self.ramp_wait_s, should_continue)

        if initial_status.laser_enabled:
            self._log("Turn-off: disabling laser drivers.")
            self.laser.set_laser_enable(False)
        else:
            self._log("Turn-off: laser drivers already disabled.")

        final_status = self._read_status()
        return {
            "ramp_steps_a": ramp_steps,
            "final_status": final_status,
        }
