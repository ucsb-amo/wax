from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import math
import struct
import time

import serial


@dataclass(frozen=True)
class FirmwareInfo:
    firmware_type: str
    major: int
    minor: int
    variant: str
    raw_response: bytes


@dataclass(frozen=True)
class SerialNumberInfo:
    year_code: int
    firmware_type: str
    serial_number: int
    raw_response: bytes


@dataclass(frozen=True)
class ReadAllFrame:
    analog: dict[str, float]
    statuses: dict[str, int]
    raw_response: bytes


@dataclass(frozen=True)
class ConvertedFrame:
    """Physical-unit values derived from a ReadAllFrame.

    Transfer functions (vendor docs):
      TSET-P / TACT-P / TSET-H / TACT-H  →  °C  (Steinhart-Hart NTC)
          T = 1 / (ln(V) / 3492 + 1/298.15) - 273.15
      IMON-PA / IMON-A / LMON             →  A
          I = V * 5
      PMON                                →  W
          P = V / Vref * Pmax
    """
    TSET_P: float
    TACT_P: float
    TSET_H: float
    TACT_H: float
    LMON: float
    PMON_W: float
    PMON_V: float
    IMON_PA: float
    IMON_A: float
    statuses: dict[str, int]

    @property
    def PMON(self) -> float:
        """Backward-compatible alias returning PMON power in watts."""
        return self.PMON_W


def _voltage_to_celsius(v: float) -> float:
    """Steinhart-Hart NTC: T[°C] = 1/(ln(V)/3492 + 1/298.15) - 273.15"""
    if v <= 0:
        return float("nan")
    return 1.0 / (math.log(v) / 3492.0 + 1.0 / 298.15) - 273.15


def _voltage_to_current(v: float) -> float:
    """I [A] = V * 5"""
    return v * 5.0


def _voltage_to_power(v: float, max_power_watts: float, pmon_full_scale_voltage: float) -> float:
    """Scale PMON voltage to optical power using a user-supplied calibration."""
    if pmon_full_scale_voltage <= 0:
        raise ValueError("pmon_full_scale_voltage must be > 0")
    return v / pmon_full_scale_voltage * max_power_watts


class ALSLaserController:
    """Serial controller for ALS laser hardware using the RCI protocol."""

    _EXPECTED_FW_TYPE = 0x42  # 'B'
    _CMD_READ_ALL = 0x01
    _READ_ALL_FRAME_LEN = 0x28

    _ANALOG_LABELS = (
        "TSET-P",
        "TACT-P",
        "LMON",
        "PMON",
        "TACT-H",
        "TSET-H",
        "IMON-PA",
        "IMON-A",
    )

    _STATUS_LABELS = (
        "STS_INTLK",
        "STS_ACOK",
        "STS_RELAY_PSU",
        "STS_RACK_PSU",
    )

    _ERROR_CODES = {
        0x00: "STS_NO_ERROR",
        0x01: "STS_ERROR_CHECKSUM",
        0x02: "STS_UNKNOWN_CMD",
        0x10: "STS_ERROR_I2C_WRITE",
        0x11: "STS_ERROR_ADC1_NOTFOUND",
        0x14: "STS_ERROR_DELAY_TOO_SHORT",
    }

    def __init__(
        self,
        port: str = "COM6",
        baudrate: int = 115200,
        timeout: float = 1.0,
        max_power_watts: float = 46.29,
        pmon_full_scale_voltage: float = 1.949,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self.set_pmon_calibration(
            max_power_watts=max_power_watts,
            pmon_full_scale_voltage=pmon_full_scale_voltage,
        )

    @staticmethod
    def crc8_maxim(data: bytes) -> int:
        """CRC-8/MAXIM over bytes from LENGTH through last DATA byte."""
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x01:
                    crc = (crc >> 1) ^ 0x8C
                else:
                    crc >>= 1
        return crc & 0xFF

    def connect(self) -> None:
        if self._ser is not None and self._ser.is_open:
            return
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def close(self) -> None:
        ser = self._ser
        self._ser = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def __del__(self) -> None:
        """Best-effort cleanup so the serial port is released on abnormal teardown."""
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "ALSLaserController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_pmon_calibration(self, max_power_watts: float, pmon_full_scale_voltage: float) -> None:
        """Set the PMON scaling used to convert monitor voltage to optical power."""
        if max_power_watts <= 0:
            raise ValueError("max_power_watts must be > 0")
        if pmon_full_scale_voltage <= 0:
            raise ValueError("pmon_full_scale_voltage must be > 0")
        self.max_power_watts = float(max_power_watts)
        self.pmon_full_scale_voltage = float(pmon_full_scale_voltage)

    def _require_connection(self) -> serial.Serial:
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("Serial port is not open. Call connect() first.")
        return self._ser

    def _read_exact(self, nbytes: int) -> bytes:
        ser = self._require_connection()
        data = ser.read(nbytes)
        if len(data) != nbytes:
            raise TimeoutError(
                f"Expected {nbytes} bytes, received {len(data)} bytes: {data.hex(' ')}"
            )
        return data

    def _read_frame(self) -> bytes:
        first = self._read_exact(1)
        frame_len = first[0]
        if frame_len < 4:
            raise ValueError(f"Invalid frame length byte: 0x{frame_len:02X}")
        return first + self._read_exact(frame_len - 1)

    def _validate_crc(self, frame: bytes) -> None:
        expected = self.crc8_maxim(frame[:-1])
        received = frame[-1]
        if expected != received:
            raise ValueError(
                f"CRC mismatch: expected 0x{expected:02X}, received 0x{received:02X}"
            )

    def _build_command_frame(self, command: int, payload: bytes = b"") -> bytes:
        frame_no_crc = bytes([3 + len(payload), command]) + payload
        crc = self.crc8_maxim(frame_no_crc)
        return frame_no_crc + bytes([crc])

    def _read_response_for_expected(self, expected_command: int, max_frames: int = 128) -> bytes:
        """Wait for expected command while ignoring interleaved CMD_READ_ALL frames."""
        for _ in range(max_frames):
            response = self._read_frame()
            self._validate_crc(response)
            cmd = response[1]
            if cmd == expected_command:
                return response
            if cmd == self._CMD_READ_ALL:
                continue
            raise ValueError(
                f"Unexpected response command while waiting for 0x{expected_command:02X}: "
                f"got 0x{cmd:02X}"
            )
        raise TimeoutError(
            f"Did not receive expected response command 0x{expected_command:02X} "
            f"within {max_frames} frames"
        )

    def _send_command(
        self,
        command: int,
        payload: bytes = b"",
        expected_command: Optional[int] = None,
        require_no_error_status: bool = True,
        clear_stale_input: bool = True,
    ) -> bytes:
        ser = self._require_connection()
        if clear_stale_input:
            ser.reset_input_buffer()
        ser.write(self._build_command_frame(command, payload))
        expected = command if expected_command is None else expected_command
        response = self._read_response_for_expected(expected_command=expected)
        status = response[2]
        if require_no_error_status and status != 0x00:
            status_name = self._ERROR_CODES.get(status, "UNKNOWN_STATUS")
            raise RuntimeError(
                f"Command 0x{command:02X} failed with status 0x{status:02X} ({status_name})"
            )
        return response

    def _expect_no_data_ack(self, response: bytes, command: int) -> None:
        if response[0] != 0x04:
            raise ValueError(
                f"Expected 4-byte ACK for command 0x{command:02X}, got length 0x{response[0]:02X}"
            )

    def _expect_u16_response_or_ack(self, response: bytes, command: int) -> Optional[int]:
        if response[0] == 0x04:
            return None
        if response[0] == 0x06:
            return (response[3] << 8) | response[4]
        raise ValueError(
            f"Expected 4-byte ACK or 6-byte u16 response for command 0x{command:02X}, "
            f"got length 0x{response[0]:02X}"
        )

    # ------------------------------------------------------------------ #
    #  Identification                                                       #
    # ------------------------------------------------------------------ #

    def handshake(self, require_no_error_status: bool = True) -> FirmwareInfo:
        """0x03 CMD_GET_FW_INFO"""
        response = self._send_command(0x03, require_no_error_status=require_no_error_status)
        if response[0] != 0x08:
            raise ValueError(f"Bad FW info frame length: 0x{response[0]:02X}")
        fw_type = response[3]
        if fw_type != self._EXPECTED_FW_TYPE:
            raise RuntimeError(
                f"Unexpected firmware type: expected 0x42 ('B'), got 0x{fw_type:02X}"
            )
        return FirmwareInfo(
            firmware_type=chr(response[3]),
            major=response[4],
            minor=response[5],
            variant=chr(response[6]),
            raw_response=response,
        )

    def cmd_get_serial(self, require_no_error_status: bool = True) -> SerialNumberInfo:
        """0x04 CMD_GET_SERIAL"""
        response = self._send_command(0x04, require_no_error_status=require_no_error_status)
        if response[0] != 0x08:
            raise ValueError(f"Bad serial response length: 0x{response[0]:02X}")
        return SerialNumberInfo(
            year_code=response[3],
            firmware_type=chr(response[4]),
            serial_number=(response[5] << 8) | response[6],
            raw_response=response,
        )

    # ------------------------------------------------------------------ #
    #  Configuration                                                        #
    # ------------------------------------------------------------------ #

    def cmd_change_poll_rate(self, interval_ms: int, require_no_error_status: bool = True) -> None:
        """0x02 CMD_CHANGE_POLL_RATE"""
        if not 10 <= interval_ms <= 0xFFFF:
            raise ValueError("interval_ms must be in [10, 65535]")
        payload = bytes([(interval_ms >> 8) & 0xFF, interval_ms & 0xFF])
        response = self._send_command(0x02, payload=payload, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(response, 0x02)

    # ------------------------------------------------------------------ #
    #  Power / stage control                                               #
    # ------------------------------------------------------------------ #

    def cmd_set_power_supply_on(self, require_no_error_status: bool = True) -> None:
        """0x08 CMD_SET_POWER_SUPPLY_ON: turn the laser rack PSU on."""
        response = self._send_command(0x08, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(response, 0x08)

    def cmd_set_power_supply_off(self, require_no_error_status: bool = True) -> None:
        """0x09 CMD_SET_POWER_SUPPLY_OFF: turn the laser rack PSU off."""
        response = self._send_command(0x09, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(response, 0x09)

    def cmd_set_second_stage_on(self, require_no_error_status: bool = True) -> None:
        """0x0A CMD_SET_SECOND_STAGE_ON: enable the amplifier stage."""
        response = self._send_command(0x0A, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(response, 0x0A)

    def cmd_set_second_stage_off(self, require_no_error_status: bool = True) -> None:
        """0x0B CMD_SET_SECOND_STAGE_OFF: disable the amplifier stage."""
        response = self._send_command(0x0B, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(response, 0x0B)

    def cmd_set_interlock_on(self, require_no_error_status: bool = True) -> None:
        """0x06 CMD_SET_INTERLOCK_ON"""
        response = self._send_command(0x06, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(response, 0x06)

    def cmd_set_interlock_off(self, require_no_error_status: bool = True) -> None:
        """0x07 CMD_SET_INTERLOCK_OFF"""
        response = self._send_command(0x07, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(response, 0x07)

    def cmd_set_power_consign(self, consign_16bit: int, require_no_error_status: bool = True) -> Optional[int]:
        """0x0C CMD_SET_POWER_CONSIGN  (16-bit, 0 – 65535 maps to 0 – Pmax).

        Some controllers return a bare 4-byte ACK, while others return a 6-byte
        response echoing the applied 16-bit consign.
        """
        if not 0 <= consign_16bit <= 0xFFFF:
            raise ValueError("consign_16bit must be in [0, 65535]")
        payload = bytes([(consign_16bit >> 8) & 0xFF, consign_16bit & 0xFF])
        response = self._send_command(0x0C, payload=payload, require_no_error_status=require_no_error_status)
        echoed_consign = self._expect_u16_response_or_ack(response, 0x0C)
        if echoed_consign is not None and echoed_consign != consign_16bit:
            raise ValueError(
                f"CMD_SET_POWER_CONSIGN echo mismatch: sent {consign_16bit}, got {echoed_consign}"
            )
        return echoed_consign

    def set_power_percent(self, percent: float, require_no_error_status: bool = True) -> Optional[int]:
        """Set output power as a percentage of Pmax (0.0 – 100.0).

        Converts to the 16-bit consign value for CMD_SET_POWER_CONSIGN:
          0 % → 0,  100 % → 65535
        """
        if not 0.0 <= percent <= 100.0:
            raise ValueError(f"percent must be in [0.0, 100.0], got {percent}")
        return self.cmd_set_power_consign(
            round(percent / 100.0 * 65535),
            require_no_error_status=require_no_error_status,
        )

    # ------------------------------------------------------------------ #
    #  Status queries                                                       #
    # ------------------------------------------------------------------ #

    def cmd_ask_poll_interval(self, require_no_error_status: bool = True) -> int:
        """0x22 CMD_ASK_POLL_INTERVAL – returns interval in ms"""
        response = self._send_command(0x22, require_no_error_status=require_no_error_status)
        if response[0] != 0x06:
            raise ValueError(f"Bad poll interval response length: 0x{response[0]:02X}")
        return (response[3] << 8) | response[4]

    def cmd_ask_power_sts(self, require_no_error_status: bool = True) -> int:
        """0x23 CMD_ASK_POWER_STS"""
        response = self._send_command(0x23, require_no_error_status=require_no_error_status)
        if response[0] != 0x05:
            raise ValueError(f"Bad power status response length: 0x{response[0]:02X}")
        return response[3]

    def cmd_ask_power_consign(self, require_no_error_status: bool = True) -> int:
        """0x24 CMD_ASK_POWER_CONSIGN – returns raw 16-bit value"""
        response = self._send_command(0x24, require_no_error_status=require_no_error_status)
        if response[0] != 0x06:
            raise ValueError(f"Bad power consign response length: 0x{response[0]:02X}")
        return (response[3] << 8) | response[4]

    def cmd_ask_secondstage_sts(self, require_no_error_status: bool = True) -> int:
        """0x25 CMD_ASK_SECONDSTAGE_STS"""
        response = self._send_command(0x25, require_no_error_status=require_no_error_status)
        if response[0] != 0x05:
            raise ValueError(f"Bad second stage status response length: 0x{response[0]:02X}")
        return response[3]

    def cmd_ask_interlock_sts(self, require_no_error_status: bool = True) -> int:
        """0x26 CMD_ASK_INTERLOCK_STS"""
        response = self._send_command(0x26, require_no_error_status=require_no_error_status)
        if response[0] != 0x05:
            raise ValueError(f"Bad interlock status response length: 0x{response[0]:02X}")
        return response[3]

    # ------------------------------------------------------------------ #
    #  Streaming / single read                                             #
    # ------------------------------------------------------------------ #

    def authorize_transfer(self, require_no_error_status: bool = True) -> None:
        """0x10 CMD_AUTHORIZE_XFER: start periodic CMD_READ_ALL stream (~50 ms)."""
        response = self._send_command(0x10, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(response, 0x10)

    def cmd_stop_xfer(self, require_no_error_status: bool = True) -> None:
        """0x11 CMD_STOP_XFER: stop the continuous CMD_READ_ALL stream."""
        response = self._send_command(0x11, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(response, 0x11)

    def cmd_get_single_read(self, require_no_error_status: bool = True) -> ReadAllFrame:
        """0x12 CMD_GET_SINGLE_READ: ACK + one CMD_READ_ALL snapshot. Stream must be stopped."""
        ack = self._send_command(0x12, require_no_error_status=require_no_error_status)
        self._expect_no_data_ack(ack, 0x12)
        return self.read_all_once(require_no_error_status=require_no_error_status)

    def stop_and_read(self, require_no_error_status: bool = True) -> ReadAllFrame:
        """CMD_STOP_XFER → CMD_GET_SINGLE_READ. Safe to call anytime."""
        try:
            self.cmd_stop_xfer(require_no_error_status=False)
        except Exception:
            pass
        return self.cmd_get_single_read(require_no_error_status=require_no_error_status)

    def stop_and_print(self, precision: int = 3) -> ReadAllFrame:
        """Stop stream, grab one snapshot, print converted values, return the frame."""
        frame = self.stop_and_read()
        print(self.format_converted(self.convert_frame(frame), precision=precision))
        return frame

    # ------------------------------------------------------------------ #
    #  Frame parsing                                                        #
    # ------------------------------------------------------------------ #

    def read_all_once(self, require_no_error_status: bool = True) -> ReadAllFrame:
        """Read and parse the next CMD_READ_ALL (0x01) frame from the bus."""
        frame = self._read_frame()
        if frame[0] != self._READ_ALL_FRAME_LEN:
            raise ValueError(f"Bad frame length byte: expected 0x28, got 0x{frame[0]:02X}")
        if frame[1] != self._CMD_READ_ALL:
            raise ValueError(f"Unexpected command byte: expected 0x01, got 0x{frame[1]:02X}")
        self._validate_crc(frame)
        status = frame[2]
        if require_no_error_status and status != 0x00:
            status_name = self._ERROR_CODES.get(status, "UNKNOWN_STATUS")
            raise RuntimeError(f"CMD_READ_ALL failed with status 0x{status:02X} ({status_name})")
        analog_block = frame[3:35]
        status_block = frame[35:39]
        analog: dict[str, float] = {}
        for i, label in enumerate(self._ANALOG_LABELS):
            start = 4 * i
            analog[label] = struct.unpack(">f", analog_block[start:start + 4])[0]
        statuses = {self._STATUS_LABELS[i]: status_block[i] for i in range(len(self._STATUS_LABELS))}
        return ReadAllFrame(analog=analog, statuses=statuses, raw_response=frame)

    # ------------------------------------------------------------------ #
    #  Unit conversion                                                      #
    # ------------------------------------------------------------------ #

    def convert_frame(self, frame: ReadAllFrame) -> ConvertedFrame:
        """Apply vendor transfer functions to produce physical-unit values."""
        a = frame.analog
        return ConvertedFrame(
            TSET_P=_voltage_to_celsius(a["TSET-P"]),
            TACT_P=_voltage_to_celsius(a["TACT-P"]),
            TSET_H=_voltage_to_celsius(a["TSET-H"]),
            TACT_H=_voltage_to_celsius(a["TACT-H"]),
            LMON=_voltage_to_current(a["LMON"]),
            PMON_W=_voltage_to_power(
                a["PMON"],
                max_power_watts=self.max_power_watts,
                pmon_full_scale_voltage=self.pmon_full_scale_voltage,
            ),
            PMON_V=a["PMON"],
            IMON_PA=_voltage_to_current(a["IMON-PA"]),
            IMON_A=_voltage_to_current(a["IMON-A"]),
            statuses=frame.statuses,
        )

    def format_converted(self, cf: ConvertedFrame, precision: int = 3) -> str:
        p = precision
        sts = " | ".join(f"{k}: {v}" for k, v in cf.statuses.items())
        return (
            f"TSET-P: {cf.TSET_P:{p+5}.{p}f} °C | "
            f"TACT-P: {cf.TACT_P:{p+5}.{p}f} °C | "
            f"TSET-H: {cf.TSET_H:{p+5}.{p}f} °C | "
            f"TACT-H: {cf.TACT_H:{p+5}.{p}f} °C | "
            f"LMON: {cf.LMON:{p+4}.{p}f} A | "
            f"PMON: {cf.PMON_W:{p+5}.{p}f} W ({cf.PMON_V:{p+4}.{p}f} V) | "
            f"IMON-PA: {cf.IMON_PA:{p+4}.{p}f} A | "
            f"IMON-A: {cf.IMON_A:{p+4}.{p}f} A || "
            f"{sts}"
        )

    def format_read_all(self, frame: ReadAllFrame, precision: int = 3) -> str:
        """Format raw voltage values (use format_converted for physical units)."""
        analog_part = " | ".join(
            f"{label}: {frame.analog[label]:.{precision}f}"
            for label in self._ANALOG_LABELS
        )
        status_part = " | ".join(
            f"{label}: {frame.statuses[label]}"
            for label in self._STATUS_LABELS
        )
        return f"{analog_part} || {status_part}"

    # ------------------------------------------------------------------ #
    #  Continuous stream loop                                               #
    # ------------------------------------------------------------------ #

    def cmd_read_all_loop(
        self,
        max_frames: Optional[int] = None,
        require_no_error_status: bool = True,
        precision: int = 3,
    ) -> None:
        """Authorize the stream and print converted values for every frame."""
        self.authorize_transfer(require_no_error_status=require_no_error_status)
        print("Started CMD_READ_ALL stream. Press interrupt to stop.")
        count = 0
        while max_frames is None or count < max_frames:
            frame = self.read_all_once(require_no_error_status=require_no_error_status)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{stamp}] {self.format_converted(self.convert_frame(frame), precision=precision)}")
            count += 1

class ALSLaserStartupController:
    """High-level wrapper that executes the recommended laser turn-on procedure."""

    def __init__(
        self,
        laser: ALSLaserController,
        imon_pa_threshold_amps: float = 3.0,
        ramp_step_percent: float = 10.0,
        ramp_wait_seconds: float = 15.0,
        turn_off_wait_seconds: float = 3.0,
        warmup_seconds: float = 30.0 * 60.0,
        poll_interval_seconds: float = 1.0,
        sleep_fn=None,
        interrogate_callback: Optional[Callable[[dict], None]] = None,
        should_interrupt: Optional[Callable[[], bool]] = None,
    ):
        if imon_pa_threshold_amps <= 0:
            raise ValueError("imon_pa_threshold_amps must be > 0")
        if ramp_step_percent <= 0 or ramp_step_percent > 100:
            raise ValueError("ramp_step_percent must be in (0, 100]")
        if ramp_wait_seconds < 0:
            raise ValueError("ramp_wait_seconds must be >= 0")
        if warmup_seconds < 0:
            raise ValueError("warmup_seconds must be >= 0")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")

        self.laser = laser
        self.imon_pa_threshold_amps = float(imon_pa_threshold_amps)
        self.ramp_step_percent = float(ramp_step_percent)
        self.ramp_wait_seconds = float(ramp_wait_seconds)
        self.turn_off_wait_seconds = float(turn_off_wait_seconds)
        self.warmup_seconds = float(warmup_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.sleep_fn = time.sleep if sleep_fn is None else sleep_fn
        self.interrogate_callback = interrogate_callback
        self.should_interrupt = should_interrupt

    def _log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] {message}")

    def _sleep_with_log(self, seconds: float, reason: str) -> None:
        if seconds <= 0:
            return
        self._log(f"Waiting {seconds:.1f} s for {reason}.")
        self._sleep_interruptibly(seconds)

    def _interrupt_requested(self) -> bool:
        return bool(self.should_interrupt is not None and self.should_interrupt())

    def _raise_if_interrupted(self) -> None:
        if self._interrupt_requested():
            self._log("Interrupt requested; aborting current wait.")
            raise InterruptedError("Sequence interrupted")

    def _sleep_interruptibly(self, seconds: float) -> None:
        remaining = float(seconds)
        while remaining > 0:
            self._raise_if_interrupted()
            chunk = min(remaining, 1.0)
            self.sleep_fn(chunk)
            remaining -= chunk
        self._raise_if_interrupted()

    def _read_converted_snapshot(self) -> ConvertedFrame:
        frame = self.laser.stop_and_read()
        converted = self.laser.convert_frame(frame)
        self._log(
            "Snapshot: "
            f"IMON-PA={converted.IMON_PA:.3f} A, "
            f"PMON={converted.PMON_W:.3f} W ({converted.PMON_V:.3f} V)"
        )
        return converted

    def _interrogate_laser_state(self, context: str) -> ConvertedFrame:
        """Query and optionally publish current laser state for external UIs."""
        frame = self.laser.stop_and_read()
        converted = self.laser.convert_frame(frame)
        power_enabled = bool(frame.statuses.get("STS_RACK_PSU", 0)) and bool(
            frame.statuses.get("STS_RELAY_PSU", 0)
        )
        interlock_enabled = bool(self.laser.cmd_ask_interlock_sts())
        second_stage_enabled = bool(self.laser.cmd_ask_secondstage_sts())
        power_setpoint_percent = self.laser.cmd_ask_power_consign() / 65535.0 * 100.0

        self._log(
            f"Interrogate ({context}): "
            f"power={'ON' if power_enabled else 'OFF'}, "
            f"interlock={'ON' if interlock_enabled else 'OFF'}, "
            f"2nd_stage={'ON' if second_stage_enabled else 'OFF'}, "
            f"setpoint={power_setpoint_percent:.1f}%, "
            f"IMON-PA={converted.IMON_PA:.3f} A, PMON={converted.PMON_W:.3f} W"
        )

        if self.interrogate_callback is not None:
            self.interrogate_callback(
                {
                    "context": context,
                    "power_enabled": power_enabled,
                    "interlock_enabled": interlock_enabled,
                    "second_stage_enabled": second_stage_enabled,
                    "power_setpoint_percent": power_setpoint_percent,
                    "temperature_act_p": converted.TACT_P,
                    "temperature_set_p": converted.TSET_P,
                    "imon_pa": converted.IMON_PA,
                    "lmon": converted.LMON,
                    "pmon_w": converted.PMON_W,
                    "connected": True,
                }
            )

        return converted

    def _is_power_enabled(self) -> bool:
        frame = self.laser.stop_and_read()
        rack_or_relay = bool(frame.statuses.get("STS_RACK_PSU", 0)) and bool(
            frame.statuses.get("STS_RELAY_PSU", 0)
        )
        return rack_or_relay

    def _is_interlock_enabled(self) -> bool:
        return bool(self.laser.cmd_ask_interlock_sts())

    def _is_second_stage_enabled(self) -> bool:
        return bool(self.laser.cmd_ask_secondstage_sts())

    def step_1_turn_laser_power_on(self) -> None:
        if self._is_power_enabled():
            self._log("Step 1: laser power supply already on; skipping.")
            self._interrogate_laser_state("after step 1 (skipped)")
            return
        self._log("Step 1: turning laser power supply on.")
        self.laser.cmd_set_power_supply_on()
        self._interrogate_laser_state("after step 1")

    def step_2_turn_interlock_on(self) -> None:
        if self._is_interlock_enabled():
            self._log("Step 2: interlock already enabled; skipping.")
            self._interrogate_laser_state("after step 2 (skipped)")
            return
        self._log("Step 2: enabling interlock.")
        self.laser.cmd_set_interlock_on()
        self._interrogate_laser_state("after step 2")

    def step_3_wait_for_imon_pa(self) -> ConvertedFrame:
        self._log(
            f"Step 3: polling IMON-PA until it exceeds {self.imon_pa_threshold_amps:.3f} A."
        )
        converted = self._read_converted_snapshot()
        if converted.IMON_PA > self.imon_pa_threshold_amps:
            self._log(
                f"Step 3: IMON-PA already above threshold ({converted.IMON_PA:.3f} A); skipping wait."
            )
            self._interrogate_laser_state("after step 3 (already above threshold)")
            return converted
        while True:
            self._raise_if_interrupted()
            converted = self._read_converted_snapshot()
            if converted.IMON_PA > self.imon_pa_threshold_amps:
                self._log(
                    f"IMON-PA threshold reached: {converted.IMON_PA:.3f} A > "
                    f"{self.imon_pa_threshold_amps:.3f} A."
                )
                self._interrogate_laser_state("after step 3 (threshold reached)")
                return converted
            self._sleep_interruptibly(self.poll_interval_seconds)

    def step_4_turn_on_second_stage(self) -> None:
        if self._is_second_stage_enabled():
            self._log("Step 4: second stage already enabled; skipping.")
            self._interrogate_laser_state("after step 4 (skipped)")
            return
        self._log("Step 4: enabling second stage.")
        self.laser.cmd_set_second_stage_on()
        self._interrogate_laser_state("after step 4")

    def step_5_ramp_to_80_percent(self) -> list[float]:
        current_percent = self._get_current_power_percent()
        self._log(
            "Step 5: ramping laser power to 80 percent from "
            f"{current_percent:.1f}% in {self.ramp_step_percent:.0f}% steps."
        )
        applied_steps: list[float] = []
        if current_percent >= 80.0 - 1e-9:
            self._log("Step 5: power already at or above 80%; skipping ramp.")
            self._interrogate_laser_state("after step 5 (skipped)")
            return applied_steps

        step = self.ramp_step_percent
        percent = min(max(current_percent + step, step), 80.0)
        while percent <= 80.0 + 1e-9:
            target_percent = min(percent, 80.0)
            echoed = self.laser.set_power_percent(target_percent)
            applied_steps.append(target_percent)
            if echoed is None:
                self._log(f"Set power to {target_percent:.0f}%.")
            else:
                self._log(f"Set power to {target_percent:.0f}% (echoed raw consign={echoed}).")
            self._interrogate_laser_state(f"step 5 ramp point {target_percent:.0f}%")
            if target_percent < 80.0:
                self._sleep_with_log(self.ramp_wait_seconds, f"power-settle at {target_percent:.0f}%")
            percent += step
        return applied_steps

    def step_6_warm_up_at_80_percent(self) -> None:
        if self._get_current_power_percent() >= 100.0 - 1e-9:
            self._log("Step 6: power already at 100%; skipping warm-up hold.")
            self._interrogate_laser_state("after step 6 (skipped)")
            return
        self._log("Step 6: warm-up hold at 80 percent.")
        self._sleep_with_log(self.warmup_seconds, "laser warm-up at 80%")
        self._interrogate_laser_state("after step 6")

    def step_7_turn_to_100_percent(self) -> Optional[int]:
        if self._get_current_power_percent() >= 100.0 - 1e-9:
            self._log("Step 7: laser power already at 100%; skipping.")
            self._interrogate_laser_state("after step 7 (skipped)")
            return None
        self._log("Step 7: setting laser power to 100 percent.")
        echoed = self.laser.set_power_percent(100.0)
        if echoed is None:
            self._log("Laser power set to 100%.")
        else:
            self._log(f"Laser power set to 100% (echoed raw consign={echoed}).")
        self._interrogate_laser_state("after step 7")
        return echoed

    def _get_current_power_percent(self) -> float:
        raw_consign = self.laser.cmd_ask_power_consign()
        return raw_consign / 65535.0 * 100.0

    def step_1_ramp_down_to_zero_percent(self) -> list[float]:
        current_percent = self._get_current_power_percent()
        self._log(
            "Turn-off Step 1: ramping laser power down from "
            f"{current_percent:.1f}% to 0% in {self.ramp_step_percent:.0f}% steps."
        )
        applied_steps: list[float] = []
        target_percent = max(current_percent - self.ramp_step_percent, 0.0)
        while True:
            echoed = self.laser.set_power_percent(target_percent)
            applied_steps.append(target_percent)
            if echoed is None:
                self._log(f"Set power to {target_percent:.1f}%.")
            else:
                self._log(f"Set power to {target_percent:.1f}% (echoed raw consign={echoed}).")
            self._interrogate_laser_state(f"turn-off ramp point {target_percent:.1f}%")
            if target_percent <= 0.0:
                break
            self._sleep_with_log(self.ramp_wait_seconds, f"power-settle at {target_percent:.1f}%")
            target_percent = max(target_percent - self.ramp_step_percent, 0.0)
        return applied_steps

    def step_2_turn_off_second_stage(self) -> None:
        if not self._is_second_stage_enabled():
            self._log("Turn-off Step 2: second stage already disabled; skipping.")
            self._interrogate_laser_state("after turn-off step 2 (skipped)")
            return
        self._log("Turn-off Step 2: disabling second stage.")
        self.laser.cmd_set_second_stage_off()
        self._sleep_with_log(self.turn_off_wait_seconds, "second stage shutdown")
        self._interrogate_laser_state("after turn-off step 2")

    def step_3_turn_off_interlock(self) -> None:
        if not self._is_interlock_enabled():
            self._log("Turn-off Step 3: interlock already disabled; skipping.")
            self._interrogate_laser_state("after turn-off step 3 (skipped)")
            return
        self._log("Turn-off Step 3: disabling interlock.")
        self.laser.cmd_set_interlock_off()
        self._sleep_with_log(self.turn_off_wait_seconds, "interlock shutdown")
        self._interrogate_laser_state("after turn-off step 3")

    def step_4_turn_off_laser_power(self) -> None:
        if not self._is_power_enabled():
            self._log("Turn-off Step 4: laser power supply already off; skipping.")
            self._interrogate_laser_state("after turn-off step 4 (skipped)")
            return
        self._log("Turn-off Step 4: turning laser power supply off.")
        self.laser.cmd_set_power_supply_off()
        self._interrogate_laser_state("after turn-off step 4")

    def run_turn_on_procedure(self) -> dict[str, object]:
        self.step_1_turn_laser_power_on()
        self.step_2_turn_interlock_on()
        threshold_frame = self.step_3_wait_for_imon_pa()
        self.step_4_turn_on_second_stage()
        ramp_steps = self.step_5_ramp_to_80_percent()
        self.step_6_warm_up_at_80_percent()
        final_echo = self.step_7_turn_to_100_percent()
        return {
            "threshold_frame": threshold_frame,
            "ramp_steps_percent": ramp_steps,
            "final_echo": final_echo,
        }

    def run_turn_off_procedure(self) -> dict[str, object]:
        ramp_steps = self.step_1_ramp_down_to_zero_percent()
        self.step_2_turn_off_second_stage()
        self.step_3_turn_off_interlock()
        self.step_4_turn_off_laser_power()
        return {
            "ramp_steps_percent": ramp_steps,
        }