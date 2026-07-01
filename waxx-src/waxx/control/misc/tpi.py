"""TPI-1005-A serial driver (binary protocol, AN-2 rev 1.27)."""

import struct
import threading
import time
from dataclasses import dataclass

import serial


BAUD = 3_000_000
QUALIFIER = bytes([0xAA, 0x55])


@dataclass
class DeviceState:
    model: str = ""
    serial_number: str = ""
    firmware: str = ""
    freq_mhz: float = 0.0
    level_dbm: int = 0
    rf_on: bool = False


class TPIError(Exception):
    pass


class TPI1005A:
    def __init__(self, port: str, timeout: float = 1.0):
        self._port = port
        self._timeout = timeout
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def open(self) -> "TPI1005A":
        self._ser = serial.Serial(
            port=self._port,
            baudrate=BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            rtscts=True,
            timeout=0.1,
        )
        time.sleep(0.05)
        self._ser.reset_input_buffer()
        self._enable_user_control()
        return self

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Packet layer
    # ------------------------------------------------------------------

    @staticmethod
    def _build(body: bytes | list) -> bytes:
        body = bytes(body)
        n = len(body)
        len_hi = (n >> 8) & 0xFF
        len_lo = n & 0xFF
        cs = (0xFF - (len_hi + len_lo + sum(body))) & 0xFF
        return bytes([0xAA, 0x55, len_hi, len_lo]) + body + bytes([cs])

    def _send_recv(self, body: bytes | list) -> bytes:
        pkt = self._build(body)
        with self._lock:
            self._ser.write(pkt)
            raw = self._read_packet()
        return raw

    def _read_packet(self) -> bytes:
        deadline = time.monotonic() + self._timeout
        buf = b""
        while time.monotonic() < deadline:
            chunk = self._ser.read(self._ser.in_waiting or 1)
            if chunk:
                buf += chunk
                if len(buf) >= 5:
                    if buf[0] == 0xAA and buf[1] == 0x55:
                        n = (buf[2] << 8) | buf[3]
                        if len(buf) >= 4 + n + 1:
                            body = buf[4:4 + n]
                            cs_exp = (0xFF - (buf[2] + buf[3] + sum(body))) & 0xFF
                            if buf[4 + n] != cs_exp:
                                raise TPIError(
                                    f"checksum error: got 0x{buf[4+n]:02x}, expected 0x{cs_exp:02x}"
                                )
                            return body
                    else:
                        idx = buf.find(QUALIFIER)
                        if idx > 0:
                            buf = buf[idx:]
                        elif idx == -1:
                            buf = b""
                deadline = time.monotonic() + 0.05
        return b""

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _enable_user_control(self):
        body = self._send_recv([0x08, 0x01])
        if not body or body[:2] != bytes([0x08, 0x01]):
            raise TPIError("failed to enable user control")

    # ------------------------------------------------------------------
    # Device info
    # ------------------------------------------------------------------

    def _read_ascii_field(self, code: int) -> str:
        body = self._send_recv([0x07, code])
        if not body or len(body) < 3:
            return ""
        return body[2:].decode("ascii", errors="replace").strip()

    def get_model(self) -> str:
        return self._read_ascii_field(0x02)

    def get_serial(self) -> str:
        return self._read_ascii_field(0x03)

    def get_firmware(self) -> str:
        return self._read_ascii_field(0x05)

    # ------------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------------

    def get_freq(self) -> float:
        """Return current frequency in MHz."""
        body = self._send_recv([0x07, 0x09])
        if not body or len(body) < 6:
            raise TPIError("no frequency response")
        freq_khz = struct.unpack_from("<I", body, 2)[0]
        return freq_khz / 1000.0

    def set_freq(self, freq_mhz: float):
        """Set frequency in MHz (35–4400)."""
        freq_khz = int(round(freq_mhz * 1000))
        if not (35000 <= freq_khz <= 4400000):
            raise ValueError(f"frequency {freq_mhz} MHz out of range (35–4400)")
        body_bytes = [0x08, 0x09] + list(struct.pack("<I", freq_khz))
        body = self._send_recv(body_bytes)
        if not body or body[:2] != bytes([0x08, 0x09]):
            raise TPIError("set_freq failed")

    # ------------------------------------------------------------------
    # Output level
    # ------------------------------------------------------------------

    def get_level(self) -> int:
        """Return current output level in dBm."""
        body = self._send_recv([0x07, 0x0A])
        if not body or len(body) < 3:
            raise TPIError("no level response")
        return struct.unpack_from("b", body, 2)[0]

    def set_level(self, dbm: int):
        """Set output level in dBm (−90 to +10 for TPI-1005-A)."""
        body = self._send_recv([0x08, 0x0A, dbm & 0xFF])
        if not body or body[:2] != bytes([0x08, 0x0A]):
            raise TPIError("set_level failed")

    # ------------------------------------------------------------------
    # RF on/off
    # ------------------------------------------------------------------

    def get_rf(self) -> bool:
        """Return True if RF output is on."""
        body = self._send_recv([0x07, 0x0B])
        if not body or len(body) < 3:
            raise TPIError("no RF state response")
        return bool(body[2])

    def set_rf(self, on: bool):
        """Turn RF output on (True) or off (False)."""
        body = self._send_recv([0x08, 0x0B, 0x01 if on else 0x00])
        if not body or body[:2] != bytes([0x08, 0x0B]):
            raise TPIError("set_rf failed")

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    def get_state(self) -> DeviceState:
        return DeviceState(
            model=self.get_model(),
            serial_number=self.get_serial(),
            firmware=self.get_firmware(),
            freq_mhz=self.get_freq(),
            level_dbm=self.get_level(),
            rf_on=self.get_rf(),
        )


def find_device() -> str | None:
    """Return the first FTDI FT230X serial port, or None."""
    import serial.tools.list_ports
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x0403 and p.pid == 0x6015:
            return p.device
    return None


def find_all_devices() -> list[str]:
    """Return all FTDI FT230X serial ports on this machine."""
    import serial.tools.list_ports
    return [p.device for p in serial.tools.list_ports.comports()
            if p.vid == 0x0403 and p.pid == 0x6015]
