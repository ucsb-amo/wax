"""A stand-in for the Quarto running the two-loop intensity PID firmware
(k-exp/kexp/control/quarto/pid_intensity_stabilize_raman), at the serial-port
level, speaking its qCommand protocol: gains ``p1 i1 p2 i2`` (``name is
%f``), ``hold`` (a qCommand bool: ``hold is true|false``), ``c`` (no reply),
``ping``, ``Unknown command: name``.  Status lines are pushed by the test with
:meth:`emit_row`, in the firmware's format.  ``hold`` is modelled: release
leaves the integrators at the held outputs.
"""

from __future__ import annotations

import threading

import serial

DEFAULT_GAINS = {"p1": -0.055, "i1": -0.006, "p2": -0.055, "i2": -0.006}


class FakeIntensityQuarto:
    def __init__(self, label: str = "raman intensity", gains: dict | None = None,
                 without: tuple = ()):
        self.label = label
        self.gains = dict(DEFAULT_GAINS if gains is None else gains)
        self.without = set(without)
        self.hold = 0
        self.override = {"override1": 0, "override2": 0}
        self.ovolt = {"ovolt1": 0.0, "ovolt2": 0.0}
        self.integ = [0.3, 0.4]
        self.out = [1.25, 2.5]
        self.received: list[str] = []
        self.opened = 0
        self.port = None
        self.fail_open = False

    def open(self, port, baud):
        if self.fail_open:
            raise serial.SerialException(f"could not open port {port!r} (fake)")
        self.opened += 1
        self.port = FakePort(self)
        return self.port

    def power_cycle(self):
        """The board resets: gains back to the sketch's defaults (the port stays)."""
        self.gains = dict(DEFAULT_GAINS)
        self.hold = 0
        self.override = {"override1": 0, "override2": 0}
        self.ovolt = {"ovolt1": 0.0, "ovolt2": 0.0}

    def emit_row(self, meas1=0.5, set1=0.5, meas2=1.0, set2=1.0, out1=None, out2=None):
        if self.port is None:
            return
        o1 = self.out[0] if out1 is None else out1
        o2 = self.out[1] if out2 is None else out2
        if not self.hold:
            self.out = [o1, o2]
        else:
            o1, o2 = self.out
        if self.override["override1"]:
            o1 = self.out[0] = self.ovolt["ovolt1"]
        if self.override["override2"]:
            o2 = self.out[1] = self.ovolt["ovolt2"]
        self.port.feed(f"{meas1:.4f},{set1:.4f},{meas2:.4f},{set2:.4f},{self.hold},{o1:.4f},{o2:.4f}\n")

    def _answer(self, line: str) -> str:
        self.received.append(line)
        parts = line.split()
        if not parts:
            return ""
        name, arg = parts[0], (parts[1] if len(parts) > 1 else None)
        if name in self.without:
            return f"Unknown command: {name}\n"
        if name == "ping":
            return self.label + "\n"
        if name == "c":
            self.integ = [0.0, 0.0]
            return ""
        if name == "hold":
            if arg in ("1", "on", "true"):
                self.hold = 1
            elif arg in ("0", "off", "false"):
                if self.hold:
                    self.integ = list(self.out)
                self.hold = 0
            return f"hold is {'true' if self.hold else 'false'}\n"
        if name in self.override:
            if arg in ("1", "on", "true"):
                self.override[name] = 1
            elif arg in ("0", "off", "false"):
                self.override[name] = 0
            return f"{name} is {'true' if self.override[name] else 'false'}\n"
        if name in self.ovolt:
            if arg is not None:
                self.ovolt[name] = float(arg)
            return f"{name} is {self.ovolt[name]:f}\n"
        if name in self.gains:
            if arg is not None:
                self.gains[name] = float(arg)
            return f"{name} is {self.gains[name]:f}\n"
        return f"Unknown command: {name}\n"


class FakePort:
    def __init__(self, dev: FakeIntensityQuarto):
        self.dev = dev
        self.is_open = True
        self._buf = b""
        self._lock = threading.Lock()
        self._data = threading.Event()
        self._partial = ""
        self.fail_next_read = False

    def feed(self, text: str) -> None:
        with self._lock:
            self._buf += text.encode()
        self._data.set()

    @property
    def in_waiting(self) -> int:
        if not self.is_open:
            raise serial.SerialException("port closed (fake)")
        with self._lock:
            return len(self._buf)

    def read(self, n: int = 1) -> bytes:
        if self.fail_next_read:
            self.fail_next_read = False
            raise serial.SerialException("ClearCommError failed (fake unplug)")
        if not self.is_open:
            raise serial.SerialException("port closed (fake)")
        self._data.wait(0.05)
        with self._lock:
            out, self._buf = self._buf[:n], self._buf[n:]
            if not self._buf:
                self._data.clear()
        return out

    def write(self, data: bytes) -> int:
        if not self.is_open:
            raise serial.SerialException("port closed (fake)")
        self._partial += data.decode()
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            reply = self.dev._answer(line.strip())
            if reply:
                self.feed(reply)
        return len(data)

    def close(self) -> None:
        self.is_open = False
        self._data.set()
