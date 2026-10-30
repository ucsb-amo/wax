"""A stand-in for the Quarto running the FP lock firmware, at the serial-port
level (``in_waiting`` / ``read`` / ``write`` / ``close``), speaking its
qCommand protocol: ``name`` -> ``name is value``, ``name value`` sets and
replies the same way, unknown names -> ``Unknown command: name``, ``ping``,
``clear``.  Rows are pushed by the test with :meth:`FakeQuarto.emit_row`;
they come out as CSV (fmt 0) or as a table line (fmt 1), like the firmware.
"""

from __future__ import annotations

import threading

import serial

FLOAT_SETTINGS = {"setp": 10.0, "p1": 0.0, "i1": -0.01, "g1": 1.0, "win": 1.0, "vth": 4.0,
                  "hys": 0.2, "tmin": 1.5}
INT_SETTINGS = {"lock": 1, "hold": 1, "nbad": 5, "ngood": 10, "nmin": 3, "fmt": 1}
READ_ONLY = {"sep": 0.0, "integ": 0.0, "state": 1, "tfirst": 0.0, "h1": 0.0, "h2": 0.0,
             "bad": 0, "unlocks": 0, "sats": 0, "period": 120.0}


class FakeQuarto:
    def __init__(self, label: str = "raman fp lock", without: tuple = ()):
        self.label = label
        self.values = {**FLOAT_SETTINGS, **INT_SETTINGS, **READ_ONLY}
        for name in without:
            self.values.pop(name, None)
        self.ints = set(INT_SETTINGS) | {"state", "bad", "unlocks", "sats"}
        self.received: list[str] = []
        self.opened = 0
        self.port = None
        self.fail_open = False

    # the serial_factory the server takes
    def open(self, port, baud):
        if self.fail_open:
            raise serial.SerialException(f"could not open port {port!r} (fake)")
        self.opened += 1
        self.port = FakePort(self)
        return self.port

    def emit_row(self, **kw):
        """Print one status line as the firmware does every ~10 sweeps."""
        v = self.values
        for k, val in kw.items():
            v[k] = val
        row = [v["sep"], v["sep"] - v["setp"], kw.get("feedback", 0.0), v["state"], v["tfirst"],
               v["h1"], v["h2"], v["bad"], v["unlocks"], v["sats"], kw.get("saturated", 0),
               kw.get("bad_none", 0), kw.get("bad_one", 0), kw.get("bad_win", 0),
               v.get("integ", 0.0), v.get("period", 120.0)]
        if v.get("fmt", 1) == 0:
            line = ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in row)
        elif v.get("fmt", 1) == 2:
            return
        else:
            line = "   ".join(str(x) for x in row) + "  LOCKED"
        self.port.feed(line + "\n")

    def _answer(self, line: str) -> str:
        self.received.append(line)
        parts = line.split()
        if not parts:
            return ""
        name = parts[0]
        if name == "ping":
            return self.label + "\n"
        if name == "clear":
            self.values["integ"] = 0.0
            return "integral and output cleared (0 V); state unchanged\n"
        if name == "help":
            return "FP peak-separation lock. Send a name to read it.\nsettings (current value):\n"
        if name not in self.values:
            return f"Unknown command: {name}\n"
        if len(parts) > 1 and name not in READ_ONLY:
            val = float(parts[1])
            self.values[name] = int(val) if name in self.ints else val
        val = self.values[name]
        return f"{name} is {val:d}\n" if name in self.ints else f"{name} is {val:f}\n"


class FakePort:
    def __init__(self, dev: FakeQuarto):
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
