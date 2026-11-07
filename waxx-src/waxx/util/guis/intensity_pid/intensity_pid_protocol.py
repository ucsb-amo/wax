"""The two-loop intensity PID firmware's serial protocol, the block means the
server keeps, and the warnings it raises.

The firmware is a qCommand sketch:

* ``name`` reads a setting, ``name value`` sets it; both reply ``name is value``
  (floats with 6 decimals: a gain below 5e-7 reads back as 0).  An unknown name
  replies ``Unknown command: name``.
* ``hold`` is a qCommand bool: ``hold`` reads it, ``hold 1`` / ``hold 0`` set
  it; reply ``hold is true|false``.  ``c`` zeroes both integrators (no reply).
* ``override1`` (bool) / ``ovolt1`` (V): while on, OUT1 = ovolt1 instead of the
  PID (same for loop 2); the firmware does not clamp ovolt.
* It prints ``meas1,set1,meas2,set2,hold,out1,out2`` every ~10 ms
  (``STREAM_FIELDS``).

Pure functions only: no I/O, no Qt.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional

__all__ = [
    "STREAM_FIELDS", "STREAM_MIN_FIELDS", "MEAN_FIELDS", "LOG_FIELDS", "Setting", "SETTINGS",
    "SETTABLE", "GAIN_NAMES",
    "OutputLimits", "parse_row", "parse_reply", "parse_unknown", "format_value",
    "parse_switch_reply", "block_mean", "Binner", "compute_warnings",
]

#: Status printout columns, in the firmware's order.
STREAM_FIELDS = ("meas1", "set1", "meas2", "set2", "hold", "out1", "out2")
#: A line without out1/out2 is still a row.
STREAM_MIN_FIELDS = 5

#: Columns of one block mean (the panel's history and the 1 s log), after the time.
MEAN_FIELDS = ("n", "meas1", "meas1_sd", "set1", "meas2", "meas2_sd", "set2",
               "out1", "out2", "hold_frac")
#: The 1 s log: the means plus the override switches when the block closed.
LOG_FIELDS = MEAN_FIELDS + ("override1", "override2")


@dataclass(frozen=True)
class Setting:
    name: str
    loop: int
    kind: str          # "gain" | "switch" | "volt"
    help: str


#: The settings the server reads and may change, in display order.
SETTINGS: tuple[Setting, ...] = (
    Setting("p1", 1, "gain", "loop 1 proportional gain: out = p*(meas - set) + integ"),
    Setting("i1", 1, "gain", "loop 1 integral gain, per ADC sample: integ += i*(meas - set)"),
    Setting("p2", 2, "gain", "loop 2 proportional gain"),
    Setting("i2", 2, "gain", "loop 2 integral gain, per ADC sample"),
    Setting("override1", 1, "switch", "1 = OUT1 is ovolt1, not the PID"),
    Setting("ovolt1", 1, "volt", "OUT1 while override1 = 1 (V)"),
    Setting("override2", 2, "switch", "1 = OUT2 is ovolt2, not the PID"),
    Setting("ovolt2", 2, "volt", "OUT2 while override2 = 1 (V)"),
)
SETTABLE = {s.name: s for s in SETTINGS}
#: The settings the server remembers and re-applies after a board reset.
GAIN_NAMES = tuple(s.name for s in SETTINGS if s.kind == "gain")

#: The firmware replies floats with %f: 6 decimals.
READBACK_RESOLUTION = 1e-6


@dataclass(frozen=True)
class OutputLimits:
    """The firmware clamps each PID output to [low, high] (V); the server keeps
    an override voltage in the same range (the firmware does not clamp it)."""
    low: float = 0.0
    high: float = 9.9


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf"
_ROW_RE = re.compile(rf"^\s*(?:{_NUM})(?:\s*,\s*(?:{_NUM}))*\s*$", re.IGNORECASE)
_REPLY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*) is (.*?)\s*$")
_UNKNOWN_RE = re.compile(r"^\s*Unknown command:\s*(\S+)")


def parse_row(line: str) -> Optional[dict]:
    """One status line as a dict (``hold`` an int), or None if it is not one."""
    if "," not in line or not _ROW_RE.match(line):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < STREAM_MIN_FIELDS or len(parts) > len(STREAM_FIELDS):
        return None
    row: dict = {}
    for name, text in zip(STREAM_FIELDS, parts):
        try:
            value = float(text)
        except ValueError:
            return None
        if name == "hold":
            if value not in (0.0, 1.0):
                return None
            value = int(value)
        row[name] = value
    return row


def parse_reply(line: str) -> Optional[tuple[str, str]]:
    """``name is value`` -> (name, value text), else None."""
    m = _REPLY_RE.match(line)
    return (m.group(1), m.group(2)) if m else None


def parse_unknown(line: str) -> Optional[str]:
    """``Unknown command: name`` -> name, else None."""
    m = _UNKNOWN_RE.match(line)
    return m.group(1) if m else None


def parse_switch_reply(text: Optional[str]) -> Optional[bool]:
    """The value of ``hold is true`` (or ``1``) as a bool (None if neither)."""
    t = (text or "").strip().split()[:1]
    if t in (["1"], ["true"]):
        return True
    if t in (["0"], ["false"]):
        return False
    return None


def format_value(setting: Setting, value, limits: OutputLimits = OutputLimits()) -> str:
    """The text sent to the firmware for ``setting = value`` (ValueError if bad)."""
    if setting.kind == "switch":
        on = parse_switch_reply(str(value).lower()) if not isinstance(value, bool) else value
        if on is None:
            raise ValueError(f"{setting.name} must be 1 or 0")
        return "1" if on else "0"
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{setting.name} must be finite")
    if setting.kind == "volt" and not limits.low <= f <= limits.high:
        raise ValueError(f"{setting.name} must be within {limits.low:g} to {limits.high:g} V")
    return repr(f)


def parse_setting(setting: Setting, text: Optional[str]) -> Optional[float]:
    """A ``name is ...`` value as a number (a switch as 1 / 0), or None."""
    if setting.kind == "switch":
        on = parse_switch_reply(text)
        return None if on is None else int(on)
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return None


def block_mean(rows: Iterable[dict]) -> Optional[dict]:
    """Means of a block of status rows (``MEAN_FIELDS``; ``*_sd`` the sample
    standard deviation, NaN for one row; ``hold_frac`` the fraction of rows
    held).  ``t`` is the time of the last row.  None for no rows."""
    rows = list(rows)
    if not rows:
        return None
    out: dict = {"t": rows[-1].get("t"), "n": len(rows)}

    def col(key):
        return [float(r[key]) for r in rows
                if isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))]

    for key in ("meas1", "set1", "meas2", "set2", "out1", "out2"):
        vals = col(key)
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    for key in ("meas1", "meas2"):
        vals = col(key)
        if len(vals) > 1:
            m = sum(vals) / len(vals)
            out[f"{key}_sd"] = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        else:
            out[f"{key}_sd"] = float("nan")
    holds = col("hold")
    out["hold_frac"] = sum(holds) / len(holds) if holds else float("nan")
    return out


class Binner:
    """Groups rows (with ``t``, s) into blocks ``width_s`` long; ``add``
    returns the mean of a block when a row past its end arrives."""

    def __init__(self, width_s: float):
        self.width_s = float(width_s)
        self._rows: list[dict] = []
        self._start: Optional[float] = None

    def add(self, row: dict) -> Optional[dict]:
        t = row["t"]
        closed = None
        if self._start is not None and t - self._start >= self.width_s:
            closed = block_mean(self._rows)
            self._rows = []
            self._start = None
        if self._start is None:
            self._start = t
        self._rows.append(row)
        return closed

    def reset(self) -> None:
        self._rows = []
        self._start = None


def compute_warnings(row: Optional[dict], *, limits: OutputLimits, com_status: str,
                     data_age_s: Optional[float], stale_after_s: float,
                     loop_names: tuple[str, str] = ("loop 1", "loop 2"),
                     settings: Optional[dict] = None,
                     extra: Iterable[dict] = ()) -> list[dict]:
    """The warnings for the current state, most serious first.

    Each is ``{"level": "error"|"warn"|"info", "code": str, "text": str}``.
    ``row`` is the latest status row; ``settings`` the server's cache (the
    override switches); ``extra`` are the server's own warnings (a gain
    re-apply).
    """
    settings = settings or {}
    overridden = {i for i in (1, 2) if settings.get(f"override{i}") == 1}
    out: list[dict] = []

    def add(level, code, text):
        out.append({"level": level, "code": code, "text": text})

    if com_status != "connected":
        add("error", "com", f"serial port {com_status}")
    elif data_age_s is None:
        add("error", "no_data", "connected, no status lines from the board yet")
    elif data_age_s > stale_after_s:
        add("error", "stale", f"no status lines from the board for {data_age_s:.0f} s")
    out.extend(dict(w) for w in extra)
    for i in sorted(overridden):
        v = settings.get(f"ovolt{i}")
        add("warn", f"override{i}", f"{loop_names[i - 1]}: output overridden to "
            f"{'?' if not isinstance(v, (int, float)) else f'{v:.3f}'} V, the PID is not driving it")
    if row is None:
        return _sorted(out)

    if row.get("hold") == 1:
        add("warn", "hold", "HELD: both outputs fixed, the loops are not correcting")
    else:
        for i, name in ((1, loop_names[0]), (2, loop_names[1])):
            if i in overridden:
                continue
            v = row.get(f"out{i}")
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            if v >= limits.high:
                add("warn", f"rail{i}", f"{name}: output at its upper limit ({v:.2f} V)")
            elif v <= limits.low:
                add("warn", f"rail{i}", f"{name}: output at its lower limit ({v:.2f} V)")
    return _sorted(out)


def _sorted(out: list[dict]) -> list[dict]:
    order = {"error": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda w: order.get(w["level"], 3))
