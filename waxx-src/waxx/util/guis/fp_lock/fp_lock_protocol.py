"""The lock firmware's serial protocol, and the warnings the server raises.

The firmware (k-exp/kexp/control/quarto/pid_frequency_stabilize_Raman_v2) is a
qCommand sketch:

* ``name`` reads a setting, ``name value`` sets it; both reply ``name is value``.
  An unknown name replies ``Unknown command: name``.
* With ``fmt 0`` it prints one CSV line every ~10 sweeps (``CSV_FIELDS``; the
  last two, ``integ`` and ``period``, only from firmware that has them).

Pure functions only: no I/O, no Qt.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "CSV_FIELDS", "CSV_MIN_FIELDS", "STATE_NAMES", "Setting", "SETTINGS", "SETTABLE",
    "parse_row", "parse_reply", "parse_unknown", "format_value", "EdgeConfig",
    "compute_warnings",
]

#: CSV columns, in the firmware's order (fmt 0).
CSV_FIELDS = (
    "sep", "err", "feedback", "state", "tfirst", "h1", "h2", "bad", "unlocks", "sats",
    "saturated", "bad_none", "bad_one", "bad_win", "integ", "period",
)
#: Firmware before integ/period printed 14 columns; older still, 11.
CSV_MIN_FIELDS = 11
_INT_FIELDS = {"state", "bad", "unlocks", "sats", "saturated", "bad_none", "bad_one", "bad_win"}

STATE_NAMES = {0: "OFF", 1: "HOLD", 2: "LOCKED"}


@dataclass(frozen=True)
class Setting:
    name: str
    kind: str          # "float" | "int" | "switch"
    unit: str
    help: str


#: Every setting the panel shows and may change, in display order.
SETTINGS: tuple[Setting, ...] = (
    Setting("lock", "switch", "", "1 = on; anything else = off (0 V, integral cleared)"),
    Setting("hold", "switch", "", "1 = skip bad sweeps and pause in HOLD; 0 = use every pair"),
    Setting("setp", "float", "ms", "peak-separation setpoint"),
    Setting("p1", "float", "V/ms", "proportional gain: feedback = g1*p1*err + integ"),
    Setting("i1", "float", "V/ms", "integral gain, each sweep: integ += g1*i1*err"),
    Setting("g1", "float", "", "scales p1 and i1"),
    Setting("win", "float", "ms", "|err| >= win: sweep skipped, output held (hold = 1)"),
    Setting("nbad", "int", "sweeps", "skipped in a row: LOCKED -> HOLD"),
    Setting("ngood", "int", "sweeps", "good in a row: HOLD -> LOCKED"),
    Setting("vth", "float", "V", "peak threshold"),
    Setting("hys", "float", "V", "a peak ends below vth - hys"),
    Setting("nmin", "int", "samples", "samples above vth for a peak to count"),
    Setting("tmin", "float", "ms", "peaks closer than this to the first are ignored"),
)
SETTABLE = {s.name: s for s in SETTINGS}

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf"
_ROW_RE = re.compile(rf"^\s*(?:{_NUM})(?:\s*,\s*(?:{_NUM}))*\s*$", re.IGNORECASE)
_REPLY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*) is (.*?)\s*$")
_UNKNOWN_RE = re.compile(r"^\s*Unknown command:\s*(\S+)")


def parse_row(line: str) -> Optional[dict]:
    """One fmt-0 CSV line as a dict (plus ``t2`` = tfirst + sep), or None if
    the line is not one."""
    if "," not in line or not _ROW_RE.match(line):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < CSV_MIN_FIELDS:
        return None
    row: dict = {}
    for name, text in zip(CSV_FIELDS, parts):
        try:
            value = float(text)
        except ValueError:
            return None
        row[name] = int(value) if name in _INT_FIELDS and math.isfinite(value) else value
    row["t2"] = row["tfirst"] + row["sep"]
    row["state_name"] = STATE_NAMES.get(row["state"], str(row["state"]))
    return row


def parse_reply(line: str) -> Optional[tuple[str, str]]:
    """``name is value`` -> (name, value text), else None."""
    m = _REPLY_RE.match(line)
    return (m.group(1), m.group(2)) if m else None


def parse_unknown(line: str) -> Optional[str]:
    """``Unknown command: name`` -> name, else None."""
    m = _UNKNOWN_RE.match(line)
    return m.group(1) if m else None


def format_value(setting: Setting, value) -> str:
    """The text sent to the firmware for ``setting = value`` (ValueError if bad)."""
    if setting.kind == "switch":
        v = int(round(float(value)))
        if v not in (0, 1):
            raise ValueError(f"{setting.name} must be 0 or 1")
        return str(v)
    if setting.kind == "int":
        f = float(value)
        if not math.isfinite(f) or f != int(f):
            raise ValueError(f"{setting.name} must be a whole number")
        return str(int(f))
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{setting.name} must be finite")
    return repr(f)


@dataclass(frozen=True)
class EdgeConfig:
    """Where the sweep starts and ends, for the peak-position warnings."""
    right_edge_ms: float = 120.0
    margin_ms: float = 10.0


def compute_warnings(row: Optional[dict], *, edge: EdgeConfig, com_status: str,
                     data_age_s: Optional[float], stale_after_s: float) -> list[dict]:
    """The warnings for the current state, most serious first.

    Each is ``{"level": "error"|"warn"|"info", "code": str, "text": str}``.
    The peak-position warnings need a measured pair (``sep`` > 0); ``t2`` is
    ``tfirst + sep`` of the latest line (the firmware keeps ``sep`` of the last
    sweep that found a pair).
    """
    out: list[dict] = []

    def add(level, code, text):
        out.append({"level": level, "code": code, "text": text})

    if com_status != "connected":
        add("error", "com", f"lock serial port {com_status}")
    elif data_age_s is None:
        add("error", "no_data", "connected, no data from the lock yet")
    elif data_age_s > stale_after_s:
        add("error", "stale", f"no data from the lock for {data_age_s:.0f} s")
    if row is None:
        return out

    if row.get("saturated"):
        add("error", "rail", f"output at the rail ({row['feedback']:+.3f} V)")
    state = row.get("state")
    if state == 1:
        add("warn", "hold", "HOLD: output frozen, waiting for good sweeps")
    elif state == 0:
        add("info", "off", "lock off (output 0 V)")

    if row.get("sep", 0) > 0 and row.get("h1", 0) > 0:
        m = edge.margin_ms
        tfirst, t2 = row["tfirst"], row["t2"]
        if tfirst < m:
            add("warn", "first_near_start",
                f"first peak at {tfirst:.2f} ms: within {m:g} ms of the sweep start")
        if t2 > edge.right_edge_ms - m:
            add("warn", "second_near_end",
                f"second peak at {t2:.2f} ms: within {m:g} ms of the sweep end "
                f"({edge.right_edge_ms:g} ms)")
    period = row.get("period")
    if isinstance(period, (int, float)) and period > 0 and edge.right_edge_ms > 0:
        if abs(period - edge.right_edge_ms) > 0.05 * edge.right_edge_ms:
            add("info", "period",
                f"measured sweep period {period:.1f} ms, the edge warning assumes "
                f"{edge.right_edge_ms:g} ms")
    order = {"error": 0, "warn": 1, "info": 2}
    out.sort(key=lambda w: order[w["level"]])
    return out
