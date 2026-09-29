"""Scan settings of a run loop, set from its card on the Sequences tab.

A loop whose experiment scans one xvar (kexp: the BEC TOF loop, ``t_tof``) can
offer its scan to be set from the GUI: a start value, an optional stop value,
the number of points between them, and the number of repeats.  Leaving the
stop value out repeats the start value alone.

The lab says what may be set (:class:`ScanSpec`: the xvar, its display unit,
bounds, defaults).  The monitor server keeps the loop's current settings (in
memory: a server restart goes back to the defaults) and hands them to every
run it launches in the environment variable :data:`ENV_VAR`; the experiment
reads them with :func:`scan_from_env` in ``prepare()`` and falls back to its
own values when it is run by hand.  The values a run used are its saved xvars
and ``N_repeats``, as for any run -- the file's source text shows only the
fallback.

Settings are a plain dict in SI units: ``{"start": float, "stop": float | None,
"n": int, "repeats": int}``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

#: The environment variable a loop's run finds its scan settings in.
ENV_VAR = "WAXX_LOOP_SCAN"

MAX_POINTS = 200
MAX_REPEATS = 100


@dataclass(frozen=True)
class ScanSpec:
    """What a loop's scan settings may be.  ``scale`` is SI per display unit
    (1e-3 for ms); ``minimum`` / ``maximum`` bound start and stop (SI); the
    defaults are the loop's settings until someone changes them."""

    xvar: str
    unit: str = ""
    scale: float = 1.0
    minimum: float | None = None
    maximum: float | None = None
    start: float = 0.0
    stop: float | None = None
    n: int = 1
    repeats: int = 1

    @classmethod
    def from_mapping(cls, m) -> "ScanSpec":
        return cls(**{k: m[k] for k in cls.__dataclass_fields__ if k in m})

    def defaults(self) -> dict:
        return normalize(self, {"start": self.start, "stop": self.stop, "n": self.n,
                                "repeats": self.repeats})

    def info(self) -> dict:
        """For the GUIs: what may be set (bounds in SI)."""
        return {"xvar": self.xvar, "unit": self.unit, "scale": self.scale,
                "minimum": self.minimum, "maximum": self.maximum,
                "max_points": MAX_POINTS, "max_repeats": MAX_REPEATS}


class ScanSettingsError(ValueError):
    pass


def normalize(spec: ScanSpec, settings) -> dict:
    """Checked settings (raises :class:`ScanSettingsError` saying why not).
    Without a stop value ``n`` is 1."""
    if not isinstance(settings, dict):
        raise ScanSettingsError("scan settings must be a dict")
    start = _number(settings.get("start"), "the start value")
    stop = settings.get("stop")
    stop = None if stop is None or stop == "" else _number(stop, "the stop value")
    for name, v in (("start", start), ("stop", stop)):
        if v is None:
            continue
        if spec.minimum is not None and v < spec.minimum:
            raise ScanSettingsError(f"the {name} value {_show(spec, v)} is below the minimum "
                                    f"{_show(spec, spec.minimum)}")
        if spec.maximum is not None and v > spec.maximum:
            raise ScanSettingsError(f"the {name} value {_show(spec, v)} is above the maximum "
                                    f"{_show(spec, spec.maximum)}")
    repeats = _count(settings.get("repeats"), "repeats", 1, MAX_REPEATS)
    if stop is None:
        n = 1
    else:
        n = _count(settings.get("n"), "the number of points", 2, MAX_POINTS)
    return {"start": start, "stop": stop, "n": n, "repeats": repeats}


def values(settings: dict) -> np.ndarray:
    """The xvar's values (SI), before repeats."""
    if settings.get("stop") is None:
        return np.array([float(settings["start"])])
    return np.linspace(float(settings["start"]), float(settings["stop"]), int(settings["n"]))


def describe(spec: ScanSpec, settings: dict) -> str:
    """One line: ``t_tof 1-4 ms, 9 points × 5 repeats (45 shots)``."""
    s = settings
    shots = int(s["n"]) * int(s["repeats"])
    if s.get("stop") is None:
        what = f"{spec.xvar} {_show(spec, s['start'])}"
    else:
        what = (f"{spec.xvar} {_num(spec, s['start'])}–{_show(spec, s['stop'])}, "
                f"{s['n']} points")
    return f"{what} × {s['repeats']} repeat{'s' if s['repeats'] != 1 else ''} ({shots} shots)"


def to_env(settings: dict) -> str:
    return json.dumps({k: settings[k] for k in ("start", "stop", "n", "repeats")})


def scan_from_env(env=None) -> dict | None:
    """In an experiment's ``prepare()``: the settings its loop launched it
    with, or None when it was not launched by a loop with scan settings.
    Raises when the variable is set but unreadable -- a run must not quietly
    fall back to other values than the ones asked for."""
    raw = (os.environ if env is None else env).get(ENV_VAR)
    if not raw:
        return None
    try:
        s = json.loads(raw)
        stop = s.get("stop")
        out = {"start": float(s["start"]), "stop": None if stop is None else float(stop),
               "n": int(s["n"]), "repeats": int(s["repeats"])}
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f"{ENV_VAR} is set but not readable ({exc!r}): {raw!r}") from exc
    if out["n"] < 1 or out["repeats"] < 1:
        raise RuntimeError(f"{ENV_VAR} asks for no shots: {raw!r}")
    return out


def _number(v, what: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ScanSettingsError(f"{what} is not a number: {v!r}") from None
    if not math.isfinite(x):
        raise ScanSettingsError(f"{what} is not finite: {v!r}")
    return x


def _count(v, what: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
        raise ScanSettingsError(f"{what} must be a whole number: {v!r}")
    n = int(v)
    if not lo <= n <= hi:
        raise ScanSettingsError(f"{what} must be {lo} to {hi}: {n}")
    return n


def _num(spec: ScanSpec, v: float) -> str:
    return f"{v / spec.scale:g}"


def _show(spec: ScanSpec, v: float) -> str:
    return f"{_num(spec, v)} {spec.unit}".rstrip()
