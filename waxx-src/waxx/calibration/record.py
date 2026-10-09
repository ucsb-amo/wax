"""CalResult: one calibration number as an analysis produced it, and the checks on it.

A ``CalResult`` is what an analysis returns (``calibrate(ad, key, **opts)``) and
what the ledger stores. ``evaluate(result, policy_entry)`` adds the flags; a
flagged result is never written back.

The flag rules are fixed and listed in ``evaluate``. A key with no policy entry
gets only the rules that need no policy: the analysis failed, the value or its
uncertainty is not finite, the uncertainty is missing.
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


@dataclass
class CalResult:
    key: str
    value: float
    unc: Optional[float]
    unit: str = ""
    run_id: int = 0
    expt_file: str = ""
    n_used: int = 0
    excluded: dict = field(default_factory=lambda: {"count": 0, "reason": ""})
    method: str = ""
    # params, their uncertainties and a goodness summary; "ok" False = the
    # analysis did not produce a number (with "reason")
    fit: dict = field(default_factory=dict)
    figure_path: Optional[str] = None
    old_value: Optional[float] = None
    rel_change: Optional[float] = None
    # [{"code": short code, "text": what was wrong}]
    flags: list = field(default_factory=list)
    applied: bool = False
    applied_file: Optional[str] = None
    applied_line: Optional[int] = None
    deferred: bool = False
    timestamp: str = ""
    analysis: str = ""
    # the params class of the run ("module:qualname"), where a write-back goes
    params_class: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    # ---- derived ------------------------------------------------------------
    @property
    def fit_ok(self) -> bool:
        """False only when the analysis says so (``fit["ok"] is False``)."""
        return self.fit.get("ok", True) is not False

    @property
    def flagged(self) -> bool:
        return bool(self.flags)

    def set_old_value(self, old_value):
        """Record the value in use and the fractional change from it."""
        self.old_value = None if old_value is None else float(old_value)
        if (self.old_value is not None and math.isfinite(self.old_value) and self.old_value != 0
                and _finite(self.value)):
            self.rel_change = (float(self.value) - self.old_value) / abs(self.old_value)
        else:
            self.rel_change = None

    # ---- JSON ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return _jsonable(asdict(self))

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "CalResult":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        for k in ("value", "unc", "old_value", "rel_change"):
            if k in kw and kw[k] is not None:
                kw[k] = float(kw[k])
        return cls(**kw)

    @classmethod
    def from_json(cls, text: str) -> "CalResult":
        return cls.from_dict(json.loads(text))


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _jsonable(obj: Any):
    """Plain JSON types. NaN / inf become the strings 'nan' / 'inf' / '-inf'
    (JSON has no such numbers); numpy scalars and arrays become Python ones."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return _jsonable(obj.tolist())
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    try:
        return _jsonable(float(obj))
    except (TypeError, ValueError):
        return repr(obj)


# ---- the checks -----------------------------------------------------------------

def _flag(code, text):
    return {"code": code, "text": text}


def evaluate(result: CalResult, policy_entry: Optional[dict] = None, *,
             allow_no_unc: bool = False) -> list:
    """The flags on ``result`` (also stored in ``result.flags``). Rules, and only these:

    - ``analysis_failed``: the analysis says its fit failed (``fit["ok"] is False``)
    - ``nonfinite``: the value, or a given uncertainty, is NaN or infinite
    - ``no_unc``: no uncertainty (unless ``allow_no_unc``)
    - ``rel_unc``: unc/|value| above the policy's ``max_rel_unc``
    - ``change``: |rel_change| above the policy's ``max_frac_change``
    - ``below_min`` / ``above_max``: the value outside the policy's ``min`` / ``max``
    - ``few_shots``: ``n_used`` below the policy's ``min_shots``

    No policy entry (None or {}) -> only the first three. A deferred result
    (the analysis did not finish in its budget) has no value, so it is flagged
    ``analysis_failed`` too."""
    pol = dict(policy_entry or {})
    flags = []
    if result.deferred:
        flags.append(_flag("analysis_failed", "the analysis did not finish within its time "
                                               "budget (deferred); there is no value"))
    elif not result.fit_ok:
        why = result.fit.get("reason") or "no reason given"
        flags.append(_flag("analysis_failed", f"the analysis failed: {why}"))
    value_ok = _finite(result.value)
    if not value_ok:
        flags.append(_flag("nonfinite", f"value is not finite ({result.value!r})"))
    if result.unc is None:
        if not allow_no_unc:
            flags.append(_flag("no_unc", "no uncertainty was given (declare "
                                         "allow_no_unc=True to accept that)"))
    elif not _finite(result.unc):
        flags.append(_flag("nonfinite", f"uncertainty is not finite ({result.unc!r})"))
    elif float(result.unc) < 0:
        flags.append(_flag("nonfinite", f"uncertainty is negative ({result.unc!r})"))

    if value_ok:
        v = float(result.value)
        lim = pol.get("max_rel_unc")
        if lim is not None and _finite(result.unc) and v != 0:
            r = abs(float(result.unc) / v)
            if r > lim:
                flags.append(_flag("rel_unc", f"relative uncertainty {r:.3g} is above the "
                                              f"policy limit {lim:g}"))
        lim = pol.get("max_frac_change")
        if lim is not None and result.rel_change is not None and abs(result.rel_change) > lim:
            flags.append(_flag("change", f"change from the value in use {result.rel_change:+.3%} "
                                         f"exceeds the policy limit +/-{lim:.3%}"))
        lo, hi = pol.get("min"), pol.get("max")
        if lo is not None and v < lo:
            flags.append(_flag("below_min", f"value {v:.6g} is below the policy minimum {lo:g}"))
        if hi is not None and v > hi:
            flags.append(_flag("above_max", f"value {v:.6g} is above the policy maximum {hi:g}"))
    lim = pol.get("min_shots")
    if lim is not None and int(result.n_used) < lim:
        flags.append(_flag("few_shots", f"{result.n_used} shots used, below the policy "
                                        f"minimum {lim}"))
    result.flags = flags
    return flags
