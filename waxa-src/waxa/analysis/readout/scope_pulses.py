"""Per-pulse integrals of a scope trace of a pulse train (the absolute photon reference).

Ported from ``k-jam/analysis/artisinal/apd_pulse_analysis_interpolate.ipynb`` (cell 6
``find_pulse_edges`` and the window loop of cell 12, commit e22b200) with the notebook
globals turned into arguments.  Two windows per pulse, with different jobs:

* **integration window** -- the WHOLE optical pulse, baseline-subtracted and integrated
  (trapezoid).  This is the photon-number source.  No pulse-shape assumption.
* **level window** -- the flat top only (diagnostics, effective pulse duration).

Pulse windows are measured, not assumed: the scope trigger fires on the commanded edge
of pulse 0, so the optical edge is found on the shot-averaged trace (first 50 %
crossing, shifted back by half the 10-90 rise time) unless ``pulse0_start`` pins it.

One change from the notebook: every pulse index is returned (NaN where a window falls
outside the trace), instead of averaging pulses per shot here; which pulses to average
is the caller's choice (the notebook's ``SKIP_FIRST_PULSE``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

DEFAULT_WINDOWS = dict(
    edge_search_start=-2.0e-6,   # start hunting for pulse 0's rising edge here (s)
    edge_smooth_time=50.0e-9,    # boxcar smoothing before edge finding (s)
    t_start_shift=2.0e-6,        # level window: start trim inside each pulse (s)
    t_end_shift=0.6e-6,          # level window: end trim inside each pulse (s)
    t_integrate_pre=1.0e-6,      # integral begins this far before the pulse start (s)
    t_integrate_post=4.0e-6,     # ... and ends this far after pulse start + duration (s)
    t_baseline_pre=5.0e-6,       # dark baseline over [start - this, start - t_integrate_pre]
)


def find_pulse_edges(t_axis, v_axis, t_search_start, t_search_stop, smooth_time=0.0):
    """Locate one pulse in the search range: t10/t50/t90 on the rise, f90/f50/f10 on the
    fall (sub-sample interpolated), rise/fall times, and the dark / flat-top levels.

    Levels come from a robust two-pass estimate (dark = median of the lower half of the
    samples; flat top = median of the samples in the top quarter of a first pass)."""
    t_axis = np.asarray(t_axis, dtype=float).ravel()
    v_axis = np.asarray(v_axis, dtype=float).ravel()
    m = (t_axis >= t_search_start) & (t_axis <= t_search_stop)
    if m.sum() < 32:
        raise ValueError(f"Only {int(m.sum())} samples in the edge-search range "
                         f"[{t_search_start*1e6:.3f}, {t_search_stop*1e6:.3f}] us.")
    ts, vs = t_axis[m], v_axis[m]
    dt = float(np.median(np.diff(ts)))
    n_sm = max(1, int(round(float(smooth_time) / dt))) if smooth_time else 1
    if n_sm > 1:
        vs = np.convolve(vs, np.ones(n_sm) / n_sm, mode="same")
        ts, vs = ts[n_sm:-n_sm], vs[n_sm:-n_sm]

    v_lo = float(np.median(vs[vs <= np.median(vs)]))
    on_rough = vs > v_lo + 0.5 * (vs.max() - v_lo)
    if not np.any(on_rough):
        raise ValueError("No pulse found in the edge-search range.")
    v_hi_rough = float(np.median(vs[on_rough]))
    on = vs > v_lo + 0.75 * (v_hi_rough - v_lo)
    v_hi = float(np.median(vs[on])) if np.any(on) else v_hi_rough
    span = v_hi - v_lo
    if span <= 0:
        raise ValueError("Flat-top level is not above the dark level.")

    def _interp(i, level):
        y = v_lo + level * span
        y0, y1 = vs[i - 1], vs[i]
        if y1 == y0:
            return float(ts[i])
        return float(ts[i - 1] + (y - y0) * (ts[i] - ts[i - 1]) / (y1 - y0))

    y50 = v_lo + 0.5 * span
    i_rise = int(np.argmax(vs > y50))
    if i_rise == 0:
        raise ValueError("The trace is already above the 50% level at the start of the edge-search "
                         "range -- move edge_search_start earlier.")
    i_fall = i_rise
    while i_fall < vs.size - 1 and vs[i_fall] > y50:
        i_fall += 1
    if vs[i_fall] > y50:
        raise ValueError("The pulse does not fall back inside the edge-search range.")

    def _back_to(i_from, level):
        y = v_lo + level * span
        i = int(i_from)
        while i > 0 and vs[i - 1] > y:
            i -= 1
        return float(ts[0]) if i == 0 else _interp(i, level)

    def _fwd_to(i_from, level, rising):
        y = v_lo + level * span
        i = int(i_from)
        while i < vs.size - 1 and ((vs[i] < y) if rising else (vs[i] > y)):
            i += 1
        return float("nan") if i == 0 else _interp(i, level)

    t50 = _back_to(i_rise, 0.5)
    t10 = _back_to(i_rise, 0.1)
    t90 = _fwd_to(i_rise, 0.9, rising=True)
    f50 = _interp(i_fall, 0.5)
    f10 = _fwd_to(i_fall, 0.1, rising=False)
    i = int(i_fall)
    while i > 0 and vs[i - 1] < v_lo + 0.9 * span:
        i -= 1
    f90 = float(ts[0]) if i == 0 else _interp(i, 0.9)
    return {"t10": t10, "t50": t50, "t90": t90, "t_rise": t90 - t10,
            "f90": f90, "f50": f50, "f10": f10, "t_fall": f10 - f90,
            "v_lo": v_lo, "v_hi": v_hi}


@dataclass(frozen=True, eq=False)
class ScopePulses:
    """Per-shot, per-pulse scope quantities.  Arrays are (n_shots, n_pulses), NaN where
    a pulse's baseline + integration span is not fully inside that shot's trace."""
    integral: np.ndarray          # baseline-subtracted integral of the whole pulse (V s)
    level: np.ndarray             # baseline-subtracted flat-top mean (V)
    baseline: np.ndarray          # dark level subtracted (V)
    valid: np.ndarray             # bool
    pulse0_edge: dict
    pulse0_start: float           # start of pulse 0 used for the windows (s, trace time)
    pulse0_start_auto: float
    t_duration: float
    t_between_pulses: float
    windows: dict
    warnings: tuple = ()
    problems: tuple = ()          # timing-check failures: the windows do not sit on the pulses

    @property
    def n_pulses(self):
        return self.integral.shape[1]

    @property
    def timing_ok(self):
        """False when the trace does not show the commanded pulse schedule (see
        :func:`check_timing`): nothing integrated from it should be trusted."""
        return not self.problems

    @property
    def t_effective(self):
        """integral / flat-top level per shot and pulse (s): the effective pulse length."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(self.level) > 0, self.integral / self.level, np.nan)

    def summary(self):
        e = self.pulse0_edge
        u = 1e6
        lines = [
            f"scope: pulse 0 rise t10/t50/t90 = {e['t10']*u:.3f}/{e['t50']*u:.3f}/{e['t90']*u:.3f} us "
            f"(turn-on {e['t_rise']*u:.3f} us), fall f90/f50/f10 = {e['f90']*u:.3f}/{e['f50']*u:.3f}/"
            f"{e['f10']*u:.3f} us (turn-off {e['t_fall']*u:.3f} us)",
            f"  dark {e['v_lo']:.4f} V, flat top {e['v_hi']:.4f} V, FWHM {(e['f50'] - e['t50'])*u:.3f} us "
            f"(commanded {self.t_duration*u:.3f} us); pulse 0 start {self.pulse0_start*u:+.3f} us "
            f"(auto {self.pulse0_start_auto*u:+.3f} us)",
            f"  {int(self.valid.sum())} of {self.valid.size} shot-pulse windows inside the traces",
        ]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        lines += [f"  TIMING CHECK FAILED: {p}" for p in self.problems]
        return "\n".join(lines)


FWHM_TOLERANCE = 0.3          # |FWHM / commanded - 1| above this fails the timing check
DARK_WINDOW_FRACTION = 0.3    # a window whose mean flat-top level is below this x pulse 0's fails


def check_timing(edge, level, t_duration, fwhm_tolerance=FWHM_TOLERANCE, dark_fraction=DARK_WINDOW_FRACTION):
    """Does the trace show the commanded schedule?  Two checks: pulse 0's FWHM against the
    commanded duration, and every pulse window lit (shot-mean baseline-subtracted flat-top
    level at least ``dark_fraction`` of pulse 0's height).  Returns a tuple of problems."""
    probs = []
    fwhm = edge["f50"] - edge["t50"]
    if not abs(fwhm / t_duration - 1.0) <= fwhm_tolerance:
        probs.append(f"pulse 0 FWHM {fwhm*1e6:.3f} us vs commanded {t_duration*1e6:.3f} us (ratio "
                     f"{fwhm / t_duration:.2f}): the trace time axis or the pulse is not what the schedule says")
    height = edge["v_hi"] - edge["v_lo"]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN column -> NaN -> "dark"
        lvl = np.nanmean(np.asarray(level, float), axis=0)
    dark = [p for p, v in enumerate(lvl) if not (v >= dark_fraction * height)]
    if dark:
        probs.append("pulse window(s) " + ", ".join(str(p + 1) for p in dark) + " hold no pulse (mean level "
                     + ", ".join(f"{lvl[p]:.3f}" for p in dark) + f" V vs pulse-0 height {height:.3f} V): the "
                     "windows placed from the commanded spacing do not sit on the pulses")
    return tuple(probs)


def _as_2d_list(a):
    if isinstance(a, np.ndarray) and a.ndim == 2:
        return [a[i] for i in range(a.shape[0])]
    return [np.asarray(x) for x in a]


def integrate_scope_pulses(t, v, *, t_duration, n_pulses, t_between_pulses, pulse0_start=None,
                           **windows) -> ScopePulses:
    """Integrate every pulse of every shot's trace (notebook cell 12).

    ``t`` and ``v`` are (n_shots, n_samples) arrays or lists of 1-D arrays (trace time 0
    = commanded edge of pulse 0).  ``t_duration`` is the commanded pulse length,
    ``t_between_pulses`` the start-to-start spacing.  Keyword overrides for the windows
    are in ``DEFAULT_WINDOWS``.
    """
    w = dict(DEFAULT_WINDOWS)
    unknown = set(windows) - set(w)
    if unknown:
        raise TypeError(f"unknown window options {sorted(unknown)}")
    w.update(windows)
    t_list, v_list = _as_2d_list(t), _as_2d_list(v)
    n_shots = len(t_list)
    if n_shots == 0 or len(v_list) != n_shots:
        raise ValueError("t and v must hold the same (nonzero) number of shots")
    n_pulses = int(n_pulses)
    t_between = float(t_between_pulses)
    t_duration = float(t_duration)
    if w["t_start_shift"] < 0 or w["t_end_shift"] < 0:
        raise ValueError("t_start_shift and t_end_shift must be non-negative.")
    if w["t_integrate_pre"] < 0 or w["t_integrate_post"] < 0:
        raise ValueError("t_integrate_pre and t_integrate_post must be non-negative.")
    if w["t_baseline_pre"] <= w["t_integrate_pre"]:
        raise ValueError("t_baseline_pre must exceed t_integrate_pre so the baseline sits outside the integral.")
    if t_duration - w["t_start_shift"] - w["t_end_shift"] <= 0:
        raise ValueError("the level window is empty: t_duration - t_start_shift - t_end_shift <= 0")
    span_per_pulse = w["t_baseline_pre"] + t_duration + w["t_integrate_post"]
    if n_pulses > 1 and span_per_pulse > t_between:
        raise ValueError(f"each pulse claims {span_per_pulse*1e6:.3f} us (baseline + integral) but the period "
                         f"is only {t_between*1e6:.3f} us -- consecutive pulses overlap.")

    # pulse 0's optical edge on the shot-averaged trace
    t_ref = np.asarray(t_list[0], dtype=float).ravel()
    v_avg = np.zeros_like(t_ref)
    for ts, vs in zip(t_list, v_list):
        ts = np.asarray(ts, dtype=float).ravel()
        vs = np.asarray(vs, dtype=float).ravel()
        same = ts.shape == t_ref.shape and np.array_equal(ts, t_ref)
        v_avg += vs if same else np.interp(t_ref, ts, vs)
    v_avg /= n_shots
    search_span = t_between if t_between > 0 else 4.0 * t_duration
    lo = max(float(w["edge_search_start"]), float(t_ref.min()))
    hi = min(lo + search_span, float(t_ref.max()))
    edge = find_pulse_edges(t_ref, v_avg, lo, hi, smooth_time=w["edge_smooth_time"])
    start_auto = edge["t50"] - 0.5 * edge["t_rise"]
    start = float(start_auto if pulse0_start is None else pulse0_start)

    warns = []
    level_lo = start + w["t_start_shift"]
    level_hi = start + t_duration - w["t_end_shift"]
    if level_lo < edge["t90"]:
        warns.append(f"level window opens at {level_lo*1e6:.3f} us, before the flat top at "
                     f"{edge['t90']*1e6:.3f} us (includes turn-on)")
    if level_hi > edge["f90"]:
        warns.append(f"level window closes at {level_hi*1e6:.3f} us, after the flat top ends at "
                     f"{edge['f90']*1e6:.3f} us (includes turn-off)")
    if start - w["t_integrate_pre"] > edge["t10"]:
        raise ValueError("the integration window opens after the rise starts -- increase t_integrate_pre")
    if start + t_duration + w["t_integrate_post"] < edge["f10"]:
        raise ValueError("the integration window closes before the fall finishes -- increase t_integrate_post")

    integral = np.full((n_shots, n_pulses), np.nan)
    level = np.full((n_shots, n_pulses), np.nan)
    baseline = np.full((n_shots, n_pulses), np.nan)
    valid = np.zeros((n_shots, n_pulses), dtype=bool)
    for s, (ts, vs) in enumerate(zip(t_list, v_list)):
        ts = np.asarray(ts, dtype=float).ravel()
        vs = np.asarray(vs, dtype=float).ravel()
        mono = bool(np.all(np.diff(ts) >= 0))
        for p in range(n_pulses):
            ps = start + p * t_between
            t_base_lo = ps - w["t_baseline_pre"]
            t_int_lo = ps - w["t_integrate_pre"]
            t_int_hi = ps + t_duration + w["t_integrate_post"]
            t_lvl_lo = ps + w["t_start_shift"]
            t_lvl_hi = ps + t_duration - w["t_end_shift"]
            if t_base_lo < ts.min() or t_int_hi > ts.max():
                continue
            if mono:   # same selections as the notebook's boolean masks, via index ranges
                b0, b1 = np.searchsorted(ts, [t_base_lo, t_int_lo], side="left")
                i0 = b1
                i1 = np.searchsorted(ts, t_int_hi, side="right")
                l0 = np.searchsorted(ts, t_lvl_lo, side="left")
                l1 = np.searchsorted(ts, t_lvl_hi, side="right")
                vb, ti, vi, vl = vs[b0:b1], ts[i0:i1], vs[i0:i1], vs[l0:l1]
            else:
                mb = (ts >= t_base_lo) & (ts < t_int_lo)
                mi = (ts >= t_int_lo) & (ts <= t_int_hi)
                ml = (ts >= t_lvl_lo) & (ts <= t_lvl_hi)
                vb, ti, vi, vl = vs[mb], ts[mi], vs[mi], vs[ml]
            if vb.size < 2 or vi.size < 2 or vl.size == 0:
                continue
            v_base = float(vb.mean())
            integral[s, p] = float(_trapz(vi - v_base, ti))
            level[s, p] = float(vl.mean() - v_base)
            baseline[s, p] = v_base
            valid[s, p] = True
    n_bad = int((~valid).sum())
    if n_bad:
        warns.append(f"{n_bad} shot-pulse window(s) fall outside their trace and are NaN")
    return ScopePulses(integral=integral, level=level, baseline=baseline, valid=valid, pulse0_edge=edge,
                       pulse0_start=start, pulse0_start_auto=float(start_auto), t_duration=t_duration,
                       t_between_pulses=t_between, windows=w, warnings=tuple(warns),
                       problems=check_timing(edge, level, t_duration))
