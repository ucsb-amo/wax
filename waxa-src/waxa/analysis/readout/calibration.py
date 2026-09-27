"""Calibrate a spin-state readout signal (e.g. an integrated APD pulse) against a Rabi flop.

Input: a pulse-length scan of a drive that rotates a spin prepared in ``S_z = +1``,
followed by ``P`` readout pulses per shot (``signal`` shaped (n_shots, P)), from one or
more runs.  Output (:class:`ReadoutCalibration`): the signal at ``S_z = +1`` and
``S_z = -1`` per pulse, per run and pooled; the ``S_z`` response shape (linear, or the
quadratic midpoint remap of the feedback model); the single-pulse noise at each
endpoint; whether the runs agree; and, when scope traces are given, the photon number
of one pulse from the absolute scope integral.

How ``S_z`` is assigned to each shot (``sz_source``)
----------------------------------------------------
``'fit'`` (default): a joint Rabi fit of every pulse (:func:`waxa.analysis.rabi.fit_rabi`,
shared omega / phase / decay, own offset and amplitude per pulse, every shot, noise
from the residuals) gives the flop coordinate

    S_z(t) = env(t) cos(omega max(t - t_dead, 0))

(``t_dead`` = commanded minus effective length; a shot shorter than the dead time is
not rotated).  ``'params'``: the notebook's ideal-flop mapping with a given pi time and
offset (:func:`.sz_response.pulse_time_to_angle_and_sz`), which is only as good as those
two numbers.  Both are always computed when ``t_pi`` is given, and the fitted pi time is
compared with it.

Endpoints
---------
Every endpoint is a least-squares fit of ``signal = A + B S_z (+ C S_z^2)`` to single
shots (:func:`.sz_response.fit_sz_response`): ``V_up = y(+1)``, ``V_down = y(-1)``.
They are therefore extrapolations of the whole flop, not the mean of the few shots that
sit at the extremes; the direct means at the extremes are reported beside them.  If the
drive does not transfer fully (e.g. it is detuned), the flop's minimum is not
``S_z = -1`` and ``V_down`` is biased toward ``V_up``: this data cannot tell.

The notebook-faithful path (``notebook``) is also run: pulse-averaged signal per shot,
grouped by pulse length, merged by ``S_z``, SEM-weighted quadratic fit, endpoint noise
read off the per-point std (``k-jam/analysis/artisinal/apd_pulse_analysis_interpolate.ipynb``
cells 9-10 and 12/16 for the scope).  Its noise numbers are the std of a PULSE-AVERAGED
signal, which is smaller than the single-pulse noise a one-pulse measurement sees;
:class:`NoiseEstimate` gives the single-pulse numbers.

Noise
-----
Residuals of single pulses about a per-run linear model (within-run noise) and about
one pooled model (which adds any run-to-run shift), taken at each endpoint
(``|S_z| >= 1 - endpoint_window``), with a shot-resampling bootstrap interval, and split
into a part common to all pulses of a shot and an independent part.

Run agreement
-------------
Nested F-tests on the pulse-averaged signal: one line for every run vs own offset per
run vs own offset and slope per run; chi2 of the per-run ``V_up``, ``V_down`` and
contrast about their weighted means; residual drift against acquisition time within
each run (when timestamps are given).  Pooling is flagged when any test has p < 0.01.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .sz_response import (collapse_by_sz, fit_sz_response, group_by_x, interp_vs_sz,
                          pulse_time_to_angle_and_sz, sz_response, weighted_lstsq)

POOLING_P_LIMIT = 0.01
DEFAULT_ENDPOINT_WINDOW = 0.1


# ------------------------------------------------------------------ small pieces
@dataclass(frozen=True)
class Endpoints:
    """``signal = A + B S_z (+ C S_z^2)`` fitted to single shots; all 1-sigma errors."""
    label: str
    v_up: float
    v_up_err: float
    v_down: float
    v_down_err: float
    contrast: float           # v_up - v_down
    contrast_err: float
    degree: int
    midpoint: float           # 0.5 exactly for degree 1
    midpoint_err: float
    rms: float                # residual std of single shots about the fit
    n_shots: int
    dof: int
    rss: float

    def to_dict(self):
        return dict(self.__dict__)


def _endpoints(label, sz, y, degree):
    sz = np.asarray(sz, float)
    y = np.asarray(y, float)
    ok = np.isfinite(sz) & np.isfinite(y)
    f = fit_sz_response(sz[ok], y[ok], degree=degree)
    dof = int(f["dof"])
    rss = float(np.sum(f["residual"] ** 2))
    return Endpoints(label=label, v_up=float(f["y_up"]), v_up_err=float(f["std_y_up"]),
                     v_down=float(f["y_down"]), v_down_err=float(f["std_y_down"]),
                     contrast=float(f["y_range"]), contrast_err=float(f["std_y_range"]), degree=int(degree),
                     midpoint=float(f["midpoint_fraction"]),
                     midpoint_err=float(f["std_midpoint_fraction"]) if degree == 2 else 0.0,
                     rms=float(np.sqrt(rss / max(dof, 1))), n_shots=int(ok.sum()), dof=dof, rss=rss)


def _chi2_consistency(values, errors):
    """(weighted mean, its error, chi2, dof, p) for values that should agree."""
    from scipy.stats import chi2 as chi2_dist
    v = np.asarray(values, float)
    e = np.asarray(errors, float)
    w = 1.0 / e ** 2
    mean = float(np.sum(w * v) / np.sum(w))
    err = float(1.0 / np.sqrt(np.sum(w)))
    c2 = float(np.sum(((v - mean) / e) ** 2))
    dof = v.size - 1
    return mean, err, c2, dof, float(chi2_dist.sf(c2, dof)) if dof > 0 else np.nan


def _f_test(rss_small, k_small, rss_big, k_big, n):
    """Nested least-squares models: (F, dfn, dfd, p) for 'the extra k_big - k_small
    parameters are not needed'."""
    from scipy.stats import f as f_dist
    dfn, dfd = k_big - k_small, n - k_big
    if dfn <= 0 or dfd <= 0 or rss_big <= 0:
        return np.nan, dfn, dfd, np.nan
    F = ((rss_small - rss_big) / dfn) / (rss_big / dfd)
    return float(F), int(dfn), int(dfd), float(f_dist.sf(F, dfn, dfd))


def _ols(design, y):
    p, cov, r = weighted_lstsq(design, y)
    return p, cov, r, float(np.sum(r ** 2))


def flop_coordinate(t, omega, t_dead, gamma=0.0, model="none"):
    """``S_z(t) = env(t) cos(omega max(t - t_dead, 0))`` for a spin prepared in +1."""
    from waxa.analysis.rabi import envelope
    t = np.asarray(t, float)
    return envelope(t, gamma, model) * np.cos(omega * np.clip(t - t_dead, 0.0, None))


def _joint_flop_fit(t, y, flop, degree, label="pooled"):
    """``y = A + B s + C s^2`` with ``s = env(t) cos(omega max(t - t_dead, 0))`` and omega,
    t_dead fitted TOGETHER with the response (env held at the flop fit's decay).

    A plain cosine fit absorbs part of a quadratic detection curve into its phase and
    frequency, so taking S_z from it and then fitting the curvature underestimates the
    curvature (a synthetic midpoint of 0.66 came back as 0.58).  Fitting both at once
    removes that bias; the omega / t_dead uncertainty is carried into every error."""
    from scipy.optimize import least_squares
    from waxa.analysis.rabi import envelope
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    env = envelope(t, flop.gamma, flop.model)
    s0 = flop_coordinate(t, flop.omega, flop.t_dead, flop.gamma, flop.model)
    C0, B0, A0 = fit_sz_response(s0, y, degree=degree)["coeff"]
    quad = degree == 2
    p0 = [A0, B0] + ([C0] if quad else []) + [flop.omega, flop.t_dead]

    def split(p):
        return p[0], p[1], (p[2] if quad else 0.0), p[-2], p[-1]

    def model(p):
        A, B, C, om, t0 = split(p)
        sz = env * np.cos(om * np.clip(t - t0, 0.0, None))
        return A + B * sz + C * sz * sz

    lo = np.full(len(p0), -np.inf)
    lo[-2] = 0.0
    r = least_squares(lambda p: y - model(p), p0, bounds=(lo, np.full(len(p0), np.inf)), x_scale="jac",
                      max_nfev=20000)
    A, B, C, om, t0 = split(r.x)
    k = r.x.size
    dof = max(y.size - k, 1)
    rss = float(np.sum(r.fun ** 2))
    J = r.jac
    try:
        cov = np.linalg.inv(J.T @ J) * rss / dof
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J) * rss / dof

    def err(g):
        g = np.asarray(g, float)
        return float(np.sqrt(max(g @ cov @ g, 0.0)))
    z = [0.0, 0.0]
    if quad:
        g_up, g_dn, g_rng = [1, 1, 1] + z, [1, -1, 1] + z, [0, 2, 0] + z
        g_mid = [0, C / (2 * B ** 2), -1 / (2 * B)] + z
    else:
        g_up, g_dn, g_rng, g_mid = [1, 1] + z, [1, -1] + z, [0, 2] + z, None
    e = Endpoints(label=label, v_up=float(A + B + C), v_up_err=err(g_up), v_down=float(A - B + C),
                  v_down_err=err(g_dn), contrast=float(2 * B), contrast_err=err(g_rng), degree=int(degree),
                  midpoint=float(0.5 - C / (2 * B)), midpoint_err=err(g_mid) if quad else 0.0,
                  rms=float(np.sqrt(rss / dof)), n_shots=int(y.size), dof=int(dof), rss=rss)
    om_f, t0_f = float(om), float(t0)
    return dict(endpoints=e, omega=om_f, omega_err=float(np.sqrt(max(cov[-2, -2], 0))), t_dead=t0_f,
                t_dead_err=float(np.sqrt(max(cov[-1, -1], 0))), success=bool(r.success), k=int(k), rss=rss,
                sz_of=lambda tt: flop_coordinate(tt, om_f, t0_f, flop.gamma, flop.model))


# --------------------------------------------------------------------- noise
@dataclass(frozen=True)
class NoiseEstimate:
    """Single-pulse noise of the signal (same units as the signal)."""
    label: str
    sigma: float
    lo: float                 # bootstrap 16th percentile (shots resampled)
    hi: float                 # bootstrap 84th percentile
    n_shots: int
    n_values: int             # shots x pulses

    def to_dict(self):
        return dict(self.__dict__)


def _sigma_boot(res, rows, label, dof_factor, n_boot, rng):
    r = np.asarray(res)[rows]
    r = r[np.all(np.isfinite(r), axis=1)] if r.ndim == 2 else r[np.isfinite(r)]
    n = r.shape[0]
    if n < 3:
        return NoiseEstimate(label, np.nan, np.nan, np.nan, int(n), int(r.size))
    sig = float(np.sqrt(np.mean(r ** 2)) * dof_factor)
    boots = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, n, n)
        boots.append(np.sqrt(np.mean(r[pick] ** 2)) * dof_factor)
    lo, hi = (float(x) for x in np.percentile(boots, [16, 84])) if boots else (np.nan, np.nan)
    return NoiseEstimate(label, sig, lo, hi, int(n), int(r.size))


# -------------------------------------------------------------------- result
@dataclass(frozen=True, eq=False)
class ReadoutCalibration:
    """Everything :func:`calibrate_readout` found.  Signal units throughout (V for an
    APD), times in s."""
    ok: bool
    reason: str = ""
    signal_name: str = "signal"
    run_ids: tuple = ()
    n_shots: int = 0
    n_pulses: int = 0
    pulses_used: tuple = ()               # 0-based pulse indices averaged for the pooled numbers
    sz_source: str = "fit"
    degree_used: int = 1
    # the flop
    flop: object = None                   # RabiFit (joint over pulses, all runs)
    flop_per_run: tuple = ()              # RabiFit per run
    t_pi_ref: float = np.nan              # the pi time the 'params' mapping uses
    t_offset_ref: float = 0.0
    sz: np.ndarray = field(default_factory=lambda: np.array([]))            # per shot, source in use
    sz_params: np.ndarray = field(default_factory=lambda: np.array([]))     # notebook mapping
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    signal: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    run: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: Optional[np.ndarray] = None
    # endpoints
    pooled: Optional[Endpoints] = None            # pulse-averaged, every run, degree_used
    pooled_linear: Optional[Endpoints] = None
    pooled_quadratic: Optional[Endpoints] = None
    per_pulse: tuple = ()                         # Endpoints per pulse (every run, linear)
    per_run: tuple = ()                           # Endpoints per run (pulse-averaged, linear)
    per_run_pulse: tuple = ()                     # [run][pulse] Endpoints (linear)
    pulse_differences: tuple = ()                 # Endpoints of y_p - mean(others), linear
    direct: dict = field(default_factory=dict)    # model-free means at the extremes
    notebook: dict = field(default_factory=dict)  # notebook-faithful S_z response
    quadratic_test: dict = field(default_factory=dict)
    # noise
    noise: dict = field(default_factory=dict)     # name -> NoiseEstimate
    noise_decomposition: dict = field(default_factory=dict)
    noise_vs_sz: dict = field(default_factory=dict)
    residual_within: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # runs
    agreement: dict = field(default_factory=dict)
    pooling_justified: bool = True
    # scope
    scope: dict = field(default_factory=dict)
    # proposal
    proposal: dict = field(default_factory=dict)
    warnings: tuple = ()
    xvarname: str = "t_pulse"
    params: object = None
    context: str = ""

    # ---------------------------------------------------------------- report
    def summary(self):
        if not self.ok:
            return f"readout calibration FAILED ({self.reason})"
        mv = 1e3
        L = []
        runs = ", ".join(map(str, self.run_ids)) or "data"
        pu = ", ".join(str(p + 1) for p in self.pulses_used)
        L.append(f"run {runs}: {self.signal_name} vs S_z, {self.n_shots} shots x {self.n_pulses} pulses; "
                 f"pooled numbers average pulses {pu}; S_z from '{self.sz_source}'")
        if self.context:
            L.append(f"  {self.context}")
        f = self.flop
        if f is not None and f.ok:
            L.append(f"  flop (joint, every pulse and run): f_Rabi {f.f_rabi/1e3:.2f} +/- {f.f_rabi_err/1e3:.2f} kHz, "
                     f"pi {f.t_pi*1e6:.3f} +/- {f.t_pi_err*1e6:.3f} us, dead time {f.t_dead*1e9:+.0f} +/- "
                     f"{f.t_dead_err*1e9:.0f} ns, model '{f.model}', {f.n_periods:.2f} periods scanned")
            if np.isfinite(self.t_pi_ref):
                d = self.t_pi_ref - f.t_pi
                L.append(f"    pi time in use {self.t_pi_ref*1e6:.4f} us (offset {self.t_offset_ref*1e9:.0f} ns): "
                         f"{d*1e9:+.0f} ns vs fit ({d/f.t_pi_err:+.1f} sigma)")
            for rid, fr in zip(self.run_ids, self.flop_per_run):
                if fr is not None and fr.ok:
                    L.append(f"    run {rid}: pi {fr.t_pi*1e6:.3f} +/- {fr.t_pi_err*1e6:.3f} us, "
                             f"f_Rabi {fr.f_rabi/1e3:.2f} +/- {fr.f_rabi_err/1e3:.2f} kHz")
            for w in f.warnings:
                L.append(f"    flop WARNING: {w}")

        def e(x: Endpoints, extra=""):
            return (f"up {x.v_up*mv:8.2f} +/- {x.v_up_err*mv:.2f}  down {x.v_down*mv:8.2f} +/- {x.v_down_err*mv:.2f}  "
                    f"contrast {x.contrast*mv:6.2f} +/- {x.contrast_err*mv:.2f} mV  (n={x.n_shots}, rms "
                    f"{x.rms*mv:.2f} mV){extra}")
        L.append("  endpoints (m" + f"V; fit of single shots on S_z, 1-sigma; per-pulse/per-run degree {self.degree_used}):")
        L.append(f"    pooled, linear      " + e(self.pooled_linear))
        q = self.pooled_quadratic
        L.append(f"    pooled, quadratic   " + e(q, f"  midpoint {q.midpoint:.3f} +/- {q.midpoint_err:.3f}"))
        qt = self.quadratic_test
        if qt:
            L.append(f"    quadratic term{' (flop refit jointly)' if qt.get('joint_flop') else ''}: midpoint - 0.5 = "
                     f"{qt['midpoint_shift']:+.3f} +/- {qt['midpoint_err']:.3f} "
                     f"({qt['n_sigma']:.1f} sigma), F-test p = {qt['p']:.3g} -> "
                     f"{'curvature resolved' if qt['significant'] else 'NOT resolved: linear map used'}")
        for x in self.per_pulse:
            L.append(f"    {x.label:<20}" + e(x))
        for x in self.per_run:
            L.append(f"    {x.label:<20}" + e(x))
        L.append("  pulse p minus the mean of the other pulses (every run; shot-common noise cancels):")
        for x in self.pulse_differences:
            su = x.v_up / x.v_up_err if x.v_up_err > 0 else np.nan
            sd = x.v_down / x.v_down_err if x.v_down_err > 0 else np.nan
            sc = x.contrast / x.contrast_err if x.contrast_err > 0 else np.nan
            L.append(f"    {x.label:<20}d_up {x.v_up*mv:+6.2f} +/- {x.v_up_err*mv:.2f} ({su:+.1f} s)  "
                     f"d_down {x.v_down*mv:+6.2f} +/- {x.v_down_err*mv:.2f} ({sd:+.1f} s)  "
                     f"d_contrast {x.contrast*mv:+6.2f} +/- {x.contrast_err*mv:.2f} ({sc:+.1f} s)")
        d = self.direct
        if d:
            L.append(f"  direct means (no model): up (t = 0, no pulse) {d['up_t0']*mv:.2f} +/- {d['up_t0_sem']*mv:.2f} mV "
                     f"(n={d['up_t0_n']} shots); flop max side (S_z > {1 - d['window']:.2f}) {d['up']*mv:.2f} +/- "
                     f"{d['up_sem']*mv:.2f} (n={d['up_n']}); min side (S_z < {-1 + d['window']:.2f}) "
                     f"{d['down']*mv:.2f} +/- {d['down_sem']*mv:.2f} (n={d['down_n']})")
        nb = self.notebook
        if nb:
            L.append(f"  notebook path (pulse-averaged, grouped by S_z, SEM-weighted deg-{nb['degree']}, "
                     f"endpoints '{nb['source']}'): up {nb['v_up']*mv:.2f} +/- {nb['v_up_err']*mv:.2f}, down "
                     f"{nb['v_down']*mv:.2f} +/- {nb['v_down_err']*mv:.2f} mV, midpoint {nb['midpoint']:.3f} +/- "
                     f"{nb['midpoint_err']:.3f}; its endpoint std (of the PULSE AVERAGE) up {nb['std_up']*mv:.2f}, "
                     f"down {nb['std_down']*mv:.2f} mV")
        L.append("  single-pulse noise (residual std; bootstrap 16-84 %):")
        for k, n in self.noise.items():
            L.append(f"    {k:<26}{n.sigma*mv:6.2f} mV  [{n.lo*mv:.2f}, {n.hi*mv:.2f}]  ({n.n_shots} shots, "
                     f"{n.n_values} values)")
        nd = self.noise_decomposition
        if nd:
            L.append(f"    split: shot-common {nd['common']*mv:.2f} mV + independent per pulse {nd['independent']*mv:.2f} mV "
                     f"(single pulse {nd['single']*mv:.2f}; mean of {nd['n_avg']} pulses {nd['avg']*mv:.2f}); "
                     f"mean pulse-pulse residual correlation {nd['mean_corr']:.2f}")
        a = self.agreement
        if a:
            L.append("  run agreement (pulse-averaged signal):")
            for key in ("offset", "slope"):
                t_ = a.get(f"f_{key}")
                if t_:
                    L.append(f"    own {key} per run: F = {t_['F']:.2f} ({t_['dfn']}, {t_['dfd']}), p = {t_['p']:.3g}")
            for key in ("v_up", "v_down", "contrast"):
                c = a.get(key)
                if c:
                    L.append(f"    {key:<9} chi2 {c['chi2']:.1f}/{c['dof']}, p = {c['p']:.3g} "
                             f"(weighted mean {c['mean']*mv:.2f} +/- {c['err']*mv:.2f} mV)")
            for rid, dr in a.get("drift", {}).items():
                L.append(f"    run {rid}: residual drift {dr['slope']*mv*60:+.2f} +/- {dr['slope_err']*mv*60:.2f} mV/min "
                         f"({dr['n_sigma']:+.1f} sigma) over {dr['span']/60:.1f} min")
            L.append(f"    -> pooling {'justified (no test at p < ' if self.pooling_justified else 'NOT justified (a test at p < '}"
                     f"{POOLING_P_LIMIT})")
        s = self.scope
        if s:
            L.append("  scope (absolute photon reference)" + ("" if s["valid"] else
                     " -- TIMING CHECK FAILED, numbers NOT used:"))
            L.append("    " + s["summary"].replace("\n", "\n    "))
            if s.get("failed"):
                L.append(f"    scope fits failed: {s['problems'][-1]}")
        if s and not s.get("failed"):
            L.append(f"    photons per pulse, single-shot fit, pooled pulses: up {s['n_up']:.1f} +/- {s['n_up_err']:.1f}, "
                     f"down {s['n_down']:.1f} +/- {s['n_down_err']:.1f}, range {s['n_range']:.1f} +/- {s['n_range_err']:.1f} "
                     f"({s['units']})")
            L.append(f"    scope midpoint (deg-2 single shots) {s['midpoint']:.3f} +/- {s['midpoint_err']:.3f}; notebook path "
                     f"{s['notebook_midpoint']:.3f} +/- {s['notebook_midpoint_err']:.3f}; APD vs scope midpoints "
                     f"{s['midpoint_n_sigma']:.1f} sigma apart")
            L.append(f"    single-pulse photon noise: up {s['sigma_up']:.1f}, down {s['sigma_down']:.1f} ({s['units']}); "
                     f"notebook (pulse-avg) up {s['notebook_std_up']:.1f}, down {s['notebook_std_down']:.1f}")
            L.append(f"    APD-scope residual correlation per pulse: "
                     + ", ".join(f"{r:.2f}" for r in s["residual_corr"]))
            L.append(f"    effective pulse duration (integral / flat top) {s['t_eff_min']*1e6:.3f}-{s['t_eff_max']*1e6:.3f} us")
        p = self.proposal
        if p:
            L.append("  proposal:")
            for k, v in p.items():
                if k != "notes":
                    L.append(f"    {k} = {v}")
            for n in p.get("notes", ()):
                L.append(f"    note: {n}")
        for w in self.warnings:
            L.append(f"  WARNING: {w}")
        return "\n".join(L)

    def __str__(self):
        return self.summary()

    def _repr_pretty_(self, p, cycle):
        p.text(self.summary())

    def to_dict(self):
        def conv(o):
            if isinstance(o, (Endpoints, NoiseEstimate)):
                return o.to_dict()
            if isinstance(o, dict):
                return {str(k): conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [conv(v) for v in o]
            if isinstance(o, np.ndarray):
                return conv(o.tolist())
            if isinstance(o, (np.floating, float)):
                return float(o) if np.isfinite(o) else None
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if hasattr(o, "to_dict") and callable(o.to_dict):
                return conv(o.to_dict())
            if isinstance(o, (str, int, bool)) or o is None:
                return o
            return str(o)
        skip = {"params", "flop", "flop_per_run", "t", "signal", "run", "timestamps", "sz", "sz_params",
                "residual_within"}
        out = {k: conv(getattr(self, k)) for k in self.__dataclass_fields__ if k not in skip}
        out["flop"] = conv(_flop_dict(self.flop))
        out["flop_per_run"] = [conv(_flop_dict(f)) for f in self.flop_per_run]
        out["shots"] = conv(dict(t=self.t, run=self.run, sz=self.sz, sz_params=self.sz_params,
                                 signal=self.signal, timestamps=self.timestamps))
        return out

    def plot_pulses(self, **kw):
        from .plotting import plot_pulses
        return plot_pulses(self, **kw)

    def plot_summary(self, **kw):
        from .plotting import plot_summary
        return plot_summary(self, **kw)


def _flop_dict(f):
    if f is None:
        return None
    keys = ("ok", "model", "noise", "omega", "omega_err", "phase", "phase_err", "gamma", "gamma_err", "t_dead",
            "t_dead_err", "t_pi", "t_pi_err", "t_half_pi", "t_half_pi_err", "t_pi_rate", "t_pi_rate_err", "chi2",
            "dof", "chi2r", "n_shots", "n_periods", "offsets", "amplitudes", "warnings", "reason", "aicc")
    d = {k: getattr(f, k, None) for k in keys}
    d["f_rabi"], d["f_rabi_err"] = f.f_rabi, f.f_rabi_err
    return d


# --------------------------------------------------------------------- entry
def calibrate_readout(t_pulse, signal, *, run=None, run_ids=None, timestamps=None, t_pi=None, t_offset=0.0,
                      sz_source="fit", pulses_used=None, degree="auto", endpoint_window=DEFAULT_ENDPOINT_WINDOW,
                      scope=None, scope_to_photons=None, n_boot=2000, seed=0, flop_model="auto",
                      signal_name="signal", xvarname="t_pulse", params=None, context="") -> ReadoutCalibration:
    """Calibrate a readout signal against a Rabi flop.

    Parameters
    ----------
    t_pulse : (n_shots,) commanded drive pulse length per shot (s).
    signal : (n_shots,) or (n_shots, n_pulses) readout per shot and readout pulse.
    run : (n_shots,) run label per shot (default: one run).  ``run_ids`` orders them.
    timestamps : (n_shots,) acquisition time per shot (s), for the drift check.
    t_pi, t_offset : pi time and offset for the ideal-flop ('params') mapping.
    sz_source : 'fit' (joint Rabi fit) or 'params'.
    pulses_used : pulse indices (0-based) averaged for the pooled numbers (default all).
    degree : 1, 2 or 'auto' (2 only if the quadratic term is resolved: its midpoint is
        more than 2 sigma from 1/2 AND the F-test gives p < 0.05; otherwise 1).
    endpoint_window : shots with ``|S_z| >= 1 - endpoint_window`` enter the endpoint noise.
    scope : optional :class:`~.scope_pulses.ScopePulses` for the same shots, in order.
    scope_to_photons : factor from scope V s to photons (None: stay in V s).
    """
    from waxa.analysis.rabi import fit_rabi

    warn = []
    t = np.asarray(t_pulse, float).ravel()
    y = np.asarray(signal, float)
    if y.ndim == 1:
        y = y[:, None]
    n, P = y.shape
    common = dict(signal_name=signal_name, xvarname=xvarname, params=params, context=context)
    if t.size != n:
        return ReadoutCalibration(False, reason=f"t_pulse has {t.size} shots, signal {n}", **common)
    run = np.zeros(n, int) if run is None else np.asarray(run).ravel()
    if run.size != n:
        return ReadoutCalibration(False, reason="run must have one label per shot", **common)
    if run_ids is None:
        _, first = np.unique(run, return_index=True)
        run_ids = tuple(run[np.sort(first)].tolist())
    run_ids = tuple(run_ids)
    ts = None if timestamps is None else np.asarray(timestamps, float).ravel()
    used = tuple(range(P)) if pulses_used is None else tuple(int(p) for p in pulses_used)
    if not used or min(used) < 0 or max(used) >= P:
        return ReadoutCalibration(False, reason=f"pulses_used {used} outside 0..{P - 1}", **common)
    finite = np.all(np.isfinite(y), axis=1) & np.isfinite(t)
    if not np.all(finite):
        warn.append(f"{int((~finite).sum())} shot(s) with a non-finite signal or pulse length are left out "
                    "of every fit")
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ the flop
    flop = fit_rabi([t[finite]] * P, [y[finite, p] for p in range(P)], model=flop_model, noise="unweighted",
                    labels=[f"pulse {p + 1}" for p in range(P)], run_ids=run_ids, xvarname=xvarname,
                    signal_name=signal_name, params=params)
    flop_runs = []
    for r in run_ids:
        m = finite & (run == r)
        flop_runs.append(fit_rabi([t[m]] * P, [y[m, p] for p in range(P)], model=flop.model if flop.ok else "none",
                                  noise="unweighted", run_ids=(r,), xvarname=xvarname, signal_name=signal_name)
                         if m.sum() >= 10 else None)
    sz_par = (pulse_time_to_angle_and_sz(t, t_pi, t_offset)[1] if t_pi is not None
              else np.full(n, np.nan))
    if sz_source == "fit":
        if not flop.ok:
            return ReadoutCalibration(False, reason=f"joint Rabi fit failed: {flop.reason}", flop=flop, **common)
        sz = flop_coordinate(t, flop.omega, flop.t_dead, flop.gamma, flop.model)
    elif sz_source == "params":
        if t_pi is None:
            return ReadoutCalibration(False, reason="sz_source='params' needs t_pi", **common)
        sz = sz_par
    else:
        return ReadoutCalibration(False, reason="sz_source must be 'fit' or 'params'", **common)
    if flop.ok and not flop.starts_populated:
        warn.append("the signal starts at its flop MINIMUM: the readout is inverted relative to the drive "
                    "(V_up < V_down); the numbers below are still V at S_z = +1 / -1")

    ybar = np.mean(y[:, list(used)], axis=1)
    fin = finite

    # ------------------------------------------------------------ endpoints
    if degree not in ("auto", 1, 2):
        return ReadoutCalibration(False, reason="degree must be 1, 2 or 'auto'", **common)
    joint = {}
    jq = None
    if sz_source == "fit":
        # the curvature test refits the flop together with the response (see _joint_flop_fit)
        jl = _joint_flop_fit(t[fin], ybar[fin], flop, 1)
        jq = _joint_flop_fit(t[fin], ybar[fin], flop, 2)
        quad = jq["endpoints"]
        F2, dfn2, dfd2, p2 = _f_test(jl["rss"], jl["k"], jq["rss"], jq["k"], int(fin.sum()))
        joint = dict(linear=jl["endpoints"], quadratic=quad,
                     omega_linear=jl["omega"], t_dead_linear=jl["t_dead"],
                     omega_quadratic=jq["omega"], omega_quadratic_err=jq["omega_err"],
                     t_dead_quadratic=jq["t_dead"], t_dead_quadratic_err=jq["t_dead_err"])
    else:
        lin0 = _endpoints("pooled", sz[fin], ybar[fin], 1)
        quad = _endpoints("pooled", sz[fin], ybar[fin], 2)
        F2, dfn2, dfd2, p2 = _f_test(lin0.rss, 2, quad.rss, 3, lin0.n_shots)
    shift = quad.midpoint - 0.5
    nsig = abs(shift) / quad.midpoint_err if quad.midpoint_err > 0 else np.inf
    resolved = bool(nsig > 2 and p2 < 0.05)
    qtest = dict(midpoint_shift=shift, midpoint_err=quad.midpoint_err, n_sigma=nsig, F=F2, dfn=dfn2, dfd=dfd2,
                 p=p2, significant=resolved, joint_flop=(sz_source == "fit"), **joint)
    deg = (2 if resolved else 1) if degree == "auto" else int(degree)
    if deg == 2 and jq is not None:
        sz = jq["sz_of"](t)             # S_z from the flop fitted together with the quadratic response
    lin = _endpoints("pooled", sz[fin], ybar[fin], 1)
    pooled = quad if deg == 2 else lin
    per_pulse = tuple(_endpoints(f"pulse {p + 1}", sz[fin], y[fin, p], deg) for p in range(P))
    per_run, per_run_pulse = [], []
    for r in run_ids:
        m = fin & (run == r)
        per_run.append(_endpoints(f"run {r}", sz[m], ybar[m], deg))
        per_run_pulse.append(tuple(_endpoints(f"run {r} pulse {p + 1}", sz[m], y[m, p], deg) for p in range(P)))
    diffs = []
    if P > 1:
        for p in range(P):
            others = [q for q in range(P) if q != p]
            d = y[:, p] - y[:, others].mean(axis=1)
            diffs.append(_endpoints(f"pulse {p + 1} - others", sz[fin], d[fin], deg))

    # direct (model-free) means at the extremes
    w = float(endpoint_window)
    t0 = fin & (t == 0)
    upm = fin & (sz >= 1 - w)
    dnm = fin & (sz <= -1 + w)

    def _ms(mask):
        v = ybar[mask]
        return (float(v.mean()) if v.size else np.nan,
                float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan, int(v.size))
    direct = {}
    direct["up_t0"], direct["up_t0_sem"], direct["up_t0_n"] = _ms(t0)
    direct["up"], direct["up_sem"], direct["up_n"] = _ms(upm)
    direct["down"], direct["down_sem"], direct["down_n"] = _ms(dnm)
    direct["window"] = w

    # notebook-faithful path (cells 9-10): pulse-averaged per shot, grouped, collapsed, SEM
    nb = {}
    try:
        xu, mean_t, std_t, cnt_t = group_by_x(t[fin], ybar[fin])
        if sz_source == "fit":
            sz_u = flop_coordinate(xu, flop.omega, flop.t_dead, flop.gamma, flop.model)
        else:
            sz_u = pulse_time_to_angle_and_sz(xu, t_pi, t_offset)[1]
        sz_ax, mean_ax, std_ax, n_ax = collapse_by_sz(sz_u, mean_t, std_t, cnt_t)
        sem_ax = std_ax / np.sqrt(np.maximum(n_ax, 1))
        resp = sz_response(sz_ax, mean_ax, yerr=sem_ax, degree=2, source="fit", pinned_comparison=False)
        sd_down, sd_up = (float(v) for v in interp_vs_sz(sz_ax, std_ax, [-1.0, 1.0], strict=False))
        nb = dict(degree=2, source="fit", v_up=resp["y_up"], v_up_err=resp["fit"]["std_y_up"],
                  v_down=resp["y_down"], v_down_err=resp["fit"]["std_y_down"],
                  midpoint=resp["midpoint_fraction"], midpoint_err=resp["std_midpoint_fraction"],
                  v_up_interp=resp["y_up_interp"], v_down_interp=resp["y_down_interp"],
                  std_up=sd_up, std_down=sd_down, rms=resp["fit"]["rms"], sz_axis=sz_ax, mean=mean_ax,
                  std=std_ax, sem=sem_ax, n=n_ax, n_pulses_averaged=len(used))
    except Exception as ex:                        # the notebook path is a cross-check only
        warn.append(f"notebook-path S_z response failed: {ex!r}")

    # ------------------------------------------------------------ residuals / noise
    res_within = np.full((n, P), np.nan)
    res_pooled = np.full((n, P), np.nan)
    k_within = 0
    for p in range(P):
        fp = fit_sz_response(sz[fin], y[fin, p], degree=deg)
        res_pooled[fin, p] = fp["residual"]
        for r in run_ids:
            m = fin & (run == r)
            fr = fit_sz_response(sz[m], y[m, p], degree=deg)
            res_within[m, p] = fr["residual"]
    n_fin = int(fin.sum())
    k_within = (deg + 1) * len(run_ids)
    fac_within = math.sqrt(n_fin / max(n_fin - k_within, 1))
    fac_pooled = math.sqrt(n_fin / max(n_fin - (deg + 1), 1))
    ul = list(used)
    noise = {}
    for tag, res, fac in (("within-run", res_within, fac_within), ("pooled", res_pooled, fac_pooled)):
        noise[f"{tag}, up"] = _sigma_boot(res[:, ul], upm, f"{tag}, up", fac, n_boot, rng)
        noise[f"{tag}, down"] = _sigma_boot(res[:, ul], dnm, f"{tag}, down", fac, n_boot, rng)
        noise[f"{tag}, all shots"] = _sigma_boot(res[:, ul], fin, f"{tag}, all shots", fac, n_boot, rng)
    for p in range(P):
        for side, mask in (("up", upm), ("down", dnm)):
            noise[f"within-run, pulse {p + 1}, {side}"] = _sigma_boot(res_within[:, [p]], mask,
                                                                      f"within-run, pulse {p + 1}, {side}",
                                                                      fac_within, 0, rng)
    rw = res_within[fin][:, ul]
    decomp = {}
    if len(ul) > 1:
        m_shot = rw.mean(axis=1)
        var_ind = float(np.sum((rw - m_shot[:, None]) ** 2) / (rw.shape[0] * (len(ul) - 1)))
        var_mean = float(np.mean(m_shot ** 2)) * fac_within ** 2
        var_com = max(var_mean - var_ind / len(ul), 0.0)
        corr = np.corrcoef(rw.T)
        decomp = dict(common=math.sqrt(var_com), independent=math.sqrt(var_ind),
                      single=math.sqrt(var_com + var_ind), avg=math.sqrt(var_mean), n_avg=len(ul),
                      mean_corr=float(np.mean(corr[np.triu_indices(len(ul), 1)])), corr=corr)
    edges = np.linspace(-1.0, 1.0, 9)
    b_s, b_sig, b_n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = fin & (sz >= lo) & (sz <= hi if hi == 1.0 else sz < hi)
        if m.sum() >= 3:
            r = res_within[m][:, ul]
            b_s.append(float(np.mean(sz[m])))
            b_sig.append(float(np.sqrt(np.mean(r ** 2)) * fac_within))
            b_n.append(int(m.sum()))
    noise_vs_sz = dict(sz=np.array(b_s), sigma=np.array(b_sig), n_shots=np.array(b_n))

    # ------------------------------------------------------------ run agreement
    agreement = {}
    pool_ok = True
    if len(run_ids) > 1:
        R = len(run_ids)
        ridx = np.array([run_ids.index(r) for r in run[fin]])
        S = sz[fin]
        Y = ybar[fin]
        onehot = (ridx[:, None] == np.arange(R)[None, :]).astype(float)
        _, _, _, rss1 = _ols(np.column_stack([np.ones_like(S), S]), Y)
        _, _, _, rss2 = _ols(np.column_stack([onehot, S]), Y)
        _, _, _, rss3 = _ols(np.column_stack([onehot, onehot * S[:, None]]), Y)
        Fo, dno, ddo, po = _f_test(rss1, 2, rss2, R + 1, Y.size)
        Fs, dns, dds, ps = _f_test(rss2, R + 1, rss3, 2 * R, Y.size)
        agreement["f_offset"] = dict(F=Fo, dfn=dno, dfd=ddo, p=po)
        agreement["f_slope"] = dict(F=Fs, dfn=dns, dfd=dds, p=ps)
        for key in ("v_up", "v_down", "contrast"):
            vals = [getattr(x, key) for x in per_run]
            errs = [getattr(x, key + "_err") for x in per_run]
            mean, err, c2, dof, pv = _chi2_consistency(vals, errs)
            agreement[key] = dict(mean=mean, err=err, chi2=c2, dof=dof, p=pv, values=vals, errors=errs)
        pvals = [po, ps] + [agreement[k]["p"] for k in ("v_up", "v_down", "contrast")]
        pool_ok = bool(all((not np.isfinite(pv)) or pv >= POOLING_P_LIMIT for pv in pvals))
        if not pool_ok:
            warn.append(f"the runs do not agree (a test at p < {POOLING_P_LIMIT}): pooled endpoints average over "
                        "a real run-to-run change; see the per-run numbers")
    if ts is not None and ts.size == n:
        drift = {}
        for r in run_ids:
            m = fin & (run == r)
            if m.sum() < 5:
                continue
            tt = ts[m] - ts[m].min()
            rr = res_within[m][:, ul].mean(axis=1)
            pr, cov, _, _ = _ols(np.column_stack([np.ones_like(tt), tt]), rr)
            se = float(np.sqrt(cov[1, 1]))
            drift[r] = dict(slope=float(pr[1]), slope_err=se, n_sigma=float(pr[1] / se) if se > 0 else np.nan,
                            span=float(tt.max()), first_shot_residual=float(rr[np.argmin(tt)]))
        agreement["drift"] = drift
        big = [r for r, dr in drift.items() if abs(dr["n_sigma"]) > 3]
        if big:
            warn.append(f"residual drift over acquisition time > 3 sigma in run(s) {big}")

    # ------------------------------------------------------------ scope
    scope_out = {}
    if scope is not None:
        try:
            scope_out = _scope_block(scope, scope_to_photons, t, sz, run, run_ids, fin, ul, upm, dnm,
                                     res_within, quad, flop, sz_source, t_pi, t_offset, fac_within, warn)
        except Exception as ex:
            # keep the record: the timing problems and the edge summary are still findings
            scope_out = dict(valid=False, failed=True, problems=tuple(scope.problems) + (f"scope fits failed: {ex!r}",),
                             summary=scope.summary(), pulse0_start=scope.pulse0_start, pulse0_edge=scope.pulse0_edge)

    if scope_out and not scope_out["valid"]:
        warn.append(f"scope timing check FAILED ({len(scope_out['problems'])} problem(s), listed in the scope section): "
                    "every scope number is reported for the record only and is NOT used")

    # ------------------------------------------------------------ proposal
    cand = [noise["within-run, up"], noise["within-run, down"]]
    if len(run_ids) > 1:
        cand += [noise["pooled, up"], noise["pooled, down"]]
    cand = [c for c in cand if np.isfinite(c.sigma)]
    sig = max(cand, key=lambda c: c.sigma) if cand else None
    proposal = {}
    if sig is not None:
        v_range = pooled.contrast
        proposal = dict(v_up=pooled.v_up, v_up_err=pooled.v_up_err, v_down=pooled.v_down,
                        v_down_err=pooled.v_down_err, v_range=v_range, v_range_err=pooled.contrast_err,
                        midpoint=pooled.midpoint if deg == 2 else 0.5,
                        midpoint_err=pooled.midpoint_err if deg == 2 else 0.0, degree=deg,
                        sigma_signal=sig.sigma, sigma_signal_source=sig.label,
                        sigma_fraction=sig.sigma / abs(v_range),
                        notes=(f"endpoints and midpoint from the degree-{deg} fit of single shots, pulses "
                               f"{', '.join(str(p + 1) for p in used)} averaged, every run pooled",
                               "sigma = the LARGEST single-pulse endpoint noise (never the smallest)"))
        # run-to-run scatter: scale the pooled errors by the PDG factor sqrt(chi2/dof) of the
        # per-run values when that exceeds 1, so a drift between runs is not hidden by pooling
        for key, perr in (("v_up", "v_up_err"), ("v_down", "v_down_err"), ("contrast", "v_range_err")):
            c = agreement.get(key)
            scale = math.sqrt(max(c["chi2"] / c["dof"], 1.0)) if c and c["dof"] > 0 else 1.0
            proposal[perr.replace("_err", "_err_with_run_scatter")] = proposal[perr] * scale
        if scope_out and scope_out["valid"]:
            proposal["n_photons_per_pulse"] = scope_out["n_range"]
            proposal["n_photons_per_pulse_err"] = scope_out["n_range_err"]
            proposal["std_n_photons"] = scope_out["n_range"] * proposal["sigma_fraction"]
    return ReadoutCalibration(
        ok=True, run_ids=run_ids, n_shots=int(n), n_pulses=int(P), pulses_used=used, sz_source=sz_source,
        degree_used=deg, flop=flop, flop_per_run=tuple(flop_runs),
        t_pi_ref=float(t_pi) if t_pi is not None else np.nan, t_offset_ref=float(t_offset), sz=sz,
        sz_params=sz_par, t=t, signal=y, run=run, timestamps=ts, pooled=pooled, pooled_linear=lin,
        pooled_quadratic=quad, per_pulse=per_pulse, per_run=tuple(per_run), per_run_pulse=tuple(per_run_pulse),
        pulse_differences=tuple(diffs), direct=direct, notebook=nb, quadratic_test=qtest, noise=noise,
        noise_decomposition=decomp, noise_vs_sz=noise_vs_sz, residual_within=res_within, agreement=agreement,
        pooling_justified=pool_ok, scope=scope_out, proposal=proposal, warnings=tuple(warn), **common)


def _scope_block(scope, to_photons, t, sz, run, run_ids, fin, ul, upm, dnm, res_within, apd_quad, flop,
                 sz_source, t_pi, t_offset, fac_within, warn):
    """Photon number from the scope integral: single-shot fits, the notebook path (cell
    16: pulse-averaged, grouped, polarity-flipped and background-subtracted), the
    single-pulse photon noise, and the APD-vs-scope midpoint and residual cross-checks."""
    k = 1.0 if to_photons is None else float(to_photons)
    units = "V s" if to_photons is None else "photons"
    ph = np.asarray(scope.integral, float) * k
    if ph.shape[0] != t.size:
        raise ValueError(f"scope has {ph.shape[0]} shots, signal {t.size}")
    okp = fin & np.all(np.isfinite(ph[:, ul]), axis=1)
    n_bad = int((fin & ~okp).sum())
    if n_bad:
        warn.append(f"scope: {n_bad} shot(s) lack a valid window on a used pulse and are left out of the scope fits")
    pbar = ph[:, ul].mean(axis=1)
    lin = _endpoints("scope pooled", sz[okp], pbar[okp], 1)
    quad = _endpoints("scope pooled", sz[okp], pbar[okp], 2)
    # notebook path
    xu, mean_t, std_t, cnt_t = group_by_x(t[okp], pbar[okp])
    if sz_source == "fit":
        sz_u = flop_coordinate(xu, flop.omega, flop.t_dead, flop.gamma, flop.model)
    else:
        sz_u = pulse_time_to_angle_and_sz(xu, t_pi, t_offset)[1]
    sz_ax, mean_ax, std_ax, n_ax = collapse_by_sz(sz_u, mean_t, std_t, cnt_t)
    sem_ax = std_ax / np.sqrt(np.maximum(n_ax, 1))
    raw = sz_response(sz_ax, mean_ax, yerr=sem_ax, degree=2, source="fit", pinned_comparison=False)
    polarity = float(np.sign(raw["y_up"] - raw["y_down"])) or 1.0
    bg = sz_response(sz_ax, polarity * (mean_ax - raw["y_down"]), yerr=sem_ax, degree=2, source="fit",
                     pinned_comparison=False)
    nb_sd_down, nb_sd_up = (float(v) for v in interp_vs_sz(sz_ax, std_ax, [-1.0, 1.0], strict=False))
    # single-pulse photon noise about per-run linear models
    res = np.full(ph.shape, np.nan)
    for p in ul:
        for r in run_ids:
            m = okp & (run == r)
            res[m, p] = fit_sz_response(sz[m], ph[m, p], degree=1)["residual"]
    rr = res[:, ul]

    def _sd(mask):
        v = rr[mask & okp]
        return float(np.sqrt(np.mean(v ** 2)) * fac_within) if v.size > 2 else np.nan
    corr = []
    for p in ul:
        m = okp & np.isfinite(res_within[:, p]) & np.isfinite(res[:, p])
        corr.append(float(np.corrcoef(res_within[m, p], res[m, p])[0, 1]) if m.sum() > 3 else np.nan)
    d_mid = apd_quad.midpoint - quad.midpoint
    s_mid = math.hypot(apd_quad.midpoint_err, quad.midpoint_err)
    teff = scope.t_effective[:, ul]
    return dict(valid=bool(scope.timing_ok), problems=tuple(scope.problems), units=units, to_photons=to_photons,
                summary=scope.summary(),
                n_up=lin.v_up, n_up_err=lin.v_up_err, n_down=lin.v_down, n_down_err=lin.v_down_err,
                n_range=abs(lin.contrast), n_range_err=lin.contrast_err, polarity=polarity,
                midpoint=quad.midpoint, midpoint_err=quad.midpoint_err,
                notebook_n_up_bg_sub=bg["y_up"], notebook_n_up_bg_sub_err=bg["fit"]["std_y_up"],
                notebook_midpoint=bg["midpoint_fraction"], notebook_midpoint_err=bg["std_midpoint_fraction"],
                notebook_std_up=nb_sd_up, notebook_std_down=nb_sd_down,
                sigma_up=_sd(upm), sigma_down=_sd(dnm), sigma_all=_sd(fin),
                midpoint_n_sigma=abs(d_mid) / s_mid if s_mid > 0 else np.nan,
                residual_corr=corr, t_eff_min=float(np.nanmin(teff)), t_eff_max=float(np.nanmax(teff)),
                n_shots=int(okp.sum()), pulse0_start=scope.pulse0_start, pulse0_edge=scope.pulse0_edge,
                linear=lin, quadratic=quad, per_shot_pulse=ph)
