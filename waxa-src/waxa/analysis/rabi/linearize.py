"""Linearise a detection signal against the spin state using a Rabi flop.

A Rabi flop drives the imaged-state population along a known shape, linear in

    v(t) = env(t) cos(omega t + phi)          (v = +1 / -1 at the flop's extremes)

If the measured signal (summed OD, atom number, ...) is a nonlinear function of the
population, the repeat-averaged flop is not a clean cosine: maxima and minima are
unequally sharp and the zero crossings sit off the midline -- a cubic of a cosine is a
cosine plus phase-locked 2nd and 3rd harmonics.  This module fits

    signal = D(c_j + a_j v(t)),     D(v) = d0 + d1 v + d2 v^2 + d3 v^3

with one detection curve D shared by every run (c_0 = 0, a_0 = 1 fix the gauge; later
runs get their own c_j, a_j because their atom numbers differ), and returns the inverse
map g = D^-1 for remapping any signal taken with the same imaging.

Why the forward direction
-------------------------
The noise lives in the measured signal, so residuals are ``(y - D(v)) / SEM`` with the
measured SEMs held fixed: an ordinary chi2 whose weights cannot depend on the fit.
Fitting the inverse polynomial g(y) directly needs the SEMs propagated through g'
(``sigma' = g' sigma``), and that fit can lower chi2 by making g steep wherever the
residuals are large -- inflating the error bars instead of straightening the data.  On
run 83092 it did exactly that (2026-09-26), so the inverse is computed numerically from
the fitted forward cubic instead.

What it reports
---------------
chi2 and AICc against the plain cosine on the same points and SEMs (AICc charges for
the two extra coefficients), the curve D with its 2nd/3rd-order coefficients and
errors, whether D is monotonic (the inverse exists only then), the pi time from the
joint covariance, and a plain Rabi fit of the remapped shots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .rabi_fit import DEFAULT_SEM_FLOOR, RabiFit, _aicc, _fit_arrays, _prepare, _wrap, envelope, fit_rabi

_P = np.polynomial.polynomial


@dataclass(frozen=True, eq=False)
class SignalMapping:
    """The fitted detection curve D (signal as a function of the flop coordinate v) and
    its inverse.  ``m(y)`` remaps a signal to units linear in the spin state (the same
    units as the input, equal to it at the run-0 flop extremes v = -1 and +1)."""
    ok: bool
    degree: int
    coeffs: np.ndarray = field(default_factory=lambda: np.array([]))       # d0..dn of D(v)
    coeffs_err: np.ndarray = field(default_factory=lambda: np.array([]))
    model: str = "none"
    omega: float = np.nan
    phase: float = np.nan
    gamma: float = 0.0
    centers: tuple = ()                  # c_j (c_0 = 0)
    scales: tuple = ()                   # a_j (a_0 = 1)
    v_range: tuple = (np.nan, np.nan)    # flop coordinate span the data cover
    monotonic: bool = False              # D' > 0 over v_range (inverse exists)
    min_slope: float = np.nan            # min D'(v) / d1 over v_range (1 = linear)
    chi2_raw: float = np.nan
    chi2_mapped: float = np.nan
    dof_raw: int = 0
    dof_mapped: int = 0
    aicc_raw: float = np.nan
    aicc_mapped: float = np.nan
    t_pi: float = np.nan
    t_pi_err: float = np.nan
    t_pi_raw: float = np.nan
    t_pi_raw_err: float = np.nan
    n_shots_unmappable: int = 0          # shots outside the invertible range (NaN after remap)
    warnings: tuple = ()
    reason: str = ""
    fit_raw: Optional[RabiFit] = None
    fit_mapped: Optional[RabiFit] = None
    run_ids: tuple = ()
    signal_name: str = "signal"
    sem_floor: float = DEFAULT_SEM_FLOOR
    fix_cosine: bool = False
    noise: str = "sem"

    # ------------------------------------------------------------------ use
    def detection(self, v):
        """D(v): the signal the fit expects at flop coordinate v."""
        return _P.polyval(np.asarray(v, float), self.coeffs)

    def _grid(self):
        lo, hi = self.v_range
        span = hi - lo
        return np.linspace(lo - 0.5 * span, hi + 0.5 * span, 20001)

    def invertible_range(self):
        """(signal_lo, signal_hi) over which D is monotonic around the data -- the
        domain of the remap."""
        v = self._grid()
        d = _P.polyval(v, _P.polyder(self.coeffs))
        i0 = int(np.argmin(np.abs(v - 0.5 * sum(self.v_range))))
        if d[i0] <= 0:
            return (np.nan, np.nan)
        lo = i0
        while lo > 0 and d[lo - 1] > 0:
            lo -= 1
        hi = i0
        while hi < v.size - 1 and d[hi + 1] > 0:
            hi += 1
        return float(self.detection(v[lo])), float(self.detection(v[hi]))

    def flop_coordinate(self, y):
        """v = D^-1(y); NaN outside :meth:`invertible_range`."""
        if not self.ok or not self.monotonic:
            raise ValueError("the fitted detection curve is not monotonic over the data: no inverse")
        v = self._grid()
        d = _P.polyval(v, _P.polyder(self.coeffs))
        ylo, yhi = self.invertible_range()
        dv = self.detection(v)
        keep = (dv >= ylo) & (dv <= yhi) & (d > 0)
        y = np.asarray(y, float)
        out = np.interp(y, dv[keep], v[keep])
        return np.where((y >= ylo) & (y <= yhi), out, np.nan)

    def __call__(self, y):
        """Remapped signal, linear in the spin state, in input units: equal to D(-1) and
        D(+1) at the run-0 flop extremes."""
        lo, hi = self.detection(-1.0), self.detection(1.0)
        return lo + (self.flop_coordinate(y) + 1) / 2 * (hi - lo)

    remap = __call__

    def fraction(self, y):
        """(v + 1) / 2 for run 0: 0 at the fitted flop minimum, 1 at the maximum.  This
        is the imaged-state population only if the pulse transfers fully."""
        return (self.flop_coordinate(y) + 1) / 2

    # ------------------------------------------------------------- report
    def summary(self):
        if not self.ok:
            return f"signal linearisation FAILED ({self.reason})"
        runs = f"run {', '.join(map(str, self.run_ids))}" if self.run_ids else "data"
        d_aic = self.aicc_mapped - self.aicc_raw
        rel = self.coeffs[2:] / self.coeffs[1]
        rel_e = self.coeffs_err[2:] / abs(self.coeffs[1])
        lines = [
            f"{runs}: {self.signal_name} = D(v), degree-{self.degree} detection curve, Rabi model '{self.model}'"
            + (" (cosine held at the plain fit)" if self.fix_cosine else "") + f", noise '{self.noise}'",
            f"  chi2/dof   cosine {self.chi2_raw:.1f}/{self.dof_raw} = {self.chi2_raw/self.dof_raw:.2f}   "
            f"cubic-of-cosine {self.chi2_mapped:.1f}/{self.dof_mapped} = {self.chi2_mapped/self.dof_mapped:.2f}",
            f"  AICc       nonlinear - linear = {d_aic:+.1f}  "
            f"({'nonlinearity favoured' if d_aic < -2 else 'nonlinearity NOT clearly favoured'})",
            f"  D(v)       d0 {self.coeffs[0]:.4g}, d1 {self.coeffs[1]:.4g}; higher orders / d1 = "
            + ", ".join(f"{r:+.3f} +/- {e:.3f}" for r, e in zip(rel, rel_e)),
            f"  slope      D'/d1 over the data: min {self.min_slope:.2f} (1 = linear); "
            + ("monotonic -> invertible" if self.monotonic else "NOT monotonic -> no inverse"),
            f"  pi pulse   cosine {self.t_pi_raw*1e6:.3f} +/- {self.t_pi_raw_err*1e6:.3f} us  ->  "
            f"with D {self.t_pi*1e6:.3f} +/- {self.t_pi_err*1e6:.3f} us",
        ]
        if self.n_shots_unmappable:
            lines.append(f"  {self.n_shots_unmappable} shot(s) fall outside the invertible range: NaN after remap, "
                         "left out of the remapped Rabi fit")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def _repr_pretty_(self, p, cycle):
        p.text(self.summary())

    def plot(self, figsize=(12.5, 4.2)):
        from .plotting import plot_linearization
        return plot_linearization(self, figsize=figsize)


def _flop_v(t, omega, phase, gamma, model):
    return envelope(t, gamma, model) * np.cos(omega * t + phase)


def linearize_signal(t, y, *, degree=3, model="none", fix_cosine=False, noise="sem",
                     sem_floor=DEFAULT_SEM_FLOOR, run_ids=(), signal_name="signal", xvarname="t_pulse",
                     params=None) -> SignalMapping:
    """Fit ``signal = D(c_j + a_j v(t))`` with a degree-``degree`` detection curve D.

    ``t, y`` are shots (lists of arrays for several runs: D and the Rabi frequency /
    phase / decay are shared; run 0 fixes the gauge c_0 = 0, a_0 = 1).  Repeats are
    required: the fit is to per-point means weighted by their SEM, exactly the points
    and weights of the plain-cosine fit it is compared with.

    ``fix_cosine=True`` holds omega, phase and decay at the plain-cosine fit and fits
    only D (and c_j, a_j): the curve that best maps the raw fitted cosine onto the data.

    ``noise`` is 'sem' (per-point SEMs, floored) or 'pooled' (one noise model
    ``sigma^2 = s0^2 + (f (y - y0))^2`` fitted to all the repeat scatter; see
    :func:`~.rabi_fit.fit_rabi`).  Both fits use the same points and weights.
    """
    from scipy.optimize import least_squares

    ts = list(t) if isinstance(t, (list, tuple)) else [t]
    ys = list(y) if isinstance(y, (list, tuple)) else [y]
    common = dict(run_ids=tuple(run_ids), signal_name=signal_name, sem_floor=float(sem_floor),
                  fix_cosine=bool(fix_cosine), noise=str(noise))
    if noise not in ("sem", "pooled"):
        return SignalMapping(False, degree, reason="noise must be 'sem' or 'pooled'", **common)
    if degree < 2:
        return SignalMapping(False, degree, reason="degree must be >= 2 (degree 1 is the plain cosine)", **common)
    datasets = [_prepare(a, b) for a, b in zip(ts, ys)]
    if not any(np.any(d.n > 1) for d in datasets):
        return SignalMapping(False, degree, reason="needs repeated pulse lengths (the fit is to repeat means)",
                             **common)
    raw = fit_rabi(ts, ys, model=model, noise=noise, sem_floor=sem_floor, run_ids=run_ids,
                   xvarname=xvarname, signal_name=signal_name, params=params)
    if not raw.ok:
        return SignalMapping(False, degree, reason=f"plain Rabi fit failed: {raw.reason}", fit_raw=raw, **common)
    T, Y, S, K, _ = _fit_arrays(datasets, noise, sem_floor)
    n_sets = len(datasets)
    has_g = model != "none"
    n_cos = 0 if fix_cosine else (3 if has_g else 2)
    nd = degree + 1

    # plain model, run j: B_j + A_j/2 v  ->  d0 = B_0, d1 = A_0/2, a_j = A_j/A_0, c_j = (B_j - B_0)/d1
    B, A = np.array(raw.offsets), np.array(raw.amplitudes)
    d_start = np.zeros(nd)
    d_start[0], d_start[1] = B[0], A[0] / 2
    ca_start = np.ravel([[(B[j] - B[0]) / d_start[1], A[j] / A[0]] for j in range(1, n_sets)])

    def split(p):
        if fix_cosine:
            om, ph, g = raw.omega, raw.phase, raw.gamma
        else:
            om, ph = p[0], p[1]
            g = p[2] if has_g else 0.0
        d = p[n_cos:n_cos + nd]
        ca = p[n_cos + nd:]
        c = np.concatenate([[0.0], ca[0::2]])
        a = np.concatenate([[1.0], ca[1::2]])
        return om, ph, g, d, c, a

    def resid(p):
        om, ph, g, d, c, a = split(p)
        return (Y - _P.polyval(c[K] + a[K] * _flop_v(T, om, ph, g, model), d)) / S

    cos0 = [] if fix_cosine else [raw.omega, raw.phase] + ([max(raw.gamma, 1e-3 * raw.omega)] if has_g else [])
    p0 = np.concatenate([cos0, d_start, ca_start])
    lo_b, hi_b = np.full(p0.size, -np.inf), np.full(p0.size, np.inf)
    if not fix_cosine:
        lo_b[0] = 0.0
        if has_g:
            lo_b[2] = 0.0
    try:
        r = least_squares(resid, p0, bounds=(lo_b, hi_b), x_scale="jac", max_nfev=40000)
    except Exception as e:
        return SignalMapping(False, degree, reason=f"fit raised {e!r}", fit_raw=raw, **common)

    om, ph, g, d, c, a = split(r.x)
    chi2 = float(np.sum(resid(r.x) ** 2))
    g_at_zero = has_g and not fix_cosine and g <= 1e-9 * om
    raw_g_zero = has_g and raw.gamma <= 1e-9 * raw.omega
    k_cos_raw = 2 + (1 if has_g and not raw_g_zero else 0)
    kpar = r.x.size - (1 if g_at_zero else 0) + (k_cos_raw if fix_cosine else 0)
    dof = max(Y.size - kpar, 1)
    J = r.jac
    keep = [i for i in range(J.shape[1]) if not (g_at_zero and i == 2)]
    Jk = J[:, keep]
    try:
        ck = np.linalg.inv(Jk.T @ Jk)
    except np.linalg.LinAlgError:
        ck = np.linalg.pinv(Jk.T @ Jk)
    cov = np.full((J.shape[1], J.shape[1]), np.nan)
    cov[np.ix_(keep, keep)] = ck
    if chi2 / dof > 1:
        cov = cov * chi2 / dof
    warn = []
    d, c = np.array(d, float), np.array(c, float)
    d_err = np.sqrt(np.clip(np.diag(cov)[n_cos:n_cos + nd], 0, None))
    if d[1] < 0:                         # signal falls as v rises: flip v -> -v so D is increasing
        if not fix_cosine:
            ph = ph + np.pi
            d = d * (-1.0) ** np.arange(nd)
            c = -c
        else:
            warn.append("D decreases with the plain-fit cosine coordinate: check the sign of the plain fit")
    ph = _wrap(ph)
    ph_eff = ph if math.cos(ph) >= 0 else _wrap(ph - np.pi)
    t_pi = (np.pi - ph_eff) / om
    if fix_cosine:
        t_pi_err = raw.t_pi_err
        warn.append("fix_cosine: omega/phase held at the plain fit, so the pi time is the plain one by construction")
    else:
        C2 = cov[:2, :2]
        grad = np.array([-(np.pi - ph_eff) / om ** 2, -1 / om])
        t_pi_err = float(np.sqrt(max(grad @ C2 @ grad, 0)))
    if not fix_cosine and abs(math.cos(ph)) < 0.3:
        warn.append(f"fitted phase {ph:+.2f} rad is near +/-pi/2: which extremum is the pi pulse is ambiguous")
    tt = np.linspace(0, T.max(), 2001)
    args = [c[j] + a[j] * _flop_v(tt, om, ph, g, model) for j in range(n_sets)]
    v_lo, v_hi = float(min(x.min() for x in args)), float(max(x.max() for x in args))
    slope = _P.polyval(np.linspace(v_lo, v_hi, 501), _P.polyder(d)) / d[1]
    monotonic = bool(np.all(slope > 0))
    if not monotonic:
        warn.append("D is not monotonic over the data: the signal does not identify the spin state uniquely "
                    "there, so no remap is available (the chi2 comparison still stands)")
    if not r.success:
        warn.append(f"optimizer: {r.message}")
    kraw = raw.popt.size - (1 if raw_g_zero else 0)
    fields = dict(
        ok=True, degree=degree, coeffs=d, coeffs_err=d_err, model=model, omega=float(om), phase=float(ph),
        gamma=float(g), centers=tuple(map(float, c)), scales=tuple(map(float, a)), v_range=(v_lo, v_hi),
        monotonic=monotonic, min_slope=float(slope.min()),
        chi2_raw=float(raw.chi2), chi2_mapped=chi2, dof_raw=int(raw.dof), dof_mapped=int(dof),
        aicc_raw=float(_aicc(raw.chi2, Y.size, kraw, True)), aicc_mapped=float(_aicc(chi2, Y.size, kpar, True)),
        t_pi=float(t_pi), t_pi_err=float(t_pi_err), t_pi_raw=raw.t_pi, t_pi_raw_err=raw.t_pi_err,
        warnings=tuple(warn), fit_raw=raw, **common)
    if monotonic:
        mapping = SignalMapping(**fields)
        mapped = [mapping(dd.y) for dd in datasets]
        fields["n_shots_unmappable"] = int(sum(np.sum(~np.isfinite(m)) for m in mapped))
        # plain Rabi fit of the remapped SHOTS (D held fixed); unmappable shots are NaN and
        # therefore left out -- counted above and printed in the summary
        fields["fit_mapped"] = fit_rabi([dd.t for dd in datasets], mapped, model=model, noise=noise,
                                        sem_floor=sem_floor, run_ids=run_ids, xvarname=xvarname,
                                        signal_name=f"g({signal_name})", params=params)
    return SignalMapping(**fields)


def linearize(source, signal="atom_number", xvar=None, *, degree=3, model="none", fix_cosine=False,
              noise="sem", roi_id="auto", sem_floor=DEFAULT_SEM_FLOOR) -> SignalMapping:
    """:func:`linearize_signal` straight from a run id, an atomdata, or a list of them."""
    from .rabi_fit import _load, _signal, _xvar
    items = list(source) if isinstance(source, (list, tuple)) else [source]
    ads = [_load(s, roi_id) for s in items]
    ts, ys, rids, name, sname = [], [], [], None, None
    for a in ads:
        tt, name = _xvar(a, xvar)
        yy, sname = _signal(a, signal)
        ts.append(tt); ys.append(yy)
        rid = getattr(getattr(a, "run_info", None), "run_id", None)
        if rid is not None:
            rids.append(int(np.ravel(rid)[0]))
    return linearize_signal(ts, ys, degree=degree, model=model, fix_cosine=fix_cosine, noise=noise,
                            sem_floor=sem_floor, run_ids=tuple(rids),
                            signal_name=sname, xvarname=name, params=getattr(ads[0], "params", None))
