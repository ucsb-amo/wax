"""Rabi oscillation fits: pi time from a pulse-length scan, without railing on bounds.

The model is ``offset + amp/2 * env(t) * cos(omega t + phi)`` with ``env`` one of
``"none"`` (1), ``"exp"`` (``exp(-gamma t)``) or ``"gauss"`` (``exp(-(gamma t)^2)``).

What it does differently from :func:`waxa.plotting.rabi_oscillation`:

* **Starting point from a frequency scan, not from peak finding.**  At each trial
  ``omega`` on a dense grid the offset, cosine and sine amplitudes are an exact
  linear least-squares problem, so the residual is known for every ``omega``.  The
  deepest few local minima are each refined with the full nonlinear model and the
  best one is kept.  Nothing depends on smoothing or on seeing clean peaks.
* **Parameterised so the answer is not a bound.**  The decay is a *rate*
  ``gamma >= 0``: ``gamma = 0`` is "no decay resolved", a result rather than a
  failure, and the other errors are then taken from the fit without it.  The
  amplitude's sign is folded into the phase instead of being bounded, and neither the
  frequency nor the offset has a bound.
* **Two pi times, stated separately.**  ``t_pi_rate = pi/omega`` is the rate.  The
  commanded pulse length that performs a pi rotation is ``t_pi = t_dead + pi/omega``,
  where ``t_dead = -phi_eff/omega`` is the fitted dead time (commanded minus
  effective length).  ``t_pi`` is the number to put in a config.  Both carry errors
  from the covariance, including the omega-phi correlation.
* **Either initial state.**  If the imaged state starts full (``cos(phi) >= 0``) the
  pi pulse is its first minimum; if it starts empty, its first maximum.
* **Repeats grouped by pulse length**, not by array reshaping, so shuffled scans and
  ragged repeats work.  Per-point SEMs weight the fit, with a floor at
  ``sem_floor * median(SEM)`` so a point whose few repeats happen to agree cannot
  dominate.  ``noise="pooled"`` instead fits one noise model to every point's
  repeat scatter, ``sigma^2(y) = s0^2 + (f (y - y0))^2`` (a floor plus a fluctuation
  of the signal above a stable offset), and weights each mean by
  ``sigma(mean)/sqrt(n)``: with few
  repeats a per-point SEM has so few degrees of freedom that a point whose repeats
  agree by chance takes over the fit.  Without repeats the fit is unweighted and the
  noise is estimated from the residuals.
* **Honest errors.**  The covariance is scaled by the reduced chi2 when that exceeds
  1 (``errors_scaled``), and ``bootstrap=N`` resamples shots within each pulse length.
* **Model choice by AICc** when ``model="auto"``, with every candidate's score kept.
* **Joint fits.**  Several datasets (e.g. runs) share omega, phi and gamma, while each
  gets its own offset and amplitude, so atom-number drift between runs is not read
  as a Rabi effect.
* **No side effects.**  Nothing is printed or plotted unless asked, and failures come
  back as ``ok=False`` with a reason instead of raising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

MODELS = ("none", "exp", "gauss")
WEIGHTED = ("sem", "pooled")
DEFAULT_SEM_FLOOR = 0.5


# --------------------------------------------------------------------------- model
def envelope(t, gamma, model):
    t = np.asarray(t, dtype=float)
    if model == "none":
        return np.ones_like(t)
    if model == "exp":
        return np.exp(-gamma * t)
    if model == "gauss":
        return np.exp(-(gamma * t) ** 2)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


def rabi_model(t, omega, phi, offset, amp, gamma=0.0, model="exp"):
    """``offset + amp/2 * env(t) * cos(omega t + phi)``; ``amp`` is peak-to-peak at t = 0."""
    return offset + 0.5 * amp * envelope(t, gamma, model) * np.cos(omega * np.asarray(t, float) + phi)


def _wrap(p):
    return (p + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ data handling
@dataclass
class _Dataset:
    t: np.ndarray            # every shot
    y: np.ndarray
    tu: np.ndarray           # unique pulse lengths
    mean: np.ndarray
    sem: np.ndarray          # NaN where a point has one shot
    n: np.ndarray


def _prepare(t, y):
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.shape != y.shape:
        raise ValueError(f"t {t.shape} and y {y.shape} must match")
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    tu, inv = np.unique(t, return_inverse=True)
    n = np.bincount(inv, minlength=tu.size)
    mean = np.bincount(inv, weights=y, minlength=tu.size) / n
    sq = np.bincount(inv, weights=(y - mean[inv]) ** 2, minlength=tu.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = np.where(n > 1, np.sqrt(sq / np.maximum(n - 1, 1)) / np.sqrt(n), np.nan)
    return _Dataset(t, y, tu, mean, sem, n)


def pooled_noise(datasets):
    """(s0, f, y0) of ``sigma^2(y) = s0^2 + (f (y - y0))^2`` per shot: a floor plus a
    fluctuation proportional to the signal above a stable offset ``y0`` (e.g. atoms or
    background that do not take part in the flop).  Maximum likelihood on the sample
    variances of every point with repeats, each a chi2 with n - 1 dof."""
    from scipy.optimize import minimize
    rows = [(d.mean[i], d.sem[i] ** 2 * d.n[i], d.n[i] - 1) for d in datasets for i in range(d.tu.size)
            if d.n[i] > 1 and np.isfinite(d.sem[i])]
    if len(rows) < 4:
        raise ValueError("noise='pooled' needs at least 4 points with repeats")
    m, s2, k = (np.array(c, float) for c in zip(*rows))
    scale = float(np.sqrt(np.median(s2))) or 1.0
    span = float(np.ptp(m)) or scale

    def nll(p):
        var = np.exp(2 * p[0]) + np.exp(2 * p[1]) * (m - p[2] * span) ** 2
        return float(np.sum(k * (s2 / var + np.log(var))))
    best = None
    for f0 in (0.03, 0.3):
        for y00 in (0.0, float(m.min()) / span):
            r = minimize(nll, [math.log(0.5 * scale), math.log(f0), y00], method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-9, maxiter=8000))
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(best.x[2] * span)


def _fit_arrays(datasets, noise, sem_floor):
    """(t, y, sigma, dataset index) actually fed to the fit, and the resolved noise mode."""
    has_repeats = any(np.any(d.n > 1) for d in datasets)
    if noise == "auto":
        noise = "sem" if has_repeats else "unweighted"
    T, Y, S, K = [], [], [], []
    if noise == "sem":
        if not has_repeats:
            raise ValueError("noise='sem' needs repeated pulse lengths")
        # per-shot scatter pooled over every dataset with repeats: it gives single-shot
        # points (and whole datasets without repeats, in a joint fit) their sigma
        pooled = np.concatenate([d.sem[np.isfinite(d.sem)] * np.sqrt(d.n[np.isfinite(d.sem)]) for d in datasets])
        per_shot_all = float(np.median(pooled))
        sems = []
        for d in datasets:
            ok = np.isfinite(d.sem)
            per_shot = float(np.median(d.sem[ok] * np.sqrt(d.n[ok]))) if ok.any() else per_shot_all
            sems.append(np.where(ok, d.sem, per_shot / np.sqrt(d.n)))
        floor = sem_floor * float(np.median(np.concatenate(sems)))
        for k, (d, sem) in enumerate(zip(datasets, sems)):
            T.append(d.tu); Y.append(d.mean); S.append(np.maximum(sem, floor)); K.append(np.full(d.tu.size, k))
    elif noise == "pooled":
        if not has_repeats:
            raise ValueError("noise='pooled' needs repeated pulse lengths")
        s0, f, y0 = pooled_noise(datasets)
        for k, d in enumerate(datasets):
            sig = np.sqrt(s0 ** 2 + (f * (d.mean - y0)) ** 2) / np.sqrt(d.n)
            T.append(d.tu); Y.append(d.mean); S.append(sig); K.append(np.full(d.tu.size, k))
    elif noise == "unweighted":
        for k, d in enumerate(datasets):
            T.append(d.t); Y.append(d.y); S.append(np.ones(d.t.size)); K.append(np.full(d.t.size, k))
    else:
        raise ValueError("noise must be 'auto', 'sem', 'pooled' or 'unweighted'")
    return np.concatenate(T), np.concatenate(Y), np.concatenate(S), np.concatenate(K).astype(int), noise


# ------------------------------------------------------------- starting points
def _frequency_scan(t, y, s, k, n_sets, n_grid=4000):
    """Weighted residual of the best linear (offset, cos, sin) fit per dataset vs omega."""
    tu = np.unique(t)
    span = tu.max() - tu.min()
    dt = np.min(np.diff(tu)) if tu.size > 1 else span
    omega_min = 2 * np.pi * 0.3 / span
    omega_max = 0.98 * np.pi / dt
    omegas = np.linspace(omega_min, omega_max, n_grid)
    w = 1.0 / s
    rss = np.empty(omegas.size)
    onehot = (k[:, None] == np.arange(n_sets)[None, :]).astype(float)
    for i, om in enumerate(omegas):
        c, sn = np.cos(om * t), np.sin(om * t)
        M = np.hstack([onehot, onehot * c[:, None], onehot * sn[:, None]]) * w[:, None]
        coef, *_ = np.linalg.lstsq(M, y * w, rcond=None)
        rss[i] = np.sum((y * w - M @ coef) ** 2)
    return omegas, rss, omega_min, np.pi / dt


def _candidates(omegas, rss, n_cand=4):
    i = np.arange(1, rss.size - 1)
    loc = i[(rss[i] <= rss[i - 1]) & (rss[i] <= rss[i + 1])]
    if loc.size == 0:
        loc = np.array([int(np.argmin(rss))])
    return omegas[loc[np.argsort(rss[loc])][:n_cand]]


def _linear_start(t, y, s, k, n_sets, omega):
    """(phi, [offset_k], [amp_k]) at fixed omega, the phase shared (amplitude-weighted)."""
    offs, amps, zs = [], [], []
    for j in range(n_sets):
        m = k == j
        M = np.column_stack([np.ones(m.sum()), np.cos(omega * t[m]), np.sin(omega * t[m])]) / s[m, None]
        (c, a, b), *_ = np.linalg.lstsq(M, y[m] / s[m], rcond=None)
        offs.append(c)
        amps.append(2 * np.hypot(a, b))
        zs.append(a - 1j * b)                       # a cos + b sin = |z| cos(omega t + arg z)
    phi = float(np.angle(np.sum(zs)))
    return phi, offs, amps


# ------------------------------------------------------------------ one model
def _fit_model(t, y, s, k, n_sets, model, omega_cands, span):
    from scipy.optimize import least_squares
    has_g = model != "none"

    def unpack(p):
        om, ph = p[0], p[1]
        g = p[2] if has_g else 0.0
        base = 3 if has_g else 2
        off = p[base::2][:n_sets]
        amp = p[base + 1::2][:n_sets]
        return om, ph, g, off, amp

    def resid(p):
        om, ph, g, off, amp = unpack(p)
        return (y - (off[k] + 0.5 * amp[k] * envelope(t, g, model) * np.cos(om * t + ph))) / s

    best = None
    for om0 in omega_cands:
        ph0, off0, amp0 = _linear_start(t, y, s, k, n_sets, om0)
        starts = [0.5 / span, 2.0 / span] if has_g else [None]
        for g0 in starts:
            p0 = [om0, ph0] + ([g0] if has_g else []) + [v for pair in zip(off0, amp0) for v in pair]
            lo = np.full(len(p0), -np.inf)
            hi = np.full(len(p0), np.inf)
            lo[0] = 0.0                                   # omega > 0 is physics, not a prior
            if has_g:
                lo[2] = 0.0                               # a decay rate cannot be negative
            try:
                r = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", method="trf", max_nfev=20000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    return best, unpack


def _covariance(r, drop=()):
    J = r.jac
    keep = [i for i in range(J.shape[1]) if i not in drop]
    Jk = J[:, keep]
    try:
        ck = np.linalg.inv(Jk.T @ Jk)
    except np.linalg.LinAlgError:
        ck = np.linalg.pinv(Jk.T @ Jk)
    cov = np.full((J.shape[1], J.shape[1]), np.nan)
    cov[np.ix_(keep, keep)] = ck
    return cov


def _aicc(chi2, n, kpar, weighted):
    base = chi2 if weighted else n * math.log(max(chi2, 1e-300) / n)
    corr = 2 * kpar * (kpar + 1) / (n - kpar - 1) if n - kpar - 1 > 0 else np.inf
    return base + 2 * kpar + corr


# ------------------------------------------------------------------ the result
@dataclass(frozen=True, eq=False)
class RabiFit:
    """Everything the fit found.  Times in s, rates in rad/s unless named ``f_``."""
    ok: bool
    model: str
    noise: str
    omega: float = np.nan
    omega_err: float = np.nan
    phase: float = np.nan              # fitted phi, wrapped to (-pi, pi]
    phase_err: float = np.nan
    gamma: float = np.nan              # decay rate (1/s); 0 = no decay resolved
    gamma_err: float = np.nan
    offsets: tuple = ()
    amplitudes: tuple = ()             # peak-to-peak at t = 0, one per dataset
    starts_populated: bool = True      # imaged state full at t = 0
    t_dead: float = np.nan             # commanded - effective pulse length
    t_dead_err: float = np.nan
    t_pi: float = np.nan               # commanded length of a pi pulse (the config number)
    t_pi_err: float = np.nan
    t_half_pi: float = np.nan
    t_half_pi_err: float = np.nan
    t_pi_rate: float = np.nan          # pi / omega
    t_pi_rate_err: float = np.nan
    chi2: float = np.nan
    dof: int = 0
    chi2r: float = np.nan
    errors_scaled: bool = False
    aicc: dict = field(default_factory=dict)
    warnings: tuple = ()
    reason: str = ""
    n_shots: int = 0
    n_points: int = 0
    n_periods: float = np.nan
    bootstrap_t_pi: Optional[tuple] = None    # (median, 16th, 84th percentile)
    noise_model: Optional[tuple] = None       # (s0, f, y0) when noise == 'pooled'
    popt: np.ndarray = field(default_factory=lambda: np.array([]))
    cov: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    datasets: tuple = ()
    labels: tuple = ()
    run_ids: tuple = ()
    xvarname: str = "t_pulse"
    signal_name: str = "signal"
    params: object = None

    # -------------------------------------------------------------- derived
    @property
    def f_rabi(self):
        return self.omega / (2 * np.pi)

    @property
    def f_rabi_err(self):
        return self.omega_err / (2 * np.pi)

    @property
    def tau(self):
        return 1 / self.gamma if self.gamma > 0 else np.inf

    @property
    def contrast(self):
        return tuple(a / (2 * o) if o else np.nan for a, o in zip(self.amplitudes, self.offsets))

    def curve(self, t, dataset=0):
        t = np.asarray(t, dtype=float)
        return rabi_model(t, self.omega, self.phase, self.offsets[dataset], self.amplitudes[dataset],
                          self.gamma if self.model != "none" else 0.0, self.model)

    # -------------------------------------------------------------- reporting
    def compare(self, value, name="t_raman_pi_pulse"):
        """How a pi time in use compares with the fit: (difference s, in sigma, text)."""
        d = float(value) - self.t_pi
        sig = d / self.t_pi_err if self.t_pi_err > 0 else np.nan
        return d, sig, (f"{name} = {value*1e6:.4f} us vs fitted {self.t_pi*1e6:.3f} +/- {self.t_pi_err*1e6:.3f} us: "
                        f"{d*1e6:+.3f} us ({100*d/self.t_pi:+.1f}%, {sig:+.1f} sigma); a pulse of value/2 is a "
                        f"{(self.omega*(value/2 - self.t_dead))/np.pi:.3f} pi rotation")

    def config_line(self, param="t_raman_pi_pulse", date=None):
        """The line to paste into a params file, lab convention: value, then '#run_id, date'."""
        import datetime
        date = date or datetime.date.today().isoformat()
        rid = ", ".join(str(r) for r in self.run_ids) or "?"
        return f"self.{param} = {self.t_pi:.4e} #{rid}, {date}"

    def summary(self):
        if not self.ok:
            return f"Rabi fit FAILED ({self.reason})"
        u = 1e6
        runs = f"run {', '.join(map(str, self.run_ids))}" if self.run_ids else "data"
        lines = [
            f"{runs}: {self.signal_name} vs {self.xvarname}, {self.n_shots} shots / {self.n_points} points, "
            f"model '{self.model}', noise '{self.noise}'"
            + (f" (per shot sigma^2 = {self.noise_model[0]:.3g}^2 + ({self.noise_model[1]:.3f} (y - "
               f"{self.noise_model[2]:.4g}))^2)"
               if self.noise_model else ""),
            f"  Rabi frequency  {self.f_rabi/1e3:.3f} +/- {self.f_rabi_err/1e3:.3f} kHz   ({self.n_periods:.1f} periods scanned)",
            f"  pi pulse        {self.t_pi*u:.3f} +/- {self.t_pi_err*u:.3f} us   (commanded length; config number)",
            f"  pi/2 pulse      {self.t_half_pi*u:.3f} +/- {self.t_half_pi_err*u:.3f} us",
            f"  pi / Omega      {self.t_pi_rate*u:.3f} +/- {self.t_pi_rate_err*u:.3f} us   (rate only)",
            f"  dead time       {self.t_dead*u*1e3:+.0f} +/- {self.t_dead_err*u*1e3:.0f} ns   "
            f"(imaged state starts {'full' if self.starts_populated else 'empty'})",
            ("  decay           " + ("not resolved (gamma = 0)" if self.model == "none" or self.gamma == 0 else
                                     f"tau = {self.tau*u:.1f} us ({self.model}), gamma {self.gamma:.3g} +/- {self.gamma_err:.2g} /s")),
            f"  contrast        " + ", ".join(f"{c:.3f}" for c in self.contrast),
            f"  chi2/dof        {self.chi2r:.2f} ({self.dof} dof){'  -> errors scaled by sqrt(chi2r)' if self.errors_scaled else ''}",
            f"  AICc            " + ", ".join(f"{m} {v - min(self.aicc.values()):+.1f}" for m, v in self.aicc.items()),
        ]
        if self.bootstrap_t_pi is not None:
            m, lo, hi = self.bootstrap_t_pi
            lines.append(f"  bootstrap pi    {m*u:.3f} us (16-84%: {lo*u:.3f} - {hi*u:.3f})")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def _repr_pretty_(self, p, cycle):
        p.text(self.summary())

    def to_dict(self):
        skip = ("datasets", "params", "popt", "cov")
        out = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in skip}
        out.update(f_rabi=self.f_rabi, f_rabi_err=self.f_rabi_err, tau=self.tau, contrast=self.contrast)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}

    def plot(self, ax=None, residuals=True, show_shots=True, reference=None, **kw):
        from .plotting import plot_rabi
        return plot_rabi(self, ax=ax, residuals=residuals, show_shots=show_shots, reference=reference, **kw)


# ------------------------------------------------------------------- entry
def fit_rabi(t, y, *, model="auto", noise="auto", sem_floor=DEFAULT_SEM_FLOOR, bootstrap=0, seed=0,
             labels=(), run_ids=(), xvarname="t_pulse", signal_name="signal", params=None) -> RabiFit:
    """Fit a Rabi oscillation.

    Parameters
    ----------
    t, y : arrays, or lists of arrays for a joint fit
        Pulse lengths (s) and signal per shot.  Several datasets share omega, phi and
        gamma; each has its own offset and amplitude.  Order and repeats are free.
    model : 'auto' | 'none' | 'exp' | 'gauss'
        Decay envelope; 'auto' picks the lowest AICc.
    noise : 'auto' | 'sem' | 'pooled' | 'unweighted'
        'sem' fits per-point means weighted by their SEM (needs repeats); 'pooled'
        weights the means by one noise model fitted to all the repeat scatter,
        ``sigma^2(y) = s0^2 + (f (y - y0))^2`` (better with few repeats); 'unweighted' fits
        every shot and takes the noise from the residuals.  'auto' picks 'sem' when
        there are repeats.
    sem_floor : float
        SEMs are floored at ``sem_floor * median(SEM)``.
    bootstrap : int
        Resamples (shots within each pulse length) for a percentile interval on t_pi.
    """
    ts = t if isinstance(t, (list, tuple)) else [t]
    ys = y if isinstance(y, (list, tuple)) else [y]
    if len(ts) != len(ys):
        raise ValueError("t and y must hold the same number of datasets")
    common = dict(labels=tuple(labels), run_ids=tuple(run_ids), xvarname=xvarname,
                  signal_name=signal_name, params=params)
    try:
        datasets = [_prepare(a, b) for a, b in zip(ts, ys)]
    except ValueError as e:
        return RabiFit(False, str(model), str(noise), reason=str(e), **common)
    n_sets = len(datasets)
    if any(d.tu.size < 5 for d in datasets):
        return RabiFit(False, str(model), str(noise), reason="fewer than 5 distinct pulse lengths",
                       datasets=tuple(datasets), **common)
    try:
        T, Y, S, K, noise = _fit_arrays(datasets, noise, sem_floor)
        if noise == "pooled":
            common["noise_model"] = pooled_noise(datasets)
        if not np.all(np.isfinite(S)) or np.any(S <= 0):
            raise ValueError("could not assign a finite, positive sigma to every point")
        span = np.ptp(np.concatenate([d.tu for d in datasets]))
        omegas, rss, omega_min, omega_nyq = _frequency_scan(T, Y, S, K, n_sets)
    except (ValueError, np.linalg.LinAlgError) as e:
        return RabiFit(False, str(model), str(noise), reason=str(e), datasets=tuple(datasets), **common)
    cands = _candidates(omegas, rss)
    models = MODELS if model == "auto" else (model,)
    fits = {}
    for m in models:
        r, unpack = _fit_model(T, Y, S, K, n_sets, m, cands, span)
        if r is not None and np.isfinite(r.cost):
            kpar = r.x.size
            fits[m] = (r, unpack, _aicc(2 * r.cost, Y.size, kpar, noise in WEIGHTED))
    if not fits:
        return RabiFit(False, str(model), noise, reason="no fit converged", datasets=tuple(datasets), **common)
    aicc = {m: v[2] for m, v in fits.items()}
    chosen = min(aicc, key=aicc.get) if model == "auto" else model
    r, unpack, _ = fits[chosen]
    res = _finish(r, unpack, chosen, noise, T, Y, S, K, n_sets, span, omega_nyq, aicc, datasets, common)
    if bootstrap and res.ok:
        res = _with_bootstrap(res, datasets, chosen, noise, sem_floor, bootstrap, seed, common)
    return res


def _finish(r, unpack, model, noise, T, Y, S, K, n_sets, span, omega_nyq, aicc, datasets, common):
    om, ph, g, off, amp = unpack(r.x)
    off, amp = np.array(off, float), np.array(amp, float)
    has_g = model != "none"
    warn = []
    g_at_zero = has_g and g <= 1e-9 * om
    drop = (2,) if g_at_zero else ()
    cov = _covariance(r, drop)
    n, kpar = Y.size, r.x.size - len(drop)
    dof = max(n - kpar, 1)
    chi2 = 2 * r.cost
    chi2r = chi2 / dof
    scaled = False
    if noise == "unweighted" or chi2r > 1:
        cov = cov * chi2r
        scaled = noise in WEIGHTED
    # fold negative amplitudes into the phase (one shared phase: fold only if all agree)
    sign = np.sign(amp)
    sign[sign == 0] = 1
    if np.all(sign < 0):
        ph, amp = ph + np.pi, -amp
        idx = [3 + 2 * j + (0 if has_g else -1) + 1 for j in range(n_sets)]
        for i in idx:
            cov[i, :] *= -1
            cov[:, i] *= -1
    elif np.any(sign < 0):
        warn.append("datasets disagree on the fringe sign -- check the signals")
    ph = _wrap(ph)
    e = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    om_e, ph_e = e[0], e[1]
    g_e = e[2] if has_g and not g_at_zero else np.nan
    starts_full = math.cos(ph) >= 0
    if abs(math.cos(ph)) < 0.3:
        warn.append(f"fitted phase {ph:+.2f} rad is near +/-pi/2: whether the imaged state starts full or "
                    "empty is ambiguous, and with it which extremum is the pi pulse")
    ph_eff = ph if starts_full else _wrap(ph - np.pi)
    C2 = cov[:2, :2]

    def t_of(theta):
        val = (theta - ph_eff) / om
        grad = np.array([-(theta - ph_eff) / om ** 2, -1 / om])
        return val, float(np.sqrt(max(grad @ C2 @ grad, 0)))

    t_pi, t_pi_e = t_of(np.pi)
    t_h, t_h_e = t_of(np.pi / 2)
    t_d, t_d_e = t_of(0.0)
    n_periods = span * om / (2 * np.pi)
    if n_periods < 1:
        warn.append(f"only {n_periods:.2f} Rabi periods in the scan -- the frequency is weakly constrained")
    if om > omega_nyq:
        warn.append("fitted frequency is above the sampling Nyquist limit -- possible alias")
    if noise in WEIGHTED and chi2r > 3:
        warn.append(f"reduced chi2 = {chi2r:.1f}: scatter well beyond the SEMs (errors scaled; check drifts/outliers)")
    if not r.success:
        warn.append(f"optimizer: {r.message}")
    if t_pi <= 0:
        warn.append("fitted pi time is not positive -- dead time exceeds half a period")
    return RabiFit(
        ok=bool(np.isfinite(om) and np.isfinite(t_pi)), model=model, noise=noise,
        omega=float(om), omega_err=float(om_e), phase=float(ph), phase_err=float(ph_e),
        gamma=float(g if has_g else 0.0), gamma_err=float(g_e),
        offsets=tuple(map(float, off)), amplitudes=tuple(map(float, amp)), starts_populated=bool(starts_full),
        t_dead=float(t_d), t_dead_err=t_d_e, t_pi=float(t_pi), t_pi_err=t_pi_e,
        t_half_pi=float(t_h), t_half_pi_err=t_h_e, t_pi_rate=float(np.pi / om),
        t_pi_rate_err=float(np.pi / om * om_e / om), chi2=float(chi2), dof=int(dof), chi2r=float(chi2r),
        errors_scaled=bool(scaled), aicc=dict(aicc), warnings=tuple(warn),
        n_shots=int(sum(d.t.size for d in datasets)), n_points=int(Y.size), n_periods=float(n_periods),
        popt=np.array(r.x), cov=cov, datasets=tuple(datasets), **common)


def _with_bootstrap(res, datasets, model, noise, sem_floor, n_boot, seed, common):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        ts, ys = [], []
        for d in datasets:
            tt, yy = [], []
            for v in d.tu:
                yv = d.y[d.t == v]
                pick = rng.integers(0, yv.size, yv.size) if yv.size > 1 else np.zeros(1, int)
                tt.append(np.full(yv.size, v)); yy.append(yv[pick])
            if all(x.size == 1 for x in yy):                 # no repeats: residual bootstrap
                model_y = res.curve(d.tu, len(ts))
                resid = d.y - res.curve(d.t, len(ts))
                yy = [np.array([m + rng.choice(resid)]) for m in model_y]
            ts.append(np.concatenate(tt)); ys.append(np.concatenate(yy))
        b = fit_rabi(ts, ys, model=model, noise=noise, sem_floor=sem_floor)
        if b.ok:
            vals.append(b.t_pi)
    if len(vals) < max(10, n_boot // 4):
        return res
    q = tuple(float(x) for x in np.percentile(vals, [50, 16, 84]))
    return RabiFit(**{**{f: getattr(res, f) for f in res.__dataclass_fields__}, "bootstrap_t_pi": q})


# ----------------------------------------------------------- atomdata adapter
def _signal(ad, signal):
    if not isinstance(signal, str):
        return np.asarray(signal, dtype=float), "signal"
    if signal == "sumod_contrast":                 # the legacy rabi_oscillation population
        return np.array([np.max(s) - np.min(s) for s in np.asarray(ad.sum_od_x)]), signal
    obj = ad
    for part in signal.split("."):
        obj = getattr(obj, part)
    return np.asarray(obj, dtype=float), signal


def _xvar(ad, xvar):
    names = [str(n) for n in ad.xvarnames]
    if xvar is None:
        if len(names) == 1:
            i = 0
        else:
            hits = [j for j, n in enumerate(names) if "pulse" in n or "rabi" in n]
            if len(hits) != 1:
                raise ValueError(f"cannot pick the pulse-length xvar from {names}; pass xvar=")
            i = hits[0]
    else:
        i = names.index(xvar) if isinstance(xvar, str) else int(xvar)
    if len(names) > 1:
        raise ValueError(f"multi-xvar run ({names}): slice it to the pulse-length axis first")
    return np.asarray(ad.xvars[i], dtype=float), names[i]


def _load(src, roi_id):
    if isinstance(src, (int, np.integer)):
        from waxa import atomdata
        return atomdata(int(src), roi_id=roi_id, lite=False)
    return src


def rabi(source, signal="atom_number", xvar=None, *, model="auto", noise="auto", joint=True,
         roi_id="auto", sem_floor=DEFAULT_SEM_FLOOR, bootstrap=0):
    """Fit a Rabi scan straight from data.

    ``source`` is an atomdata (or anything with ``xvarnames``, ``xvars`` and the
    signal), a run id (loaded read-only with ``roi_id``, ``lite=False``), or a list of
    either.  A list is fit jointly (shared omega, phi, gamma) unless ``joint=False``,
    which returns one fit per item.
    """
    items = list(source) if isinstance(source, (list, tuple)) else [source]
    ads = [_load(s, roi_id) for s in items]
    if not joint and len(ads) > 1:
        return [rabi(a, signal, xvar, model=model, noise=noise, sem_floor=sem_floor, bootstrap=bootstrap)
                for a in ads]
    ts, ys, rids, name, sname = [], [], [], None, None
    for a in ads:
        tt, name = _xvar(a, xvar)
        yy, sname = _signal(a, signal)
        ts.append(tt); ys.append(yy)
        rid = getattr(getattr(a, "run_info", None), "run_id", None)
        rids.append(int(np.ravel(rid)[0]) if rid is not None else None)
    return fit_rabi(ts, ys, model=model, noise=noise, sem_floor=sem_floor, bootstrap=bootstrap,
                    labels=tuple(f"run {r}" for r in rids), run_ids=tuple(r for r in rids if r is not None),
                    xvarname=name, signal_name=sname, params=getattr(ads[0], "params", None))
