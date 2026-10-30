"""Drive dead time from split pulses: the pulse area lost per switch-on.

Data: shots that each apply ``N`` sub-pulses of commanded length ``tau`` to a spin
prepared in ``S_z = +1`` (``N = 0``: no pulse), then read a signal linear in ``S_z``
(``signal`` shaped (n_shots,) or (n_shots, P) reads, averaged per shot).  The model is

    t_eff = N max(tau - t_dead, 0)                  (the drive time that rotates)
    y     = A + B env(t_eff) cos(omega t_eff)

with ``env`` one of :func:`waxa.analysis.rabi.envelope`'s models (``"none"``, ``"exp"``,
``"gauss"``, a decay rate ``gamma >= 0``), picked by AICc when ``model="auto"``.  Every
shot is fitted by least squares (unweighted, the noise from the residuals, the
covariance scaled by it).

Why split pulses: at a fixed total ``T = N tau``, ``omega t_eff = omega T - N omega t_dead``,
so t_dead is the slope against N and omega the intercept.  In a pulse-length scan
t_dead only shifts the whole flop, which a fit trades against omega.

Why a decay envelope: within one commanded total the decay is nearly the same for
every N, so a decay that is not modelled looks like a smaller contrast for the
longer total, which the fit then reads partly as t_dead.  On synthetic data an
8 us decay pulled t_dead from 127 to 92 +/- 18 ns with no flag from the checks
below; the envelope removes that bias.  The summary lists t_dead under every
model, so the dependence on the choice is visible.

Reported
--------
* t_dead +/- its covariance error, and the profile interval (``delta chi2 <= 1``, the
  other parameters refitted at each t_dead): the two differ when the fit is not
  parabolic, e.g. near ``t_dead = tau`` where a sub-pulse stops rotating at all.
* omega as the rate pi time pi/omega, compared with ``t_pi_ref``.
* Lack of fit: the repeats give a pure-error variance, so the scatter of the
  per-configuration means about the model is tested against it (F test).
* Groups: every group (one commanded total) with two or more N values is also
  fitted on its own, together with the single-N groups (the references).  Their
  t_dead values should agree; they share the reference shots, so the chi2 across
  groups is approximate.  A disagreement points at something the model lacks.

Not modelled: a nonlinear readout (an ``S_z^2`` term), a detuning.  Either would
bias t_dead; the lack-of-fit test and the group comparison are the checks.  Nothing
is left out except shots with a non-finite N, tau or read, which are counted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .rabi_fit import MODELS, envelope

T_DEAD_SEARCH = (-0.3e-6, 1.5e-6)     # start-grid range for t_dead (s)
OMEGA_SEARCH = (0.6, 1.4)             # start-grid range, as a fraction of pi / t_pi_ref
DECAY_TIME_STARTS = (100e-6, 30e-6, 10e-6, 4e-6)   # starts for 1/gamma (s)


def split_pulse_theta(n_split, tau, omega, t_dead):
    """Rotation angle of N sub-pulses of commanded length tau."""
    return omega * effective_time(n_split, tau, t_dead)


def effective_time(n_split, tau, t_dead):
    """Drive time that rotates: N max(tau - t_dead, 0)."""
    return np.asarray(n_split, float) * np.clip(np.asarray(tau, float) - t_dead, 0.0, None)


def split_pulse_model(n_split, tau, A, B, omega, t_dead, gamma=0.0, model="none"):
    t_eff = effective_time(n_split, tau, t_dead)
    return A + B * envelope(t_eff, gamma, model) * np.cos(omega * t_eff)


def _p_model(p, n, tau, model):
    return split_pulse_model(n, tau, p[0], p[1], p[2], p[3], p[4] if model != "none" else 0.0, model)


def _linear_ab(c, y):
    """A, B, rss of ``y = A + B c`` (exact)."""
    dc = c - c.mean()
    var = float(dc @ dc)
    B = float(dc @ y) / var if var > 0 else 0.0
    A = float(y.mean() - B * c.mean())
    r = y - A - B * c
    return A, B, float(r @ r)


def _best_omega(n, tau, y, t_dead, om_lo, om_hi, n_grid=41):
    """omega minimising rss at a fixed t_dead (no decay): a grid, then a bounded refinement."""
    from scipy.optimize import minimize_scalar
    grid = np.linspace(om_lo, om_hi, n_grid)
    rss = [_linear_ab(np.cos(split_pulse_theta(n, tau, om, t_dead)), y)[2] for om in grid]
    k = int(np.argmin(rss))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, n_grid - 1)]
    r = minimize_scalar(lambda om: _linear_ab(np.cos(split_pulse_theta(n, tau, om, t_dead)), y)[2],
                        bounds=(lo, hi), method="bounded", options=dict(xatol=1e-6 * om_hi))
    return float(r.x), float(r.fun)


def _aicc(rss, n, k):
    corr = 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf
    return n * math.log(max(rss, 1e-300) / n) + 2 * k + corr


def _lsq(n, tau, y, model, p0, fixed_td=None):
    """least squares from p0; with fixed_td, t_dead is held (for the profile)."""
    from scipy.optimize import least_squares
    p0 = np.asarray(p0, float)
    free = [i for i in range(p0.size) if not (fixed_td is not None and i == 3)]
    lo = np.full(p0.size, -np.inf)
    lo[2] = 0.0
    if p0.size > 4:
        lo[4] = 0.0

    def resid(q):
        p = p0.copy()
        p[free] = q
        if fixed_td is not None:
            p[3] = fixed_td
        return y - _p_model(p, n, tau, model)

    start = np.maximum(p0[free], lo[free])
    r = least_squares(resid, start, bounds=(lo[free], np.full(len(free), np.inf)), x_scale="jac", max_nfev=20000)
    p = p0.copy()
    p[free] = r.x
    if fixed_td is not None:
        p[3] = fixed_td
    return p, r


def _fit(n, tau, y, omega_ref, model, start=None):
    """Least squares on (A, B, omega, t_dead[, gamma]).  Returns a dict, or None."""
    k = 4 if model == "none" else 5
    dof = y.size - k
    if dof < 1:
        return None
    if start is None:
        om_lo, om_hi = OMEGA_SEARCH[0] * omega_ref, OMEGA_SEARCH[1] * omega_ref
        td_max = min(T_DEAD_SEARCH[1], float(np.max(tau[n > 0])))
        best = None
        for td in np.arange(T_DEAD_SEARCH[0], td_max, 10e-9):
            om, rss = _best_omega(n, tau, y, td, om_lo, om_hi)
            if best is None or rss < best[2]:
                best = (td, om, rss)
        td0, om0, _ = best
        A0, B0, _ = _linear_ab(np.cos(split_pulse_theta(n, tau, om0, td0)), y)
        start = [A0, B0, om0, td0]
    starts = [list(start[:4])] if model == "none" else [list(start[:4]) + [1 / tdec] for tdec in DECAY_TIME_STARTS]
    best = None
    for p0 in starts:
        p, r = _lsq(n, tau, y, model, p0)
        rss = float(np.sum(r.fun ** 2))
        if best is None or rss < best[2]:
            best = (p, r, rss)
    p, r, rss = best
    J = r.jac
    try:
        cov = np.linalg.inv(J.T @ J) * rss / dof
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J) * rss / dof
    return dict(model=model, p=p, A=float(p[0]), B=float(p[1]), omega=float(p[2]), t_dead=float(p[3]),
                gamma=float(p[4]) if k == 5 else 0.0, cov=cov, rss=rss, dof=int(dof), k=k,
                aicc=_aicc(rss, y.size, k), success=bool(r.success))


def _profile(n, tau, y, f, step=2e-9):
    """delta chi2 against t_dead (the other parameters refitted), chi2 scaled by the
    best fit's residual variance.  Returns (grid, delta chi2, (lo, hi) where <= 1)."""
    s2 = f["rss"] / f["dof"]
    se = np.sqrt(max(f["cov"][3, 3], 0.0))
    hw = max(6 * se, 50e-9)
    grid = np.arange(f["t_dead"] - hw, f["t_dead"] + hw + step / 2, step)
    rss = np.empty(grid.size)
    for i, td in enumerate(grid):
        p, r = _lsq(n, tau, y, f["model"], f["p"], fixed_td=td)
        rss[i] = float(np.sum(r.fun ** 2))
    d = (rss - min(rss.min(), f["rss"])) / s2
    inside = np.nonzero(d <= 1.0)[0]
    if inside.size == 0:
        return grid, d, (np.nan, np.nan)
    lo = grid[inside[0]] if inside[0] > 0 else np.nan        # nan: the interval runs off the grid
    hi = grid[inside[-1]] if inside[-1] < grid.size - 1 else np.nan
    return grid, d, (float(lo), float(hi))


def _configs(n, tau, y):
    """Per-configuration (N, tau) means: list of dicts, and the pure-error rss / dof."""
    key = np.round(tau * 1e12).astype(np.int64) * 1000 + n.astype(np.int64)
    out, ss_pe, df_pe = [], 0.0, 0
    for k in np.unique(key):
        m = key == k
        v = y[m]
        out.append(dict(n=int(n[m][0]), tau=float(tau[m][0]), n_shots=int(m.sum()), mean=float(v.mean()),
                        sem=float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan))
        ss_pe += float(np.sum((v - v.mean()) ** 2))
        df_pe += v.size - 1
    return out, ss_pe, df_pe


def _groups_from_totals(n, tau, gap=0.25e-6):
    """Group ids by commanded total N tau (totals within ``gap`` of each other are one
    group); N = 0 is its own group."""
    total = n * tau
    g = np.full(n.size, -1, int)
    g[n == 0] = 0
    idx = np.nonzero(n > 0)[0]
    order = idx[np.argsort(total[idx])]
    gid = 0
    prev = None
    for i in order:
        if prev is None or total[i] - prev > gap:
            gid += 1
        g[i] = gid
        prev = total[i]
    return g


@dataclass(eq=False)
class SplitPulseDeadTime:
    ok: bool
    reason: str = ""
    model: str = ""                        # decay envelope used
    t_dead: float = np.nan                 # s, commanded minus effective length per sub-pulse
    t_dead_err: float = np.nan             # covariance, scaled by the residuals
    t_dead_interval: tuple = (np.nan, np.nan)   # profile delta chi2 <= 1
    omega: float = np.nan                  # rad/s
    omega_err: float = np.nan
    t_pi_rate: float = np.nan              # pi / omega
    t_pi_rate_err: float = np.nan
    gamma: float = 0.0                     # decay rate of the envelope (1/s); 0 for 'none'
    gamma_err: float = 0.0
    v_up: float = np.nan                   # A + B (S_z = +1)
    v_down: float = np.nan                 # A - B (S_z = -1, no decay)
    contrast: float = np.nan               # 2 B
    rms: float = np.nan                    # residual rms per shot
    n_shots: int = 0
    n_excluded: int = 0
    dof: int = 0
    models: dict = field(default_factory=dict)   # every model tried: aicc, t_dead, t_dead_err, gamma
    lack_of_fit: dict = field(default_factory=dict)
    groups: list = field(default_factory=list)
    group_chi2: float = np.nan
    group_dof: int = 0
    group_p: float = np.nan
    configs: list = field(default_factory=list)
    profile: tuple = ()
    t_pi_ref: float = np.nan
    t_dead_ref: float = np.nan
    run_ids: tuple = ()
    signal_name: str = "APD"
    warnings: tuple = ()
    data: dict = field(default_factory=dict, repr=False)

    def summary(self):
        runs = ", ".join(str(r) for r in self.run_ids) or "?"
        if not self.ok:
            return f"split-pulse dead time, run {runs}: FAILED ({self.reason})"
        ns = 1e9
        lo, hi = self.t_dead_interval
        L = [f"split-pulse dead time, run {runs}: {self.n_shots} shots ({self.n_excluded} excluded: non-finite), "
             f"decay model '{self.model}'",
             f"  t_dead     {self.t_dead*ns:+.1f} +/- {self.t_dead_err*ns:.1f} ns   "
             f"(profile delta chi2 <= 1: {lo*ns:+.1f} .. {hi*ns:+.1f} ns)"]
        if np.isfinite(self.t_dead_ref):
            pull = (self.t_dead - self.t_dead_ref) / self.t_dead_err if self.t_dead_err > 0 else np.nan
            L.append(f"             params value {self.t_dead_ref*ns:.1f} ns: difference "
                     f"{(self.t_dead - self.t_dead_ref)*ns:+.1f} ns ({pull:+.1f} sigma)")
        L.append(f"  pi/omega   {self.t_pi_rate*1e6:.4f} +/- {self.t_pi_rate_err*1e6:.4f} us (rate pi time)")
        if np.isfinite(self.t_pi_ref):
            pull = (self.t_pi_rate - self.t_pi_ref) / self.t_pi_rate_err if self.t_pi_rate_err > 0 else np.nan
            L.append(f"             t_pi_ref {self.t_pi_ref*1e6:.4f} us: {pull:+.1f} sigma")
        if self.model != "none":
            L.append(f"  decay      gamma {self.gamma*1e-3:.1f} +/- {self.gamma_err*1e-3:.1f} /ms "
                     f"(1/gamma {1/self.gamma*1e6 if self.gamma > 0 else np.inf:.1f} us)")
        L.append(f"  {self.signal_name} up {self.v_up*1e3:.2f} mV, down {self.v_down*1e3:.2f} mV, contrast "
                 f"{self.contrast*1e3:.2f} mV, residual rms {self.rms*1e3:.2f} mV/shot (dof {self.dof})")
        if self.models:
            a0 = min(m["aicc"] for m in self.models.values())
            L.append("  models     " + "; ".join(
                f"{k}: dAICc {m['aicc'] - a0:+.1f}, t_dead {m['t_dead']*ns:+.1f} +/- {m['t_dead_err']*ns:.1f} ns"
                for k, m in self.models.items()))
        lf = self.lack_of_fit
        if lf:
            L.append(f"  lack of fit: F = {lf['F']:.2f} ({lf['dfn']}, {lf['dfd']}), p = {lf['p']:.3g}"
                     + ("   <- the model does not describe the configuration means" if lf["p"] < 0.01 else ""))
        for g in self.groups:
            if g.get("ok"):
                L.append(f"  group {g['group']} (total {g['total']*1e6:.3f} us, N = {g['n_values']}): "
                         f"t_dead {g['t_dead']*ns:+.1f} +/- {g['t_dead_err']*ns:.1f} ns, "
                         f"pi/omega {g['t_pi_rate']*1e6:.4f} us")
            else:
                L.append(f"  group {g['group']}: no fit ({g.get('reason', '')})")
        if self.group_dof > 0:
            L.append(f"  groups agree? chi2 {self.group_chi2:.2f} / {self.group_dof}, p = {self.group_p:.3g} "
                     "(approximate: the groups share the reference shots)")
        L.append("  per configuration (N, tau): mean +/- sem, model, residual / sem")
        for c in self.configs:
            z = (c["mean"] - c["model"]) / c["sem"] if np.isfinite(c["sem"]) and c["sem"] > 0 else np.nan
            L.append(f"    N={c['n']:2d} tau={c['tau']*1e6:7.4f} us  {c['mean']*1e3:8.2f} +/- {c['sem']*1e3:5.2f} mV  "
                     f"model {c['model']*1e3:8.2f}  {z:+5.1f}  (n={c['n_shots']})")
        for w in self.warnings:
            L.append(f"  WARNING: {w}")
        return "\n".join(L)

    def __repr__(self):
        return self.summary()

    def to_dict(self):
        out = {k: getattr(self, k) for k in ("ok", "reason", "model", "t_dead", "t_dead_err", "t_dead_interval",
                                             "omega", "omega_err", "t_pi_rate", "t_pi_rate_err", "gamma",
                                             "gamma_err", "v_up", "v_down", "contrast", "rms", "n_shots",
                                             "n_excluded", "dof", "models", "lack_of_fit", "groups", "group_chi2",
                                             "group_dof", "group_p", "configs", "t_pi_ref", "t_dead_ref",
                                             "run_ids", "warnings")}
        out["t_dead_interval"] = list(out["t_dead_interval"])
        out["run_ids"] = list(out["run_ids"])
        out["warnings"] = list(out["warnings"])
        return out

    def plot(self, ax=None):
        """Signal against N, one panel per split group, with the joint model; the
        reference levels as horizontal lines."""
        import matplotlib.pyplot as plt
        d = self.data
        split = list(self.groups)
        if ax is None:
            fig, axs = plt.subplots(1, max(len(split), 1), figsize=(4.2 * max(len(split), 1), 3.4), squeeze=False)
            axs = axs[0]
        else:
            axs = np.atleast_1d(ax)
            fig = axs[0].figure
        runs = ", ".join(str(r) for r in self.run_ids) or "?"
        A, B = (self.v_up + self.v_down) / 2, self.contrast / 2
        for a, g in zip(axs, split):
            cs = [c for c in self.configs if c["group"] == g["group"]]
            nn = np.array([c["n"] for c in cs])
            m = d["group"] == g["group"]
            a.plot(d["n"][m], d["y"][m] * 1e3, ".", color="0.7", ms=3, zorder=0, label="shots")
            a.errorbar(nn, [c["mean"] * 1e3 for c in cs], yerr=[c["sem"] * 1e3 for c in cs], fmt="o", capsize=3,
                       label="mean +/- sem")
            nf = np.linspace(1, nn.max(), 200)
            a.plot(nf, 1e3 * split_pulse_model(nf, g["total"] / nf, A, B, self.omega, self.t_dead, self.gamma,
                                               self.model), "-", label="joint fit")
            a.axhline(self.v_up * 1e3, ls=":", color="k", lw=0.8)
            a.axhline(self.v_down * 1e3, ls=":", color="k", lw=0.8)
            a.set_xlabel("sub-pulses N")
            a.set_ylabel(f"{self.signal_name} (mV)")
            a.set_title(f"total {g['total']*1e6:.3f} us", fontsize=9)
        axs[0].legend(fontsize=7)
        fig.suptitle(f"run {runs} | t_dead {self.t_dead*1e9:.1f} +/- {self.t_dead_err*1e9:.1f} ns ('{self.model}'), "
                     f"pi/omega {self.t_pi_rate*1e6:.3f} us (t_pi_ref {self.t_pi_ref*1e6:.3f} us)", fontsize=9)
        fig.tight_layout()
        return fig


def fit_split_pulse_dead_time(n_split, tau, signal, *, t_pi_ref, group=None, model="auto", t_dead_ref=np.nan,
                              run_ids=(), signal_name="APD") -> SplitPulseDeadTime:
    """Fit the dead time of a split-pulse scan; see the module docstring.

    n_split, tau : per shot, the number of sub-pulses and the commanded sub-pulse length (s).
    signal : per shot, (n_shots,) or (n_shots, P) reads (averaged per shot).
    t_pi_ref : rate pi time (s), only to place the start grid for omega and to compare.
    group : per-shot group id (e.g. data.raman_split[:, 2]); else grouped by commanded total.
    model : decay envelope, 'auto' (lowest AICc) or one of 'none', 'exp', 'gauss'.
    t_dead_ref : the dead time in use, to compare.
    """
    from scipy.stats import chi2 as chi2_dist, f as f_dist
    if model != "auto" and model not in MODELS:
        raise ValueError(f"model must be 'auto' or one of {MODELS}, got {model!r}")
    n = np.asarray(n_split, float).ravel()
    tau = np.asarray(tau, float).ravel()
    s = np.asarray(signal, float)
    s = s.reshape(s.shape[0], -1)
    if not (n.size == tau.size == s.shape[0]):
        raise ValueError(f"n_split ({n.size}), tau ({tau.size}) and signal ({s.shape[0]}) disagree on the shot count")
    g = None if group is None else np.asarray(group, float).ravel()
    ok = np.isfinite(n) & np.isfinite(tau) & np.all(np.isfinite(s), axis=1)
    if g is not None:
        ok &= np.isfinite(g)
    n_excl = int(np.sum(~ok))
    n, tau, y = np.round(n[ok]), tau[ok], s[ok].mean(axis=1)
    g = _groups_from_totals(n, tau) if g is None else np.round(g[ok]).astype(int)
    common = dict(t_pi_ref=float(t_pi_ref), t_dead_ref=float(t_dead_ref), run_ids=tuple(int(r) for r in run_ids),
                  signal_name=signal_name, n_excluded=n_excl, n_shots=int(y.size))
    if np.unique(n).size < 3 or y.size < 8:
        return SplitPulseDeadTime(False, reason=f"too few shots or N values ({y.size} shots, N in {np.unique(n)})",
                                  **common)
    omega_ref = np.pi / float(t_pi_ref)

    f0 = _fit(n, tau, y, omega_ref, "none")
    if f0 is None:
        return SplitPulseDeadTime(False, reason="least squares failed", **common)
    fits = {"none": f0}
    for m in (MODELS[1:] if model == "auto" else ([model] if model != "none" else [])):
        fm = _fit(n, tau, y, omega_ref, m, start=f0["p"])
        if fm is not None:
            fits[m] = fm
    chosen = model if model != "auto" else min(fits, key=lambda k: fits[k]["aicc"])
    f = fits[chosen]
    if not np.all(np.isfinite(f["cov"])):
        return SplitPulseDeadTime(False, reason="covariance not finite", **common)
    models = {k: dict(aicc=v["aicc"], t_dead=v["t_dead"], t_dead_err=float(np.sqrt(max(v["cov"][3, 3], 0))),
                      gamma=v["gamma"]) for k, v in fits.items()}
    warn = []
    if not f["success"]:
        warn.append("least squares reports no convergence")
    om, td, cov = f["omega"], f["t_dead"], f["cov"]
    om_e, td_e = float(np.sqrt(max(cov[2, 2], 0))), float(np.sqrt(max(cov[3, 3], 0)))
    gam_e = float(np.sqrt(max(cov[4, 4], 0))) if f["k"] == 5 else 0.0
    if f["k"] == 5 and f["gamma"] <= 0:
        warn.append(f"decay model '{chosen}' railed at gamma = 0 (no decay resolved)")
    if np.any((n > 0) & (tau <= td)):
        warn.append(f"t_dead {td*1e9:.0f} ns is at or above the shortest sub-pulses: those rotate nothing in the "
                    "model, and the covariance error is not reliable (use the profile interval)")
    span = om * td * (n.max() - 1)
    if abs(span) > np.pi:
        warn.append(f"the rotation lost over the scan ({span/np.pi:.2f} pi) exceeds pi: check the per-N means "
                    "by eye for an alias")
    spread = [v["t_dead"] for v in fits.values()]
    if len(spread) > 1 and np.ptp(spread) > td_e:
        warn.append(f"t_dead depends on the decay model by {np.ptp(spread)*1e9:.0f} ns (> its 1 sigma error)")
    grid, dchi2, interval = _profile(n, tau, y, f)
    if not np.all(np.isfinite(interval)):
        warn.append("the profile interval runs off its grid (one side unbounded within +/- 6 sigma)")

    cfg, ss_pe, df_pe = _configs(n, tau, y)
    for c in cfg:
        m = (n == c["n"]) & np.isclose(tau, c["tau"], rtol=0, atol=1e-12)
        c["group"] = int(np.bincount(g[m] - g.min()).argmax() + g.min())
        c["model"] = float(_p_model(f["p"], c["n"], c["tau"], chosen))
    lof = {}
    df_lof = len(cfg) - f["k"]
    if df_lof > 0 and df_pe > 0 and ss_pe > 0:
        ss_lof = f["rss"] - ss_pe
        F = (ss_lof / df_lof) / (ss_pe / df_pe)
        lof = dict(F=float(F), dfn=int(df_lof), dfd=int(df_pe), p=float(f_dist.sf(F, df_lof, df_pe)))
        if lof["p"] < 0.01:
            warn.append(f"lack of fit p = {lof['p']:.2g}: the configuration means scatter about the model more "
                        "than the repeats allow; t_dead and its error are suspect")

    # groups: those with >= 2 N values are fitted alone, with every single-N group as reference
    gids = sorted(set(g.tolist()))
    n_per_group = {k: np.unique(n[g == k]).size for k in gids}
    refs = np.isin(g, [k for k in gids if n_per_group[k] < 2])
    groups = []
    for k in gids:
        if n_per_group[k] < 2:
            continue
        m = refs | (g == k)
        info = dict(group=int(k), total=float(np.median((n * tau)[g == k])),
                    n_values=[int(v) for v in np.unique(n[g == k])], n_shots=int(np.sum(g == k)))
        fg = _fit(n[m], tau[m], y[m], omega_ref, chosen, start=f["p"])
        if fg is None or not np.all(np.isfinite(fg["cov"])):
            info.update(ok=False, reason="fit failed")
        else:
            oe = float(np.sqrt(max(fg["cov"][2, 2], 0)))
            info.update(ok=True, t_dead=fg["t_dead"], t_dead_err=float(np.sqrt(max(fg["cov"][3, 3], 0))),
                        omega=fg["omega"], t_pi_rate=np.pi / fg["omega"], t_pi_rate_err=np.pi * oe / fg["omega"] ** 2)
        groups.append(info)
    good = [x for x in groups if x.get("ok") and x["t_dead_err"] > 0]
    g_chi2, g_dof, g_p = np.nan, 0, np.nan
    if len(good) >= 2:
        v = np.array([x["t_dead"] for x in good])
        w = 1 / np.array([x["t_dead_err"] for x in good]) ** 2
        mean = np.sum(w * v) / np.sum(w)
        g_chi2, g_dof = float(np.sum(w * (v - mean) ** 2)), len(good) - 1
        g_p = float(chi2_dist.sf(g_chi2, g_dof))
        if g_p < 0.01:
            warn.append(f"the groups disagree on t_dead (p = {g_p:.2g}): something the model lacks "
                        "(a nonlinear readout, a length-dependent edge, a decay unlike the envelope)")

    return SplitPulseDeadTime(
        True, model=chosen, t_dead=td, t_dead_err=td_e, t_dead_interval=interval, omega=om, omega_err=om_e,
        t_pi_rate=np.pi / om, t_pi_rate_err=np.pi * om_e / om ** 2, gamma=f["gamma"], gamma_err=gam_e,
        v_up=f["A"] + f["B"], v_down=f["A"] - f["B"], contrast=2 * f["B"], rms=float(np.sqrt(f["rss"] / f["dof"])),
        dof=int(f["dof"]), models=models, lack_of_fit=lof, groups=groups, group_chi2=g_chi2, group_dof=g_dof,
        group_p=g_p, configs=cfg, profile=(grid, dchi2), warnings=tuple(warn), data=dict(n=n, tau=tau, y=y, group=g),
        **common)
