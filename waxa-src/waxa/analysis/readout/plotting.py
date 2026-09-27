"""Figures for :class:`waxa.analysis.readout.ReadoutCalibration`."""

import numpy as np


def _unit(cal, values):
    try:
        from waxa.plotting.units import detect_unit
        return detect_unit(xvarnames=[cal.xvarname], xvar_values=values, params_obj=cal.params)
    except Exception:
        return "", 1.0, cal.xvarname


def _signal_unit(cal):
    """Display unit for the signal axis: mV for a volt-scale signal."""
    return ("mV", 1e3) if np.nanmax(np.abs(cal.signal)) < 10 else ("", 1.0)


def _title_runs(cal):
    return "run " + ", ".join(map(str, cal.run_ids)) if cal.run_ids else "data"


def _curve_sz(cal, tt):
    from .calibration import flop_coordinate
    from .sz_response import pulse_time_to_angle_and_sz
    if cal.sz_source == "fit":
        f = cal.flop
        return flop_coordinate(tt, f.omega, f.t_dead, f.gamma, f.model)
    return pulse_time_to_angle_and_sz(tt, cal.t_pi_ref, cal.t_offset_ref)[1]


def plot_pulses(cal, figsize=None):
    """Top row: every shot's signal against the drive pulse length, one column per readout
    pulse, coloured by run, with the linear S_z fit of that pulse (every run pooled, the
    flop from the joint Rabi fit) and its S_z = +1 / -1 levels.  Bottom row: residuals
    about each run's own linear fit, with the within-run single-pulse sigma shaded.
    Returns (fig, axes)."""
    import matplotlib.pyplot as plt

    P = cal.n_pulses
    unit, mult, label = _unit(cal, cal.t)
    su, sm = _signal_unit(cal)
    figsize = figsize or (3.1 * P + 1.0, 6.2)
    fig, axes = plt.subplots(2, P, figsize=figsize, sharex=True, sharey="row", squeeze=False,
                             gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.06, "wspace": 0.06})
    tt = np.linspace(0, np.nanmax(cal.t), 600)
    s_curve = _curve_sz(cal, tt)
    for p in range(P):
        a, ar = axes[0, p], axes[1, p]
        e = cal.per_pulse[p]
        c0, b0 = 0.5 * (e.v_up + e.v_down), 0.5 * (e.v_up - e.v_down)
        for j, r in enumerate(cal.run_ids):
            m = cal.run == r
            a.plot(cal.t[m] * mult, cal.signal[m, p] * sm, ".", color=f"C{j}", ms=4, alpha=0.7,
                   label=f"run {r}" if p == 0 else None)
            ar.plot(cal.t[m] * mult, cal.residual_within[m, p] * sm, ".", color=f"C{j}", ms=4, alpha=0.7)
        a.plot(tt * mult, (c0 + b0 * s_curve) * sm, "-", color="k", lw=1.6, label="fit (pooled)" if p == 0 else None)
        a.axhline(e.v_up * sm, color="C2", ls="--", lw=1)
        a.axhline(e.v_down * sm, color="C3", ls="--", lw=1)
        a.set_title(f"pulse {p + 1}{' (not averaged)' if p not in cal.pulses_used else ''}\n"
                    f"up {e.v_up*sm:.1f}$\\pm${e.v_up_err*sm:.1f}, down {e.v_down*sm:.1f}$\\pm${e.v_down_err*sm:.1f} {su}",
                    fontsize=9)
        nz = cal.noise.get(f"within-run, pulse {p + 1}, up")
        nd = cal.noise.get(f"within-run, pulse {p + 1}, down")
        ar.axhline(0, color="k", lw=0.8)
        for n_, col in ((nz, "C2"), (nd, "C3")):
            if n_ is not None and np.isfinite(n_.sigma):
                ar.axhspan(-n_.sigma * sm, n_.sigma * sm, color=col, alpha=0.08, lw=0)
        ar.set_xlabel(f"{label} ({unit})" if unit else label)
    axes[0, 0].set_ylabel(f"{cal.signal_name} ({su})" if su else cal.signal_name)
    axes[1, 0].set_ylabel(f"residual ({su})" if su else "residual")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    f = cal.flop
    flop_txt = (f"flop: pi {f.t_pi*1e6:.3f}$\\pm${f.t_pi_err*1e6:.3f} us, dead time {f.t_dead*1e9:.0f} ns"
                if f is not None and f.ok else "")
    fig.suptitle(f"{_title_runs(cal)} | {cal.signal_name} vs drive pulse length, S_z from '{cal.sz_source}' | "
                 f"{flop_txt}" + (f"\n{cal.context}" if cal.context else ""), fontsize=10, y=1.03)
    return fig, axes


def plot_summary(cal, figsize=(12.5, 9.0)):
    """(a) pulse-averaged signal against S_z with the linear and quadratic fits; (b) V_up /
    V_down per run; (c) per readout pulse; (d) single-pulse residual sigma against S_z
    with the proposed sigma.  Returns (fig, axes)."""
    import matplotlib.pyplot as plt

    su, sm = _signal_unit(cal)
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    a, b, c, d = ax.ravel()
    ul = list(cal.pulses_used)
    ybar = cal.signal[:, ul].mean(axis=1)
    for j, r in enumerate(cal.run_ids):
        m = cal.run == r
        a.plot(cal.sz[m], ybar[m] * sm, ".", color=f"C{j}", alpha=0.45, ms=4, label=f"run {r} shots")
    nb = cal.notebook
    if nb:
        a.errorbar(nb["sz_axis"], nb["mean"] * sm, yerr=nb["sem"] * sm, fmt="o", color="k", ms=4, capsize=2,
                   label="mean $\\pm$ SEM per pulse length")
    sg = np.linspace(-1, 1, 200)
    for e, ls, col, nm in ((cal.pooled_linear, "-", "C3", "linear"), (cal.pooled_quadratic, "--", "C1", "quadratic")):
        cc = 0.5 * (e.v_up + e.v_down)
        bb = 0.5 * (e.v_up - e.v_down)
        # the quadratic's curvature from its midpoint: C = (0.5 - midpoint) * range
        C = (0.5 - e.midpoint) * e.contrast if e.degree == 2 else 0.0
        a.plot(sg, (cc - C + bb * sg + C * sg ** 2) * sm, ls, color=col, lw=1.6,
               label=f"{nm}: up {e.v_up*sm:.1f}, down {e.v_down*sm:.1f} {su}"
               + (f", midpoint {e.midpoint:.2f}$\\pm${e.midpoint_err:.2f}" if e.degree == 2 else ""))
    a.set_xlabel("$S_z$")
    a.set_ylabel(f"{cal.signal_name}, mean of pulses {', '.join(str(p + 1) for p in ul)} ({su})")
    a.legend(fontsize=7)
    a.set_title(f"{_title_runs(cal)}: response vs $S_z$", fontsize=10)

    x = np.arange(len(cal.per_run))
    for k, (key, col) in enumerate((("v_up", "C2"), ("v_down", "C3"))):
        vals = np.array([getattr(e, key) for e in cal.per_run]) * sm
        errs = np.array([getattr(e, key + "_err") for e in cal.per_run]) * sm
        b.errorbar(x + 0.05 * (k - 0.5), vals, yerr=errs, fmt="o", color=col, capsize=3, label=key)
        pv = getattr(cal.pooled_linear, key) * sm
        pe = getattr(cal.pooled_linear, key + "_err") * sm
        b.axhspan(pv - pe, pv + pe, color=col, alpha=0.12, lw=0)
    b.set_xticks(x)
    b.set_xticklabels([str(r) for r in cal.run_ids])
    b.set_xlabel("run (acquisition order)")
    b.set_ylabel(f"endpoint ({su})")
    ag = cal.agreement
    ttl = "per run (pulse-averaged, linear); band = pooled"
    if ag.get("f_offset"):
        ttl += f"\nF-test own offset p = {ag['f_offset']['p']:.2g}, own slope p = {ag['f_slope']['p']:.2g}"
    b.set_title(f"{_title_runs(cal)}: " + ttl, fontsize=9)
    b.legend(fontsize=8)

    xp = np.arange(1, cal.n_pulses + 1)
    for k, (key, col) in enumerate((("v_up", "C2"), ("v_down", "C3"))):
        vals = np.array([getattr(e, key) for e in cal.per_pulse]) * sm
        errs = np.array([getattr(e, key + "_err") for e in cal.per_pulse]) * sm
        c.errorbar(xp + 0.05 * (k - 0.5), vals, yerr=errs, fmt="o", color=col, capsize=3, label=key)
    c.set_xticks(xp)
    c.set_xlabel("readout pulse")
    c.set_ylabel(f"endpoint ({su})")
    c.set_title(f"{_title_runs(cal)}: per readout pulse (every run, linear)", fontsize=10)
    c.legend(fontsize=8)

    nv = cal.noise_vs_sz
    if len(nv.get("sz", [])):
        d.plot(nv["sz"], nv["sigma"] * sm, "o-", color="C0", label="within-run, used pulses, per S_z bin")
        for i, (s_, n_) in enumerate(zip(nv["sz"], nv["n_shots"])):
            d.annotate(str(n_), (s_, nv["sigma"][i] * sm), textcoords="offset points", xytext=(0, 5),
                       fontsize=7, ha="center")
    for key, col, xpos in (("within-run, up", "C2", 1.0), ("within-run, down", "C3", -1.0),
                           ("pooled, up", "C2", 0.97), ("pooled, down", "C3", -0.97)):
        n_ = cal.noise.get(key)
        if n_ is not None and np.isfinite(n_.sigma):
            d.errorbar([xpos], [n_.sigma * sm], yerr=[[(n_.sigma - n_.lo) * sm], [(n_.hi - n_.sigma) * sm]],
                       fmt="s" if key.startswith("within") else "D", color=col, capsize=3, mfc="none" if
                       key.startswith("pooled") else col, label=key)
    if cal.proposal:
        d.axhline(cal.proposal["sigma_signal"] * sm, color="k", ls="--", lw=1,
                  label=f"proposed sigma {cal.proposal['sigma_signal']*sm:.2f} {su}")
    d.set_xlabel("$S_z$")
    d.set_ylabel(f"single-pulse residual std ({su})")
    d.set_ylim(bottom=0)
    d.legend(fontsize=7)
    d.set_title(f"{_title_runs(cal)}: noise (numbers = shots per bin)", fontsize=10)
    fig.tight_layout()
    return fig, ax
