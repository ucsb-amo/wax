"""Plot for :class:`waxa.analysis.rabi.RabiFit`: data, fit, pi / pi/2 markers, residuals."""

import numpy as np


def _unit(fit, values):
    try:
        from waxa.plotting.units import detect_unit
        return detect_unit(xvarnames=[fit.xvarname], xvar_values=values, params_obj=fit.params)
    except Exception:
        return "µs", 1e6, fit.xvarname


def plot_rabi(fit, ax=None, residuals=True, show_shots=True, reference=None, figsize=(8, 5.5)):
    """Data (per-point mean +/- SEM where there are repeats, single shots faint), the
    fitted curve, the pi and pi/2 pulse lengths, and optionally a pi time in use
    (``reference``, s) with its half.  Returns (fig, axes)."""
    import matplotlib.pyplot as plt

    if not fit.datasets:
        raise ValueError("nothing to plot: the fit holds no data")
    allt = np.concatenate([d.tu for d in fit.datasets])
    unit, mult, label = _unit(fit, allt)
    if ax is None:
        if residuals:
            fig, (ax, axr) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axr = None
    else:
        fig, axr = ax.figure, None
    tf = np.linspace(0, allt.max(), 800)
    for j, d in enumerate(fit.datasets):
        c = f"C{j}"
        name = fit.labels[j] if j < len(fit.labels) else f"dataset {j}"
        if show_shots and np.any(d.n > 1):
            ax.plot(d.t * mult, d.y, ".", color=c, alpha=0.3, ms=4)
        rep = d.n > 1
        ax.errorbar(d.tu * mult, d.mean, yerr=np.where(rep, d.sem, np.nan), fmt="o", color=c, ms=4.5,
                    capsize=2, label=name + (" (mean +/- SEM)" if rep.any() else ""))
        if fit.ok:
            ax.plot(tf * mult, fit.curve(tf, j), "-", color="k" if len(fit.datasets) == 1 else c, lw=1.8)
            if axr is not None:
                r = d.mean - fit.curve(d.tu, j)
                axr.errorbar(d.tu * mult, r, yerr=np.where(rep, d.sem, np.nan), fmt="o", color=c, ms=3.5, capsize=2)
    if fit.ok:
        for tv, ls, nm in ((fit.t_pi, "--", "pi"), (fit.t_half_pi, ":", "pi/2")):
            ax.axvline(tv * mult, color="C3", ls=ls, lw=1.2)
        ax.plot([], [], "--", color="C3", label=f"pi {fit.t_pi*1e6:.3f} +/- {fit.t_pi_err*1e6:.3f} us")
        ax.plot([], [], ":", color="C3", label=f"pi/2 {fit.t_half_pi*1e6:.3f} +/- {fit.t_half_pi_err*1e6:.3f} us")
    if reference is not None:
        ax.axvline(reference * mult, color="C2", lw=1.2, alpha=0.8)
        ax.axvline(reference / 2 * mult, color="C2", lw=1.0, alpha=0.5)
        ax.plot([], [], "-", color="C2", label=f"in use: pi {reference*1e6:.3f} us (and pi/2)")
    runs = ", ".join(map(str, fit.run_ids)) if fit.run_ids else ""
    head = f"run {runs} | " if runs else ""
    ax.set_title(head + (f"f_Rabi = {fit.f_rabi/1e3:.2f} +/- {fit.f_rabi_err/1e3:.2f} kHz, model '{fit.model}'"
                         if fit.ok else f"fit failed: {fit.reason}"), fontsize=10)
    ax.set_ylabel(fit.signal_name)
    ax.legend(fontsize=8)
    if axr is not None:
        axr.axhline(0, color="k", lw=0.8)
        axr.set_ylabel("residual")
        axr.set_xlabel(f"{label} ({unit})")
    else:
        ax.set_xlabel(f"{label} ({unit})")
    return fig, (ax, axr) if axr is not None else (ax,)


def plot_linearization(m, figsize=(12.5, 4.2)):
    """Three panels for a :class:`~.linearize.SignalMapping`: the repeat means against
    their fitted flop coordinate v with the detection curve D and the straight chord
    through D(-1), D(+1) (a linear detector puts every point on the chord); the means
    against pulse length with the plain cosine and the cubic-of-cosine fits; and both
    fits' residuals in units of the SEM.  Returns (fig, axes)."""
    import matplotlib.pyplot as plt
    from .linearize import _fit_arrays, _flop_v

    if not m.ok:
        raise ValueError(f"nothing to plot: {m.reason}")
    raw = m.fit_raw
    T, Y, S, K, _ = _fit_arrays(list(raw.datasets), m.noise, m.sem_floor)
    unit, mult, label = _unit(raw, T)
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.1, 1.5, 1.5]})
    runs = ", ".join(map(str, m.run_ids)) if m.run_ids else ""
    V = np.asarray(m.centers)[K] + np.asarray(m.scales)[K] * _flop_v(T, m.omega, m.phase, m.gamma, m.model)
    vv = np.linspace(min(m.v_range[0], -1), max(m.v_range[1], 1), 300)
    lo, hi = m.detection(-1.0), m.detection(1.0)
    a0.plot(vv, lo + (vv + 1) / 2 * (hi - lo), "--", color="0.6", lw=1, label="linear (chord)")
    a0.plot(vv, m.detection(vv), "-", color="C3", lw=2, label=f"D, degree {m.degree}")
    for j in range(len(m.centers)):
        sel = K == j
        a0.errorbar(V[sel], Y[sel], yerr=S[sel], fmt="o", color=f"C{j}", ms=3.5, capsize=2,
                    label=f"run {m.run_ids[j]}" if j < len(m.run_ids) else None)
    a0.set_xlabel("flop coordinate v = env cos(omega t + phi)")
    a0.set_ylabel(m.signal_name)
    a0.legend(fontsize=7)
    a0.set_title("detection curve" + ("" if m.monotonic else " (NOT monotonic)"), fontsize=10)

    tf = np.linspace(0, T.max(), 800)
    for j in range(len(m.centers)):
        sel = K == j
        c = f"C{j}"
        a1.errorbar(T[sel] * mult, Y[sel], yerr=S[sel], fmt="o", color=c, ms=4, capsize=2)
        a1.plot(tf * mult, raw.curve(tf, j), ":", color="0.35", lw=1.3, label="cosine" if j == 0 else None)
        vf = m.centers[j] + m.scales[j] * _flop_v(tf, m.omega, m.phase, m.gamma, m.model)
        a1.plot(tf * mult, m.detection(vf), "-", color="k" if len(m.centers) == 1 else c, lw=1.6,
                label="D(cosine)" if j == 0 else None)
        pr = (Y[sel] - raw.curve(T[sel], j)) / S[sel]
        pm = (Y[sel] - m.detection(V[sel])) / S[sel]
        a2.plot(T[sel] * mult, pr, "o-", color="0.6", ms=3.5, lw=0.8, label="cosine" if j == 0 else None)
        a2.plot(T[sel] * mult, pm, "o-", color=c, ms=4, lw=1, label=f"D(cosine)" if j == 0 else None)
    a1.axvline(m.t_pi * mult, color="C3", ls="--", lw=1)
    a1.plot([], [], "--", color="C3", label=f"pi {m.t_pi*1e6:.3f} +/- {m.t_pi_err*1e6:.3f} us")
    a1.set_xlabel(f"{label} ({unit})")
    a1.set_ylabel(f"{m.signal_name} (mean +/- SEM)")
    a1.legend(fontsize=7)
    a1.set_title(f"run {runs}" if runs else "data", fontsize=10)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xlabel(f"{label} ({unit})")
    a2.set_ylabel("residual / SEM")
    a2.legend(fontsize=8)
    a2.set_title(f"chi2/dof {m.chi2_raw/m.dof_raw:.2f} -> {m.chi2_mapped/m.dof_mapped:.2f}, "
                 f"dAICc {m.aicc_mapped - m.aicc_raw:+.1f}", fontsize=10)
    fig.tight_layout()
    return fig, (a0, a1, a2)
