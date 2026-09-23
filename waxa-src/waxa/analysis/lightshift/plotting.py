"""Plots for :class:`waxa.analysis.lightshift.RamseyLightShift`.

Every point of :func:`plot_light_shift` is a panel of :func:`plot_fringes`: both
read the same fit objects.
"""

import numpy as np
import matplotlib.pyplot as plt

from waxa.plotting.units import detect_unit

COLOR_LIGHT = "tab:orange"      # fringe with the light on
COLOR_REFERENCE = "tab:blue"    # fringe with the light off (measured or assumed)


def _axis_unit(ls, name, values):
    unit, mult, label = detect_unit(xvarnames=[name], xvar_values=values,
                                    params_obj=ls.params)
    return unit, mult, label


def _cell_label(ls, where):
    parts = []
    for name, value in where.items():
        unit, mult, label = _axis_unit(ls, name, [value])
        parts.append(f"{label} = {value * mult:.4g} {unit}".rstrip())
    return "\n".join(parts)


def plot_fringes(ls, ncols=4, figsize=None):
    """One panel per scan cell: repeat means (sem bars), the fit, the reference."""
    cells = [c for c in ls.cells() if np.any(np.isfinite(c[2].mean))]
    n = len(cells)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False, layout="constrained",
                             figsize=figsize or (3.2 * ncols, 2.6 * nrows))
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    for i, (idx, where, fit, ref) in enumerate(cells):
        ax = axes.flat[i]
        # draw the fit across the whole scanned phase window, not just one period
        scanned = np.concatenate([fit.phase_values] + ([ref.phase_values] if ref is not None else []))
        scanned = scanned[np.isfinite(scanned)]
        grid = np.linspace(scanned.min(), scanned.max(), 400)
        ax.errorbar(fit.phase_values, fit.mean, yerr=fit.sem, fmt="o", ms=3, capsize=2,
                    color=COLOR_LIGHT, label="light on: mean, sem")
        if fit.ok:
            ax.plot(grid, fit.model(grid), "--", color=COLOR_LIGHT, label="fit")
        if ref is not None:
            ax.errorbar(ref.phase_values, ref.mean, yerr=ref.sem, fmt="o", ms=3, capsize=2,
                        color=COLOR_REFERENCE, label="light off: mean, sem")
            if ref.ok:
                ax.plot(grid, ref.model(grid), "--", color=COLOR_REFERENCE, label="fit")
        elif fit.ok:
            ax.plot(grid, fit.amplitude * np.cos(grid - ls.reference_phase) + fit.offset,
                    ":", color=COLOR_REFERENCE, alpha=0.7,
                    label=f"assumed reference ({ls.reference_phase:.2f} rad)")
        f, err = ls.f_lightshift_Hz[idx], ls.f_lightshift_err_Hz[idx]
        title = _cell_label(ls, where)
        title += ("\n" if title else "") + (
            rf"$\Delta\phi$ = {ls.phase_shift[idx]:.2f}({ls.phase_shift_err[idx]:.2f}) rad, "
            rf"$f_{{ls}}$ = {f / 1e3:.1f}({err / 1e3:.1f}) kHz" if fit.ok else "fit failed")
        ax.set_title(title, fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel(ls.signal_name)
        if i // ncols == nrows - 1:
            ax.set_xlabel(f"{ls.phase_name} (rad)")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=5)
    fig.suptitle(f"{ls.run_id_title}\nRamsey fringe per scan cell, "
                 f"{ls.n_ok}/{ls.n_total} fits ok, pulse {ls.t_pulse * 1e6:.3g} µs",
                 fontsize=9)
    return fig, axes


def plot_light_shift(ls, ax=None, x=None, xlabel=None, figsize=(6, 4), cmap="plasma",
                     labels=None, **errorbar_kwargs):
    """Light shift (kHz) against the last scan axis, one curve per leading cell.

    ``x`` replaces the last scan axis's values (same length), e.g. with the probe
    offset from the midpoint; ``labels`` formats a leading cell's dict of
    ``{scan name: value}`` into a legend entry.
    """
    if not ls.scan_names:
        raise ValueError("This result is a single light shift; there is nothing to plot against.")
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    name, values = ls.scan_names[-1], ls.scan_values[-1]
    if x is None:
        unit, mult, label = _axis_unit(ls, name, values)
        x = values * mult
        xlabel = xlabel or (f"{label} ({unit})" if unit else label)
    lead_shape = ls.scan_shape[:-1]
    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.8, max(int(np.prod(lead_shape)), 1)))
    style = dict(fmt="o", ms=5, capsize=3)
    style.update(errorbar_kwargs)
    for color, lead in zip(colors, np.ndindex(lead_shape)):
        where = {n: v[i] for n, v, i in zip(ls.scan_names, ls.scan_values, lead)}
        label = labels(where) if labels else (_cell_label(ls, where).replace("\n", ", ") or None)
        f, err = ls.f_lightshift_Hz[lead] / 1e3, ls.f_lightshift_err_Hz[lead] / 1e3
        ok = np.isfinite(f)
        ax.errorbar(np.asarray(x)[ok], f[ok], yerr=err[ok], color=color, label=label, **style)
    ax.set_xlabel(xlabel or name)
    ax.set_ylabel("light shift (kHz)")
    if created:
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.grid(alpha=0.2)
        ax.set_title(f"{ls.run_id_title}\nlight shift from the Ramsey phase jump, "
                     f"pulse {ls.t_pulse * 1e6:.3g} µs"
                     + ("" if ls.reference_measured
                        else f", reference assumed at {ls.reference_phase:.2f} rad"),
                     fontsize=9)
        if lead_shape:
            ax.legend()
    return ax
