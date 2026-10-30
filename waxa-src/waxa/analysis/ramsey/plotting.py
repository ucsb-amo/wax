"""Plots for :class:`waxa.analysis.ramsey.RamseyPhaseScan` and :class:`PhaseModelFit`.

Every point of :func:`plot_phase_shift` is a panel of :func:`plot_fringes`: both
read the same fit objects.
"""

import numpy as np
import matplotlib.pyplot as plt

from waxa.plotting.units import detect_unit

COLOR_FRINGE = "tab:orange"
COLOR_REFERENCE = "tab:blue"    # the reference fringe (measured or assumed)


def _axis_unit(scan, name, values):
    unit, mult, label = detect_unit(xvarnames=[name], xvar_values=values,
                                    params_obj=scan.params)
    return unit, mult, label


def _cell_label(scan, where):
    parts = []
    for name, value in where.items():
        unit, mult, label = _axis_unit(scan, name, [value])
        parts.append(f"{label} = {value * mult:.4g} {unit}".rstrip())
    return "\n".join(parts)


def _default_annotate(scan, idx, fit):
    if not fit.ok:
        return "fit failed"
    return rf"$\Delta\phi$ = {scan.phase_shift[idx]:.2f}({scan.phase_shift_err[idx]:.2f}) rad"


def plot_fringes(scan, ncols=4, figsize=None, labels=("fringe", "reference"),
                 annotate=None, subtitle=None):
    """One panel per scan cell: repeat means (sem bars), the fit, the reference.

    ``labels`` names the fringe and the reference in the legend; ``annotate(scan,
    idx, fit)`` gives each panel's result line; ``subtitle`` is added to the
    figure title.
    """
    annotate = annotate or _default_annotate
    cells = [c for c in scan.cells() if np.any(np.isfinite(c[2].mean))]
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
                    color=COLOR_FRINGE, label=f"{labels[0]}: mean, sem")
        if fit.ok:
            ax.plot(grid, fit.model(grid), "--", color=COLOR_FRINGE, label="fit")
        if ref is not None:
            ax.errorbar(ref.phase_values, ref.mean, yerr=ref.sem, fmt="o", ms=3, capsize=2,
                        color=COLOR_REFERENCE, label=f"{labels[1]}: mean, sem")
            if ref.ok:
                ax.plot(grid, ref.model(grid), "--", color=COLOR_REFERENCE, label="fit")
        elif fit.ok:
            ax.plot(grid, fit.amplitude * np.cos(grid - scan.reference_phase) + fit.offset,
                    ":", color=COLOR_REFERENCE, alpha=0.7,
                    label=f"assumed reference ({scan.reference_phase:.2f} rad)")
        title = _cell_label(scan, where)
        title += ("\n" if title else "") + annotate(scan, idx, fit)
        ax.set_title(title, fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel(scan.signal_name)
        if i // ncols == nrows - 1:
            ax.set_xlabel(f"{scan.phase_name} (rad)")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=5)
    fig.suptitle(f"{scan.run_id_title}\nRamsey fringe per scan cell, "
                 f"{scan.n_ok}/{scan.n_total} fits ok" + (f", {subtitle}" if subtitle else ""),
                 fontsize=9)
    return fig, axes


def plot_vs_last_axis(scan, y, yerr, ylabel, ax=None, x=None, xlabel=None, figsize=(6, 4),
                      cmap="plasma", labels=None, title=None, **errorbar_kwargs):
    """``y`` (scan-shaped) against the last scan axis, one curve per leading cell.

    ``x`` replaces the last scan axis's values (same length); ``labels`` formats a
    leading cell's dict of ``{scan name: value}`` into a legend entry.
    """
    if not scan.scan_names:
        raise ValueError("This result is a single cell; there is nothing to plot against.")
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    name, values = scan.scan_names[-1], scan.scan_values[-1]
    if x is None:
        unit, mult, label = _axis_unit(scan, name, values)
        x = values * mult
        xlabel = xlabel or (f"{label} ({unit})" if unit else label)
    lead_shape = scan.scan_shape[:-1]
    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.8, max(int(np.prod(lead_shape)), 1)))
    style = dict(fmt="o", ms=5, capsize=3)
    style.update(errorbar_kwargs)
    for color, lead in zip(colors, np.ndindex(lead_shape)):
        where = {n: v[i] for n, v, i in zip(scan.scan_names, scan.scan_values, lead)}
        label = labels(where) if labels else (_cell_label(scan, where).replace("\n", ", ") or None)
        yy, ee = np.asarray(y[lead], dtype=float), np.asarray(yerr[lead], dtype=float)
        ok = np.isfinite(yy)
        ax.errorbar(np.asarray(x)[ok], yy[ok], yerr=ee[ok], color=color, label=label, **style)
    ax.set_xlabel(xlabel or name)
    ax.set_ylabel(ylabel)
    if created:
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.grid(alpha=0.2)
        if title:
            ax.set_title(title, fontsize=9)
        if lead_shape:
            ax.legend()
    return ax


def _reference_note(scan):
    if scan.reference_measured:
        return f"reference measured ({scan.reference_name})"
    return f"reference assumed at {scan.reference_phase:.2f} rad"


def plot_phase_shift(scan, **kwargs):
    """Phase shift (rad) against the last scan axis, one curve per leading cell."""
    kwargs.setdefault("title", f"{scan.run_id_title}\nRamsey phase shift, {_reference_note(scan)}")
    return plot_vs_last_axis(scan, scan.phase_shift, scan.phase_shift_err,
                             "phase shift (rad)", **kwargs)


def plot_contrast(scan, **kwargs):
    """Fringe contrast (amplitude / offset) against the last scan axis."""
    kwargs.setdefault("title", f"{scan.run_id_title}\nRamsey fringe contrast")
    return plot_vs_last_axis(scan, scan.contrast, scan.contrast_err, "fringe contrast", **kwargs)


def plot_model_fit(mf, ax=None, figsize=(6, 4), cmap="plasma", n_grid=300):
    """Phase shifts against the model's x with the fitted curve, one per other cell."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.8, max(int(np.prod(mf.other_shape)), 1)))
    finite_x = mf.x[np.isfinite(mf.x)]
    grid = np.linspace(finite_x.min(), finite_x.max(), n_grid)
    for color, idx in zip(colors, np.ndindex(mf.other_shape)):
        y, e, ok = mf.cell_data(idx)
        where = ", ".join(f"{n} = {v[i]:.4g}" for n, v, i in
                          zip(mf.other_names, mf.other_values, idx))
        ax.errorbar(mf.x[ok], y[ok], yerr=e[ok], fmt="o", ms=5, capsize=3, color=color,
                    label=where or "data")
        if np.all(np.isfinite(mf.popt[idx])):
            ax.plot(grid, mf.model(grid, *mf.popt[idx]), "--", color=color,
                    label="fit" if mf.other_shape == () else None)
    ax.set_xlabel(mf.x_label)
    ax.set_ylabel("phase shift (rad)" + (" (unwrapped)" if mf.unwrapped else ""))
    ax.grid(alpha=0.2)
    if created:
        if mf.other_shape == ():
            fit = ", ".join(f"{n} {mf.popt[k]:.4g}({mf.perr[k]:.2g})" for k, n in enumerate(mf.names))
            title = f"{mf.scan.run_id_title}\n{fit}, " + rf"$\chi^2_r$ {float(mf.chi2_red):.2f}"
        else:
            title = f"{mf.scan.run_id_title}\nphase model fit along {mf.axis_name}"
        ax.set_title(title, fontsize=9)
        if mf.other_shape:
            ax.legend(fontsize=7)
    return ax
