"""Plots for :class:`waxa.analysis.lightshift.RamseyLightShift`.

The general plots are :mod:`waxa.analysis.ramsey.plotting`; these add the light
shift in kHz.  Every point of :func:`plot_light_shift` is a panel of
:func:`plot_fringes`: both read the same fit objects.
"""

from waxa.analysis.ramsey import plotting as _ramsey_plotting
from waxa.analysis.ramsey.plotting import COLOR_FRINGE as COLOR_LIGHT, COLOR_REFERENCE  # noqa: F401


def _annotate(ls, idx, fit):
    if not fit.ok:
        return "fit failed"
    f, err = ls.f_lightshift_Hz[idx], ls.f_lightshift_err_Hz[idx]
    return (rf"$\Delta\phi$ = {ls.phase_shift[idx]:.2f}({ls.phase_shift_err[idx]:.2f}) rad, "
            rf"$f_{{ls}}$ = {f / 1e3:.1f}({err / 1e3:.1f}) kHz")


def plot_fringes(ls, ncols=4, figsize=None):
    """One panel per scan cell: repeat means (sem bars), the fit, the reference."""
    return _ramsey_plotting.plot_fringes(ls, ncols=ncols, figsize=figsize,
                                         labels=("light on", "light off"), annotate=_annotate,
                                         subtitle=f"pulse {ls.t_pulse * 1e6:.3g} µs")


def plot_light_shift(ls, ax=None, x=None, xlabel=None, figsize=(6, 4), cmap="plasma",
                     labels=None, **errorbar_kwargs):
    """Light shift (kHz) against the last scan axis, one curve per leading cell.

    ``x`` replaces the last scan axis's values (same length), e.g. with the probe
    offset from the midpoint; ``labels`` formats a leading cell's dict of
    ``{scan name: value}`` into a legend entry.
    """
    title = (f"{ls.run_id_title}\nlight shift from the Ramsey phase jump, "
             f"pulse {ls.t_pulse * 1e6:.3g} µs"
             + ("" if ls.reference_measured
                else f", reference assumed at {ls.reference_phase:.2f} rad"))
    return _ramsey_plotting.plot_vs_last_axis(
        ls, ls.f_lightshift_Hz / 1e3, ls.f_lightshift_err_Hz / 1e3, "light shift (kHz)",
        ax=ax, x=x, xlabel=xlabel, figsize=figsize, cmap=cmap, labels=labels, title=title,
        **errorbar_kwargs)
