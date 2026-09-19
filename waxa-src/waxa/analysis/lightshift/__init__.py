"""waxa.analysis.lightshift -- light shifts from the phase jump of a Ramsey fringe.

The sequence is pi/2 - (light pulse of length ``t_pulse``) - pi/2 with the phase
of the second pulse scanned.  Light on the atoms during the gap shifts the qubit
splitting by ``f_ls`` and so moves the fringe by ``2 pi f_ls t_pulse``.  The
fringe phase is read from a cosine fit and compared either with the fringe taken
with the light off (when the run scans an on/off xvar such as ``with_imaging``)
or with an assumed reference phase (when it does not).  That phase is pi for this
sequence, not 0: see ``ramsey_phase.DEFAULT_REFERENCE_PHASE``.

Quick start
-----------
>>> from waxa import atomdata
>>> from waxa.analysis.lightshift import ramsey_light_shift
>>> ls = ramsey_light_shift(atomdata(0))     # works on an AtomdataVault as well
>>> ls.f_lightshift_Hz, ls.f_lightshift_err_Hz   # one entry per scan cell
>>> ls.plot_fringes(); ls.plot()
>>> print(ls.config_line())

Every xvar other than the phase (and the on/off xvar, if there is one) is a scan
axis of the result, so a (compression, phase, detuning) vault gives light shifts
of shape (n_compression, n_detuning).

Modules
-------
:mod:`.ramsey_phase`  ``fit_fringe``, ``ramsey_light_shift``, the result classes,
                      and ``probe_offset_from_midpoint``.
:mod:`.plotting`      the fringe grid and the light shift versus scan plot.
"""

from .ramsey_phase import (
    FringeFit,
    RamseyLightShift,
    cosine_model,
    fit_fringe,
    probe_offset_from_midpoint,
    ramsey_light_shift,
    wrap_phase_shift,
)
from .plotting import plot_fringes, plot_light_shift

__all__ = [
    "FringeFit",
    "RamseyLightShift",
    "cosine_model",
    "fit_fringe",
    "plot_fringes",
    "plot_light_shift",
    "probe_offset_from_midpoint",
    "ramsey_light_shift",
    "wrap_phase_shift",
]
