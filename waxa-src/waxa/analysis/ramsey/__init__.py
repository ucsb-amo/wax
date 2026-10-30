"""waxa.analysis.ramsey -- the phase of a Ramsey fringe from a phase scan, and models of it.

The sequence is (pulse) - (something in the gap) - (pulse at a scanned phase).
Whatever acts in the gap rotates the spin about z and moves the fringe; the
fringe phase is read from a cosine fit in every scan cell and compared either
with a reference fringe (one value of an xvar, e.g. ``with_imaging = 0``) or
with an assumed reference phase (pi for the K-machine absorption sequence, at
any first-pulse area: see ``phase_scan.DEFAULT_REFERENCE_PHASE``).  The phase
shift can then be fit along a scan axis with any model.

Quick start
-----------
>>> from waxa import atomdata
>>> from waxa.analysis.ramsey import ramsey_phase_scan
>>> scan = ramsey_phase_scan(atomdata(0))     # works on an AtomdataVault as well
>>> scan.phase_shift, scan.phase_shift_err, scan.contrast   # one entry per scan cell
>>> scan.plot_fringes(); scan.plot()
>>> # e.g. the read twist (check_twist): phase vs s_z = cos(pi theta_area)
>>> mf = scan.fit_model(lambda s, alpha, twist: alpha - twist * s,
...                     x=lambda a: np.cos(np.pi * a), x_label="s_z")
>>> mf.value("twist"), mf.error("twist"); mf.plot()

Every xvar other than the phase (and the reference xvar, unless it holds several
non-reference values) is a scan axis of the result.

Uses: :mod:`waxa.analysis.lightshift` (the phase shift as a frequency).

Modules
-------
:mod:`.phase_scan`  ``fit_fringe``, ``ramsey_phase_scan``, ``fit_phase_model`` and
                    their result classes.
:mod:`.plotting`    the fringe grid, phase shift / contrast versus scan, model fits.
"""

from .phase_scan import (
    DEFAULT_REFERENCE_PHASE,
    DEFAULT_SIGN,
    DEFAULT_WINDOW_MIN,
    FringeFit,
    PhaseModelFit,
    RamseyPhaseScan,
    cosine_model,
    fit_fringe,
    fit_phase_model,
    ramsey_phase_scan,
    wrap_phase_shift,
)
from .plotting import plot_contrast, plot_fringes, plot_model_fit, plot_phase_shift

__all__ = [
    "DEFAULT_REFERENCE_PHASE",
    "DEFAULT_SIGN",
    "DEFAULT_WINDOW_MIN",
    "FringeFit",
    "PhaseModelFit",
    "RamseyPhaseScan",
    "cosine_model",
    "fit_fringe",
    "fit_phase_model",
    "plot_contrast",
    "plot_fringes",
    "plot_model_fit",
    "plot_phase_shift",
    "ramsey_phase_scan",
    "wrap_phase_shift",
]
