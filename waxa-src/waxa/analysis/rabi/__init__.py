"""waxa.analysis.rabi -- Rabi oscillations: the pi time from a pulse-length scan.

Replaces the fitting inside :func:`waxa.plotting.rabi_oscillation` (which stays for
old notebooks) with a fit that starts from a frequency scan, cannot rail on a decay
bound (the decay is a rate, and zero means "not resolved"), reports the commanded pi
time separately from pi/Omega, handles repeats in any order, picks the decay model
by AICc, and can fit several runs jointly.  See :mod:`.rabi_fit` for the design.

Quick start
-----------
>>> from waxa.analysis.rabi import rabi
>>> fit = rabi(83092)                 # run id, atomdata, or a list of either (joint fit)
>>> fit                               # prints the summary in a notebook
>>> fit.t_pi, fit.t_pi_err            # commanded pi-pulse length (s) -- the config number
>>> fit.compare(ad.p.t_raman_pi_pulse)
>>> fit.plot(reference=ad.p.t_raman_pi_pulse)
>>> print(fit.config_line("t_raman_pi_pulse"))

Is the signal linear in the spin state?  Fit a monotonic cubic map with the flop:

>>> from waxa.analysis.rabi import linearize
>>> m = linearize(83092)              # needs repeats; m(y) remaps any signal array
>>> m; m.plot()

On arrays: ``fit_rabi(t, y)`` / ``linearize_signal(t, y)`` (lists of arrays for a joint fit).
Command line: ``python -m waxa.analysis.rabi 83092 [83091 ...] --compare t_raman_pi_pulse``.

Modules
-------
:mod:`.rabi_fit`   ``fit_rabi``, ``rabi``, ``RabiFit``, ``rabi_model``.
:mod:`.linearize`  ``linearize``, ``linearize_signal``, ``SignalMapping``.
:mod:`.plotting`   ``plot_rabi``, ``plot_linearization``.
"""

from .rabi_fit import MODELS, RabiFit, envelope, fit_rabi, rabi, rabi_model
from .linearize import SignalMapping, linearize, linearize_signal
from .plotting import plot_linearization, plot_rabi

__all__ = ["MODELS", "RabiFit", "SignalMapping", "envelope", "fit_rabi", "linearize", "linearize_signal",
           "plot_linearization", "plot_rabi", "rabi", "rabi_model"]
