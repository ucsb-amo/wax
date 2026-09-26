"""waxa.analysis -- per-experiment analysis, one subpackage per kind of measurement.

A subpackage here turns an ``atomdata`` or an ``AtomdataVault`` into the physical
quantity the experiment measures, together with the plots of that quantity.  It
is the layer above :mod:`waxa.fitting` (fit models) and :mod:`waxa.plotting`
(generic views of a run): those know nothing about what a scan means, while an
analysis module knows the sequence and returns a result object carrying the
number, its uncertainty, the fits behind it, and the run ids it came from.

Quick start
-----------
>>> from waxa import atomdata
>>> from waxa.analysis.lightshift import ramsey_light_shift
>>> ls = ramsey_light_shift(atomdata(0))
>>> print(ls.config_line())

Nothing is imported eagerly: the subpackages pull in matplotlib and scipy, and
waxa is imported by loky worker subprocesses that need neither.  Import the
subpackage you want by name.

Modules
-------
:mod:`.lightshift`  light shift from the phase jump of a Ramsey fringe
                    (``ramsey_light_shift``).
:mod:`.rabi`        pi time from a Rabi pulse-length scan (``rabi``, ``fit_rabi``;
                    ``python -m waxa.analysis.rabi RUN``).

Adding an experiment: give it its own subpackage (``waxa/analysis/<name>/``)
with the analysis in one module and its plots in ``plotting.py``, a package
docstring with a quick start and a module map, and a test on synthetic data
under ``waxa-src/tests/``.  Follow the shape of :mod:`.lightshift`: a function
that takes ``ad`` and returns a frozen result class whose ``plot*`` methods call
into its own plotting module, so a notebook is a load, a call, and two plots.
"""

__all__ = ["lightshift"]
