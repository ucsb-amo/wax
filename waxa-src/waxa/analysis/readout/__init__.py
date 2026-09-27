"""waxa.analysis.readout -- calibrate a spin-state readout signal against a Rabi flop.

A drive pulse of scanned length rotates a spin prepared in ``S_z = +1``; one or more
readout pulses per shot (an integrated APD pulse, say) then measure it.  This package
turns such a scan into the readout calibration: the signal at ``S_z = +1`` and ``-1``,
the ``S_z`` response shape (linear, or the quadratic "midpoint remap"), the single-pulse
noise at each endpoint, whether several runs agree, and -- from scope traces of the
same pulses -- the photon number of one pulse.

It is the importable form of ``k-jam/analysis/artisinal/apd_pulse_analysis_interpolate.ipynb``
(the S_z response fit and the scope integration are ported cell by cell), extended with a
fitted flop, per-pulse and per-run endpoints, single-pulse noise and run-agreement tests.
Machine-specific names (which params hold the calibration, the scope gain chain) live
in the caller, e.g. ``kexp.analysis.apd_state_mapping``.

Quick start
-----------
>>> from waxa.analysis.readout import calibrate_readout
>>> cal = calibrate_readout(t_pulse, apd, run=run_of_shot, t_pi=t_pi, pulses_used=[1, 2, 3, 4])
>>> cal                      # summary: endpoints, noise, run agreement, proposal
>>> cal.plot_pulses(); cal.plot_summary()
>>> cal.to_dict()            # JSON-able

Modules
-------
:mod:`.sz_response`   the notebook's S_z helpers and quadratic response fit
                      (``fit_sz_response``, ``sz_response``, ``pulse_time_to_angle_and_sz``).
:mod:`.scope_pulses`  ``integrate_scope_pulses``, ``find_pulse_edges``, ``ScopePulses``.
:mod:`.calibration`   ``calibrate_readout``, ``ReadoutCalibration``, ``Endpoints``,
                      ``NoiseEstimate``, ``flop_coordinate``.
:mod:`.plotting`      ``plot_pulses``, ``plot_summary``.
"""

from .sz_response import (collapse_by_sz, fit_sz_response, group_by_x, interp_vs_sz,
                          pulse_time_to_angle_and_sz, sz_response)
from .scope_pulses import DEFAULT_WINDOWS, ScopePulses, check_timing, find_pulse_edges, integrate_scope_pulses
from .calibration import (DEFAULT_ENDPOINT_WINDOW, POOLING_P_LIMIT, Endpoints, NoiseEstimate,
                          ReadoutCalibration, calibrate_readout, flop_coordinate)
from .plotting import plot_pulses, plot_summary

__all__ = [
    "DEFAULT_ENDPOINT_WINDOW", "DEFAULT_WINDOWS", "Endpoints", "NoiseEstimate", "POOLING_P_LIMIT",
    "ReadoutCalibration", "ScopePulses", "calibrate_readout", "check_timing", "collapse_by_sz", "find_pulse_edges",
    "fit_sz_response", "flop_coordinate", "group_by_x", "integrate_scope_pulses", "interp_vs_sz",
    "plot_pulses", "plot_summary", "pulse_time_to_angle_and_sz", "sz_response",
]
