"""Light shift from the phase of a Ramsey fringe (see the package docstring).

The fringe fitting is :mod:`waxa.analysis.ramsey`; this module turns its phase
shift into a frequency, ``phase_shift / (2 pi t_pulse)``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from waxa.analysis.ramsey.phase_scan import (  # noqa: F401  (re-exported)
    DEFAULT_REFERENCE_PHASE,
    DEFAULT_SIGN,
    DEFAULT_WINDOW_MIN,
    MIN_FRINGE_POINTS,
    FringeFit,
    RamseyPhaseScan,
    cosine_model,
    fit_fringe,
    ramsey_phase_scan,
    wrap_phase_shift,
)

# At the imaging midpoint the lit fringe sits near pi/2, where a reference of 0
# gives the right magnitude with the wrong sign and hides the mistake; away from
# the midpoint it is wrong by up to 1 / (2 t_pulse).  See DEFAULT_REFERENCE_PHASE.


def probe_offset_from_midpoint(f_probe, f_up_resonance, half_splitting_Hz):
    """Probe offset (Hz) from the midpoint of the two imaging transitions.

    ``f_probe`` and ``f_up_resonance`` are on the experiment's own detuning axis
    (``frequency_detuned_hf_midpoint``, ``frequency_detuned_hf_f1m1``), which
    increases with optical frequency.  |up> is the LOWER of the two lines, so the
    midpoint sits ``half_splitting_Hz`` above it; take that from kamo at the
    measured field rather than from a second experiment parameter.
    """
    return np.asarray(f_probe, dtype=float) - (float(f_up_resonance) + float(half_splitting_Hz))


@dataclass
class RamseyLightShift(RamseyPhaseScan):
    """Light shifts on the grid of scan axes (every xvar but phase and on/off).

    A :class:`waxa.analysis.ramsey.RamseyPhaseScan` with the light pulse length.
    """

    t_pulse: float = np.nan                # s

    @property
    def f_lightshift_Hz(self):
        return self.phase_shift / (2 * np.pi * self.t_pulse)

    @property
    def f_lightshift_err_Hz(self):
        return self.phase_shift_err / (2 * np.pi * self.t_pulse)

    def config_line(self, param="frequency_lightshift"):
        """The paste-into-config line of a single-cell (calibration) result."""
        if self.fits.size != 1:
            raise ValueError("config_line is for a single light shift; this result has "
                             f"shape {self.scan_shape}.")
        f, err = float(np.ravel(self.f_lightshift_Hz)[0]), float(np.ravel(self.f_lightshift_err_Hz)[0])
        amp = getattr(self.params, "amp_imaging", None)
        note = f", imaging amp {amp}" if amp is not None else ""
        note += f", t_pulse {self.t_pulse * 1e6:.4g} us"
        return (f"self.{param} = {f:.4g}  # Hz, +/- {err:.2g} Hz{note} "
                f"#{', '.join(str(r) for r in self.run_ids)}")

    def plot_fringes(self, **kwargs):
        from .plotting import plot_fringes
        return plot_fringes(self, **kwargs)

    def plot(self, **kwargs):
        from .plotting import plot_light_shift
        return plot_light_shift(self, **kwargs)


def ramsey_light_shift(ad, signal="atom_number", *, phase_xvar=None, onoff_xvar=None,
                       t_pulse=None, reference_phase=DEFAULT_REFERENCE_PHASE,
                       sign=DEFAULT_SIGN, window_min=DEFAULT_WINDOW_MIN,
                       reference_value=None) -> RamseyLightShift:
    """Light shift from the Ramsey phase jump, for an atomdata or an AtomdataVault.

    Parameters
    ----------
    ad : atomdata or AtomdataVault
        Must scan the phase of the second pi/2 pulse on one xvar.  Repeats may sit
        on any axis; all shots sharing a scan cell and a phase are pooled.
    signal : str or array
        Attribute of ``ad`` (dots allowed, e.g. ``"data.apd"``) or a scan-shaped
        array: what oscillates with the phase.
    phase_xvar, onoff_xvar : str or int, optional
        Default: the xvar with "phase" in its name, and an xvar whose values are
        exactly an off (0) and an on (1) state, if there is one.
    t_pulse : float, optional
        Length of the light pulse (s).  Default ``ad.params.t_ramsey``.
    reference_phase : float
        Fringe phase with no light (rad), used only when there is no on/off xvar
        to measure it from.  Default pi (see ``DEFAULT_REFERENCE_PHASE``); an
        error here is a common-mode offset of ``d_phase / (2 pi t_pulse)`` on
        every light shift, so prefer a value measured on an on/off run taken
        with the same Raman settings.
    sign, window_min
        ``phase_shift = wrap(sign * (phase - reference), window_min)``.
    reference_value : optional
        The value of the on/off xvar that is the reference (default: off).

    The general form (any reference arm, models of the phase along a scan axis)
    is :func:`waxa.analysis.ramsey.ramsey_phase_scan`.
    """
    scan = ramsey_phase_scan(ad, signal, phase_xvar=phase_xvar,
                             reference_xvar="auto" if onoff_xvar is None else onoff_xvar,
                             reference_value=reference_value, reference_phase=reference_phase,
                             sign=sign, window_min=window_min, stacklevel=3)
    if t_pulse is None:
        t_pulse = float(ad.params.t_ramsey)
    return RamseyLightShift(**{f.name: getattr(scan, f.name) for f in fields(scan)},
                            t_pulse=float(t_pulse))
