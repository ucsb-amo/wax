"""Light shift from the phase of a Ramsey fringe (see the package docstring)."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

# The light pulls the fringe one way, so the phase jump is wrapped into the
# one-sided window [window_min, window_min + 2 pi) instead of (-pi, pi]: a large
# shift keeps its sign, and the margin below zero keeps noise around "no shift"
# from aliasing up to 2 pi.  What remains ambiguous is a whole fringe,
# 1 / t_pulse in frequency.
DEFAULT_WINDOW_MIN = -np.pi / 2
# With cosine_model below and the K-machine Raman phase convention, a positive
# differential shift moves the fitted phase DOWN.
DEFAULT_SIGN = -1
# Two pi/2 pulses at zero relative phase add up to a pi pulse and empty the state
# the absorption image counts, so the fringe with no light is -cos: its phase is
# pi, not 0.  Measured on six with_imaging on/off runs (71315-71317, 71439, 71446,
# 71455): 2.84 to 3.18 rad, mean 3.03.  At the imaging midpoint the lit fringe sits
# near pi/2, where a reference of 0 gives the right magnitude with the wrong sign
# and hides the mistake; away from the midpoint it is wrong by up to 1 / (2 t_pulse).
DEFAULT_REFERENCE_PHASE = np.pi
MIN_FRINGE_POINTS = 4

_ON = {"1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}


def cosine_model(phase, amplitude, phase_offset, offset):
    return amplitude * np.cos(phase - phase_offset) + offset


def wrap_phase_shift(shift, window_min=DEFAULT_WINDOW_MIN):
    """Wrap a phase (rad) into ``[window_min, window_min + 2 pi)``."""
    return (np.asarray(shift, dtype=float) - window_min) % (2 * np.pi) + window_min


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
class FringeFit:
    """One cosine fit, with the repeat-averaged data it was fit to."""

    phase_values: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    amplitude: float = np.nan
    phase: float = np.nan           # rad, in [0, 2 pi)
    offset: float = np.nan
    cov: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan))
    ok: bool = False

    @property
    def popt(self):
        return np.array([self.amplitude, self.phase, self.offset])

    @property
    def phase_err(self):
        return float(np.sqrt(self.cov[1, 1])) if np.isfinite(self.cov[1, 1]) else np.nan

    @property
    def contrast(self):
        return self.amplitude / self.offset if self.offset else np.nan

    def model(self, phase):
        return cosine_model(np.asarray(phase, dtype=float), *self.popt)


def fit_fringe(phase_values, mean, sem=None) -> FringeFit:
    """Fit ``amplitude * cos(phase - phase_offset) + offset``.

    The starting point is the exact linear least-squares solution for
    ``a cos + b sin + c``, so the nonlinear fit only has to supply the covariance.
    The fit is unweighted: a sem from a handful of repeats is too noisy to weight
    by.  A fringe with fewer than four finite points, or a fit that does not
    converge, comes back with ``ok=False`` and NaN parameters.
    """
    x = np.asarray(phase_values, dtype=float)
    y = np.asarray(mean, dtype=float)
    sem = np.zeros_like(y) if sem is None else np.asarray(sem, dtype=float)
    out = FringeFit(phase_values=x, mean=y, sem=sem)

    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < MIN_FRINGE_POINTS:
        return out
    xv, yv = x[valid], y[valid]
    design = np.column_stack([np.cos(xv), np.sin(xv), np.ones_like(xv)])
    (a, b, c), *_ = np.linalg.lstsq(design, yv, rcond=None)
    p0 = [np.hypot(a, b), np.arctan2(b, a), c]
    try:
        popt, pcov = curve_fit(cosine_model, xv, yv, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError):
        return out

    amplitude, phase, offset = popt
    if amplitude < 0:
        amplitude, phase = -amplitude, phase + np.pi
    out.amplitude, out.phase, out.offset = float(amplitude), float(phase % (2 * np.pi)), float(offset)
    out.cov = np.asarray(pcov, dtype=float)
    out.ok = bool(np.all(np.isfinite(popt)))
    return out


@dataclass
class RamseyLightShift:
    """Light shifts on the grid of scan axes (every xvar but phase and on/off).

    The array fields all have shape ``scan_shape``; for a run with no scan axis
    that is ``()`` and they are 0-d.  ``reference_fits`` is None when the
    reference phase was assumed instead of measured.
    """

    scan_names: Tuple[str, ...]
    scan_values: Tuple[np.ndarray, ...]
    phase_name: str
    signal_name: str
    fits: np.ndarray                       # object array of FringeFit
    reference_fits: Optional[np.ndarray]
    reference_phase: float
    phase_shift: np.ndarray                # rad
    phase_shift_err: np.ndarray
    t_pulse: float                         # s
    sign: int
    window_min: float
    run_ids: Tuple[int, ...]
    params: object = None

    @property
    def scan_shape(self):
        return self.fits.shape

    @property
    def f_lightshift_Hz(self):
        return self.phase_shift / (2 * np.pi * self.t_pulse)

    @property
    def f_lightshift_err_Hz(self):
        return self.phase_shift_err / (2 * np.pi * self.t_pulse)

    @property
    def ok(self):
        return np.isfinite(self.phase_shift)

    @property
    def n_ok(self):
        return int(np.count_nonzero(self.ok))

    @property
    def n_total(self):
        """Cells that held a fringe at all (NaN-padded vault cells do not count)."""
        return int(sum(np.any(np.isfinite(f.mean)) for f in self.fits.ravel()))

    @property
    def reference_measured(self):
        return self.reference_fits is not None

    @property
    def run_id_title(self):
        ids = ", ".join(str(r) for r in self.run_ids)
        return f"Run ID: {ids}" if len(self.run_ids) == 1 else f"Run IDs: {ids}"

    def cells(self):
        """Iterate ``(index, {scan name: value}, fit, reference fit or None)``."""
        for idx in np.ndindex(self.scan_shape):
            where = {n: v[i] for n, v, i in zip(self.scan_names, self.scan_values, idx)}
            ref = self.reference_fits[idx] if self.reference_measured else None
            yield idx, where, self.fits[idx], ref

    def config_line(self, param="omega_lightshift"):
        """The paste-into-config line of a single-cell (calibration) result."""
        if self.fits.size != 1:
            raise ValueError("config_line is for a single light shift; this result has "
                             f"shape {self.scan_shape}.")
        f, err = float(np.ravel(self.f_lightshift_Hz)[0]), float(np.ravel(self.f_lightshift_err_Hz)[0])
        amp = getattr(self.params, "amp_imaging", None)
        note = f", imaging amp {amp}" if amp is not None else ""
        return (f"self.p.{param} = 2 * np.pi * {f:.4g}  # rad/s, +/- {err:.2g} Hz{note} "
                f"#{', '.join(str(r) for r in self.run_ids)}")

    def plot_fringes(self, **kwargs):
        from .plotting import plot_fringes
        return plot_fringes(self, **kwargs)

    def plot(self, **kwargs):
        from .plotting import plot_light_shift
        return plot_light_shift(self, **kwargs)


# ------------------------------------------------------------------ helpers

def _name(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _onoff_state(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number is not None:
        if np.isclose(number, 0):
            return "off"
        if np.isclose(number, 1):
            return "on"
        return None
    text = str(value).strip().lower()
    return "on" if text in _ON else "off" if text in _OFF else None


def _find_axis(names, wanted, what, guess):
    if wanted is not None:
        if isinstance(wanted, (int, np.integer)):
            return int(wanted)
        if wanted not in names:
            raise KeyError(f"{what} xvar {wanted!r} is not one of {names}.")
        return names.index(wanted)
    return guess()


def _resolve_signal(ad, signal):
    if isinstance(signal, str):
        obj = ad
        for part in signal.split("."):
            obj = getattr(obj, part)
        return np.asarray(obj, dtype=float), signal
    return np.asarray(signal, dtype=float), "signal"


def _fringe(sub, phase_axis, phase_inverse, phase_unique):
    """Pool every shot in ``sub`` by phase value -> FringeFit."""
    shots = np.moveaxis(sub, phase_axis, -1).reshape(-1, sub.shape[phase_axis])
    mean = np.full(phase_unique.size, np.nan)
    sem = np.zeros(phase_unique.size)
    for k in range(phase_unique.size):
        values = shots[:, phase_inverse == k].ravel()
        values = values[np.isfinite(values)]
        if values.size:
            mean[k] = values.mean()
        if values.size > 1:
            sem[k] = values.std(ddof=1) / np.sqrt(values.size)
    return fit_fringe(phase_unique, mean, sem)


# --------------------------------------------------------------------- main

def ramsey_light_shift(ad, signal="atom_number", *, phase_xvar=None, onoff_xvar=None,
                       t_pulse=None, reference_phase=DEFAULT_REFERENCE_PHASE,
                       sign=DEFAULT_SIGN,
                       window_min=DEFAULT_WINDOW_MIN) -> RamseyLightShift:
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
    """
    names = [_name(n) for n in ad.xvarnames]
    xvars = [np.asarray(x) for x in ad.xvars]
    sig, signal_name = _resolve_signal(ad, signal)
    dims = tuple(len(x) for x in xvars)
    if sig.shape != dims:
        raise ValueError(f"signal has shape {sig.shape}; expected one value per shot, {dims}.")

    def _guess_phase():
        hits = [i for i, n in enumerate(names) if "phase" in n.lower()]
        if len(hits) != 1:
            raise ValueError(f"Could not pick the phase xvar out of {names}; pass phase_xvar=.")
        return hits[0]

    def _guess_onoff():
        for i, x in enumerate(xvars):
            if i != i_phase and {_onoff_state(v) for v in np.unique(x)} == {"on", "off"}:
                return i
        return None

    i_phase = _find_axis(names, phase_xvar, "phase", _guess_phase)
    i_onoff = _find_axis(names, onoff_xvar, "on/off", _guess_onoff)
    scan_axes = [i for i in range(len(names)) if i not in (i_phase, i_onoff)]

    phase_unique, phase_inverse = np.unique(xvars[i_phase].astype(float), return_inverse=True)
    scan_unique = [np.unique(xvars[i], return_inverse=True) for i in scan_axes]
    scan_shape = tuple(u.size for u, _ in scan_unique)
    if i_onoff is not None:
        states = np.array([_onoff_state(v) for v in xvars[i_onoff]])

    def _fit_cell(idx, state=None):
        take = [np.arange(n) for n in dims]
        for axis, (_, inverse), u in zip(scan_axes, scan_unique, idx):
            take[axis] = np.flatnonzero(inverse == u)
        if state is not None:
            take[i_onoff] = np.flatnonzero(states == state)
        return _fringe(sig[np.ix_(*take)], i_phase, phase_inverse, phase_unique)

    fits = np.empty(scan_shape, dtype=object)
    refs = np.empty(scan_shape, dtype=object) if i_onoff is not None else None
    shift = np.full(scan_shape, np.nan)
    shift_err = np.full(scan_shape, np.nan)
    for idx in itertools.product(*(range(n) for n in scan_shape)):
        fit = fits[idx] = _fit_cell(idx, "on" if i_onoff is not None else None)
        ref_phase, ref_err = float(reference_phase), 0.0
        if refs is not None:
            ref = refs[idx] = _fit_cell(idx, "off")
            ref_phase, ref_err = ref.phase, ref.phase_err
        if fit.ok and np.isfinite(ref_phase):
            shift[idx] = wrap_phase_shift(sign * (fit.phase - ref_phase), window_min)
            shift_err[idx] = np.hypot(fit.phase_err, ref_err)

    if t_pulse is None:
        t_pulse = float(ad.params.t_ramsey)
    run_ids = tuple(int(r) for r in np.atleast_1d(ad.run_info.run_id))
    result = RamseyLightShift(
        scan_names=tuple(names[i] for i in scan_axes),
        scan_values=tuple(u for u, _ in scan_unique),
        phase_name=names[i_phase], signal_name=signal_name,
        fits=fits, reference_fits=refs, reference_phase=float(reference_phase),
        phase_shift=shift, phase_shift_err=shift_err, t_pulse=float(t_pulse),
        sign=int(sign), window_min=float(window_min), run_ids=run_ids, params=ad.params,
    )
    if result.n_ok < result.n_total:
        warnings.warn(f"ramsey_light_shift: {result.n_total - result.n_ok} of "
                      f"{result.n_total} fringe fits failed.", stacklevel=2)
    with np.errstate(invalid="ignore"):
        edge = np.minimum(shift - window_min, window_min + 2 * np.pi - shift)
        railed = result.ok & (edge < 0.05)
    if np.any(railed):
        warnings.warn("ramsey_light_shift: a phase shift sits on the edge of the wrap "
                      f"window ({window_min:+.2f} rad); it may be aliased by a whole "
                      "fringe. Move window_min.", stacklevel=2)
    return result
