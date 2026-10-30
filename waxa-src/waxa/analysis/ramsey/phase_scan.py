"""The phase of a Ramsey fringe, cell by cell, and models of it (see the package docstring)."""

from __future__ import annotations

import inspect
import itertools
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

# Whatever moves the fringe moves it one way, so the phase shift is wrapped into
# the one-sided window [window_min, window_min + 2 pi) instead of (-pi, pi]: a
# large shift keeps its sign, and the margin below zero keeps noise around "no
# shift" from aliasing up to 2 pi.  What remains ambiguous is a whole fringe.
DEFAULT_WINDOW_MIN = -np.pi / 2
# With cosine_model below and the K-machine Raman phase convention, a positive
# differential shift moves the fitted phase DOWN.
DEFAULT_SIGN = -1
# Two pi/2 pulses at zero relative phase add up to a pi pulse and empty the state
# the absorption image counts, so the fringe with nothing in the gap is -cos: its
# phase is pi, not 0.  Measured on six with_imaging on/off runs (71315-71317, 71439,
# 71446, 71455): 2.84 to 3.18 rad, mean 3.03.  It stays pi for any first-pulse
# area theta and second area beta: the final s_z is
# cos(theta) cos(beta) - sin(theta) sin(beta) cos(phi).
DEFAULT_REFERENCE_PHASE = np.pi
MIN_FRINGE_POINTS = 4

_ON = {"1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}


def cosine_model(phase, amplitude, phase_offset, offset):
    return amplitude * np.cos(phase - phase_offset) + offset


def wrap_phase_shift(shift, window_min=DEFAULT_WINDOW_MIN):
    """Wrap a phase (rad) into ``[window_min, window_min + 2 pi)``."""
    return (np.asarray(shift, dtype=float) - window_min) % (2 * np.pi) + window_min


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

    @property
    def contrast_err(self):
        """From the fit covariance of amplitude and offset (their correlation included)."""
        a, o = self.amplitude, self.offset
        if not o:
            return np.nan
        var = (self.cov[0, 0] / o ** 2 + a ** 2 * self.cov[2, 2] / o ** 4
               - 2 * a * self.cov[0, 2] / o ** 3)
        return float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan

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
    cov = np.asarray(pcov, dtype=float)
    if amplitude < 0:
        amplitude, phase = -amplitude, phase + np.pi
        cov = cov * np.outer([-1, 1, 1], [-1, 1, 1])     # amplitude flipped sign
    out.amplitude, out.phase, out.offset = float(amplitude), float(phase % (2 * np.pi)), float(offset)
    out.cov = cov
    out.ok = bool(np.all(np.isfinite(popt)))
    return out


@dataclass
class RamseyPhaseScan:
    """Fringe phase shifts on the grid of scan axes (every xvar but the phase, and
    the reference xvar when it holds only one non-reference value).

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
    sign: int
    window_min: float
    run_ids: Tuple[int, ...]
    params: object = None
    reference_name: Optional[str] = None   # the xvar the reference fringe came from

    @property
    def scan_shape(self):
        return self.fits.shape

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
    def contrast(self):
        """Fringe amplitude / offset per cell (NaN where the fit failed)."""
        return self._from_fits(lambda f: f.contrast)

    @property
    def contrast_err(self):
        return self._from_fits(lambda f: f.contrast_err)

    @property
    def run_id_title(self):
        ids = ", ".join(str(r) for r in self.run_ids)
        return f"Run ID: {ids}" if len(self.run_ids) == 1 else f"Run IDs: {ids}"

    def _from_fits(self, get):
        out = np.full(self.scan_shape, np.nan)
        for idx in np.ndindex(self.scan_shape):
            if self.fits[idx].ok:
                out[idx] = get(self.fits[idx])
        return out

    def cells(self):
        """Iterate ``(index, {scan name: value}, fit, reference fit or None)``."""
        for idx in np.ndindex(self.scan_shape):
            where = {n: v[i] for n, v, i in zip(self.scan_names, self.scan_values, idx)}
            ref = self.reference_fits[idx] if self.reference_measured else None
            yield idx, where, self.fits[idx], ref

    def fit_model(self, model, p0=None, **kwargs) -> "PhaseModelFit":
        """``fit_phase_model(self, model, p0, **kwargs)``."""
        return fit_phase_model(self, model, p0, **kwargs)

    def plot_fringes(self, **kwargs):
        from .plotting import plot_fringes
        return plot_fringes(self, **kwargs)

    def plot(self, **kwargs):
        from .plotting import plot_phase_shift
        return plot_phase_shift(self, **kwargs)


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


def _matches(value, wanted):
    """``value`` is the reference value ``wanted`` (None: the off state)."""
    if wanted is None:
        return _onoff_state(value) == "off"
    try:
        return bool(np.isclose(float(value), float(wanted)))
    except (TypeError, ValueError):
        return _name(value).strip().lower() == _name(wanted).strip().lower()


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

def ramsey_phase_scan(ad, signal="atom_number", *, phase_xvar=None, reference_xvar="auto",
                      reference_value=None, reference_phase=DEFAULT_REFERENCE_PHASE,
                      sign=DEFAULT_SIGN, window_min=DEFAULT_WINDOW_MIN,
                      stacklevel=2) -> RamseyPhaseScan:
    """Phase shift of a Ramsey fringe in every scan cell, for an atomdata or an AtomdataVault.

    ``phase_shift = wrap(sign * (phase - reference), window_min)`` per cell, where
    ``phase`` is the fitted phase of the fringe in the second pulse's phase.

    Parameters
    ----------
    ad : atomdata or AtomdataVault
        Must scan the phase of the second pulse on one xvar.  Repeats may sit on
        any axis; all shots sharing a scan cell and a phase are pooled.
    signal : str or array
        Attribute of ``ad`` (dots allowed, e.g. ``"data.apd"``) or a scan-shaped
        array: what oscillates with the phase.
    phase_xvar : str or int, optional
        Default: the one xvar with "phase" in its name.
    reference_xvar : "auto", str, int or None
        The xvar that holds the reference fringe.  ``"auto"``: an xvar whose values
        are exactly an off (0) and an on (1) state, if there is one.  None: no
        reference fringe, ``reference_phase`` is assumed.
    reference_value : optional
        The value of ``reference_xvar`` that marks the reference fringe (default:
        its off state).  Every other value is compared with it.  With one other
        value the xvar is not a scan axis of the result; with several it is, and
        its values are the non-reference ones.
    reference_phase : float
        Fringe phase with nothing in the gap (rad), used only without a reference
        fringe.  Default pi (see ``DEFAULT_REFERENCE_PHASE``); an error here is a
        common-mode offset on every phase shift.
    sign, window_min
        See ``DEFAULT_SIGN`` and ``DEFAULT_WINDOW_MIN``.
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

    def _guess_reference():
        for i, x in enumerate(xvars):
            if i != i_phase and {_onoff_state(v) for v in np.unique(x)} == {"on", "off"}:
                return i
        return None

    i_phase = _find_axis(names, phase_xvar, "phase", _guess_phase)
    if isinstance(reference_xvar, str) and reference_xvar == "auto":
        i_ref = _guess_reference()
    else:
        i_ref = _find_axis(names, reference_xvar, "reference", lambda: None)
    if i_ref == i_phase:
        raise ValueError("The reference xvar cannot be the phase xvar.")

    is_ref = None
    if i_ref is not None:
        is_ref = np.array([_matches(v, reference_value) for v in xvars[i_ref]])
        if not is_ref.any():
            raise ValueError(f"No value of {names[i_ref]!r} ({np.unique(xvars[i_ref])}) is the "
                             f"reference value {reference_value!r}.")
        if is_ref.all():
            raise ValueError(f"Every value of {names[i_ref]!r} is the reference value; "
                             "there is no fringe to compare with it.")
        ref_unique, ref_inverse = np.unique(xvars[i_ref][~is_ref], return_inverse=True)
        ref_is_scan = ref_unique.size > 1

    phase_unique, phase_inverse = np.unique(xvars[i_phase].astype(float), return_inverse=True)
    scan_axes = [i for i in range(len(names)) if i != i_phase and (i != i_ref or (i_ref is not None and ref_is_scan))]
    scan_unique = []
    for i in scan_axes:
        if i == i_ref:
            scan_unique.append((ref_unique, None))
        else:
            scan_unique.append(np.unique(xvars[i], return_inverse=True))
    scan_shape = tuple(u.size for u, _ in scan_unique)

    def _fit_cell(idx, reference=False):
        take = [np.arange(n) for n in dims]
        for axis, (unique, inverse), u in zip(scan_axes, scan_unique, idx):
            if axis != i_ref:
                take[axis] = np.flatnonzero(inverse == u)
        if i_ref is not None:
            if reference:
                take[i_ref] = np.flatnonzero(is_ref)
            elif ref_is_scan:
                u = idx[scan_axes.index(i_ref)]
                take[i_ref] = np.flatnonzero(~is_ref)[ref_inverse == u]
            else:
                take[i_ref] = np.flatnonzero(~is_ref)
        return _fringe(sig[np.ix_(*take)], i_phase, phase_inverse, phase_unique)

    fits = np.empty(scan_shape, dtype=object)
    refs = np.empty(scan_shape, dtype=object) if i_ref is not None else None
    shift = np.full(scan_shape, np.nan)
    shift_err = np.full(scan_shape, np.nan)
    for idx in itertools.product(*(range(n) for n in scan_shape)):
        fit = fits[idx] = _fit_cell(idx)
        ref_phase, ref_err = float(reference_phase), 0.0
        if refs is not None:
            ref = refs[idx] = _fit_cell(idx, reference=True)
            ref_phase, ref_err = ref.phase, ref.phase_err
        if fit.ok and np.isfinite(ref_phase):
            shift[idx] = wrap_phase_shift(sign * (fit.phase - ref_phase), window_min)
            shift_err[idx] = np.hypot(fit.phase_err, ref_err)

    run_ids = tuple(int(r) for r in np.atleast_1d(ad.run_info.run_id))
    result = RamseyPhaseScan(
        scan_names=tuple(names[i] for i in scan_axes),
        scan_values=tuple(u for u, _ in scan_unique),
        phase_name=names[i_phase], signal_name=signal_name,
        fits=fits, reference_fits=refs, reference_phase=float(reference_phase),
        phase_shift=shift, phase_shift_err=shift_err,
        sign=int(sign), window_min=float(window_min), run_ids=run_ids, params=ad.params,
        reference_name=names[i_ref] if i_ref is not None else None,
    )
    if result.n_ok < result.n_total:
        warnings.warn(f"ramsey phase scan: {result.n_total - result.n_ok} of "
                      f"{result.n_total} fringe fits failed.", stacklevel=stacklevel)
    with np.errstate(invalid="ignore"):
        edge = np.minimum(shift - window_min, window_min + 2 * np.pi - shift)
        railed = result.ok & (edge < 0.05)
    if np.any(railed):
        warnings.warn("ramsey phase scan: a phase shift sits on the edge of the wrap "
                      f"window ({window_min:+.2f} rad); it may be aliased by a whole "
                      "fringe. Move window_min.", stacklevel=stacklevel)
    return result


# ------------------------------------------------------------- phase models

@dataclass
class PhaseModelFit:
    """``model(x, *params)`` fit to the phase shift along one scan axis, once per
    cell of the other scan axes (``other_shape``; ``()`` when there are none).

    ``popt``/``perr`` have shape ``other_shape + (n_params,)``.  ``perr`` is the
    fit's (``absolute_sigma``, from the fringe fits' phase errors) times
    ``error_scale = sqrt(max(1, chi2_red))``: never shrunk below what the fringe
    fits give, inflated when the points scatter more than they say.  On
    synthetic fringes (5 points, linear model) that covers: pull std 0.99,
    70 % within 1 sigma, 95 % within 2 (tests/test_ramsey_phase_scan.py).
    """

    scan: RamseyPhaseScan
    model: Callable
    names: Tuple[str, ...]
    axis_name: str
    axis_values: np.ndarray
    x: np.ndarray                          # the model's abscissa, one per axis value
    x_label: str
    other_names: Tuple[str, ...]
    other_values: Tuple[np.ndarray, ...]
    popt: np.ndarray
    perr: np.ndarray
    pcov: np.ndarray                       # scaled like perr
    chi2_red: np.ndarray
    dof: np.ndarray
    n_used: np.ndarray
    error_scale: np.ndarray
    unwrapped: bool = False

    @property
    def other_shape(self):
        return self.chi2_red.shape

    def _take(self, arr, name):
        k = self.names.index(name)
        out = arr[..., k]
        return float(out) if np.ndim(out) == 0 else out

    def value(self, name):
        return self._take(self.popt, name)

    def error(self, name):
        return self._take(self.perr, name)

    def cell_data(self, idx=()):
        """``(phase_shift, error, used)`` along the axis for one cell of the other axes,
        in axis order (unwrapped when the fit was); ``used`` marks the points fit."""
        sl = self._slice(idx)
        y = np.asarray(self.scan.phase_shift[sl], dtype=float)
        e = np.asarray(self.scan.phase_shift_err[sl], dtype=float)
        ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
        if self.unwrapped and np.count_nonzero(ok) > 1:
            y = y.copy()
            y[ok] = np.unwrap(y[ok])
        return y, e, ok

    def _slice(self, idx):
        idx = tuple(np.atleast_1d(idx)) if idx != () else ()
        axis = self.scan.scan_names.index(self.axis_name)
        sl, it = [], iter(idx)
        for i in range(len(self.scan.scan_names)):
            sl.append(slice(None) if i == axis else next(it))
        return tuple(sl)

    def summary(self):
        lines = []
        for idx in np.ndindex(self.other_shape):
            where = ", ".join(f"{n} = {v[i]:.4g}" for n, v, i in
                              zip(self.other_names, self.other_values, idx))
            fit = ", ".join(f"{n} = {self.popt[idx][k]:.4g} +/- {self.perr[idx][k]:.2g}"
                            for k, n in enumerate(self.names))
            lines.append(f"{where + ': ' if where else ''}{fit}; chi2_red {self.chi2_red[idx]:.2f} "
                         f"({int(self.dof[idx])} dof, errors x {self.error_scale[idx]:.2f}), "
                         f"{int(self.n_used[idx])}/{self.axis_values.size} points")
        return "\n".join(lines)

    def plot(self, **kwargs):
        from .plotting import plot_model_fit
        return plot_model_fit(self, **kwargs)


def fit_phase_model(scan, model, p0=None, *, axis=None, x=None, x_label=None, names=None,
                    unwrap=False) -> PhaseModelFit:
    """Fit ``model(x, *params)`` to the phase shifts along one scan axis.

    Parameters
    ----------
    scan : RamseyPhaseScan (or RamseyLightShift)
    model : callable ``model(x, *params)``
    p0 : sequence, optional
        Starting parameters (default all 1).
    axis : str or int, optional
        The scan axis to fit along (default: the only one).  The fit is repeated
        for every cell of the other scan axes.
    x : array or callable, optional
        The model's abscissa: an array with one entry per value of ``axis`` (in
        ``scan.scan_values`` order), or a function of those values, e.g.
        ``lambda a: np.cos(np.pi * a)``.  Default: the values themselves.
    x_label : str, optional
        For plots (default: the axis name, or ``f(<axis name>)`` for a computed x).
    names : sequence of str, optional
        Parameter names (default: from the model's signature).
    unwrap : bool
        ``np.unwrap`` the phase shifts along the axis (in axis order) before the
        fit, for a model that spans more than the wrap window.  Only safe when
        neighbouring points differ by less than pi.

    Points with a failed fringe fit (NaN) are left out; each cell's count is in
    ``n_used``, and a warning names how many were left out in all.
    """
    if not scan.scan_names:
        raise ValueError("This scan has no scan axis to fit a model along.")
    if axis is None:
        if len(scan.scan_names) != 1:
            raise ValueError(f"Several scan axes {scan.scan_names}; pass axis=.")
        i_axis = 0
    elif isinstance(axis, (int, np.integer)):
        i_axis = int(axis)
    else:
        i_axis = list(scan.scan_names).index(axis)
    axis_name = scan.scan_names[i_axis]
    axis_values = np.asarray(scan.scan_values[i_axis], dtype=float)

    if x is None:
        x_arr, default_label = axis_values, axis_name
    elif callable(x):
        x_arr, default_label = np.asarray(x(axis_values), dtype=float), f"f({axis_name})"
    else:
        x_arr, default_label = np.asarray(x, dtype=float), axis_name
    if x_arr.shape != axis_values.shape:
        raise ValueError(f"x has shape {x_arr.shape}; expected one entry per {axis_name} "
                         f"value, {axis_values.shape}.")

    if names is None:
        params = list(inspect.signature(model).parameters)[1:]
        names = tuple(params)
    names = tuple(names)
    n_par = len(names)
    p0 = np.ones(n_par) if p0 is None else np.asarray(p0, dtype=float)
    if p0.size != n_par:
        raise ValueError(f"p0 has {p0.size} entries for {n_par} parameters {names}.")

    other = [i for i in range(len(scan.scan_names)) if i != i_axis]
    other_shape = tuple(scan.scan_values[i].size for i in other)
    popt = np.full(other_shape + (n_par,), np.nan)
    perr = np.full(other_shape + (n_par,), np.nan)
    pcov = np.full(other_shape + (n_par, n_par), np.nan)
    chi2_red = np.full(other_shape, np.nan)
    dof = np.zeros(other_shape, dtype=int)
    n_used = np.zeros(other_shape, dtype=int)
    scale = np.full(other_shape, np.nan)

    result = PhaseModelFit(
        scan=scan, model=model, names=names, axis_name=axis_name, axis_values=axis_values,
        x=x_arr, x_label=x_label or default_label,
        other_names=tuple(scan.scan_names[i] for i in other),
        other_values=tuple(scan.scan_values[i] for i in other),
        popt=popt, perr=perr, pcov=pcov, chi2_red=chi2_red, dof=dof, n_used=n_used,
        error_scale=scale, unwrapped=bool(unwrap))

    n_left_out, n_unfit = 0, 0
    for idx in np.ndindex(other_shape):
        y, e, ok = result.cell_data(idx)
        n_used[idx] = np.count_nonzero(ok)
        n_left_out += int(ok.size - n_used[idx])
        if n_used[idx] < n_par:
            n_unfit += 1
            continue
        try:
            p, c = curve_fit(model, x_arr[ok], y[ok], p0=p0, sigma=e[ok], absolute_sigma=True,
                             maxfev=20000)
        except (RuntimeError, ValueError):
            n_unfit += 1
            continue
        dof[idx] = n_used[idx] - n_par
        resid = (y[ok] - model(x_arr[ok], *p)) / e[ok]
        chi2_red[idx] = np.sum(resid ** 2) / dof[idx] if dof[idx] > 0 else np.nan
        scale[idx] = np.sqrt(max(1.0, chi2_red[idx])) if np.isfinite(chi2_red[idx]) else 1.0
        popt[idx] = p
        pcov[idx] = np.asarray(c, dtype=float) * scale[idx] ** 2
        perr[idx] = np.sqrt(np.diag(pcov[idx]))
    if n_left_out:
        warnings.warn(f"fit_phase_model: {n_left_out} point(s) left out (no phase shift: "
                      "failed fringe fit or no data in that cell).",
                      stacklevel=2)
    if n_unfit:
        warnings.warn(f"fit_phase_model: {n_unfit} cell(s) not fit (too few points or no "
                      "convergence).", stacklevel=2)
    if np.any(dof[np.isfinite(popt[..., 0])] == 0):
        warnings.warn("fit_phase_model: a fit has 0 degrees of freedom; its errors are the "
                      "fringe fits' alone, not checked against the scatter.", stacklevel=2)
    return result
