"""Signal versus spin state ``S_z``: the quadratic response fit behind the feedback readout.

Ported from ``k-jam/analysis/artisinal/apd_pulse_analysis_interpolate.ipynb`` (cells 6
and 7, as of commit e22b200, 2026-09-10) with the notebook globals turned into
arguments.  The math is unchanged:

    y(S_z) = A + B S_z + C S_z^2

which is exactly the forward model of ``Feedback.expected_photon_fraction``
(``kexp/base/feedback.py``),

    p1(S_z) = (1 + S_z)/2 + delta (1 - S_z^2),      y = y_down + y_range p1,

with ``delta = midpoint - 1/2``, so ``y_range = 2B``, ``midpoint = 1/2 - C/y_range``,
``y_down = y(-1)`` and ``y_up = y(+1)``.  ``degree=1`` forces the symmetric linear
response (``midpoint = 1/2`` exactly).

Nothing here knows about APDs, runs or parameter names: ``y`` can be any signal and
``sz`` any spin coordinate.  Covariances are scaled by the reduced chi2, as
``np.polyfit(..., cov=True)`` does, so they are meaningful whether the error bars
passed in are absolute sigmas or only relative weights.
"""

from __future__ import annotations

import numpy as np

SZ_FIT_TINY = 1e-15
DEFAULT_ENDPOINT_TOL = 2e-2


# ---------------------------------------------------------------- mapping (cell 6)
def pulse_time_to_angle_and_sz(t_pulse, t_pi, t_offset=0.0):
    """Pulse time -> (Bloch angle, S_z) for an ideal resonant flop, folded into one
    Rabi half-cycle (notebook ``raman_time_to_angle_and_sz``).

    The effective pulse is ``t - t_offset`` clipped at zero; ``angle = pi t_eff / t_pi``
    (0 = all up, pi = all down) and ``S_z = sin(pi/2 - angle) = cos(angle)``.
    """
    t_pi = float(t_pi)
    if t_pi <= 0:
        raise ValueError("t_pi must be positive")
    t = np.clip(np.asarray(t_pulse, dtype=float) - float(t_offset), 0.0, None)
    t_mod = np.mod(t, 2.0 * t_pi)
    t_fold = np.minimum(t_mod, 2.0 * t_pi - t_mod)
    angle = np.pi * t_fold / t_pi
    return angle, np.sin(0.5 * np.pi - angle)


def group_by_x(xvals, yvals):
    """Mean / std (ddof=1, 0 for a single value) / count of y for each unique x, in
    first-appearance order."""
    xvals = np.asarray(xvals, dtype=float)
    yvals = np.asarray(yvals, dtype=float)
    x_unique, first_idx = np.unique(xvals, return_index=True)
    x_unique = x_unique[np.argsort(first_idx)]
    mean_y = np.zeros_like(x_unique, dtype=float)
    std_y = np.zeros_like(x_unique, dtype=float)
    count = np.zeros(x_unique.size, dtype=int)
    for i, xv in enumerate(x_unique):
        vals = yvals[np.isclose(xvals, xv, rtol=0, atol=1e-15)]
        mean_y[i] = np.mean(vals)
        std_y[i] = np.std(vals, ddof=1) if vals.size > 1 else 0.0
        count[i] = vals.size
    return x_unique, mean_y, std_y, count


def collapse_by_sz(sz_vals, mean_vals, std_vals, counts=None, decimals=12):
    """Merge points that land on the same S_z (pulse times mirrored about the Rabi
    cycle).  Means are averaged, stds combined in quadrature-mean, counts summed;
    output sorted by S_z."""
    sz_vals = np.asarray(sz_vals, dtype=float)
    mean_vals = np.asarray(mean_vals, dtype=float)
    std_vals = np.asarray(std_vals, dtype=float)
    if not (sz_vals.shape == mean_vals.shape == std_vals.shape):
        raise ValueError("collapse_by_sz expects arrays with identical shapes.")
    counts = np.ones(sz_vals.size, dtype=int) if counts is None else np.asarray(counts, dtype=int)
    uniq_key, inv = np.unique(np.round(sz_vals, decimals=decimals), return_inverse=True)
    sz_out = np.zeros(uniq_key.size)
    mean_out = np.zeros(uniq_key.size)
    std_out = np.zeros(uniq_key.size)
    count_out = np.zeros(uniq_key.size, dtype=int)
    for i in range(uniq_key.size):
        m = inv == i
        sz_out[i] = np.mean(sz_vals[m])
        mean_out[i] = np.mean(mean_vals[m])
        std_out[i] = np.sqrt(np.mean(std_vals[m] ** 2))
        count_out[i] = int(np.sum(counts[m]))
    order = np.argsort(sz_out)
    return sz_out[order], mean_out[order], std_out[order], count_out[order]


def interp_vs_sz(sz_axis, y_axis, sz_targets, endpoint_tol=DEFAULT_ENDPOINT_TOL, strict=True):
    """Linear interpolation of y onto ``sz_targets``, clipped at the measured S_z range.

    Targets outside the measured range by more than ``endpoint_tol`` raise when
    ``strict`` (clipping there is silent extrapolation)."""
    sz_axis = np.asarray(sz_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    sz_targets = np.atleast_1d(np.asarray(sz_targets, dtype=float))
    order = np.argsort(sz_axis)
    sz_sorted, y_sorted = sz_axis[order], y_axis[order]
    overshoot = np.maximum(sz_sorted[0] - sz_targets, sz_targets - sz_sorted[-1])
    if strict and np.any(overshoot > endpoint_tol):
        raise ValueError(
            f"Interpolation targets {sz_targets} lie outside the measured S_z range "
            f"[{sz_sorted[0]:.3f}, {sz_sorted[-1]:.3f}] by up to {overshoot.max():.3g} "
            f"(tol {endpoint_tol:.3g}).  Use endpoint source 'fit' to extrapolate with the model.")
    return np.interp(np.clip(sz_targets, sz_sorted[0], sz_sorted[-1]), sz_sorted, y_sorted)


def weighted_rms(residual, weights=None):
    residual = np.asarray(residual, dtype=float)
    if weights is None or np.all(np.asarray(weights) == 0):
        return float(np.sqrt(np.mean(residual ** 2)))
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(w * residual ** 2) / np.sum(w)))


# --------------------------------------------------------------- the fit (cell 7)
def weighted_lstsq(design, target, wsq=None):
    """Weighted least squares -> (params, covariance scaled by reduced chi2, residual)."""
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    sw = np.ones(target.size) if wsq is None else np.sqrt(np.asarray(wsq, dtype=float))
    params, *_ = np.linalg.lstsq(design * sw[:, None], target * sw, rcond=None)
    resid = target - design @ params
    dof = target.size - params.size
    xtx_inv = np.linalg.pinv((design * sw[:, None]).T @ (design * sw[:, None]))
    if dof > 0:
        cov = xtx_inv * (float(np.sum((resid * sw) ** 2)) / dof)
    else:
        cov = np.full_like(xtx_inv, np.nan)
    return params, cov, resid


def fit_sz_response(sz, y, yerr=None, degree=2, pin_endpoints=False, y_down_pin=None, y_up_pin=None):
    """Fit ``y(S_z)`` and translate the coefficients into feedback-model parameters.

    ``degree=2`` leaves the midpoint remap free; ``degree=1`` forces the symmetric
    linear response.  ``pin_endpoints=True`` clamps ``y(-1)``, ``y(+1)`` to the pins and
    fits only the curvature (comparison only).  ``yerr`` (1-sigma) weights the fit;
    it is ignored (unweighted fit) unless every entry is finite and positive.

    Returns a dict: ``y_up``, ``y_down``, ``y_range``, their 1-sigma errors
    ``std_y_up`` / ``std_y_down``, ``midpoint_fraction`` +/- ``std_midpoint_fraction``,
    ``monotonic`` (``|midpoint - 1/2| < 1/4``), ``coeff`` ``(C, B, A)`` in np.polyval
    order, ``cov``, ``residual``, ``rms``, ``chi2``, ``dof``, ``predict``.
    """
    degree = int(degree)
    sz = np.asarray(sz, dtype=float)
    y = np.asarray(y, dtype=float)
    if sz.shape != y.shape:
        raise ValueError("fit_sz_response expects sz and y with the same shape.")
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2.")
    if sz.size < degree + 1:
        raise ValueError(f"Need >= {degree + 1} S_z points for a degree-{degree} fit.")

    weighted, wsq = False, None
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        if np.all(np.isfinite(yerr)) and np.all(yerr > 0):
            wsq = 1.0 / yerr ** 2
            weighted = True

    if pin_endpoints:
        if degree != 2:
            raise ValueError("Endpoint pinning is only defined for the quadratic model.")
        if y_down_pin is None or y_up_pin is None:
            raise ValueError("pin_endpoints=True needs y_down_pin and y_up_pin.")
        y_d, y_u = float(y_down_pin), float(y_up_pin)
        B = 0.5 * (y_u - y_d)
        line = 0.5 * (y_u + y_d) + B * sz
        basis = sz ** 2 - 1.0
        params, cov_p, resid = weighted_lstsq(basis[:, None], y - line, wsq)
        C = float(params[0])
        coeff = np.array([C, B, 0.5 * (y_u + y_d) - C])
        cov = np.zeros((3, 3))
        cov[0, 0] = cov_p[0, 0]
        n_par = 1
    else:
        design = np.vander(sz, degree + 1)
        params, cov_p, resid = weighted_lstsq(design, y, wsq)
        if degree == 2:
            coeff, cov = np.asarray(params, dtype=float), cov_p
        else:
            coeff = np.array([0.0, params[0], params[1]])
            cov = np.zeros((3, 3))
            cov[1:, 1:] = cov_p
        n_par = degree + 1

    C, B, A = (float(v) for v in coeff)
    y_range = 2.0 * B
    if abs(y_range) < SZ_FIT_TINY:
        raise ValueError("Degenerate fit: the fitted S_z response range is zero.")
    midpoint = 0.5 - C / y_range
    delta = midpoint - 0.5

    def _sigma(grad):
        if cov is None or not np.all(np.isfinite(cov)):
            return np.nan
        g = np.asarray(grad, dtype=float)
        return float(np.sqrt(max(float(g @ cov @ g), 0.0)))

    residual = y - np.polyval(coeff, sz)
    chi2 = float(np.sum(residual ** 2 * (wsq if wsq is not None else 1.0)))
    return {
        "degree": degree,
        "pinned": bool(pin_endpoints),
        "weighted": weighted,
        "n_points": int(sz.size),
        "dof": int(sz.size - n_par),
        "coeff": coeff,
        "cov": cov,
        "y_down": A - B + C,
        "y_up": A + B + C,
        "y_range": y_range,
        "std_y_down": _sigma([1.0, -1.0, 1.0]),
        "std_y_up": _sigma([1.0, 1.0, 1.0]),
        "std_y_range": _sigma([0.0, 2.0, 0.0]),
        "midpoint_fraction": midpoint,
        "std_midpoint_fraction": _sigma([-1.0 / y_range, 2.0 * C / y_range ** 2, 0.0]),
        # p1 is monotonic on [-1, 1] only for |delta| < 0.25 (midpoint in (0.25, 0.75))
        "monotonic": bool(abs(delta) < 0.25),
        "slope_at_minus1": 0.5 + 2.0 * delta,
        "slope_at_plus1": 0.5 - 2.0 * delta,
        "residual": residual,
        "rms": weighted_rms(residual, wsq),
        "chi2": chi2,
        "predict": (lambda s, _c=coeff: np.polyval(_c, np.asarray(s, dtype=float))),
    }


def sz_response(sz, y, yerr=None, degree=2, source="fit", pinned_comparison=True,
                endpoint_tol=DEFAULT_ENDPOINT_TOL):
    """Fit ``y(S_z)`` and pick endpoints / midpoint from ``source`` ('fit' | 'interp').

    Always returns the free fit AND the interpolated endpoints, whichever is used."""
    if source not in ("fit", "interp"):
        raise ValueError("source must be 'fit' or 'interp'.")
    degree = int(degree)
    sz = np.asarray(sz, dtype=float)
    y = np.asarray(y, dtype=float)
    fit = fit_sz_response(sz, y, yerr=yerr, degree=degree)
    y_down_interp, y_up_interp = (float(v) for v in interp_vs_sz(
        sz, y, [-1.0, 1.0], endpoint_tol=endpoint_tol, strict=(source == "interp")))
    fit_pinned = None
    if pinned_comparison and degree == 2 and np.any(np.abs(sz) < 1.0 - 1e-9):
        fit_pinned = fit_sz_response(sz, y, yerr=yerr, degree=2, pin_endpoints=True,
                                     y_down_pin=y_down_interp, y_up_pin=y_up_interp)
    if source == "fit":
        y_down, y_up = fit["y_down"], fit["y_up"]
        midpoint, std_midpoint = fit["midpoint_fraction"], fit["std_midpoint_fraction"]
    else:
        y_down, y_up = y_down_interp, y_up_interp
        y_range_interp = y_up - y_down
        y_at_zero = float(interp_vs_sz(sz, y, [0.0], strict=False)[0])
        midpoint = ((y_at_zero - y_down) / y_range_interp
                    if abs(y_range_interp) > SZ_FIT_TINY else np.nan)
        std_midpoint = np.nan
    return {
        "source": source,
        "sz": sz,
        "y": y,
        "yerr": None if yerr is None else np.asarray(yerr, dtype=float),
        "fit": fit,
        "fit_pinned": fit_pinned,
        "y_down": float(y_down),
        "y_up": float(y_up),
        "y_range": float(y_up - y_down),
        "midpoint_fraction": float(midpoint),
        "std_midpoint_fraction": float(std_midpoint),
        "y_down_interp": y_down_interp,
        "y_up_interp": y_up_interp,
    }
