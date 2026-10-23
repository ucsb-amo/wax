import numpy as np
import pytest

from waxa.fitting.log_photodiode import fit_log_photodiode


def _synthetic(k=0.5, v0=2.0, p_ref=1e-3, p_dark=1e-7, noise=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    p = np.logspace(-8, -1, 36)
    v = v0 + k * np.log10((p + p_dark) / p_ref) + rng.normal(0, noise, p.size)
    v = np.minimum(v, v0 + k * 1.5)          # compression above 30 mW
    return p, v


def test_recovers_slope_and_intercept_inside_cuts():
    k, v0 = 0.5, 2.0
    p, v = _synthetic(k, v0)
    v_floor = v0 + k * np.log10(1e-7 / 1e-3)
    fit = fit_log_photodiode(p, v, 1e-3, v_floor_cut=v_floor + 2 * k, v_sat_cut=v0 + k * 1.4)
    assert fit.k == pytest.approx(k, abs=5 * fit.k_err + 1e-3)
    assert fit.v_ref == pytest.approx(v0, abs=5 * fit.v_ref_err + 1e-3)
    assert fit.residual_max_frac < 0.03
    reasons = [e[3] for e in fit.excluded]
    assert any("floor" in r for r in reasons) and any("saturation" in r for r in reasons)
    assert fit.n_used + len(fit.excluded) == p.size
    assert fit.v_to_power(fit.power_to_v(2e-3)) == pytest.approx(2e-3)


def test_floor_left_in_shows_in_residuals():
    p, v = _synthetic()
    fit = fit_log_photodiode(p, v, 1e-3)
    assert fit.residual_max_frac > 0.5          # the flattened floor is not hidden


def test_bad_points_excluded_with_reasons():
    p = np.array([1e-4, 1e-3, 0., 1e-2, np.nan, 1e-1])
    v = np.array([1.5, 2.0, 1.0, 2.5, 2.2, 3.0])
    fit = fit_log_photodiode(p, v, 1e-3)
    assert {e[0] for e in fit.excluded} == {2, 4}
    assert fit.n_used == 4
    with pytest.raises(ValueError):
        fit_log_photodiode(p[:3], v[:3], 1e-3)
