"""waxa.analysis.ramsey on synthetic fringes with a known phase shift."""

import warnings
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from waxa.analysis.ramsey import fit_fringe, fit_phase_model, ramsey_phase_scan

PHASES = np.linspace(0.0, 3 * np.pi, 21)


def _fringe(dphi, rng, phases=PHASES, n=1.e4, contrast=0.8, noise=400.):
    # nothing in the gap: phase pi; a positive phase shift moves the fitted phase down
    return n * (1 + contrast * np.cos(phases - (np.pi - dphi))) + rng.normal(0, noise, phases.size)


def _fake_ad(xvarnames, xvars, signal, run_id=1):
    return SimpleNamespace(xvarnames=list(xvarnames), xvars=list(xvars), atom_number=signal,
                           params=SimpleNamespace(t_ramsey=5e-6), run_info=SimpleNamespace(run_id=run_id))


def _twist_ad(rng, alpha=1.191, twist=1.35, thetas=(0.2, 0.35, 0.5, 0.65, 0.8), n_rep=2):
    th = np.repeat(np.asarray(thetas), n_rep)                     # repeats on the outer axis
    an = np.stack([_fringe(alpha - twist * np.cos(np.pi * t), rng, contrast=0.7 * np.sin(np.pi * t))
                   for t in th])
    return _fake_ad(["theta_area", "relative_phase"], [th, PHASES], an)


def _twist(scan):
    return scan.fit_model(lambda s, alpha, twist: alpha - twist * s, p0=[1., 1.],
                          x=lambda a: np.cos(np.pi * a), x_label="s_z")


def test_negative_amplitude_is_flipped_into_the_phase():
    x = np.linspace(0, 2 * np.pi, 12)
    fit = fit_fringe(x, 3 - 2 * np.cos(x - 1.0))                  # fitted with amplitude flipped
    assert fit.ok and np.isclose(fit.contrast, 2 / 3) and np.isclose(fit.phase, 1.0 + np.pi)


def test_twist_model_recovers_slope_and_intercept():
    scan = ramsey_phase_scan(_twist_ad(np.random.default_rng(0)))
    assert scan.scan_names == ("theta_area",) and not scan.reference_measured
    mf = _twist(scan)
    assert mf.names == ("alpha", "twist") and mf.n_used == 5 and mf.dof == 3
    assert abs(mf.value("twist") - 1.35) < 4 * mf.error("twist")
    assert abs(mf.value("alpha") - 1.191) < 4 * mf.error("alpha")
    assert mf.error_scale >= 1.0
    mf.plot(); scan.plot(); scan.plot_fringes()
    matplotlib.pyplot.close("all")


def test_twist_error_bar_covers():
    """sqrt(max(1, chi2_red)) scaling: pull std ~1 (0.99 on 3000 seeds, 2026-10-28)."""
    rng = np.random.default_rng(3)
    pulls = []
    for _ in range(300):
        mf = _twist(ramsey_phase_scan(_twist_ad(rng)))
        pulls.append((mf.value("twist") - 1.35) / mf.error("twist"))
    pulls = np.array(pulls)
    assert 0.85 < pulls.std() < 1.2
    assert 0.9 < np.mean(np.abs(pulls) < 2) <= 1.0


def test_reference_arm_with_several_values_is_a_scan_axis():
    rng = np.random.default_rng(4)
    arms = np.array([0, 1, 2, 0, 1, 2])                           # 0 = reference, repeats on this axis
    truth = {0: 0., 1: 0.6, 2: 1.4}
    an = np.stack([_fringe(truth[a] + 0.3, rng) for a in arms])   # 0.3: a common offset the reference removes
    scan = ramsey_phase_scan(_fake_ad(["arm", "relative_phase"], [arms, PHASES], an),
                             reference_xvar="arm", reference_value=0)
    assert scan.reference_measured and scan.reference_name == "arm"
    assert scan.scan_names == ("arm",) and np.array_equal(scan.scan_values[0], [1, 2])
    assert np.allclose(scan.phase_shift, [0.6, 1.4], atol=0.06)


def test_auto_on_off_reference_collapses_the_axis():
    rng = np.random.default_rng(5)
    on = np.array([0, 1, 0, 1])
    an = np.stack([_fringe(0.9 * w + 0.2, rng) for w in on])
    scan = ramsey_phase_scan(_fake_ad(["with_imaging", "relative_phase"], [on, PHASES], an))
    assert scan.reference_measured and scan.scan_shape == ()
    assert abs(float(scan.phase_shift) - 0.9) < 0.06
    no_ref = ramsey_phase_scan(_fake_ad(["with_imaging", "relative_phase"], [on, PHASES], an),
                               reference_xvar=None)
    assert not no_ref.reference_measured and no_ref.scan_names == ("with_imaging",)


def test_reference_value_must_exist():
    an = np.zeros((2, PHASES.size))
    with pytest.raises(ValueError, match="reference value"):
        ramsey_phase_scan(_fake_ad(["arm", "relative_phase"], [np.array([1, 2]), PHASES], an),
                          reference_xvar="arm", reference_value=7)


def test_model_fit_per_cell_of_another_axis_and_exclusions():
    rng = np.random.default_rng(6)
    amps = np.array([0.1, 0.2])
    thetas = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    an = np.empty((amps.size, thetas.size, PHASES.size))
    for i, a in enumerate(amps):
        for j, t in enumerate(thetas):
            an[i, j] = _fringe(10 * a - 5 * a * np.cos(np.pi * t), rng)
    an[1, 2] = np.nan                                             # one cell without data
    ad = _fake_ad(["amp_imaging", "theta_area", "relative_phase"], [amps, thetas, PHASES], an)
    scan = ramsey_phase_scan(ad)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mf = scan.fit_model(lambda s, alpha, twist: alpha - twist * s, axis="theta_area",
                            x=lambda a: np.cos(np.pi * a))
    assert any("1 point(s) left out" in str(w.message) for w in record)
    assert mf.other_names == ("amp_imaging",) and mf.popt.shape == (2, 2)
    assert np.array_equal(mf.n_used, [5, 4])
    assert np.allclose(mf.value("twist"), 5 * amps, atol=4 * mf.error("twist"))
    assert "amp_imaging = 0.1" in mf.summary()


def test_unwrap_follows_a_model_past_the_window():
    rng = np.random.default_rng(7)
    xs = np.linspace(0, 1, 9)
    an = np.stack([_fringe(7.0 * x, rng, noise=100.) for x in xs])   # 0 .. 7 rad: wraps once
    scan = ramsey_phase_scan(_fake_ad(["v", "relative_phase"], [xs, PHASES], an),
                             window_min=-0.5)
    wrapped = fit_phase_model(scan, lambda x, a, b: a + b * x)
    unwrapped = fit_phase_model(scan, lambda x, a, b: a + b * x, unwrap=True)
    assert abs(unwrapped.value("b") - 7.0) < 0.1
    assert abs(wrapped.value("b") - 7.0) > 1.0
