"""waxa.analysis.lightshift on synthetic fringes with a known light shift."""

import warnings
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np

from waxa.analysis.lightshift import (fit_fringe, probe_offset_from_midpoint,
                             ramsey_light_shift, wrap_phase_shift)

T_RAMSEY = 5.0e-6
PHASES = np.repeat(np.linspace(0.0, 2 * np.pi, 12), 5)      # repeats on the phase axis


def _fringe(f_ls_Hz, rng, phases=PHASES, n=1000.0, noise=15.0):
    # no light: the fringe is -cos (phase pi); a positive shift moves the phase down
    phi = np.pi - 2 * np.pi * f_ls_Hz * T_RAMSEY
    return n * (1 + 0.8 * np.cos(phases - phi)) + rng.normal(0, noise, phases.size)


def _fake_ad(xvarnames, xvars, atom_number, run_id=1):
    return SimpleNamespace(
        xvarnames=list(xvarnames), xvars=list(xvars), atom_number=atom_number,
        params=SimpleNamespace(t_ramsey=T_RAMSEY, amp_imaging=0.3),
        run_info=SimpleNamespace(run_id=run_id),
    )


def test_fit_fringe_recovers_phase_and_covariance():
    x = np.linspace(0, 2 * np.pi, 12)
    fit = fit_fringe(x, 3 + 2 * np.cos(x - 4.0))
    assert fit.ok
    assert np.allclose(fit.popt, [2.0, 4.0, 3.0], atol=1e-6)
    assert fit.cov.shape == (3, 3)


def test_fit_fringe_reports_failure_instead_of_raising():
    fit = fit_fringe(np.linspace(0, 6, 12), np.full(12, np.nan))
    assert not fit.ok and np.isnan(fit.phase)


def test_wrap_window_is_one_sided():
    assert np.isclose(wrap_phase_shift(3.5, -np.pi / 2), 3.5)          # not aliased to -2.78
    assert np.isclose(wrap_phase_shift(-0.2, -np.pi / 2), -0.2)        # noise about zero survives


def test_detuning_scan_with_assumed_reference():
    rng = np.random.default_rng(0)
    truth = np.array([10e3, 35e3, 80e3, 120e3])
    detunings = np.linspace(-540e6, -475e6, truth.size)
    an = np.stack([_fringe(f, rng) for f in truth], axis=1)            # (phase, detuning)
    ls = ramsey_light_shift(_fake_ad(["relative_phase", "frequency_detuned_hf_midpoint"],
                                     [PHASES, detunings], an))
    assert ls.scan_names == ("frequency_detuned_hf_midpoint",)
    assert not ls.reference_measured and ls.n_ok == ls.n_total == truth.size
    assert np.all(np.abs(ls.f_lightshift_Hz - truth) < 4 * ls.f_lightshift_err_Hz + 200)
    fig, _ = ls.plot_fringes()
    ls.plot(x=probe_offset_from_midpoint(detunings, -568e6, 55.11e6) / 1e6,
            xlabel="probe offset from the midpoint (MHz)")
    matplotlib.pyplot.close("all")


def test_on_off_axis_measures_the_reference():
    rng = np.random.default_rng(1)
    # repeats live on the on/off axis, as in the calibration experiment
    with_imaging = np.array([0, 1, 0, 1, 0, 1])
    phases = np.linspace(0, 2 * np.pi, 12)
    an = np.stack([_fringe(47e3 * w, rng, phases) for w in with_imaging], axis=0)
    ls = ramsey_light_shift(_fake_ad(["with_imaging", "relative_phase"],
                                     [with_imaging, phases], an))
    assert ls.reference_measured and ls.scan_shape == ()
    assert abs(float(ls.f_lightshift_Hz) - 47e3) < 1.5e3
    assert "frequency_lightshift" in ls.config_line() and "#1" in ls.config_line()


def test_nan_padded_vault_cells_are_skipped_not_failed():
    rng = np.random.default_rng(2)
    truth = np.array([[20e3, np.nan, 60e3], [np.nan, 40e3, 90e3]])    # (compression, detuning)
    an = np.full((2, PHASES.size, 3), np.nan)
    for i, j in zip(*np.nonzero(np.isfinite(truth))):
        an[i, :, j] = _fringe(truth[i, j], rng)
    ad = _fake_ad(["v_pd_hf_tweezer_squeeze_power", "relative_phase", "frequency_detuned_hf_midpoint"],
                  [np.array([0.444, 3.94]), PHASES, np.array([-540e6, -532e6, -523e6])], an,
                  run_id=[71160, 71287])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ls = ramsey_light_shift(ad)
    assert not [w for w in record if "failed" in str(w.message)]
    assert ls.scan_shape == (2, 3) and ls.n_ok == ls.n_total == 4
    ok = np.isfinite(truth)
    assert np.array_equal(ls.ok, ok)
    assert np.all(np.abs(ls.f_lightshift_Hz[ok] - truth[ok]) < 2e3)
    assert ls.run_id_title == "Run IDs: 71160, 71287"
