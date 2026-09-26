"""waxa.analysis.rabi on synthetic Rabi flops with a known pi time.  Writes no files."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from waxa.analysis.rabi import fit_rabi, rabi, rabi_model
from waxa.analysis.rabi.__main__ import build_parser

F_RABI = 80e3
OMEGA = 2 * np.pi * F_RABI
T = np.linspace(0, 30e-6, 31)


def _flop(rng, t, t_dead=0.0, gamma=0.0, model="exp", n0=1500.0, contrast=0.9, noise=0.05, start_full=True,
          mult=True):
    phi = -OMEGA * t_dead + (0.0 if start_full else np.pi)
    y = rabi_model(t, OMEGA, phi, n0, 2 * contrast * n0, gamma, model)
    sd = noise * (y if mult else n0)
    return y + rng.normal(0, 1, t.size) * sd


def _shuffled_repeats(rng, nrep=3):
    t = np.repeat(T, nrep)
    rng.shuffle(t)
    return t


def _within(fit, value, n=4):
    return abs(fit.t_pi - value) < n * fit.t_pi_err


def test_repeats_shuffled_with_decay_and_dead_time():
    rng = np.random.default_rng(1)
    t = _shuffled_repeats(rng)
    fit = fit_rabi(t, _flop(rng, t, t_dead=0.3e-6, gamma=1 / 60e-6))
    true_pi = 0.3e-6 + np.pi / OMEGA
    assert fit.ok and fit.noise == "sem"
    assert _within(fit, true_pi), fit.summary()
    assert abs(fit.t_dead - 0.3e-6) < 4 * fit.t_dead_err
    assert abs(fit.f_rabi - F_RABI) < 4 * fit.f_rabi_err
    assert fit.starts_populated


def test_no_decay_is_a_result_not_a_railed_bound():
    rng = np.random.default_rng(2)
    t = _shuffled_repeats(rng)
    fit = fit_rabi(t, _flop(rng, t, noise=0.02, mult=False))
    assert fit.ok
    assert fit.model == "none" or fit.gamma < 1 / 1e-3        # decay unresolved or negligible
    assert not any("bound" in w for w in fit.warnings)
    assert _within(fit, np.pi / OMEGA)


def test_single_shots_fit_unweighted():
    rng = np.random.default_rng(3)
    fit = fit_rabi(T, _flop(rng, T, gamma=1 / 80e-6))
    assert fit.ok and fit.noise == "unweighted" and fit.n_points == T.size
    assert _within(fit, np.pi / OMEGA)


def test_initially_empty_state():
    rng = np.random.default_rng(4)
    t = _shuffled_repeats(rng)
    fit = fit_rabi(t, _flop(rng, t, start_full=False, t_dead=0.2e-6, noise=0.02, mult=False))
    assert fit.ok and not fit.starts_populated
    assert _within(fit, 0.2e-6 + np.pi / OMEGA)


def test_joint_fit_shares_frequency_with_separate_offsets():
    rng = np.random.default_rng(5)
    t1, t2 = _shuffled_repeats(rng), T.copy()
    y1 = _flop(rng, t1, n0=1500, noise=0.03)
    y2 = _flop(rng, t2, n0=900, noise=0.03)
    fit = fit_rabi([t1, t2], [y1, y2], noise="unweighted", labels=("a", "b"))
    assert fit.ok and len(fit.offsets) == 2
    assert abs(fit.offsets[0] - 1500) < 100 and abs(fit.offsets[1] - 900) < 100
    assert _within(fit, np.pi / OMEGA)


def test_joint_fit_mixing_repeats_and_single_shots():
    rng = np.random.default_rng(10)
    t1, t2 = _shuffled_repeats(rng), T.copy()          # one run with repeats, one without
    fit = fit_rabi([t1, t2], [_flop(rng, t1, noise=0.03), _flop(rng, t2, n0=1200, noise=0.03)])
    assert fit.ok and fit.noise == "sem" and np.all(np.isfinite(fit.popt))
    assert _within(fit, np.pi / OMEGA)


def test_short_scan_warns():
    rng = np.random.default_rng(6)
    t = np.linspace(0, 7e-6, 15)
    fit = fit_rabi(t, _flop(rng, t, noise=0.01, mult=False))
    assert fit.ok
    assert any("periods" in w for w in fit.warnings)


def test_too_few_points_is_a_reported_failure():
    fit = fit_rabi([0, 1e-6, 2e-6], [1, 2, 3])
    assert not fit.ok and "5 distinct" in fit.reason
    assert "FAILED" in fit.summary()


def test_compare_and_config_line():
    rng = np.random.default_rng(7)
    t = _shuffled_repeats(rng)
    fit = fit_rabi(t, _flop(rng, t, noise=0.02, mult=False), run_ids=(123,))
    d, sig, text = fit.compare(7.121e-6)
    assert d > 0 and np.isfinite(sig) and "t_raman_pi_pulse" in text
    line = fit.config_line("t_raman_pi_pulse", date="2026-09-26")
    assert line.startswith("self.t_raman_pi_pulse = ") and line.endswith("#123, 2026-09-26")


def test_adapter_on_an_atomdata_like_object():
    rng = np.random.default_rng(8)
    t = _shuffled_repeats(rng)
    ad = SimpleNamespace(xvarnames=["t_raman_pulse"], xvars=[t], atom_number=_flop(rng, t, noise=0.03),
                         run_info=SimpleNamespace(run_id=42), params=SimpleNamespace(t_raman_pi_pulse=7.121e-6))
    fit = rabi(ad)
    assert fit.ok and fit.run_ids == (42,) and fit.xvarname == "t_raman_pulse"
    assert _within(fit, np.pi / OMEGA)
    fig, axes = fit.plot(reference=ad.params.t_raman_pi_pulse)      # smoke test, nothing saved
    assert len(axes) == 2


def test_cli_parser():
    a = build_parser().parse_args(["83092", "83091", "--separate", "--compare", "t_raman_pi_pulse",
                                   "--model", "exp", "--bootstrap", "10"])
    assert a.runs == [83092, 83091] and a.separate and a.model == "exp" and a.bootstrap == 10


def test_bootstrap_interval_brackets_the_fit():
    rng = np.random.default_rng(9)
    t = _shuffled_repeats(rng)
    fit = fit_rabi(t, _flop(rng, t, noise=0.03), bootstrap=40, seed=1)
    assert fit.bootstrap_t_pi is not None
    med, lo, hi = fit.bootstrap_t_pi
    assert lo <= med <= hi and lo < fit.t_pi + 3 * fit.t_pi_err and hi > fit.t_pi - 3 * fit.t_pi_err
