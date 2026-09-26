"""waxa.analysis.rabi.linearize on synthetic flops with a known detection nonlinearity.  Writes no files."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np

from waxa.analysis.rabi import linearize, linearize_signal

F_RABI = 80e3
OMEGA = 2 * np.pi * F_RABI
T = np.linspace(0, 30e-6, 31)
N0 = 2000.0


def _population(t, t_dead=0.2e-6, contrast=0.95):
    """Imaged-state population, starting full."""
    return 0.5 + 0.5 * contrast * np.cos(OMEGA * (t - t_dead))


def _shots(rng, h, nrep=5, noise=0.015, t_dead=0.2e-6, atoms=1.0):
    """Shots of N0 h(atoms P): the detector nonlinearity h acts on the absolute signal."""
    t = np.repeat(T, nrep)
    rng.shuffle(t)
    return t, N0 * h(atoms * _population(t, t_dead)) + rng.normal(0, noise * N0, t.size)


def _linearity_error(m, h):
    """Max deviation of g(h(P)) from a straight line in P, as a fraction of its span."""
    p = np.linspace(0.05, 0.95, 50)
    z = m(N0 * h(p))
    resid = z - np.polyval(np.polyfit(p, z, 1), p)
    return np.max(np.abs(resid)) / np.ptp(z)


def _convex(p):
    """Mild convex distortion, h(0) = 0 and h(1) = 1: slope 0.7 at the bottom, 1.3 at the top."""
    return p + 0.3 * p * (p - 1)


def test_recovers_a_convex_distortion():
    rng = np.random.default_rng(11)
    t, y = _shots(rng, _convex)
    m = linearize_signal(t, y)
    assert m.ok, m.reason
    assert m.chi2_raw / m.dof_raw > 10                       # the distortion is visible in the raw fit
    assert m.chi2_mapped / m.dof_mapped < 2
    assert m.aicc_mapped < m.aicc_raw - 10
    p = np.linspace(0.05, 0.95, 50)
    raw_err = np.max(np.abs(_convex(p) - np.polyval(np.polyfit(p, _convex(p), 1), p))) / np.ptp(_convex(p))
    assert _linearity_error(m, _convex) < 0.25 * raw_err     # g(h(P)) much straighter than h(P)
    true_pi = 0.2e-6 + np.pi / OMEGA
    assert abs(m.t_pi - true_pi) < 4 * m.t_pi_err, m.summary()
    assert m.monotonic and m.fit_mapped is not None and m.fit_mapped.ok
    assert m.n_shots_unmappable == 0


def test_steep_distortion_is_handled_by_the_forward_cubic():
    # the inverse of p**1.6 is infinitely steep at zero signal, but D (signal vs flop) is smooth
    rng = np.random.default_rng(11)
    h = lambda p: p ** 1.6
    t, y = _shots(rng, h)
    m = linearize_signal(t, y)
    assert m.ok and m.monotonic
    assert m.chi2_raw / m.dof_raw > 30 and m.chi2_mapped / m.dof_mapped < 2
    assert _linearity_error(m, h) < 0.03


def test_linear_signal_gives_a_near_identity_map():
    rng = np.random.default_rng(12)
    t, y = _shots(rng, lambda p: p)
    m = linearize_signal(t, y)
    assert m.ok
    assert np.all(np.abs(m.coeffs[2:]) < 4 * m.coeffs_err[2:]), m.summary()
    assert m.aicc_mapped > m.aicc_raw - 4                    # the map buys nothing it has to pay for
    assert abs(m.t_pi - m.t_pi_raw) < m.t_pi_raw_err


def test_fix_cosine_keeps_the_raw_pi_time():
    rng = np.random.default_rng(13)
    t, y = _shots(rng, _convex)
    m = linearize_signal(t, y, fix_cosine=True)
    assert m.ok and m.t_pi == m.t_pi_raw
    assert any("fix_cosine" in w for w in m.warnings)
    assert m.chi2_mapped < m.chi2_raw


def test_joint_runs_share_one_map():
    rng = np.random.default_rng(14)
    t1, y1 = _shots(rng, _convex)
    t2, y2 = _shots(rng, _convex, nrep=3, atoms=0.8)          # fewer atoms, same detector
    m = linearize_signal([t1, t2], [y1, y2], run_ids=(1, 2))
    assert m.ok and len(m.centers) == 2
    assert m.chi2_mapped / m.dof_mapped < 2
    assert abs(m.scales[1] - 0.8) < 0.1                       # run 2 covers 0.8 of run 1's population span


def test_single_shots_are_refused():
    rng = np.random.default_rng(15)
    y = N0 * _population(T) + rng.normal(0, 20, T.size)
    m = linearize_signal(T, y)
    assert not m.ok and "repeat" in m.reason
    assert "FAILED" in m.summary()


def test_adapter_and_plot():
    rng = np.random.default_rng(16)
    t, y = _shots(rng, _convex)
    ad = SimpleNamespace(xvarnames=["t_raman_pulse"], xvars=[t], atom_number=y,
                         run_info=SimpleNamespace(run_id=7), params=SimpleNamespace())
    m = linearize(ad)
    assert m.ok and m.run_ids == (7,)
    assert "detection curve" in m.summary() and m.monotonic
    fig, axes = m.plot()                                       # smoke test, nothing saved
    assert len(axes) == 3
