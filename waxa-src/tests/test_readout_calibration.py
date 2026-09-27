"""waxa.analysis.readout on synthetic flops with known endpoints, noise and response.  Writes no files."""

import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from waxa.analysis.readout import (calibrate_readout, check_timing, find_pulse_edges, fit_sz_response,
                                   integrate_scope_pulses, pulse_time_to_angle_and_sz, sz_response)

T_PI = 6.5e-6
T_DEAD = 0.3e-6
OMEGA = np.pi / (T_PI - T_DEAD)
V_UP, V_DOWN = -0.170, -0.208
P = 5


def _sz(t):
    return np.cos(OMEGA * np.clip(t - T_DEAD, 0, None))


def _runs(rng, n_runs=3, reps=2, common=4e-3, indep=6e-3, v_up=None, v_down=None, midpoint=0.5):
    """Shots of P readout pulses after a flop: shot-common + independent Gaussian noise."""
    v_up = [V_UP] * n_runs if v_up is None else v_up
    v_down = [V_DOWN] * n_runs if v_down is None else v_down
    t_grid = T_PI * np.linspace(0, 2, 21)
    T, Y, R, TS = [], [], [], []
    for r in range(n_runs):
        t = np.repeat(t_grid, reps)
        rng.shuffle(t)
        s = _sz(t)
        p1 = 0.5 * (1 + s) + (midpoint - 0.5) * (1 - s ** 2)
        clean = v_down[r] + (v_up[r] - v_down[r]) * p1
        y = clean[:, None] + rng.normal(0, common, (t.size, 1)) + rng.normal(0, indep, (t.size, P))
        T.append(t); Y.append(y); R.append(np.full(t.size, 1000 + r)); TS.append(300.0 * r + 5.5 * np.arange(t.size))
    return np.concatenate(T), np.concatenate(Y), np.concatenate(R), np.concatenate(TS)


def test_sz_mapping_folds_and_clips():
    ang, sz = pulse_time_to_angle_and_sz(np.array([0.0, 0.1e-6, 1.1e-6, 2.1e-6, 3.1e-6]), 1e-6, 0.1e-6)
    assert np.allclose(sz, [1.0, 1.0, -1.0, 1.0, -1.0])
    assert np.allclose(ang, [0, 0, np.pi, 0, np.pi])


def test_quadratic_response_is_exact_on_model_data():
    s = np.linspace(-0.95, 0.95, 15)
    mid, y_d, y_u = 0.62, -0.21, -0.17
    y = y_d + (y_u - y_d) * (0.5 * (1 + s) + (mid - 0.5) * (1 - s ** 2))
    f = fit_sz_response(s, y, degree=2)
    assert f["y_up"] == pytest.approx(y_u, abs=1e-12)
    assert f["y_down"] == pytest.approx(y_d, abs=1e-12)
    assert f["midpoint_fraction"] == pytest.approx(mid, abs=1e-10)
    lin = fit_sz_response(s, y, degree=1)
    assert lin["midpoint_fraction"] == 0.5
    r = sz_response(s, y, degree=2, source="fit")
    assert r["y_up"] == pytest.approx(y_u, abs=1e-12) and r["fit_pinned"] is not None


def test_recovers_endpoints_noise_and_agreeing_runs():
    rng = np.random.default_rng(3)
    t, y, run, ts = _runs(rng)
    cal = calibrate_readout(t, y, run=run, timestamps=ts, t_pi=T_PI, t_offset=T_DEAD, n_boot=300)
    assert cal.ok, cal.reason
    e = cal.pooled_linear
    assert abs(e.v_up - V_UP) < 4 * e.v_up_err
    assert abs(e.v_down - V_DOWN) < 4 * e.v_down_err
    assert abs(cal.flop.t_dead - T_DEAD) < 4 * cal.flop.t_dead_err
    assert cal.degree_used == 1 and not cal.quadratic_test["significant"]
    single = np.hypot(4e-3, 6e-3)
    for key in ("within-run, up", "within-run, down", "within-run, all shots"):
        assert cal.noise[key].sigma == pytest.approx(single, rel=0.2), key
    d = cal.noise_decomposition
    assert d["common"] == pytest.approx(4e-3, rel=0.35)
    assert d["independent"] == pytest.approx(6e-3, rel=0.15)
    assert d["avg"] < d["single"]                                   # the pulse average understates one pulse
    assert cal.agreement["f_offset"]["p"] > 1e-3 and cal.agreement["v_up"]["p"] > 1e-3
    p = cal.proposal
    assert p["midpoint"] == 0.5 and p["degree"] == 1
    assert p["sigma_signal"] == max(cal.noise[k].sigma for k in
                                    ("within-run, up", "within-run, down", "pooled, up", "pooled, down"))
    assert p["sigma_fraction"] == pytest.approx(p["sigma_signal"] / p["v_range"])
    for dd in cal.pulse_differences:                               # identical pulses
        assert abs(dd.v_up) < 4 * dd.v_up_err and abs(dd.v_down) < 4 * dd.v_down_err
    json.dumps(cal.to_dict())                                       # JSON-able, no NaN objects
    assert "pooled, linear" in cal.summary()


def test_flags_a_run_that_moved():
    rng = np.random.default_rng(5)
    t, y, run, ts = _runs(rng, v_up=[V_UP, V_UP, V_UP + 0.009])
    cal = calibrate_readout(t, y, run=run, timestamps=ts, t_pi=T_PI, n_boot=100)
    assert cal.ok
    assert not cal.pooling_justified
    assert cal.agreement["v_up"]["p"] < 0.01
    moved = cal.per_run[2].v_up - cal.per_run[0].v_up
    assert abs(moved - 0.009) < 4 * np.hypot(cal.per_run[2].v_up_err, cal.per_run[0].v_up_err)
    assert cal.proposal["v_up_err_with_run_scatter"] > cal.proposal["v_up_err"]
    assert any("do not agree" in w for w in cal.warnings)


def test_resolves_a_real_midpoint_remap():
    rng = np.random.default_rng(7)
    t, y, run, ts = _runs(rng, n_runs=3, reps=6, common=1e-3, indep=2e-3, midpoint=0.66)
    cal = calibrate_readout(t, y, run=run, t_pi=T_PI, n_boot=0)
    assert cal.ok
    assert cal.quadratic_test["significant"] and cal.degree_used == 2
    q = cal.pooled_quadratic
    assert abs(q.midpoint - 0.66) < 4 * q.midpoint_err
    assert cal.proposal["midpoint"] == pytest.approx(q.midpoint)


def test_inverted_readout_keeps_endpoint_meaning():
    rng = np.random.default_rng(9)
    t, y, run, ts = _runs(rng, n_runs=1, v_up=[V_DOWN], v_down=[V_UP])
    cal = calibrate_readout(t, y, run=run, t_pi=T_PI, n_boot=0)
    assert cal.ok
    assert cal.pooled_linear.v_up < cal.pooled_linear.v_down
    assert abs(cal.pooled_linear.v_up - V_DOWN) < 4 * cal.pooled_linear.v_up_err
    assert any("inverted" in w for w in cal.warnings)


def test_params_sz_source_and_bad_inputs():
    rng = np.random.default_rng(11)
    t, y, run, ts = _runs(rng, n_runs=1)
    cal = calibrate_readout(t, y, run=run, t_pi=T_PI, t_offset=T_DEAD, sz_source="params", n_boot=0)
    assert cal.ok and cal.sz_source == "params"
    assert np.allclose(cal.sz, pulse_time_to_angle_and_sz(t, T_PI, T_DEAD)[1])
    assert not calibrate_readout(t, y[:-1], run=run).ok
    assert not calibrate_readout(t, y, run=run, pulses_used=[7]).ok
    assert not calibrate_readout(t, y, run=run, sz_source="params").ok       # needs t_pi


def _traces(n_shots=4, heights=None, compress=1.0, t_between=25e-6, width=5e-6, delay=1.5e-6, dt=2e-9):
    """Trapezoid pulses (0.3 us edges) on a noiseless trace; ``compress`` scales the recorded time axis."""
    t_true = np.arange(-20e-6, 5 * t_between + 10e-6, dt)
    heights = np.ones((n_shots, P)) if heights is None else heights
    v = np.zeros((n_shots, t_true.size))
    edge = 0.3e-6
    for s in range(n_shots):
        for p in range(P):
            t0 = delay + p * t_between
            ramp = np.clip((t_true - t0) / edge, 0, 1) * np.clip((t0 + width - t_true) / edge, 0, 1)
            v[s] += heights[s, p] * ramp
    return np.tile(t_true * compress, (n_shots, 1)), v


def test_scope_integrals_and_timing_check():
    heights = np.array([[1.0, 0.9, 0.8, 0.7, 0.6]] * 3)
    t, v = _traces(n_shots=3, heights=heights)
    sp = integrate_scope_pulses(t, v, t_duration=5e-6, n_pulses=P, t_between_pulses=25e-6)
    assert sp.timing_ok, sp.problems
    expect = heights * (5e-6 - 0.3e-6)                              # trapezoid area
    assert np.allclose(sp.integral, expect, rtol=0.01)
    assert sp.pulse0_start == pytest.approx(1.5e-6, abs=0.1e-6)
    # a recorded time axis compressed by 2 (the 83148-83150 scope traces) must fail the check
    tc, vc = _traces(n_shots=3, heights=heights, compress=0.5)
    bad = integrate_scope_pulses(tc, vc, t_duration=5e-6, n_pulses=P, t_between_pulses=25e-6, t_start_shift=1e-6,
                                 t_end_shift=0.2e-6)
    assert not bad.timing_ok
    assert any("FWHM" in p for p in bad.problems) and any("hold no pulse" in p for p in bad.problems)
    e = find_pulse_edges(t[0], v[0], -5e-6, 20e-6)
    assert check_timing(e, sp.level, 5e-6) == ()


def test_scope_enters_proposal_only_when_timing_ok():
    rng = np.random.default_rng(13)
    t, y, run, ts = _runs(rng, n_runs=1, reps=2, common=5e-4, indep=5e-4)   # precise flop: this tests the scope
    n = t.size
    heights = (0.5 + 0.2 * (1 + _sz(t)))[:, None] * np.ones((1, P))
    tt, vv = _traces(n_shots=n, heights=heights, dt=10e-9)
    sp = integrate_scope_pulses(tt, vv, t_duration=5e-6, n_pulses=P, t_between_pulses=25e-6)
    cal = calibrate_readout(t, y, run=run, t_pi=T_PI, scope=sp, scope_to_photons=1e9, n_boot=0)
    assert cal.scope["valid"] and "n_photons_per_pulse" in cal.proposal
    assert cal.proposal["n_photons_per_pulse"] == pytest.approx(0.4 * (5e-6 - 0.3e-6) * 1e9, rel=0.05)
    tc, vc = _traces(n_shots=n, heights=heights, dt=10e-9, compress=0.5)
    spc = integrate_scope_pulses(tc, vc, t_duration=5e-6, n_pulses=P, t_between_pulses=25e-6, t_start_shift=1e-6,
                                 t_end_shift=0.2e-6)
    cal2 = calibrate_readout(t, y, run=run, t_pi=T_PI, scope=spc, scope_to_photons=1e9, n_boot=0)
    assert not cal2.scope["valid"] and "n_photons_per_pulse" not in cal2.proposal
    assert any("timing check FAILED" in w for w in cal2.warnings)


def test_plots_build():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(17)
    t, y, run, ts = _runs(rng)
    cal = calibrate_readout(t, y, run=run, timestamps=ts, t_pi=T_PI, n_boot=0, xvarname="t_raman_pulse")
    fig1, _ = cal.plot_pulses()
    fig2, _ = cal.plot_summary()
    plt.close(fig1)
    plt.close(fig2)
