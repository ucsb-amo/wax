"""waxa.analysis.rabi.dead_time on synthetic split-pulse scans with a known dead time.  Writes no files."""

import matplotlib
matplotlib.use("Agg")
import numpy as np

from waxa.analysis.rabi import fit_split_pulse_dead_time
from waxa.analysis.rabi.dead_time import split_pulse_theta

T_PI = 3.9229e-6          # rate pi time
T_DEAD = 127e-9
V_UP, V_DOWN = 0.100, 0.070
SIGMA = 0.004             # per shot (reads averaged)


def _table(t_pi=T_PI, t0=127e-9, grid=8e-9):
    """The configuration table of raman_dead_time_split_pulse.py: (group, N, tau)."""
    configs = [(0, 0, 0.), (1, 1, t_pi + t0)]
    configs += [(2, n, 0.5 * t_pi + t0) for n in [1, 2, 3, 4]]
    configs += [(3, n, 1.5 * t_pi + t0) for n in [1, 2, 4, 6, 8, 10]]
    return [(g, n, grid * round(T / max(n, 1) / grid) if n else 0.) for g, n, T in configs]


def _shots(rng, t_dead=T_DEAD, n_rep=8, n_reads=5, decay=None):
    tab = _table()
    g = np.repeat([c[0] for c in tab], n_rep)
    n = np.repeat([c[1] for c in tab], n_rep).astype(float)
    tau = np.repeat([c[2] for c in tab], n_rep)
    sz = np.cos(split_pulse_theta(n, tau, np.pi / T_PI, t_dead))
    if decay is not None:
        sz = sz * np.exp(-n * np.clip(tau - t_dead, 0, None) / decay)
    y = (V_UP + V_DOWN) / 2 + (V_UP - V_DOWN) / 2 * sz
    reads = y[:, None] + rng.normal(0, SIGMA * np.sqrt(n_reads), (y.size, n_reads))
    p = rng.permutation(y.size)
    return n[p], tau[p], reads[p], g[p]


def test_recovers_dead_time_and_rate():
    rng = np.random.default_rng(1)
    n, tau, reads, g = _shots(rng)
    fit = fit_split_pulse_dead_time(n, tau, reads, group=g, t_pi_ref=T_PI, t_dead_ref=T_DEAD, run_ids=(1,))
    assert fit.ok, fit.summary()
    assert abs(fit.t_dead - T_DEAD) < 4 * fit.t_dead_err, fit.summary()
    assert fit.t_dead_err < 30e-9, fit.summary()
    assert abs(fit.t_pi_rate - T_PI) < 4 * fit.t_pi_rate_err, fit.summary()
    lo, hi = fit.t_dead_interval
    assert lo < fit.t_dead < hi
    assert abs((hi - lo) / 2 - fit.t_dead_err) < 0.5 * fit.t_dead_err      # near-parabolic here
    assert [x["group"] for x in fit.groups] == [2, 3] and all(x["ok"] for x in fit.groups)
    assert fit.lack_of_fit["p"] > 1e-3
    assert abs(fit.v_up - V_UP) < 3e-3 and abs(fit.v_down - V_DOWN) < 3e-3
    fit.plot()
    assert "t_dead" in fit.summary()


def test_errors_cover_the_truth():
    pulls = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        n, tau, reads, g = _shots(rng)
        fit = fit_split_pulse_dead_time(n, tau, reads, group=g, t_pi_ref=T_PI)
        assert fit.ok
        pulls.append((fit.t_dead - T_DEAD) / fit.t_dead_err)
    pulls = np.array(pulls)
    assert 0.6 < pulls.std() < 1.5, pulls
    assert abs(pulls.mean()) < 0.8, pulls


def test_wrong_reference_pi_time_and_zero_dead_time():
    rng = np.random.default_rng(2)
    n, tau, reads, g = _shots(rng, t_dead=0.0)
    fit = fit_split_pulse_dead_time(n, tau, reads, group=g, t_pi_ref=1.06 * T_PI)
    assert fit.ok
    assert abs(fit.t_dead) < 4 * fit.t_dead_err, fit.summary()
    assert abs(fit.t_pi_rate - T_PI) < 4 * fit.t_pi_rate_err, fit.summary()


def test_groups_found_from_totals_and_bad_shots_counted():
    rng = np.random.default_rng(3)
    n, tau, reads, g = _shots(rng)
    reads[0, 2] = np.nan
    tau[5] = np.nan
    fit = fit_split_pulse_dead_time(n, tau, reads, t_pi_ref=T_PI)
    assert fit.ok and fit.n_excluded == 2 and fit.n_shots == n.size - 2
    assert len(fit.groups) == 2
    assert abs(fit.t_dead - T_DEAD) < 4 * fit.t_dead_err


def test_decay_is_modelled_not_read_as_dead_time():
    # without the envelope this data gave t_dead 92 +/- 18 ns and no flag
    rng = np.random.default_rng(4)
    n, tau, reads, g = _shots(rng, decay=8e-6, n_rep=16)
    fit = fit_split_pulse_dead_time(n, tau, reads, group=g, t_pi_ref=T_PI)
    assert fit.ok and fit.model in ("exp", "gauss"), fit.summary()
    assert abs(fit.t_dead - T_DEAD) < 4 * fit.t_dead_err, fit.summary()
    assert "none" in fit.models                            # the no-decay answer stays in the report
    assert "models" in fit.summary()
