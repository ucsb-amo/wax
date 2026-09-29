"""timestamp_shot_end is carried into AtomdataVault, shot for shot."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from waxa.atomdata_vault import AtomdataVault
from test_vault_stacked import PHASES, _chunk, _vault

T0 = 1.8e9  # a plausible unix time


def _flat_chunk(run_id, xvals, shot_end=None):
    xvals = np.asarray(xvals, dtype=float)
    n = xvals.size
    # atom_number encodes (run, x) so the timestamps can be checked against it
    an = run_id * 1000.0 + xvals
    c = SimpleNamespace(
        Nvars=1, xvarnames=["t_tof"], xvars=[xvals], xvardims=np.array([n]),
        params=SimpleNamespace(t_tof=xvals, N_repeats=1, N_shots=n,
                               N_shots_with_repeats=n, N_pwa_per_shot=1),
        camera_params=SimpleNamespace(), roi=None, _has_images=False,
        data=SimpleNamespace(keys=["an"], an=an), scope_data={},
        run_info=SimpleNamespace(run_id=run_id, imaging_type=0),
        _analysis_tags=SimpleNamespace(xvars_shuffled=False),
        images=np.array([]), image_timestamps=np.array([]),
    )
    if shot_end is not None:
        c.timestamp_shot_end = np.asarray(shot_end, dtype=float)
    return c


def _flat_vault(monkeypatch, chunks):
    monkeypatch.setattr(AtomdataVault, "_materialize_inputs",
                        lambda self, *a, **k: list(chunks))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = AtomdataVault([c.run_info.run_id for c in chunks],
                          ignore_images=True, auto_lite_threshold=None)
    return v, [str(w.message) for w in caught]


def _stamp(run_id, xvals):
    return T0 + run_id + np.asarray(xvals) * 1e-3


def test_flat_vault_concatenates_and_sorts_with_the_shots(monkeypatch):
    xa, xb = [3.0, 1.0, 2.0], [2.5, 0.5]
    a = _flat_chunk(7, xa, _stamp(7, xa))
    b = _flat_chunk(5, xb, _stamp(5, xb))
    v, msgs = _flat_vault(monkeypatch, [a, b])
    assert not any("timestamp_shot_end" in m for m in msgs)
    assert v.timestamp_shot_end.shape == (5,)
    # sorted by t_tof; each stamp still belongs to its own shot
    assert np.array_equal(v.xvars[0], [0.5, 1.0, 2.0, 2.5, 3.0])
    expect = _stamp(v.shot_run_id, v.xvars[0])
    assert np.array_equal(v.timestamp_shot_end, expect)

    v.drop_runs([7])
    assert np.array_equal(v.timestamp_shot_end, _stamp(5, [0.5, 2.5]))


def test_flat_vault_nan_fills_and_names_runs_without_stamps(monkeypatch):
    xa, xb, xc = [1.0, 2.0], [1.5], [0.5, 2.5]
    a = _flat_chunk(7, xa, _stamp(7, xa))
    b = _flat_chunk(5, xb)                          # not recorded
    c = _flat_chunk(9, xc, _stamp(9, [0.5]))        # one stamp for two shots
    v, msgs = _flat_vault(monkeypatch, [a, b, c])
    warn = [m for m in msgs if "timestamp_shot_end" in m]
    assert len(warn) == 1
    assert "2 of 3 runs" in warn[0]
    assert "5 (not recorded)" in warn[0] and "9 (shape (1,) != scan (2,))" in warn[0]
    ok = v.shot_run_id == 7
    assert np.array_equal(v.timestamp_shot_end[ok], _stamp(7, v.xvars[0][ok]))
    assert np.all(np.isnan(v.timestamp_shot_end[~ok]))


def test_stacked_vault_stacks_onto_the_union_grid():
    dets_a, dets_b = [-568e6, -559e6, -550e6], [-559e6, -540e6]
    a, b, c = _chunk(7, 3.94, dets_a), _chunk(5, 0.444, dets_b), _chunk(9, 1.0, dets_a)
    a.timestamp_shot_end = T0 + a.atom_number
    b.timestamp_shot_end = T0 + b.atom_number
    # run 9 has none
    with pytest.warns(UserWarning, match=r"1 of 3 runs: 9 \(not recorded\)"):
        v = _vault_warn([a, b, c])
    assert v.timestamp_shot_end.shape == v.atom_number.shape
    rec = v.shot_run_id != 9
    assert np.allclose(v.timestamp_shot_end[rec] - T0, v.atom_number[rec],
                       atol=1e-5, rtol=0, equal_nan=True)
    assert np.all(np.isnan(v.timestamp_shot_end[v.shot_run_id == 9]))
    assert np.all(np.isnan(v.timestamp_shot_end[~v.stack_mask]))


def _vault_warn(chunks):
    # like test_vault_stacked._vault, but lets the vault's warnings through
    v = object.__new__(AtomdataVault)
    v._lite, v.server_talk, v._ignore_images = False, None, True
    v._drop_raw_images, v._merge_overlap, v._stacked = False, True, True
    v._timing_enabled, v._timing = False, {}
    v._validate_inputs_nd(chunks, ignore_images=True)
    v._assemble_stacked(chunks, "v_squeeze", "auto", True, None)
    return v
