"""The end-of-run save with data that went into the file during the run:
containers pre-allocated with their fill value, keys written at their final
slots left alone by the unshuffle, scopes pushed per shot kept, and a stash
that carries no bulk array."""
import pickle

import h5py
import numpy as np
import pytest
from end_run_helpers import make_run_file, write_during_run, finish_run


SORT = [[2, 0, 3, 1]]       # shot j of the scan lands at slot SORT[0][j]


def _init(dv_shapes, sort_idx=()):
    return {"capture_images": False, "xvarnames": ["t"], "sort_idx": list(sort_idx),
            "sort_N": [len(s) for s in sort_idx], "datavault_shapes": dv_shapes,
            "params": {"t": [1, 2, 3, 4]}, "camera_params": {}}


def _end(datavault, sort_idx=(), scope=None, params=None):
    return {"params": params or {"t": [1, 2, 3, 4]}, "datavault": datavault,
            "sort_idx": list(sort_idx), "sort_N": [len(s) for s in sort_idx],
            "xvardims": [4], "N_shots_with_repeats": 4, "N_pwa_per_shot": 1,
            "capture_images": False, "scope_data_taken": scope is not None,
            "scope_data": scope or [], "expt_filepath": "", "expt_file_text": "",
            "params_file_text": "", "base_class_texts": {}, "extra_file_texts": {}}


def test_pre_allocated_containers_carry_their_fill_value(tmp_path):
    path = str(tmp_path / "r.hdf5")
    make_run_file(path, _init({
        "img": {"shape": (4, 2, 2), "dtype": "uint8", "external": False, "fill": 0},
        "img_meta": {"shape": (4, 3), "dtype": "float64", "external": False, "fill": float("nan")},
        "apd": {"shape": (4,), "dtype": "float64", "external": False, "fill": -1.0},
        "old": {"shape": (4,), "dtype": "int64", "external": False},
    }))
    with h5py.File(path, "r") as f:
        assert np.all(np.isnan(f["data/img_meta"][()]))
        assert np.all(f["data/apd"][()] == -1.0)
        assert np.all(f["data/old"][()] == 0) and np.all(f["data/img"][()] == 0)


def test_keys_written_at_their_final_slots_are_left_alone_by_the_unshuffle(tmp_path):
    path = str(tmp_path / "r.hdf5")
    ds = make_run_file(path, _init({
        "img": {"shape": (4, 2), "dtype": "uint8", "external": False, "fill": 0},
        "kernel": {"shape": (4,), "dtype": "float64", "external": False},
    }, sort_idx=SORT))
    # the writer put shot j's frame at its final slot SORT[0][j]
    final = {SORT[0][j]: np.full(2, j + 1, np.uint8) for j in range(4)}
    write_during_run(path, {"img": final})
    # a kernel container still travels in END_RUN in scan order
    kernel = np.array([10., 20., 30., 40.])
    finish_run(ds, path, _end({
        "img": {"data": None, "data_gotten": True, "external": True, "final_order": True},
        "kernel": {"data": kernel, "data_gotten": True, "external": False},
    }, sort_idx=SORT), shot_timestamps=[1., 2., 3., 4.])
    with h5py.File(path, "r") as f:
        assert f.attrs["run_complete"] and f.attrs["unshuffle_applied"]
        for j in range(4):
            assert np.all(f["data/img"][SORT[0][j]] == j + 1)
            assert f["data/kernel"][SORT[0][j]] == kernel[j]
            assert f["data/timestamp_shot_end"][SORT[0][j]] == j + 1


def test_an_external_key_not_in_final_order_is_still_unshuffled_in_place(tmp_path):
    path = str(tmp_path / "r.hdf5")
    ds = make_run_file(path, _init({
        "ext": {"shape": (4,), "dtype": "float64", "external": True},
    }, sort_idx=SORT))
    write_during_run(path, {"ext": {j: float(j + 1) for j in range(4)}})   # scan order
    finish_run(ds, path, _end({
        "ext": {"data": None, "data_gotten": True, "external": True},
    }, sort_idx=SORT))
    with h5py.File(path, "r") as f:
        for j in range(4):
            assert f["data/ext"][SORT[0][j]] == j + 1


def test_pushed_scope_traces_stay_and_a_payload_scope_is_rewritten_beside_them(tmp_path):
    path = str(tmp_path / "r.hdf5")
    ds = make_run_file(path, _init({}))
    t = np.linspace(0, 1, 5, dtype=np.float32)
    v = np.arange(4 * 1 * 5, dtype=np.float32).reshape(4, 1, 5)
    write_during_run(path, {"scope_data/PD/t": {None: t}, "scope_data/PD/v": {None: v}})
    other = np.stack([np.stack([t, t * 2])] * 4)[:, None]       # (4, 1 ch, 2, 5)
    finish_run(ds, path, _end({}, scope=[
        {"label": "PD", "data": None, "pushed": True},
        {"label": "APD", "data": other},
    ]))
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["data/scope_data/PD/v"][()], v)
        assert np.array_equal(f["data/scope_data/PD/t"][()], t)
        assert np.array_equal(f["data/scope_data/APD/v"][()], other[:, :, 1, :])
        assert f.attrs["run_complete"]


def test_the_stash_keeps_the_small_payload_and_drops_the_bulk():
    from waxa.data.data_saver import _payload_without_bulk, STASH_MAX_ARRAY_BYTES
    big = np.zeros(STASH_MAX_ARRAY_BYTES + 1, np.uint8)
    payload = _end({
        "big": {"data": big, "data_gotten": True, "external": False},
        "apd": {"data": np.ones(4), "data_gotten": True, "external": False},
    }, scope=[{"label": "PD", "data": np.zeros((4, 1, 2, 3 * STASH_MAX_ARRAY_BYTES // 8), np.float32)}])
    small = _payload_without_bulk(payload)
    assert small["datavault"]["big"]["data"] is None and small["datavault"]["big"]["stash_dropped"]
    assert np.array_equal(small["datavault"]["apd"]["data"], np.ones(4))
    assert small["scope_data"][0]["data"] is None
    assert small["stash_dropped"] == ["big", "scope PD"]
    assert payload["datavault"]["big"]["data"] is big          # the original is untouched
    assert len(pickle.dumps(small)) < 10_000
