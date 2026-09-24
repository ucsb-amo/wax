"""The file attributes of a run liveOD finalized with frames missing, and how
the run lookup treats them. Every file here is a fresh HDF5 in tmp_path.
"""
import os

import h5py
import pytest


def _fresh_run_file(folder):
    path = os.path.join(str(folder), "0000101.hdf5")
    with h5py.File(path, "w") as f:
        f.create_group("data")
        f.create_group("params")
        f.create_group("run_info")
        f.attrs["run_complete"] = False
        f.attrs["xvarnames"] = []
    return path


def _finalize(path, incomplete):
    from waxa.data.data_saver import DataSaver
    ds = DataSaver(os.path.dirname(path))     # the saver's root: this test's folder
    out = ds._compute_end_run_outputs(
        {}, {"images": None, "image_timestamps": None, "external": {}, "torn": False}, None)
    ds._write_end_run_outputs(path, {}, out, incomplete)


def test_a_complete_run_is_marked_complete_on_every_flag(tmp_path):
    path = _fresh_run_file(tmp_path)
    _finalize(path, None)
    with h5py.File(path, "r") as f:
        assert bool(f.attrs["run_complete"]) is True
        assert bool(f.attrs["run_finalized"]) is True
        assert bool(f.attrs["data_complete"]) is True
        assert "incomplete_reason" not in f.attrs


def test_an_incomplete_run_keeps_run_complete_false_and_says_why(tmp_path):
    path = _fresh_run_file(tmp_path)
    _finalize(path, {"reason": "60/63 images received; camera timed out",
                     "images_expected": 63, "images_received": 60})
    with h5py.File(path, "r") as f:
        assert bool(f.attrs["run_complete"]) is False
        assert bool(f.attrs["run_finalized"]) is True
        assert bool(f.attrs["data_complete"]) is False
        assert f.attrs["incomplete_reason"] == "60/63 images received; camera timed out"
        assert int(f.attrs["images_expected"]) == 63 and int(f.attrs["images_received"]) == 60


def test_run_lookup_treats_a_finalized_incomplete_run_as_finished(tmp_path):
    from waxa.data.server_talk import server_talk
    path = _fresh_run_file(tmp_path)
    is_done = lambda: server_talk._is_completed_run(object(), path)
    assert is_done() is False                                 # still being written
    _finalize(path, {"reason": "1/3 images received", "images_expected": 3, "images_received": 1})
    assert is_done() is True                                  # finished, though incomplete
    with h5py.File(path, "r+") as f:
        del f.attrs["run_finalized"]                          # a file from before the attr
    assert is_done() is False


def test_stash_carries_the_incomplete_marker(tmp_path, monkeypatch):
    import pickle
    from waxa.data import data_saver
    monkeypatch.setattr(data_saver, "pending_save_dir", lambda: str(tmp_path))
    marker = {"reason": "2/3 images received", "images_expected": 3, "images_received": 2}
    stash = data_saver.stash_end_run_payload({"a": 1}, "somewhere.hdf5", 101, [1.0], incomplete=marker)
    with open(stash, "rb") as fh:
        assert pickle.load(fh)["incomplete"] == marker
    stash = data_saver.stash_end_run_payload({"a": 1}, "somewhere.hdf5", 102, [1.0])
    with open(stash, "rb") as fh:
        assert pickle.load(fh)["incomplete"] is None
