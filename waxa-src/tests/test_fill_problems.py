"""The run file's fill returns its PROBLEMS (queue phase 2, 2026-10-14): every
value that is not in the new file as the payload had it, as a list, besides
the old prints; and liveOD's RESERVE_RUN dry run fills an in-memory file with
the same function. Every file here is in pytest's tmp_path (or in memory)."""
import os

import h5py
import numpy as np
import pytest

FILL = "_populate" + "_data_file"          # the fill, by name


def _saver(folder, server_talk=None):
    from waxa.data.data_saver import DataSaver
    return DataSaver(str(folder), server_talk=server_talk)


def _fill_on_disk(folder, payload):
    path = os.path.join(str(folder), "0000101.hdf5")
    with h5py.File(path, "w") as f:
        problems = getattr(_saver(folder), FILL)(f, payload, 101)
    return path, problems


BAD = {
    "params": {"t_tof": 1e-3, "nothing": None, "a/b": 2.0, "d": {"k": 1}},
    "camera_params": {"key": "andor", "sensor_roi": None},
    "xvarnames": ["t_tof"], "capture_images": True, "images_shape": (4, 8, 8),
    "datavault_shapes": {"apd": {"shape": [4], "dtype": "float64", "fill": float("nan")},
                         "bad": {"shape": [4], "dtype": "float64", "fill": "nope"},
                         "x/y": {"shape": [4], "dtype": "int64", "fill": -1}},
}


def test_every_unstored_value_is_a_problem_and_the_prints_stay(tmp_path, capsys):
    path, problems = _fill_on_disk(tmp_path, BAD)
    text = "\n".join(problems)
    assert "params/nothing not stored (NoneType value)" in text
    assert "params/d not stored (dict value)" in text
    assert "camera_params/sensor_roi not stored (NoneType value)" in text
    assert "data/bad: DataVault container not pre-allocated" in text
    assert "params/a/b: '/' in the name" in text and "data/x/y: '/' in the name" in text
    assert len(problems) == 6
    out = capsys.readouterr().out
    assert "[DataSaver] WARNING: params/nothing not stored" in out
    assert "[DataSaver] Could not pre-allocate DataVault 'bad'" in out
    with h5py.File(path, "r") as f:                 # written as before
        assert "t_tof" in f["params"] and "nothing" not in f["params"]
        assert isinstance(f["params/a"], h5py.Group) and "b" in f["params/a"]
        assert "apd" in f["data"] and "bad" not in f["data"]


def test_a_storable_payload_has_no_problems(tmp_path):
    _, problems = _fill_on_disk(tmp_path, {"params": {"t_tof": 1e-3, "N": 2},
                                           "camera_params": {"key": "andor"},
                                           "xvarnames": ["t_tof"]})
    assert problems == []


def test_a_dropped_run_info_value_is_named_no_longer_silent(tmp_path):
    from types import SimpleNamespace
    ri = SimpleNamespace(run_id=5, expt_class="x", odd=None)
    with h5py.File(os.path.join(str(tmp_path), "r.h5"), "w") as f:
        problems = _saver(tmp_path)._run_info_proxy_to_h5(f, ri)
        assert "run_id" in f["run_info"] and "odd" not in f["run_info"]
    assert any("run_info attribute odd not stored" in p for p in problems)
    assert any("run_info/odd not stored" in p for p in problems)


def test_the_dry_run_finds_what_the_disk_fill_finds_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = sorted(os.listdir(tmp_path))
    dry = _saver(tmp_path / "nowhere").dry_run_fill(BAD)
    assert sorted(os.listdir(tmp_path)) == before            # no file, anywhere here
    assert not (tmp_path / "nowhere").exists()
    disk_dir = tmp_path / "disk"
    disk_dir.mkdir()
    _, problems = _fill_on_disk(disk_dir, BAD)
    assert dry["error"] is None and dry["problems"] == problems


@pytest.mark.parametrize("change, words", [
    ({"datavault_shapes": {"k": {"shape": [2], "dtype": "notadtype"}}}, "TypeError"),
    ({"datavault_shapes": {"k": {"dtype": "float64"}}}, "KeyError"),
    ({"imaging_type": "absorption"}, "ValueError"),
    ({"xvarnames": ["t", 3]}, ""),
])
def test_what_would_raise_is_the_dry_runs_error(tmp_path, change, words):
    payload = dict({"params": {"t": 1.0}, "xvarnames": []}, **change)
    dry = _saver(tmp_path).dry_run_fill(payload)
    assert dry["error"] and words in dry["error"]
    with pytest.raises(Exception):                            # the real fill raises too
        _fill_on_disk(tmp_path, payload)


class _Talk:
    """The server_talk calls reserve makes, inside tmp_path."""

    def __init__(self):
        self.next = 1

    def check_for_mapped_data_dir(self):
        return True

    def get_run_id(self):
        return self.next

    def get_latest_run_id_any(self):
        return None

    def set_run_id(self, value):
        self.next = value


def test_the_real_reserve_keeps_its_fills_problems(tmp_path):
    saver = _saver(tmp_path, server_talk=_Talk())
    payload = dict(BAD, run_date_str="2026-10-14", run_datetime_str="2026-10-14_10-00-00",
                   expt_class="probe")
    run_id, path = saver.reserve_run_id_and_path(payload)
    assert os.path.dirname(path) == os.path.join(str(tmp_path), "2026-10-14")
    assert len(saver.last_fill_problems) == 6
    saver.reserve_run_id_and_path({"params": {"t": 1.0}, "xvarnames": [],
                                   "run_date_str": "2026-10-14",
                                   "run_datetime_str": "2026-10-14_10-00-01"})
    assert saver.last_fill_problems == []                     # per call, not accumulated


def test_the_data_dir_check_is_read_only(tmp_path):
    assert _saver(tmp_path).data_dir_reachable() == (True, "")
    ok, why = _saver(tmp_path / "gone").data_dir_reachable()
    assert not ok and "not found" in why and not (tmp_path / "gone").exists()
