"""B2: a params / camera_params value h5py cannot store is left out of the new
run file (it always was), but the saver now names it instead of dropping it in
silence. Every file here is a fresh HDF5 in tmp_path.

Passes once the guarded proposal for waxa/data/data_saver.py (LIVEOD, B2) is
applied.
"""
import os

import h5py


def _populate(folder, payload):
    from waxa.data.data_saver import DataSaver
    ds = DataSaver(str(folder))              # the saver's root: this test's folder
    path = os.path.join(str(folder), "0000101.hdf5")
    with h5py.File(path, "w") as f:
        ds._populate_data_file(f, payload, 101)
    return path


def test_a_value_h5py_cannot_store_is_named(tmp_path, capsys):
    payload = {"params": {"t_tof": 1.0e-3, "not_a_number": None},
               "camera_params": {"key": "andor", "gain": 300.0, "sensor_roi": None},
               "xvarnames": ["t_tof"]}
    path = _populate(tmp_path, payload)
    out = capsys.readouterr().out
    assert "[DataSaver] WARNING: params/not_a_number not stored (NoneType value)" in out
    assert "[DataSaver] WARNING: camera_params/sensor_roi not stored (NoneType value)" in out
    with h5py.File(path, "r") as f:
        assert "t_tof" in f["params"] and "not_a_number" not in f["params"]
        assert "gain" in f["camera_params"] and "sensor_roi" not in f["camera_params"]


def test_a_storable_payload_says_nothing(tmp_path, capsys):
    _populate(tmp_path, {"params": {"t_tof": 1.0e-3, "N_repeats": 2},
                         "camera_params": {"key": "andor", "sensor_roi": (0, 512, 0, 512, 1, 1)},
                         "xvarnames": []})
    assert "not stored" not in capsys.readouterr().out
