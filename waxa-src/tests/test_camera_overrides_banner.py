"""The camera overrides record (INTERFACES section 4) on the analysis side: the
banner, ``ad.camera_overrides``, lite copies, and AtomdataVault saying when its
runs' cameras differ. Every file is a fresh HDF5 in tmp_path.

Passes once the guarded proposals for waxa/atomdata_base.py and
waxa/atomdata_vault.py (LIVEOD) are applied; the lite-copy test passes already.
"""
import json
import os
import types
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

PERSIST = {"schema": 1, "camera_key": "andor", "persist_since": "2026-09-26T14:02:11",
           "fields": {"gain": {"requested": 300, "applied": 30, "origin": "persist"}}}
BOTH = {"schema": 1, "camera_key": "andor", "persist_since": "2026-09-26T14:02:11",
        "fields": {"gain": {"requested": 300, "applied": 30, "origin": "persist"},
                   "exposure_time": {"requested": 1.5e-05, "applied": 1.9e-05,
                                     "origin": "clamped"}}}


def test_the_banner_is_the_one_line_in_the_contract():
    from waxa.atomdata_base import camera_overrides_banner, read_camera_overrides
    record = read_camera_overrides({"camera_overrides": json.dumps(PERSIST)})
    assert record == PERSIST
    assert camera_overrides_banner(80713, record) == (
        "!! run 80713: camera settings differ from ad.camera_params (the request): "
        "gain 300 -> 30 (persist)")
    assert camera_overrides_banner(80713, BOTH).endswith(
        "gain 300 -> 30 (persist), exposure_time 1.5e-05 -> 1.9e-05 (clamped)")


def test_no_record_is_an_empty_dict_and_no_banner():
    from waxa.atomdata_base import camera_overrides_banner, read_camera_overrides
    assert read_camera_overrides({}) == {}
    assert camera_overrides_banner(1, {}) == ""


def test_a_record_stored_as_bytes_reads_the_same():
    from waxa.atomdata_base import read_camera_overrides
    assert read_camera_overrides({"camera_overrides": json.dumps(PERSIST).encode()}) == PERSIST


def test_an_unreadable_record_is_said_not_hidden():
    from waxa.atomdata_base import camera_overrides_banner, read_camera_overrides
    record = read_camera_overrides({"camera_overrides": "{not json"})
    assert "unreadable" in record and record["raw"] == "{not json"
    assert "could not be read" in camera_overrides_banner(5, record)


def test_ad_camera_overrides_defaults_to_empty():
    from waxa.atomdata_base import atomdata_base
    ad = object.__new__(atomdata_base)
    assert ad.camera_overrides == {}
    ad.camera_overrides = PERSIST
    assert ad.camera_overrides == PERSIST and ad.camera_overrides is not PERSIST


def _chunk(run_id, gain, overrides):
    return types.SimpleNamespace(
        camera_params=types.SimpleNamespace(key="andor", gain=gain, resolution=np.array([512, 512])),
        run_info=types.SimpleNamespace(run_id=run_id), camera_overrides=overrides)


def test_the_vault_says_when_its_runs_cameras_differ():
    from waxa.atomdata_vault import AtomdataVault
    vault = object.__new__(AtomdataVault)
    with pytest.warns(UserWarning) as caught:
        vault._note_camera_differences([_chunk(1, 300.0, {}), _chunk(2, 30.0, PERSIST)])
    text = " ".join(str(w.message) for w in caught)
    assert "camera_params differ" in text and "gain" in text
    assert "in run(s) 2 the camera applied settings other than camera_params" in text
    assert vault.camera_param_disagreements == {"gain": {1: 300.0, 2: 30.0}}
    assert vault.camera_overrides_by_run == {1: {}, 2: PERSIST}
    assert vault.camera_overrides == {}                       # the first run's, like camera_params


def test_the_vault_is_quiet_when_its_runs_agree():
    from waxa.atomdata_vault import AtomdataVault
    vault = object.__new__(AtomdataVault)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vault._note_camera_differences([_chunk(1, 300.0, {}), _chunk(2, 300.0, {})])
    assert vault.camera_param_disagreements == {} and vault.camera_overrides == {}


def test_a_lite_copy_carries_the_record(tmp_path):
    from waxa.data.server_talk import server_talk
    run_id, date = 71299, "2026-09-26"
    folder = tmp_path / date
    folder.mkdir()
    raw = str(folder / f"{run_id:07d}_{date}_12-00-00_FakeExpt.hdf5")
    with h5py.File(raw, "w") as f:
        ri = f.create_group("run_info")
        ri["run_id"] = run_id
        ri["run_date_str"] = date
        ri["run_datetime_str"] = f"{date}_12-00-00"
        ri["expt_class"] = "FakeExpt"
        f.create_group("params")["N_pwa_per_shot"] = 1
        f.create_group("data")["images"] = np.zeros((3, 20, 20), np.uint16)
        f.attrs["run_complete"] = True
        f.attrs["camera_overrides"] = json.dumps(PERSIST)
    lite = server_talk(data_dir=str(tmp_path)).create_lite_copy(run_id, roix=[0, 10], roiy=[0, 10],
                                                                path=raw)
    with h5py.File(lite, "r") as f:
        assert json.loads(f.attrs["camera_overrides"]) == PERSIST
