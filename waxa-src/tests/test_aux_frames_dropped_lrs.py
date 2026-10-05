"""The aux_frames_dropped record on the analysis side: per-shot auxiliary
frames (camera streams / diagnostic images) the experiment could not get
into the run's file. ``ad.aux_frames_dropped``, the ``!!`` banner, and the
vault's warning. Every file is a fresh HDF5 in tmp_path."""
import json
import os
import types
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from end_run_helpers import make_run_file, finish_run

DROPS = {"queue_max_bytes": 52428800, "peak_bytes": 52000000, "retries": 2,
         "keys": {"img_mot": {"sent": 97, "dropped": 3,
                              "reasons": {"queue_full": 2, "push_failed": 1},
                              "first_dropped_slots": [[4], [5], [9]],
                              "clears_sent": 0, "clears_failed": 0,
                              "clear_fail_reasons": {}, "first_clear_failed_slots": []},
                  "img_mot_meta": {"sent": 100, "dropped": 0, "reasons": {},
                                   "first_dropped_slots": [], "clears_sent": 0,
                                   "clears_failed": 0, "clear_fail_reasons": {},
                                   "first_clear_failed_slots": []}}}
NONE_DROPPED = {"queue_max_bytes": 52428800, "peak_bytes": 1000, "retries": 2,
                "keys": {"img_mot": dict(DROPS["keys"]["img_mot_meta"])}}


def test_the_record_reads_back_and_the_banner_names_key_count_and_reason():
    from waxa.atomdata_base import aux_frames_dropped_banner, read_aux_frames_dropped
    record = read_aux_frames_dropped({"aux_frames_dropped": json.dumps(DROPS)})
    assert record == DROPS
    banner = aux_frames_dropped_banner(85001, record)
    assert banner.startswith("!! run 85001: per-shot auxiliary data never reached the file")
    assert "img_mot 3 dropped" in banner and "queue_full" in banner
    assert "img_mot_meta" not in banner


def test_no_record_and_nothing_dropped_give_no_banner():
    from waxa.atomdata_base import aux_frames_dropped_banner, read_aux_frames_dropped
    assert read_aux_frames_dropped({}) == {}
    assert aux_frames_dropped_banner(1, {}) == ""
    assert aux_frames_dropped_banner(1, NONE_DROPPED) == ""


def test_a_failed_clear_is_in_the_banner_and_bytes_read_the_same():
    from waxa.atomdata_base import aux_frames_dropped_banner, read_aux_frames_dropped
    rec = json.loads(json.dumps(NONE_DROPPED))
    rec["keys"]["img_mot"].update(clears_failed=1, first_clear_failed_slots=[[2]])
    got = read_aux_frames_dropped({"aux_frames_dropped": json.dumps(rec).encode()})
    assert got == rec
    assert "warm-up slot clears not sent" in aux_frames_dropped_banner(3, got)


def test_an_unreadable_record_is_said_not_hidden():
    from waxa.atomdata_base import aux_frames_dropped_banner, read_aux_frames_dropped
    record = read_aux_frames_dropped({"aux_frames_dropped": "{oops"})
    assert "unreadable" in record and record["raw"] == "{oops"
    assert "could not be read" in aux_frames_dropped_banner(5, record)


def test_ad_aux_frames_dropped_defaults_to_empty():
    from waxa.atomdata_base import atomdata_base
    ad = object.__new__(atomdata_base)
    assert ad.aux_frames_dropped == {}
    ad.aux_frames_dropped = DROPS
    assert ad.aux_frames_dropped == DROPS and ad.aux_frames_dropped is not DROPS


def _chunk(run_id, drops):
    return types.SimpleNamespace(
        camera_params=types.SimpleNamespace(key="andor"),
        run_info=types.SimpleNamespace(run_id=run_id), camera_overrides={},
        aux_frames_dropped=drops)


def test_the_vault_says_which_runs_lost_frames():
    from waxa.atomdata_vault import AtomdataVault
    vault = object.__new__(AtomdataVault)
    with pytest.warns(UserWarning, match=r"in run\(s\) 2 some per-shot auxiliary frames"):
        vault._note_camera_differences([_chunk(1, NONE_DROPPED), _chunk(2, DROPS),
                                        _chunk(3, {})])
    assert vault.aux_frames_dropped_by_run == {1: NONE_DROPPED, 2: DROPS, 3: {}}
    vault2 = object.__new__(AtomdataVault)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vault2._note_camera_differences([_chunk(1, NONE_DROPPED), _chunk(2, {})])


def _run_file(tmp_path, extra):
    """A run file made, filled and finished the way liveOD does it."""
    run_id, date = 85001, "2026-10-05"
    folder = tmp_path / date
    folder.mkdir()
    path = str(folder / f"{run_id:07d}_{date}_12-00-00_FakeExpt.hdf5")
    init = {"capture_images": False, "xvarnames": ["t"], "sort_idx": [], "sort_N": [],
            "datavault_shapes": {
                "img_mot": {"shape": (4, 2, 2), "dtype": "uint8", "external": True, "fill": 0},
                "img_mot_meta": {"shape": (4, 7), "dtype": "float64", "external": True,
                                 "fill": float("nan")}},
            "params": {"t": [1., 2., 3., 4.]}, "camera_params": {},
            "run_date_str": date, "run_datetime_str": f"{date}_12-00-00",
            "expt_class": "FakeExpt"}
    ds = make_run_file(path, init, run_id=run_id)
    end = {"params": {"t": [1., 2., 3., 4.]},
           "datavault": {k: {"data": None, "data_gotten": True, "external": True,
                             "final_order": True} for k in ("img_mot", "img_mot_meta")},
           "sort_idx": [], "sort_N": [], "xvardims": [4], "N_shots_with_repeats": 4,
           "N_pwa_per_shot": 1, "capture_images": False, "scope_data_taken": False,
           "scope_data": [], "expt_filepath": "", "expt_file_text": "",
           "params_file_text": "", "base_class_texts": {}, "extra_file_texts": extra}
    finish_run(ds, path, end, shot_timestamps=[1., 2., 3., 4.])
    return path


def test_atomdata_loads_the_record_and_prints_the_banner(tmp_path, capsys):
    # (not ``from waxa import atomdata``: once the waxa.atomdata submodule is
    # imported, as the vault tests here do, the package attribute is the module)
    from waxa.atomdata import atomdata
    from waxa.data.server_talk import server_talk
    path = _run_file(tmp_path, {"aux_frames_dropped": json.dumps(DROPS)})
    with h5py.File(path, "r") as f:
        assert json.loads(f.attrs["aux_frames_dropped"]) == DROPS
    ad = atomdata(path=path, ignore_images=True,
                  server_talk=server_talk(data_dir=str(tmp_path)))
    assert ad.aux_frames_dropped == DROPS
    assert "aux_frames_dropped" not in ad.run_records       # its own attribute
    assert "!! run 85001: per-shot auxiliary data never reached the file" in \
        capsys.readouterr().out
    # the dropped slots hold the fill: frame zeros, NaN record
    assert np.all(ad.data.img_mot == 0) and np.isnan(ad.data.img_mot_meta).all()


def test_atomdata_without_the_record_is_empty_and_quiet(tmp_path, capsys):
    from waxa.atomdata import atomdata
    from waxa.data.server_talk import server_talk
    path = _run_file(tmp_path, {})
    ad = atomdata(path=path, ignore_images=True,
                  server_talk=server_talk(data_dir=str(tmp_path)))
    assert ad.aux_frames_dropped == {}
    assert "auxiliary data never reached" not in capsys.readouterr().out
