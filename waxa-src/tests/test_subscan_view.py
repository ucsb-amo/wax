"""A run with an interleaved sub-scan (waxx SubScan) in atomdata: its
containers leave ad.data and load as ad.cal, a scan of its own; siblings and
slices carry it; the vault takes it as part='cal' and a main vault gets a lazy
.cal. Files are made the way liveOD makes them, in pytest's temp folder."""
import json
import os

import h5py
import numpy as np
import pytest

from end_run_helpers import make_run_file, write_during_run, finish_run
from waxa.atomdata import atomdata
from waxa.atomdata_vault import AtomdataVault
from waxa.data.server_talk import server_talk

N_DATA, N_CAL, P = 4, 6, 3
DATE = "2026-10-09"


def _cal_scan(prefix="cal_"):
    return {
        "format": 1, "name": "cal", "prefix": prefix, "every_n_shots": 2, "at_start": True,
        "alternate": True, "reads_params": ["amp_x"], "n_points_per_event": 2, "n_events": 3,
        "N": N_CAL, "event_starts": [0, 2, 4], "xvarnames": ["t_pulse"],
        "xvars": {"t_pulse": [0.0, 2.5e-6, 2.5e-6, 0.0, 0.0, 2.5e-6]},
        "point_keys": ["t_pulse", "mode"],
        "point_xvars": {"t_pulse": [0.0, 2.5e-6, 2.5e-6, 0.0, 0.0, 2.5e-6], "mode": [0, 1, 1, 0, 0, 1]},
        "xvardims": [N_CAL], "keys": ["apd", "mode", "time", "main_index", "status"],
        "enabled": True, "disabled_reason": "",
        "log": {"points": [{"slot": i, "ok": True} for i in range(N_CAL)], "windows": [{"mode": 0, "v_up": -0.08}],
                "summary": {"n_points": N_CAL, "n_taken": N_CAL, "n_ok": N_CAL, "n_failed": 0}, "notes": []},
    }


def _make_run(folder, run_id, with_cal=True, seed=0):
    """A run file with a 4-shot main scan and, when asked, a 6-point sub-scan."""
    folder.mkdir(parents=True, exist_ok=True)
    path = str(folder / f"{run_id:07d}_{DATE}_12-00-00_FakeExpt.hdf5")
    rng = np.random.default_rng(seed)
    main = {"apd": rng.normal(size=(N_DATA, P))}
    cal = {}
    if with_cal:
        cal = {"cal_apd": rng.normal(size=(N_CAL, P)) + 10.0,
               "cal_mode": np.array([0., 1., 1., 0., 0., 1.]),
               "cal_time": 1000.0 + 6.0 * np.arange(N_CAL) + 100 * seed,
               "cal_main_index": np.array([[-1, -1], [-1, -1], [1, 1], [1, 1], [3, 3], [3, 3]], dtype=np.int64),
               "cal_status": np.ones(N_CAL, dtype=np.int32)}
    containers = dict(main, **cal)
    params = {"t": [1., 1., 2., 2.], "N_repeats": 2, "N_pwa_per_shot": 1, "amp_x": 0.2,
              "t_pulse": 0.0, "mode": 0.0, "N_shots": 2, "N_shots_with_repeats": N_DATA, "N_img": 0}
    shapes = {k: {"shape": tuple(v.shape), "dtype": str(v.dtype), "external": True,
                  "fill": -1 if v.dtype.kind == "i" else np.nan}
              for k, v in containers.items()}
    init = {"capture_images": False, "xvarnames": ["t"], "sort_idx": [], "sort_N": [],
            "datavault_shapes": shapes, "params": params, "camera_params": {},
            "run_date_str": DATE, "run_datetime_str": f"{DATE}_12-00-00", "expt_class": "FakeExpt"}
    ds = make_run_file(path, init, run_id=run_id)
    write_during_run(path, {k: {i: v[i] for i in range(v.shape[0])} for k, v in containers.items()})
    extra = {"cal_scan": json.dumps(_cal_scan())} if with_cal else {}
    end = {"params": params,
           "datavault": {k: {"data": None, "data_gotten": True, "external": True, "final_order": True}
                         for k in containers},
           "sort_idx": [], "sort_N": [], "xvardims": [N_DATA], "N_shots_with_repeats": N_DATA,
           "N_pwa_per_shot": 1, "capture_images": False, "scope_data_taken": False, "scope_data": [],
           "expt_filepath": "", "expt_file_text": "", "params_file_text": "", "base_class_texts": {},
           "extra_file_texts": extra}
    finish_run(ds, path, end, shot_timestamps=[2000.0 + 5.0 * i for i in range(N_DATA)])
    return path, containers


def _load(path):
    return atomdata(path=path, roi_id="auto", lite=False, ignore_images=True)


def test_the_sub_scans_containers_leave_ad_data_and_load_as_ad_cal(tmp_path):
    path, c = _make_run(tmp_path / DATE, 90001)
    ad = _load(path)
    assert ad.data.keys == ["apd"] and ad.has_cal
    cal = ad.cal
    assert cal is ad.cal                                     # built once
    assert cal.xvarnames == ["t_pulse"] and cal.Nvars == 1 and list(cal.xvardims) == [N_CAL]
    assert np.allclose(cal.xvars[0], [0.0, 2.5e-6, 2.5e-6, 0.0, 0.0, 2.5e-6])
    assert cal.data.keys == ["apd", "mode", "time", "main_index", "status"]
    assert np.allclose(cal.data.apd, c["cal_apd"]) and np.allclose(cal.data.mode, c["cal_mode"])
    assert np.array_equal(cal.data.main_index, c["cal_main_index"])
    assert np.allclose(cal.timestamp_shot_end, c["cal_time"])
    assert cal.run_info.run_id == ad.run_info.run_id and cal.p is ad.p
    assert cal.cal_scan["every_n_shots"] == 2 and cal.calibration_log["summary"]["n_ok"] == N_CAL
    assert cal.calibration_log["windows"][0]["v_up"] == -0.08
    assert cal.part == "cal" and cal.roi is None and cal.od is None
    # the points alternate, so repeat statistics pass through rather than average
    assert np.allclose(cal.avg.data.apd, cal.data.apd)
    assert np.array_equal(cal.acquisition_order(), np.arange(N_CAL))
    assert np.array_equal(ad.acquisition_order(), np.arange(N_DATA))
    assert "cal_scan" in ad.run_records


def test_a_run_without_a_sub_scan_has_none(tmp_path):
    path, _ = _make_run(tmp_path / DATE, 90002, with_cal=False)
    ad = _load(path)
    assert not ad.has_cal and ad.cal is None and ad.cal_scan == {}
    assert ad.data.keys == ["apd"]


def test_siblings_slices_and_reshuffles_carry_the_sub_scan(tmp_path):
    path, c = _make_run(tmp_path / DATE, 90003)
    ad = _load(path)
    assert ad.avg.cal is not None and np.allclose(ad.avg.cal.data.apd, c["cal_apd"])
    assert ad.std.has_cal and ad.sem.has_cal
    sl = ad.slice_atomdata(which_xvar_idx=0, xvar_value=2.0)
    assert sl.has_cal and np.allclose(sl.cal.data.apd, c["cal_apd"])
    ad.reshuffle()
    ad.unshuffle()
    assert ad.data.keys == ["apd"] and np.allclose(ad.cal.data.apd, c["cal_apd"])


def test_a_lite_copy_carries_the_sub_scan(tmp_path):
    folder = tmp_path / DATE
    path, c = _make_run(folder, 90004)
    talk = server_talk(data_dir=str(tmp_path))
    ad = atomdata(path=path, ignore_images=True, server_talk=talk)
    ad.save_lite_copy()
    lite = next((tmp_path / "_lite" / DATE).glob("*.hdf5"))
    with h5py.File(lite, "r") as f:
        assert "cal_apd" in f["data"] and "cal_scan" in f.attrs and "cal_apd" not in ad.data.keys
        np.testing.assert_allclose(f["data"]["cal_apd"][()], c["cal_apd"])
        np.testing.assert_array_equal(f["data"]["cal_main_index"][()], c["cal_main_index"])
    ad2 = atomdata(path=str(lite), ignore_images=True, server_talk=talk)
    assert ad2.has_cal and np.allclose(ad2.cal.data.apd, c["cal_apd"]) and ad2.data.keys == ["apd"]


def test_a_vault_of_sub_scans_and_a_main_vaults_lazy_cal(tmp_path):
    p1, c1 = _make_run(tmp_path / DATE, 90005, seed=1)
    p2, c2 = _make_run(tmp_path / DATE, 90006, seed=2)
    p3, _ = _make_run(tmp_path / DATE, 90007, with_cal=False, seed=3)
    ads = [_load(p) for p in (p1, p2, p3)]
    cal_vault = AtomdataVault([a.cal for a in ads if a.cal is not None], lite=False,
                              ignore_images=True, uniform_roi=False, sort=False)
    assert cal_vault.data.apd.shape == (2 * N_CAL, P)
    assert cal_vault.shot_run_id.tolist() == [90005] * N_CAL + [90006] * N_CAL
    assert np.allclose(cal_vault.timestamp_shot_end, np.concatenate([c1["cal_time"], c2["cal_time"]]))
    assert np.allclose(cal_vault.data.apd, np.concatenate([c1["cal_apd"], c2["cal_apd"]]))
    main = AtomdataVault(ads, lite=False, ignore_images=True, uniform_roi=False)
    assert main.data.keys == ["apd"] and main.data.apd.shape == (3 * N_DATA, P)
    assert main.cal is not None and main.cal.data.apd.shape == (2 * N_CAL, P)
    assert main.cal is main.cal                                  # cached
    main.drop_runs([90005])
    assert main.cal.data.apd.shape == (N_CAL, P) and main.cal.shot_run_id.tolist() == [90006] * N_CAL
    main.drop_runs([90006])
    assert main.cal is None
