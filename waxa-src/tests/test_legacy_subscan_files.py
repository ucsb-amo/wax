"""Run files from the interleaved readout sub-scan era (waxx SubScan,
2026-10-10 .. 2026-11, removed 2026-11-04) still load: the sub-scan's cal_*
containers stay out of ad.data (their first axis is not the scan grid) and are
kept raw in ad.legacy_cal_data; slices, reshuffles, lite copies and vaults are
unaffected. Files are made the way liveOD made them, in pytest's temp folder."""
import json

import h5py
import numpy as np

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


def test_legacy_sub_scan_containers_stay_out_of_ad_data(tmp_path):
    path, c = _make_run(tmp_path / DATE, 90001)
    ad = _load(path)
    assert ad.data.keys == ["apd"]
    assert sorted(ad.legacy_cal_data) == sorted(k for k in c if k.startswith("cal_"))
    np.testing.assert_allclose(ad.legacy_cal_data["cal_apd"], c["cal_apd"])
    assert ad.cal_scan["prefix"] == "cal_" and ad.cal_scan["every_n_shots"] == 2
    assert not hasattr(ad, "cal") and not hasattr(ad, "has_cal")
    assert np.array_equal(ad.acquisition_order(), np.arange(N_DATA))


def test_a_run_without_a_sub_scan_has_no_legacy_data(tmp_path):
    path, _ = _make_run(tmp_path / DATE, 90002, with_cal=False)
    ad = _load(path)
    assert ad.cal_scan == {} and ad.legacy_cal_data == {}
    assert ad.data.keys == ["apd"]


def test_siblings_slices_and_reshuffles_ignore_the_legacy_containers(tmp_path):
    path, c = _make_run(tmp_path / DATE, 90003)
    ad = _load(path)
    assert ad.avg.data.keys == ["apd"]
    sl = ad.slice_atomdata(which_xvar_idx=0, xvar_value=2.0)
    assert sl.data.keys == ["apd"]
    ad.reshuffle()
    ad.unshuffle()
    assert ad.data.keys == ["apd"]
    np.testing.assert_allclose(ad.legacy_cal_data["cal_apd"], c["cal_apd"])


def test_a_lite_copy_keeps_the_legacy_containers(tmp_path):
    folder = tmp_path / DATE
    path, c = _make_run(folder, 90004)
    talk = server_talk(data_dir=str(tmp_path))
    ad = atomdata(path=path, ignore_images=True, server_talk=talk)
    ad.save_lite_copy()
    lite = next((tmp_path / "_lite" / DATE).glob("*.hdf5"))
    with h5py.File(lite, "r") as f:
        assert "cal_apd" in f["data"] and "cal_scan" in f.attrs
        np.testing.assert_allclose(f["data"]["cal_apd"][()], c["cal_apd"])
    ad2 = atomdata(path=str(lite), ignore_images=True, server_talk=talk)
    assert ad2.data.keys == ["apd"]
    np.testing.assert_allclose(ad2.legacy_cal_data["cal_apd"], c["cal_apd"])


def test_a_vault_of_legacy_and_plain_runs(tmp_path):
    paths = [_make_run(tmp_path / DATE, 90005, seed=1)[0],
             _make_run(tmp_path / DATE, 90006, with_cal=False, seed=2)[0]]
    main = AtomdataVault([_load(p) for p in paths], lite=False, ignore_images=True, uniform_roi=False)
    assert main.data.keys == ["apd"] and main.data.apd.shape == (2 * N_DATA, P)
    assert not hasattr(main, "cal")
