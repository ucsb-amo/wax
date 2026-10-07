"""Lite copies leave the camera-stream frame stacks behind, their per-shot
records kept, unless asked to keep the frames -- on the in-memory path
(atomdata.save_lite_copy) and the disk path (server_talk.create_lite_copy).
Every file is a fresh HDF5 in tmp_path."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np

from synthetic_run_file import make_loadable_run

DATE = "2026-10-05"
N, H, W = 4, 10, 12
ROIX, ROIY = [2, 8], [1, 7]


def test_stream_frame_keys_are_the_ones_with_a_record_beside_them():
    from waxa.data.camera_frames import stream_frame_keys
    assert stream_frame_keys(["a", "a_meta", "b", "b_seq", "c", "d_meta", "apd"]) == ["a", "b"]
    assert stream_frame_keys([]) == []


def _raw_run(tmp_path, run_id=71234):
    folder = tmp_path / DATE
    folder.mkdir(exist_ok=True)
    path = str(folder / f"{run_id:07d}_{DATE}_12-00-00_FakeExpt.hdf5")
    rng = np.random.default_rng(run_id)
    with h5py.File(path, "w") as f:
        ri = f.create_group("run_info")
        ri["run_id"] = run_id
        ri["run_date_str"] = DATE
        ri["run_datetime_str"] = f"{DATE}_12-00-00"
        ri["expt_class"] = "FakeExpt"
        f.create_group("params")["N_pwa_per_shot"] = 1
        d = f.create_group("data")
        d["images"] = rng.integers(0, 4000, size=(N, H, W), dtype=np.uint16)
        d["apd"] = np.arange(float(N))
        d["img_x"] = rng.integers(0, 255, size=(N, 3, 3), dtype=np.uint8)
        d["img_x_meta"] = np.ones((N, 7))
        d["img_y"] = rng.integers(0, 255, size=(N, 3, 3), dtype=np.uint8)
        d["img_y_seq"] = np.arange(N)                      # the 09-30 / 10-01 record form
        f.attrs["run_complete"] = True
    return path


def test_disk_path_drops_the_frames_and_keeps_the_records(tmp_path):
    from waxa.data.server_talk import server_talk
    raw = _raw_run(tmp_path)
    talk = server_talk(data_dir=str(tmp_path))
    lite = talk.create_lite_copy(71234, roix=ROIX, roiy=ROIY, path=raw)
    with h5py.File(lite, "r") as f, h5py.File(raw, "r") as src:
        keys = set(f["data"].keys())
        assert "img_x" not in keys and "img_y" not in keys
        assert {"img_x_meta", "img_y_seq", "apd", "images"} <= keys
        assert json.loads(f.attrs["lite_dropped_keys"]) == ["img_x", "img_y"]
        np.testing.assert_array_equal(f["data"]["images"][()], src["data"]["images"][:, 1:7, 2:8])
        np.testing.assert_array_equal(f["data"]["img_x_meta"][()], np.ones((N, 7)))
    lite2 = talk.create_lite_copy(71234, roix=ROIX, roiy=ROIY, path=raw, include_streams=True)
    with h5py.File(lite2, "r") as f, h5py.File(raw, "r") as src:
        np.testing.assert_array_equal(f["data"]["img_x"][()], src["data"]["img_x"][()])
        np.testing.assert_array_equal(f["data"]["img_y"][()], src["data"]["img_y"][()])
        assert "lite_dropped_keys" not in f.attrs


def test_memory_path_drops_the_frames_and_the_lite_copy_still_answers(tmp_path, capsys):
    from waxa.atomdata import atomdata
    from waxa.data.server_talk import server_talk
    from waxa.data.camera_frames import frames_present
    frames = np.random.default_rng(5).integers(0, 255, size=(N, 3, 3), dtype=np.uint8)
    meta = np.ones((N, 7))
    meta[1, 0] = np.nan                                    # shot 1 has no frame
    path = make_loadable_run(tmp_path / DATE,
                             {"img_x": frames, "img_x_meta": meta, "apd": np.arange(float(N))})
    talk = server_talk(data_dir=str(tmp_path))
    ad = atomdata(path=path, ignore_images=True, server_talk=talk)

    ad.save_lite_copy()
    assert "1 camera-stream frame stacks left out" in capsys.readouterr().out
    lite = next((tmp_path / "_lite" / DATE).glob("*.hdf5"))
    with h5py.File(lite, "r") as f:
        assert "img_x" not in f["data"] and "img_x_meta" in f["data"] and "apd" in f["data"]
        assert json.loads(f.attrs["lite_dropped_keys"]) == ["img_x"]
    ad2 = atomdata(path=str(lite), ignore_images=True, server_talk=talk)
    assert json.loads(ad2.run_records["lite_dropped_keys"]) == ["img_x"]
    assert "img_x" not in ad2.data.keys
    np.testing.assert_array_equal(frames_present(ad2, "img_x"), [True, False, True, True])
    np.testing.assert_array_equal(ad2.data.apd, np.arange(float(N)))

    ad.save_lite_copy(include_streams=True)
    assert "left out" not in capsys.readouterr().out
    with h5py.File(lite, "r") as f:
        np.testing.assert_array_equal(f["data"]["img_x"][()], frames)
        assert "lite_dropped_keys" not in f.attrs
