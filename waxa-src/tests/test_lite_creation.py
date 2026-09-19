"""Lite-copy creation from the data browser: headless crop, no partial files,
and the ROI dialog refusing to open off the Qt main thread (which froze the
browser)."""

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

from waxa.data.server_talk import server_talk

RUN_ID = 71234
DATE = "2026-09-19"
N_IMG, H, W = 6, 40, 50
ROIX, ROIY = [10, 30], [5, 25]


def _make_run(data_dir, run_id=RUN_ID, images=True, saved_roi=None):
    folder = os.path.join(data_dir, DATE)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{run_id:07d}_{DATE}_12-00-00_FakeExpt.hdf5")
    rng = np.random.default_rng(run_id)
    with h5py.File(path, "w") as f:
        ri = f.create_group("run_info")
        ri["run_id"] = run_id
        ri["run_date_str"] = DATE
        ri["run_datetime_str"] = f"{DATE}_12-00-00"
        ri["expt_class"] = "FakeExpt"
        f.create_group("params")["N_pwa_per_shot"] = 1
        d = f.create_group("data")
        if images:
            d["images"] = rng.integers(0, 4000, size=(N_IMG, H, W), dtype=np.uint16)
        d["apd"] = np.arange(4.0)
        scope = d.create_group("scope_data").create_group("scope0")
        scope["t"] = np.linspace(0, 1, 16)
        f.attrs["run_complete"] = True
        if saved_roi is not None:
            f.attrs["roix"], f.attrs["roiy"] = saved_roi
    return path


def _raw_snapshot(path):
    with h5py.File(path, "r") as f:
        return dict(f.attrs), f["data"]["images"][()] if "images" in f["data"] else None


def test_explicit_roi_crops_without_touching_raw(tmp_path):
    raw = _make_run(str(tmp_path))
    attrs_before, imgs = _raw_snapshot(raw)
    talk = server_talk(data_dir=str(tmp_path))

    lite = talk.create_lite_copy(RUN_ID, roix=ROIX, roiy=ROIY, path=raw)

    assert lite is not None and os.path.exists(lite)
    assert not os.path.exists(lite + ".tmp")
    with h5py.File(lite, "r") as f:
        out = f["data"]["images"][()]
        np.testing.assert_array_equal(out, imgs[:, 5:25, 10:30])
        assert list(f.attrs["roix"]) == [0, 20] and list(f.attrs["roiy"]) == [0, 20]
        assert list(f.attrs["lite_source_roix"]) == ROIX
        assert f["data"]["scope_data"]["scope0"]["t"].dtype == np.float32
        np.testing.assert_array_equal(f["data"]["apd"][()], np.arange(4.0))
    attrs_after, imgs_after = _raw_snapshot(raw)
    assert attrs_after.keys() == attrs_before.keys()
    np.testing.assert_array_equal(imgs_after, imgs)


def test_cancel_leaves_no_file(tmp_path):
    raw = _make_run(str(tmp_path))
    talk = server_talk(data_dir=str(tmp_path))
    calls = []

    def cancel_after_two():
        calls.append(1)
        return len(calls) > 2

    assert talk.create_lite_copy(RUN_ID, roix=ROIX, roiy=ROIY, path=raw,
                                 should_cancel=cancel_after_two) is None
    lite_dir = os.path.join(str(tmp_path), "_lite", DATE)
    assert os.listdir(lite_dir) == []


def test_bad_roi_raises_and_leaves_no_file(tmp_path):
    raw = _make_run(str(tmp_path))
    talk = server_talk(data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="does not fit"):
        talk.create_lite_copy(RUN_ID, roix=[0, W + 5], roiy=ROIY, path=raw)
    assert os.listdir(os.path.join(str(tmp_path), "_lite", DATE)) == []


def test_run_without_images(tmp_path):
    raw = _make_run(str(tmp_path), images=False, saved_roi=(ROIX, ROIY))
    lite = server_talk(data_dir=str(tmp_path)).create_lite_copy(
        RUN_ID, roix=ROIX, roiy=ROIY, path=raw)
    with h5py.File(lite, "r") as f:
        assert "images" not in f["data"]
        assert not f.attrs["has_images"]
        assert "roix" not in f.attrs


def test_read_saved_roi(tmp_path):
    from waxa.roi import read_saved_roi

    assert read_saved_roi(_make_run(str(tmp_path), run_id=1)) is None
    assert read_saved_roi(_make_run(str(tmp_path), run_id=2, saved_roi=(ROIX, ROIY))) == (ROIX, ROIY)
    blank = ([-1, -1], [-1, -1])
    assert read_saved_roi(_make_run(str(tmp_path), run_id=3, saved_roi=blank)) is None


# --- Qt: worker, thread guard, dialog pre-fill ------------------------------

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_worker_batch_continues_past_a_failure(tmp_path, qapp):
    from waxa.browser.scanner import LiteCreateWorker

    good = _make_run(str(tmp_path), run_id=10)
    _make_run(str(tmp_path), run_id=11, images=True)
    # run 11 with an ROI that does not fit its frame size
    with h5py.File(_make_run(str(tmp_path), run_id=12), "a") as f:
        del f["data"]["images"]
        f["data"]["images"] = np.zeros((2, 8, 8), dtype=np.uint16)
    runs = [(10, good), (12, ""), (11, "")]
    worker = LiteCreateWorker(str(tmp_path), runs, ROIX, ROIY)
    created, errors, done = [], [], []
    worker.created.connect(lambda rid, p: created.append(rid))
    worker.error.connect(errors.append)
    worker.completed.connect(lambda n, t, c: done.append((n, t, c)))

    worker.run()  # synchronously, on this thread

    assert created == [10, 11]
    assert len(errors) == 1 and "run 12" in errors[0]
    assert done == [(2, 3, False)]


def test_worker_cancel_before_start(tmp_path, qapp):
    from waxa.browser.scanner import LiteCreateWorker

    raw = _make_run(str(tmp_path))
    worker = LiteCreateWorker(str(tmp_path), [(RUN_ID, raw)], ROIX, ROIY)
    done = []
    worker.completed.connect(lambda n, t, c: done.append((n, t, c)))
    worker.cancel()
    worker.run()
    assert done == [(0, 1, True)]
    assert not os.path.exists(os.path.join(str(tmp_path), "_lite"))


def _in_thread(fn):
    box = {}

    def target():
        try:
            fn()
        except BaseException as exc:
            box["exc"] = exc

    t = threading.Thread(target=target)
    t.start()
    t.join(10)
    assert not t.is_alive(), "call hung instead of refusing"
    return box.get("exc")


def test_roi_dialog_refuses_worker_thread(qapp):
    from waxa.roi import pick_roi, roi_creator

    exc = _in_thread(lambda: pick_roi(1, file_path="unused.hdf5"))
    assert isinstance(exc, RuntimeError)

    creator = roi_creator(1, "", None, precomputed_ods=np.zeros((1, 8, 8)))
    exc = _in_thread(creator.get_roi_rectangle)
    assert isinstance(exc, RuntimeError) and "main thread" in str(exc)


def test_dialog_prefills_initial_roi(qapp):
    from waxa.roi import SAVED_PRESET_KEY, _RoiSelectorDialog, roi_creator

    creator = roi_creator(1, "", None, precomputed_ods=np.ones((2, H, W)),
                          initial_roi=(ROIX, ROIY))
    creator.cmap_juice_factor = 1.0  # set by get_roi_rectangle before the dialog
    dialog = _RoiSelectorDialog(creator, [])
    try:
        assert dialog.active_roi_source == f"preset:{SAVED_PRESET_KEY}"
        x0, x1, y0, y1 = dialog.active_roi_bounds
        assert [x0, x1] == ROIX and [y0, y1] == ROIY
    finally:
        dialog.close()
        creator.close()
