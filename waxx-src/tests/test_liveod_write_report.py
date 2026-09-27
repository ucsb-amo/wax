"""B3: every frame handed to the image writer is accounted for, and the account
reaches END_RUN's completeness decision. Nothing the camera delivered is dropped
for looking wrong: frames that are not the shape/dtype the run declared are all
written and the run is marked incomplete; only a frame that cannot go into the
run's dataset (a mid-run change) is not written, and is counted; write errors are
counted. Every file lives in tmp_path; the saver is the stand-in from
live_od_data_fakes, and the writer and RunFile are the real ones.

Passes once the guarded proposals for waxx/util/live_od/data/{image_writer,
run_file}.py (LIVEOD, B3, second version) are applied.
"""
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from live_od_data_fakes import FakeSaver, patch_payload_stash
import liveod_qt_helpers as qt


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def calls(monkeypatch):
    from waxx.util.live_od.data import run_file
    recorded = []
    patch_payload_stash(monkeypatch, run_file, recorded)
    return recorded


def _spin(app, condition, timeout=10.0):
    t0 = time.time()
    while not condition() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return condition()


def _camera_run(folder, shape=(3, 4, 5), dtype="uint16"):
    """(RunFile, saver, run_id, path): a camera run's file as INIT_RUN leaves it,
    declaring its images' shape and dtype (None: declaring nothing, as a file from
    an older path)."""
    from waxx.util.live_od.data.run_file import RunFile
    saver = FakeSaver(folder)
    rf = RunFile(saver)
    rf.begin(save_data=True, has_writer=True)
    run_id, path = rf.reserve({})
    if shape is not None:
        with h5py.File(path, "r+") as f:
            f.attrs["images_shape"] = list(shape)
            f.attrs["images_dtype"] = dtype
    return rf, saver, run_id, path


def _write(app, rf, path, frames, n_img=3, slots=None):
    """Feed the frames to a real ImageWriter (frame i into slot ``slots[i]``),
    wait for it and open the RunFile's writer gate; its thread is joined
    whatever happens."""
    from waxx.util.live_od.data.image_writer import ImageWriter
    done = threading.Event()
    writer = ImageWriter(path)
    writer.start(n_img=n_img, check_interrupt_method=lambda: False,
                 done_signal=done.set, failed_signal=lambda reason: None)
    try:
        for i, img in enumerate(frames):
            writer.put(img, i if slots is None else slots[i], 100.0 + i)
        writer.finish(interrupted=False)
        assert _spin(app, done.is_set)
    finally:
        if writer.started and not done.is_set():
            writer._worker.interrupted = True                   # not finished: stop it, write nothing more
            writer._save_queue.put(None)
        if writer.started:
            assert qt.join_or_keep(writer._worker) == ""
    rf.writer_finished()
    return writer


def _images(path):
    with h5py.File(path, "r") as f:
        return f["data/images"][()]


def test_a_matching_run_is_complete(app, tmp_path, calls):
    rf, saver, run_id, path = _camera_run(tmp_path)
    _write(app, rf, path, [np.full((4, 5), i, np.uint16) for i in range(3)])
    assert [int(im[0, 0]) for im in _images(path)] == [0, 1, 2]
    assert rf.save({}, run_id, [], images_expected=3, images_received=3) is None


def test_frames_not_as_declared_are_all_written_and_the_run_is_incomplete(app, tmp_path, calls):
    rf, saver, run_id, path = _camera_run(tmp_path, shape=(3, 4, 5), dtype="uint16")
    _write(app, rf, path, [np.full((2, 2), i, np.uint8) for i in range(3)])
    images = _images(path)
    assert images.shape == (3, 2, 2) and images.dtype == np.uint8      # the frames as they came
    assert [int(im[0, 0]) for im in images] == [0, 1, 2]               # every one of them
    incomplete = rf.save({}, run_id, [], images_expected=3, images_received=3)
    assert incomplete["reason"] == "frames are (2, 2) uint8 but the run declared (4, 5) uint16"
    assert saver.incomplete == incomplete


def test_a_frame_that_changes_mid_run_is_not_written_and_counted(app, tmp_path, calls):
    rf, saver, run_id, path = _camera_run(tmp_path)
    _write(app, rf, path, [np.full((4, 5), 1, np.uint16),
                           np.zeros((2, 2), np.uint16),                # another shape
                           np.full((4, 5), 3, np.uint8)])              # another dtype
    images = _images(path)
    assert images.shape == (3, 4, 5) and int(images[0, 0, 0]) == 1
    incomplete = rf.save({}, run_id, [], images_expected=3, images_received=3)
    assert incomplete["reason"] == (
        "1/3 images written to the file (2 not written; first: frame 1: (2, 2) uint16 does not "
        "fit the run's images (4, 5) uint16; not written)")


def test_a_write_error_is_counted(app, tmp_path, calls):
    from waxx.util.live_od.data.image_writer import take_write_report
    rf, saver, run_id, path = _camera_run(tmp_path)
    _write(app, rf, path, [np.full((4, 5), i, np.uint16) for i in range(2)],
           slots=[0, 7])                                               # no slot 7 in a 3-frame run
    written, not_written, mismatches = take_write_report(path).snapshot()
    assert written == 1 and mismatches == []
    assert len(not_written) == 1 and not_written[0].startswith("frame 7: write failed")


def test_a_write_error_makes_the_run_incomplete(app, tmp_path, calls):
    rf, saver, run_id, path = _camera_run(tmp_path)
    _write(app, rf, path, [np.full((4, 5), i, np.uint16) for i in range(3)], slots=[0, 1, 9])
    incomplete = rf.save({}, run_id, [], images_expected=3, images_received=3)
    assert incomplete["reason"].startswith("2/3 images written to the file (1 not written; "
                                           "first: frame 9: write failed")


def test_a_file_that_declares_nothing_is_written_as_before(app, tmp_path, calls):
    rf, saver, run_id, path = _camera_run(tmp_path, shape=None)
    _write(app, rf, path, [np.full((4, 5), i, np.uint16) for i in range(3)])
    assert _images(path).shape == (3, 4, 5)
    assert rf.save({}, run_id, [], images_expected=3, images_received=3) is None


def test_without_a_writer_nothing_changes(tmp_path, calls):
    from waxx.util.live_od.data.run_file import RunFile
    rf = RunFile(FakeSaver(tmp_path))
    rf.begin(save_data=True, has_writer=False)
    run_id, _ = rf.reserve({})
    assert rf.save({}, run_id, [], images_expected=0, images_received=0) is None


def test_a_discarded_run_drops_its_report(tmp_path):
    from waxx.util.live_od.data.image_writer import new_write_report, take_write_report
    from waxx.util.live_od.data.run_file import RunFile
    rf = RunFile(FakeSaver(tmp_path))
    rf.begin(save_data=True, has_writer=False)
    _, path = rf.reserve({})
    new_write_report(path)
    rf.discard()
    assert take_write_report(path) is None
