"""waxx.util.live_od.data: what liveOD does to a run's data file, and the server
and DataHandler driving it. Every file here lives in pytest's tmp_path; the saver
is a stand-in (live_od_data_fakes) and nothing opens a socket or a camera.
"""
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from live_od_data_fakes import FakeSaver, patch_payload_stash, new_data_file as _reserve


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def calls(monkeypatch):
    from waxx.util.live_od.data import run_file
    recorded = []
    patch_payload_stash(monkeypatch, run_file, recorded)
    return recorded


# ----------------------------------------------------------------------
# RunFile
# ----------------------------------------------------------------------

def test_run_file_reserves_then_saves_and_forgets_the_path(tmp_path, calls):
    from waxx.util.live_od.data.run_file import RunFile
    saver = FakeSaver(tmp_path)
    rf = RunFile(saver)
    assert rf.writer_done.is_set() and not rf.pending

    rf.begin(save_data=True, has_writer=False)
    run_id, path = rf.reserve({})
    assert run_id == 101 and os.path.exists(path) and rf.pending

    rf.save({}, run_id, [1.0, 2.0])
    assert saver.saved == [(path, [1.0, 2.0])]
    assert calls == ["stash", ("clear", "stash.pkl")]        # stashed before the save, cleared after
    assert rf.filepath == "" and not rf.pending

    rf.discard()                                             # a late reset: the saved run is out of reach
    assert os.path.exists(path)


def test_run_file_save_waits_for_the_image_writer(tmp_path, calls, monkeypatch):
    from waxx.util.live_od.data import run_file
    rf = run_file.RunFile(FakeSaver(tmp_path))
    rf.begin(save_data=True, has_writer=True)
    run_id, _ = rf.reserve({})
    assert not rf.writer_done.is_set()
    threading.Timer(0.2, rf.writer_finished).start()
    t0 = time.time()
    rf.save({}, run_id, [])
    assert time.time() - t0 >= 0.15


def test_run_file_failed_save_keeps_the_stash_and_says_how_to_finish(tmp_path, calls):
    from waxx.util.live_od.data.run_file import RunFile, RunFileSaveError
    rf = RunFile(FakeSaver(tmp_path, fail_save=True))
    rf.begin(save_data=True, has_writer=False)
    run_id, path = rf.reserve({})
    with pytest.raises(RunFileSaveError) as err:
        rf.save({}, run_id, [])
    assert err.value.cause == "drive went away"
    assert "stash.pkl" in str(err.value) and "preserved" in str(err.value)
    assert calls == ["stash"]                                # not cleared
    assert rf.filepath == path and os.path.exists(path)


def test_run_file_failed_reserve_releases_the_writer_gate(tmp_path):
    from waxx.util.live_od.data.run_file import RunFile
    rf = RunFile(FakeSaver(tmp_path, fail_reserve=True))
    rf.begin(save_data=True, has_writer=True)
    with pytest.raises(OSError):
        rf.reserve({})
    assert rf.writer_done.is_set() and rf.filepath == "" and not rf.pending


def test_run_file_discard_deletes_only_the_unsaved_run(tmp_path):
    from waxx.util.live_od.data.run_file import RunFile
    rf = RunFile(FakeSaver(tmp_path))
    rf.begin(save_data=True, has_writer=False)
    _, path = rf.reserve({})
    bystander = tmp_path / "0000100.hdf5"
    bystander.write_bytes(b"an earlier run")
    rf.discard()
    assert not os.path.exists(path) and bystander.exists() and rf.filepath == ""
    rf.discard()                                             # nothing left: a no-op


def test_run_file_without_save_data_has_nothing_to_do(tmp_path):
    from waxx.util.live_od.data.run_file import RunFile
    rf = RunFile(FakeSaver(tmp_path))
    rf.begin(save_data=False, has_writer=True)
    assert not rf.pending and rf.filepath == ""
    rf.discard()
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------
# The server, driving a RunFile
# ----------------------------------------------------------------------

def _init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 2, "expt_class": "scope_test"}
    msg.update(kw)
    return msg


@pytest.fixture
def server(app, tmp_path, calls):
    from waxx.util.live_od.live_od_server import LiveODServer
    saver = FakeSaver(tmp_path)
    srv = LiveODServer(server_talk=None, data_saver=saver)   # never started: no socket, no beacon
    states = []
    srv.run_state_signal.connect(lambda state, detail: states.append(state))
    return srv, saver, states


def test_server_saves_a_run(server):
    srv, saver, states = server
    reply = srv._handle_init_run(_init_msg())
    assert reply["ok"] and reply["run_id"] == 101 and os.path.exists(reply["filepath"])
    assert srv._current_filepath == reply["filepath"] and srv._current_save_data   # the old names still read
    srv._handle_shot_complete({"shot_idx": 0, "N_shots_total": 1})
    assert srv._handle_end_run({})["ok"]
    assert [path for path, _ in saver.saved] == [reply["filepath"]] and len(saver.saved[0][1]) == 1
    assert states == ["running", "saving", "saved"]
    assert srv._current_filepath == "" and os.path.exists(reply["filepath"])


def test_server_reports_a_failed_save_and_ends_the_run(server):
    srv, saver, states = server
    saver.fail_save = True
    path = srv._handle_init_run(_init_msg())["filepath"]
    reply = srv._handle_end_run({})
    assert not reply["ok"] and "drive went away" in reply["error"] and "stash.pkl" in reply["error"]
    assert states[-1] == "error" and srv._run_in_progress is False
    assert os.path.exists(path)                              # a failed save deletes nothing


def test_server_deletes_the_file_of_a_reset_run_on_abort(server):
    srv, saver, states = server
    path = srv._handle_init_run(_init_msg())["filepath"]
    srv._handle_reset({})
    srv._handle_abort_run({})
    assert not os.path.exists(path) and saver.saved == []
    assert states[-1] == "aborted" and srv._run_in_progress is False


def test_server_deletes_the_file_of_a_reset_run_at_end_run(server):
    srv, saver, states = server
    path = srv._handle_init_run(_init_msg())["filepath"]
    srv._handle_reset({})
    assert srv._handle_end_run({})["ok"]
    assert not os.path.exists(path) and saver.saved == []


def test_server_finalizes_a_reset_whose_experiment_died_at_the_next_init(server):
    srv, saver, states = server
    first = srv._handle_init_run(_init_msg())["filepath"]
    srv._handle_reset({})                                    # ...and the experiment never answers
    second = srv._handle_init_run(_init_msg())["filepath"]
    assert not os.path.exists(first) and os.path.exists(second)
    assert srv._handle_end_run({})["ok"] and [p for p, _ in saver.saved] == [second]


def test_server_camera_run_gates_the_save_on_the_writer(server):
    srv, saver, states = server
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    assert not srv._data_handler_done_event.is_set()
    threading.Timer(0.2, srv.on_data_handler_done).start()
    t0 = time.time()
    assert srv._handle_end_run({})["ok"]
    assert time.time() - t0 >= 0.15 and len(saver.saved) == 1


# ----------------------------------------------------------------------
# ImageWriter, and DataHandler feeding it
# ----------------------------------------------------------------------

def _spin(app, condition, timeout=10.0):
    t0 = time.time()
    while not condition() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return condition()


def _frames(n):
    return [np.full((4, 5), i, np.uint16) for i in range(n)]


def test_image_writer_writes_the_images_it_is_given(app, tmp_path):
    from waxx.util.live_od.data.image_writer import ImageWriter
    _, path = _reserve(tmp_path)
    done, failed = threading.Event(), []
    writer = ImageWriter(path)
    assert not writer.started
    writer.start(n_img=3, check_interrupt_method=lambda: False,
                 done_signal=done.set, failed_signal=failed.append)
    for i, img in enumerate(_frames(3)):
        writer.put(img, i, 100.0 + i)
    writer.finish(interrupted=False)
    assert _spin(app, done.is_set) and failed == []
    writer._worker.wait()
    with h5py.File(path, "r") as f:
        assert f["data/images"].shape == (3, 4, 5)
        assert [int(f["data/images"][i, 0, 0]) for i in range(3)] == [0, 1, 2]
        assert list(f["data/image_timestamps"]) == [100.0, 101.0, 102.0]


def test_image_writer_interrupted_drains_without_writing(app, tmp_path):
    from waxx.util.live_od.data.image_writer import ImageWriter
    _, path = _reserve(tmp_path)
    done = threading.Event()
    writer = ImageWriter(path)
    writer.start(n_img=3, check_interrupt_method=lambda: False,
                 done_signal=done.set, failed_signal=lambda reason: None)
    writer._worker.interrupted = True
    for i, img in enumerate(_frames(3)):
        writer.put(img, i, 0.0)
    writer.finish(interrupted=True)
    assert _spin(app, done.is_set)
    writer._worker.wait()
    with h5py.File(path, "r") as f:
        assert "images" not in f["data"]


def test_image_writer_discard_deletes_the_file_unless_told_not_to(tmp_path):
    from waxx.util.live_od.data.image_writer import ImageWriter
    _, path = _reserve(tmp_path)
    ImageWriter(path).discard(False)
    assert os.path.exists(path)
    ImageWriter(path).discard()
    assert not os.path.exists(path)
    ImageWriter(path).discard()                              # already gone: a no-op
    ImageWriter("").discard()


def test_data_handler_shows_and_saves_every_image(app, tmp_path, monkeypatch):
    from queue import Queue
    from waxx.util.live_od import config as live_od_config
    from waxx.util.live_od.camera_mother import DataHandler, SaveWorker
    from waxx.util.live_od.data import image_writer
    assert SaveWorker is image_writer.SaveWorker             # the old import path, the same class

    monkeypatch.setattr(live_od_config, "_active", live_od_config.LiveODConfig())
    _, path = _reserve(tmp_path)
    queue = Queue()
    handler = DataHandler(queue, path, save_data=True, imaging_type=0, camera_key="cam_a",
                          params_payload={"N_img": 3}, n_img=3, n_shots=1, n_pwa_per_shot=3)
    shown, done = [], threading.Event()
    handler.got_image_from_queue.connect(lambda img: shown.append(int(img[0, 0])))
    handler.done_writing_signal.connect(done.set)
    handler.get_img_number(3, 1, 3)
    for i, img in enumerate(_frames(3)):
        queue.put((img, None, i))
    handler.start()
    assert _spin(app, done.is_set)
    handler.wait()
    handler.writer._worker.wait()
    assert _spin(app, lambda: len(shown) == 3) and shown == [0, 1, 2]
    with h5py.File(path, "r") as f:
        assert [int(f["data/images"][i, 0, 0]) for i in range(3)] == [0, 1, 2]


def test_data_handler_without_save_data_never_starts_a_writer(app, monkeypatch):
    from queue import Queue
    from waxx.util.live_od import config as live_od_config
    from waxx.util.live_od.camera_mother import DataHandler
    monkeypatch.setattr(live_od_config, "_active", live_od_config.LiveODConfig())
    queue = Queue()
    handler = DataHandler(queue, "", save_data=False, imaging_type=0, n_img=2, n_shots=1, n_pwa_per_shot=2)
    done = threading.Event()
    handler.done_writing_signal.connect(done.set)
    handler.get_img_number(2, 1, 2)
    for i, img in enumerate(_frames(2)):
        queue.put((img, None, i))
    handler.start()
    assert _spin(app, done.is_set)
    handler.wait()
    assert not handler.writer.started
