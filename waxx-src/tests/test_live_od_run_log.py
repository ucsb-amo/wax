"""liveOD's run-tagged log buffer and GET_LOG, and what the server and camera
threads do with a run whose frames do not all arrive: the DataHandler stops when
the grab ends, the save reports the shortfall, and the run's outcome says so.

Every file here lives in pytest's tmp_path; the saver is a stand-in
(live_od_data_fakes) and nothing opens a socket or a camera.
"""
import logging
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


@pytest.fixture
def buffer():
    from waxx.util.live_od.log import get_log_buffer
    buf = get_log_buffer()
    buf.clear()
    yield buf
    buf.clear()


def _init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 1, "expt_class": "smear_test"}
    msg.update(kw)
    return msg


@pytest.fixture
def server(app, tmp_path, calls, buffer):
    from waxx.util.live_od.live_od_server import LiveODServer
    saver = FakeSaver(tmp_path)
    srv = LiveODServer(server_talk=None, data_saver=saver)   # never started: no socket, no beacon
    states = []
    srv.run_state_signal.connect(lambda state, detail: states.append((state, detail)))
    return srv, saver, states


def _msgs(records):
    return [r["msg"] for r in records]


# ----------------------------------------------------------------------
# LogBuffer
# ----------------------------------------------------------------------

def test_log_buffer_tags_records_with_the_run_that_was_current(buffer):
    from waxx.util.live_od.log import get_logger
    log = get_logger("test")
    log.info("before any run")
    seq_a = buffer.begin_run(101, "expt_a")
    log.info("a one")
    log.debug("a two")
    seq_b = buffer.begin_run(0, "expt_b")                    # an unsaved run
    log.warning("b one")
    t_mid = time.time()
    time.sleep(0.01)
    log.error("b two")

    assert (seq_a, seq_b) == (1, 2)
    assert _msgs(buffer.records(run_id=101)) == ["a one", "a two"]
    assert _msgs(buffer.records()) == ["b one", "b two"]                  # the current run
    assert _msgs(buffer.records(seq=1)) == ["a one", "a two"]
    assert _msgs(buffer.records(run_id="all")) == ["before any run", "a one", "a two", "b one", "b two"]
    assert _msgs(buffer.records(run_id=101, min_level=logging.INFO)) == ["a one"]
    assert _msgs(buffer.records(since=t_mid)) == ["b two"]
    assert _msgs(buffer.records(run_id="all", limit=2)) == ["b one", "b two"]
    assert buffer.records(run_id=999) == []
    assert all(r["run_id"] == 101 and r["seq"] == 1 for r in buffer.records(run_id=101))


def test_log_buffer_keeps_a_run_index_with_outcomes(buffer):
    buffer.begin_run(101, "expt_a", n_shots_expected=4)
    buffer.update_run(images_received=7)
    buffer.end_run("saved_incomplete", "7/12 images received", n_shots=4)
    buffer.begin_run(102, "expt_b")

    runs = buffer.runs()
    assert [r["run_id"] for r in runs] == [101, 102]
    first, second = runs
    assert first["outcome"] == "saved_incomplete" and first["detail"] == "7/12 images received"
    assert first["n_shots_expected"] == 4 and first["images_received"] == 7 and first["n_shots"] == 4
    assert first["t_end"] >= first["t_start"]
    assert second["outcome"] == "in_progress" and second["t_end"] is None
    assert buffer.find_run(run_id=101)["seq"] == 1
    assert buffer.find_run()["run_id"] == 102
    assert buffer.find_run(run_id=555) is None and buffer.find_run(seq=9) is None
    buffer.clear()
    assert buffer.runs() == [] and buffer.find_run() is None
    buffer.end_run("saved")                                  # nothing current: a no-op


# ----------------------------------------------------------------------
# GET_LOG
# ----------------------------------------------------------------------

def test_server_answers_get_log_with_a_runs_records_and_outcome(server, buffer):
    srv, saver, states = server
    run_id = srv._handle_init_run(_init_msg())["run_id"]
    live = srv._handle_get_log({})
    assert live["ok"] and live["run"]["run_id"] == run_id
    assert live["run"]["outcome"] == "in_progress" and live["run"]["run_in_progress"] is True
    srv._handle_shot_complete({"shot_idx": 0, "N_shots_total": 1})
    assert srv._handle_end_run({})["ok"]

    reply = srv._handle_get_log({"run_id": run_id})
    assert reply["ok"] and reply["run"]["outcome"] == "saved" and reply["run"]["n_shots"] == 1
    msgs = _msgs(reply["records"])
    assert any(m.startswith(f"INIT_RUN: run_id={run_id}") for m in msgs)
    assert f"END_RUN: run_id={run_id} saved." in msgs
    assert any(m.startswith("shot 1/1") for m in msgs)

    only_warnings = srv._handle_get_log({"run_id": run_id, "min_level": logging.WARNING})
    assert only_warnings["ok"] and only_warnings["records"] == []

    missing = srv._handle_get_log({"run_id": 999})
    assert not missing["ok"] and "999" in missing["error"]

    runs = srv._handle_get_log({"runs": True})["runs"]
    assert [r["run_id"] for r in runs] == [run_id] and runs[0]["outcome"] == "saved"

    poll = srv._handle_poll({})
    assert poll["last_outcome"]["outcome"] == "saved" and poll["run_state"] == "saved"
    assert poll["expt_name"] == "smear_test" and poll["images_expected"] == 0


def test_server_get_log_tells_unsaved_runs_apart_by_seq(server, buffer):
    srv, saver, states = server
    srv._handle_init_run(_init_msg(save_data=False))
    srv._handle_end_run({})
    srv._handle_init_run(_init_msg(save_data=False))
    runs = srv._handle_get_log({"runs": True})["runs"]
    assert [r["run_id"] for r in runs] == [0, 0] and [r["seq"] for r in runs] == [1, 2]
    assert runs[0]["outcome"] == "nothing_written" and runs[1]["outcome"] == "in_progress"
    first = srv._handle_get_log({"seq": 1})
    assert first["ok"] and "END_RUN: save_data=False, nothing written." in _msgs(first["records"])
    assert "END_RUN: save_data=False, nothing written." not in _msgs(srv._handle_get_log({"seq": 2})["records"])


def test_server_get_log_before_any_run(server, buffer):
    srv, saver, states = server
    assert not srv._handle_get_log({})["ok"]
    assert srv._handle_get_log({"runs": True}) == {"ok": True, "runs": []}
    everything = srv._handle_get_log({"run_id": "all"})
    assert everything["ok"] and everything["run"] is None


# ----------------------------------------------------------------------
# A camera run whose frames do not all arrive
# ----------------------------------------------------------------------

def _camera_init(srv, n_img=3, n_shots=1):
    return srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a",
                                          params={"N_img": n_img}, N_shots_with_repeats=n_shots))


def test_server_saves_a_short_camera_run_as_incomplete(server, buffer):
    srv, saver, states = server
    run_id = _camera_init(srv)["run_id"]
    assert srv._images_expected == 3
    srv.on_image_received()
    srv.on_image_received()
    srv._handle_shot_complete({"shot_idx": 0, "N_shots_total": 1})
    srv.on_data_handler_done()                               # the writer closed the file
    reply = srv._handle_end_run({})

    assert reply["ok"]
    assert reply["incomplete"] == {"reason": "2/3 images received", "images_expected": 3, "images_received": 2}
    assert saver.incomplete == reply["incomplete"]           # the saver was told, so the file says so
    assert states[-1] == ("saved", "INCOMPLETE: 2/3 images received")
    assert srv._run_in_progress is False and srv._current_filepath == ""

    run = srv._handle_get_log({"run_id": run_id})
    assert run["run"]["outcome"] == "saved_incomplete" and run["run"]["images_received"] == 2
    errors = [r for r in run["records"] if r["level"] >= logging.ERROR]
    assert len(errors) == 1 and "saved INCOMPLETE" in errors[0]["msg"]
    assert srv._handle_poll({})["last_outcome"]["outcome"] == "saved_incomplete"


def test_server_saves_a_full_camera_run_as_complete(server, buffer):
    srv, saver, states = server
    _camera_init(srv)
    for _ in range(3):
        srv.on_image_received()
    srv.on_data_handler_done()
    reply = srv._handle_end_run({})
    assert reply == {"ok": True} and saver.incomplete is None
    assert states[-1] == ("saved", "")
    assert srv._handle_get_log({"runs": True})["runs"][-1]["outcome"] == "saved"


def test_server_records_a_grab_failure_in_the_incomplete_reason(server, buffer):
    srv, saver, states = server
    _camera_init(srv)
    srv.on_image_received()
    srv.on_grab_failed("camera timed out: No Andor image within 60 s (got 1/3)")
    srv.on_data_handler_done()
    reply = srv._handle_end_run({})
    assert reply["incomplete"]["reason"] == "1/3 images received; camera timed out: No Andor image within 60 s (got 1/3)"
    assert srv._handle_get_log({})["run"]["grab_failure"].startswith("camera timed out")


def test_server_counts_frames_per_run(server, buffer):
    srv, saver, states = server
    _camera_init(srv)
    srv.on_image_received()
    srv.on_data_handler_done()
    srv._handle_end_run({})
    _camera_init(srv)                                         # the next run starts from zero
    assert srv._images_received_now() == 0 and srv._grab_failure == ""
    for _ in range(3):
        srv.on_image_received()
    srv.on_data_handler_done()
    assert srv._handle_end_run({}) == {"ok": True}


def test_server_warns_once_the_camera_is_a_whole_shot_behind(server, buffer):
    srv, saver, states = server
    _camera_init(srv, n_img=6, n_shots=2)                     # 3 frames a shot
    srv._handle_shot_complete({"shot_idx": 0, "N_shots_total": 2})
    warnings = lambda: [r["msg"] for r in srv._handle_get_log({})["records"] if r["level"] == logging.WARNING]
    assert warnings() == []                                   # nothing is due yet from earlier shots
    srv._handle_shot_complete({"shot_idx": 1, "N_shots_total": 2})
    assert len(warnings()) == 1 and "3 frame(s) behind after shot 2/2" in warnings()[0]
    srv._handle_shot_complete({"shot_idx": 1, "N_shots_total": 2})
    assert len(warnings()) == 1                               # the same shortfall is not repeated


def test_server_does_not_warn_when_frames_keep_up(server, buffer):
    srv, saver, states = server
    _camera_init(srv, n_img=6, n_shots=2)
    for _ in range(3):
        srv.on_image_received()
    srv._handle_shot_complete({"shot_idx": 0, "N_shots_total": 2})
    srv._handle_shot_complete({"shot_idx": 1, "N_shots_total": 2})
    assert not [r for r in srv._handle_get_log({})["records"] if r["level"] >= logging.WARNING]


def test_run_file_save_reports_what_is_missing(tmp_path, calls, monkeypatch):
    from waxx.util.live_od.data import run_file
    monkeypatch.setattr(run_file, "DATA_SAVER_TIMEOUT", 0.2)
    saver = FakeSaver(tmp_path)
    rf = run_file.RunFile(saver)
    rf.begin(save_data=True, has_writer=True)
    run_id, _ = rf.reserve({})
    incomplete = rf.save({}, run_id, [], images_expected=3, images_received=1)   # the writer never finishes
    assert incomplete["reason"] == "image writer did not finish within 0 s; 1/3 images received"
    assert incomplete["images_expected"] == 3 and incomplete["images_received"] == 1
    assert saver.incomplete == incomplete

    rf.begin(save_data=True, has_writer=False)
    run_id, _ = rf.reserve({})
    assert rf.save({}, run_id, [], images_expected=3, images_received=3) is None
    assert saver.incomplete is None


# ----------------------------------------------------------------------
# DataHandler: the grab ends before the last frame
# ----------------------------------------------------------------------

def _spin(app, condition, timeout=10.0):
    t0 = time.time()
    while not condition() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_data_handler_stops_when_the_grab_ends_short(app, tmp_path, monkeypatch):
    from queue import Queue
    from waxx.util.live_od import config as live_od_config
    from waxx.util.live_od.camera_mother import DataHandler
    monkeypatch.setattr(live_od_config, "_active", live_od_config.LiveODConfig())
    _, path = _reserve(tmp_path)
    queue = Queue()
    handler = DataHandler(queue, path, save_data=True, imaging_type=0, camera_key="cam_a",
                          params_payload={"N_img": 3}, n_img=3, n_shots=1, n_pwa_per_shot=3)
    done = threading.Event()
    handler.done_writing_signal.connect(done.set)
    handler.get_img_number(3, 1, 3)
    for i in range(2):                                        # two of the three frames
        queue.put((np.full((4, 5), i + 1, np.uint16), None, i))
    handler.start()
    time.sleep(0.3)
    assert not done.is_set()                                  # still waiting for the third
    handler.grab_finished()                                   # the camera gave up
    assert _spin(app, done.is_set)
    handler.wait()
    handler.writer._worker.wait()
    assert handler.images_received == 2
    with h5py.File(path, "r") as f:                           # the file is closed and readable
        assert [int(f["data/images"][i, 0, 0]) for i in range(3)] == [1, 2, 0]
        assert f["data/image_timestamps"][2] == 0.0


def test_data_handler_grab_finished_after_every_frame_is_harmless(app, monkeypatch):
    from queue import Queue
    from waxx.util.live_od import config as live_od_config
    from waxx.util.live_od.camera_mother import DataHandler
    monkeypatch.setattr(live_od_config, "_active", live_od_config.LiveODConfig())
    queue = Queue()
    handler = DataHandler(queue, "", save_data=False, imaging_type=0, n_img=2, n_shots=1, n_pwa_per_shot=2)
    done = threading.Event()
    handler.done_writing_signal.connect(done.set)
    handler.get_img_number(2, 1, 2)
    for i in range(2):
        queue.put((np.zeros((2, 2), np.uint16), None, i))
    handler.start()
    assert _spin(app, done.is_set)
    handler.grab_finished()
    handler.wait()
    assert handler.images_received == 2
