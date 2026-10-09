"""A run being saved is never closed under its save (review B1, 2026-10-09).

During an asynchronous END_RUN save the run stays in progress, and an Abort
pressed then shows "aborting": a RUN_EXITED / ABORT_RUN / second Reset would
close it as aborted and delete the file being written. POLL now says
``save_in_progress``; the server refuses those messages while a save runs; the
gate (run_gate) calls such a run live and never sends RUN_EXITED for it.

The server is never started (no socket, no beacon); its handlers are called
directly, the save is held open with an Event, and its file lives in tmp_path.
"""
import os
import socket
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from live_od_data_fakes import FakeSaver, patch_payload_stash
import liveod_qt_helpers as qt

from waxx.util.device_state import run_gate


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def held(app, tmp_path, monkeypatch):
    """A server whose run 101 is in its asynchronous save, held until ``release``."""
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    saver = FakeSaver(tmp_path)
    srv = LiveODServer(server_talk=None, data_saver=saver)   # never started
    gate = threading.Event()
    real = saver.save_data_from_payload

    def slow(*a, **k):
        gate.wait(10)
        return real(*a, **k)
    monkeypatch.setattr(saver, "save_data_from_payload", slow)
    init = srv._handle_init_run({
        "tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
        "params": {"N_img": 1}, "N_shots_with_repeats": 1, "expt_class": "save_guard",
        "client_pid": 4242, "client_host": socket.gethostname(), "launcher": "run_lock"})
    assert srv._handle_end_run({"run_token": init["run_token"], "async_save": True})["saving"]
    yield srv, init, gate
    gate.set()
    srv._run_file.wait_save(10)


def _wait(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, "timed out"
        time.sleep(0.01)


def test_poll_says_a_save_is_running(held):
    srv, init, gate = held
    poll = srv._handle_poll({"tag": "POLL"})
    assert poll["run_in_progress"] and poll["save_in_progress"] is True
    assert poll["save_status"]["state"] == "saving"
    assert poll["save_status"]["run_id"] == init["run_id"]
    gate.set()
    _wait(lambda: not srv._run_in_progress)
    poll = srv._handle_poll({"tag": "POLL"})
    assert poll["save_in_progress"] is False and poll["save_status"]["state"] == "saved"


def test_reset_during_save_closes_nothing_and_deletes_nothing(held):
    srv, init, gate = held
    path, rid = init["filepath"], init["run_id"]
    srv._handle_reset({"tag": "RESET"})
    poll = srv._handle_poll({"tag": "POLL"})
    assert poll["reset_requested"] and poll["run_state"] == "aborting"
    # the server refuses to close the run under its save
    out = srv._handle_run_exited({"tag": "RUN_EXITED", "run_id": rid, "reason": "x"})
    assert out["ok"] is False and out["saving"] is True
    out = srv._handle_abort_run({"tag": "ABORT_RUN"})
    assert out["ok"] is False and out["saving"] is True
    srv._abort_requested_at = time.time() - 60          # a second Reset, long after
    assert srv.abort_again() is False
    assert srv._run_in_progress and os.path.exists(path)
    # the gate: live, not waived, even with the client's process gone
    st = run_gate.classify(poll, None, pid_alive=lambda pid: False)
    assert st.state == "live" and not st.waivable
    # the helper sends nothing
    sent = []

    class Client:
        def poll(self):
            return srv._handle_poll({"tag": "POLL"})

        def _send_recv(self, msg):
            sent.append(msg)
            return srv._handle_run_exited(msg)
    out = run_gate.tell_live_od_run_exited(Client(), rid, "x", pid_alive=lambda pid: False)
    assert not out["sent"] and sent == []
    # the save finishes and keeps the file
    gate.set()
    _wait(lambda: not srv._run_in_progress)
    assert os.path.exists(path)
    assert srv._last_outcome["outcome"] == "saved"


@pytest.fixture
def plain(app, tmp_path, monkeypatch):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    saver = FakeSaver(tmp_path)
    return LiveODServer(server_talk=None, data_saver=saver), saver


def _init(srv, **kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 1}, "N_shots_with_repeats": 1, "expt_class": "save_guard"}
    msg.update(kw)
    return srv._handle_init_run(msg)


def test_an_abort_between_runs_keeps_a_failed_saves_file(plain):
    """Review S3: an Abort pressed with no run in progress used to be finalized at
    the next INIT_RUN: the previous run recorded as discarded, and its file
    deleted when its save had failed (the path is kept for a retry)."""
    srv, saver = plain
    saver.fail_save = True
    first = _init(srv)
    assert srv._handle_end_run({"run_token": first["run_token"]})["ok"] is False
    assert not srv._run_in_progress and os.path.exists(first["filepath"])
    assert srv._last_outcome["outcome"] == "save_failed"
    srv._handle_reset({"tag": "RESET"})                  # Abort, no run in progress
    assert srv._reset_requested
    saver.fail_save = False
    second = _init(srv)
    assert second["ok"] and not srv._reset_requested
    assert os.path.exists(first["filepath"])             # kept
    # the failed run's record is not overwritten with "discarded / reset"
    assert srv._last_outcome["run_id"] == first["run_id"]
    assert srv._last_outcome["outcome"] == "save_failed"


def test_an_abort_of_a_run_in_progress_is_still_finalized_at_init_run(plain):
    srv, saver = plain
    first = _init(srv)
    srv._handle_reset({"tag": "RESET"})                  # its experiment never answers
    second = _init(srv)
    assert second["ok"]
    assert not os.path.exists(first["filepath"])         # discarded, as for any abort
    assert srv._last_outcome["run_id"] == first["run_id"]
    assert srv._last_outcome["outcome"] == "discarded"


def test_dead_client_during_save_is_live(held):
    srv, init, gate = held
    poll = srv._handle_poll({"tag": "POLL"})
    st = run_gate.classify(poll, None, pid_alive=lambda pid: False)
    assert st.state == "live" and not st.waivable and "being saved" in st.reason
