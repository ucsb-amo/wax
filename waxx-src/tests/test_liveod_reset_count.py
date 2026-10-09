"""liveOD counts every Abort by source, and RESET can name its run (review
B2, S9): POLL's reset_count / reset_counts / last_reset; RESET's optional
source ("person" when absent), run_id and run_token -- a RESET naming a run
that is not the one in progress is refused.  The server is never started (no
socket, no beacon); its handlers are called directly and its data file lives
in pytest's tmp_path."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from live_od_data_fakes import FakeSaver, patch_payload_stash
import liveod_qt_helpers as qt


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def srv(app, tmp_path, monkeypatch):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    server = LiveODServer(server_talk=None, data_saver=FakeSaver(tmp_path))  # never started
    yield server
    server._run_file.finish_writer()
    assert server._run_file.writer_done.wait(10), "the run's ImageWriter did not finish"


def _init(srv, **kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 2, "expt_class": "count_test"}
    msg.update(kw)
    return srv._handle_init_run(msg)


def _poll(srv):
    return srv._handle_poll({"tag": "POLL"})


def test_poll_counts_every_abort_by_source(srv):
    poll = _poll(srv)
    assert poll["reset_count"] == 0 and poll["last_reset"] is None
    assert poll["reset_counts"] == {"person": 0, "queue": 0, "agent": 0}
    run = _init(srv)["run_id"]
    assert srv._handle_reset({"tag": "RESET", "source": "queue", "run_id": run})["ok"]
    poll = _poll(srv)
    assert poll["reset_count"] == 1 and poll["reset_counts"]["queue"] == 1
    last = poll["last_reset"]
    assert last["source"] == "queue" and last["run_id"] == run and last["count"] == 1
    # the window's own reset() for the same press is not counted again
    assert srv.request_reset("person") is False
    assert _poll(srv)["reset_count"] == 1


def test_a_reset_without_a_source_is_a_persons_and_the_count_outlives_the_level(srv):
    _init(srv)
    srv._handle_reset({"tag": "RESET"})
    srv._reset_requested = False                     # cleared (ABORT_RUN / INIT_RUN)
    poll = _poll(srv)
    assert not poll["reset_requested"]
    assert poll["reset_counts"]["person"] == 1 and poll["last_reset"]["source"] == "person"
    assert srv.request_reset("person") is True       # the window's own button, a new press
    assert _poll(srv)["reset_counts"]["person"] == 2


@pytest.mark.parametrize("msg, words", [
    ({"run_id": 999}, "names run 999"),
    ({"run_token": "not-this-runs"}, "token"),
    ({"source": "robot"}, "unknown RESET source"),
    ({"run_id": "x"}, "bad run_id"),
])
def test_a_reset_for_another_run_is_refused_and_changes_nothing(srv, msg, words):
    _init(srv)
    reply = srv._handle_reset(dict({"tag": "RESET"}, **msg))
    assert reply["ok"] is False and reply["refused"] and words in reply["error"]
    poll = _poll(srv)
    assert not poll["reset_requested"] and poll["reset_count"] == 0


def test_a_reset_naming_a_run_with_no_run_in_progress_is_refused(srv):
    reply = srv._handle_reset({"tag": "RESET", "run_id": 5, "source": "agent"})
    assert reply["refused"] and reply["run_id"] is None and "none" in reply["error"]


def test_the_right_run_id_and_token_are_accepted(srv):
    reply = _init(srv)
    out = srv._handle_reset({"tag": "RESET", "source": "agent", "run_id": reply["run_id"],
                             "run_token": reply["run_token"]})
    assert out["ok"] and out["reset_count"] == 1
    assert _poll(srv)["reset_counts"]["agent"] == 1
