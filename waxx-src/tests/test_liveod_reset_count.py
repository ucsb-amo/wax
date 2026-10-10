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
    assert poll["reset_counts"] == {"person": 0, "queue": 0, "agent": 0, "liveod": 0}
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


# -- the window's reset(): only its own button counts a person's Reset (review NEW-1) --------

def _window(srv):
    from types import SimpleNamespace
    from waxx.util.live_od.gui.main_window import LiveODWindow

    class Win:
        reset = LiveODWindow.reset
        _reset_from_server = LiveODWindow._reset_from_server

        def __init__(self):
            self.live_od_server = srv
            self.messages = []
            self._run_active = True
            self.the_baby = None
            self.data_handler = None
            self.camera_nanny = SimpleNamespace(interrupted=False)

        def msg(self, text, level=None):
            self.messages.append(text)

        def _update_run_buttons(self):
            pass
    return Win()


def test_a_remote_queue_reset_is_never_recounted_as_a_persons(srv):
    run = _init(srv)["run_id"]
    srv._handle_reset({"tag": "RESET", "source": "queue", "run_id": run})
    # the experiment's ABORT_RUN clears the flag before the GUI thread's slot runs
    srv._reset_requested = False
    _window(srv)._reset_from_server()                 # the queued reset_signal
    poll = _poll(srv)
    assert poll["reset_counts"] == {"person": 0, "queue": 1, "agent": 0, "liveod": 0}
    assert poll["reset_requested"] is False           # not re-armed for the next run


def test_a_finalized_runs_signal_sets_nothing(srv):
    _init(srv)
    _window(srv)._reset_from_server()                 # e.g. a run the server finalized
    poll = _poll(srv)
    assert poll["reset_count"] == 0 and poll["reset_requested"] is False


def test_the_windows_own_button_is_a_persons_reset(srv):
    _init(srv)
    _window(srv).reset()
    poll = _poll(srv)
    assert poll["reset_counts"]["person"] == 1 and poll["reset_requested"] is True
    assert poll["last_reset"]["source"] == "person"


def test_an_unusable_data_file_aborts_as_liveod_not_a_person(srv):
    _init(srv)
    _window(srv).reset(source="liveod")
    poll = _poll(srv)
    assert poll["reset_counts"]["liveod"] == 1 and poll["reset_counts"]["person"] == 0
    assert poll["reset_requested"] is True


def test_poll_counts_the_arrays_pushed_during_the_run(srv):
    from types import SimpleNamespace
    import numpy as np
    assert _poll(srv)["aux_items_received"] == 0
    _init(srv)
    srv._note_aux_items([SimpleNamespace(key="img_mot", index=(0,), offset=None,
                                         array=np.zeros(3))])
    assert _poll(srv)["aux_items_received"] == 1
    _init(srv)                                         # a new run starts at 0
    assert _poll(srv)["aux_items_received"] == 0


# -- a Reset on a run whose process is gone keeps its file (user ruling 2026-10-09) --------------

import socket as _socket

HERE = _socket.gethostname()


def _alive(monkeypatch, alive):
    from waxx.util.device_state import run_gate
    monkeypatch.setattr(run_gate, "pid_alive", lambda pid: alive)


def test_a_reset_after_the_process_died_keeps_the_file(srv, monkeypatch):
    run = _init(srv, client_pid=4242, client_host=HERE)
    srv._shot_timestamps = [1.0, 2.0, 3.0]                # it took shots
    _alive(monkeypatch, False)
    reply = srv._handle_reset({"tag": "RESET"})
    assert reply["ok"] and reply["kept"] and "file is KEPT" in reply["message"]
    srv._check_exited_run()                               # the server loop's next pass
    poll = _poll(srv)
    assert not poll["run_in_progress"] and not poll["reset_requested"]
    last = poll["last_outcome"]
    assert last["outcome"] == "exited" and last["run_id"] == run["run_id"]
    assert last["detail"] == srv.DEAD_CLIENT_RESET_WHY
    assert os.path.exists(run["filepath"])                # not deleted
    assert poll["reset_counts"]["person"] == 1            # the press is counted
    from waxx.util.live_od.gui.remote_viewer_window import RemoteViewerWindow
    assert "file is KEPT" in RemoteViewerWindow.reply_notice_text("Reset", reply)


def test_an_abort_pressed_before_death_then_a_reset_after_keeps_the_file(srv, monkeypatch):
    run = _init(srv, client_pid=4242, client_host=HERE)
    srv._shot_timestamps = [1.0]
    _alive(monkeypatch, True)
    srv._handle_reset({"tag": "RESET"})                   # alive: an Abort, as ever
    assert _poll(srv)["reset_requested"] is True
    _alive(monkeypatch, False)                            # it dies before answering
    reply = srv._handle_reset({"tag": "RESET"})           # a person presses again
    assert reply.get("kept") is True
    assert _poll(srv)["reset_requested"] is False         # the pending Abort is spent
    srv._check_exited_run()
    _init(srv)                                            # the next run starts
    assert os.path.exists(run["filepath"])


def test_the_windows_button_on_a_dead_run_keeps_the_file(srv, monkeypatch):
    run = _init(srv, client_pid=4242, client_host=HERE)
    _alive(monkeypatch, False)
    win = _window(srv)
    win.reset()
    assert any("file is KEPT" in m for m in win.messages)
    srv._check_exited_run()
    assert _poll(srv)["last_outcome"]["detail"] == srv.DEAD_CLIENT_RESET_WHY
    assert os.path.exists(run["filepath"])


@pytest.mark.parametrize("client, alive", [
    ({"client_pid": 4242, "client_host": HERE}, True),           # a live client
    ({"client_pid": 4242, "client_host": "other-pc"}, False),    # another host
    ({"client_host": HERE}, False),                              # no pid recorded
])
def test_otherwise_a_reset_is_the_abort_it_always_was(srv, monkeypatch, client, alive):
    _init(srv, **client)
    _alive(monkeypatch, alive)
    reply = srv._handle_reset({"tag": "RESET"})
    assert reply["ok"] and "kept" not in reply
    assert _poll(srv)["reset_requested"] is True          # the experiment aborts; discard


def test_an_empty_dead_run_with_an_abort_is_discarded_at_the_next_init_run(srv, monkeypatch):
    # run 85528's case: Abort pending, process killed, no shot, no frame
    run = _init(srv, client_pid=4242, client_host=HERE)
    _alive(monkeypatch, True)
    srv._handle_reset({"tag": "RESET"})
    _alive(monkeypatch, False)
    _init(srv)                                            # the gate waived it: next run
    assert not os.path.exists(run["filepath"])            # discarded, as today


# -- a reused pid (review F3) ---------------------------------------------------------

def _started_now(monkeypatch, value):
    from waxx.util.device_state import detached
    monkeypatch.setattr(detached, "process_started", lambda pid=None: value)


@pytest.mark.parametrize("now, dead", [
    (1000.0, False),             # the same process: alive, a normal Abort
    (1000.004, False),           # within detached.SAME_PROCESS_S (rounding)
    (2000.0, True),              # the pid was reused: the client is gone
    (None, False),               # cannot be read: never taken for gone
])
def test_a_live_pid_is_the_client_only_when_its_creation_time_matches(srv, monkeypatch,
                                                                       now, dead):
    _init(srv, client_pid=4242, client_host=HERE, client_started=1000.0)
    _alive(monkeypatch, True)
    _started_now(monkeypatch, now)
    assert srv.client_known_dead() is dead


def test_without_a_recorded_creation_time_a_live_pid_is_the_client(srv, monkeypatch):
    _init(srv, client_pid=4242, client_host=HERE)                 # an older client
    _alive(monkeypatch, True)
    _started_now(monkeypatch, 2000.0)
    assert srv.client_known_dead() is False


def test_a_reset_on_a_reused_pid_keeps_the_file(srv, monkeypatch):
    run = _init(srv, client_pid=4242, client_host=HERE, client_started=1000.0)
    srv._shot_timestamps = [1.0, 2.0]
    _alive(monkeypatch, True)
    _started_now(monkeypatch, 2000.0)                             # another process now
    reply = srv._handle_reset({"tag": "RESET"})
    assert reply["ok"] and reply["kept"]
    srv._check_exited_run()
    assert _poll(srv)["last_outcome"]["detail"] == srv.DEAD_CLIENT_RESET_WHY
    assert os.path.exists(run["filepath"])


# -- RUN_EXITED on a process's behalf during an Abort (review F4) --------------------

@pytest.mark.parametrize("data", ["shots", "pushed"])
def test_a_notice_on_behalf_during_an_abort_on_a_run_with_data_is_refused(srv, monkeypatch, data):
    run = _init(srv, client_pid=4242, client_host=HERE)
    _alive(monkeypatch, True)                              # alive when the Abort is set
    if data == "shots":
        srv._shot_timestamps = [1.0]
    else:
        srv._aux_items_received = 2
    srv._handle_reset({"tag": "RESET", "source": "queue"})
    before = _poll(srv)
    reply = srv._handle_run_exited({"tag": "RUN_EXITED", "run_id": run["run_id"],
                                    "reason": "sent by a helper"})
    assert reply["ok"] is False and reply["refused"] and reply["abort_pending_with_data"]
    assert "a person decides" in reply["error"]
    after = _poll(srv)                                     # nothing changed
    assert after["run_in_progress"] and after["reset_requested"]
    assert after["last_outcome"] == before["last_outcome"]
    assert os.path.exists(run["filepath"])


def test_a_notice_on_behalf_during_an_abort_on_an_empty_run_is_the_abort(srv, monkeypatch):
    run = _init(srv, client_pid=4242, client_host=HERE)
    _alive(monkeypatch, True)
    srv._handle_reset({"tag": "RESET", "source": "queue"})
    reply = srv._handle_run_exited({"tag": "RUN_EXITED", "run_id": run["run_id"],
                                    "reason": "sent by a helper"})
    assert reply["ok"]
    assert not _poll(srv)["run_in_progress"]
    assert not os.path.exists(run["filepath"])             # nothing in it: discarded


def test_the_processs_own_notice_during_an_abort_is_still_its_answer(srv, monkeypatch):
    run = _init(srv, client_pid=4242, client_host=HERE)
    _alive(monkeypatch, True)
    srv._shot_timestamps = [1.0]
    srv._handle_reset({"tag": "RESET"})                    # a person's Abort
    reply = srv._handle_run_exited({"tag": "RUN_EXITED", "run_token": run["run_token"],
                                    "reason": "uncaught KeyboardInterrupt"})
    assert reply["ok"] and not _poll(srv)["run_in_progress"]
    assert _poll(srv)["last_outcome"]["outcome"] == "discarded"


def test_the_helper_reports_the_servers_refusal(srv, monkeypatch):
    from waxx.util.device_state import run_gate
    run = _init(srv, client_pid=4242, client_host=HERE)
    _alive(monkeypatch, True)
    srv._handle_reset({"tag": "RESET", "source": "queue"})
    stale = _poll(srv)                                     # taken before the shot came
    srv._shot_timestamps = [1.0]
    out = run_gate.tell_live_od_run_exited(
        None, run["run_id"], "sent by a helper", poll=stale, pid_alive=lambda pid: False,
        send=lambda rid, why: srv._handle_run_exited({"tag": "RUN_EXITED", "run_id": rid,
                                                      "reason": why}))
    assert out["sent"] and out["ok"] is False and "liveOD refused" in out["why"]
    assert os.path.exists(run["filepath"]) and _poll(srv)["run_in_progress"]


# -- INIT_RUN's finalize of an unanswered Abort (review F5) ---------------------------

def _abort_then(srv, monkeypatch, shots, alive_at_init, **client):
    run = _init(srv, **client)
    srv._shot_timestamps = [1.0] * shots
    _alive(monkeypatch, True)                              # alive when the Abort is set
    srv._handle_reset({"tag": "RESET", "source": "queue"})
    assert _poll(srv)["reset_requested"] is True
    _alive(monkeypatch, alive_at_init)
    nxt = _init(srv)                                       # the next run starts
    return run, nxt


def test_a_dead_client_run_with_data_is_kept_at_the_next_init_run(srv, monkeypatch):
    run, nxt = _abort_then(srv, monkeypatch, 3, False, client_pid=4242, client_host=HERE)
    assert os.path.exists(run["filepath"])                 # kept, not discarded
    poll = _poll(srv)
    last = poll["last_outcome"]
    assert last["run_id"] == run["run_id"] and last["outcome"] == "exited"
    assert last["detail"] == srv.DEAD_CLIENT_KEPT_AT_INIT_WHY
    assert last["n_shots"] == 3                            # the kept run's own counts
    assert poll["run_in_progress"] and poll["run_id"] == nxt["run_id"]
    assert not poll["reset_requested"]                     # the new run is not aborted


@pytest.mark.parametrize("shots, alive, client", [
    (0, False, {"client_pid": 4242, "client_host": HERE}),        # dead, no data
    (3, True, {"client_pid": 4242, "client_host": HERE}),         # live client
    (3, False, {"client_pid": 4242, "client_host": "other-pc"}),  # unknown: another host
    (3, False, {"client_host": HERE}),                            # unknown: no pid
])
def test_otherwise_the_next_init_run_discards_it_as_before(srv, monkeypatch, shots, alive,
                                                           client):
    run, _ = _abort_then(srv, monkeypatch, shots, alive, **client)
    assert not os.path.exists(run["filepath"])
    last = _poll(srv)["last_outcome"]
    assert last["run_id"] == run["run_id"] and last["outcome"] == "discarded"


# -- the file a camera thread already deleted at the Abort (review G2) -----------------

def test_a_file_already_gone_is_never_reported_kept(srv, monkeypatch):
    # a person's Abort on a camera run while the client was alive: the grab
    # loop's dishonorable_death deletes the file at once; then the client dies.
    # The deletion is simulated: os.path.exists says the file is gone (no file
    # is removed by this test).
    run = _init(srv, client_pid=4242, client_host=HERE)
    srv._shot_timestamps = [1.0, 2.0]
    _alive(monkeypatch, True)
    srv._handle_reset({"tag": "RESET", "source": "person"})
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists",
                        lambda p: False if p == run["filepath"] else real_exists(p))
    _alive(monkeypatch, False)
    _init(srv)                                             # the next run starts
    last = _poll(srv)["last_outcome"]
    assert last["run_id"] == run["run_id"] and last["outcome"] == "discarded"
    assert last["detail"].startswith(srv.ALREADY_DELETED_AT_ABORT_WHY)
    assert "the Abort was at" in last["detail"]
    assert "kept" not in last["detail"].lower()
    monkeypatch.undo()
    assert os.path.exists(run["filepath"])                 # this path never deleted it


def test_a_run_without_a_file_is_finalized_as_before(srv, monkeypatch):
    run = _init(srv, client_pid=4242, client_host=HERE, save_data=False)
    srv._shot_timestamps = [1.0]
    _alive(monkeypatch, True)
    srv._handle_reset({"tag": "RESET"})
    _alive(monkeypatch, False)
    _init(srv)
    last = _poll(srv)["last_outcome"]
    assert last["run_id"] == run["run_id"] and last["outcome"] == "discarded"
