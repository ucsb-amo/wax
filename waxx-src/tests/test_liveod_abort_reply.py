"""A run whose experiment stops talking: the exit notice (RUN_EXITED) and an abort
nobody answers ("no_reply"). Run 83110 (2026-09-26) is the case: its process died
after its one shot without END_RUN, an Abort pressed afterwards had no one to
acknowledge it, and the pill sat on "Aborting" until the next run started.

The server is never started (no socket, no beacon); the client's transport is
either scripted or the handlers themselves; every data file lives in pytest's
tmp_path.
"""
import logging
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from live_od_data_fakes import FakeSaver, patch_payload_stash
import liveod_qt_helpers as qt


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def records():
    """WARNING+ records of the liveOD logger (it may not propagate to the root)."""
    got = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            got.append(record)
    handler = ListHandler(logging.WARNING)
    log = logging.getLogger("waxx.live_od")
    log.addHandler(handler)
    yield got
    log.removeHandler(handler)


@pytest.fixture
def server(app, tmp_path, monkeypatch):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    saver = FakeSaver(tmp_path)
    srv = LiveODServer(server_talk=None, data_saver=saver)   # never started
    states = []
    srv.run_state_signal.connect(lambda state, detail: states.append((state, detail)))
    done = []
    srv.run_done_signal.connect(lambda: done.append(True))
    return srv, saver, states, done


def _init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 2, "expt_class": "exit_test"}
    msg.update(kw)
    return msg


def dispatch(srv):
    """The server's message loop, minus the socket."""
    handlers = {"INIT_RUN": srv._handle_init_run, "WAIT_CAM_READY": srv._handle_wait_cam_ready,
                "SHOT_COMPLETE": srv._handle_shot_complete, "END_RUN": srv._handle_end_run,
                "ABORT_RUN": srv._handle_abort_run, "RUN_EXITED": srv._handle_run_exited,
                "POLL": srv._handle_poll}

    def transport(payload, rcvtimeo_ms=None):
        return handlers[payload["tag"]](dict(payload))
    return transport


def client_on(transport, monkeypatch=None):
    """A client with no discovery and no socket. Its exit notice goes through
    ``transport`` too (``_send_once`` is the only other way out)."""
    from waxx.util.live_od.live_od_client import LiveODClient
    client = LiveODClient.__new__(LiveODClient)
    client.last_reset_requested = False
    client._send_recv = transport
    client._send_once = lambda payload, timeout_ms: transport(payload)
    return client


@pytest.fixture
def no_atexit(monkeypatch):
    """Record exit handlers instead of registering them with the interpreter."""
    from waxx.util.live_od import live_od_client
    registered = []
    monkeypatch.setattr(live_od_client.atexit, "register", registered.append)
    return registered


# ----------------------------------------------------------------------
# RUN_EXITED on the server
# ----------------------------------------------------------------------

def test_an_exit_during_an_abort_acknowledges_it(server):
    srv, _, states, done = server
    reply = srv._handle_init_run(_init_msg())
    srv._handle_reset({})
    assert srv._handle_run_exited({"run_token": reply["run_token"],
                                   "reason": "uncaught KeyboardInterrupt"})["ok"]
    # exactly what ABORT_RUN does: the run's file goes, as for any abort
    assert [s for s, _ in states] == ["running", "aborting", "aborted"]
    assert not os.path.exists(reply["filepath"]) and srv._run_in_progress is False
    assert srv._last_outcome["outcome"] == "discarded" and done == [True]


def test_an_exit_without_an_abort_keeps_the_file(server, records):
    srv, saver, states, done = server
    reply = srv._handle_init_run(_init_msg())
    srv._handle_shot_complete({"shot_idx": 0, "N_shots_total": 2})
    assert srv._handle_run_exited({"run_token": reply["run_token"],
                                   "reason": "uncaught RuntimeError: boom"})["ok"]
    state, detail = states[-1]
    assert state == "exited" and "RuntimeError: boom" in detail
    assert os.path.exists(reply["filepath"]) and saver.saved == []     # untouched
    assert srv._run_in_progress is False and done == [True]
    assert srv._last_outcome["outcome"] == "exited"
    assert "RuntimeError: boom" in srv._last_outcome["detail"]
    poll = srv._handle_poll({})
    assert poll["run_state"] == "exited" and poll["run_in_progress"] is False
    assert any("without END_RUN" in r.getMessage() for r in records)
    # the next run does not touch it either (no abort was pending)
    srv._handle_init_run(_init_msg())
    assert os.path.exists(reply["filepath"])


def test_an_exit_while_frames_are_still_due_leaves_the_camera_run_alone(server):
    """The camera thread still owns the file: the run stays in progress (as after
    a crash today) and the GUI is not told the run is done, which would drop its
    handle on that thread. Only the state and the outcome say what happened."""
    srv, _, states, done = server
    reply = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    srv.on_image_received(None)
    assert srv._handle_run_exited({"run_token": reply["run_token"], "reason": ""})["ok"]
    state, detail = states[-1]
    assert state == "exited" and "1/3" in detail
    assert srv._run_in_progress is True and done == []
    assert os.path.exists(reply["filepath"])


def test_a_superseded_runs_exit_changes_nothing(server):
    srv, _, states, _ = server
    old = srv._handle_init_run(_init_msg())
    new = srv._handle_init_run(_init_msg())
    srv._handle_reset({})                          # the new run is being aborted
    reply = srv._handle_run_exited({"run_token": old["run_token"], "reason": ""})
    assert reply["stale_run"] and srv._reset_requested is True
    assert os.path.exists(new["filepath"]) and srv._run_in_progress is True
    assert states[-1][0] == "aborting"


def test_an_exit_after_the_run_ended_is_ignored(server):
    srv, saver, states, _ = server
    reply = srv._handle_init_run(_init_msg())
    srv._handle_end_run({"run_token": reply["run_token"]})
    assert srv._handle_run_exited({"run_token": reply["run_token"], "reason": ""})["ok"]
    assert states[-1][0] == "saved" and srv._last_outcome["outcome"] == "saved"
    assert [p for p, _ in saver.saved] == [reply["filepath"]]


# ----------------------------------------------------------------------
# an abort nobody answers
# ----------------------------------------------------------------------

def test_an_abort_nobody_answers_becomes_no_reply(server, records):
    from waxx.util.live_od import live_od_server as mod
    srv, _, states, _ = server
    reply = srv._handle_init_run(_init_msg())
    srv._handle_reset({})
    t_abort = srv._abort_requested_at
    limit = srv._abort_reply_limit()
    assert limit == mod.ABORT_REPLY_MIN_S                     # no shot yet
    srv._check_abort_reply(now=t_abort + limit - 1.0)
    assert states[-1][0] == "aborting"
    srv._check_abort_reply(now=t_abort + limit + 1.0)
    state, detail = states[-1]
    assert state == "no_reply" and "no answer from the experiment" in detail
    assert "discards its file" in detail                      # what the next run start does
    # nothing is decided on a guess: a live experiment is still told to stop,
    # and the file is untouched until something answers
    assert srv._reset_requested is True and srv._run_in_progress is True
    assert os.path.exists(reply["filepath"])
    assert srv._handle_poll({})["run_state"] == "no_reply"
    warned = [r for r in records if "no answer from the experiment" in r.getMessage()]
    srv._check_abort_reply(now=t_abort + limit + 5.0)
    assert len(warned) == 1 and len([s for s, _ in states if s == "no_reply"]) == 1
    # pressing Abort again does not flip it back to "aborting"
    srv.note_reset_requested()
    assert states[-1][0] == "no_reply"
    # a late acknowledgement still closes it out as before
    srv._handle_abort_run({"run_token": reply["run_token"]})
    assert states[-1][0] == "aborted" and not os.path.exists(reply["filepath"])
    assert srv._abort_requested_at is None


def test_the_reply_limit_follows_the_shot_period(server):
    from waxx.util.live_od import live_od_server as mod
    srv, _, _, _ = server
    srv._handle_init_run(_init_msg())
    srv._init_run_time = 1000.0
    srv._shot_timestamps = [1035.0]                           # first shot, from INIT_RUN
    assert srv._abort_reply_limit() == pytest.approx(mod.ABORT_REPLY_SHOT_FACTOR * 35.0)
    srv._shot_durations = [8.0, 12.0, 9.0]                    # the longest recent period
    assert srv._abort_reply_limit() == pytest.approx(max(mod.ABORT_REPLY_MIN_S,
                                                         mod.ABORT_REPLY_SHOT_FACTOR * 12.0))


def test_no_reply_is_only_for_an_abort_in_progress(server):
    srv, _, states, _ = server
    srv._check_abort_reply(now=1e12)                          # no run at all
    reply = srv._handle_init_run(_init_msg())
    srv._check_abort_reply(now=1e12)                          # a run, no abort
    assert [s for s, _ in states] == ["running"]
    srv._handle_reset({})
    srv._handle_end_run({"run_token": reply["run_token"]})    # answered by END_RUN
    srv._check_abort_reply(now=1e12)
    assert [s for s, _ in states] == ["running", "aborting", "aborted"]


def test_a_new_run_starts_clean_after_no_reply(server):
    srv, _, states, _ = server
    old = srv._handle_init_run(_init_msg())
    srv._handle_reset({})
    srv._check_abort_reply(now=srv._abort_requested_at + 1e6)
    new = srv._handle_init_run(_init_msg())                   # closes the old one out, as before
    assert not os.path.exists(old["filepath"]) and os.path.exists(new["filepath"])
    assert states[-1][0] == "running" and srv._abort_requested_at is None
    assert srv._reset_requested is False


# ----------------------------------------------------------------------
# the client's exit notice
# ----------------------------------------------------------------------

def _scripted():
    sent = []

    def transport(payload, rcvtimeo_ms=None):
        sent.append(dict(payload))
        if payload["tag"] == "INIT_RUN":
            return {"ok": True, "run_id": 5, "filepath": "", "run_token": "abc123"}
        return {"ok": True, "reset_requested": False}
    return sent, transport


def test_exit_notice_for_a_run_left_open(no_atexit, monkeypatch, capsys):
    sent, transport = _scripted()
    c = client_on(transport)
    c.init_run({})
    c.init_run({})                                            # a second run: one handler
    assert no_atexit == [c.notify_exit]
    monkeypatch.setattr(sys, "last_exc", RuntimeError("boom"), raising=False)
    c.notify_exit()
    assert sent[-1]["tag"] == "RUN_EXITED" and sent[-1]["run_token"] == "abc123"
    assert sent[-1]["reason"] == "uncaught RuntimeError: boom"
    out = capsys.readouterr().out
    assert "without END_RUN" in out and out.isascii()
    c.notify_exit()                                           # once
    assert [p["tag"] for p in sent].count("RUN_EXITED") == 1


@pytest.mark.parametrize("close", ["end_run", "abort_run"])
def test_no_exit_notice_once_the_run_is_closed(no_atexit, close):
    sent, transport = _scripted()
    c = client_on(transport)
    c.init_run({})
    getattr(c, close)({}) if close == "end_run" else c.abort_run()
    c.notify_exit()
    assert "RUN_EXITED" not in [p["tag"] for p in sent]


def test_an_abort_that_did_not_reach_the_server_leaves_the_notice(no_atexit):
    sent, transport = _scripted()

    def flaky(payload, rcvtimeo_ms=None):
        if payload["tag"] == "ABORT_RUN":
            raise ConnectionError("no reply")
        return transport(payload)
    c = client_on(flaky)
    c.init_run({})
    c.abort_run()                                             # best effort, swallowed
    c.notify_exit()
    assert sent[-1]["tag"] == "RUN_EXITED"


def test_the_exit_notice_never_raises(no_atexit, capsys):
    _, transport = _scripted()
    c = client_on(transport)
    c.init_run({})

    def dead(payload, timeout_ms):
        raise OSError("liveOD is gone")
    c._send_once = dead
    c.notify_exit()                                           # no exception
    out = capsys.readouterr().out
    assert "could not tell liveOD" in out and out.isascii()


def test_no_exit_notice_without_a_run(no_atexit):
    sent, transport = _scripted()
    c = client_on(transport)
    c.notify_exit()
    assert sent == [] and no_atexit == []


def test_an_experiment_that_dies_mid_abort_ends_the_abort(server, no_atexit):
    """End to end: Abort pressed, the process exits instead of answering."""
    srv, _, states, _ = server
    c = client_on(dispatch(srv))
    path = c.init_run(_init_msg())["filepath"]
    srv._handle_reset({})
    c.notify_exit()
    assert states[-1][0] == "aborted" and not os.path.exists(path)
