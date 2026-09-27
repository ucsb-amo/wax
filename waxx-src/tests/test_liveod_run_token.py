"""Run identity (B1), the camera overrides record, WAIT_CAM_READY failing fast,
the camera-action rule (B7) and the END_RUN frame-alignment check, through the
server's handlers and the client. The server is never started (no socket, no
beacon); the client's transport is either scripted or the handlers themselves;
every data file lives in pytest's tmp_path.
"""
import json
import logging
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
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
    return srv, saver


def _init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 1, "expt_class": "token_test"}
    msg.update(kw)
    return msg


def dispatch(srv):
    """The server's message loop, minus the socket."""
    handlers = {"INIT_RUN": srv._handle_init_run, "WAIT_CAM_READY": srv._handle_wait_cam_ready,
                "SHOT_COMPLETE": srv._handle_shot_complete, "END_RUN": srv._handle_end_run,
                "ABORT_RUN": srv._handle_abort_run, "POLL": srv._handle_poll}

    def transport(payload, rcvtimeo_ms=None):
        return handlers[payload["tag"]](dict(payload))
    return transport


def client_on(transport):
    from waxx.util.live_od.live_od_client import LiveODClient
    client = LiveODClient.__new__(LiveODClient)      # no discovery, no socket
    client.last_reset_requested = False
    client._send_recv = transport
    return client


# ----------------------------------------------------------------------
# the token on the server
# ----------------------------------------------------------------------

def test_every_init_run_issues_a_new_token(server):
    srv, _ = server
    a = srv._handle_init_run(_init_msg())["run_token"]
    b = srv._handle_init_run(_init_msg())["run_token"]
    assert a != b and len(a) == 32 and int(a, 16) >= 0


def test_a_superseded_runs_end_run_is_ignored_and_the_new_file_survives(server, records):
    srv, saver = server
    old = srv._handle_init_run(_init_msg())
    new = srv._handle_init_run(_init_msg())                  # takes over, as before
    reply = srv._handle_end_run({"run_token": old["run_token"]})
    assert reply["ok"] is False and reply["stale_run"] is True
    assert reply["error"] == f"END_RUN for superseded run (token {old['run_token']}); ignored"
    assert saver.saved == [] and srv._run_in_progress is True
    assert os.path.exists(new["filepath"]) and os.path.exists(old["filepath"])
    assert any("superseded" in r.getMessage() for r in records)
    # the current run still ends normally
    assert srv._handle_end_run({"run_token": new["run_token"]})["ok"]
    assert [p for p, _ in saver.saved] == [new["filepath"]]


def test_a_superseded_runs_abort_does_not_discard_the_new_runs_file(server):
    srv, saver = server
    old = srv._handle_init_run(_init_msg())
    new = srv._handle_init_run(_init_msg())
    reply = srv._handle_abort_run({"run_token": old["run_token"]})
    assert reply["stale_run"] and os.path.exists(new["filepath"]) and srv._run_in_progress
    assert srv._run_state != "aborted"
    # the current run's own abort still discards its file (as before)
    assert srv._handle_abort_run({"run_token": new["run_token"]})["ok"]
    assert not os.path.exists(new["filepath"]) and srv._run_in_progress is False


def test_a_superseded_runs_shots_and_camera_wait_change_nothing(server):
    srv, _ = server
    old = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    reply = srv._handle_shot_complete({"run_token": old["run_token"], "shot_idx": 0, "N_shots_total": 1})
    assert reply["stale_run"] and srv._shot_timestamps == [] and srv._shot_mono == []
    reply = srv._handle_wait_cam_ready({"run_token": old["run_token"], "timeout": 0.01})
    assert reply["stale_run"] and "timed_out" not in reply and srv._t_ready_mono is None


def test_messages_without_a_token_are_taken_as_before(server):
    srv, saver = server
    path = srv._handle_init_run(_init_msg())["filepath"]
    assert srv._handle_shot_complete({"shot_idx": 0, "N_shots_total": 1})["ok"]
    assert srv._handle_end_run({})["ok"] and [p for p, _ in saver.saved] == [path]


# ----------------------------------------------------------------------
# the token in the client, end to end through the handlers
# ----------------------------------------------------------------------

def test_client_sends_the_token_on_every_run_message():
    sent = []

    def transport(payload, rcvtimeo_ms=None):
        sent.append(dict(payload))
        tag = payload["tag"]
        if tag == "INIT_RUN":
            return {"ok": True, "run_id": 5, "filepath": "", "run_token": "abc123"}
        if tag == "WAIT_CAM_READY":
            return {"ok": True, "ready": True, "reset_requested": False}
        if tag == "SHOT_COMPLETE":
            return {"ok": True, "reset_requested": False, "adjust_values": {}}
        return {"ok": True}
    c = client_on(transport)
    c.init_run({})
    c.wait_cam_ready(timeout=1.0)
    c.shot_complete(0, 1, {})
    c.end_run({})
    c.abort_run()
    assert [p["tag"] for p in sent] == ["INIT_RUN", "WAIT_CAM_READY", "SHOT_COMPLETE", "END_RUN", "ABORT_RUN"]
    assert "run_token" not in sent[0] and all(p["run_token"] == "abc123" for p in sent[1:])


def test_client_against_an_older_server_sends_no_token():
    sent = []

    def transport(payload, rcvtimeo_ms=None):
        sent.append(dict(payload))
        return {"ok": True, "run_id": 5, "filepath": "", "reset_requested": False}
    c = client_on(transport)
    c.init_run({})
    c.shot_complete(0, 1, {})
    assert all("run_token" not in p for p in sent)


def test_a_superseded_experiment_is_stopped_and_cannot_touch_the_new_run(server, capsys):
    srv, saver = server
    transport = dispatch(srv)
    old, new = client_on(transport), client_on(transport)
    old_path = old.init_run(_init_msg())["filepath"]
    new_path = new.init_run(_init_msg())["filepath"]          # a second launch takes liveOD over
    assert old.shot_complete(0, 1, {}) is True                 # told to stop, like a reset
    assert old.last_reset_requested is True
    assert "newer run" in capsys.readouterr().out
    old.abort_run()                                            # best effort, ignored by the server
    with pytest.raises(RuntimeError, match="superseded"):
        old.end_run({})
    assert os.path.exists(new_path) and os.path.exists(old_path) and saver.saved == []
    assert new.shot_complete(0, 1, {}) is False
    assert new.end_run({}) and [p for p, _ in saver.saved] == [new_path]


def test_a_superseded_camera_wait_fails_at_once(server):
    srv, _ = server
    transport = dispatch(srv)
    old, new = client_on(transport), client_on(transport)
    old.init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    new.init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    srv.on_cam_ready()                                         # the new run's camera is ready
    with pytest.raises(ValueError, match="superseded"):
        old.wait_cam_ready(timeout=5.0)
    assert new.wait_cam_ready(timeout=1.0) is True


# ----------------------------------------------------------------------
# WAIT_CAM_READY when the camera failed before it was armed
# ----------------------------------------------------------------------

def test_a_grab_that_failed_before_ready_fails_the_wait_at_once(server):
    srv, _ = server
    client = client_on(dispatch(srv))
    client.init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    srv.on_grab_failed("camera timed out: acquisition would not start")
    reply = srv._handle_wait_cam_ready({"timeout": 0.01})
    assert reply["ok"] is False and "timed_out" not in reply
    assert "failed before it was ready" in reply["error"]
    with pytest.raises(ValueError, match="acquisition would not start"):
        client.wait_cam_ready(timeout=30.0)


def test_ready_is_recorded_once_for_the_alignment_check(server):
    srv, _ = server
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    srv.on_cam_ready()
    assert srv._handle_wait_cam_ready({"timeout": 0.01})["ready"]
    t_ready = srv._t_ready_mono
    assert t_ready is not None
    srv._handle_wait_cam_ready({"timeout": 0.01})
    assert srv._t_ready_mono == t_ready


# ----------------------------------------------------------------------
# the camera overrides record
# ----------------------------------------------------------------------

RECORD = {"schema": 1, "camera_key": "cam_a", "persist_since": None,
          "fields": {"exposure_time": {"requested": 1.5e-05, "applied": 1.9e-05,
                                       "origin": "clamped"}}}


def test_clamps_are_logged_polled_and_written_at_end_run(server, records, capsys):
    srv, saver = server
    client = client_on(dispatch(srv))
    client.init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    srv.on_camera_overrides("cam_a", {"exposure_time": (np.float64(1.5e-05), np.float64(1.9e-05))})
    warnings = [r.getMessage() for r in records if "CAMERA OVERRIDE" in r.getMessage()]
    assert len(warnings) == 1 and "exposure_time requested 1.5e-05 -> applied 1.9e-05" in warnings[0]
    assert srv._handle_poll({})["camera_overrides"] == RECORD
    srv.on_cam_ready()
    assert client.wait_cam_ready(timeout=1.0) is True
    banner = capsys.readouterr().out
    assert "!! CAMERA SETTINGS DIFFER FROM camera_params (cam_a)" in banner
    assert "exposure_time: requested 1.5e-05 -> applied 1.9e-05 (clamped)" in banner
    assert banner.isascii()

    srv.on_data_handler_done()                                 # no image writer in this test
    payload = {"extra_file_texts": {"device_state_at_start": "{}"}}
    client.end_run(payload)
    sent = saver.payloads[-1]["extra_file_texts"]
    assert json.loads(sent["camera_overrides"]) == RECORD
    assert sent["device_state_at_start"] == "{}"
    assert "camera_overrides" not in payload["extra_file_texts"]     # the caller's dict untouched


def test_no_overrides_no_record(server):
    srv, saver = server
    srv._handle_init_run(_init_msg())
    srv.on_camera_overrides("cam_a", {})
    assert srv._handle_poll({})["camera_overrides"] == {}
    msg = {"extra_file_texts": {"a": "b"}}
    assert srv._with_camera_overrides(msg) is msg
    srv._handle_end_run(msg)
    assert "camera_overrides" not in saver.payloads[-1]["extra_file_texts"]


def test_a_new_run_starts_without_the_last_runs_overrides(server):
    srv, _ = server
    srv._handle_init_run(_init_msg())
    srv.on_camera_overrides("cam_a", {"gain": (300, 30)})
    assert srv.camera_overrides_record()
    srv._handle_init_run(_init_msg())
    assert srv.camera_overrides_record() == {}


# ----------------------------------------------------------------------
# who may open / close a camera (B7)
# ----------------------------------------------------------------------

def test_camera_action_allowed(server):
    srv, _ = server
    assert srv.camera_action_allowed("cam_a", "open") == (True, "")
    assert srv.camera_action_allowed("cam_a", "explode")[0] is False
    assert srv.camera_action_allowed("", "close") == (False, "Missing camera_key")
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    run_id = srv._current_run_id
    assert srv.camera_action_allowed("cam_b", "close") == (True, "")
    ok, reason = srv.camera_action_allowed("cam_a", "close")
    assert ok is False and reason == (f"Camera control rejected: run {run_id} in progress "
                                      f"(uses cam_a); only closing a camera the run does not use "
                                      f"is allowed")
    assert srv.camera_action_allowed("cam_b", "open")[0] is False
    assert srv.camera_action_allowed("cam_b", "toggle")[0] is False
    # the remote path gives the same answer, in the same words
    reply = srv._handle_camera_control({"camera_key": "cam_a", "action": "close"})
    assert reply["error"] == reason and reply["run_camera_key"] == "cam_a"


# ----------------------------------------------------------------------
# frame alignment at END_RUN
# ----------------------------------------------------------------------

def _camera_run(srv, n_shots=2, per_shot=3):
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a",
                                   params={"N_img": n_shots * per_shot},
                                   N_shots_with_repeats=n_shots))
    srv.on_data_handler_done()                                 # no image writer in these tests


def _feed(srv, frame_t, shot_t, t_ready=0.0):
    """Arrival and completion times as the handlers record them (monotonic s)."""
    for _ in frame_t:
        srv.on_image_received(None)
    srv._frame_times[:] = list(frame_t)
    for i, _ in enumerate(shot_t):
        srv._handle_shot_complete({"shot_idx": i, "N_shots_total": len(shot_t)})
    srv._shot_mono[:] = list(shot_t)
    srv._t_ready_mono = t_ready


GOOD = [9.90, 9.95, 10.002, 19.90, 19.95, 20.002]


def test_aligned_frames_leave_the_run_complete(server):
    srv, saver = server
    _camera_run(srv)
    _feed(srv, GOOD, [10.0, 20.0])
    reply = srv._handle_end_run({})
    assert reply["ok"] and "incomplete" not in reply and saver.incomplete is None


def test_a_frame_before_its_shot_could_start_marks_the_run_incomplete(server, records):
    srv, saver = server
    _camera_run(srv)
    _feed(srv, [-0.5] + GOOD[:-1], [10.0, 20.0])            # stray before ready, count still 6
    reply = srv._handle_end_run({})
    assert reply["incomplete"]["reason"].startswith("FRAME ALIGNMENT SUSPECT: frame 0")
    assert "FRAME ALIGNMENT SUSPECT" in saver.incomplete["reason"]
    assert saver.incomplete["images_received"] == 6 == saver.incomplete["images_expected"]


def test_an_ambiguous_late_frame_is_only_a_warning(server, records):
    srv, saver = server
    _camera_run(srv)
    late = GOOD[:2] + [13.0] + [t + 3.0 for t in GOOD[3:]]
    _feed(srv, late, [10.0, 20.0])
    reply = srv._handle_end_run({})
    assert "incomplete" not in reply
    assert any("Frame alignment (not conclusive)" in r.getMessage() for r in records)
