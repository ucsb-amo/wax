"""FIX-B review findings on the liveOD server and client, through the server's
handlers and the client (the flag-off path):

* m1  a camera thread's report carries its run's token; a replaced run's is dropped;
* m3  WAIT_CAM_READY: only an old server's reply is judged by its error text;
* m7  the camera_overrides record never blocks the END_RUN save;
* m9  SHOT_COMPLETE says the camera's grab has ended; the client warns once;
* m10 a refused INIT_RUN leaves the run in progress as it was (the camera host's
      refusal here; the data-file refusal waits for the guarded edit, see
      test_fixb_init_run.py);
* m15 a token this liveOD never issued is "unknown (restarted?)", not "superseded";
* M2  END_RUN's alignment check looks at the Andor's last slot, not a Basler's.

The server is never started (no socket, no beacon); every data file lives in
pytest's tmp_path.
"""
import json
import logging
import os
import types

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


def messages(records, level=logging.WARNING):
    return [r.getMessage() for r in records if r.levelno >= level]


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
           "params": {"N_img": 3}, "N_shots_with_repeats": 1, "expt_class": "fixb_test"}
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


def scripted(replies):
    """A transport that answers WAIT_CAM_READY with ``replies`` in turn (the last
    one over and over); returns (transport, list of tags sent)."""
    sent = []

    def transport(payload, rcvtimeo_ms=None):
        sent.append(payload["tag"])
        n = sum(1 for t in sent if t == "WAIT_CAM_READY")
        return dict(replies[min(n - 1, len(replies) - 1)])
    return transport, sent


# ----------------------------------------------------------------------
# m1: reports from a replaced run's camera threads
# ----------------------------------------------------------------------

def test_reports_from_a_replaced_runs_camera_thread_are_dropped(server, records):
    srv, _ = server
    old = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))["run_token"]
    new = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))["run_token"]
    srv.on_cam_ready(run_token=old)
    srv.on_grab_failed("camera timed out: late", run_token=old)
    srv.on_camera_overrides("cam_a", {"gain": (300, 30)}, run_token=old)
    for _ in range(5):
        srv.on_image_received(None, run_token=old)
    assert not srv._cam_ready_event.is_set()
    assert srv._grab_failure == "" and srv.camera_overrides_record() == {}
    assert srv._images_received_now() == 0 and srv._frame_times == []
    # one WARNING per kind of report, not one per frame
    stale = [m for m in messages(records) if "camera thread of an earlier run" in m]
    assert len(stale) == 4 and all(old[:8] in m and new[:8] in m for m in stale)

    # the current run's threads, and a caller that does not stamp, get through
    srv.on_image_received(None, run_token=new)
    srv.on_image_received(None)
    assert srv._images_received_now() == 2
    srv.on_grab_failed("camera timed out: now", run_token=new)
    assert srv._grab_failure == "camera timed out: now"
    srv.on_cam_ready(run_token=new)
    assert srv._cam_ready_event.is_set()


def test_every_camera_run_leaves_a_token_for_its_spawn_in_order(server):
    srv, _ = server
    a = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))["run_token"]
    srv._handle_init_run(_init_msg())                                  # no camera: no spawn
    b = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))["run_token"]
    assert srv.take_spawn_token() == a and srv.take_spawn_token() == b
    assert srv.take_spawn_token() == b                                 # none waiting: the current run's


# ----------------------------------------------------------------------
# m3: WAIT_CAM_READY failures and slice timeouts
# ----------------------------------------------------------------------

def test_a_new_servers_fast_failure_raises_at_once_whatever_its_text():
    transport, sent = scripted([{"ok": False, "ready": False, "reset_requested": False,
                                 "error": "camera failed before it was ready: "
                                          "TimeoutError: acquisition timeout on the camera"}])
    client = client_on(transport)
    with pytest.raises(ValueError, match="acquisition timeout on the camera"):
        client.wait_cam_ready(timeout=30.0)
    assert sent == ["WAIT_CAM_READY"]


def test_timed_out_false_is_a_failure_and_true_is_ask_again():
    transport, sent = scripted([{"ok": False, "ready": False, "timed_out": True,
                                 "reset_requested": False, "error": "Camera ready timeout"},
                                {"ok": False, "ready": False, "timed_out": False,
                                 "error": "timeout of some other kind"}])
    client = client_on(transport)
    with pytest.raises(ValueError, match="other kind"):
        client.wait_cam_ready(timeout=30.0)
    assert sent == ["WAIT_CAM_READY", "WAIT_CAM_READY"]


def test_a_stale_run_is_a_failure_even_when_its_text_says_timeout():
    transport, sent = scripted([{"ok": False, "stale_run": True,
                                 "error": "WAIT_CAM_READY for superseded run (timeout); ignored"}])
    with pytest.raises(ValueError, match="superseded"):
        client_on(transport).wait_cam_ready(timeout=30.0)
    assert sent == ["WAIT_CAM_READY"]


def test_an_old_servers_timeout_text_still_means_not_ready_yet():
    transport, sent = scripted([{"ok": False, "error": "Camera ready timeout"},
                                {"ok": False, "error": "Camera ready timeout"},
                                {"ok": True, "ready": True}])
    assert client_on(transport).wait_cam_ready(timeout=30.0) is True
    assert sent == ["WAIT_CAM_READY"] * 3


def test_an_old_servers_instant_timeout_replies_are_paced(monkeypatch):
    """An old server failing fast with "timeout" in its text looks like a slice
    timeout; the client no longer asks again at once, over and over."""
    from waxx.util.live_od import live_od_client
    monkeypatch.setattr(live_od_client, "CAM_READY_SLICE_S", 0.05)
    transport, sent = scripted([{"ok": False, "error": "TimeoutException: camera timeout"}])
    with pytest.raises(ValueError, match="timed out after"):
        client_on(transport).wait_cam_ready(timeout=0.3)
    assert 3 <= len(sent) <= 12


def test_the_server_says_reset_requested_on_its_fast_failure(server):
    """What the client keys on: a current server's fast failure carries
    reset_requested and no timed_out."""
    srv, _ = server
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    srv.on_grab_failed("TimeoutError: acquisition timeout")
    reply = srv._handle_wait_cam_ready({"timeout": 0.01})
    assert "reset_requested" in reply and "timed_out" not in reply
    from waxx.util.live_od.live_od_client import LiveODClient
    assert LiveODClient._wait_not_ready_yet(reply) is False


# ----------------------------------------------------------------------
# m7: the overrides record never blocks the save
# ----------------------------------------------------------------------

class Opaque:
    def __repr__(self):
        return "<Opaque driver value>"


def test_values_json_cannot_hold_are_written_as_their_repr(server):
    srv, saver = server
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    srv.on_data_handler_done()                                 # no image writer in this test
    roi = np.array([0, 512])
    srv.on_camera_overrides("cam_a", {"sensor_roi": (roi, (0, 256))})
    srv._merge_camera_overrides("cam_a", {"gain": {"requested": 300, "applied": Opaque(),
                                                   "origin": "persist"}})
    poll = srv._handle_poll({})["camera_overrides"]
    assert json.loads(json.JSONEncoder().encode(poll)) == poll         # plain values only
    reply = srv._handle_end_run({"extra_file_texts": {"device_state_at_start": "{}"}})
    assert reply["ok"] and len(saver.saved) == 1
    written = json.loads(saver.payloads[-1]["extra_file_texts"]["camera_overrides"])
    assert written["fields"]["sensor_roi"] == {"requested": repr(roi), "applied": [0, 256],
                                               "origin": "clamped"}
    assert written["fields"]["gain"] == {"requested": 300, "applied": "<Opaque driver value>",
                                         "origin": "persist"}


def test_a_record_that_cannot_be_encoded_is_still_written_field_by_field(server):
    srv, saver = server
    srv._handle_init_run(_init_msg())
    loop = {"requested": 1, "origin": "clamped"}
    loop["applied"] = loop                                       # a self-reference
    srv._merge_camera_overrides("cam_a", {"loop": loop})
    assert srv._handle_end_run({})["ok"] and len(saver.saved) == 1
    written = json.loads(saver.payloads[-1]["extra_file_texts"]["camera_overrides"])
    assert written["fields"]["loop"]["requested"] == "1"
    assert written["fields"]["loop"]["origin"] == "clamped"


def test_a_record_that_cannot_be_added_does_not_stop_the_save(server, records, monkeypatch):
    srv, saver = server
    srv._handle_init_run(_init_msg())
    srv.on_camera_overrides("cam_a", {"gain": (300, 30)})

    def broken():
        raise RuntimeError("no record today")
    monkeypatch.setattr(srv, "camera_overrides_record", broken)
    reply = srv._handle_end_run({"extra_file_texts": {"a": "b"}})
    assert reply["ok"] and len(saver.saved) == 1
    assert saver.payloads[-1]["extra_file_texts"] == {"a": "b"}
    errors = messages(records, logging.ERROR)
    assert any("saved WITHOUT it" in m and "no record today" in m and "'gain'" in m
               for m in errors)


# ----------------------------------------------------------------------
# m9: the experiment is told the grab has ended
# ----------------------------------------------------------------------

def test_shot_complete_says_when_the_grab_has_ended(server):
    srv, _ = server
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a",
                                   N_shots_with_repeats=3))
    assert "grab_failure" not in srv._handle_shot_complete({"shot_idx": 0, "N_shots_total": 3})
    srv.on_grab_failed("camera timed out: FrameLostError: frame 4 of 9 lost")
    reply = srv._handle_shot_complete({"shot_idx": 1, "N_shots_total": 3})
    assert reply["ok"] and reply["reset_requested"] is False
    assert reply["grab_failure"] == "camera timed out: FrameLostError: frame 4 of 9 lost"


def test_the_client_warns_once_per_run_and_goes_on(server, capsys):
    srv, _ = server
    client = client_on(dispatch(srv))
    client.init_run(_init_msg(capture_images=True, camera_key="cam_a", N_shots_with_repeats=3))
    assert client.shot_complete(0, 3, {}) is False
    assert "!!" not in capsys.readouterr().out
    srv.on_grab_failed("camera timed out: frame 4 lost (µs clock)")
    assert client.shot_complete(1, 3, {}) is False               # not a reset: the run goes on
    out = capsys.readouterr().out
    assert "!! liveOD: THE CAMERA STOPPED RECORDING THIS RUN" in out
    assert "frame 4 lost" in out and "NOT recorded" in out and out.isascii()
    assert client.shot_complete(2, 3, {}) is False
    assert "!!" not in capsys.readouterr().out                   # once
    # the next run is warned again
    client.init_run(_init_msg(capture_images=True, camera_key="cam_a", N_shots_with_repeats=3))
    srv.on_grab_failed("camera timed out: again")
    client.shot_complete(0, 3, {})
    assert "again" in capsys.readouterr().out


# ----------------------------------------------------------------------
# m10: a camera-host refusal leaves the run in progress as it was
# ----------------------------------------------------------------------

class FakeHost:
    """The camera host's run surface, with begin_run refusing on demand."""

    def __init__(self):
        self.refuse = None
        self.begun, self.ended = [], []

    def begin_run(self, token, camera_key, capture_images, camera_params=None, images_shape=None):
        if self.refuse:
            raise RuntimeError(self.refuse)
        self.begun.append(token)
        return types.SimpleNamespace(camera_key=camera_key, refused={}, persist_on=False,
                                     overrides={}, persist_since=None)

    def note_run_id(self, token, run_id):
        pass

    def end_run(self, token, reason="END_RUN"):
        self.ended.append((token, reason))
        return types.SimpleNamespace(overrides={}, problems=[], camera_key="", persist_since=None)

    def poll_cameras(self):
        return {}


def test_a_host_refusal_leaves_the_run_in_progress_as_it_was(server, records):
    srv, saver = server
    host = FakeHost()
    srv.set_camera_host(host)
    first = srv._handle_init_run(_init_msg())
    token = first["run_token"]
    state = srv._run_state
    host.refuse = "HostRefused: cam_b is held elsewhere"
    reply = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_b"))
    assert reply["ok"] is False and "cam_b is held elsewhere" in reply["error"]
    assert "no run id used" in reply["error"]
    # nothing of the first run changed
    assert srv._run_token == token and srv._run_in_progress is True
    assert srv._run_state == state and srv._current_camera_key == ""
    assert srv._host_run_token == token and host.ended == []
    assert list(srv._spawn_tokens) == []
    assert any("changed nothing" in m for m in messages(records))
    # and it ends as it would have
    assert srv._handle_shot_complete({"run_token": token, "shot_idx": 0, "N_shots_total": 1})["ok"]
    assert srv._handle_end_run({"run_token": token})["ok"]
    assert [p for p, _ in saver.saved] == [first["filepath"]]
    assert host.ended == [(token, "END_RUN")]


def test_a_host_refusal_with_no_run_in_progress_shows_the_error(server):
    srv, _ = server
    host = FakeHost()
    host.refuse = "HostRefused: frame transfer is refused"
    srv.set_camera_host(host)
    reply = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_b"))
    assert reply["ok"] is False and srv._run_state == "error"
    assert srv._run_token == "" and srv._run_in_progress is False


def test_an_accepted_init_run_ends_the_previous_runs_camera_hold(server):
    srv, _ = server
    host = FakeHost()
    srv.set_camera_host(host)
    a = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))["run_token"]
    b = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_b"))["run_token"]
    assert host.begun == [a, b] and srv._host_run_token == b
    assert host.ended == [(a, "superseded by a new INIT_RUN")]


# ----------------------------------------------------------------------
# m15: superseded or unknown
# ----------------------------------------------------------------------

def test_a_token_this_liveod_never_issued_is_unknown_not_superseded(server, records):
    srv, saver = server
    first = srv._handle_init_run(_init_msg())
    reply = srv._handle_end_run({"run_token": "f" * 32})
    assert reply["ok"] is False and reply["stale_run"] is True and reply["unknown_run"] is True
    assert "unknown to this liveOD (restarted?)" in reply["error"]
    assert "superseded" not in reply["error"]
    assert saver.saved == [] and srv._run_in_progress is True
    assert any("unknown to this liveOD" in m and "restarted" in m for m in messages(records))
    # a superseded run is still called that
    srv._handle_init_run(_init_msg())
    reply = srv._handle_end_run({"run_token": first["run_token"]})
    assert "superseded" in reply["error"] and "unknown_run" not in reply


def test_after_a_restart_every_old_token_is_unknown(server, capsys):
    srv, _ = server                                   # a fresh liveOD: no INIT_RUN yet
    client = client_on(dispatch(srv))
    client._run_token = "a" * 32                      # from before the restart
    assert client.shot_complete(3, 10, {}) is True    # stopped, like a reset
    out = capsys.readouterr().out
    assert "does not know this run (was it restarted?)" in out
    assert "newer run" not in out


# ----------------------------------------------------------------------
# M2: the slow-readout camera's last slot at END_RUN
# ----------------------------------------------------------------------

# each shot's last frame arrives before its SHOT_COMPLETE (10.0, 20.0)
LAST_SLOT_EARLY = [9.90, 9.93, 9.96, 19.90, 19.93, 19.96]


@pytest.mark.parametrize("camera_type, noted", [("andor", True), ("basler", False)])
def test_the_last_slot_note_is_for_a_slow_readout_camera_only(server, records, monkeypatch,
                                                              camera_type, noted):
    from waxx.util.live_od import config as live_od_config
    monkeypatch.setattr(live_od_config, "_active", live_od_config.LiveODConfig(
        resolve_camera_params=lambda key: types.SimpleNamespace(key=key, camera_type=camera_type)))
    srv, saver = server
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a",
                                   params={"N_img": 6}, N_shots_with_repeats=2))
    srv.on_data_handler_done()
    for _ in LAST_SLOT_EARLY:
        srv.on_image_received(None)
    srv._frame_times[:] = LAST_SLOT_EARLY
    for i in range(2):
        srv._handle_shot_complete({"shot_idx": i, "N_shots_total": 2})
    srv._shot_mono[:] = [10.0, 20.0]
    srv._t_ready_mono = 0.0
    reply = srv._handle_end_run({})
    assert reply["ok"] and "incomplete" not in reply              # a note, never an issue
    shift = [m for m in messages(records) if "possible one-slot shift" in m]
    assert bool(shift) is noted


def test_a_camera_table_that_cannot_answer_leaves_the_rest_of_the_check(server, records,
                                                                        monkeypatch):
    from waxx.util.live_od import config as live_od_config

    def broken(key):
        raise KeyError(key)
    monkeypatch.setattr(live_od_config, "_active",
                        live_od_config.LiveODConfig(resolve_camera_params=broken))
    srv, saver = server
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a",
                                   params={"N_img": 6}, N_shots_with_repeats=2))
    srv.on_data_handler_done()
    frames = [-0.5, 9.9, 9.95, 19.9, 19.95, 20.002]           # frame 0 before "ready"
    for _ in frames:
        srv.on_image_received(None)
    srv._frame_times[:] = frames
    for i in range(2):
        srv._handle_shot_complete({"shot_idx": i, "N_shots_total": 2})
    srv._shot_mono[:] = [10.0, 20.0]
    srv._t_ready_mono = 0.0
    reply = srv._handle_end_run({})
    assert reply["incomplete"]["reason"].startswith("FRAME ALIGNMENT SUSPECT: frame 0")
    assert any("last-slot check is off" in m for m in messages(records))
