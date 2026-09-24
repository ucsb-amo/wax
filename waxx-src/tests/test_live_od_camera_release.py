"""Releasing a camera from liveOD while nothing uses it: the server's
CAMERA_CONTROL guard, camera state in POLL, and the client helpers that wait
for the GUI to finish.  Nothing opens a socket or a camera; the client's
transport is a scripted stand-in.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from live_od_data_fakes import FakeSaver


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def server(app, tmp_path):
    from waxx.util.live_od.live_od_server import LiveODServer
    srv = LiveODServer(server_talk=None, data_saver=FakeSaver(tmp_path))   # never started
    emitted = []
    srv.camera_control_signal.connect(lambda key, action: emitted.append((key, action)))
    return srv, emitted


def _run_on(srv, camera_key, capture_images=True, run_id=77):
    srv._run_in_progress = True
    srv._current_capture_images = capture_images
    srv._current_camera_key = camera_key
    srv._current_run_id = run_id


# ----------------------------------------------------------------------
# Server guard
# ----------------------------------------------------------------------

def test_idle_server_allows_open_close_toggle(server):
    srv, emitted = server
    for action in ("open", "close", "toggle"):
        assert srv._handle_camera_control({"camera_key": "xy_basler", "action": action}) == {"ok": True}
    assert emitted == [("xy_basler", "open"), ("xy_basler", "close"), ("xy_basler", "toggle")]


def test_run_refuses_closing_its_own_camera(server):
    srv, emitted = server
    _run_on(srv, "z_basler")
    reply = srv._handle_camera_control({"camera_key": "z_basler", "action": "close"})
    assert reply["ok"] is False and reply["run_camera_key"] == "z_basler"
    assert "run 77" in reply["error"] and "z_basler" in reply["error"]
    assert emitted == []


def test_run_allows_closing_a_camera_it_does_not_use(server):
    srv, emitted = server
    _run_on(srv, "z_basler")
    assert srv._handle_camera_control({"camera_key": "xy_basler", "action": "close"}) == {"ok": True}
    assert emitted == [("xy_basler", "close")]


def test_run_refuses_any_open_or_toggle(server):
    srv, emitted = server
    _run_on(srv, "z_basler")
    for action in ("open", "toggle"):
        reply = srv._handle_camera_control({"camera_key": "xy_basler", "action": action})
        assert reply["ok"] is False and reply["run_in_progress"] is True
    assert emitted == []


def test_no_camera_run_uses_no_camera(server):
    srv, emitted = server
    _run_on(srv, "z_basler", capture_images=False)     # key left over, but no frames taken
    assert srv._handle_camera_control({"camera_key": "z_basler", "action": "close"}) == {"ok": True}
    reply = srv._handle_camera_control({"camera_key": "z_basler", "action": "open"})
    assert reply["ok"] is False and reply["run_camera_key"] == ""
    assert srv._handle_poll({})["run_camera_key"] == ""


def test_bad_requests(server):
    srv, _ = server
    assert srv._handle_camera_control({"camera_key": "", "action": "close"})["ok"] is False
    assert srv._handle_camera_control({"camera_key": "x", "action": "explode"})["ok"] is False


# ----------------------------------------------------------------------
# POLL reports the cameras
# ----------------------------------------------------------------------

def test_poll_reports_camera_states_from_the_provider(server):
    srv, _ = server
    assert srv._handle_poll({})["cameras"] == {}          # no GUI attached
    states = {"xy_basler": {"state": "open", "camera_type": "basler", "serial_no": "40316451"},
              "andor": {"state": "closed", "camera_type": "andor", "serial_no": ""}}
    srv.set_camera_state_provider(lambda: states)
    reply = srv._handle_poll({})
    assert reply["cameras"] == states and reply["run_camera_key"] == ""
    _run_on(srv, "xy_basler")
    assert srv._handle_poll({})["run_camera_key"] == "xy_basler"


def test_poll_survives_a_broken_provider(server):
    srv, _ = server

    def boom():
        raise RuntimeError("gui gone")
    srv.set_camera_state_provider(boom)
    assert srv._handle_poll({})["cameras"] == {}


# ----------------------------------------------------------------------
# Client helpers, against a scripted transport
# ----------------------------------------------------------------------

class ScriptedLiveOD:
    """Stands in for LiveODServer over the wire: holds camera states, applies a
    CAMERA_CONTROL after ``latency`` seconds (the GUI is asynchronous), and can
    be told to refuse like the server does during a run."""

    def __init__(self, states, latency=0.3, refuse=None):
        self.states = {k: dict(v) for k, v in states.items()}
        self.latency = latency
        self.refuse = refuse or {}
        self.sent = []
        self._pending = []

    def __call__(self, payload, rcvtimeo_ms=None):
        self.sent.append(payload)
        now = time.monotonic()
        for t, key, state in list(self._pending):
            if now >= t:
                self.states[key]["state"] = state
                self._pending.remove((t, key, state))
        tag = payload["tag"]
        if tag == "POLL":
            return {"ok": True, "run_in_progress": False, "run_id": 1, "cameras": self.states,
                    "run_camera_key": ""}
        if tag == "CAMERA_CONTROL":
            key, action = payload["camera_key"], payload["action"]
            if key in self.refuse:
                return {"ok": False, "error": self.refuse[key]}
            target = "closed" if action == "close" else "open"
            self._pending.append((now + self.latency, key, target))
            return {"ok": True}
        return {"ok": False, "error": f"Unknown tag: {tag}"}


def make_client(transport):
    from waxx.util.live_od.live_od_client import LiveODClient
    client = LiveODClient.__new__(LiveODClient)
    client._send_recv = transport
    return client


STATES = {"xy_basler": {"state": "open", "camera_type": "basler", "serial_no": "40316451"},
          "z_basler": {"state": "closed", "camera_type": "basler", "serial_no": "40416468"}}


def test_release_camera_waits_until_the_gui_has_closed_it():
    t = ScriptedLiveOD(STATES, latency=0.3)
    client = make_client(t)
    t0 = time.monotonic()
    info = client.release_camera("xy_basler", timeout=5.0)
    assert info["state"] == "closed" and info["serial_no"] == "40316451"
    assert 0.25 < time.monotonic() - t0 < 3.0
    assert [p for p in t.sent if p["tag"] == "CAMERA_CONTROL"] == \
        [{"tag": "CAMERA_CONTROL", "camera_key": "xy_basler", "action": "close"}]


def test_release_of_a_closed_camera_sends_nothing():
    t = ScriptedLiveOD(STATES)
    info = make_client(t).release_camera("z_basler")
    assert info["state"] == "closed"
    assert not [p for p in t.sent if p["tag"] == "CAMERA_CONTROL"]


def test_release_refused_by_the_server_is_an_error_with_its_reason():
    t = ScriptedLiveOD(STATES, refuse={"xy_basler": "Camera control rejected: run 77 in progress (uses xy_basler)"})
    with pytest.raises(RuntimeError, match="rejected.*run 77"):
        make_client(t).release_camera("xy_basler")


def test_release_unknown_camera():
    with pytest.raises(KeyError, match="no camera 'andor'"):
        make_client(ScriptedLiveOD(STATES)).release_camera("andor")


def test_release_times_out_when_the_gui_never_closes_it():
    t = ScriptedLiveOD(STATES, latency=60.0)
    with pytest.raises(TimeoutError, match="still 'open'"):
        make_client(t).release_camera("xy_basler", timeout=0.5)


def test_open_camera():
    t = ScriptedLiveOD(STATES, latency=0.2)
    info = make_client(t).open_camera("z_basler", timeout=5.0)
    assert info["state"] == "open"


def test_cameras_on_an_old_server():
    def old(payload, rcvtimeo_ms=None):
        return {"ok": True, "run_in_progress": False}      # no "cameras" key
    with pytest.raises(LookupError, match="does not report camera state"):
        make_client(old).cameras()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def test_cli_release_and_list(monkeypatch, capsys):
    from waxx.util.live_od import camera_cli
    t = ScriptedLiveOD(STATES, latency=0.1)
    monkeypatch.setattr(camera_cli, "_connect", lambda timeout: make_client(t))
    assert camera_cli.main(["release", "xy_basler"]) == 0
    assert "released" in capsys.readouterr().out
    assert camera_cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "xy_basler" in out and "closed" in out and "40316451" in out


def test_cli_release_refused_exit_code(monkeypatch, capsys):
    from waxx.util.live_od import camera_cli
    t = ScriptedLiveOD(STATES, refuse={"xy_basler": "Camera control rejected: run 77 in progress (uses xy_basler)"})
    monkeypatch.setattr(camera_cli, "_connect", lambda timeout: make_client(t))
    assert camera_cli.main(["release", "xy_basler"]) == camera_cli.EXIT_REFUSED
    assert "rejected" in capsys.readouterr().err


def test_cli_grab_refuses_without_release_when_liveod_holds_the_camera(monkeypatch, capsys):
    from waxx.util.live_od import camera_cli
    t = ScriptedLiveOD(STATES)
    monkeypatch.setattr(camera_cli, "_connect", lambda timeout: make_client(t))
    assert camera_cli.main(["grab", "xy_basler"]) == camera_cli.EXIT_REFUSED
    assert "--release" in capsys.readouterr().err


def test_cli_grab_releases_grabs_by_serial_and_reopens(monkeypatch, capsys):
    import beacon.basler.frame_grabber as fg
    from waxx.util.live_od import camera_cli
    t = ScriptedLiveOD(STATES, latency=0.1)
    monkeypatch.setattr(camera_cli, "_connect", lambda timeout: make_client(t))
    grabbed = []

    def fake_grab(camera, timeout=None, trigger_mode=None, fresh=True, collect_for=2.0):
        import numpy as np
        grabbed.append((camera, timeout, trigger_mode, fresh))
        return fg.Frame(image=np.zeros((2, 3), dtype="uint8"), timestamp=1.0, gain=6.0,
                        exposure_us=19.0, max_pixel_value=255, serial=camera, user_id="xy_basler",
                        model="acA", server_id="basler_server:cam-pc", trigger_mode="On", waited_s=0.5)
    monkeypatch.setattr(fg, "grab_frame", fake_grab)
    monkeypatch.setattr(camera_cli.time, "sleep", lambda s: None)

    assert camera_cli.main(["grab", "xy_basler", "--release", "--reopen",
                            "--frame-timeout", "4", "--trigger-mode", "On"]) == 0
    assert grabbed == [("40316451", 4.0, "On", True)]      # by serial, not by key
    out = capsys.readouterr().out
    assert "released from liveOD" in out and "back in liveOD" in out
    assert t.states["xy_basler"]["state"] == "open"
    actions = [p["action"] for p in t.sent if p["tag"] == "CAMERA_CONTROL"]
    assert actions == ["close", "open"]
