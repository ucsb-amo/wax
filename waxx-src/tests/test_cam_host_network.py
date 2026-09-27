"""The camera host's network side, on 127.0.0.1 only and with no beacon sent
(cam_host_helpers.quiet_network):

* borrowing a Basler from a beacon camera server -- a real CameraServerCore
  (policy "on_demand") with a FakeBackend stands in for it: RELINQUISH at
  INIT_RUN, the claim renewed (and re-made when that server restarts),
  RETURN when liveOD lets the camera go, a clear refusal naming whoever
  holds it;
* who may write (T3): programs on this PC only, never a persisted field,
  never an owner-only one;
* a remote SNAP after a run gets the live profile back first;
* live requests: liveOD's (start_stream / stop_stream) and a program's on this
  PC (START_LIVE / STOP_LIVE) are kept apart, and the stream stops when the
  last one is given back; START_LIVE never opens a camera, is refused during a
  run and to other PCs, and gets the live profile back first after a run;
* LocalHostStream (the in-process viewer source) and HostQtBridge.
"""
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import zmq

import cam_host_helpers as h
from cam_host_helpers import ANDOR, BASLER, Fakes, andor_params, basler_params, wait_for
import liveod_qt_helpers as qt

CID = "basler_usb:s1"


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def env(monkeypatch, tmp_path):
    h.quiet_network(monkeypatch, tmp_path)
    before = set(threading.enumerate())
    made = {"hosts": [], "cores": []}
    yield made
    left = []
    for host in made["hosts"]:
        left += h.stop_host(host)
    for core in made["cores"]:
        core.stop(10)
    left += h.join_new_threads(before)
    assert not left, f"threads still running: {left}"


def standin(env, server_id="camera_server:standin"):
    """A beacon camera server with the Basler, on 127.0.0.1."""
    from beacon.camera.core import CameraServerCore
    from beacon.camera.fake_backend import FakeBackend
    core = CameraServerCore(server_id, policy="on_demand", bind_host="127.0.0.1",
                            check_duplicate=False)
    fb = FakeBackend(CID, category="basler_usb")
    core.add_camera(CID, lambda: fb, category="basler_usb", info={"model": "FakeCam"})
    core.start_in_thread()
    env["cores"].append(core)
    return core, fb


def ref_of(core):
    from waxx.util.live_od.camera_host.claims import ServerRef
    return ServerRef(core.server_id, "127.0.0.1", core.port)


def host_for(env, resolver, cams=(BASLER,), ttl_s=30.0, **kw):
    from waxx.util.live_od.config import LiveODConfig
    cfg = LiveODConfig(camera_params_list=list(cams))
    host = h.make_host(cfg, Fakes(), keeper=h.keeper_for(resolver, ttl_s=ttl_s), **kw)
    host.start()
    env["hosts"].append(host)
    return host


def v2(host_port, header, timeout_s=3.0):
    from waxx.util.live_od.camera_host.claims import v2_request
    ctx = zmq.Context()
    try:
        return v2_request(ctx, "127.0.0.1", host_port, header, timeout_s, label="test")
    finally:
        ctx.term()


# ----------------------------------------------------------------------
# RELINQUISH / RETURN
# ----------------------------------------------------------------------

def test_a_basler_is_borrowed_at_init_run_and_given_back(env):
    beacon, bfb = standin(env)
    beacon.attach(CID, "anon", "anon").result(5)               # an old viewer is watching
    wait_for(lambda: bfb.opened)
    host = host_for(env, lambda cid: [ref_of(beacon)] if cid == CID else [])
    start = host.begin_run("tok", "cam_b", True, camera_params=basler_params())
    assert start.locked
    assert not bfb.opened                                       # beacon let go (verified close)
    res = beacon.reservation(CID)
    assert res["holder"]["label"] == "liveOD" and res["holder"]["holder_id"] == "liveod:test"
    assert host.snapshot()["cameras"]["cam_b"]["claims"][0]["server_id"] == beacon.server_id
    host.note_run_id("tok", 5)
    host.arm_run("tok", basler_params(), 1).result(5)
    host.end_run("tok")
    assert beacon.reservation(CID) is not None                  # liveOD keeps it after the run
    host.request("cam_b", "close").result(5)
    assert beacon.reservation(CID) is None                      # RETURN_CAMERA
    wait_for(lambda: bfb.opened, what="beacon reopening for its suspended viewer")


def test_a_camera_held_elsewhere_refuses_init_run_naming_the_holder(env):
    from beacon.camera.reservations import Holder
    from waxx.util.live_od.camera_host import HostRefused
    beacon, bfb = standin(env)
    assert beacon.relinquish([CID], Holder("other", label="spot finder", host="PC2"), 30.0)[CID]["ok"]
    host = host_for(env, lambda cid: [ref_of(beacon)])
    with pytest.raises(HostRefused, match=r"held elsewhere.*held by spot finder on PC2"):
        host.begin_run("tok", "cam_b", True, camera_params=basler_params())
    snap = host.snapshot()["cameras"]["cam_b"]
    assert snap["host_state"] == "held_elsewhere" and snap["state"] == "closed"
    assert host.worker("cam_b").locked_by is None and not host.has_run("tok")


def test_the_claim_is_renewed_and_made_again_when_beacon_restarts(env):
    from beacon.camera.core import CameraServerCore
    from beacon.camera.fake_backend import FakeBackend
    beacon, _ = standin(env)
    current = {"ref": ref_of(beacon)}
    host = host_for(env, lambda cid: [current["ref"]], ttl_s=1.5)   # renewed every 0.5 s
    host.request("cam_b", "open").result(5)
    time.sleep(2.2)
    assert beacon.reservation(CID) is not None                  # renewed past its ttl
    beacon.stop(10)
    again = CameraServerCore(beacon.server_id, policy="on_demand", bind_host="127.0.0.1",
                             check_duplicate=False)
    fb2 = FakeBackend(CID, category="basler_usb")
    again.add_camera(CID, lambda: fb2, category="basler_usb")
    again.start_in_thread()
    env["cores"].append(again)
    current["ref"] = ref_of(again)
    wait_for(lambda: again.reservation(CID) is not None, timeout=8.0,
             what="the claim made again on the restarted server")
    assert again.reservation(CID)["holder"]["holder_id"] == "liveod:test"


# ----------------------------------------------------------------------
# who may write (T3)
# ----------------------------------------------------------------------

def test_remote_writes_view_only_persist_and_owner_rules(env):
    host = host_for(env, lambda cid: [], cams=(BASLER, ANDOR))
    wp = host._write_policy
    ok, why = wp("10.255.255.1", CID, ["gain"])
    assert not ok and "view only" in why and "10.255.255.1" in why
    assert not wp("", CID, ["snap"])[0]
    assert wp("127.0.0.1", CID, ["gain"]) == (True, "")
    ok, why = wp("127.0.0.1", "andor_emccd:cam_a", ["cooler"])
    assert not ok and "owner-only" in why
    assert not wp("10.255.255.1", CID, ["relinquish"])[0]
    host.request("cam_b", "open").result(5)
    host.set_persist("cam_b", True)
    ok, why = wp("127.0.0.1", CID, ["gain", "exposure_time"])
    assert not ok and "persist is on for cam_b" in why and "['gain']" in why
    assert wp("127.0.0.1", CID, ["exposure_time"]) == (True, "")


def test_a_persisted_field_is_refused_over_the_wire(env):
    host = host_for(env, lambda cid: [], serve=True)
    host.request("cam_b", "open").result(5)
    host.set_persist("cam_b", True)
    port = host.core.port
    r = v2(port, {"cmd": "SET_SETTINGS", "camera_id": CID, "values": {"gain": 1.0}})
    assert r["ok"] is False and r["code"] == "refused" and "persist is on" in r["reason"]
    r = v2(port, {"cmd": "SET_SETTINGS", "camera_id": CID, "values": {"exposure_time": 2e-3}})
    assert r["ok"] is True and r["readback"]["exposure_time"]["value"] == pytest.approx(2e-3)


def test_a_remote_snap_after_a_run_gets_the_live_profile_first(env):
    host = host_for(env, lambda cid: [], serve=True)
    host.request("cam_b", "open").result(5)
    host.begin_run("tok", "cam_b", True, camera_params=basler_params())
    host.arm_run("tok", basler_params(), 1).result(5)
    host.end_run("tok")                                         # idle at the run's trigger "On"
    assert host.worker("cam_b").settings["trigger_mode"] == "On"
    port = host.core.port
    r = v2(port, {"cmd": "SNAP", "camera_id": CID, "timeout_s": 2.0})
    assert r["ok"], r
    r = v2(port, {"cmd": "WAIT_FRAME", "camera_id": CID, "snap_id": r["snap_id"],
                  "timeout_s": 3.0}, timeout_s=5.0)
    assert r["ok"] and r["frame"]["source"] == "snap", r
    assert r["frame"]["settings"]["trigger_mode"] == "Off"      # not the run's trigger


# ----------------------------------------------------------------------
# live requests: kept per requester; the last one out stops the stream
# ----------------------------------------------------------------------

SF = "spot-finder-test"             # a program on this PC: its v2 client_id


def live(host, cmd, client_id=SF, camera_id=CID):
    return v2(host.core.port, {"cmd": cmd, "camera_id": camera_id, "client_id": client_id},
              timeout_s=5.0)


def cam(host, key="cam_b"):
    return host.snapshot()["cameras"][key]


def test_liveods_stop_never_ends_a_stream_another_program_asked_for(env):
    host = host_for(env, lambda cid: [], serve=True)
    w = host.worker("cam_b")
    host.start_stream("cam_b").result(5)
    assert w.state == "streaming" and cam(host)["live_requesters"] == ["liveod"]
    r = live(host, "START_LIVE")
    assert r["ok"] and r["live_requesters"] == 2
    host.stop_stream("cam_b").result(5)                        # liveOD's live view closes
    time.sleep(0.2)
    assert w.state == "streaming" and cam(host)["live_requesters"] == [f"v2:{SF}"]
    r = live(host, "STOP_LIVE", client_id="someone-else")      # never asked: changes nothing
    assert r["ok"] and r["stopped"] is False and w.state == "streaming"
    r = live(host, "STOP_LIVE")                                # the last requester
    assert r["ok"] and r["stopped"] is True
    wait_for(lambda: w.state == "idle", what="the stream stopped")
    assert cam(host)["live_requesters"] == []


def test_a_program_streams_an_idle_camera_until_it_gives_its_request_back(env):
    host = host_for(env, lambda cid: [], serve=True)
    w = host.worker("cam_b")
    host.request("cam_b", "open").result(5)
    assert w.state == "idle"
    r = live(host, "START_LIVE")
    assert r["ok"] and w.state == "streaming"
    host.stop_stream("cam_b").result(5)                        # liveOD never asked: nothing stops
    time.sleep(0.2)
    assert w.state == "streaming"
    host.start_stream("cam_b").result(5)                       # liveOD's view joins
    r = live(host, "STOP_LIVE")
    assert r["ok"] and r["stopped"] is False                   # liveOD still wants it
    assert w.state == "streaming" and cam(host)["live_requesters"] == ["liveod"]
    host.stop_stream("cam_b").result(5)
    wait_for(lambda: w.state == "idle", what="the stream stopped")


def test_start_live_never_opens_and_is_refused_to_other_pcs_and_during_a_run(env):
    host = host_for(env, lambda cid: [], serve=True)
    w = host.worker("cam_b")
    r = live(host, "START_LIVE")
    assert r["ok"] is False and r["code"] == "not_open"
    time.sleep(0.2)
    assert not w.is_open and cam(host)["live_requesters"] == []
    host.request("cam_b", "open").result(5)
    # another PC: view only, over the wire as in the policy
    ok, why = host._write_policy("10.255.255.1", CID, ["__live__"])
    assert not ok and "view only" in why and "live-stream requests" in why
    assert host._write_policy("127.0.0.1", CID, ["__live__"]) == (True, "")
    host._is_local = lambda addr: False
    try:
        r = live(host, "START_LIVE")
        assert r["ok"] is False and r["code"] == "refused" and "view only" in r["reason"]
        time.sleep(0.2)
        assert w.state == "idle"                               # nothing started
    finally:
        del host._is_local
    # a run: its lock clears every request, START_LIVE is refused, no stream resumes
    host.start_stream("cam_b").result(5)
    assert live(host, "START_LIVE")["ok"]
    host.begin_run("tok", "cam_b", True, camera_params=basler_params())
    assert cam(host)["live_requesters"] == []
    r = live(host, "START_LIVE")
    assert r["ok"] is False and r["code"] == "run_locked"
    from beacon.camera.worker import LockedError
    with pytest.raises(LockedError, match="camera is locked"):
        host.start_stream("cam_b").result(5)
    host.arm_run("tok", basler_params(), 1).result(5)
    host.end_run("tok")
    time.sleep(0.2)
    assert w.state == "idle" and cam(host)["live_requesters"] == []


def test_start_live_after_a_run_gets_the_live_profile_first(env):
    host = host_for(env, lambda cid: [], serve=True)
    w = host.worker("cam_b")
    host.request("cam_b", "open").result(5)
    host.begin_run("tok", "cam_b", True, camera_params=basler_params())
    host.arm_run("tok", basler_params(), 1).result(5)
    host.end_run("tok")                                         # idle at the run's trigger "On"
    assert w.settings["trigger_mode"] == "On"
    assert live(host, "START_LIVE")["ok"]
    assert w.state == "streaming" and w.settings["trigger_mode"] == "Off"
    r = v2(host.core.port, {"cmd": "WAIT_FRAME", "camera_id": CID, "timeout_s": 3.0,
                            "client_id": SF}, timeout_s=5.0)
    assert r["ok"] and r["frame"]["source"] == "live", r
    assert r["frame"]["settings"]["trigger_mode"] == "Off"     # not the run's trigger
    assert live(host, "STOP_LIVE")["stopped"] is True


def test_local_stream_open_and_close_ask_for_no_live_stream(env):
    from waxx.util.live_od.camera_host.local_stream import LocalHostStream
    host = host_for(env, lambda cid: [], cams=(ANDOR,))
    w = host.worker("cam_a")
    host.request("cam_a", "open").result(5)
    s = LocalHostStream(host, "cam_a")
    assert s.open()["ok"]
    time.sleep(0.2)
    assert w.state == "idle" and cam(host, "cam_a")["live_requesters"] == []
    host.start_stream("cam_a").result(5)                       # the live window's own call
    assert s.close()["ok"]                                     # the viewer leaves: no stop
    time.sleep(0.2)
    assert w.state == "streaming" and cam(host, "cam_a")["live_requesters"] == ["liveod"]
    host.stop_stream("cam_a").result(5)
    wait_for(lambda: w.state == "idle", what="the stream stopped")


def test_the_network_face_lists_the_cameras(env):
    host = host_for(env, lambda cid: [], cams=(BASLER, ANDOR), serve=True)
    r = v2(host.core.port, {"cmd": "LIST_CAMERAS"})
    got = {c["camera_id"]: c for c in r["cameras"]}
    assert set(got) == {CID, "andor_emccd:cam_a"}
    assert got[CID]["owner_key"] == "cam_b" and got[CID]["state"] == "closed"
    hello = r["_hello"]
    assert hello["server_id"] == "camera_server:test:liveod" and hello["policy"] == "persistent"


# ----------------------------------------------------------------------
# LocalHostStream, HostQtBridge
# ----------------------------------------------------------------------

def test_local_stream_frames_settings_and_pacing(env):
    from waxx.util.live_od.camera_host.local_stream import LocalHostStream
    host = host_for(env, lambda cid: [], cams=(ANDOR,))
    from beacon.camera.viewer.sources import missing_members
    s = LocalHostStream(host, "cam_a", min_interval_s=0.1, run_interval_s=0.4)
    assert missing_members(s) == []                           # the viewer's source protocol
    assert s.protocol == "local" and s.camera_id == "andor_emccd:cam_a" and s.name == "cam_a"
    assert s.open()["ok"]
    assert host.snapshot()["cameras"]["cam_a"]["n_subs"] == 1
    assert host.snapshot()["cameras"]["cam_a"]["host_state"] == "closed"    # open() starts nothing
    host.start_stream("cam_a").result(5)
    f1 = s.next_frame(2.0)
    f2 = s.next_frame(2.0)
    assert f1.source == "live" and f2.seq > f1.seq
    d = s.describe()
    assert d["ok"] and d["schema"]["name"] == "andor_emccd"
    r = s.set_settings({"exposure_time": 2e-3})
    assert r["ok"] and r["readback"]["exposure_time"]["value"] == pytest.approx(2e-3)
    r = s.set_settings({"gain": 150})
    assert r["ok"] is False and r["code"] == "refused" and "unlock" in r["error"]
    r = s.set_settings({"gain": 150}, confirmed=frozenset({"gain"}))  # the viewer's unlock
    assert r["ok"] and r["readback"]["gain"]["value"] == 150
    assert s.set_settings({"exposure_time": 3e-3})["ok"]      # still above the cap: stays unlocked
    assert s.set_settings({"gain": 50})["ok"]                 # back under the cap: locked again
    assert s.set_settings({"gain": 150})["ok"] is False
    host.begin_run("tok", "cam_a", True, camera_params=andor_params())
    host.note_run_id("tok", 80001)
    host.arm_run("tok", andor_params(), 4).result(5)
    fb = host.worker("cam_a")._backend
    got = []
    for _ in range(2):
        fb.trigger(1)
        for _ in range(5):              # a live frame from just before the lock may come first
            f = s.next_frame(2.0)
            if f is not None and f.source == "run":
                break
        got.append((f.source, f.run_tag, time.monotonic()))
    assert [g[0] for g in got] == ["run", "run"]
    assert got[0][1].startswith("80001:")
    assert got[1][2] - got[0][2] >= 0.35                       # ~2 Hz during a run
    assert s.status()["run"]["active"] is True
    host.end_run("tok")
    assert s.close()["ok"] and host.snapshot()["cameras"]["cam_a"]["n_subs"] == 0


def test_bridge_emits_snapshots_on_the_gui_thread(env, app):
    from waxx.util.live_od.camera_host.qt_bridge import HostQtBridge
    host = host_for(env, lambda cid: [], cams=(ANDOR,))
    bridge = HostQtBridge(host)
    got = []
    bridge.snapshot_changed.connect(lambda snap: got.append((threading.current_thread(), snap)))
    try:
        host.set_persist("cam_a", True)

        def seen():
            app.processEvents()
            return any(s["cameras"]["cam_a"]["persist"] for _, s in got)
        wait_for(seen, what="a snapshot with persist on")
        assert all(t is threading.main_thread() for t, _ in got)
        bridge.close()
        n = len(got)
        host.set_persist("cam_a", False)
        time.sleep(1.3)
        app.processEvents()
        assert len(got) == n                                    # closed: nothing more
    finally:
        from PyQt6 import sip
        bridge.close()
        sip.delete(bridge)
        app.processEvents()
