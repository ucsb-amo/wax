"""The camera host's run path (waxx.util.live_od.camera_host) against beacon's
FakeBackend: one owner thread per camera, the run lock, the full run profile,
arming, runs that stop at N, frames that never reach the camera thread,
lost frames, faults, shutdown, runs that take no lock, and Persist.

No real camera, no socket outside 127.0.0.1, no beacon (cam_host_helpers).
"""
import threading
import time
import types
from queue import Queue

import numpy as np
import pytest

import cam_host_helpers as h
from cam_host_helpers import ANDOR, APD, BASLER, Fakes, andor_params, basler_params, wait_for


@pytest.fixture
def make(monkeypatch, tmp_path):
    h.quiet_network(monkeypatch, tmp_path)
    before = set(threading.enumerate())
    hosts = []

    def _make(cams=(BASLER,), fakes=None, constraints=None, **kw):
        from waxx.util.live_od.config import LiveODConfig
        cfg = LiveODConfig(camera_params_list=list(cams),
                           camera_constraints=dict(constraints or {}))
        fakes = fakes if fakes is not None else Fakes()
        host = h.make_host(cfg, fakes, **kw)
        host.start()
        hosts.append(host)
        return host, fakes
    yield _make
    left = []
    for host in hosts:
        left += h.stop_host(host)
    left += h.join_new_threads(before)
    assert not left, f"threads still running: {left}"


def arm(host, fakes, token="tok", key="cam_b", params=None, n=3, run_id=7):
    params = params if params is not None else (basler_params() if key == "cam_b" else andor_params())
    host.begin_run(token, key, True, camera_params=params)
    host.note_run_id(token, run_id)
    res = host.arm_run(token, params, n).result(5)
    return res


def grab_in_thread(handle, n, q=None, check=None):
    q = q if q is not None else Queue()
    out = {"exc": None, "armed": 0}

    def run():
        try:
            handle.start_grab(n, q, check, on_armed=lambda: out.__setitem__("armed", out["armed"] + 1))
        except BaseException as exc:
            out["exc"] = exc
    t = threading.Thread(target=run, name="test-grab")
    t.start()
    return t, q, out


# ----------------------------------------------------------------------
# one owner thread; the lock
# ----------------------------------------------------------------------

def test_only_the_worker_thread_calls_the_backend(make):
    host, fakes = make()
    host.start_stream("cam_b").result(5)
    fb = fakes["cam_b"]
    wait_for(lambda: host.wait_frame("cam_b", timeout_s=0.5) is not None, what="a live frame")
    host.set_live("cam_b", {"gain": 5.0})
    arm(host, fakes)
    handle = host.attach_run("tok")
    t, q, out = grab_in_thread(handle, 3)
    wait_for(lambda: out["armed"] == 1)
    fb.trigger(3)
    t.join(5)
    assert out["exc"] is None and q.qsize() == 3
    host.end_run("tok")
    assert fb.foreign_calls == []
    worker = host.worker("cam_b")
    assert {name for _, name in fb.calls} == {worker.name}


def test_a_lock_queued_behind_a_command_still_refuses_it(make):
    from beacon.camera.worker import LockedError
    host, fakes = make()
    host.request("cam_b", "open").result(5)
    fb, w = fakes["cam_b"], host.worker("cam_b")
    fb.hang["read_settings"] = 0.4
    busy = w.submit("read_settings")
    wait_for(lambda: fb.counters["read_settings"] == 1)       # the worker is inside it
    queued = w.submit("apply", values={"gain": 9.0})             # e.g. a live change, queued first
    start = host.begin_run("tok", "cam_b", True, camera_params=basler_params())  # then the run
    busy.result(2)
    assert start.locked
    with pytest.raises(LockedError):
        queued.result(2)
    assert fb.settings["gain"] != 9.0
    assert host.snapshot()["cameras"]["cam_b"]["state"] == "grabbing"


def test_a_run_applies_the_whole_profile_not_changes_on_top(make):
    host, fakes = make()
    host.start_stream("cam_b").result(5)
    host.set_live("cam_b", {"gain": 20.0, "exposure_time": 5e-3, "trigger_source": "Line3"})
    fb = fakes["cam_b"]
    arm(host, fakes, params=basler_params(gain=3.0, exposure_time=2e-4, trigger_source="Line2"))
    profile, purpose = fb.applied[-1]
    assert purpose == "run"
    expected = {"exposure_time": 2e-4, "gain": 3.0, "trigger_source": "Line2", "trigger_mode": "On",
                "trigger_selector": "FrameStart", "line_mode": "Input", "pixel_format": "Mono8",
                "exposure_auto": "Off", "gain_auto": "Off", "user_set": "Default"}
    assert profile == expected
    assert {k: fb.settings[k] for k in expected} == expected   # every "register" is the run's
    assert fb.acquisitions[-1] == ("run", 3)
    host.end_run("tok")


# ----------------------------------------------------------------------
# arming
# ----------------------------------------------------------------------

def test_armed_with_no_triggers_and_ready_is_on_armed(make):
    host, fakes = make()
    res = arm(host, fakes, n=2)
    fb = fakes["cam_b"]
    assert res.stale_discarded == 0 and fb.acquiring
    handle = host.attach_run("tok")
    assert handle.is_opened()
    t, q, out = grab_in_thread(handle, 2)
    wait_for(lambda: out["armed"] == 1)
    time.sleep(0.1)
    assert q.qsize() == 0                                     # nothing without a trigger
    fb.trigger(2)
    t.join(5)
    assert [x[2] for x in list(q.queue)] == [0, 1] and out["exc"] is None
    host.end_run("tok")


def test_warmup_extra_lengthens_only_the_first_frame_wait(make):
    host, fakes = make()
    arm(host, fakes, n=2)
    fb = fakes["cam_b"]
    handle = host.attach_run("tok")
    handle.first_timeout_s = handle.next_timeout_s = 0.3
    q, out = Queue(), {"exc": None}

    def run():
        try:
            handle.start_grab(2, q, None, first_frame_extra_s=0.6)
        except BaseException as exc:
            out["exc"] = exc
    t = threading.Thread(target=run, name="test-grab")
    t.start()
    time.sleep(0.5)                                           # past first_timeout_s alone
    fb.trigger(1)
    wait_for(lambda: q.qsize() == 1, what="the late first frame")
    t.join(5)                                                 # no second trigger: 0.3 s, not extended
    assert isinstance(out["exc"], TimeoutError) and "got 1/2" in str(out["exc"])
    assert "warm-up" not in str(out["exc"])
    host.end_run("tok")


def test_a_free_running_camera_is_refused_at_the_arm(make):
    from beacon.camera.worker import ArmError
    from waxx.util.live_od.camera_host import HostNanny
    host, fakes = make(fakes=Fakes(cam_b={"run_free_runs": True, "cycle_s": 0.005}))
    host.begin_run("tok", "cam_b", True, camera_params=basler_params())
    fut = host.arm_run("tok", basler_params(), 3)
    with pytest.raises(ArmError, match="free-running"):
        fut.result(5)
    camera = HostNanny(host, "tok").persistent_get_camera(types.SimpleNamespace(key="cam_b"))
    assert not camera.is_opened()                             # DummyCamera: "camera not ready"
    host.end_run("tok")


def test_the_run_stops_at_n_and_no_extra_frame_is_taken(make):
    host, fakes = make(fakes=Fakes(cam_b={"ignore_n_frames": True}))  # a device that does not stop
    arm(host, fakes, n=3)
    fb = fakes["cam_b"]
    handle = host.attach_run("tok")
    t, q, out = grab_in_thread(handle, 3)
    wait_for(lambda: out["armed"] == 1)
    fb.trigger(3)
    t.join(5)
    wait_for(lambda: not fb.acquiring, what="the worker stopping the acquisition")
    assert fb.trigger(4) == 0                                  # acquisition stopped at 3
    s = host.end_run("tok")
    assert (s.delivered, s.lost_idx, s.surplus, s.stopped_early) == (3, (), 0, False)
    assert q.qsize() == 3


def test_frames_of_another_run_or_acquisition_never_reach_the_camera_thread(make):
    from beacon.camera.frames import Frame
    host, fakes = make()
    arm(host, fakes, n=2)
    run = host._run_of("tok")
    sink = run.sink

    def frame(**kw):
        d = dict(image=np.full((4, 6), 9, np.uint16), camera_id="basler_usb:s1", seq=10**6,
                 instance="x", acq_gen=run.arm_result.acq_gen, settings_rev=0, source="run",
                 run_tag=run.run_tag, hw_idx=0)
        d.update(kw)
        return Frame(**d)
    sink(frame(run_tag="99:ffffffff"))                        # another run
    sink(frame(acq_gen=run.arm_result.acq_gen - 1))           # an earlier acquisition
    sink(frame(source="live"))                                 # not a run frame
    handle = host.attach_run("tok")
    t, q, out = grab_in_thread(handle, 2)
    wait_for(lambda: out["armed"] == 1)
    time.sleep(0.1)
    assert q.qsize() == 0
    fakes["cam_b"].trigger(2)
    t.join(5)
    got = list(q.queue)
    assert [idx for _, _, idx in got] == [0, 1]
    assert all(int(img[0, 0]) != 9 for img, _, _ in got)      # the real frames, not the injected
    assert sink.counts()["stale"] == 1 and sink.counts()["foreign"] == 1
    host.end_run("tok")


def test_queued_frames_are_private_writable_copies(make):
    host, fakes = make()
    arm(host, fakes, n=1)
    handle = host.attach_run("tok")
    t, q, out = grab_in_thread(handle, 1)
    wait_for(lambda: out["armed"] == 1)
    fakes["cam_b"].trigger(1)
    t.join(5)
    img, _, idx = q.get()
    shared = host.worker("cam_b").slot.latest(sources=("run",))
    assert idx == 0 and img.flags.writeable and not shared.image.flags.writeable
    assert img is not shared.image and np.array_equal(img, shared.image)
    host.end_run("tok")


# ----------------------------------------------------------------------
# lost frames, faults
# ----------------------------------------------------------------------

def test_a_lost_frame_keeps_its_slot_and_ends_the_grab(make):
    from waxx.control.cameras.errors import FrameLostError
    host, fakes = make()
    arm(host, fakes, n=4)
    fb = fakes["cam_b"]
    handle = host.attach_run("tok")
    t, q, out = grab_in_thread(handle, 4)
    wait_for(lambda: out["armed"] == 1)
    fb.trigger(1)
    fb.lose_next(1)
    fb.trigger(2)
    t.join(5)
    assert isinstance(out["exc"], FrameLostError) and out["exc"].lost == (1,)
    assert [idx for _, _, idx in list(q.queue)] == [0, 2]     # frame 2 in slot 2, not slot 1
    s = host.end_run("tok")
    assert s.lost_idx == (1,) and any("lost" in p for p in s.problems)


def test_the_last_frame_lost_is_reported_when_the_acquisition_ends(make):
    from waxx.control.cameras.errors import FrameLostError
    host, fakes = make()
    arm(host, fakes, n=2)
    fb = fakes["cam_b"]
    handle = host.attach_run("tok")
    t, q, out = grab_in_thread(handle, 2)
    wait_for(lambda: out["armed"] == 1)
    fb.trigger(1)
    fb.lose_next(1)
    fb.trigger(1)
    t.join(5)
    assert isinstance(out["exc"], FrameLostError) and out["exc"].lost == (1,)
    assert [idx for _, _, idx in list(q.queue)] == [0]
    host.end_run("tok")


def test_a_driver_error_during_the_run_is_a_run_fault_with_no_reopen(make):
    from waxx.util.live_od.camera_host.legacy import CameraRunFault
    host, fakes = make()
    arm(host, fakes, n=3)
    fb = fakes["cam_b"]
    opens = fb.counters["open"]
    handle = host.attach_run("tok")
    t, q, out = grab_in_thread(handle, 3)
    wait_for(lambda: out["armed"] == 1)
    fb.trigger(1)
    wait_for(lambda: q.qsize() == 1)
    fb.fail_next("retrieve", RuntimeError("USB transfer failed"))
    t.join(5)
    assert isinstance(out["exc"], CameraRunFault) and "USB transfer failed" in str(out["exc"])
    snap = host.snapshot()["cameras"]["cam_b"]
    assert snap["host_state"] == "run_fault" and snap["state"] == "failed"
    time.sleep(0.3)
    assert fb.counters["open"] == opens                        # no reopen while the run holds it
    s = host.end_run("tok")
    assert s.run_fault and any("camera error" in p for p in s.problems)
    wait_for(lambda: host.snapshot()["cameras"]["cam_b"]["host_state"] == "idle")
    assert fb.counters["open"] == opens + 1                    # recovered after the run, idle


def test_shutdown_is_bounded_when_a_camera_hangs(make):
    host, fakes = make()
    host.request("cam_b", "open").result(5)
    fakes["cam_b"].hang["close"] = 2.0
    t0 = time.monotonic()
    report = host.shutdown(0.5)
    assert time.monotonic() - t0 < 1.8
    assert report["cameras_closed"] is False
    assert host.shutdown(0.5) == {"already": True}             # idempotent


# ----------------------------------------------------------------------
# runs that take no lock; the lock's lifetime
# ----------------------------------------------------------------------

def test_no_camera_and_apd_runs_take_no_lock(make):
    from waxx.util.live_od.camera_host import HostRefused
    host, fakes = make(cams=(BASLER, APD))
    s = host.begin_run("t1", "cam_b", False, camera_params=basler_params())
    assert not s.locked and host.worker("cam_b").locked_by is None
    s = host.begin_run("t2", "apd", True, camera_params={"key": "apd"})
    assert not s.locked and host.worker("cam_b").locked_by is None
    with pytest.raises(HostRefused, match="apd"):
        host.arm_run("t2", {}, 1).result(1)
    assert host.snapshot()["cameras"]["apd"]["host_state"] == "virtual"
    assert "cam_b" not in fakes                                # never opened


def test_a_new_run_on_the_same_camera_ends_the_old_ones_hold(make):
    host, fakes = make()
    arm(host, fakes, token="old", n=2)
    host.begin_run("new", "cam_b", True, camera_params=basler_params())
    assert host.worker("cam_b").locked_by == "new" and not host.has_run("old")


def test_the_camera_stays_idle_at_the_run_settings_after_the_run(make):
    host, fakes = make()
    host.start_stream("cam_b").result(5)
    arm(host, fakes, params=basler_params(gain=7.0), n=1)
    fakes["cam_b"].trigger(1)
    host.end_run("tok")
    snap = host.snapshot()["cameras"]["cam_b"]
    assert snap["host_state"] == "idle"                        # no stream resumes (Q5)
    assert fakes["cam_b"].settings["gain"] == 7.0 and fakes["cam_b"].settings["trigger_mode"] == "On"
    host.start_stream("cam_b").result(5)                       # a stream puts the live profile back
    assert fakes["cam_b"].settings["trigger_mode"] == "Off"


def test_operator_release_ends_a_hold_and_closes(make, caplog):
    host, fakes = make()
    arm(host, fakes, n=2)
    host.operator_release("cam_b").result(5)
    assert not host.has_run("tok")
    assert host.snapshot()["cameras"]["cam_b"]["host_state"] == "closed"


def test_close_is_refused_while_a_run_holds_the_camera(make):
    from waxx.util.live_od.camera_host import HostRefused
    host, fakes = make()
    arm(host, fakes, n=2)
    with pytest.raises(HostRefused, match="holds the camera"):
        host.request("cam_b", "close").result(5)
    assert host.is_open("cam_b")
    host.end_run("tok")


# ----------------------------------------------------------------------
# Persist (D-a)
# ----------------------------------------------------------------------

def test_persist_takes_only_the_whitelist_never_exposure(make):
    from beacon.camera.backend import ApplyRefused
    host, fakes = make(cams=(ANDOR,))
    host.start_stream("cam_a").result(5)
    host.set_live("cam_a", {"gain": 30, "exposure_time": 2e-3, "vs_speed": 2})
    ps = host.set_persist("cam_a", True)
    assert ps.on and ps.since is not None
    assert ps.values == {"gain": 30, "hs_speed": 0, "preamp": 2, "vs_speed": 2, "vs_amp": 3,
                         "baseline_clamp": 1}
    assert "exposure_time" not in ps.values
    with pytest.raises(ApplyRefused, match="exposure_time"):
        host.set_persist("cam_a", True, values={"exposure_time": 1e-3})


def test_persist_goes_on_top_of_the_run_and_is_recorded(make):
    host, fakes = make(cams=(ANDOR,))
    host.start_stream("cam_a").result(5)
    host.set_live("cam_a", {"gain": 30})
    host.set_persist("cam_a", True)
    start = host.begin_run("tok", "cam_a", True, camera_params=andor_params(gain=300))
    assert start.persist_on
    assert start.overrides == {"gain": {"requested": 300, "applied": 30, "origin": "persist"}}
    text = start.persist_warning(80713)
    assert text.startswith("PERSISTED CAMERA SETTINGS: run 80713 on cam_a does NOT use the "
                           "experiment's camera_params for 1 field(s):")
    assert "gain  300 -> 30 (persisted)" in text and start.persist_since in text
    host.note_run_id("tok", 80713)
    host.arm_run("tok", andor_params(gain=300), 1).result(5)
    assert fakes["cam_a"].applied[-1][0]["gain"] == 30
    s = host.end_run("tok")
    assert s.overrides["gain"]["origin"] == "persist" and s.persist_since == start.persist_since


def test_persist_is_off_when_unset_and_on_a_new_host(make):
    host, fakes = make(cams=(ANDOR,))
    host.start_stream("cam_a").result(5)
    host.set_persist("cam_a", True)
    assert host.persist("cam_a").on
    host.set_persist("cam_a", False)
    assert not host.persist("cam_a").on and host.persist("cam_a").values == {}
    host.set_persist("cam_a", True)
    host2, _ = make(cams=(ANDOR,))
    assert not host2.persist("cam_a").on                       # a restart clears it


def test_the_snapshot_carries_the_last_runs_request(make):
    import json
    from beacon.camera.schema import ANDOR_EMCCD
    from waxx.util.live_od.camera_host import HostRefused
    host, fakes = make(cams=(ANDOR,))

    def snap():
        return host.snapshot()["cameras"]["cam_a"]
    assert snap()["run_request"] == {}                         # no run yet
    host.start_stream("cam_a").result(5)
    host.set_live("cam_a", {"gain": 30})
    host.set_persist("cam_a", True)
    params = andor_params(gain=300, exposure_time=1e-5)       # 1e-5 s: below the fake's minimum
    host.begin_run("tok", "cam_a", True, camera_params=params)
    req = snap()["run_request"]
    assert set(req) == set(ANDOR_EMCCD.keys(run="param"))
    # the request (camera_params), not what Persist or a clamp made of it
    assert req["gain"] == 300 and req["exposure_time"] == 1e-5
    assert req["sensor_roi"] == [0, 512, 0, 512, 1, 1] and req["trigger_mode"] == "ext"
    assert json.loads(json.dumps(req)) == req
    host.note_run_id("tok", 80713)
    host.arm_run("tok", params, 1).result(5)
    s = snap()
    assert s["run_request"] == req
    assert s["settings"]["gain"] == 30 and s["settings"]["exposure_time"] == pytest.approx(2e-5)
    host.end_run("tok")
    assert snap()["run_request"] == req                        # the last run's, after it ended
    with pytest.raises(HostRefused, match="lacks run field"):
        host.begin_run("bad", "cam_a", True, camera_params={"exposure_time": 1e-3})
    assert snap()["run_request"] == req                        # a refused run is no run
    host.set_persist("cam_a", False)
    host.begin_run("t2", "cam_a", True, camera_params=andor_params(gain=5))
    assert snap()["run_request"]["gain"] == 5 and snap()["run_request"]["exposure_time"] == 1e-3
    host.end_run("t2")


def test_persist_cannot_change_while_a_run_holds_the_camera(make):
    from waxx.util.live_od.camera_host import HostRefused
    host, fakes = make(cams=(ANDOR,))
    host.begin_run("tok", "cam_a", True, camera_params=andor_params())
    with pytest.raises(HostRefused, match="holds the camera"):
        host.set_persist("cam_a", True)
    host.end_run("tok")


def test_a_refused_constraint_refuses_the_run_before_anything_is_locked(make):
    from beacon.camera.schema import Constraint
    from waxx.util.live_od.camera_host import HostRefused
    du897 = Constraint(when=(("vs_speed", "==", 0), ("vs_amp", "==", 0)), level="refuse",
                       reason="0.3 us vertical clock with Normal amplitude transfers no charge")
    host, fakes = make(cams=(ANDOR,), constraints={"andor_emccd": (du897,)})
    with pytest.raises(HostRefused, match="vs_speed=0, vs_amp=0 refused.*no charge"):
        host.begin_run("tok", "cam_a", True, camera_params=andor_params(vs_speed=0, vs_amp=0))
    assert host.worker("cam_a").locked_by is None and not host.has_run("tok")
    with pytest.raises(HostRefused, match="lacks run field"):
        host.begin_run("tok", "cam_a", True, camera_params={"exposure_time": 1e-3})


def test_live_em_gain_is_capped_unless_unlocked(make):
    from beacon.camera.backend import ApplyRefused
    host, fakes = make(cams=(ANDOR,))
    host.start_stream("cam_a").result(5)
    with pytest.raises(ApplyRefused, match="gain=150.*unlock"):
        host.set_live("cam_a", {"gain": 150})
    host.set_live("cam_a", {"gain": 150, "em_gain_unlocked": True})
    assert fakes["cam_a"].settings["gain"] == 150
    with pytest.raises(ApplyRefused, match="frame_transfer"):
        host.set_live("cam_a", {"frame_transfer": 1})
