"""The camera host with its real EMCCD backend (waxx EMCCDBackend over the real
AndorEMCCD and pylablib code) against the fake SDK2 library (fakes/fake_sdk2.py):
the host's default backend factory, the lab's constraint handed to the
backend, a run's frames in their hardware-index slots, and the shutdown order
(acquisition stopped, shutter closed, SDK shut down). The OS device lock lives
in tmp_path.
"""
import sys
import threading
from pathlib import Path
from queue import Queue

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
import fake_sdk2  # noqa: E402

import cam_host_helpers as h  # noqa: E402
from cam_host_helpers import ANDOR, andor_params, wait_for  # noqa: E402

SHUTTER_MODE = {0: "auto", 1: "open", 2: "closed"}


@pytest.fixture
def host(monkeypatch, tmp_path):
    from beacon.camera.schema import Constraint
    from waxx.control.cameras import device_lock as dl
    from waxx.util.live_od.config import LiveODConfig
    from waxx.util.live_od.camera_host import CameraHost, HostServerCore
    import functools
    h.quiet_network(monkeypatch, tmp_path)
    monkeypatch.setenv(dl.ENV_LOCK_DIR, str(tmp_path / "locks"))
    fake = fake_sdk2.FakeSDK2Lib()
    fake_sdk2.install(monkeypatch, fake)
    du897 = Constraint(when=(("vs_speed", "==", 0), ("vs_amp", "==", 0)), level="refuse",
                       reason="0.3 us vertical clock with Normal amplitude transfers no charge")
    cfg = LiveODConfig(camera_params_list=[ANDOR], camera_constraints={"andor_emccd": (du897,)})
    before = set(threading.enumerate())
    host = CameraHost(cfg, core_factory=functools.partial(HostServerCore, check_duplicate=False,
                                                          bind_host="127.0.0.1"),
                      keeper=h.keeper_for(), serve=False, local_addresses={"127.0.0.1"})
    host.start()
    yield host, fake
    left = h.stop_host(host)
    left += h.join_new_threads(before)
    for lock in list(dl._HELD.values()):
        lock.release()
    assert not left, left


def test_a_run_on_the_emccd_backend(host):
    host, fake = host
    start = host.begin_run("tok", "cam_a", True, camera_params=andor_params(gain=300))
    assert start.locked
    host.note_run_id("tok", 81000)
    res = host.arm_run("tok", andor_params(gain=300), 3).result(15)
    assert res.stale_discarded == 0
    assert fake.hw["trigger"] == 1 and fake.hw["em_gain"] == 300           # ext, the run's gain
    assert SHUTTER_MODE[fake.hw["shutter"][1]] == "open"
    handle = host.attach_run("tok")
    q, out = Queue(), {}

    def grab():
        try:
            handle.start_grab(3, q, None, on_armed=lambda: out.setdefault("armed", True))
        except BaseException as exc:
            out["exc"] = exc
    t = threading.Thread(target=grab)
    t.start()
    wait_for(lambda: out.get("armed"))
    fake.trigger(3)
    t.join(10)
    assert "exc" not in out
    frames = list(q.queue)
    assert [idx for _, _, idx in frames] == [0, 1, 2]
    acq_gen = fake_sdk2.FakeSDK2Lib.decode(frames[0][0])[0]
    assert [fake_sdk2.FakeSDK2Lib.decode(img) for img, _, _ in frames] == [(acq_gen, i) for i in range(3)]
    s = host.end_run("tok")
    assert s.delivered == 3 and s.lost_idx == ()


@pytest.mark.parametrize("how", ["abort", "end", "superseded"])
def test_an_extreme_gain_run_leaves_the_camera_at_300(host, how):
    host, fake = host
    p = andor_params(gain=800, extreme_emccd_gain=True)
    host.begin_run("tok", "cam_a", True, camera_params=p)
    host.note_run_id("tok", 81001)
    host.arm_run("tok", p, 3).result(15)
    assert fake.hw["em_gain"] == 800 and fake.hw["em_advanced"] == 1       # the run's
    if how == "abort":
        s = host.abort_run("tok")
    elif how == "end":
        s = host.end_run("tok")
    else:                                                                  # a new run takes over
        host.begin_run("tok2", "cam_a", True, camera_params=andor_params(gain=300))
        s = None
    assert fake.hw["em_gain"] == 300 and fake.hw["em_advanced"] == 0
    if s is not None:
        assert s.errors == ()
    assert host.worker("cam_a").locked_by in (None, "tok2")


def test_a_dead_experiments_extreme_gain_is_set_back_while_its_run_stays_open(host):
    host, fake = host
    p = andor_params(gain=800, extreme_emccd_gain=True)
    host.begin_run("tok", "cam_a", True, camera_params=p)
    assert not host.extreme_gain_pending("tok")                           # not armed yet
    host.arm_run("tok", p, 3).result(15)
    assert host.extreme_gain_pending("tok")
    assert host.undo_extreme_em_gain("tok", "the experiment's process is gone")
    assert fake.hw["em_gain"] == 300 and fake.hw["em_advanced"] == 0
    assert not host.extreme_gain_pending("tok")
    assert host.active_run("cam_a") == "tok"                              # the run stays open
    assert not host.undo_extreme_em_gain("tok", "again")                  # once
    start = len(fake.log)
    s = host.abort_run("tok")                                             # the Reset, later
    assert s.errors == ()
    assert "SetEMCCDGain" not in [n for n, _ in fake.log[start:]]
    assert fake.hw["em_gain"] == 300 and fake.hw["em_advanced"] == 0


def test_an_ordinary_run_has_nothing_to_set_back(host):
    host, fake = host
    host.begin_run("tok", "cam_a", True, camera_params=andor_params(gain=300))
    host.arm_run("tok", andor_params(gain=300), 1).result(15)
    assert not host.extreme_gain_pending("tok")
    assert not host.undo_extreme_em_gain("tok", "x")
    assert not host.extreme_gain_pending("no such token")
    host.end_run("tok")


def test_an_extreme_gain_run_without_the_opt_in_is_refused_at_init_run(host):
    from waxx.util.live_od.camera_host import HostRefused
    host, fake = host
    with pytest.raises(HostRefused, match=r"self\._extreme_emccd_gain = True"):
        host.begin_run("tok", "cam_a", True, camera_params=andor_params(gain=800))
    assert host.worker("cam_a").locked_by is None


def test_an_ordinary_run_end_does_not_touch_the_gain(host):
    host, fake = host
    host.begin_run("tok", "cam_a", True, camera_params=andor_params(gain=250))
    host.arm_run("tok", andor_params(gain=250), 1).result(15)
    start = len(fake.log)
    host.abort_run("tok")
    assert fake.hw["em_gain"] == 250
    assert "SetEMCCDGain" not in [n for n, _ in fake.log[start:]]


def test_the_lab_constraint_reaches_the_backend(host):
    from beacon.camera.backend import ApplyRefused
    host, fake = host
    host.request("cam_a", "open").result(15)
    w = host.worker("cam_a")
    live = dict(host._live_profile("cam_a"), vs_speed=0, vs_amp=0)
    with pytest.raises(ApplyRefused, match="no charge"):                  # the backend's own check
        w.call("apply", 10, values=live, purpose="live")
    with pytest.raises(ApplyRefused, match="no charge"):                  # and the host's, before it
        host.set_live("cam_a", {"vs_speed": 0, "vs_amp": 0})


def test_shutdown_stops_closes_the_shutter_then_shuts_down(host):
    host, fake = host
    host.start_stream("cam_a").result(15)
    start = len(fake.log)
    report = host.shutdown(10.0)
    assert report["cameras_closed"] is True
    names = [n for n, _ in fake.log[start:]]
    assert names.index("AbortAcquisition") < names.index("SetShutterEx") < names.index("ShutDown")
    assert SHUTTER_MODE[fake.hw["shutter"][1]] == "closed"
