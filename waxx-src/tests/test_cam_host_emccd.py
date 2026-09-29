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
