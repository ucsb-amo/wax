"""liveOD's camera host takes the Andor over from the server that has it -- the
SLM spot finder's, while the spot finder has the camera open -- as it borrows
a Basler from the beacon server: RELINQUISH (the other server closes the
camera, verified, before answering), the claim kept while liveOD has the
camera, RETURN when liveOD closes it.

127.0.0.1 only, and no core ever beacons (cam_host_helpers.quiet_network). The
spot finder's server is a stand-in: a real CameraServerCore, policy
"persistent" as the spot finder's is, with a FakeBackend Andor it has opened.
"""
import threading

import pytest

import cam_host_helpers as h
from cam_host_helpers import ANDOR, BASLER, Fakes, andor_params, basler_params, wait_for

ANDOR_ID = "andor_emccd:cam_a"
BASLER_ID = "basler_usb:s1"
SPOT_ID = "camera_server:test:spot_finder"


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


def spot_finder_standin(env, server_id=SPOT_ID, **backend_kw):
    """The spot finder's server with the Andor it opened itself."""
    from beacon.camera.core import CameraServerCore
    from beacon.camera.fake_backend import FakeBackend
    core = CameraServerCore(server_id, policy="persistent", bind_host="127.0.0.1",
                            check_duplicate=False)
    fb = FakeBackend(ANDOR_ID, category="andor_emccd", shape=(4, 6),
                     choices=dict(h.ANDOR_CHOICES), ranges=dict(h.ANDOR_RANGES),
                     settings={"trigger_mode": "int", "gain": 1, "exposure_time": 1e-3},
                     **backend_kw)
    core.add_camera(ANDOR_ID, lambda: fb, category="andor_emccd")
    core.start_in_thread()
    env["cores"].append(core)
    core.worker(ANDOR_ID).call("open", 5.0)
    return core, fb


def ref_of(core):
    from waxx.util.live_od.camera_host.claims import ServerRef
    return ServerRef(core.server_id, "127.0.0.1", core.port)


def host_for(env, resolver, cams=(ANDOR,)):
    """A host whose keeper records every claim's (camera_id, timeout_s)."""
    from waxx.util.live_od.config import LiveODConfig
    keeper = h.keeper_for(resolver)
    keeper.timeouts = []
    claim = keeper.claim

    def recording_claim(camera_id, timeout_s=3.0):
        keeper.timeouts.append((camera_id, timeout_s))
        return claim(camera_id, timeout_s)

    keeper.claim = recording_claim
    host = h.make_host(LiveODConfig(camera_params_list=list(cams)), Fakes(), keeper=keeper)
    host.start()
    env["hosts"].append(host)
    return host, keeper


def test_the_andor_is_taken_over_from_the_spot_finder_at_init_run(env):
    from waxx.util.live_od.camera_host.host import ANDOR_CLAIM_TIMEOUT_S
    spot, sfb = spot_finder_standin(env)
    host, keeper = host_for(env, lambda cid: [ref_of(spot)] if cid == ANDOR_ID else [])
    start = host.begin_run("tok", "cam_a", True, camera_params=andor_params())
    assert start.locked
    assert not sfb.opened                           # the spot finder let go first (verified)
    assert keeper.timeouts == [(ANDOR_ID, ANDOR_CLAIM_TIMEOUT_S)]
    res = spot.reservation(ANDOR_ID)
    assert res["holder"]["label"] == "liveOD" and res["holder"]["holder_id"] == "liveod:test"
    assert host.snapshot()["cameras"]["cam_a"]["claims"][0]["server_id"] == SPOT_ID
    # liveOD opens the Andor itself and carries on with the run
    host.note_run_id("tok", 5)
    host.arm_run("tok", andor_params(), 1).result(5)
    assert host.is_open("cam_a")
    host.end_run("tok")
    assert spot.reservation(ANDOR_ID) is not None   # liveOD keeps it after the run
    host.request("cam_a", "close").result(10)
    assert spot.reservation(ANDOR_ID) is None       # RETURN_CAMERA
    assert not sfb.opened                           # the spot finder reopens it only itself


def test_opening_the_andor_in_liveod_takes_it_over_too(env):
    spot, sfb = spot_finder_standin(env)
    host, _ = host_for(env, lambda cid: [ref_of(spot)] if cid == ANDOR_ID else [])
    host.request("cam_a", "open", origin="claim at start").result(10)
    assert host.is_open("cam_a") and not sfb.opened
    assert spot.reservation(ANDOR_ID)["holder"]["label"] == "liveOD"
    assert host.snapshot()["cameras"]["cam_a"]["host_state"] == "idle"


def test_nothing_is_claimed_when_no_other_server_lists_the_andor(env):
    host, keeper = host_for(env, lambda cid: [])
    start = host.begin_run("tok", "cam_a", True, camera_params=andor_params())
    assert start.locked
    assert keeper.timeouts and keeper.claims() == {}
    assert host.snapshot()["cameras"]["cam_a"]["claims"] == []


def test_a_spot_finder_that_cannot_let_go_refuses_init_run_saying_why(env):
    from waxx.util.live_od.camera_host import HostRefused
    spot, sfb = spot_finder_standin(env)
    sfb.attached_after_close = True                 # the device stays attached after the close
    host, _ = host_for(env, lambda cid: [ref_of(spot)] if cid == ANDOR_ID else [])
    with pytest.raises(HostRefused, match=r"held elsewhere.*still attached"):
        host.begin_run("tok", "cam_a", True, camera_params=andor_params())
    snap = host.snapshot()["cameras"]["cam_a"]
    assert snap["host_state"] == "held_elsewhere" and snap["state"] == "closed"
    assert host.worker("cam_a").locked_by is None and not host.has_run("tok")


def test_a_basler_claim_keeps_its_short_bound(env):
    from beacon.camera.core import CameraServerCore
    from beacon.camera.fake_backend import FakeBackend
    from waxx.util.live_od.camera_host.host import CLAIM_TIMEOUT_S
    beacon = CameraServerCore("camera_server:standin", policy="on_demand", bind_host="127.0.0.1",
                              check_duplicate=False)
    fb = FakeBackend(BASLER_ID, category="basler_usb")
    beacon.add_camera(BASLER_ID, lambda: fb, category="basler_usb")
    beacon.start_in_thread()
    env["cores"].append(beacon)
    host, keeper = host_for(env, lambda cid: [ref_of(beacon)] if cid == BASLER_ID else [],
                            cams=(BASLER,))
    assert host.begin_run("tok", "cam_b", True, camera_params=basler_params()).locked
    assert keeper.timeouts == [(BASLER_ID, CLAIM_TIMEOUT_S)]
    wait_for(lambda: beacon.reservation(BASLER_ID) is not None)
