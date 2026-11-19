"""The Andor camera server (beacon.andor: the generic camera server with the
Andor provider) over the real EMCCDBackend / AndorEMCCD and pylablib, against
the fake SDK2 (fakes/fake_sdk2.py): served by its serial without touching the
device, opened on demand, a camera with another serial refused, and lent to
liveOD under the id liveOD's camera host gives it.  Bound to 127.0.0.1, no
beacon (conftest), state and device locks in the test's own folders."""
import sys
import time
from pathlib import Path

import pytest
import zmq

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
import fake_sdk2  # noqa: E402

SERIAL = "6321"         # what the fake SDK reports
CID = f"andor_emccd:{SERIAL}"
HOLDER = {"holder_id": "liveod-1234", "server_id": "camera_server:kong:liveod",
          "host": "kong", "pid": 1234, "label": "liveOD"}


@pytest.fixture
def serve(monkeypatch, tmp_path):
    from beacon.camera import device_lock as dl
    monkeypatch.setenv("BEACON_STATE_DIR", str(tmp_path / "beacon_state"))
    monkeypatch.setenv(dl.ENV_LOCK_DIR, str(tmp_path / "locks"))
    fake = fake_sdk2.FakeSDK2Lib(serial=int(SERIAL))
    fake_sdk2.install(monkeypatch, fake)
    from beacon.andor.camera_server import AndorCameraServer
    made = []

    def make(serials=(SERIAL,), **kw):
        s = AndorCameraServer(serials, bind_host="127.0.0.1", check_duplicate=False,
                              defaults=False, **kw)
        s.start_in_thread()
        made.append(s)
        return s
    make.fake = fake
    yield make
    for s in made:
        s.stop(20)
    for lock in list(dl._HELD.values()):
        lock.release()


def req(server, cmd, **header):
    from beacon.camera.lending import v2_request
    return v2_request(zmq.Context.instance(), "127.0.0.1", server.port,
                      {"cmd": cmd, **header}, 30.0, label="test")


def listed(server):
    (cam,) = req(server, "LIST_CAMERAS")["cameras"]
    return cam


def wait_for(cond, timeout_s=10.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        got = cond()
        if got:
            return got
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def test_du897_server_serves_its_serial_and_opens_only_on_demand(serve):
    from beacon.camera.server import CameraServer
    s = serve()
    fake = serve.fake
    assert isinstance(s, CameraServer) and s.server_id.endswith(":andor")
    assert s.relinquish_timeout_s == 15.0 and s.open_timeout_s == 60.0
    cam = listed(s)
    assert cam["camera_id"] == CID and cam["category"] == "andor_emccd"
    assert cam["serial"] == SERIAL
    # serving it touched nothing: the SDK was never initialised
    assert not fake.initialized and fake.calls("Initialize") == []
    head = req(s, "OPEN_CAMERA", camera_id=CID, client_id="viewer")
    assert head["ok"], head
    assert fake.initialized
    d = req(s, "DESCRIBE", camera_id=CID)
    assert d["schema"]["name"] == "andor_emccd" and d["serial"] == SERIAL
    # the live profile: free run, the category's readout settings
    assert d["settings"]["trigger_mode"] == "int"
    assert (d["settings"]["vs_speed"], d["settings"]["vs_amp"]) == (1, 3)


def test_du897_server_readout_and_constraints_reach_the_backend(serve):
    from beacon.camera.schema import Constraint
    rule = Constraint(when=(("vs_speed", "==", 0), ("vs_amp", "==", 0)), level="refuse",
                      reason="no charge moves")
    s = serve(readout={"hs_speed": 1, "preamp": 1}, constraints=(rule,))
    assert req(s, "OPEN_CAMERA", camera_id=CID, client_id="viewer")["ok"]
    d = req(s, "DESCRIBE", camera_id=CID)
    assert (d["settings"]["hs_speed"], d["settings"]["preamp"]) == (1, 1)
    head = req(s, "SET_SETTINGS", camera_id=CID, values={"vs_speed": 0, "vs_amp": 0})
    assert not head["ok"] or head.get("errors"), head
    assert "no charge moves" in str(head)
    d = req(s, "DESCRIBE", camera_id=CID)
    assert (d["settings"]["vs_speed"], d["settings"]["vs_amp"]) == (1, 3)


def test_du897_server_refuses_a_camera_reporting_another_serial(serve):
    s = serve(serials=("9999",))
    assert listed(s)["camera_id"] == "andor_emccd:9999"
    head = req(s, "OPEN_CAMERA", camera_id="andor_emccd:9999", client_id="viewer")
    assert not head["ok"]
    assert "expected Andor serial 9999" in head["error"] and SERIAL in head["error"]
    # nothing is left open or locked
    assert not serve.fake.initialized
    from beacon.camera.device_lock import DeviceLock
    from beacon.drivers.andor.params import DEVICE_LOCK_KEY
    lock = DeviceLock(DEVICE_LOCK_KEY)
    lock.acquire()
    lock.release()


def test_du897_server_without_a_serial_serves_nothing(serve):
    s = serve(serials=())
    assert req(s, "LIST_CAMERAS")["cameras"] == []


def test_du897_server_lends_to_liveod_with_a_verified_close(serve):
    s = serve()
    fake = serve.fake
    assert req(s, "OPEN_CAMERA", camera_id=CID, client_id="viewer")["ok"]
    assert fake.initialized
    head = req(s, "RELINQUISH_CAMERA", camera_ids=[CID], holder=HOLDER, ttl_s=30.0)
    assert head["ok"], head
    # answered after the close: SDK shut down, device lock free for liveOD
    assert not fake.initialized and fake.shutdown_count == 1
    from beacon.camera.device_lock import DeviceLock
    from beacon.drivers.andor.params import DEVICE_LOCK_KEY
    lock = DeviceLock(DEVICE_LOCK_KEY)
    lock.acquire()
    lock.release()
    assert listed(s)["state"] == "reserved"
    # reserved for liveOD: no one else opens it meanwhile
    assert not req(s, "OPEN_CAMERA", camera_id=CID, client_id="other")["ok"]
    head = req(s, "RETURN_CAMERA", camera_ids=[CID], holder_id=HOLDER["holder_id"])
    assert head["ok"], head
    wait_for(lambda: listed(s)["state"] != "reserved")


def test_liveod_names_and_builds_the_camera_the_server_serves():
    from waxx.control.cameras.camera_param_classes import AndorParams
    from waxx.util.live_od.camera_host.host import camera_specs, default_backend_factory
    (spec,) = camera_specs([AndorParams(serial_number=SERIAL, key="andor")])
    assert spec.camera_id == CID and spec.serial == SERIAL
    assert default_backend_factory(spec)._expected_serial == SERIAL
    # no serial in the table: the old id, no check
    (old,) = camera_specs([AndorParams(key="andor")])
    assert old.camera_id == "andor_emccd:andor"
    assert default_backend_factory(old)._expected_serial is None


def test_du897_old_waxx_names_are_the_beacon_modules():
    import beacon.camera.device_lock
    import beacon.camera.errors
    import beacon.drivers.andor.backend
    import beacon.drivers.andor.driver
    import waxx.control.cameras.andor
    import waxx.control.cameras.device_lock
    import waxx.control.cameras.emccd_backend
    import waxx.control.cameras.errors
    assert waxx.control.cameras.andor is beacon.drivers.andor.driver
    assert waxx.control.cameras.emccd_backend is beacon.drivers.andor.backend
    assert waxx.control.cameras.device_lock is beacon.camera.device_lock
    assert waxx.control.cameras.errors is beacon.camera.errors
    from waxx.control.cameras.camera_param_classes import RunFieldRefused, AndorParams
    from beacon.drivers.andor import params
    assert RunFieldRefused is params.RunFieldRefused
    assert AndorParams.DETECTOR_SHAPE == params.DETECTOR_SHAPE
    from waxx.config.timeouts import CAMERA_GRAB_TIMEOUT_ANDOR
    assert CAMERA_GRAB_TIMEOUT_ANDOR == params.GRAB_TIMEOUT_S == 60.
