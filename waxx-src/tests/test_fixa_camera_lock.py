"""The camera thread (CameraBaby) and CameraNanny with stand-in cameras:

* a run whose camera settings are refused fails its WAIT_CAM_READY at once,
  naming the refusal, instead of after the whole ready timeout (m2);
* one thread at a time holds a camera from applying its run's settings to the
  end of its grab, and a thread stopped (superseded) before or while applying
  them grabs nothing (m8).

No camera, no file outside tmp_path, no socket, no beacon.
"""
import logging
import os
import threading
import time
import types
from queue import Queue

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt

import liveod_qt_helpers as qt


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def records():
    got = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            got.append(record.getMessage())
    handler = ListHandler(logging.INFO)
    log = logging.getLogger("waxx.live_od")
    old_level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    yield got
    log.removeHandler(handler)
    log.setLevel(old_level)


def stand_in_handler(camera_params, n_img=2):
    discarded = []
    return types.SimpleNamespace(
        camera_params=camera_params,
        params=types.SimpleNamespace(N_img=n_img, N_shots=1, N_pwa_per_shot=1),
        read_params=lambda: None, grab_finished=lambda: None,
        writer=types.SimpleNamespace(discard=lambda delete=True: discarded.append(delete)),
        discarded=discarded)


def basler_params(exposure_time, gain=6.0):
    return types.SimpleNamespace(key="xy_basler", camera_type="basler", exposure_time=exposure_time,
                                 gain=gain, trigger_source="Line1")


def andor_params(**kw):
    p = types.SimpleNamespace(key="andor", camera_type="andor", exposure_time=10e-6, gain=300.,
                              preamp=2, hs_speed=0, vs_speed=1, vs_amp=3, baseline_clamp=1)
    vars(p).update(kw)
    return p


def wait_for(predicate, timeout_s=5.0):
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ----------------------------------------------------------------------
# m2: refused settings fail WAIT_CAM_READY at once
# ----------------------------------------------------------------------

class RefusingAndor:
    """An open Andor-like camera whose run settings are refused at ``step``."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = []

    def is_opened(self):
        return True

    def Close(self):
        pass

    def stop_grab(self):
        self.calls.append("stop_grab")

    def start_grab(self, *a, **k):
        raise AssertionError("a refused camera must not grab")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            self.calls.append(name)
            if name == "set_amp_mode_checked":
                raise self.exc
        return call


def _refusals():
    from beacon.camera.backend import ApplyRefused
    from waxx.control.cameras.andor import AmpModeUnavailable
    return [AmpModeUnavailable("amplifier mode (channel 0, output amp 0, hs_speed 3, preamp 2) "
                               "is not available on this camera; nothing was sent."),
            ApplyRefused("frame_transfer", "1 is refused: D-g, no frame transfer")]


@pytest.fixture
def server(app, tmp_path, monkeypatch):
    from live_od_data_fakes import FakeSaver, patch_payload_stash
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    return LiveODServer(server_talk=None, data_saver=FakeSaver(tmp_path))   # never started


def client_on(srv):
    from waxx.util.live_od.live_od_client import LiveODClient
    handlers = {"INIT_RUN": srv._handle_init_run, "WAIT_CAM_READY": srv._handle_wait_cam_ready}
    client = LiveODClient.__new__(LiveODClient)      # no discovery, no socket
    client.last_reset_requested = False
    client._send_recv = lambda payload, rcvtimeo_ms=None: handlers[payload["tag"]](dict(payload))
    return client


@pytest.mark.parametrize("which", [0, 1], ids=["amp_mode_unavailable", "apply_refused"])
def test_refused_settings_fail_the_ready_wait_at_once_naming_the_refusal(server, which):
    from waxx.util.live_od.camera_mother import CameraBaby
    from waxx.util.live_od.camera_nanny import CameraNanny
    refusal = _refusals()[which]
    srv, client = server, client_on(server)
    client.init_run({"tag": "INIT_RUN", "save_data": True, "capture_images": True,
                     "camera_key": "andor", "params": {"N_img": 2}, "N_shots_with_repeats": 1,
                     "expt_class": "fixa_test"})
    cam = RefusingAndor(refusal)
    nanny = CameraNanny()
    nanny.get_camera = lambda params: cam
    handler = stand_in_handler(andor_params(hs_speed=3))
    baby = CameraBaby(handler, "Refused", Queue(), nanny)
    baby.grab_failed_signal.connect(srv.on_grab_failed, Qt.ConnectionType.DirectConnection)
    baby.run()                                               # on this thread
    assert "start_grab" not in cam.calls
    assert type(refusal).__name__ in srv._grab_failure and str(refusal) in srv._grab_failure
    t0 = time.monotonic()
    with pytest.raises(ValueError) as err:
        client.wait_cam_ready(timeout=30.0)
    assert time.monotonic() - t0 < 2.0                       # at once, not after the timeout
    text = str(err.value)
    assert "camera failed before it was ready" in text
    assert "the run's settings were not applied" in text and str(refusal) in text


def test_a_stopped_thread_reports_no_grab_failure(app):
    from waxx.util.live_od.camera_mother import CameraBaby
    from waxx.util.live_od.camera_nanny import CameraNanny
    cam = RefusingAndor(_refusals()[0])
    nanny = CameraNanny()
    nanny.get_camera = lambda params: cam
    baby = CameraBaby(stand_in_handler(andor_params()), "Stopped", Queue(), nanny)
    failures = []
    baby.grab_failed_signal.connect(failures.append, Qt.ConnectionType.DirectConnection)
    baby.request_stop()                                      # superseded before it began
    baby.run()
    assert failures == [] and cam.calls == []                # nothing applied, nothing reported


def test_the_nanny_reports_why_it_gave_no_camera():
    from waxx.control.cameras import DummyCamera
    from waxx.util.live_od.camera_nanny import CameraNanny
    report = {}
    out = CameraNanny().update_params(RefusingAndor(_refusals()[0]), andor_params(), report=report)
    assert isinstance(out, DummyCamera)
    assert report["error"].startswith("AmpModeUnavailable: amplifier mode")


# ----------------------------------------------------------------------
# m8: one thread at a time from update_params to the end of the grab
# ----------------------------------------------------------------------

class SharedCam:
    """One physical camera, as the drivers have it: a reentrant per-device grab
    lock that start_grab holds, a non-blocking stop_grab. Records every call
    with the exposure the camera had at the time."""

    def __init__(self):
        self.lock = threading.RLock()
        self.calls = []
        self.exposure = None
        self.on_set_gain = None

    def is_opened(self):
        return True

    def Close(self):
        pass

    def grab_lock(self):
        return self.lock

    def IsGrabbing(self):
        return False

    def set_exposure(self, value):
        self.calls.append(("set_exposure", value))
        self.exposure = value

    def set_gain(self, value):
        self.calls.append(("set_gain", value))
        if self.on_set_gain is not None:
            self.on_set_gain()

    def configure_trigger(self, source):
        self.calls.append(("configure_trigger", self.exposure))

    def last_clamps(self):
        return {}

    def start_grab(self, N_img, output_queue=None, check_interrupt_method=None, on_armed=None):
        with self.lock:
            self.calls.append(("start_grab", self.exposure))
            if on_armed is not None:
                on_armed()
            while not check_interrupt_method():
                time.sleep(0.005)
            self.calls.append(("grab_end", self.exposure))

    def stop_grab(self):
        if self.lock.acquire(blocking=False):
            try:
                self.calls.append(("stop_grab", self.exposure))
            finally:
                self.lock.release()

    def names(self):
        return [c[0] for c in self.calls]


def make_baby(cam, nanny, name, exposure_time):
    from waxx.util.live_od.camera_mother import CameraBaby
    return CameraBaby(stand_in_handler(basler_params(exposure_time)), name, Queue(), nanny)


def shared_nanny(cam):
    from waxx.util.live_od.camera_nanny import CameraNanny
    nanny = CameraNanny()
    nanny.get_camera = lambda params: cam
    return nanny


def test_the_next_runs_settings_wait_for_the_old_threads_grab(app, records):
    cam = SharedCam()
    nanny = shared_nanny(cam)
    old, new = make_baby(cam, nanny, "Old", 1e-3), make_baby(cam, nanny, "New", 2e-3)
    try:
        old.start()
        assert wait_for(lambda: ("start_grab", 1e-3) in cam.calls)
        new.start()
        time.sleep(0.3)
        assert ("set_exposure", 2e-3) not in cam.calls        # waits: the old grab holds the camera
        assert any("waiting for another camera thread" in r for r in records)
        old.request_stop()                                    # superseded: its grab ends
        assert wait_for(lambda: ("start_grab", 2e-3) in cam.calls)
    finally:
        old.request_stop()
        new.request_stop()
        assert qt.join_or_keep(old) == "" and qt.join_or_keep(new) == ""
    seq = cam.calls
    assert seq.index(("grab_end", 1e-3)) < seq.index(("set_exposure", 2e-3))
    # the new run's settings, all of them, then its grab at its own exposure
    new_part = seq[seq.index(("set_exposure", 2e-3)):]
    assert [c[0] for c in new_part][:4] == ["set_exposure", "set_gain", "configure_trigger",
                                            "start_grab"]
    assert ("start_grab", 2e-3) in new_part
    assert cam.lock.acquire(blocking=False)                   # both threads let go
    cam.lock.release()


def test_a_thread_stopped_while_waiting_applies_nothing(app):
    cam = SharedCam()
    nanny = shared_nanny(cam)
    baby = make_baby(cam, nanny, "Superseded", 1e-3)
    failures = []
    baby.grab_failed_signal.connect(failures.append, Qt.ConnectionType.DirectConnection)
    cam.lock.acquire()                                        # someone else holds the camera
    try:
        baby.start()
        time.sleep(0.3)
        assert baby.isRunning() and cam.calls == []
        baby.request_stop()
        assert qt.join_or_keep(baby) == ""
    finally:
        cam.lock.release()
    assert cam.calls == []                                    # no setter, no grab
    assert failures == []


def test_a_thread_stopped_while_applying_does_not_grab_and_lets_go(app):
    cam = SharedCam()
    nanny = shared_nanny(cam)
    old, new = make_baby(cam, nanny, "Old", 1e-3), make_baby(cam, nanny, "New", 2e-3)
    cam.on_set_gain = old.request_stop                        # superseded mid-apply
    try:
        old.start()
        assert qt.join_or_keep(old) == ""
        assert "start_grab" not in cam.names()
        cam.on_set_gain = None
        new.start()
        assert wait_for(lambda: ("start_grab", 2e-3) in cam.calls)   # the lock was let go
    finally:
        new.request_stop()
        assert qt.join_or_keep(new) == ""
    assert cam.names().count("start_grab") == 1               # only the new run grabbed


# ----------------------------------------------------------------------
# which lock
# ----------------------------------------------------------------------

def test_camera_lock_is_the_drivers_reentrant_grab_lock_else_the_nannys():
    from waxx.control.cameras import DummyCamera
    from waxx.util.live_od.camera_nanny import CameraNanny
    nanny = CameraNanny()
    cam = SharedCam()
    assert nanny.camera_lock(cam, basler_params(1e-3)) is cam.lock

    class PlainLock(SharedCam):                               # not reentrant: start_grab would deadlock
        def grab_lock(self):
            return threading.Lock()

    class NoLock:
        def is_opened(self):
            return True

        def Close(self):
            pass
    a = nanny.camera_lock(PlainLock(), basler_params(1e-3))
    b = nanny.camera_lock(NoLock(), basler_params(1e-3))
    c = nanny.camera_lock(NoLock(), andor_params())
    assert a is b and a is not c                              # the nanny's own, per camera key
    assert a.acquire(blocking=False) and a.acquire(blocking=False)   # reentrant
    a.release()
    a.release()
    assert nanny.camera_lock(DummyCamera(), basler_params(1e-3)) is None
    assert nanny.close_all() == {}                            # the locks are not cameras
