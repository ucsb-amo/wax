"""The camera thread (CameraBaby) and CameraNanny, with stand-in cameras: "ready"
is reported only once the driver says acquisition is running (B4), a baby has
its own stop (B5), the run's trigger and shutter are re-asserted and the
driver's clamps collected (B8), and close_all closes through the safe Close().
No camera, no file, no socket.
"""
import os
import threading
import time
import types
from queue import Queue

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import Qt

import liveod_qt_helpers as qt


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


# ----------------------------------------------------------------------
# stand-ins
# ----------------------------------------------------------------------

class ArmingCamera:
    """start_grab as the drivers have it since 2026-09-26: on_armed once the
    acquisition runs, before the first frame."""

    def __init__(self, events):
        self.events = events

    def is_opened(self):
        return True

    def start_grab(self, N_img, output_queue=None, check_interrupt_method=None, on_armed=None):
        self.events.append("start_grab")
        self.events.append("acquisition running")
        if on_armed is not None:
            on_armed()
        for i in range(N_img):
            output_queue.put((np.zeros((2, 2), np.uint16), 0.0, i))
            self.events.append(f"frame {i}")

    def stop_grab(self):
        self.events.append("stop_grab")


class OldCamera(ArmingCamera):
    """A driver from before on_armed."""

    def start_grab(self, N_img, output_queue=None, check_interrupt_method=None):
        super().start_grab(N_img, output_queue, check_interrupt_method)


class StandInNanny:
    def __init__(self, camera, clamps=None):
        self.camera, self.clamps = camera, clamps or {}
        self.interrupted = False
        self.break_checks = []

    def persistent_get_camera(self, camera_params, break_check=None):
        self.break_checks.append(break_check)
        return self.camera

    def update_params(self, camera, camera_params, report=None):
        if report is not None:
            report["clamps"] = dict(self.clamps)
        return camera


def stand_in_handler(n_img=3):
    discarded = []
    return types.SimpleNamespace(
        camera_params=types.SimpleNamespace(key="cam_a", camera_type="fake"),
        params=types.SimpleNamespace(N_img=n_img, N_shots=1, N_pwa_per_shot=1),
        read_params=lambda: None, grab_finished=lambda: None,
        writer=types.SimpleNamespace(discard=lambda delete=True: discarded.append(delete)),
        discarded=discarded)


def make_baby(camera, clamps=None, n_img=3):
    from waxx.util.live_od.camera_mother import CameraBaby
    events = camera.events
    handler = stand_in_handler(n_img)
    nanny = StandInNanny(camera, clamps)
    baby = CameraBaby(handler, "Tester", Queue(), nanny)
    baby.cam_status_signal.connect(lambda s: events.append(f"status {s}"),
                                   Qt.ConnectionType.DirectConnection)
    return baby, handler, nanny


# ----------------------------------------------------------------------
# B4: ready only once acquisition is running
# ----------------------------------------------------------------------

def test_status_2_is_sent_from_on_armed_after_acquisition_starts(app):
    events = []
    baby, handler, nanny = make_baby(ArmingCamera(events))
    baby.run()                                               # on this thread
    assert events.index("status 1") < events.index("start_grab")
    assert events.index("acquisition running") < events.index("status 2") < events.index("frame 0")
    assert events.count("status 2") == 1 and events.index("status 2") + 1 == events.index("status 3")
    assert events[-1] == "status -1" and handler.discarded == []
    assert nanny.break_checks == [baby.break_check]          # the baby's own stop, not the nanny's


def test_an_old_driver_still_gets_ready_but_it_is_said(app):
    import logging
    events, warned = [], []

    class H(logging.Handler):
        def emit(self, record):
            warned.append(record.getMessage())
    h = H(logging.WARNING)
    logging.getLogger("waxx.live_od").addHandler(h)
    try:
        baby, _, _ = make_baby(OldCamera(events))
        baby.run()
    finally:
        logging.getLogger("waxx.live_od").removeHandler(h)
    assert events.index("status 2") < events.index("start_grab")      # as before
    assert any("no on_armed" in w for w in warned)


def test_on_armed_called_twice_reports_ready_once(app):
    events = []

    class Twice(ArmingCamera):
        def start_grab(self, N_img, output_queue=None, check_interrupt_method=None, on_armed=None):
            on_armed()
            on_armed()
    baby, _, _ = make_baby(Twice(events))
    baby.run()
    assert events.count("status 2") == 1


def test_clamps_leave_the_camera_thread_once(app):
    events, got = [], []
    baby, _, _ = make_baby(ArmingCamera(events), clamps={"exposure_time": (1.9e-05, 2.1e-05)})
    baby.camera_overrides_signal.connect(lambda key, clamps: got.append((key, clamps)),
                                         Qt.ConnectionType.DirectConnection)
    baby.run()
    assert got == [("cam_a", {"exposure_time": (1.9e-05, 2.1e-05)})]


# ----------------------------------------------------------------------
# B5: each baby has its own stop
# ----------------------------------------------------------------------

def test_a_stopped_baby_leaves_the_camera_wait_quietly(app, monkeypatch):
    from waxx.control.cameras import DummyCamera
    from waxx.util.live_od import camera_nanny as nanny_module
    from waxx.util.live_od.camera_mother import CameraBaby
    monkeypatch.setattr(nanny_module, "CHECK_PERIOD", 0.2)
    monkeypatch.setattr(nanny_module, "CHECK_EVERY", 0.02)
    nanny = nanny_module.CameraNanny()
    nanny.get_camera = lambda params: DummyCamera()           # the camera is never reachable
    handler = stand_in_handler()
    baby = CameraBaby(handler, "Waiter", Queue(), nanny)
    baby.start()
    try:
        time.sleep(0.3)
        assert baby.isRunning()
        nanny.interrupted = False                              # what spawn_baby does for a new run
    finally:
        baby.request_stop()
        stopped = qt.join_or_keep(baby)                        # never left running, whatever failed
    assert stopped == ""
    assert handler.discarded == []                             # its run may be saved: nothing discarded


def test_the_nanny_flag_still_stops_callers_without_their_own_check(app, monkeypatch):
    from waxx.control.cameras import DummyCamera
    from waxx.util.live_od import camera_nanny as nanny_module
    monkeypatch.setattr(nanny_module, "CHECK_PERIOD", 0.1)
    monkeypatch.setattr(nanny_module, "CHECK_EVERY", 0.01)
    nanny = nanny_module.CameraNanny()
    nanny.get_camera = lambda params: DummyCamera()
    timer = threading.Timer(0.15, lambda: setattr(nanny, "interrupted", True))
    timer.start()
    try:
        t0 = time.time()
        cam = nanny.persistent_get_camera(types.SimpleNamespace(key="cam_a"))
        assert isinstance(cam, DummyCamera) and time.time() - t0 < 2.0
    finally:
        nanny.interrupted = True
        assert qt.join_or_keep(timer) == ""


# ----------------------------------------------------------------------
# B8: the run's trigger (and the Andor's shutter) are re-asserted every run
# ----------------------------------------------------------------------

class Recorder:
    """Records every driver call. ``absent``: methods this driver does not have."""
    absent = ("close_safely",)

    def __init__(self):
        self.calls = []
        self.opened = True

    def __getattr__(self, name):
        if name.startswith("_") or name in type(self).absent:
            raise AttributeError(name)

        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return call

    def is_opened(self):
        return self.opened

    def Close(self):
        self.calls.append(("Close", (), {}))
        self.opened = False


class FakeBasler(Recorder):
    """BaslerUSB's surface: configure_trigger, last_clamps; no close_safely."""

    def IsGrabbing(self):
        return False

    def last_clamps(self):
        return {"exposure_time": (1.9e-05, 2.1e-05)}


class FakeAndor(Recorder):
    """AndorEMCCD's surface: apply_run_fields, set_amp_mode_checked, close_safely."""
    absent = ()

    def close_safely(self):
        self.calls.append(("close_safely", (), {}))
        self.opened = False
        return []


class OldBasler(Recorder):
    """No configure_trigger, no last_clamps."""
    absent = ("close_safely", "configure_trigger", "last_clamps")

    def IsGrabbing(self):
        return False


class OldAndor(Recorder):
    absent = ("close_safely", "apply_run_fields", "set_amp_mode_checked")


def basler_params(**kw):
    p = types.SimpleNamespace(key="xy_basler", camera_type="basler", exposure_time=19e-6,
                              gain=6.0, trigger_source="Line2")
    vars(p).update(kw)
    return p


def andor_params(**kw):
    p = types.SimpleNamespace(key="andor", camera_type="andor", exposure_time=10e-6, gain=300.,
                              preamp=2, hs_speed=0, vs_speed=1, vs_amp=3, baseline_clamp=1)
    vars(p).update(kw)
    return p


def names(camera):
    return [c[0] for c in camera.calls]


def test_basler_trigger_is_reasserted_and_clamps_collected():
    from waxx.util.live_od.camera_nanny import CameraNanny
    cam, report = FakeBasler(), {}
    out = CameraNanny().update_params(cam, basler_params(), report=report)
    assert out is cam
    assert ("configure_trigger", ("Line2",), {}) in cam.calls
    assert names(cam).index("set_gain") < names(cam).index("configure_trigger")
    assert report["clamps"] == {"exposure_time": (1.9e-05, 2.1e-05)}


def test_andor_run_fields_are_applied_every_run():
    from waxx.util.live_od.camera_nanny import CameraNanny
    cam = FakeAndor()
    params = andor_params(trigger_mode=b"ext", frame_transfer=0, sensor_roi=(0, 512, 0, 512, 1, 1))
    assert CameraNanny().update_params(cam, params, report={}) is cam
    assert ("apply_run_fields", ("ext", 0, (0, 512, 0, 512, 1, 1)), {}) in cam.calls


def test_andor_amplifier_mode_is_one_checked_call():
    """hs_speed and preamp together (DRV): as two calls pylablib could swap in a
    preamp the old hs_speed offers."""
    from waxx.util.live_od.camera_nanny import CameraNanny
    cam = FakeAndor()
    CameraNanny().update_params(cam, andor_params(hs_speed=1, preamp=2))
    assert ("set_amp_mode_checked", (), {"channel": 0, "oamp": 0, "hsspeed": 1, "preamp": 2}) in cam.calls
    assert "set_amp_mode" not in names(cam) and "set_hsspeed" not in names(cam)


def test_an_unavailable_amplifier_mode_means_no_camera_for_the_run():
    from waxx.control.cameras import DummyCamera
    from waxx.util.live_od.camera_nanny import CameraNanny

    class NoSuchMode(FakeAndor):
        def set_amp_mode_checked(self, **kw):
            raise ValueError("amplifier mode (channel 0, output amp 0, hs_speed 3, preamp 2) "
                             "is not available on this camera; nothing was sent.")
    assert isinstance(CameraNanny().update_params(NoSuchMode(), andor_params(hs_speed=3)), DummyCamera)


def test_an_old_andor_driver_gets_the_two_calls_as_before():
    from waxx.util.live_od.camera_nanny import CameraNanny
    cam = OldAndor()
    CameraNanny().update_params(cam, andor_params(hs_speed=1, preamp=2))
    assert ("set_amp_mode", (), {"preamp": 2}) in cam.calls
    assert ("set_hsspeed", (), {"hs_speed": 1}) in cam.calls


def test_an_old_payload_means_what_liveod_always_did():
    from waxx.util.live_od.camera_nanny import CameraNanny
    cam = FakeAndor()
    CameraNanny().update_params(cam, andor_params())           # no run fields in the payload
    assert ("apply_run_fields", ("ext", 0, (0, 512, 0, 512, 1, 1)), {}) in cam.calls


def test_a_refused_run_field_means_no_camera_for_the_run():
    from waxx.control.cameras import DummyCamera
    from waxx.util.live_od.camera_nanny import CameraNanny

    class Refusing(FakeAndor):
        def apply_run_fields(self, trigger_mode, frame_transfer, sensor_roi):
            raise ValueError(f"frame_transfer={frame_transfer} refused for runs (D-g: no frame transfer)")
    out = CameraNanny().update_params(Refusing(), andor_params(frame_transfer=1))
    assert isinstance(out, DummyCamera)


def test_an_old_driver_still_works_and_is_warned_about_once():
    import logging
    from waxx.util.live_od.camera_nanny import CameraNanny
    warned = []

    class H(logging.Handler):
        def emit(self, record):
            warned.append(record.getMessage())
    h = H(logging.WARNING)
    logging.getLogger("waxx.live_od").addHandler(h)
    try:
        nanny = CameraNanny()
        for _ in range(2):
            basler, andor, report = OldBasler(), OldAndor(), {}
            assert nanny.update_params(basler, basler_params(), report=report) is basler
            assert report["clamps"] == {}
            assert nanny.update_params(andor, andor_params()) is andor
    finally:
        logging.getLogger("waxx.live_od").removeHandler(h)
    assert sum("configure_trigger()" in w for w in warned) == 1
    assert sum("apply_run_fields()" in w for w in warned) == 1
    assert sum("set_amp_mode_checked()" in w for w in warned) == 1


# ----------------------------------------------------------------------
# close_all (shutdown)
# ----------------------------------------------------------------------

def test_close_all_closes_safely_and_forgets_the_camera():
    from waxx.util.live_od.camera_nanny import CameraNanny
    nanny = CameraNanny()
    nanny.andor, nanny.xy_basler = FakeAndor(), FakeBasler()
    andor, basler = nanny.andor, nanny.xy_basler

    class Stuck(FakeBasler):
        def Close(self):
            raise RuntimeError("device busy")
    nanny.z_basler = Stuck()
    results = nanny.close_all()
    assert results["andor"] == "" and results["xy_basler"] == ""
    assert "device busy" in results["z_basler"]
    assert names(andor) == ["stop_grab", "close_safely"]       # the grab stopped first; never Close()
    assert names(basler) == ["stop_grab", "Close"]              # no close_safely: Close()
    assert not hasattr(nanny, "andor") and not hasattr(nanny, "xy_basler")
    assert hasattr(nanny, "z_basler")                           # still open: kept
    assert "andor" not in nanny.close_all()                     # closed once only
    assert nanny.interrupted is False                           # settings are not cameras


class NodeNotExisting(Exception):
    """Stands in for pypylon's genicam LogicalErrorException: NOT an AttributeError."""


class PylonLikeBasler(FakeBasler):
    """As pypylon's InstantCamera (BaslerUSB's base): a name the driver lacks is
    looked up as a GenICam node, and a missing node raises NodeNotExisting, which
    getattr(obj, name, None) and hasattr() let through."""

    def __getattr__(self, name):
        if name in type(self).absent:
            raise NodeNotExisting(f"Node not existing: {name}")
        return super().__getattr__(name)


def test_close_all_closes_a_pylon_like_basler():
    """2026-09-27, liveOD shutdown: getattr(obj, "close_safely", None) raised
    "Node not existing" on the real x_basler, so Close() never ran."""
    from waxx.util.live_od.camera_nanny import CameraNanny
    nanny = CameraNanny()
    nanny.x_basler = cam = PylonLikeBasler()
    results = nanny.close_all()
    assert results["x_basler"] == ""
    assert names(cam) == ["stop_grab", "Close"]
    assert not hasattr(nanny, "x_basler")


def test_run_settings_reach_a_pylon_like_basler_without_optional_methods():
    from waxx.util.live_od.camera_nanny import CameraNanny

    class OldPylonBasler(PylonLikeBasler):
        absent = ("close_safely", "configure_trigger", "last_clamps")
    cam, report = OldPylonBasler(), {}
    assert CameraNanny().update_params(cam, basler_params(), report=report) is cam
    assert "error" not in report
    assert "set_gain" in names(cam) and "configure_trigger" not in names(cam)


def test_close_all_reports_step_errors_and_keeps_a_camera_still_open():
    from waxx.util.live_od.camera_nanny import CameraNanny

    class ShutterFailed(FakeAndor):
        def close_safely(self):
            self.opened = False
            return ["setup_shutter('closed'): DRV_ACQUIRING"]

    class SdkCloseFailed(FakeAndor):
        def close_safely(self):
            return ["close(): DRV_ERROR_ACK"]                   # still open afterwards
    nanny = CameraNanny()
    nanny.andor, nanny.andor_b = ShutterFailed(), SdkCloseFailed()
    results = nanny.close_all()
    assert results["andor"] == "setup_shutter('closed'): DRV_ACQUIRING"
    assert results["andor_b"] == "close(): DRV_ERROR_ACK"
    assert not hasattr(nanny, "andor")                          # closed, with a step error
    assert hasattr(nanny, "andor_b")                            # not closed: kept
