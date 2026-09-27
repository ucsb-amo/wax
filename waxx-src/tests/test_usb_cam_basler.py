"""BaslerUSB (waxx) against a fake vendor pylon module (fakes/fake_pylon.py):
trigger re-assert, clamps (incl. the min-gain fix), on_armed, OneByOne, lost
frames, and the device lock."""
import importlib.util
import logging
import sys
import types
from pathlib import Path
from queue import Queue

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
import fake_pylon  # noqa: E402

from waxx.control.cameras import device_lock as dl  # noqa: E402
from waxx.control.cameras.device_lock import DeviceLock, DeviceBusy  # noqa: E402
from waxx.control.cameras.errors import FrameLostError  # noqa: E402

DRIVER = Path(__file__).resolve().parents[1] / "waxx" / "control" / "cameras" / "basler_usb.py"
VENDOR = "py" + "pylon"
TRIGGER_WRITES = [("LineSelector", "Line2"), ("LineMode", "Input"),
                  ("TriggerSelector", "FrameStart"), ("TriggerMode", "On"),
                  ("TriggerSource", "Line2")]


@pytest.fixture(autouse=True)
def release_leftover_locks():
    yield
    for lock in list(dl._HELD.values()):
        lock.release()


@pytest.fixture
def usb(monkeypatch, tmp_path):
    """The driver module, loaded under a private name against the fake pylon."""
    monkeypatch.setenv(dl.ENV_LOCK_DIR, str(tmp_path / "locks"))
    pkg = types.ModuleType(VENDOR)
    pkg.pylon = fake_pylon
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, VENDOR, pkg)
    monkeypatch.setitem(sys.modules, VENDOR + ".pylon", fake_pylon)
    fake_pylon.reset(serials=("40316451", "40320384"), gain_range=(1.0, 24.0))
    spec = importlib.util.spec_from_file_location("waxx_usb_cam_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "TIMEOUT_INIT", 0.5)
    monkeypatch.setattr(mod, "TIMEOUT_RUN", 0.5)
    monkeypatch.setattr(mod, "GRAB_POLL_INTERVAL", 0.01)
    return mod


@pytest.fixture
def cam(usb):
    c = usb.BaslerUSB(ExposureTime=100e-6, Gain=6.0, TriggerSource="Line2",
                      BaslerSerialNumber="40316451")
    yield c
    if c.IsOpen():
        c.Close()


def test_no_module_level_artiq_import():
    src = DRIVER.read_text(encoding="utf-8")
    assert "from artiq" not in src and "import artiq" not in src


# -- trigger -----------------------------------------------------------------------
def test_init_sets_the_trigger_as_before(cam):
    writes = cam._fake["writes"]
    assert writes[:2] == [("UserSetSelector", "Default"), ("UserSetLoad", "Execute")]
    assert writes[2:7] == TRIGGER_WRITES


def test_configure_trigger_reasserts_every_trigger_node(cam):
    cam.TriggerMode.SetValue("Off")           # changed since open (e.g. a viewer)
    cam.TriggerSource.SetValue("Software")
    start = len(cam._fake["writes"])
    cam.configure_trigger("Line2")
    assert cam._fake["writes"][start:] == TRIGGER_WRITES
    assert cam.TriggerMode.GetValue() == "On" and cam.TriggerSource.GetValue() == "Line2"


# -- clamps --------------------------------------------------------------------------
def test_in_range_values_are_not_clamps(cam):
    assert cam.last_clamps() == {}
    assert cam.ExposureTime.GetValue() == pytest.approx(100.0)
    assert cam.Gain.GetValue() == 6.0


def test_clamps_are_logged_and_remembered(cam, usb, caplog):
    with caplog.at_level(logging.WARNING, logger=usb.__name__):
        cam.set_exposure(10e-6)               # camera minimum 19 us
        cam.set_gain(0.0)                     # camera minimum 1 dB: was never clamped (B16)
    assert cam.last_clamps() == {"exposure_time": (10e-6, pytest.approx(19e-6)),
                                 "gain": (0.0, 1.0)}
    assert cam.ExposureTime.GetValue() == 19.0 and cam.Gain.GetValue() == 1.0
    assert "exposure_time 10 us is outside the camera's range; set to 19 us" in caplog.text
    assert "gain 0 dB is outside the camera's range; set to 1 dB" in caplog.text
    cam.set_gain(30.0)
    assert cam.last_clamps()["gain"] == (30.0, 24.0)
    cam.set_exposure(200e-6)                  # back in range: no longer a clamp
    cam.set_gain(12.0)
    assert cam.last_clamps() == {}


def test_last_clamps_is_a_copy(cam):
    cam.set_gain(0.0)
    cam.last_clamps().clear()
    assert "gain" in cam.last_clamps()


class _Rounding(fake_pylon._Node):
    """A node that keeps what it is sent rounded to ``step`` (as a camera does to
    its increment), or scaled by ``factor`` (float noise)."""

    def __init__(self, node, step=None, factor=1.0):
        super().__init__(node._cam, node._name, node._value, node._lo, node._hi)
        self._step, self._factor = step, factor

    def SetValue(self, value):
        super().SetValue(value)
        if self._step:
            self._value = round(value / self._step) * self._step
        self._value *= self._factor


def test_camera_rounding_beyond_tolerance_is_a_clamp(cam, usb, caplog):
    nodes = cam._fake["nodes"]
    nodes["ExposureTime"] = _Rounding(nodes["ExposureTime"], step=1.0)       # 1 us steps
    nodes["Gain"] = _Rounding(nodes["Gain"], step=0.1)
    with caplog.at_level(logging.WARNING, logger=usb.__name__):
        cam.set_exposure(19.4e-6)             # -> 19 us: 0.4 us off, tolerance 0.1% = 19 ns
        cam.set_gain(6.06)                    # -> 6.1 dB: 0.04 dB off, tolerance 0.01 dB
    assert cam.last_clamps() == {"exposure_time": (19.4e-6, pytest.approx(19e-6)),
                                 "gain": (6.06, pytest.approx(6.1))}
    assert "exposure_time 19.4 us was rounded by the camera; set to 19 us" in caplog.text
    assert "gain 6.06 dB was rounded by the camera; set to 6.1 dB" in caplog.text


def test_rounding_within_tolerance_is_not_a_clamp(cam):
    nodes = cam._fake["nodes"]
    nodes["ExposureTime"] = _Rounding(nodes["ExposureTime"], step=1.0)
    nodes["Gain"] = _Rounding(nodes["Gain"], step=0.01)
    cam.set_exposure(1000.4e-6)               # -> 1000 us: 0.4 us off, tolerance 1.0004 us
    cam.set_gain(6.004)                       # -> 6.00 dB: 0.004 dB off
    assert cam.last_clamps() == {}


def test_float_noise_in_the_readback_is_not_a_clamp(cam):
    nodes = cam._fake["nodes"]
    nodes["ExposureTime"] = _Rounding(nodes["ExposureTime"], factor=1 + 1e-12)
    nodes["Gain"] = _Rounding(nodes["Gain"], factor=1 + 1e-12)
    cam.set_exposure(100e-6)
    cam.set_gain(6.0)
    assert cam.last_clamps() == {}
    cam.set_exposure(5e-6)                    # out of range is still a clamp, read back
    assert cam.last_clamps()["exposure_time"] == (5e-6, pytest.approx(19e-6))


# -- grab ----------------------------------------------------------------------------
def _frames(n, h=4, w=6):
    return [("ok", np.full((h, w), i, dtype=np.uint8), 1000 + i) for i in range(n)]


def test_on_armed_is_called_once_after_grabbing_started(cam):
    armed = []

    def on_armed():
        armed.append(list(cam._fake["calls"]))
        cam._fake["results"].extend(_frames(3))

    q = Queue()
    cam.start_grab(3, q, on_armed=on_armed)
    assert len(armed) == 1
    calls = armed[0]
    assert calls[-2:] == ["StartGrabbingMax", "IsGrabbing"]      # armed, not yet waiting
    assert "RetrieveResult" not in calls
    assert cam._fake["strategy"] == fake_pylon.GrabStrategy_OneByOne
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert [idx for _, _, idx in items] == [0, 1, 2]
    assert [int(img[0, 0]) for img, _, _ in items] == [0, 1, 2]
    assert cam._fake["calls"][-1] == "StopGrabbing"


def test_on_armed_is_an_explicit_keyword(usb):
    # liveOD passes on_armed only when inspect.signature shows the parameter
    import inspect
    params = inspect.signature(usb.BaslerUSB.start_grab).parameters
    assert "on_armed" in params
    assert params["on_armed"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_without_output_queue_or_callback_still_grabs(cam):
    cam._fake["results"].extend(_frames(1))
    cam.start_grab(1)


def test_a_failed_grab_raises_and_nothing_shifts(cam):
    cam._fake["results"].extend(_frames(2) + [("fail", "Payload data has been discarded")]
                                + _frames(1))
    q = Queue()
    with pytest.raises(FrameLostError) as err:
        cam.start_grab(4, q)
    assert isinstance(err.value, TimeoutError)
    assert err.value.lost == (2,)
    assert "frame 2 of 4 lost: Payload data has been discarded" in str(err.value)
    assert [idx for _, _, idx in (q.get_nowait() for _ in range(q.qsize()))] == [0, 1]
    assert cam._fake["calls"][-1] == "StopGrabbing"


def _with_counters(monkeypatch, block_ids=None, image_numbers=None):
    """Grab results carry these BlockID / ImageNumber values, frame by frame
    (None: the result has no such counter)."""
    real = fake_pylon.InstantCamera.RetrieveResult
    values = iter(zip(block_ids or [None] * 100, image_numbers or [None] * 100))

    def retrieve(self, timeout_ms, handling):
        result = real(self, timeout_ms, handling)
        if result.IsValid():
            block, number = next(values)
            if block is not None:
                result.BlockID = block
            if number is not None:
                result.GetImageNumber = lambda n=number: n
        return result
    monkeypatch.setattr(fake_pylon.InstantCamera, "RetrieveResult", retrieve)


def _indices(q):
    return [idx for _, _, idx in (q.get_nowait() for _ in range(q.qsize()))]


def test_a_block_id_gap_is_a_lost_frame_and_nothing_shifts(cam, monkeypatch):
    _with_counters(monkeypatch, block_ids=[10, 11, 13, 14])     # the camera's frame 12 never came
    cam._fake["results"].extend(_frames(4))
    q = Queue()
    with pytest.raises(FrameLostError) as err:
        cam.start_grab(5, q)
    assert err.value.lost == (2,)
    assert "frame(s) [2] of 5 lost without a failed grab result (BlockID went from 11 to 13" in str(err.value)
    assert "queued as frame 3" in str(err.value)
    assert _indices(q) == [0, 1, 3]            # the frame after the gap in its own slot, slot 2 empty
    assert cam._fake["calls"][-1] == "StopGrabbing"


def test_an_image_number_gap_is_a_lost_frame(cam, monkeypatch):
    _with_counters(monkeypatch, image_numbers=[1, 2, 5])        # two frames skipped by the host
    cam._fake["results"].extend(_frames(3))
    q = Queue()
    with pytest.raises(FrameLostError) as err:
        cam.start_grab(4, q)
    assert err.value.lost == (2, 3)
    assert "ImageNumber went from 2 to 5" in str(err.value) and "beyond the run" in str(err.value)
    assert _indices(q) == [0, 1]               # its slot (4) is past N_img: not queued


def test_consecutive_counters_grab_everything(cam, monkeypatch):
    _with_counters(monkeypatch, block_ids=[7, 8, 9], image_numbers=[1, 2, 3])
    cam._fake["results"].extend(_frames(3))
    q = Queue()
    cam.start_grab(3, q)
    assert _indices(q) == [0, 1, 2]


def test_a_counter_that_does_not_count_is_turned_off_and_said(cam, usb, monkeypatch, caplog):
    _with_counters(monkeypatch, block_ids=[0, 0, 0])            # e.g. a device that reports none
    cam._fake["results"].extend(_frames(3))
    q = Queue()
    with caplog.at_level(logging.WARNING, logger=usb.__name__):
        cam.start_grab(3, q)
    assert _indices(q) == [0, 1, 2]
    assert caplog.text.count("BlockID went from 0 to 0; lost frames are not checked") == 1


def test_no_trigger_times_out(cam):
    with pytest.raises(TimeoutError, match="got 0/2"):
        cam.start_grab(2, Queue(), on_armed=lambda: None)


def test_not_grabbing_after_start_is_an_error_and_never_arms(cam, monkeypatch):
    monkeypatch.setattr(type(cam), "IsGrabbing", lambda self: False)
    armed = []
    with pytest.raises(RuntimeError, match="not grabbing; not arming"):
        cam.start_grab(1, Queue(), on_armed=lambda: armed.append(1))
    assert armed == []


# -- device lock ----------------------------------------------------------------------------
def test_second_open_of_the_same_camera_is_busy_until_close(cam, usb):
    with pytest.raises(DeviceBusy, match="Basler camera 40316451 is held by pid"):
        usb.BaslerUSB(BaslerSerialNumber="40316451")
    other = usb.BaslerUSB(BaslerSerialNumber="40320384")   # a different camera is free
    other.Close()
    cam.Close()
    again = usb.BaslerUSB(BaslerSerialNumber="40316451")
    again.Close()


def test_a_busy_camera_is_never_attached(usb, monkeypatch):
    holder = DeviceLock("basler:40316451").acquire()
    attached = []
    monkeypatch.setattr(fake_pylon.InstantCamera, "Attach",
                        lambda self, dev: attached.append(dev))
    with pytest.raises(DeviceBusy):
        usb.BaslerUSB(BaslerSerialNumber="40316451")
    assert attached == []
    holder.release()


def test_close_twice_on_an_old_object_keeps_the_new_ones_lock(cam, usb):
    cam.Close()
    newer = usb.BaslerUSB(BaslerSerialNumber="40316451")
    cam.Close()                               # a stale handle closed again
    with pytest.raises(DeviceBusy):
        DeviceLock("basler:40316451").acquire()
    newer.close()
    DeviceLock("basler:40316451").acquire().release()


def test_open_failure_releases_the_lock(usb, monkeypatch):
    def broken_open(self):
        raise fake_pylon.RuntimeException("device is in use")
    monkeypatch.setattr(fake_pylon.InstantCamera, "Open", broken_open)
    with pytest.raises(fake_pylon.RuntimeException):
        usb.BaslerUSB(BaslerSerialNumber="40316451")
    DeviceLock("basler:40316451").acquire().release()


def test_first_device_mode_locks_by_the_attached_serial(usb):
    c = usb.BaslerUSB(BaslerSerialNumber="")
    with pytest.raises(DeviceBusy, match="40316451"):
        DeviceLock("basler:40316451").acquire()
    c.Close()
