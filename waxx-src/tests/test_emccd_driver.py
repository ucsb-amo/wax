"""AndorEMCCD (waxx) driving the real pylablib AndorSDK2Camera against a fake
SDK2 library (fakes/fake_sdk2.py): acquisition-mode codes, amplifier mode,
Close order, the device lock, start_grab's on_armed / hardware index, and
apply_run_fields."""
import logging
import sys
from pathlib import Path
from queue import Queue

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
import fake_sdk2  # noqa: E402

from pylablib.devices import Andor  # noqa: E402
from beacon.camera.backend import ApplyRefused, ApplyMismatch, Readback  # noqa: E402

import waxx.control.cameras.andor as andor_mod  # noqa: E402
from waxx.control.cameras import device_lock as dl  # noqa: E402
from waxx.control.cameras.device_lock import DeviceLock, DeviceBusy  # noqa: E402
from waxx.control.cameras.errors import FrameLostError  # noqa: E402

FULL = (0, 512, 0, 512, 1, 1)
SHUTTER_MODE = {0: "auto", 1: "open", 2: "closed"}


@pytest.fixture(autouse=True)
def release_leftover_locks():
    # a failing test must not leave the process-wide lock held for the next
    yield
    for lock in list(dl._HELD.values()):
        lock.release()


@pytest.fixture
def make_sdk(monkeypatch, tmp_path):
    monkeypatch.setenv(dl.ENV_LOCK_DIR, str(tmp_path / "locks"))
    monkeypatch.setattr(andor_mod, "TIMEOUT", 2.0)

    def make(**kwargs):
        fake = fake_sdk2.FakeSDK2Lib(**kwargs)
        fake_sdk2.install(monkeypatch, fake)
        return fake
    return make


@pytest.fixture
def sdk(make_sdk):
    return make_sdk()


@pytest.fixture
def cam(sdk):
    c = andor_mod.AndorEMCCD()
    yield c
    if c.is_opened():
        c.close()


def shutter_modes(fake):
    """Internal shutter modes sent, in order."""
    return [SHUTTER_MODE[args[1]] for name, args in fake.log
            if name in ("SetShutter", "SetShutterEx")]


# -- init ------------------------------------------------------------------------
def test_init_leaves_the_camera_in_the_run_state(sdk):
    cam = andor_mod.AndorEMCCD(hs_speed=1, vs_speed=2, vs_amp=3, preamp=2, gain=30,
                               baseline_clamp=1)
    hw = sdk.hw
    assert hw["acq_mode"] == 5                   # continuous, not accumulate
    assert hw["trigger"] == 1 and hw["ft"] == 0 and hw["read_mode"] == 4
    assert hw["image"] == (1, 1, 1, 512, 1, 512)
    assert (hw["adc"], hw["oamp"], hw["hs"], hw["preamp"]) == (0, 0, 1, 2)
    assert (cam.get_channel(), cam.get_oamp(), cam.get_hsspeed(), cam.get_preamp()) == (0, 0, 1, 2)
    assert (hw["vs"], hw["vs_amp"]) == (2, 3)
    assert (hw["em_gain_mode"], hw["em_advanced"], hw["em_gain"]) == (3, 0, 30)
    assert hw["baseline_clamp"] == 1 and hw["cooler_mode"] == 1 and hw["camlink"] == 1
    assert SHUTTER_MODE[hw["shutter"][1]] == "open"
    assert SHUTTER_MODE[hw["shutter_ext"]] == "open"
    # one amplifier-mode write for hs_speed (B10): no bare SetHSSpeed outside set_amp_mode
    hs_calls = [i for i, (n, _) in enumerate(sdk.log) if n == "SetHSSpeed"]
    for i in hs_calls:
        assert sdk.log[i - 1][0] == "SetOutputAmplifier"
    assert "auto" not in shutter_modes(sdk)
    assert all(args[0] == 0 for _, args in sdk.calls("SetEMAdvanced"))
    cam.close()


def test_last_clamps_compare_the_request_with_the_camera(make_sdk):
    sdk = make_sdk(exposure_quantum=0.3e-6)
    cam = andor_mod.AndorEMCCD(ExposureTime=10e-6, gain=300)
    assert set(cam.last_clamps()) == {"exposure_time"}            # 10 us -> 10.2 us
    req, applied = cam.last_clamps()["exposure_time"]
    assert req == 10e-6 and applied == pytest.approx(10.2e-6, rel=1e-5)
    cam.set_exposure(9e-6)                                        # a whole number of quanta
    assert cam.last_clamps() == {}
    cam.set_EMCCD_gain(30.5, advanced=False)                      # the SDK takes an int
    assert cam.last_clamps() == {"gain": (30.5, 30.0)}
    cam.close()


def test_float32_rounding_is_not_a_clamp(cam, sdk):
    cam.set_exposure(10e-6)
    cam.set_EMCCD_gain(300, advanced=False)
    assert cam.last_clamps() == {}


# -- acquisition mode (B9) -----------------------------------------------------------
@pytest.mark.parametrize("alias, code", [("single", 1), ("accum", 2), ("kinetic", 3),
                                         ("fast_kinetic", 4), ("cont", 5)])
@pytest.mark.parametrize("setup_params", [True, False])
def test_set_acquisition_mode_sends_the_right_sdk_code(cam, sdk, alias, code, setup_params):
    start = len(sdk.log)
    assert cam.set_acquisition_mode(alias, setup_params=setup_params) == alias
    sent = [args[0] for name, args in sdk.log[start:] if name == "SetAcquisitionMode"]
    assert sent == [code]
    assert sdk.hw["acq_mode"] == code
    assert cam.get_acquisition_mode() == alias


def test_setup_params_reapplies_the_modes_last_parameters(cam, sdk):
    cam.setup_cont_mode(0.25)
    cam.set_acquisition_mode("single")
    start = len(sdk.log)
    cam.setup_acquisition("cont")
    assert ("SetAcquisitionMode", (5,)) in sdk.log[start:]
    assert ("SetKineticCycleTime", (0.25,)) in sdk.log[start:]
    assert sdk.hw["acq_mode"] == 5


def test_vendored_set_acquisition_mode_is_still_off_by_one(cam, sdk):
    """Pins the upstream bug the override exists for (AndorSDK2.py:710-718).
    If this starts failing, pylablib was fixed and the override can go."""
    vendored = Andor.AndorSDK2Camera.set_acquisition_mode
    start = len(sdk.log)
    vendored(cam, "single")
    sent = [args[0] for name, args in sdk.log[start:] if name == "SetAcquisitionMode"]
    assert sent == [2]                      # "single" ran the accum setup
    with pytest.raises(TypeError):
        vendored(cam, "fast_kinetic")       # unpacks the scalar cont cycle time


# -- amplifier mode (B10) ------------------------------------------------------------
def test_hs_speed_cache_matches_hardware_after_a_later_set_amp_mode(cam, sdk):
    cam.set_hsspeed(hs_speed=2)
    assert sdk.hw["hs"] == 2 == cam.get_hsspeed()
    cam.set_amp_mode(preamp=1)              # nanny's per-run call
    assert sdk.hw["hs"] == 2 == cam.get_hsspeed()
    assert sdk.hw["preamp"] == 1 == cam.get_preamp()


def test_unavailable_amp_mode_is_refused_before_anything_is_sent(make_sdk):
    sdk = make_sdk(unavailable_amp=[(0, 0, 3, 2)])
    cam = andor_mod.AndorEMCCD(hs_speed=0, preamp=2)
    start = len(sdk.log)
    with pytest.raises(andor_mod.AmpModeUnavailable, match="hs_speed 3, preamp 2"):
        cam.set_hsspeed(hs_speed=3)         # pylablib would switch the preamp silently
    with pytest.raises(andor_mod.AmpModeUnavailable):
        cam.set_hsspeed(hs_speed=9)         # pylablib would truncate to the slowest
    assert not [n for n, _ in sdk.log[start:] if n.startswith("Set")]
    assert sdk.hw["hs"] == 0 == cam.get_hsspeed()
    cam.close()


def test_init_with_an_unavailable_amp_mode_fails_cleanly(make_sdk):
    sdk = make_sdk(unavailable_amp=[(0, 0, 1, 2)])
    with pytest.raises(andor_mod.AmpModeUnavailable):
        andor_mod.AndorEMCCD(hs_speed=1, preamp=2)
    assert sdk.shutdown_count == 1 and not sdk.initialized
    DeviceLock(andor_mod.DEVICE_LOCK_KEY).acquire().release()   # lock released


# -- Close / Open (B11) -------------------------------------------------------------------
def test_close_stops_before_closing_the_shutter(cam, sdk):
    cam.set_trigger_mode("int")
    cam.start_acquisition(mode="cont")
    assert sdk.acquiring
    start = len(sdk.log)
    assert cam.Close() == []
    names = [n for n, _ in sdk.log[start:]]
    shutter_i = start + names.index("SetShutterEx")
    assert SHUTTER_MODE[sdk.log[shutter_i][1][1]] == "closed"
    assert sdk.index("AbortAcquisition", start) < shutter_i
    assert shutter_i < sdk.index("SetCoolerMode", start) < sdk.index("ShutDown", start)
    assert sdk.hw["cooler_mode"] == 1
    assert not cam.is_opened()
    DeviceLock(andor_mod.DEVICE_LOCK_KEY).acquire().release()


def test_shutdown_still_runs_when_the_shutter_command_fails(cam, sdk):
    sdk.fail_next("SetShutterEx")
    errors = cam.Close()
    assert len(errors) == 1 and "setup_shutter('closed')" in errors[0]
    assert sdk.shutdown_count == 1 and not cam.is_opened()
    DeviceLock(andor_mod.DEVICE_LOCK_KEY).acquire().release()


def test_close_safely_never_raises_and_reports_every_failure(cam, sdk):
    sdk.fail_next("SetShutterEx")
    sdk.fail_next("ShutDown")
    errors = cam.close_safely()
    assert len(errors) == 2
    assert "setup_shutter('closed')" in errors[0] and "ShutDown" in errors[1]
    assert not cam.is_opened()
    assert cam.close_safely() == []          # already closed
    DeviceLock(andor_mod.DEVICE_LOCK_KEY).acquire().release()


def test_reopening_in_place_is_refused(cam, sdk):
    cam.Open()                               # open: nothing to do
    assert cam.is_opened()
    cam.Close()
    n_init = len(sdk.calls("Initialize"))
    with pytest.raises(Andor.AndorError, match="(?i)reopening"):
        cam.Open()
    with pytest.raises(Andor.AndorError, match="(?i)reopening"):
        cam.open()
    assert len(sdk.calls("Initialize")) == n_init


# -- device lock ----------------------------------------------------------------------
def test_second_handle_is_busy_and_never_touches_the_sdk(cam, sdk):
    n_init = len(sdk.calls("Initialize"))
    with pytest.raises(DeviceBusy, match="Andor SDK is held by pid"):
        andor_mod.AndorEMCCD()
    assert len(sdk.calls("Initialize")) == n_init
    cam.Close()
    andor_mod.AndorEMCCD().Close()           # free again after Close


def test_lock_is_taken_before_the_sdk(sdk):
    holder = DeviceLock(andor_mod.DEVICE_LOCK_KEY).acquire()
    try:
        with pytest.raises(DeviceBusy):
            andor_mod.AndorEMCCD()
        assert sdk.calls("Initialize") == [] and sdk.calls("GetAvailableCameras") == []
    finally:
        holder.release()


def test_constructor_failure_releases_the_lock_and_the_sdk(sdk):
    sdk.fail_next("SetEMGainMode")
    with pytest.raises(Exception):
        andor_mod.AndorEMCCD()
    assert sdk.shutdown_count == 1
    DeviceLock(andor_mod.DEVICE_LOCK_KEY).acquire().release()


# -- start_grab (B4, B14) ------------------------------------------------------------------
def test_start_grab_arms_once_after_acquisition_is_running(cam, sdk):
    seen = []

    def on_armed():
        seen.append((sdk.acquiring, sdk.index("StartAcquisition") >= 0, len(sdk.log)))
        sdk.trigger(3)                       # the experiment's triggers come after arming

    q = Queue()
    frames = cam.start_grab(3, output_queue=q, on_armed=on_armed)
    assert len(seen) == 1
    acquiring, started, at = seen[0]
    assert acquiring and started
    assert sdk.index("StartAcquisition") < at
    assert "WaitForAcquisitionByHandleTimeOut" not in [n for n, _ in sdk.log[:at]][
        sdk.index("StartAcquisition"):]
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert [idx for _, _, idx in items] == [0, 1, 2]
    for img, t, idx in items:
        assert fake_sdk2.FakeSDK2Lib.decode(img) == (sdk.acq_gen, idx)
        assert isinstance(t, float)
    assert len(frames) == 3
    assert not sdk.acquiring                 # stopped after N_img


def test_a_lost_frame_raises_and_nothing_shifts(make_sdk):
    sdk = make_sdk(buffer_size=4)
    cam = andor_mod.AndorEMCCD()
    q = Queue()
    with pytest.raises(FrameLostError) as err:
        cam.start_grab(8, output_queue=q, on_armed=lambda: sdk.trigger(8))
    assert isinstance(err.value, TimeoutError)
    assert err.value.lost == (0, 1, 2, 3, 4)
    assert "[0, 1, 2, 3, 4]" in str(err.value)
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert [idx for _, _, idx in items] == [5, 6, 7]
    for img, _, idx in items:
        assert fake_sdk2.FakeSDK2Lib.decode(img)[1] == idx     # each frame in its own slot
    assert not sdk.acquiring
    cam.close()


def test_vendored_missing_frame_none_fails_on_a_lost_frame(make_sdk):
    """Why start_grab reads an explicit range with missing_frame="skip": pylablib
    prepends None for an overwritten frame, then _convert_frame_format calls
    frames[0].ndim on it; and with rng=None it drops the lost count entirely.
    If this starts failing, pylablib was fixed."""
    sdk = make_sdk(buffer_size=4)
    cam = andor_mod.AndorEMCCD()
    cam.set_trigger_mode("ext")
    cam.start_acquisition(mode="cont")
    sdk.trigger(8)
    frames, rng = cam.read_multiple_images(missing_frame="none", return_rng=True, peek=True)
    assert rng == (5, 8) and len(frames) == 3            # 0..4 lost without a trace
    assert all(f is not None for f in frames)
    with pytest.raises(AttributeError, match="ndim"):
        cam.read_multiple_images(rng=(0, None), missing_frame="none", return_rng=True)
    cam.close()


def test_no_trigger_times_out_with_the_builtin_timeout(cam, sdk, monkeypatch):
    monkeypatch.setattr(andor_mod, "TIMEOUT", 0.2)
    armed = []
    with pytest.raises(TimeoutError, match="got 0/2"):
        cam.start_grab(2, output_queue=Queue(), on_armed=lambda: armed.append(1))
    assert armed == [1] and not sdk.acquiring


def test_first_frame_extra_lengthens_only_the_first_wait(cam, sdk, monkeypatch):
    waits = []
    real = cam.wait_for_frame

    def wait(*args, timeout=None, **kwargs):
        waits.append(timeout)
        if len(waits) == 2:
            sdk.trigger(1)                   # the second shot's trigger
        return real(*args, timeout=timeout, **kwargs)
    monkeypatch.setattr(cam, "wait_for_frame", wait)
    q = Queue()
    cam.start_grab(2, output_queue=q, on_armed=lambda: sdk.trigger(1), first_frame_extra_s=90.)
    assert waits == [andor_mod.TIMEOUT + 90., andor_mod.TIMEOUT]
    assert [idx for _, _, idx in (q.get_nowait() for _ in range(q.qsize()))] == [0, 1]


def test_without_extra_every_wait_is_the_usual_timeout(cam, sdk, monkeypatch):
    waits = []
    real = cam.wait_for_frame

    def wait(*args, timeout=None, **kwargs):
        waits.append(timeout)
        return real(*args, timeout=timeout, **kwargs)
    monkeypatch.setattr(cam, "wait_for_frame", wait)
    cam.start_grab(1, output_queue=Queue(), on_armed=lambda: sdk.trigger(1))
    assert waits == [andor_mod.TIMEOUT]


def test_a_first_frame_timeout_with_extra_says_so(cam, sdk, monkeypatch):
    monkeypatch.setattr(andor_mod, "TIMEOUT", 0.2)
    with pytest.raises(TimeoutError, match="got 0/2") as err:
        cam.start_grab(2, output_queue=Queue(), on_armed=lambda: None, first_frame_extra_s=0.2)
    assert "s for warm-up shots)" in str(err.value) and not sdk.acquiring


def test_surplus_frames_are_not_queued(cam, sdk, caplog):
    q = Queue()
    with caplog.at_level(logging.WARNING, logger=andor_mod.__name__):
        cam.start_grab(2, output_queue=q, on_armed=lambda: sdk.trigger(3))
    assert [idx for _, _, idx in (q.get_nowait() for _ in range(q.qsize()))] == [0, 1]
    assert "1 frame(s) beyond the 2 expected" in caplog.text


def test_interrupt_returns_without_error(cam, sdk):
    q = Queue()
    stop = []
    cam.start_grab(3, output_queue=q, check_interrupt_method=lambda: bool(stop),
                   on_armed=lambda: stop.append(1))
    assert q.qsize() == 0 and not sdk.acquiring


# -- grab lock (run takeover) -------------------------------------------------------------------
def _grab_in_thread(cam, n, armed_event, result):
    import threading

    def run():
        try:
            result["frames"] = cam.start_grab(n, output_queue=Queue(),
                                              on_armed=armed_event.set)
        except Exception as e:           # pragma: no cover - reported by the test
            result["error"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_stop_grab_from_another_thread_does_not_stop_a_running_grab(cam, sdk):
    import threading
    armed, result = threading.Event(), {}
    t = _grab_in_thread(cam, 2, armed, result)          # the new run's grab
    assert armed.wait(5)
    cam.stop_grab()                                     # a superseded baby's death handler
    assert sdk.acquiring, "stop_grab stopped a grab another thread owns"
    sdk.trigger(2)
    t.join(5)
    assert not t.is_alive() and "error" not in result
    assert len(result["frames"]) == 2


def test_grabs_are_serialized(cam, sdk):
    import threading, time
    armed1, armed2, r1, r2 = threading.Event(), threading.Event(), {}, {}
    t1 = _grab_in_thread(cam, 1, armed1, r1)
    assert armed1.wait(5)
    t2 = _grab_in_thread(cam, 1, armed2, r2)
    time.sleep(0.2)
    assert not armed2.is_set()                          # waits for the first grab's lock
    sdk.trigger(1)
    t1.join(5)
    assert armed2.wait(5)
    sdk.trigger(1)
    t2.join(5)
    assert len(r1["frames"]) == 1 and len(r2["frames"]) == 1
    assert "error" not in r1 and "error" not in r2


def test_stop_grab_stops_when_no_grab_owns_the_camera(cam, sdk):
    cam.set_trigger_mode("int")
    cam.start_acquisition(mode="cont")
    cam.stop_grab()
    assert not sdk.acquiring
    cam.stop_grab()                                     # nothing to stop: no error


def test_on_armed_is_an_explicit_keyword():
    # liveOD passes on_armed only when inspect.signature shows the parameter
    import inspect
    params = inspect.signature(andor_mod.AndorEMCCD.start_grab).parameters
    assert "on_armed" in params
    assert params["on_armed"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# -- apply_run_fields (B8) --------------------------------------------------------------------
def test_apply_run_fields_restores_the_run_state(cam, sdk):
    cam.set_trigger_mode("int")
    cam.enable_frame_transfer_mode(True)
    cam.setup_image_mode(0, 256, 0, 256, 2, 2)
    cam.set_acquisition_mode("single")
    cam.setup_shutter("closed")
    cam.start_acquisition(mode="cont")
    rb = cam.apply_run_fields("ext", 0, FULL)
    hw = sdk.hw
    assert not sdk.acquiring
    assert hw["trigger"] == 1 and hw["ft"] == 0 and hw["acq_mode"] == 5
    assert hw["image"] == (1, 1, 1, 512, 1, 512)
    assert SHUTTER_MODE[hw["shutter"][1]] == "open"
    assert rb == {
        "trigger_mode": Readback("ext", "driver_cache"),
        "frame_transfer": Readback(0, "driver_cache"),
        "sensor_roi": Readback(FULL, "driver_cache"),
        "acq_mode": Readback("cont", "driver_cache"),
        "shutter": Readback("open", "commanded"),
    }


@pytest.mark.parametrize("kwargs, field", [
    (dict(trigger_mode="int"), "trigger_mode"),
    (dict(trigger_mode="ext_start"), "trigger_mode"),
    (dict(frame_transfer=1), "frame_transfer"),
    (dict(sensor_roi=(0, 256, 0, 256, 1, 1)), "sensor_roi"),
    (dict(sensor_roi=(0, 512, 0, 512, 2, 2)), "sensor_roi"),
])
def test_apply_run_fields_refuses_before_sending_anything(cam, sdk, kwargs, field):
    start = len(sdk.log)
    with pytest.raises(ApplyRefused) as err:
        cam.apply_run_fields(**kwargs)
    assert err.value.field == field
    assert repr(kwargs[field]) in str(err.value)
    assert sdk.log[start:] == []


def test_apply_run_fields_normalises_numpy_and_bytes(cam, sdk):
    import numpy as np
    rb = cam.apply_run_fields(b"ext", np.int64(0), np.array(FULL))
    assert rb["sensor_roi"].value == FULL


def test_a_truncated_image_area_is_a_mismatch(make_sdk):
    sdk = make_sdk(detector=(500, 500))
    cam = andor_mod.AndorEMCCD()
    with pytest.raises(ApplyMismatch) as err:
        cam.apply_run_fields()
    assert err.value.mismatches == {"sensor_roi": (FULL, (0, 500, 0, 500, 1, 1))}
    cam.close()


def test_a_camera_without_shutter_control_reports_it(make_sdk):
    sdk = make_sdk(has_shutter=False)
    cam = andor_mod.AndorEMCCD()
    assert cam.apply_run_fields()["shutter"] == Readback(None, "unsupported")
    assert sdk.calls("SetShutter") == [] and sdk.calls("SetShutterEx") == []
    assert cam.Close() == []


# -- shutter: the internal one is the one controlled -------------------------------------------
def test_the_internal_shutter_is_controlled_the_external_held_open(cam, sdk):
    # SHUTTEREX camera: SetShutterEx only (the manual forbids SetShutter there),
    # the mode on the internal shutter, extmode permanently open, min times
    # in the SDK's (closing, opening) order
    assert sdk.calls("SetShutter") == []
    typ, mode, closing, opening = sdk.hw["shutter"]
    assert (typ, SHUTTER_MODE[mode], closing, opening) == (0, "open", 27, 27)
    assert SHUTTER_MODE[sdk.hw["shutter_ext"]] == "open"
    cam.setup_shutter("closed")
    assert SHUTTER_MODE[sdk.hw["shutter"][1]] == "closed"
    assert SHUTTER_MODE[sdk.hw["shutter_ext"]] == "open"
    assert cam.shutter_readback() == Readback("closed", "commanded")
    assert sdk.calls("SetShutterEx")[-1][1] == (0, 2, 27, 27, 1)
    cam.Close()
    assert SHUTTER_MODE[sdk.hw["shutter"][1]] == "closed"
    assert SHUTTER_MODE[sdk.hw["shutter_ext"]] == "open"
    assert sdk.calls("SetShutter") == []


def test_without_independent_control_setshutter_drives_the_internal_shutter(make_sdk):
    sdk = make_sdk(has_shutter_ex=False)
    cam = andor_mod.AndorEMCCD()
    assert sdk.calls("SetShutterEx") == []
    assert SHUTTER_MODE[sdk.hw["shutter"][1]] == "open"
    assert cam.apply_run_fields()["shutter"] == Readback("open", "commanded")
    assert cam.Close() == []
    assert SHUTTER_MODE[sdk.hw["shutter"][1]] == "closed"
    assert sdk.calls("SetShutterEx") == []
