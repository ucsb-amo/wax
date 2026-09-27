"""EMCCDBackend (the Andor as a beacon CameraBackend) over the real AndorEMCCD
and pylablib, against the fake SDK2 library (fakes/fake_sdk2.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
import fake_sdk2  # noqa: E402

from beacon.camera.backend import (CameraBackend, CameraUnavailable, CameraError, ApplyRefused,
                                   ApplyMismatch, CloseReport, Readback, clamps)  # noqa: E402

import waxx.control.cameras.andor as andor_mod  # noqa: E402
from waxx.control.cameras import device_lock as dl  # noqa: E402
from waxx.control.cameras.device_lock import DeviceLock  # noqa: E402
from waxx.control.cameras.emccd_backend import EMCCDBackend  # noqa: E402

FULL = (0, 512, 0, 512, 1, 1)
SHUTTER_MODE = {0: "auto", 1: "open", 2: "closed"}
DU897_VS = ((("vs_speed", 0), ("vs_amp", 0)),
            "0.3 us vertical clock with Normal amplitude transfers no charge (run 80708)")

RUN = dict(exposure_time=10e-6, gain=300, trigger_mode="ext", frame_transfer=0, sensor_roi=FULL,
           hs_speed=0, preamp=2, vs_speed=1, vs_amp=3, baseline_clamp=1, shutter="open")
LIVE = dict(exposure_time=1e-3, gain=50, trigger_mode="int", frame_transfer=0, sensor_roi=FULL,
            hs_speed=1, preamp=1, vs_speed=2, vs_amp=1, baseline_clamp=0, shutter="closed")


@pytest.fixture(autouse=True)
def release_leftover_locks():
    yield
    for lock in list(dl._HELD.values()):
        lock.release()


@pytest.fixture
def make_backend(monkeypatch, tmp_path):
    monkeypatch.setenv(dl.ENV_LOCK_DIR, str(tmp_path / "locks"))
    made = []

    def make(sdk_kwargs=None, **backend_kwargs):
        fake = fake_sdk2.FakeSDK2Lib(**(sdk_kwargs or {}))
        fake_sdk2.install(monkeypatch, fake)
        backend_kwargs.setdefault("refuse_combos", (DU897_VS,))
        be = EMCCDBackend(**backend_kwargs)
        be.open()
        made.append(be)
        return be, fake
    yield make
    for be in made:
        be.close()


@pytest.fixture
def backend(make_backend):
    return make_backend()


def cam_of(be):
    return be._cam


def run_registers(fake, p):
    """The fake's hardware registers that a profile p determines."""
    hw = fake.hw
    return {
        "trigger": hw["trigger"], "ft": hw["ft"], "image": hw["image"], "acq_mode": hw["acq_mode"],
        "read_mode": hw["read_mode"], "fast_ext": hw["fast_ext"], "invert": hw["trigger_invert"],
        "term": hw["trigger_term"], "adc": hw["adc"], "oamp": hw["oamp"], "hs": hw["hs"],
        "preamp": hw["preamp"], "vs": hw["vs"], "vs_amp": hw["vs_amp"],
        "em_gain_mode": hw["em_gain_mode"], "em_advanced": hw["em_advanced"],
        "em_gain": hw["em_gain"], "exposure": hw["exposure"], "clamp": hw["baseline_clamp"],
        "shutter": SHUTTER_MODE[hw["shutter"][1]], "camlink": hw["camlink"],
        "cooler_mode": hw["cooler_mode"],
    }


def expected_registers(p):
    h0, h1, v0, v1, hb, vb = p["sensor_roi"]
    return {
        "trigger": {"int": 0, "ext": 1, "software": 10}[p["trigger_mode"]], "ft": 0,
        "image": (hb, vb, h0 + 1, h1, v0 + 1, v1), "acq_mode": 5, "read_mode": 4, "fast_ext": 0,
        "invert": 0, "term": 1, "adc": 0, "oamp": 0, "hs": p["hs_speed"], "preamp": p["preamp"],
        "vs": p["vs_speed"], "vs_amp": p["vs_amp"], "em_gain_mode": 3, "em_advanced": 0,
        "em_gain": p["gain"], "exposure": p["exposure_time"], "clamp": p["baseline_clamp"],
        "shutter": p["shutter"], "camlink": 1, "cooler_mode": 1,
    }


# -- open -------------------------------------------------------------------------
def test_open_names_the_camera_by_serial(backend):
    be, fake = backend
    assert isinstance(be, CameraBackend)
    assert be.category == "andor_emccd"
    assert be.camera_id == "andor_emccd:6321"
    info = be.open()                         # idempotent
    assert info["serial"] == "6321" and info["model"] == "DU897_BV"
    assert info["detector"] == (512, 512)
    assert fake.hw["em_gain"] == 0           # opened at EM gain 0 unless told otherwise


def test_open_when_another_process_holds_the_sdk(monkeypatch, tmp_path):
    monkeypatch.setenv(dl.ENV_LOCK_DIR, str(tmp_path / "locks"))
    fake = fake_sdk2.FakeSDK2Lib()
    fake_sdk2.install(monkeypatch, fake)
    holder = DeviceLock(andor_mod.DEVICE_LOCK_KEY).acquire()
    try:
        with pytest.raises(CameraUnavailable) as err:
            EMCCDBackend().open()
        assert err.value.holder["pid"]
        assert "Andor SDK is held by pid" in str(err.value)
        assert fake.calls("Initialize") == []
    finally:
        holder.release()


def test_open_refuses_an_unexpected_serial(monkeypatch, tmp_path):
    monkeypatch.setenv(dl.ENV_LOCK_DIR, str(tmp_path / "locks"))
    fake = fake_sdk2.FakeSDK2Lib(serial=1111)
    fake_sdk2.install(monkeypatch, fake)
    with pytest.raises(CameraError, match="expected Andor serial 6321, found 1111"):
        EMCCDBackend(serial=6321).open()
    assert fake.shutdown_count == 1          # closed again
    DeviceLock(andor_mod.DEVICE_LOCK_KEY).acquire().release()


def test_describe_dynamic(backend):
    be, _ = backend
    d = be.describe_dynamic()
    assert d["hs_speed"] == {"choices": [0, 1, 2, 3],
                             "labels": ["17 MHz", "10 MHz", "5 MHz", "1 MHz"]}
    assert d["vs_speed"]["labels"][:2] == ["0.3 µs", "0.5 µs"]
    assert d["vs_amp"]["labels"] == ["Normal", "+1", "+2", "+3", "+4"]
    assert d["preamp"]["labels"] == ["x1", "x2.4", "x5.1"]
    assert d["preamp"]["by_hs_speed"][0] == [0, 1, 2]
    assert d["gain"] == {"range": (0, 300), "live_cap": 100}
    assert d["trigger_mode"]["choices"] == ["int", "software", "ext"]
    assert d["exposure_time"]["range"][0] == 0.0


def test_read_settings_before_any_apply(backend):
    be, _ = backend
    rb = be.read_settings()
    assert rb["shutter"] == Readback("open", "commanded")        # AndorEMCCD opened it
    assert rb["trigger_mode"] == Readback("ext", "driver_cache")
    assert rb["sensor_roi"] == Readback(FULL, "driver_cache")


def test_a_camera_without_shutter_control(make_backend):
    be, fake = make_backend(sdk_kwargs=dict(has_shutter=False))
    assert be.apply(RUN, "run")["shutter"] == Readback(None, "unsupported")
    assert fake.calls("SetShutter") == []


# -- apply: a full state, in order --------------------------------------------------
def test_apply_is_a_full_state_not_a_change_on_top(backend):
    be, fake = backend
    be.apply(LIVE, "live")
    be.start_acquisition("live")
    fake.advance(0.1)
    cam = cam_of(be)
    # leftovers from elsewhere (e.g. the spot finder's own calls)
    cam.stop_acquisition()
    cam.enable_frame_transfer_mode(True)
    cam.setup_image_mode(0, 256, 0, 256, 2, 2)
    cam.set_fast_trigger_mode(1)
    cam.setup_ext_trigger(invert=True, term_highZ=False)
    cam.set_acquisition_mode("single")
    cam.start_acquisition(mode="cont")
    rb = be.apply(RUN, "run")
    assert run_registers(fake, RUN) == expected_registers(RUN)
    assert not fake.acquiring
    assert rb["trigger_mode"] == Readback("ext", "driver_cache")
    assert rb["sensor_roi"] == Readback(FULL, "driver_cache")
    assert rb["gain"] == Readback(300, "hw")
    assert rb["vs_amp"] == Readback(3, "commanded")
    assert rb["shutter"] == Readback("open", "commanded")
    assert rb["baseline_clamp"] == Readback(1, "hw")
    assert rb["exposure_time"].source == "hw"
    assert rb["cycle_time"].value > 0 and rb["readout_time"].value > 0
    assert rb["temperature"] == Readback(-60.0, "hw")


def test_apply_live_then_run_registers_follow_each(backend):
    be, fake = backend
    be.apply(LIVE, "live")
    assert run_registers(fake, LIVE) == expected_registers(LIVE)
    be.apply(RUN, "run")
    assert run_registers(fake, RUN) == expected_registers(RUN)


def test_apply_order(backend):
    be, fake = backend
    be.apply(LIVE, "live")
    be.start_acquisition("live")
    fake.advance(0.05)
    start = len(fake.log)
    be.apply(RUN, "run")
    be.start_acquisition("run", n_frames=3)
    log = fake.log[start:]
    names = [n for n, _ in log]
    first_set = next(i for i, n in enumerate(names) if n.startswith(("Set", "Free", "Prepare")))
    assert names.index("AbortAcquisition") < first_set            # stop first
    assert names.index("SetEMGainMode") < names.index("SetEMCCDGain")
    assert names.index("SetOutputAmplifier") < names.index("SetHSSpeed")
    assert names.index("SetAcquisitionMode") < names.index("SetFrameTransferMode")
    start_i = names.index("StartAcquisition")
    for getter in ("GetAcquisitionTimings", "GetReadOutTime", "GetKeepCleanTime",
                   "GetBaselineClamp", "GetEMCCDGain", "GetTemperatureF", "IsTriggerModeAvailable"):
        assert getter in names[:start_i], getter
        assert getter not in names[start_i:], getter
    assert all(args[0] == 0 for n, args in fake.log if n == "SetEMAdvanced")
    assert "auto" not in [SHUTTER_MODE[a[1]] for n, a in fake.log if n == "SetShutter"]


# -- apply: refusals send nothing ---------------------------------------------------------
def _refused(be, fake, values, purpose, field, words):
    start = len(fake.log)
    with pytest.raises(ApplyRefused) as err:
        be.apply(values, purpose)
    assert err.value.field == field
    assert words in str(err.value)
    assert fake.log[start:] == [], "a refusal must not touch the device"


@pytest.mark.parametrize("purpose, change, field, words", [
    ("run", dict(frame_transfer=1), "frame_transfer", "frame transfer is never used"),
    ("live", dict(frame_transfer=1), "frame_transfer", "frame transfer is never used"),
    ("run", dict(trigger_mode="int"), "trigger_mode", "runs accept only"),
    ("run", dict(trigger_mode="ext_start"), "trigger_mode", "ext_start free-runs"),
    ("live", dict(trigger_mode="ext"), "trigger_mode", "live streaming accepts only"),
    ("run", dict(sensor_roi=(0, 256, 0, 512, 1, 1)), "sensor_roi", "only the full frame"),
    ("live", dict(sensor_roi=(0, 512, 0, 512, 2, 2)), "sensor_roi", "binning 2x2"),
    ("run", dict(gain=301), "gain", "never above 300"),
    ("live", dict(gain=101), "gain", "above the live cap 100"),
    ("live", dict(gain=301, em_gain_unlocked=True), "gain", "never above 300"),
    ("run", dict(vs_speed=0, vs_amp=0), "vs_speed, vs_amp", "run 80708"),
    ("live", dict(vs_speed=0, vs_amp=0), "vs_speed, vs_amp", "transfers no charge"),
    ("live", dict(shutter="auto"), "shutter", "never 'auto'"),
    ("run", dict(shutter="closed"), "shutter", "shutter open"),
    ("run", dict(exposure_time=0.0), "exposure_time", "positive exposure"),
    ("run", dict(hs_speed=7), "hs_speed", "not an amplifier mode"),
    ("run", dict(vs_speed=9), "vs_speed", "out of range"),
    ("run", dict(baseline_clamp=2), "baseline_clamp", "not 0 or 1"),
    ("run", dict(em_advanced=1), "em_advanced", "pinned to 0"),
    ("run", dict(acquisition_mode="single"), "acquisition_mode", "pinned to 'cont'"),
    ("run", dict(color="red"), "color", "not an andor_emccd setting"),
])
def test_refusals(backend, purpose, change, field, words):
    be, fake = backend
    values = dict(RUN if purpose == "run" else LIVE, **change)
    _refused(be, fake, values, purpose, field, words)


def test_a_profile_missing_a_run_parameter_is_refused(backend):
    be, fake = backend
    values = dict(RUN)
    del values["vs_amp"]
    _refused(be, fake, values, "run", "vs_amp", "full profile")


def test_unlock_lifts_the_live_cap(backend):
    be, fake = backend
    be.apply(dict(LIVE, gain=250, em_gain_unlocked=True), "live")
    assert fake.hw["em_gain"] == 250


def test_status_keys_and_owner_settings(backend):
    be, fake = backend
    rb = be.apply(dict(RUN, temperature=-70.0, cooler_status="stabilized",
                       temperature_setpoint=-65, cooler=True, fan="low"), "run")
    assert fake.hw["temperature_setpoint"] == -65 and fake.hw["fan"] == 1
    assert rb["temperature_setpoint"].value == -65 and rb["fan"].value == "low"


# -- apply: mismatches ------------------------------------------------------------------------
def test_a_truncated_sensor_roi_is_a_mismatch(make_backend):
    be, fake = make_backend(sdk_kwargs=dict(detector=(500, 500)))
    with pytest.raises(ApplyMismatch) as err:
        be.apply(RUN, "run")
    assert err.value.mismatches["sensor_roi"] == (FULL, (0, 500, 0, 500, 1, 1))
    with pytest.raises(CameraError, match="last successful apply"):
        be.start_acquisition("run", n_frames=1)


def test_exposure_readback_tolerance(make_backend):
    be, fake = make_backend(sdk_kwargs=dict(exposure_scale=1.2))
    with pytest.raises(ApplyMismatch) as err:
        be.apply(RUN, "run")
    req, got = err.value.mismatches["exposure_time"]
    assert req == 10e-6 and got == pytest.approx(12e-6, rel=1e-5)


def test_an_em_gain_the_camera_raises_is_a_clamp_not_a_refusal(make_backend):
    """2026-09-27, kong: the host opened the Andor at live EM gain 1, the camera
    read back 4, and the open failed ("read-back differs"). Now it is recorded as
    a clamp, for live and run profiles alike."""
    be, fake = make_backend(sdk_kwargs=dict(em_gain_floor=4))
    rb = be.apply(dict(LIVE, gain=1), "live")
    assert rb["gain"] == Readback(4, "hw", origin="clamped", requested=1)
    assert clamps(rb)["gain"] == (1, 4)
    rb = be.apply(RUN, "run")                 # 300: taken as asked
    assert rb["gain"] == Readback(300, "hw")
    assert "gain" not in clamps(rb)


def test_a_small_exposure_quantisation_is_accepted_and_marked(make_backend):
    be, fake = make_backend(sdk_kwargs=dict(exposure_quantum=0.3e-6))
    rb = be.apply(RUN, "run")                 # 10 us -> 10.2 us: within 5 %
    assert rb["exposure_time"].origin == "clamped"
    assert rb["exposure_time"].requested == 10e-6
    assert clamps(rb) == {"exposure_time": (10e-6, rb["exposure_time"].value)}


def test_float32_rounding_is_not_a_clamp(backend):
    be, _ = backend
    rb = be.apply(RUN, "run")
    assert rb["exposure_time"].origin == ""
    assert clamps(rb) == {}


# -- acquisition ------------------------------------------------------------------------------
def test_start_needs_the_matching_apply(backend):
    be, _ = backend
    with pytest.raises(CameraError, match="apply a live profile first"):
        be.start_acquisition("live")
    be.apply(RUN, "run")
    with pytest.raises(CameraError, match="apply a live profile first"):
        be.start_acquisition("snap")
    with pytest.raises(ValueError, match="n_frames"):
        be.start_acquisition("run")


def test_run_acquisition_delivers_hardware_indices_and_stops_after_n(backend):
    be, fake = backend
    be.apply(RUN, "run")
    be.start_acquisition("run", n_frames=3)
    st = be.acquisition_state()
    assert st["acquiring"] and st["frames_done"] == 0 and st["cycle_s"] > 0
    assert be.retrieve(0.05) == []            # no trigger, no frame
    fake.trigger(3)
    frames = be.retrieve(0.5)
    assert [f.hw_idx for f in frames] == [0, 1, 2]
    for f in frames:
        assert fake_sdk2.FakeSDK2Lib.decode(f.image) == (fake.acq_gen, f.hw_idx)
        assert not f.image.flags.writeable and f.image.flags.owndata
        assert f.t_host > 0
    assert not be.acquisition_state()["acquiring"]     # stopped by itself after n_frames
    fake.trigger(2)
    assert be.retrieve(0.05) == []


def test_run_lost_frames_keep_their_index(make_backend):
    be, fake = make_backend(sdk_kwargs=dict(buffer_size=4))
    be.apply(RUN, "run")
    be.start_acquisition("run", n_frames=8)
    fake.trigger(8)
    frames = be.retrieve(0.5)
    assert [f.hw_idx for f in frames] == list(range(8))
    assert [f.image is None for f in frames] == [True] * 5 + [False] * 3
    for f in frames[5:]:
        assert fake_sdk2.FakeSDK2Lib.decode(f.image)[1] == f.hw_idx   # nothing shifted


def test_live_stream_runs_on_the_internal_trigger(backend):
    be, fake = backend
    rb = be.apply(LIVE, "live")
    be.start_acquisition("live")
    frames = be.retrieve(1.0)
    assert frames and frames[0].hw_idx == 0
    fake.advance(3 * rb["cycle_time"].value)
    more = be.retrieve(1.0)
    assert [f.hw_idx for f in more] == list(range(1, 1 + len(more)))
    assert be.acquisition_state()["frames_lost"] == 0
    be.stop_acquisition()
    assert not fake.acquiring


def test_snap_returns_one_fresh_frame_and_stops(backend):
    be, fake = backend
    be.apply(LIVE, "live")
    be.start_acquisition("snap")
    frames = be.retrieve(1.0)
    assert len(frames) == 1 and frames[0].hw_idx == 0
    assert fake_sdk2.FakeSDK2Lib.decode(frames[0].image)[0] == fake.acq_gen
    assert not fake.acquiring


def test_status_tolerates_drv_acquiring(backend):
    be, fake = backend
    fresh = be.status()
    assert fresh["temperature"] == -60.0 and fresh["stale"] == ()
    be.apply(LIVE, "live")
    be.start_acquisition("live")
    fake.temperature = -55.0
    during = be.status()                     # GetTemperatureF refuses while acquiring
    assert "temperature" in during["stale"]
    assert during["temperature"] == -60.0    # last known
    assert during["acquiring"] is True
    assert be.read_settings()["gain"] == Readback(50, "hw")    # last apply's readback
    be.stop_acquisition()
    assert be.status()["temperature"] == -55.0


# -- close ---------------------------------------------------------------------------------------
def test_close_never_raises_and_reports(make_backend):
    be, fake = make_backend()
    be.apply(LIVE, "live")
    be.start_acquisition("live")
    fake.fail_next("SetShutter")
    fake.fail_next("ShutDown")
    report = be.close()
    assert isinstance(report, CloseReport)
    assert report.attached_after is False
    assert any("setup_shutter" in e for e in report.errors)
    assert any("close" in e for e in report.errors)
    assert be.close() == CloseReport(attached_after=False, accessible_after=None, errors=())


def test_clean_close_stops_closes_the_shutter_and_frees_the_sdk(make_backend):
    be, fake = make_backend()
    be.apply(LIVE, "live")
    be.start_acquisition("live")
    start = len(fake.log)
    report = be.close()
    assert report == CloseReport(attached_after=False, accessible_after=True, errors=())
    names = [n for n, _ in fake.log[start:]]
    assert names.index("AbortAcquisition") < names.index("SetShutter") < names.index("ShutDown")
    assert fake.hw["cooler_mode"] == 1
    DeviceLock(andor_mod.DEVICE_LOCK_KEY).acquire().release()
