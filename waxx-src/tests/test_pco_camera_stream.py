"""A PCO sCMOS as an auxiliary camera: its camera-table entry (PCOParams) and
camera streams on it.

Offline.  The unit tests use the in-memory fakes of test_camera_stream_client.
The end-to-end ones run the real stream clients over the real beacon
CameraStream against a CameraServerCore on 127.0.0.1 whose camera is beacon's
PcoBackend on its fake camera (beacon's tests/fake_pco.py): no SDK, no
hardware, no beacon broadcast (conftest).  They are skipped with a beacon that
has no PCO backend.
"""
import json
import os
import sys
import time

import numpy as np
import pytest

from waxx.control.cameras.camera_param_classes import PCOParams, RunFieldRefused
from waxx.control.cameras.camera_stream_client import (
    CameraStreamClient, TriggeredCameraStreamClient, _differs)

from test_camera_stream_client import F, M, FakeExpt, FakeStream, FakeTTL, wait_drained

S = "61000123"
ROI_4x4 = (0, 2560, 0, 2160, 4, 4)


# ---- the camera-table entry ------------------------------------------------

def test_pco_params_fields():
    p = PCOParams(serial_number=S, exposure_time=2e-3, trigger_ttl="pco_trigger")
    assert p.camera_type == "pco" and p.serial_no == S
    assert p.pixel_format == "Mono16" and p.trigger_source == "ExposureTrigger"
    assert p.sensor_roi == ROI_4x4 and p.resolution == (540, 640)     # (rows, cols)
    assert p.exposure_time == 2e-3 and p.gain == 0.0 and p.readout_rate == "slow"
    assert p.trigger_ttl == "pco_trigger" and p.trigger_ttl_ch == -1
    assert p.stream_settings() == {"sensor_roi": ROI_4x4, "readout_rate": "slow"}
    assert PCOParams(sensor_roi=[640, 1920, 540, 1620, 2, 2]).resolution == (540, 640)


def test_pco_params_refuses_bad_fields_and_being_the_run_camera():
    with pytest.raises(RunFieldRefused, match="PCOParams.readout_rate"):
        PCOParams(readout_rate="medium")
    with pytest.raises(RunFieldRefused, match="PCOParams.sensor_roi"):
        PCOParams(sensor_roi=(0, 2560, 0, 2160))
    with pytest.raises(RunFieldRefused, match="PCOParams.sensor_roi"):
        PCOParams(sensor_roi=(0, 2560, 0, 2160, 0, 1))
    p = PCOParams(serial_number=S, key="pco_edge")
    with pytest.raises(RunFieldRefused, match="cannot be a run's acquisition camera"):
        p.prepare_for_run()


def test_pco_image_dtype():
    pytest.importorskip("beacon.camera.pco_backend")
    assert PCOParams().image_dtype() == np.uint16


# ---- the stream client, on fakes -------------------------------------------

def test_differs_compares_a_sensor_roi_by_value():
    assert not _differs([0, 2560, 0, 2160, 4, 4], ROI_4x4)
    assert _differs([0, 2560, 0, 2160, 2, 2], ROI_4x4)
    assert _differs(None, ROI_4x4) and _differs("x", ROI_4x4)


def test_settings_are_applied_with_the_exposure_and_put_back():
    fe = FakeExpt()
    before = [0, 2560, 0, 2160, 2, 2]
    stream = FakeStream(settings={"exposure_time": 5e-3, "gain": 0.0, "trigger_mode": "Off",
                                  "sensor_roi": list(before), "readout_rate": "slow"})
    cs = CameraStreamClient(fe, S, "img_test", _stream=stream, _server_defaults={},
                            exposure_time=1e-3, gain=0.0, crop_to_saved_roi=False,
                            settings={"sensor_roi": ROI_4x4, "readout_rate": "slow"})
    try:
        (sent,) = stream.set_calls                 # one write; what already holds is not sent
        assert sent == {"exposure_time": 1e-3, "sensor_roi": ROI_4x4}
    finally:
        cs.finish()
    assert stream.set_calls[-1] == {"exposure_time": 5e-3, "sensor_roi": before}
    rec = json.loads(fe._extra_file_texts["camera_stream_img_test"])
    assert rec["settings_asked"] == {"sensor_roi": list(ROI_4x4), "readout_rate": "slow"}


def test_a_setting_with_its_own_argument_is_refused_in_settings():
    with pytest.raises(ValueError, match="have their own"):
        CameraStreamClient(FakeExpt(), S, "img_test", _stream=FakeStream(),
                           _server_defaults={}, settings={"gain": 3.0})


def test_no_saved_defaults_are_read_when_told_there_are_none(monkeypatch):
    stream = FakeStream()

    def no_store(self):
        raise AssertionError("the Basler defaults store was asked")
    monkeypatch.setattr(CameraStreamClient, "_resolve_and_connect", lambda self, d: stream)
    monkeypatch.setattr(CameraStreamClient, "_read_server_defaults", no_store)
    cs = CameraStreamClient(FakeExpt(), S, "img_test", exposure_time=1e-3,
                            read_saved_defaults=False)
    try:
        assert cs._defaults == {} and cs._roi is None and cs._frame_shape == (10, 12)
    finally:
        cs.finish()


# ---- end to end: the real clients, CameraStream, core and PcoBackend --------

def _fake_pco():
    pytest.importorskip("beacon.camera.pco_backend")
    import beacon
    tests = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(beacon.__file__))),
                         "tests")
    if not os.path.exists(os.path.join(tests, "fake_pco.py")):
        pytest.skip("beacon's tests/fake_pco.py is not beside the beacon package")
    if tests not in sys.path:
        sys.path.append(tests)
    import fake_pco
    return fake_pco


class _NoOtherServer:
    def find(self, query, refresh=False):
        from beacon.camera.directory import CameraNotFound
        raise CameraNotFound(f"{query}: no other server (test)")

    def close(self):
        pass


@pytest.fixture
def pco_served(monkeypatch):
    fake_pco = _fake_pco()
    from beacon.camera.core import CameraServerCore
    from beacon.camera.directory import CameraEntry
    from beacon.camera.pco_backend import PcoBackend
    from beacon.camera.stream import CameraStream
    monkeypatch.setattr(CameraServerCore, "DUPLICATE_WINDOW_S", 0.0)

    cam = fake_pco.FakePCOCamera(S, interface="cl")
    cid = f"pco_sc2:{S}"
    core = CameraServerCore("camera_server:pcotest", policy="on_demand",
                            bind_host="127.0.0.1", check_duplicate=False)
    core.add_camera(cid, lambda: PcoBackend(S, camera_factory=lambda: cam),
                    category="pco_sc2", info={"model": "pco.edge 5.5", "serial": S})
    core.start_in_thread()
    entry = CameraEntry(camera_id=cid, category="pco_sc2", serial=S, name="",
                        model="pco.edge 5.5", server_id=core.server_id, host="127.0.0.1",
                        port=core.port, state="closed", holder=None, protocol="v2",
                        server_ids=core.server_ids)
    stream = CameraStream(entry, label="test", directory=_NoOtherServer())
    stream.get_settings()                               # attaches (opens the camera)
    # free run: the fake exposes a frame whenever the server is waiting for one
    ticking = {"on": True}

    def tick():
        while ticking["on"]:
            try:
                if cam.acquisition_in_progress() and cam.get_trigger_mode() == "int":
                    cam.tick(1)
            except Exception:
                pass
            time.sleep(0.02)
    import threading
    t = threading.Thread(target=tick, daemon=True)
    t.start()
    yield cam, core, stream, cid
    ticking["on"] = False
    t.join(1)
    try:
        stream.close()
    except Exception:
        pass
    core.stop(10)


def _entry_kw(p):
    return dict(pixel_format=p.pixel_format, settings=p.stream_settings(),
                read_saved_defaults=False, exposure_time=p.exposure_time, gain=p.gain)


def test_free_run_stream_on_a_pco_end_to_end(pco_served):
    cam, core, stream, cid = pco_served
    p = PCOParams(serial_number=S, exposure_time=2e-3)
    fe = FakeExpt()
    cs = CameraStreamClient(fe, S, "img_pco", _stream=stream, _server_defaults={},
                            crop_to_saved_roi=False, t_late_tolerance=2.0, **_entry_kw(p))
    try:
        assert cs._frame_shape == (540, 640) and cs._frame_dtype == np.uint16
        got = dict(core.worker(cid).settings)
        assert tuple(got["sensor_roi"]) == ROI_4x4 and got["exposure_time"] == 2e-3
        fe.data.init()
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        assert cs.counts["ok"] == 1 and cs.counts["failed"] == 0, (cs.counts, cs.events)
        img = F(fe, "img_pco")[0]
        assert img.shape == (540, 640) and img.dtype == np.uint16
        # the fake camera's frames are zeros but a frame counter: the record says a frame came
        assert not np.isnan(M(fe, "img_pco", "seq")[0])
        assert M(fe, "img_pco", "exposure")[0] == pytest.approx(2e-3)
    finally:
        cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_pco"])
    assert rec["pixel_format"] == "Mono16" and rec["frame_dtype"] == "uint16"
    assert rec["settings_applied"]["sensor_roi"] == list(ROI_4x4)


def test_a_frame_wider_than_the_camera_link_cap_fails_the_setup(pco_served):
    cam, core, stream, cid = pco_served
    with pytest.raises(RuntimeError, match="640 px cap"):
        CameraStreamClient(FakeExpt(), S, "img_pco", _stream=stream, _server_defaults={},
                           crop_to_saved_roi=False, pixel_format="Mono16",
                           read_saved_defaults=False, exposure_time=1e-3, gain=0.0,
                           settings={"sensor_roi": (0, 2560, 0, 2160, 1, 1)})


def test_triggered_stream_on_a_pco_end_to_end(pco_served):
    cam, core, stream, cid = pco_served
    p = PCOParams(serial_number=S, exposure_time=1e-3)
    fe = FakeExpt()
    cs = TriggeredCameraStreamClient(
        fe, S, ["img_a", "img_b"], ttl=FakeTTL(), trigger_source=p.trigger_source,
        exposure_delay=p.exposure_delay_configured,
        exposure_delay_from_camera=p.exposure_delay_from_camera,
        _stream=stream, _server_defaults={}, crop_to_saved_roi=False,
        t_match_window=2.0, t_stray_grace=0.2, **_entry_kw(p))
    try:
        got = dict(core.worker(cid).settings)
        # the camera's own "triggered" mode, and nothing it does not have: no
        # trigger line to choose, no gain, no pixel format (the Basler words
        # the client used to send every camera)
        assert got["trigger_mode"] == "On" and cam.get_trigger_mode() == "ext"
        assert cs._modes_from_camera and cs._category == "pco_sc2"
        assert not {"trigger_source", "trigger_selector", "line_mode", "gain",
                    "pixel_format"} & set(got)
        fe.data.init()
        for j in (0, 1):
            cs.announce_trigger_mu(0, j)
            time.sleep(0.1)
            cam.trigger(1)
            deadline = time.monotonic() + 5.0
            while cs.counts["ok"] < j + 1 and time.monotonic() < deadline:
                fe.drain_shot_data(0.2)
                time.sleep(0.02)
        assert cs.counts["ok"] == 2 and cs.counts["failed"] == 0, (cs.counts, cs.events)
        for k in ("img_a", "img_b"):
            img = F(fe, k)[0]
            assert img.shape == (540, 640) and img.dtype == np.uint16
        assert not np.isnan(M(fe, "img_a", "seq")[0]) and not np.isnan(M(fe, "img_b", "seq")[0])
    finally:
        cs.finish()
    assert dict(core.worker(cid).settings)["trigger_mode"] == "Off"     # back in free run
    rec = json.loads(fe._extra_file_texts["camera_stream_img_a"])
    assert rec["camera_category"] == "pco_sc2" and rec["modes_from_camera"] is True


def test_a_gain_is_refused_for_a_camera_without_one(pco_served):
    cam, core, stream, cid = pco_served
    with pytest.raises(RuntimeError, match="has no gain setting"):
        CameraStreamClient(FakeExpt(), S, "img_pco", _stream=stream, _server_defaults={},
                           crop_to_saved_roi=False, pixel_format="Mono16",
                           exposure_time=1e-3, gain=6.0)


def test_a_camera_whose_category_has_no_triggered_mode_cannot_be_a_triggered_stream():
    from test_camera_stream_client import TrigFakeStream

    class FreeRunOnly(TrigFakeStream):
        def describe(self):
            return {"category": "andor_emccd",
                    "schema": {"name": "andor_emccd",
                               "settings": [{"key": "exposure_time"}, {"key": "gain"},
                                            {"key": "trigger_mode"}],
                               "modes": {"free_run": {"trigger_mode": "Off"}}}}
    stream = FreeRunOnly()
    with pytest.raises(RuntimeError, match="has no triggered mode"):
        TriggeredCameraStreamClient(FakeExpt(), S, "img_test", ttl=FakeTTL(), _stream=stream,
                                    _server_defaults={"exposure_time": 19e-6, "gain": 0.0})
    assert stream.closed and not stream.live                 # put back, nothing left on


def test_saved_defaults_come_from_the_cameras_own_server(pco_served, monkeypatch):
    cam, core, stream, cid = pco_served
    from beacon.camera.defaults import JsonDefaultsStore            # noqa: F401 (the store's home)
    from beacon.camera.stream import Unsupported
    # this test server keeps none: a camera that is not a Basler then has none, noted
    with pytest.raises(Unsupported):
        stream.get_defaults()
    client = CameraStreamClient.__new__(CameraStreamClient)
    client._stream, client.label, client.serial = stream, "t", S
    client._lock, client.counts, client.events = __import__("threading").RLock(), {}, []
    assert client._read_server_defaults() == {}
    assert client.counts.get("no_saved_defaults") == 1
    # a server that keeps them: its record, with the viewer's ROI as ``roi``
    monkeypatch.setattr(stream, "get_defaults", lambda: {
        "settings": {"exposure_time": 4e-3}, "viewer": {"roi": [1, 2, 30, 40]}, "keys": []})
    assert client._read_server_defaults() == {"exposure_time": 4e-3, "roi": [1, 2, 30, 40]}
