"""Camera streams built without a connection and connected later (queue
phase 2: the run's claim connects them, after a warm-launched run's kernel
was compiled), and DataVault host containers resized until the run starts.

Offline: the streams are the real waxx classes on in-memory fakes from
test_camera_stream_client (no network, no UDP, no servers).
"""
import json

import numpy as np
import pytest

from waxx.config.data_vault import DataVault, HostDataContainer
from waxx.control.cameras.camera_stream_client import (
    CameraStreamClient, TriggeredCameraStreamClient, PROVISIONAL_FRAME_SHAPE,
    connect_streams)

from test_camera_stream_client import (DEFAULTS, FakeExpt, FakeStream, FakeTTL,
                                       TrigFakeStream)


class CountingStream(FakeStream):
    """A FakeStream that counts every call (a connect-later stream must make
    none before connect())."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_calls = 0

    def snap(self, *a, **k):
        self.n_calls += 1
        return super().snap(*a, **k)

    def get_settings(self):
        self.n_calls += 1
        return super().get_settings()

    def set(self, **values):
        self.n_calls += 1
        return super().set(**values)


def _free(fe, stream, **kw):
    return CameraStreamClient(fe, "12345", "img_test", _stream=stream,
                              _server_defaults=DEFAULTS, connect=False, **kw)


def _trig(fe, stream, key="img_test", **kw):
    return TriggeredCameraStreamClient(
        fe, "12345", key, ttl=FakeTTL(), trigger_source="Line2",
        exposure_delay=17e-6, _stream=stream, _server_defaults=DEFAULTS,
        connect=False, t_match_window=0.6, t_stray_grace=0.1, **kw)


def test_built_without_connection_registers_provisional_containers():
    fe = FakeExpt()
    stream = CountingStream()
    cs = _free(fe, stream)
    try:
        assert stream.n_calls == 0 and cs._stream is None
        assert not cs.live and cs._worker is None
        assert fe.camera_streams == [cs]
        dc = fe.data.img_test
        assert isinstance(dc, HostDataContainer) and dc.stream_only
        assert dc._per_shot_data_shape == PROVISIONAL_FRAME_SHAPE
    finally:
        cs.finish()


def test_connect_then_activate_sets_the_probe_shape_after_init():
    fe = FakeExpt()
    stream = CountingStream()
    cs = _free(fe, stream)
    fe.data.init()
    try:
        cs.connect()
        assert stream.n_calls > 0 and cs._frame_shape == (6, 5)
        assert fe.data.img_test._per_shot_data_shape == PROVISIONAL_FRAME_SHAPE
        cs.activate()
        assert cs.live and cs._worker.is_alive()
        dc = fe.data.img_test
        assert dc._per_shot_data_shape == (6, 5) and dc.stream_only
        assert dc._run_data.shape == (4, 6, 5)
        assert fe.data.img_test_meta._run_data.shape[0] == 4
    finally:
        cs.finish()
    assert stream.closed


def test_a_triggered_stream_arms_only_at_connect():
    fe = FakeExpt()
    stream = TrigFakeStream()
    cs = _trig(fe, stream, key=["img_a", "img_b"])
    try:
        assert stream._settings["trigger_mode"] == "Off" and not stream.live
        fe.data.init()
        connect_streams([cs])
        assert cs.live
        assert stream._settings["trigger_mode"] == "On" and stream.live
        for k in ("img_a", "img_b"):
            assert getattr(fe.data, k)._per_shot_data_shape == (6, 5)
    finally:
        cs.finish()
    assert stream._settings["trigger_mode"] == "Off"


def test_a_dead_triggered_stream_counts_its_edges_and_stores_nothing():
    fe = FakeExpt()
    stream = TrigFakeStream()
    stream.start_live = lambda: {}          # the stream never starts: connect fails
    cs = _trig(fe, stream, key=["img_a", "img_b"], on_error="dummy")
    fe.data.init()
    res = connect_streams([cs])
    assert res[0][1] is not None and not cs.live
    assert isinstance(cs, TriggeredCameraStreamClient)          # no class swap
    assert stream.closed
    for k in ("img_a", "img_b"):
        dc = getattr(fe.data, k)
        assert dc._per_shot_data_shape == (1,) and dc._run_data.dtype == np.uint8
        assert not dc.stream_only
    cs.announce_trigger_mu(0, 0)
    cs.announce_trigger_mu(0, 1)
    assert cs.counts["edges_not_served"] == 2 and cs.counts["requested"] == 0
    assert fe.live_od_client.calls == []                        # nothing pushed
    cs.finish()
    for i, k in enumerate(("img_a", "img_b")):
        rec = json.loads(fe._extra_file_texts[f"camera_stream_{k}"])
        assert rec["dummy"] and rec["dead"] and rec["triggered"]
        assert rec["edges_not_served"] == 2 and "did not start" in rec["reason"]


def test_a_dead_free_run_stream_counts_its_snaps():
    fe = FakeExpt()
    stream = CountingStream()
    stream.snap_hook = lambda rev: RuntimeError("no camera (test)")
    cs = _free(fe, stream, on_error="dummy")
    fe.data.init()
    connect_streams([cs])
    assert not cs.live and "no camera" in cs._dead_reason
    cs.request_snap_mu(0, 0.)
    assert cs.counts["edges_not_served"] == 1 and cs._queue.empty()
    cs.finish()
    assert json.loads(fe._extra_file_texts["camera_stream_img_test"])["dead"]


def test_raise_policy_raises_and_marks_the_stream_dead():
    fe = FakeExpt()
    stream = CountingStream()
    stream.snap_hook = lambda rev: RuntimeError("no camera (test)")
    cs = _free(fe, stream, on_error="raise")
    fe.data.init()
    with pytest.raises(RuntimeError, match="no camera"):
        connect_streams([cs])
    assert not cs.live and cs._dead_reason


def test_a_stream_never_connected_finishes_and_exits_cleanly():
    """A run that stops between prepare and its claim (a warm run told to
    stand down): finish() and the exit cleanup touch no camera."""
    fe = FakeExpt()
    stream = CountingStream()
    cs = _free(fe, stream)
    cs.finish()
    cs._atexit_cleanup()
    other = _free(FakeExpt(), stream)
    other._atexit_cleanup()                 # the abort path (no end_wax)
    assert stream.n_calls == 0
    rec = json.loads(fe._extra_file_texts["camera_stream_img_test"])
    assert rec["dead"] and "never connected" in rec["reason"]


def test_connect_true_keeps_the_old_construction():
    fe = FakeExpt()
    stream = CountingStream()
    cs = CameraStreamClient(fe, "12345", "img_test", _stream=stream,
                            _server_defaults=DEFAULTS)
    try:
        assert cs.live and stream.n_calls > 0
        assert fe.data.img_test._per_shot_data_shape == (6, 5)
        assert connect_streams([cs]) == []                      # nothing to connect
    finally:
        cs.finish()


# ---- DataVault.resize_host_container -----------------------------------------

class _E:
    def __init__(self):
        self.xvardims = [3, 2]
        self._run_started = False


def test_host_container_resized_before_and_after_init_until_the_start():
    e = _E()
    d = DataVault(expt=e)
    d.frame = d.add_host_data_container((1,), np.uint8, fill_value=0, keep_run_data=False)
    d.kept = d.add_host_data_container((1,), np.uint8, fill_value=7)
    d.resize_host_container(d.frame, (4, 5), np.uint16)
    assert d.frame._per_shot_data_shape == (4, 5)
    d.init()
    assert d.frame._run_data.shape == (3, 2, 4, 5) and d.frame._run_data.dtype == np.uint16
    d.resize_host_container(d.frame, (6, 7), np.uint8)
    assert d.frame._run_data.shape == (3, 2, 6, 7) and d.frame.stream_only
    d.resize_host_container(d.kept, (2,), np.int16)
    assert d.kept._run_data.shape == (3, 2, 2) and (d.kept._run_data == 7).all()
    d.resize_host_container(d.frame, (1,), np.uint8, keep_run_data=True)
    assert d.frame._run_data.shape == (3, 2) and not d.frame.stream_only
    d.frame.put_shot_data_host((0, 1), 5)                      # a real array again
    e._run_started = True
    with pytest.raises(RuntimeError, match="has started"):
        d.resize_host_container(d.kept, (3,), np.int16)


def test_a_kernel_container_is_never_resized():
    e = _E()
    d = DataVault(expt=e)
    d.x = d.add_data_container(1)
    with pytest.raises(TypeError, match="not a host container"):
        d.resize_host_container(d.x, (2,), np.float64)


# ---- re-merge 2026-10-21: main's per-frame ROIs and camera exposure delay ----

class _Entry:
    def __init__(self, delay):
        self.exposure_start_delay_s = delay
        self.server_id = "camera_server:test"


class _Directory:
    """A directory whose entry reports ``delay`` (no network)."""
    def __init__(self, delay):
        self.entry = _Entry(delay)
        self.finds = []

    def find(self, query, *, refresh=False):
        self.finds.append(query)
        return self.entry


def test_per_frame_rois_take_their_shapes_at_the_late_connect():
    fe = FakeExpt()
    stream = TrigFakeStream()
    cs = _trig(fe, stream, key=["img_a", "img_b", "img_c"], crop_to_saved_roi=False,
               roi=[0, 0, 6, 4], frame_rois={"img_a": [1, 2, 4, 7], 1: None},
               no_roi_downsample=2)
    try:
        for k in ("img_a", "img_b", "img_c"):
            assert getattr(fe.data, k)._per_shot_data_shape == PROVISIONAL_FRAME_SHAPE
        fe.data.init()
        connect_streams([cs])
        assert cs.live
        for k, shape in (("img_a", (5, 3)), ("img_b", (5, 6)), ("img_c", (4, 6))):
            assert getattr(fe.data, k)._per_shot_data_shape == shape
            assert fe.data.__dict__[k]._run_data.shape == (4, *shape)
    finally:
        cs.finish()


def test_the_exposure_delay_is_decided_before_the_compile_and_kept_at_connect(capsys):
    """Built without a connection: the kernel-read exposure_delay comes from
    the camera server's entry at construction (before bind copies it and
    before a warm run's compile); the connect records the camera's value and
    never changes the kernel's."""
    fe = FakeExpt()
    stream = TrigFakeStream()
    stream.exposure_start_delay_s = 23.5e-6
    directory = _Directory(21.0e-6)
    cs = _trig(fe, stream, directory=directory)
    try:
        assert directory.finds == ["12345"] and stream._settings["trigger_mode"] == "Off"
        assert cs.exposure_delay == 21.0e-6
        assert cs.exposure_delay_source == "camera_before_claim"
        fe.data.init()
        connect_streams([cs])
        assert cs.live
        assert cs.exposure_delay == 21.0e-6                     # unchanged by the claim
        assert cs.exposure_delay_at_connect == 23.5e-6
        out = capsys.readouterr().out
        assert "reports an exposure delay of 23.50 us" in out and "kernel uses 21.00 us" in out
        extra = cs._provenance_extra()
        assert extra["exposure_delay"] == 21.0e-6
        assert extra["exposure_delay_source"] == "camera_before_claim"
        assert extra["exposure_delay_at_connect"] == 23.5e-6
    finally:
        cs.finish()


def test_the_exposure_delay_falls_back_to_configured_before_the_compile(capsys):
    fe = FakeExpt()
    stream = TrigFakeStream()
    stream.exposure_start_delay_s = 23.5e-6
    cs = _trig(fe, stream)              # the conftest's directory finds nothing
    try:
        assert cs.exposure_delay == 17e-6 and cs.exposure_delay_source == "configured"
        assert "before the compile" in capsys.readouterr().out
        fe.data.init()
        connect_streams([cs])
        assert cs.exposure_delay == 17e-6 and cs.exposure_delay_at_connect == 23.5e-6
    finally:
        cs.finish()
