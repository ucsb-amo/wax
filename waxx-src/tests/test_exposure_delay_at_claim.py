"""A triggered camera stream's exposure delay: provisional at the compile,
decided at the run's claim with this run's settings, taken by the kernel at
its start through ONE RPC (Scanner.load_exposure_delays).

Run 87656 (2026-10-21): the claim connected the diagnostic streams after the
compile, the kernel kept the camera server's last report (the previous run's
settings: 23593 us) while the camera said 21131 us at the claim, and those
frames fired 2.46 ms early.

Offline: the real waxx classes on the in-memory fakes of
test_camera_stream_client (no network, no UDP, no servers). The kernel's
loader is run on the host through its Python body; a separate test compiles
it (and a diagnostic image's trigger) with ARTIQ, no core device.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from artiq.experiment import delay, kernel

from waxx.base.scanner import Scanner
from waxx.control.cameras import exposure_delay as xd
from waxx.control.cameras.camera_param_classes import BaslerParams
from waxx.control.cameras.camera_stream_client import (
    CameraStreamClient, DummyCameraStreamClient, InactiveCameraStream,
    TriggeredCameraStreamClient, connect_streams)
from waxx.control.cameras.diagnostic_frame import diagnostic_frame

from test_camera_stream_client import FakeExpt, FakeTTL, TrigFakeStream
from test_camera_stream_connect_later import CountingStream, _Directory, _free, _trig

AT_CLAIM = 21131e-6         # run 87656, 2D-MOT basler: the camera at the claim
BEFORE = 23593e-6           # ... and what the kernel used (the previous run's)


class Expt(FakeExpt):
    """FakeExpt with the Scanner's loader, its RPC and the records hook."""

    _exposure_delays_at_claim = Scanner._exposure_delays_at_claim

    def __init__(self):
        super().__init__()
        xd.init_cells(self)
        self._exposure_delays_loaded = False
        self.hook_calls = 0

    def exposure_delays_loaded(self):
        self.hook_calls += 1


def run_loader(expt):
    """The body of the @kernel Scanner.load_exposure_delays, run on the host."""
    Scanner.load_exposure_delays.artiq_embedded.function(expt)


def stream(expt, at_claim=AT_CLAIM, before=BEFORE, key="img_test", **kw):
    s = TrigFakeStream()
    s.exposure_start_delay_s = at_claim
    return _trig(expt, s, key=key, directory=_Directory(before), **kw), s


# ---- the RPC -------------------------------------------------------------

def test_the_rpc_returns_the_values_decided_at_the_claim_in_cell_order():
    e = Expt()
    a, _ = stream(e, at_claim=21e-6, before=30e-6, key="img_a")
    b, _ = stream(e, at_claim=22e-6, before=31e-6, key="img_b")
    try:
        assert xd.cells(e) == [xd.dummy_cell(e), a.exposure_delay_cell, b.exposure_delay_cell]
        e.data.init()
        connect_streams([a, b])
        d = e._exposure_delays_at_claim()
        assert d.dtype == np.float64 and d.shape == (4,)
        assert list(d) == [0., 21e-6, 22e-6, 1.]          # spare, a, b, final
        assert e.hook_calls == 1
    finally:
        a.finish()
        b.finish()


def test_before_the_claim_the_rpc_gives_the_compiled_values_and_is_not_final():
    e = Expt()
    a, _ = stream(e)
    try:
        d = e._exposure_delays_at_claim()
        assert list(d) == [0., BEFORE, 0.]
        run_loader(e)
        assert a.exposure_delay_cell.value == BEFORE
        assert bool(e._exposure_delays_loaded) is False           # stays armed for the next call
        assert a.exposure_delay_to_kernel == "compiled_in"
    finally:
        a.finish()


# ---- the kernel's value ---------------------------------------------------

def test_the_kernel_takes_the_value_reported_at_the_claim_not_the_directory_one():
    e = Expt()
    a, cam = stream(e)
    try:
        assert a.exposure_delay_cell.value == BEFORE          # compiled in (provisional)
        e.data.init()
        connect_streams([a])
        assert a.exposure_delay_cell.value == BEFORE          # the claim touches no kernel value
        run_loader(e)
        assert a.exposure_delay_cell.value == AT_CLAIM == a.exposure_delay_at_connect
        assert bool(e._exposure_delays_loaded) is True
        assert a.exposure_delay == AT_CLAIM
        assert a.exposure_delay_source == "camera_at_claim"
        assert a.exposure_delay_to_kernel == "loaded_at_kernel_start"
        # a second load (scan() after init_kernel) makes no RPC
        cam.exposure_start_delay_s = 1.
        e._exposure_delays_at_claim = lambda: pytest.fail("a second RPC")
        run_loader(e)
        assert a.exposure_delay_cell.value == AT_CLAIM
    finally:
        a.finish()


def test_the_claim_value_is_read_after_our_trigger_settings():
    """The camera's report depends on its settings: the value taken is the
    one it gives once trigger mode is on (as on main before the landing)."""
    e = Expt()
    s = TrigFakeStream()
    real_status = s.status

    def status(refresh=True):
        st = real_status(refresh)
        st.raw["exposure_start_delay_s"] = (AT_CLAIM if s._settings.get("trigger_mode") == "On"
                                            else BEFORE)
        return st

    s.status = status
    a = _trig(e, s, directory=_Directory(BEFORE))
    try:
        e.data.init()
        connect_streams([a])
        run_loader(e)
        assert a.exposure_delay_cell.value == AT_CLAIM
    finally:
        a.finish()


# ---- fallbacks -------------------------------------------------------------

def test_a_camera_that_reports_no_delay_at_the_claim_gives_the_configured_one_loudly(capsys):
    e = Expt()
    a, _ = stream(e, at_claim=None)
    try:
        e.data.init()
        connect_streams([a])
        out = capsys.readouterr().out
        assert "!!" in out and "not read from the camera at the claim" in out
        run_loader(e)
        assert a.exposure_delay_cell.value == 17e-6           # configured, not the directory's
        assert a.exposure_delay_source == "configured"
        rec = a.exposure_delay_record()
        assert rec["exposure_delay_before_claim"] == BEFORE
        assert rec["exposure_delay_at_connect"] is None
    finally:
        a.finish()


def test_exposure_delay_from_camera_off_is_configured_throughout():
    e = Expt()
    a, cam = stream(e, exposure_delay_from_camera=False)
    try:
        assert a.exposure_delay_cell.value == 17e-6           # no directory read
        e.data.init()
        connect_streams([a])
        run_loader(e)
        assert a.exposure_delay_cell.value == 17e-6
        assert a.exposure_delay_source == "configured" and a.exposure_delay_at_connect is None
    finally:
        a.finish()


def test_a_stream_that_fails_to_connect_takes_the_configured_delay():
    e = Expt()
    s = TrigFakeStream()
    s.start_live = lambda: {}               # never streams: the connect fails
    a = _trig(e, s, directory=_Directory(BEFORE), on_error="dummy")
    try:
        e.data.init()
        connect_streams([a])
        assert not a.live
        run_loader(e)
        assert a.exposure_delay_cell.value == 17e-6
        assert a.exposure_delay_at_claim_source == "configured"
        assert bool(e._exposure_delays_loaded) is True
    finally:
        a.finish()


# ---- zero streams, snap streams, placeholders -------------------------------

def test_no_streams_the_spare_cell_alone():
    e = Expt()
    assert len(xd.cells(e)) == 1
    d = e._exposure_delays_at_claim()
    assert list(d) == [0., 1.]
    run_loader(e)
    assert bool(e._exposure_delays_loaded) is True and xd.dummy_cell(e).value == 0.


def test_snap_streams_and_placeholders_have_no_cell():
    e = Expt()
    f = _free(e, CountingStream())
    DummyCameraStreamClient(e, "img_dummy", reason="test", triggered=True)
    InactiveCameraStream(e, "img_off", reason="test")
    try:
        assert len(xd.cells(e)) == 1                          # the spare one only
        e.data.init()
        connect_streams([f])
        run_loader(e)
        assert bool(e._exposure_delays_loaded) is True
        assert not hasattr(f, "exposure_delay_cell")
    finally:
        f.finish()


def test_a_stream_built_with_a_connection_is_compiled_in_and_not_registered():
    e = Expt()
    s = TrigFakeStream()
    s.exposure_start_delay_s = AT_CLAIM
    a = TriggeredCameraStreamClient(e, "12345", "img_test", ttl=FakeTTL(),
                                    trigger_source="Line2", exposure_delay=17e-6,
                                    _stream=s, _server_defaults={"exposure_time": 19e-6,
                                                                 "gain": 0.0, "roi": None})
    try:
        assert a.exposure_delay_cell.value == AT_CLAIM        # read at its connect, pre-compile
        assert a.exposure_delay_source == "camera"
        assert len(xd.cells(e)) == 1
        assert a.exposure_delay_record()["exposure_delay_to_kernel"] == "compiled_in"
    finally:
        a.finish()


def test_the_rpc_never_raises():
    e = Expt()
    a, _ = stream(e)
    try:
        a._exposure_delay_for_kernel = lambda: 1 / 0
        d = e._exposure_delays_at_claim()
        assert list(d) == [0., BEFORE, 0.]                     # its compiled value, not final
    finally:
        a.finish()


# ---- records ----------------------------------------------------------------

def test_the_provenance_says_which_value_the_kernel_used():
    e = Expt()
    a, _ = stream(e)
    e.data.init()
    connect_streams([a])
    run_loader(e)
    a.finish()
    rec = json.loads(e._extra_file_texts["camera_stream_img_test"])
    assert rec["exposure_delay"] == AT_CLAIM
    assert rec["exposure_delay_source"] == "camera_at_claim"
    assert rec["exposure_delay_to_kernel"] == "loaded_at_kernel_start"
    assert rec["exposure_delay_before_claim"] == BEFORE
    assert rec["exposure_delay_before_claim_source"] == "camera_before_claim"
    assert rec["exposure_delay_at_connect"] == AT_CLAIM
    assert rec["exposure_delay_at_claim"] == AT_CLAIM
    assert rec["exposure_delay_configured"] == 17e-6


def test_the_provenance_of_a_kernel_that_never_loaded_says_compiled_in():
    e = Expt()
    a, _ = stream(e)
    e.data.init()
    connect_streams([a])
    a.finish()
    rec = json.loads(e._extra_file_texts["camera_stream_img_test"])
    assert rec["exposure_delay"] == BEFORE                    # what the kernel really used
    assert rec["exposure_delay_source"] == "camera_before_claim"
    assert rec["exposure_delay_to_kernel"] == "compiled_in"
    assert rec["exposure_delay_at_claim"] == AT_CLAIM


# ---- the diagnostic frame -----------------------------------------------------

def _cam(serial, key):
    return BaslerParams(serial_number=serial, exposure_delay=17e-6, key=key)


class _Frame(diagnostic_frame):
    def __init__(self):
        self.setup()
        cam_a, cam_b = _cam("12345", "cam_a"), _cam("67890", "cam_b")
        self.img_a1 = self.image(cam_a, t_offset=-60e-3, exposure_time=3.5e-3, gain=20.)
        self.img_a2 = self.image(cam_a, t_offset=-25e-3, exposure_time=3e-3, gain=30.)
        self.img_b = self.image(cam_b, t_offset=1e-3, exposure_time=2e-6, gain=0.)
        self.cleanup()


def _bind(e, enabled=True):
    made = {}

    def make_streams(expt, specs):
        out = []
        for spec in specs:
            if spec["camera"].key == "cam_a":
                cs, _ = stream(expt, key=spec["key"])
                made["a"] = cs
            else:
                cs = DummyCameraStreamClient(expt, spec["key"], reason="test", triggered=True)
            out.append(cs)
        return out

    return _Frame().bind(e, enabled, make_streams, trigger_ttl=lambda c: FakeTTL()), made


def test_diagnostic_images_read_their_streams_cell_and_the_record_follows():
    e = Expt()
    diag, made = _bind(e)
    a = made["a"]
    try:
        assert diag.img_a1.exposure_delay_cell is a.exposure_delay_cell
        assert diag.img_a2.exposure_delay_cell is a.exposure_delay_cell
        assert diag.img_b.exposure_delay_cell is xd.dummy_cell(e)     # takes no frames
        e.data.init()
        connect_streams([a])
        r = diag.record()["images"]
        assert r["img_a1"]["exposure_delay"] == BEFORE                  # before the kernel loads
        assert r["img_a1"]["exposure_delay_to_kernel"] == "compiled_in"
        assert r["img_a1"]["exposure_delay_at_claim"] == AT_CLAIM
        run_loader(e)
        assert diag.img_a1.exposure_delay_cell.value == AT_CLAIM
        r = diag.record()["images"]
        for k in ("img_a1", "img_a2"):
            assert r[k]["exposure_delay"] == AT_CLAIM
            assert r[k]["exposure_delay_source"] == "camera_at_claim"
            assert r[k]["exposure_delay_before_claim"] == BEFORE
            assert r[k]["exposure_delay_to_kernel"] == "loaded_at_kernel_start"
        assert r["img_b"]["exposure_delay"] is None
        json.dumps(diag.record())
    finally:
        a.finish()


def test_diagnostics_off_every_image_reads_the_spare_cell():
    e = Expt()
    diag, made = _bind(e, enabled=False)
    assert not made
    assert all(img.exposure_delay_cell is xd.dummy_cell(e) for img in diag.images().values())
    run_loader(e)
    assert bool(e._exposure_delays_loaded) is True
    assert all(v["exposure_delay"] is None for v in diag.record()["images"].values())


# ---- ARTIQ: the loader and the triggers compile ---------------------------------

class _TTL:
    def __init__(self, core):
        self.core = core

    @kernel
    def pulse(self, t):
        delay(t)


class _Compiled(Expt):
    """The Scanner's loader plus diagnostic images' triggers in one kernel,
    compiled offline through waxx's Core (host=None: nothing can connect)."""

    load_exposure_delays = Scanner.load_exposure_delays

    def __init__(self, n_streams, diag_enabled=True):
        from waxx.util.artiq.core import Core
        super().__init__()
        dmgr = SimpleNamespace()
        self.core = Core(dmgr, None, 1e-9, target="cortexa9")
        dmgr.get = lambda name: self.core
        self.streams = []
        for i in range(n_streams):
            cs, _ = stream(self, key=f"img_{i}")
            cs.ttl = _TTL(self.core)
            self.streams.append(cs)
        made = iter(self.streams)

        def make_streams(expt, specs):
            return [next(made, None) or DummyCameraStreamClient(expt, s["key"], triggered=True)
                    for s in specs]

        self.diag = _Frame().bind(self, diag_enabled, make_streams,
                                  trigger_ttl=lambda c: _TTL(self.core))

    @kernel
    def run(self):
        self.load_exposure_delays()
        self.diag.img_a1.trigger()
        self.diag.img_a2.trigger()
        self.diag.img_b.trigger()


def _compile(obj):
    return obj.core.compile(type(obj).run, [obj], {}, attribute_writeback=True,
                            print_as_rpc=False)


@pytest.mark.parametrize("n_streams,diag_enabled", [(0, False), (0, True), (1, True), (2, True)])
def test_the_loader_and_the_triggers_compile(n_streams, diag_enabled):
    obj = _Compiled(n_streams, diag_enabled)
    try:
        embedding_map = _compile(obj)[0]
        attrs = {(type(o).__name__, a) for _, o, t in embedding_map.iter_objects()
                 for a in getattr(t, "attributes", {})}
        assert ("ExposureDelayCell", "value") in attrs
        assert ("_Compiled", "_exposure_delay_cells") in attrs
        # an image reads its cell; it holds no delay of its own
        assert ("DiagnosticImage", "exposure_delay_cell") in attrs
        assert ("DiagnosticImage", "exposure_delay") not in attrs
    finally:
        for cs in obj.streams:
            cs.finish()


def test_a_stream_trigger_compiles_with_its_cell():
    class _S(_Compiled):
        @kernel
        def run(self):
            self.load_exposure_delays()
            self.streams[0].trigger(0)

    obj = _S(1)
    try:
        _compile(obj)
    finally:
        obj.streams[0].finish()
