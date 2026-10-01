"""Offline tests for waxx.control.cameras.camera_stream_client.

No network, no UDP, no servers: the client is constructed with the _stream /
_server_defaults test hooks, and the fake expt carries a real DataVault.
"""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from waxx.config.data_vault import DataVault, HostDataContainer
from waxx.control.cameras.camera_stream_client import (CameraStreamClient,
                                                       crop_to_roi)


class FakeFrame:
    def __init__(self, seq, shape=(10, 12), value=7, settings_rev=1):
        self.image = np.full(shape, value, dtype=np.uint8)
        self.seq = seq
        self.t_mono = time.monotonic()
        self.settings_rev = settings_rev
        self.settings = {"trigger_mode": "Off"}


class FakeStream:
    """Stands in for beacon.camera.stream.CameraStream."""

    def __init__(self, settings=None, shape=(10, 12)):
        self.entry = SimpleNamespace(host="127.0.0.1")
        self.camera_id = "basler_usb:12345"
        self.server_id = "camera_server:fake"
        self.shape = shape
        self._settings = dict(settings or {"exposure_time": 300e-6,
                                           "gain": 12.0,
                                           "trigger_mode": "Off"})
        self.rev = 1
        self._seq = 0
        self.set_calls = []
        self.closed = False
        self.snap_hook = None     # optional callable -> raise / frame

    def snap(self, timeout=5.0, policy="restart", require_rev=None):
        if self.snap_hook is not None:
            r = self.snap_hook(require_rev)
            if isinstance(r, Exception):
                raise r
            if r is not None:
                return r
        if require_rev is not None and require_rev != self.rev:
            from beacon.camera.stream import SettingsChanged
            raise SettingsChanged(require_rev, self.rev)
        self._seq += 1
        return FakeFrame(self._seq, shape=self.shape, settings_rev=self.rev)

    def get_settings(self):
        return dict(self._settings), self.rev

    def set(self, **values):
        self.set_calls.append(dict(values))
        self.rev += 1
        self._settings.update(values)
        rb = {k: {"value": v, "source": "live", "origin": "applied",
                  "requested": v} for k, v in values.items()}
        return {"readback": rb, "settings_rev": self.rev, "errors": {}}

    def close(self):
        self.closed = True


class FakeCore:
    def mu_to_seconds(self, mu):
        return float(mu) * 1e-9


class FakeExpt:
    def __init__(self, xvardims=(4,)):
        self.core = FakeCore()
        self.data = DataVault(expt=self)
        self.xvardims = list(xvardims)
        self.scan_xvars = [SimpleNamespace(counter=0)]
        self.camera_streams = []
        self._extra_file_texts = {}
        self.setup_camera = False


DEFAULTS = {"exposure_time": 19e-6, "gain": 0.0, "trigger_mode": "Off",
            "roi": [2, 3, 7, 9]}


def make_client(fe=None, stream=None, defaults=DEFAULTS, **kw):
    fe = fe or FakeExpt()
    stream = stream or FakeStream()
    cs = CameraStreamClient(fe, "12345", "img_test", _stream=stream,
                            _server_defaults=defaults,
                            t_late_tolerance=0.5, **kw)
    return fe, stream, cs


def wait_drained(cs, timeout=5.0):
    deadline = time.monotonic() + timeout
    while (not cs._queue.empty() or cs._busy) and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)


def test_construction_applies_defaults_and_crops():
    fe, stream, cs = make_client()
    try:
        # saved defaults differ from live settings -> one set() call
        assert stream.set_calls == [{"exposure_time": 19e-6, "gain": 0.0}]
        # frame cropped to the saved ROI [x1 y1 x2 y2] = [2,3,7,9] -> (6, 5)
        assert cs._frame_shape == (6, 5)
        assert cs._frame_dtype == np.uint8
        # containers registered under the user's key, host-only
        for key in ("img_test", "img_test_seq", "img_test_t",
                    "img_test_t_target"):
            assert isinstance(getattr(fe.data, key), HostDataContainer)
        assert fe.camera_streams == [cs]
    finally:
        cs.finish()


def test_overrides_beat_defaults():
    fe, stream, cs = make_client(stream=FakeStream(), exposure_time=42e-6,
                                 gain=3.0)
    try:
        assert stream.set_calls == [{"exposure_time": 42e-6, "gain": 3.0}]
    finally:
        cs.finish()


def test_no_set_when_already_at_defaults():
    stream = FakeStream(settings={"exposure_time": 19e-6, "gain": 0.0,
                                  "trigger_mode": "Off"})
    fe, stream, cs = make_client(stream=stream)
    try:
        assert stream.set_calls == []
        assert cs._priors == {}
    finally:
        cs.finish()


def test_host_containers_stay_out_of_kernel_lists():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        for lst in (fe.data._list_1d_f64, fe.data._list_2d_f64,
                    fe.data._list_1d_i32, fe.data._list_2d_i32,
                    fe.data._list_1d_i64, fe.data._list_2d_i64):
            assert all(dc._is_sentinel for dc in lst)
        assert set(fe.data.keys) == {"img_test", "img_test_seq", "img_test_t",
                                     "img_test_t_target"}
        # sized to (*xvardims, *per_shot) with fill values intact
        assert fe.data.img_test._run_data.shape == (4, 6, 5)
        assert fe.data.img_test_seq._run_data.shape == (4,)
        assert np.all(fe.data.img_test_seq._run_data == -1)
        assert np.all(np.isnan(fe.data.img_test_t._run_data))
        # END_RUN saves host containers (data_gotten True from birth)
        assert fe.data.img_test._data_gotten
    finally:
        cs.finish()


def test_per_shot_capture_lands_at_counter_index():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        for shot in (0, 2):
            fe.scan_xvars[0].counter = shot
            cs.request_snap_mu(0, 0.0)
            wait_drained(cs)
        assert fe.data.img_test_seq._run_data[0] > 0
        assert fe.data.img_test_seq._run_data[1] == -1
        assert fe.data.img_test_seq._run_data[2] > fe.data.img_test_seq._run_data[0]
        assert np.all(fe.data.img_test._run_data[0] == 7)
        assert np.isfinite(fe.data.img_test_t._run_data[0])
        assert np.isfinite(fe.data.img_test_t_target._run_data[0])
        assert cs._shot_log["ok"] == 2
    finally:
        cs.finish()


def test_scheduled_delay_is_honored():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        t0 = time.monotonic()
        cs.request_snap_mu(0, 0.3)
        wait_drained(cs, timeout=5.0)
        t_frame = fe.data.img_test_t._run_data[0]
        assert t_frame - t0 >= 0.29
    finally:
        cs.finish()


def test_late_request_skipped_not_snapped():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        cs.request_snap_mu(0, -5.0)       # target 5 s in the past
        wait_drained(cs)
        assert cs._shot_log["late"] == 1
        assert fe.data.img_test_seq._run_data[0] == -1
    finally:
        cs.finish()


def test_duplicate_request_clears_slot():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        seq1 = fe.data.img_test_seq._run_data[0]
        assert seq1 > 0
        # same index requested again, but the snap now fails
        stream.snap_hook = lambda rev: RuntimeError("boom")
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        assert cs._shot_log["duplicate"] == 1
        assert fe.data.img_test_seq._run_data[0] == -1   # cleared, not stale
        assert cs._shot_log["failed"] == 1
    finally:
        cs.finish()


def test_snap_failure_is_recorded_never_raised():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        from beacon.camera.stream import RunLocked
        stream.snap_hook = lambda rev: RunLocked("0")
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        assert cs._shot_log["failed"] == 1
        assert fe.data.img_test_seq._run_data[0] == -1
    finally:
        cs.finish()


def test_settings_changed_reapplies_once():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        stream.rev += 5      # someone changed settings since our pin
        n_sets = len(stream.set_calls)
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        assert len(stream.set_calls) == n_sets + 1     # one reapply
        assert cs._shot_log["ok"] == 1
        assert cs._rev == stream.rev
    finally:
        cs.finish()


def test_finish_writes_provenance_and_restores():
    fe, stream, cs = make_client()
    fe.data.init()
    cs.request_snap_mu(0, 0.0)
    wait_drained(cs)
    cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_test"])
    assert rec["key"] == "img_test"
    assert rec["shots"]["ok"] == 1
    assert rec["frame_shape"] == [6, 5]
    # priors restored: the last set() call puts back the original values
    assert stream.set_calls[-1] == {"exposure_time": 300e-6, "gain": 12.0}
    assert stream.closed
    # finish is idempotent
    cs.finish()


def test_no_restore_when_someone_else_changed_settings():
    fe, stream, cs = make_client()
    fe.data.init()
    n = len(stream.set_calls)
    stream.rev += 1          # a viewer changed something after our pin
    cs.finish()
    assert len(stream.set_calls) == n    # no restore
    assert stream.closed


def test_rpc_handler_never_raises():
    fe, stream, cs = make_client()
    try:
        fe.scan_xvars = None        # force an internal error
        cs.request_snap_mu(0, 0.0)  # must not raise
        assert cs._shot_log["rpc_errors"] == 1
    finally:
        fe.scan_xvars = [SimpleNamespace(counter=0)]
        cs.finish()


def test_key_collision_and_late_registration_refused():
    fe = FakeExpt()
    fe.data.existing = fe.data.add_data_container()
    with pytest.raises(ValueError):
        CameraStreamClient(fe, "12345", "existing", _stream=FakeStream(),
                           _server_defaults=DEFAULTS)
    fe2 = FakeExpt()
    fe2.data.init()   # too late
    with pytest.raises(RuntimeError):
        CameraStreamClient(fe2, "12345", "img_test", _stream=FakeStream(),
                           _server_defaults=DEFAULTS)


def test_dummy_registers_placeholder_containers():
    from waxx.control.cameras.camera_stream_client import (
        DummyCameraStreamClient)
    fe = FakeExpt()
    cs = DummyCameraStreamClient(fe, "img_test", reason="run camera")
    fe.data.init()
    assert fe.camera_streams == [cs]
    # one byte per shot instead of a frame; every shot marked "no frame"
    assert fe.data.img_test._run_data.shape == (4,)
    assert fe.data.img_test._run_data.dtype == np.uint8
    assert (fe.data.img_test_seq._run_data == -1).all()
    assert np.isnan(fe.data.img_test_t._run_data).all()
    assert np.isnan(fe.data.img_test_t_target._run_data).all()
    cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_test"])
    assert rec["dummy"] is True and rec["reason"] == "run camera"


class FakeDirectory:
    def __init__(self, entry):
        self.entry = entry
        self.calls = []

    def find(self, query, *, refresh=False):
        self.calls.append(refresh)
        return self.entry


def _entry(server_id, state="open", holder=None):
    return SimpleNamespace(camera_id="basler_usb:12345", server_id=server_id,
                           state=state, holder=holder)


def test_pinned_directory_never_leaves_its_server():
    from beacon.camera.directory import CameraNotFound
    from waxx.control.cameras.camera_stream_client import _PinnedDirectory
    liveod = "camera_server:pc:liveod"
    d = FakeDirectory(_entry(liveod))
    pinned = _PinnedDirectory(d, liveod)
    assert pinned.find("12345", refresh=True) is d.entry
    # liveOD stalls: only the lender's "reserved" listing is left
    d.entry = _entry("camera_server:pc", state="reserved")
    with pytest.raises(CameraNotFound):
        pinned.find("12345", refresh=True)


class FakeLiveOD:
    def __init__(self, state):
        self.state = state
        self.opened = []

    def cameras(self):
        return {"cam_a": {"state": self.state, "serial_no": "12345"}}

    def open_camera(self, key):
        self.opened.append(key)
        self.state = "open"


def _bare_client(live_od_client):
    cs = object.__new__(CameraStreamClient)
    cs.label, cs.serial = "camera_stream:img_test", "12345"
    cs._live_od_client = live_od_client
    return cs


@pytest.fixture
def stream_made(monkeypatch):
    import beacon.camera.stream as bcs
    made = []
    monkeypatch.setattr(
        bcs, "CameraStream",
        lambda entry, label, directory: made.append((entry, directory))
        or SimpleNamespace(entry=entry))
    return made


@pytest.mark.parametrize("state", ["closed", "failed", "loading"])
def test_lease_asked_from_liveod_and_stream_pinned(stream_made, state):
    from waxx.control.cameras.camera_stream_client import _PinnedDirectory
    lod = FakeLiveOD(state)
    d = FakeDirectory(_entry("camera_server:pc:liveod"))
    _bare_client(lod)._resolve_and_connect(d)
    assert lod.opened == ["cam_a"]
    assert d.calls == [True]
    assert isinstance(stream_made[0][1], _PinnedDirectory)


def test_open_liveod_camera_needs_no_lease(stream_made):
    lod = FakeLiveOD("open")
    _bare_client(lod)._resolve_and_connect(
        FakeDirectory(_entry("camera_server:pc:liveod")))
    assert lod.opened == []


def test_liveod_camera_never_opened_on_the_lender(stream_made):
    # liveOD lists it but its camera server did not answer the directory
    with pytest.raises(RuntimeError, match="behind liveOD's back"):
        _bare_client(FakeLiveOD("open"))._resolve_and_connect(
            FakeDirectory(_entry("camera_server:pc", state="reserved")))
    # no liveOD client at all, camera lent to liveOD
    holder = {"server_id": "camera_server:pc:liveod"}
    with pytest.raises(RuntimeError, match="lent to liveOD"):
        _bare_client(None)._resolve_and_connect(
            FakeDirectory(_entry("camera_server:pc", state="reserved",
                                 holder=holder)))
    assert stream_made == []


def test_camera_not_on_liveod_is_not_pinned(stream_made):
    d = FakeDirectory(_entry("camera_server:pc"))
    _bare_client(None)._resolve_and_connect(d)
    assert stream_made[0][1] is d


def test_crop_to_roi_edge_cases():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert crop_to_roi(img, None).shape == (10, 10)
    assert crop_to_roi(img, [5, 5, 5, 9]).shape == (10, 10)   # degenerate
    assert crop_to_roi(img, [-5, -5, 50, 50]).shape == (10, 10)  # clamped
    c = crop_to_roi(img, [2, 3, 7, 9])
    assert c.shape == (6, 5) and c[0, 0] == 32
