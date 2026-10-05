"""Offline tests for waxx.control.cameras.camera_stream_client.

No network, no UDP, no servers: the client is constructed with the _stream /
_server_defaults test hooks, and the fake expt carries a real DataVault and
Expt's real shot queue, whose PUT_DATA goes to an in-memory stand-in for the
run's file (FakePutData). Stream containers are stream-only: the frames are
read back from that stand-in (F / M), never from the containers.
"""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from waxa.data.camera_frames import META_FIELDS, N_META
from waxx.config.data_vault import DataVault, HostDataContainer
from waxx.control.cameras.camera_stream_client import (CameraStreamClient,
                                                       crop_to_roi)


class FakeFrame:
    def __init__(self, seq, shape=(10, 12), value=7, settings_rev=1):
        self.image = np.full(shape, value, dtype=np.uint8)
        self.seq = seq
        self.t_mono = time.monotonic()
        self.settings_rev = settings_rev
        self.settings = {"trigger_mode": "Off", "exposure_time": 19e-6,
                         "gain": 0.0}
        self.hw_idx = seq
        self.hw_ts = 1000 * seq


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


class FakePutData:
    """liveOD's PUT_DATA as the run's file sees it: each key's dataset made
    at its full_shape with its fill on first use, written at each index.
    ``refuse(specs)`` -> True makes liveOD refuse that message."""

    def __init__(self):
        self.arrays = {}
        self.calls = []
        self.history = []           # (key, index, array) of every write taken
        self.refuse = None
        self.enabled = True
        self._lock = threading.Lock()

    def supports(self, feature):
        return self.enabled and feature == "put_data"

    def put_data(self, specs):
        with self._lock:
            self.calls.append([(s["key"], tuple(s["index"])) for s in specs])
            if self.refuse is not None and self.refuse(specs):
                return {"ok": False, "error": "refused (test)"}
            for s in specs:
                arr = np.asarray(s["array"])
                self.history.append((s["key"], tuple(s["index"]), arr.copy()))
                if s["key"] not in self.arrays:
                    self.arrays[s["key"]] = np.full(s["full_shape"], s["fill"],
                                                    dtype=arr.dtype)
                self.arrays[s["key"]][tuple(s["index"])] = arr
            return {"ok": True}


def _expt_methods():
    """Expt's real queue / push methods, to bind onto a fake expt."""
    from waxx.base.expt import Expt
    names = ("current_shot_index", "final_shot_index", "push_data_enabled",
             "shot_data_queue", "queue_shot_data", "queue_shot_clear",
             "_queue_shot_specs", "_warn_no_put_data", "_push_queued",
             "drain_shot_data", "_stream_only_keys", "_close_shot_data_queue",
             "push_shot_data", "push_raw", "_push", "_note_push_failure",
             "_push_whole_container", "_serialize_end_payload",
             "PUSH_QUEUE_MAX_BYTES", "PUSH_QUEUE_RETRIES", "T_PUSH_QUEUE_CLOSE_S",
             "BULK_PUSH_BYTES", "PUT_DATA_SLICE_BYTES")
    return {n: vars(Expt)[n] for n in names}


class FakeExpt:
    locals().update(_expt_methods())

    def __init__(self, xvardims=(4,)):
        self.core = FakeCore()
        self.data = DataVault(expt=self)
        self.xvardims = list(xvardims)
        self.scan_xvars = [SimpleNamespace(counter=0)]
        self.camera_streams = []
        self._extra_file_texts = {}
        self.setup_camera = False
        self.sort_idx, self.sort_N = [], []
        self.run_info = SimpleNamespace(save_data=True)
        self.live_od_client = FakePutData()
        self._shot_queue = None
        self._push_failed = set()
        self._n_pushed = self._n_push_failed = 0


DEFAULTS = {"exposure_time": 19e-6, "gain": 0.0, "trigger_mode": "Off",
            "roi": [2, 3, 7, 9]}


def F(fe, key):
    """``key`` as the run's file holds it: what was pushed, else the fill
    liveOD pre-allocated (the container's zero-memory view, read only)."""
    fe.drain_shot_data(5.0)
    got = fe.live_od_client.arrays.get(key)
    return got if got is not None else np.array(fe.data.__dict__[key]._run_data)


def M(fe, key, field):
    """The META_FIELDS column ``field`` of ``<key>_meta``, shaped (*xvardims,)."""
    return F(fe, key + "_meta")[..., META_FIELDS.index(field)]


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
    cs._expt.drain_shot_data(timeout)       # and the shot queue delivered it


def test_construction_applies_defaults_and_crops():
    fe, stream, cs = make_client()
    try:
        # saved defaults differ from live settings -> one set() call
        assert stream.set_calls == [{"exposure_time": 19e-6, "gain": 0.0}]
        # frame cropped to the saved ROI [x1 y1 x2 y2] = [2,3,7,9] -> (6, 5)
        assert cs._frame_shape == (6, 5)
        assert cs._frame_dtype == np.uint8
        # containers registered under the user's key, host-only: the frame
        # and ONE record per shot
        for key in ("img_test", "img_test_meta"):
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
        assert set(fe.data.keys) == {"img_test", "img_test_meta"}
        # sized to (*xvardims, *per_shot) with fill values intact
        assert fe.data.img_test._run_data.shape == (4, 6, 5)
        assert fe.data.img_test_meta._run_data.shape == (4, N_META)
        assert np.isnan(fe.data.img_test_meta._run_data).all()
        # stream-only: the shape is a view of one fill value, no frames kept
        for key in ("img_test", "img_test_meta"):
            dc = getattr(fe.data, key)
            assert dc.stream_only and dc._run_data.strides == (0,) * dc._run_data.ndim
        assert fe.data.img_test._data_gotten
    finally:
        cs.finish()


def test_duplicate_request_queues_a_clear_ahead_of_the_new_frame():
    """A warm-up shot's frame already in the file is cleared before the
    real shot's frame goes in, in that order; the real frame then wins."""
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        first_seq = M(fe, "img_test", "seq")[0]
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        calls = [c for c in fe.live_od_client.calls]
        # frame, then clear, then the new frame: all for slot 0
        assert len(calls) == 3 and all(c[0] == ("img_test", (0,)) for c in calls)
        seqs = [a[0] for k, i, a in fe.live_od_client.history if k == "img_test_meta"]
        assert seqs[0] == first_seq and np.isnan(seqs[1]) and seqs[2] > first_seq
        frames = [a for k, i, a in fe.live_od_client.history if k == "img_test"]
        assert np.all(frames[1] == 0) and np.all(frames[2] == 7)
        assert cs.counts["slot_clears_sent"] == 1 and cs.counts["duplicate"] == 1
        assert M(fe, "img_test", "seq")[0] > first_seq
        assert cs._n_missing() == 0
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
        seq = M(fe, "img_test", "seq")
        assert seq[0] > 0
        assert np.isnan(seq[1])
        assert seq[2] > seq[0]
        assert np.all(F(fe, "img_test")[0] == 7)
        assert np.isfinite(M(fe, "img_test", "t")[0])
        assert np.isfinite(M(fe, "img_test", "t_target")[0])
        assert M(fe, "img_test", "exposure")[0] == 19e-6
        assert M(fe, "img_test", "hw_idx")[0] == seq[0]
        assert cs.counts["ok"] == 2
    finally:
        cs.finish()


def test_scheduled_delay_is_honored():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        t0 = time.monotonic()
        cs.request_snap_mu(0, 0.3)
        wait_drained(cs, timeout=5.0)
        t_frame = M(fe, "img_test", "t")[0]
        assert t_frame - t0 >= 0.29
    finally:
        cs.finish()


def test_late_request_skipped_not_snapped():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        cs.request_snap_mu(0, -5.0)       # target 5 s in the past
        wait_drained(cs)
        assert cs.counts["late"] == 1
        assert np.isnan(M(fe, "img_test", "seq")[0])
    finally:
        cs.finish()


def test_duplicate_request_clears_slot():
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        assert M(fe, "img_test", "seq")[0] > 0
        # same index requested again, but the snap now fails
        stream.snap_hook = lambda rev: RuntimeError("boom")
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        assert cs.counts["duplicate"] == 1
        assert np.isnan(M(fe, "img_test", "seq")[0])   # cleared, not stale
        assert np.all(F(fe, "img_test")[0] == 0)
        assert cs.counts["failed"] == 1
        # the slot is missing (the first frame was cleared, the second failed)
        assert cs._n_missing() == 1
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
        assert cs.counts["failed"] == 1
        assert np.isnan(M(fe, "img_test", "seq")[0])
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
        assert cs.counts["ok"] == 1
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


def test_someone_elses_change_keeps_their_exposure_and_trigger_stays_off():
    stream = FakeStream(settings={"exposure_time": 300e-6, "gain": 12.0,
                                  "trigger_mode": "On"})
    fe, stream, cs = make_client(stream=stream)
    fe.data.init()
    n = len(stream.set_calls)
    stream.rev += 1          # a viewer changed something after our pin
    cs.finish()
    # their exposure/gain stay; the trigger mode is Off after a run whatever
    # it was before, so nothing is put back to On
    assert stream.set_calls[n:] == []
    assert stream._settings["trigger_mode"] == "Off"
    assert stream.closed


def test_trigger_mode_is_off_after_a_run_even_if_someone_turned_it_on():
    fe, stream, cs = make_triggered()
    fe.data.init()
    stream.rev += 1          # someone changed the camera during the run
    cs.finish()
    assert stream._settings["trigger_mode"] == "Off"
    assert stream._settings["trigger_source"] == "Line1"
    # their exposure / gain were left alone
    assert stream._settings["exposure_time"] == 19e-6


def test_rpc_handler_never_raises():
    fe, stream, cs = make_client()
    try:
        fe.scan_xvars = None        # force an internal error
        cs.request_snap_mu(0, 0.0)  # must not raise
        assert cs.counts["rpc_errors"] == 1
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
    assert fe.data.img_test_meta._run_data.shape == (4, N_META)
    assert np.isnan(fe.data.img_test_meta._run_data).all()
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


def test_no_saved_roi_downsample_fallback():
    from waxx.control.cameras.camera_stream_client import block_reduce
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    r = block_reduce(img, 2)
    assert r.shape == (5, 5) and r.dtype == np.uint8 and r[0, 0] == 6   # mean(0,1,10,11)=5.5 -> 6
    assert block_reduce(img, 1) is img
    assert block_reduce(np.ones((3, 3), np.uint8), 4).shape == (3, 3)  # too small: untouched
    # a stream whose server has no saved ROI, asked to downsample 2x
    no_roi = {"exposure_time": 19e-6, "gain": 0.0, "trigger_mode": "Off"}
    fe, stream, cs = make_client(defaults=no_roi, no_roi_downsample=2)
    fe.data.init()
    try:
        assert cs._downsample == 2 and cs._frame_shape == (5, 6)
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        assert fe.data.img_test._run_data.shape == (4, 5, 6)
        assert np.all(F(fe, "img_test")[0] == 7)
    finally:
        cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_test"])
    assert rec["downsample"] == 2 and rec["sensor_frame_shape"] == [10, 12]
    # with a saved ROI the option does nothing
    fe2, stream2, cs2 = make_client(no_roi_downsample=2)
    try:
        assert cs2._downsample == 1 and cs2._frame_shape == (6, 5)
    finally:
        cs2.finish()


def test_roi_override_beats_the_saved_roi():
    fe, stream, cs = make_client(roi=[1, 1, 4, 5])      # saved roi is [2,3,7,9]
    try:
        assert cs._roi == [1, 1, 4, 5] and cs._frame_shape == (4, 3)
    finally:
        cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_test"])
    assert rec["roi"] == [1, 1, 4, 5] and rec["overrides"]["roi"] == [1, 1, 4, 5]


def test_crop_to_roi_edge_cases():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert crop_to_roi(img, None).shape == (10, 10)
    assert crop_to_roi(img, [5, 5, 5, 9]).shape == (10, 10)   # degenerate
    assert crop_to_roi(img, [-5, -5, 50, 50]).shape == (10, 10)  # clamped
    c = crop_to_roi(img, [2, 3, 7, 9])
    assert c.shape == (6, 5) and c[0, 0] == 32


# ---------------------------------------------------------------------------
# Triggered mode
# ---------------------------------------------------------------------------

class TrigFrame:
    def __init__(self, seq, hw_idx, rev, shape=(10, 12), value=9, mode="On",
                 acq_gen=1):
        self.image = np.full(shape, value, dtype=np.uint8)
        self.seq = seq
        self.hw_idx = hw_idx
        self.hw_ts = 1000 * seq
        self.acq_gen = acq_gen
        self.t_mono = time.monotonic()
        self.settings_rev = rev
        self.settings = {"trigger_mode": mode, "exposure_time": 19e-6,
                         "gain": 0.0}


class TrigFakeStream(FakeStream):
    """FakeStream plus the live-stream surface a triggered client uses.
    ``edge()`` stands in for a TTL edge reaching the camera."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._settings.setdefault("trigger_source", "Line1")
        self.live = False
        self.live_calls = []
        self.instance = "inst-a"
        self._frames = []
        self._flock = threading.Lock()
        self._hw_idx = 0

    def start_live(self):
        self.live = True
        self.live_calls.append("start")
        return {}

    def stop_live(self):
        self.live = False
        self.live_calls.append("stop")
        return {}

    def status(self, refresh=True):
        return SimpleNamespace(state="streaming" if self.live else "idle",
                               instance=self.instance,
                               raw={"seq": self._seq})

    def edge(self, skip=0, **kw):
        """One triggered frame; ``skip`` frames before it were produced and
        never delivered (conflated). The frame carries the stream's current
        exposure / gain, as a real camera's does."""
        with self._flock:
            self._seq += 1 + skip
            self._hw_idx += 1 + skip
            f = TrigFrame(self._seq, self._hw_idx, kw.pop("rev", self.rev),
                          shape=self.shape, **kw)
            f.settings["exposure_time"] = self._settings.get("exposure_time")
            f.settings["gain"] = self._settings.get("gain")
            self._frames.append(f)

    def latest(self, after_seq=None, timeout=2.0, sources=("live",),
               include_run=False):
        deadline = time.monotonic() + timeout
        while True:
            with self._flock:
                if self._frames:
                    return self._frames.pop(0)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)


class FakeTTL:
    key = "mot_basler_trigger"
    ch = 51


def make_triggered(fe=None, stream=None, **kw):
    from waxx.control.cameras.camera_stream_client import (
        TriggeredCameraStreamClient)
    fe = fe or FakeExpt()
    stream = stream or TrigFakeStream()
    kw.setdefault("t_match_window", 0.6)
    kw.setdefault("t_stray_grace", 0.1)
    cs = TriggeredCameraStreamClient(
        fe, "12345", "img_test", ttl=FakeTTL(), trigger_source="Line2",
        exposure_delay=17e-6, _stream=stream, _server_defaults=DEFAULTS, **kw)
    return fe, stream, cs


def make_triggered_keys(keys, fe=None, stream=None, **kw):
    from waxx.control.cameras.camera_stream_client import (
        TriggeredCameraStreamClient)
    fe = fe or FakeExpt()
    stream = stream or TrigFakeStream()
    kw.setdefault("t_match_window", 0.6)
    kw.setdefault("t_stray_grace", 0.1)
    cs = TriggeredCameraStreamClient(
        fe, "12345", keys, ttl=FakeTTL(), trigger_source="Line2",
        _stream=stream, _server_defaults=DEFAULTS, **kw)
    return fe, stream, cs


def wait_settled(cs, timeout=5.0):
    deadline = time.monotonic() + timeout
    while cs._inflight() and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    cs._expt.drain_shot_data(timeout)       # and the shot queue delivered it


def test_triggered_construction_leaves_camera_waiting_for_trigger():
    fe, stream, cs = make_triggered()
    try:
        assert stream._settings["trigger_mode"] == "On"
        assert stream._settings["trigger_source"] == "Line2"
        assert stream.live and stream.live_calls == ["start"]
        assert cs._rev == stream.rev
        for key in ("img_test", "img_test_meta"):
            assert isinstance(getattr(fe.data, key), HostDataContainer)
        assert cs._frame_shape == (6, 5)
    finally:
        cs.finish()


def test_triggered_construction_fails_when_not_streaming():
    stream = TrigFakeStream()
    stream.start_live = lambda: {}          # the stream never starts
    from waxx.control.cameras import camera_stream_client as m
    fe = FakeExpt()
    with pytest.raises(RuntimeError, match="did not start"):
        m.TriggeredCameraStreamClient(
            fe, "12345", "img_test", ttl=FakeTTL(), trigger_source="Line2",
            _stream=stream, _server_defaults=DEFAULTS)
    assert stream.closed


def test_triggered_frames_land_at_their_shot():
    fe, stream, cs = make_triggered()
    fe.data.init()
    try:
        for shot in (0, 2):
            fe.scan_xvars[0].counter = shot
            cs.announce_trigger_mu(0, 0)
            stream.edge()
            wait_settled(cs)
        seq = M(fe, "img_test", "seq")
        assert seq[0] > 0 and np.isnan(seq[1]) and seq[2] > seq[0]
        assert np.all(F(fe, "img_test")[0] == 9)
        assert np.all(F(fe, "img_test")[1] == 0)
        hw = M(fe, "img_test", "hw_idx")
        assert hw[2] == hw[0] + 1
        assert M(fe, "img_test", "hw_ts")[0] > 0
        assert np.isfinite(M(fe, "img_test", "t_target")[0])
        # the settings each frame was taken with, NaN where there is none
        assert M(fe, "img_test", "exposure")[0] == 19e-6
        assert M(fe, "img_test", "gain")[0] == 0.0
        assert np.isnan(M(fe, "img_test", "exposure")[1])
        assert cs.counts["ok"] == 2 and cs._n_missing() == 0
    finally:
        cs.finish()


def test_triggered_missing_frame_is_recorded_not_filled():
    fe, stream, cs = make_triggered()
    fe.data.init()
    try:
        cs.announce_trigger_mu(0, 0)          # no edge reaches the camera
        wait_settled(cs)
        assert np.isnan(M(fe, "img_test", "seq")[0])
        assert cs._n_missing() == 1
        assert any(e["event"] == "no_frame" for e in cs.events)
    finally:
        cs.finish()


def test_triggered_stray_frame_is_not_a_shot():
    fe, stream, cs = make_triggered()
    fe.data.init()
    try:
        stream.edge()                      # an edge nobody announced
        time.sleep(0.5)
        assert cs.counts["stray_frames"] == 1
        assert np.isnan(M(fe, "img_test", "seq")).all()
        # the next announced trigger still pairs with its own frame
        cs.announce_trigger_mu(0, 0)
        stream.edge()
        wait_settled(cs)
        assert M(fe, "img_test", "seq")[0] > 0
    finally:
        cs.finish()


def test_triggered_counter_gap_marks_the_shot_missing():
    fe, stream, cs = make_triggered()
    fe.data.init()
    try:
        cs.announce_trigger_mu(0, 0)
        stream.edge()
        wait_settled(cs)
        fe.scan_xvars[0].counter = 1
        cs.announce_trigger_mu(0, 0)
        stream.edge(skip=1)                # one frame was never delivered
        wait_settled(cs)
        seq = M(fe, "img_test", "seq")
        assert seq[0] > 0 and np.isnan(seq[1])
        assert cs.counts["gaps"] == 1
        # contiguous again afterwards
        fe.scan_xvars[0].counter = 2
        cs.announce_trigger_mu(0, 0)
        stream.edge()
        wait_settled(cs)
        assert M(fe, "img_test", "seq")[2] > 0
    finally:
        cs.finish()


def test_triggered_free_run_frame_never_stored():
    fe, stream, cs = make_triggered()
    fe.data.init()
    try:
        cs.announce_trigger_mu(0, 0)
        stream.edge(mode="Off")
        wait_settled(cs)
        assert np.isnan(M(fe, "img_test", "seq")[0])
        assert cs.counts["stale_frames"] == 1
    finally:
        cs.finish()


def test_triggered_settings_change_fails_shot_and_reasserts():
    fe, stream, cs = make_triggered()
    fe.data.init()
    try:
        stream.rev += 3                    # a viewer changed something
        cs.announce_trigger_mu(0, 0)
        stream.edge()
        wait_settled(cs)
        assert np.isnan(M(fe, "img_test", "seq")[0])
        assert any(e["event"] == "settings_changed" for e in cs.events)
        assert cs._rev == stream.rev       # trigger mode put back, new rev
        fe.scan_xvars[0].counter = 1
        cs.announce_trigger_mu(0, 0)
        stream.edge()
        wait_settled(cs)
        assert M(fe, "img_test", "seq")[1] > 0
    finally:
        cs.finish()


def test_triggered_two_keys_two_frames_per_shot():
    fe, stream, cs = make_triggered_keys(["img_a", "img_b"])
    fe.data.init()
    try:
        assert cs.keys == ("img_a", "img_b")
        for k in ("img_a", "img_b"):
            assert fe.data.__dict__[k]._run_data.shape == (4, 6, 5)
            assert fe.data.__dict__[k + "_meta"]._run_data.shape == (4, N_META)
        assert set(fe.data.keys) == {"img_a", "img_a_meta", "img_b",
                                     "img_b_meta"}
        fe.scan_xvars[0].counter = 1
        cs.announce_trigger_mu(0, 0)
        stream.edge(value=1)
        wait_settled(cs)
        cs.announce_trigger_mu(0, 1)
        stream.edge(value=2)
        wait_settled(cs)
        assert np.all(F(fe, "img_a")[1] == 1)
        assert np.all(F(fe, "img_b")[1] == 2)
        assert M(fe, "img_b", "seq")[1] == M(fe, "img_a", "seq")[1] + 1
        assert np.isnan(M(fe, "img_a", "seq")[0])
        assert cs._n_missing() == 0
        # a bad frame index: the edge is expected (the TTL pulsed), not kept
        cs.announce_trigger_mu(0, 7)
        stream.edge(value=3)
        wait_settled(cs)
        assert cs.counts["bad_frame_index"] == 1
        assert np.all(F(fe, "img_a")[1] == 1)
        assert np.all(F(fe, "img_b")[1] == 2)
        # a second announcement of the same frame slot (a warm-up shot)
        # clears only that slot
        cs.announce_trigger_mu(0, 0)
        wait_settled(cs)       # no edge: frame 0 ends up missing
        assert np.isnan(M(fe, "img_a", "seq")[1])
        assert M(fe, "img_b", "seq")[1] > 0
        assert cs.counts["duplicate"] == 1 and cs._n_missing() == 1
    finally:
        cs.finish()
    rec_a = json.loads(fe._extra_file_texts["camera_stream_img_a"])
    rec_b = json.loads(fe._extra_file_texts["camera_stream_img_b"])
    assert rec_a["keys"] == ["img_a", "img_b"] and rec_a["frame"] == 0
    assert rec_b["key"] == "img_b" and rec_b["frame"] == 1


def test_triggered_pairing_follows_expected_time_not_announcement_order():
    """The kernel may announce a later-timeline edge first (delay(-x))."""
    fe, stream, cs = make_triggered_keys(["img_late", "img_early"])
    fe.data.init()
    try:
        # frame 0 is announced first but is 0.3 s later on the timeline
        cs.announce_trigger_mu(int(0.3e9), 0)
        cs.announce_trigger_mu(0, 1)
        stream.edge(value=5)             # the early one exposes first
        time.sleep(0.3)
        stream.edge(value=6)
        wait_settled(cs)
        assert np.all(F(fe, "img_early")[0] == 5)
        assert np.all(F(fe, "img_late")[0] == 6)
        assert cs._n_missing() == 0
    finally:
        cs.finish()


def test_siblings_on_a_shared_line_discard_each_others_frames():
    from waxx.control.cameras.camera_stream_client import link_trigger_line
    fe = FakeExpt()
    sa, sb = TrigFakeStream(), TrigFakeStream()
    sb.camera_id = "basler_usb:67890"
    _, _, a = make_triggered_keys(["img_a"], fe=fe, stream=sa)
    _, _, b = make_triggered_keys(["img_b"], fe=fe, stream=sb)
    link_trigger_line(a, b)
    fe.data.init()
    try:
        # an edge fired for A exposes both cameras
        a.announce_trigger_mu(0, 0)
        sa.edge(value=1)
        sb.edge(value=1)
        wait_settled(a)
        wait_settled(b)
        fe.scan_xvars[0].counter = 1
        b.announce_trigger_mu(0, 0)
        sa.edge(value=2)
        sb.edge(value=2)
        wait_settled(a)
        wait_settled(b)
        assert np.all(F(fe, "img_a")[0] == 1)
        assert np.isnan(M(fe, "img_a", "seq")[1])
        assert np.isnan(M(fe, "img_b", "seq")[0])
        assert np.all(F(fe, "img_b")[1] == 2)
        assert a.counts["sibling_frames_discarded"] == 1
        assert b.counts["sibling_frames_discarded"] == 1
        assert a._n_missing() == 0 and b._n_missing() == 0
        assert not a.counts.get("stray_frames")
        assert not b.counts.get("gaps")
        # a sibling edge the camera never saw costs nothing of ours
        fe.scan_xvars[0].counter = 2
        a.announce_trigger_mu(0, 0)
        sa.edge(value=3)                 # B misses its (unwanted) frame
        wait_settled(a)
        wait_settled(b)
        assert np.all(F(fe, "img_a")[2] == 3)
        assert b._n_missing() == 0 and b.counts["failed"] == 0
        assert b.counts.get("sibling_frame_no_frame") == 1
    finally:
        a.finish()
        b.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_a"])
    assert rec["siblings_on_line"] == [["img_b"]]


def test_inactive_stream_has_no_containers_and_is_not_registered():
    from waxx.control.cameras.camera_stream_client import (
        InactiveCameraStream)
    fe = FakeExpt()
    cs = InactiveCameraStream(fe, ["img_x", "img_y"], reason="off")
    fe.data.init()
    assert fe.camera_streams == []
    assert fe.data.keys == []
    assert cs.keys == ("img_x", "img_y") and cs.key == "img_x"
    cs.finish()
    assert fe._extra_file_texts == {}


def test_finish_joins_the_worker_before_closing():
    """The stream is closed only after the worker left its last request
    (run 84562 hung in zmq term() on a socket a racing worker had made)."""
    order = []
    fe, stream, cs = make_triggered()
    orig_close = stream.close
    stream.close = lambda: (order.append(("close", cs._worker.is_alive())),
                            orig_close())
    fe.data.init()
    cs.finish()
    assert order == [("close", False)]
    assert not cs._worker.is_alive()


def test_triggered_finish_restores_free_run_and_records_mode():
    fe, stream, cs = make_triggered()
    fe.data.init()
    cs.announce_trigger_mu(0, 0)
    stream.edge()
    wait_settled(cs)
    cs.finish()
    assert stream.live_calls == ["start", "stop"]
    assert stream._settings["trigger_mode"] == "Off"
    assert stream._settings["trigger_source"] == "Line1"
    assert stream.closed
    rec = json.loads(fe._extra_file_texts["camera_stream_img_test"])
    assert rec["mode"] == "triggered"
    assert rec["trigger_ttl"] == "mot_basler_trigger"
    assert rec["trigger_ttl_ch"] == 51
    assert rec["keys"] == ["img_test"] and rec["frame"] == 0
    assert rec["siblings_on_line"] == []
    assert rec["n_shots_missing_frame"] == 0


def test_leftover_trigger_mode_is_cleared_before_the_probe():
    stream = FakeStream(settings={"exposure_time": 19e-6, "gain": 0.0,
                                  "trigger_mode": "On"})
    fe, stream, cs = make_client(stream=stream)
    try:
        assert stream.set_calls[0] == {"trigger_mode": "Off"}
        assert cs._priors == {"trigger_mode": "On"}
    finally:
        cs.finish()


def test_wrong_mode_calls_are_counted_never_raised():
    fe, stream, cs = make_triggered()
    try:
        cs.note_wrong_mode_call()
        assert cs.counts["wrong_mode_calls"] == 1
    finally:
        cs.finish()


def test_per_frame_settings_applied_before_the_edge_and_recorded():
    fe, stream, cs = make_triggered_keys(
        ["img_dim", "img_bright"],
        frame_settings={"img_bright": {"exposure_time": 2e-3, "gain": 20.}})
    fe.data.init()
    try:
        assert cs._base_settings == {"exposure_time": 19e-6, "gain": 0.0}
        n_sets = len(stream.set_calls)
        # frame 0: the stream's own settings, nothing to apply
        cs.announce_trigger_mu(int(0.3e9), 0)      # edge 0.3 s ahead
        time.sleep(0.15)
        assert len(stream.set_calls) == n_sets
        stream.edge(value=1)
        wait_settled(cs)
        # frame 1 announced 0.3 s ahead: its settings go in before the edge
        cs.announce_trigger_mu(int(0.3e9), 1)
        time.sleep(0.15)
        assert stream.set_calls[-1] == {"exposure_time": 2e-3, "gain": 20.}
        assert cs._rev == stream.rev
        stream.edge(value=2)
        wait_settled(cs)
        assert np.all(F(fe, "img_bright")[0] == 2)
        assert M(fe, "img_bright", "exposure")[0] == 2e-3
        assert M(fe, "img_bright", "gain")[0] == 20.
        assert M(fe, "img_dim", "exposure")[0] == 19e-6
        assert cs.counts["frame_settings_applied"] >= 1
        # next shot, frame 0 again: back to the stream's own settings
        fe.scan_xvars[0].counter = 1
        cs.announce_trigger_mu(int(0.3e9), 0)
        time.sleep(0.15)
        assert stream.set_calls[-1] == {"exposure_time": 19e-6, "gain": 0.0}
        stream.edge(value=3)
        wait_settled(cs)
        assert M(fe, "img_dim", "exposure")[1] == 19e-6
        assert cs._n_missing() == 0
    finally:
        cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_bright"])
    assert rec["frame_settings"] == {"img_bright": {"exposure_time": 2e-3,
                                                    "gain": 20.}}


def test_next_frames_settings_go_in_as_soon_as_the_previous_frame_is_in():
    """Frames are listed in sequence order: after frame 0 arrived, frame 1's
    settings are set at once, so an edge announced with little slack still
    gets them."""
    fe, stream, cs = make_triggered_keys(
        ["img_dim", "img_bright"],
        frame_settings={"img_bright": {"exposure_time": 2e-3, "gain": 20.}})
    fe.data.init()
    try:
        cs.announce_trigger_mu(int(0.1e9), 0)
        stream.edge(value=1)
        wait_settled(cs)
        time.sleep(0.2)
        assert stream.set_calls[-1] == {"exposure_time": 2e-3, "gain": 20.}
        # announced with no slack at all: still taken at its own settings
        cs.announce_trigger_mu(0, 1)
        stream.edge(value=2)
        wait_settled(cs)
        assert M(fe, "img_bright", "exposure")[0] == 2e-3
        assert not cs.counts.get("frame_settings_late")
        # and the camera goes back to frame 0's settings for the next shot
        time.sleep(0.2)
        assert stream.set_calls[-1] == {"exposure_time": 19e-6, "gain": 0.0}
    finally:
        cs.finish()


def test_a_run_that_takes_only_the_first_frame_learns_its_settings():
    """Only frame 0 is requested each shot (an experiment that stops before
    the frame-1 stage): frame 0's settings go in before the first edge, and
    after one repeat the worker knows frame 0 follows frame 0."""
    fe, stream, cs = make_triggered_keys(
        ["img_dim", "img_bright"],
        frame_settings={"img_dim": {"exposure_time": 2e-3, "gain": 20.}})
    fe.data.init()
    try:
        time.sleep(0.2)                            # before any edge
        assert stream.set_calls[-1] == {"exposure_time": 2e-3, "gain": 20.}
        cs.announce_trigger_mu(0, 0)
        stream.edge(value=1)
        wait_settled(cs)
        assert M(fe, "img_dim", "exposure")[0] == 2e-3
        time.sleep(0.2)                            # guesses frame 1 next
        assert stream.set_calls[-1] == {"exposure_time": 19e-6, "gain": 0.0}
        fe.scan_xvars[0].counter = 1
        cs.announce_trigger_mu(0, 0)               # frame 0 again, no slack
        stream.edge(value=2)
        wait_settled(cs)
        assert cs.counts["frame_settings_late"] == 1
        time.sleep(0.2)                            # learned: 0 follows 0
        assert stream.set_calls[-1] == {"exposure_time": 2e-3, "gain": 20.}
        fe.scan_xvars[0].counter = 2
        cs.announce_trigger_mu(0, 0)
        stream.edge(value=3)
        wait_settled(cs)
        assert M(fe, "img_dim", "exposure")[2] == 2e-3
        assert cs.counts["frame_settings_late"] == 1
    finally:
        cs.finish()


def test_per_frame_settings_too_late_frame_kept_and_flagged():
    fe, stream, cs = make_triggered_keys(
        ["img_dim", "img_bright"],
        frame_settings={1: {"exposure_time": 2e-3, "gain": 20.}})
    fe.data.init()
    try:
        n_sets = len(stream.set_calls)
        cs.announce_trigger_mu(0, 1)               # edge now: no time
        stream.edge(value=4)
        wait_settled(cs)
        assert len(stream.set_calls) == n_sets     # nothing applied
        assert np.all(F(fe, "img_bright")[0] == 4)
        assert M(fe, "img_bright", "exposure")[0] == 19e-6   # the truth
        assert cs.counts["frame_settings_late"] == 1
    finally:
        cs.finish()


def test_per_frame_settings_refuse_bad_frame_or_key():
    with pytest.raises(ValueError, match="no such frame"):
        make_triggered_keys(["img_a"], frame_settings={3: {"gain": 1.}})
    with pytest.raises(ValueError, match="cannot change between frames"):
        make_triggered_keys(["img_a"],
                            frame_settings={0: {"trigger_mode": "Off"}})


def test_dummy_for_triggered_keys_has_one_record_per_key():
    from waxx.control.cameras.camera_stream_client import (
        DummyCameraStreamClient)
    fe = FakeExpt()
    cs = DummyCameraStreamClient(fe, ["img_test", "img_other"],
                                 reason="run camera", triggered=True)
    fe.data.init()
    assert set(fe.data.keys) == {"img_test", "img_test_meta", "img_other",
                                 "img_other_meta"}
    assert np.isnan(fe.data.img_other_meta._run_data).all()
    cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_other"])
    assert rec["dummy"] and rec["triggered"] and rec["keys"] == [
        "img_test", "img_other"]
