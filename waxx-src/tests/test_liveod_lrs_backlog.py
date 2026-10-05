"""Long-run stability, phase C (2026-10-05): what liveOD does per shot and per
second of uptime that it need not.

* pushed arrays: the broadcast carries a small notice per PUT_DATA (no arrays),
  sent from the server thread; GET_AUX_DATA still has the latest array per key;
* the image writers' backlog (items and bytes queued, not yet written) is in
  POLL, with a WARNING past a threshold and nothing ever dropped;
* the live scalar plot redraws at most twice a second, whatever the shot rate;
* the next-run-id label is read off the GUI thread, one read at a time;
* the image dispatcher blocks on its queue instead of waking every millisecond.

Everything runs against fakes in pytest's tmp_path: no socket is bound, no
beacon is sent, no camera is opened.
"""
import logging
import pickle
import queue
import threading
import time

import h5py
import numpy as np
import pytest
import liveod_qt_helpers as qt
from live_od_data_fakes import FakeSaver, patch_payload_stash


@pytest.fixture
def app():
    return qt.session_app()


@pytest.fixture
def log_records():
    """(levelno, message) of every liveOD record at INFO and above."""
    got = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            got.append((record.levelno, record.getMessage()))
    handler = ListHandler(logging.INFO)
    log = logging.getLogger("waxx.live_od")
    old_level = log.level
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    yield got
    log.removeHandler(handler)
    log.setLevel(old_level)


def _spin(app, cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    return cond()


# ----------------------------------------------------------------------
# C1: pushed arrays -> a notice on the broadcast, the arrays on request
# ----------------------------------------------------------------------

@pytest.fixture
def server(app, tmp_path, monkeypatch):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    srv = LiveODServer(server_talk=None, data_saver=FakeSaver(tmp_path))   # never started
    yield srv
    srv._run_file.finish_writer()
    w = srv._run_file.writer
    if w is not None:
        w.wait_done(5)


def _init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 2, "expt_class": "lrs_test"}
    msg.update(kw)
    return msg


def _put_msg(token, items):
    specs, buffers = [], []
    for key, index, arr, full_shape, fill in items:
        arr = np.ascontiguousarray(arr)
        specs.append({"key": key, "index": index, "shape": list(arr.shape),
                      "dtype": arr.dtype.str, "full_shape": full_shape, "fill": fill})
        buffers.append(arr.tobytes())
    return {"tag": "PUT_DATA", "run_token": token, "items": specs}, buffers


def _no_arrays(obj) -> bool:
    if isinstance(obj, np.ndarray):
        return False
    if isinstance(obj, dict):
        return all(_no_arrays(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_no_arrays(v) for v in obj)
    return True


@pytest.mark.parametrize("save_data", [True, False])
def test_a_push_sends_one_notice_without_arrays_from_the_server_thread(server, save_data):
    from waxx.util.live_od.live_od_server import LiveODServer
    srv = server
    # the per-item GUI-thread signal is gone
    assert not hasattr(LiveODServer, "aux_data_signal")
    notices, threads = [], []
    srv.set_aux_notice_sink(lambda n: (notices.append(n), threads.append(threading.get_ident())))
    init = srv._handle_init_run(_init_msg(save_data=save_data))
    token = init["run_token"]
    frame = np.arange(4 * 6, dtype=np.uint16).reshape(4, 6)
    meta = np.array([1.0, 2.0, 3.0])
    msg, bufs = _put_msg(token, [("img_a", [1], frame, [2, 4, 6], 0),
                                 ("img_a_meta", [1], meta, [2, 3], float("nan"))])
    reply = srv._handle_put_data(msg, bufs)
    assert reply["ok"] and reply["written"] is save_data

    assert len(notices) == 1                       # one notice per PUT_DATA, not per item
    assert threads == [threading.get_ident()]      # called right here, no hop to the GUI thread
    note = notices[0]
    assert _no_arrays(note)
    assert note["run_id"] == init["run_id"] and note["t"] > 0
    assert note["items"] == [
        {"key": "img_a", "index": [1], "offset": None, "shape": [4, 6],
         "dtype": frame.dtype.str, "nbytes": frame.nbytes},
        {"key": "img_a_meta", "index": [1], "offset": None, "shape": [3],
         "dtype": meta.dtype.str, "nbytes": meta.nbytes},
    ]
    # the arrays themselves: the latest per key, on request
    latest = srv._handle_get_aux_data({})["items"]
    assert np.array_equal(latest["img_a"]["array"], frame)
    assert np.array_equal(latest["img_a_meta"]["array"], meta)
    assert srv._handle_end_run({"run_token": token})["ok"]


def test_a_slice_of_a_whole_array_gets_no_notice(server):
    srv = server
    notices = []
    srv.set_aux_notice_sink(notices.append)
    init = srv._handle_init_run(_init_msg())
    msg, bufs = _put_msg(init["run_token"], [("big", None, np.zeros((2, 3), np.uint8), [4, 3], 0)])
    msg["items"][0]["offset"] = 0
    assert srv._handle_put_data(msg, bufs)["written"]
    assert notices == [] and srv._handle_get_aux_data({})["items"] == {}
    assert srv._handle_end_run({"run_token": init["run_token"]})["ok"]


def test_a_failing_notice_sink_does_not_fail_the_push(server):
    srv = server

    def broken(_notice):
        raise RuntimeError("broadcaster gone")
    srv.set_aux_notice_sink(broken)
    init = srv._handle_init_run(_init_msg(save_data=False))
    msg, bufs = _put_msg(init["run_token"], [("img_a", [0], np.ones(3, np.uint8), [2, 3], 0)])
    assert srv._handle_put_data(msg, bufs)["ok"]
    assert "img_a" in srv._handle_get_aux_data({})["items"]


def test_the_broadcaster_queues_a_small_aux_notice(app):
    from waxx.util.live_od.live_od_broadcaster import LiveODBroadcaster
    b = LiveODBroadcaster()                         # never started: nothing bound, no beacon
    assert not hasattr(b, "broadcast_aux_data")
    notice = {"run_id": 7, "t": 1.0,
              "items": [{"key": f"img_{i}", "index": [3], "offset": None, "shape": [1200, 1920],
                         "dtype": "<u2", "nbytes": 1200 * 1920 * 2} for i in range(9)]}
    b.broadcast_aux_notice(notice)
    msg = b._queue.get_nowait()
    assert msg["tag"] == "AUX_NOTICE" and msg["run_id"] == 7 and len(msg["items"]) == 9
    assert len(pickle.dumps(msg)) < 4096           # nine 4.6 MB frames, announced in < 4 kB


# ----------------------------------------------------------------------
# C2: the writers' backlog, visible and loud, never dropped
# ----------------------------------------------------------------------

def test_the_gauge_warns_once_per_crossing_and_says_when_it_drains(log_records):
    from waxx.util.live_od.data.image_writer import QueueGauge
    g = QueueGauge(warn_bytes=1000)
    frames = [(np.zeros(300, np.uint8), i, 0.0) for i in range(6)]
    for f in frames[:3]:
        g.added(f)
    assert g.snapshot()["bytes"] == 900 and not [r for r in log_records if r[0] >= logging.WARNING]
    g.added(frames[3])                              # 1200 >= 1000: one warning
    g.added(frames[4])                              # still above: no second one
    warnings = [m for lvl, m in log_records if lvl == logging.WARNING]
    assert len(warnings) == 1 and "backlog" in warnings[0] and "Nothing is dropped" in warnings[0]
    for f in frames[:3]:
        g.taken(f)                                  # 600: above half, no all-clear yet
    assert not [m for lvl, m in log_records if lvl == logging.INFO]
    g.taken(frames[3])                              # 300 < 500: all clear
    infos = [m for lvl, m in log_records if lvl == logging.INFO]
    assert len(infos) == 1 and "down to" in infos[0]
    snap = g.snapshot()
    assert snap == {"items": 1, "bytes": 300, "peak_bytes": 1500, "warn_bytes": 1000}
    for f in frames[:4]:
        g.added(f)                                  # crosses again: warns again
    assert len([m for lvl, m in log_records if lvl == logging.WARNING]) == 2


def test_the_writer_counts_what_it_holds_until_written(app, tmp_path):
    from waxx.util.live_od.data.image_writer import ImageWriter, PutItem, QueueGauge
    path = str(tmp_path / "w.hdf5")
    with h5py.File(path, "x") as f:
        f.create_group("data")
    gauge = QueueGauge(warn_bytes=10**9)
    w = ImageWriter(path, server_owned=True, gauge=gauge)
    # the drive is slow: the worker cannot open the file until released
    release = threading.Event()
    real_wait = w.wait_for_data_available

    def slow_wait(*a, **k):
        release.wait(10)
        return real_wait(*a, **k)
    w.wait_for_data_available = slow_wait
    done = threading.Event()
    w.start_for_server(n_img=2, on_done=done.set)
    frames = [np.full((3, 4), i + 1, np.uint16) for i in range(2)]
    w.put(frames[0], 0, 1.0)
    w.put(frames[1], 1, 2.0)
    assert w.put_data(PutItem("aux", (0,), np.arange(5, dtype=np.float64), full_shape=(2, 5)))
    snap = gauge.snapshot()
    assert snap["items"] == 3 and snap["bytes"] == 2 * frames[0].nbytes + 5 * 8
    release.set()
    w.finish()
    assert done.wait(10)
    assert gauge.snapshot()["items"] == 0 and gauge.snapshot()["bytes"] == 0
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["data/images"][1], frames[1])
        assert np.array_equal(f["data/aux"][0], np.arange(5))


def test_a_frame_after_finish_is_reported_not_written_and_not_left_queued(app, tmp_path):
    from waxx.util.live_od.data.image_writer import ImageWriter, QueueGauge
    path = str(tmp_path / "w.hdf5")
    with h5py.File(path, "x") as f:
        f.create_group("data")
    gauge = QueueGauge(warn_bytes=10**9)
    w = ImageWriter(path, server_owned=True, gauge=gauge)
    done = threading.Event()
    w.start_for_server(n_img=2, on_done=done.set)
    w.report = w._worker.report                    # what a DataHandler's start registers
    w.put(np.zeros((2, 2), np.uint16), 0, 0.0)
    w.finish()
    w.put(np.ones((2, 2), np.uint16), 1, 0.0)     # too late
    assert done.wait(10)
    written, not_written, _ = w.report.snapshot()
    assert written == 1
    assert len(not_written) == 1 and "after the writer was finished" in not_written[0]
    assert gauge.snapshot()["items"] == 0


def test_poll_reports_the_writer_backlog(server):
    reply = server._handle_poll({})
    assert set(reply["writer_queue"]) == {"items", "bytes", "peak_bytes", "warn_bytes"}
    assert reply["writer_queue"]["warn_bytes"] == 200 * 2**20


# ----------------------------------------------------------------------
# C3: the live scalar plot redraws at most twice a second
# ----------------------------------------------------------------------

def test_the_scalar_plot_coalesces_redraws(app, monkeypatch):
    from waxx.util.live_od.gui import live_scalar_plot_window as lsp
    calls = []
    real = lsp.LiveScalarPlotWindow._refresh_plot

    def counting(self, *a):
        calls.append(time.monotonic())
        return real(self, *a)
    monkeypatch.setattr(lsp.LiveScalarPlotWindow, "_refresh_plot", counting)
    win = lsp.LiveScalarPlotWindow()
    try:
        win.show()
        app.processEvents()
        win.on_new_run(1, ["t_tof"])
        win._last_refresh_t = time.monotonic()      # as if it had just redrawn
        calls.clear()
        for i in range(50):                         # a burst of shots
            win.on_shot_scalars({"shot_idx": i, "atom_number": 1e5 + i})
        assert calls == [] and win._refresh_timer.isActive()
        assert _spin(app, lambda: len(calls) >= 1, timeout=3.0)
        _spin(app, lambda: False, timeout=0.2)      # nothing more is pending
        assert len(calls) == 1
        assert len(win._scatter_item.getData()[0]) == 50
        # a shot long after the last redraw is drawn at once
        time.sleep(lsp.LiveScalarPlotWindow.REFRESH_MIN_INTERVAL_MS / 1000 + 0.05)
        win.on_shot_scalars({"shot_idx": 50, "atom_number": 2e5})
        assert len(calls) == 2 and len(win._scatter_item.getData()[0]) == 51
    finally:
        win._refresh_timer.stop()
        win.close()
        qt.delete_widgets(app, [win])


def test_the_scalar_plot_takes_the_last_n_without_copying_the_rest(app):
    from waxx.util.live_od.gui.live_scalar_plot_window import LiveScalarPlotWindow
    win = LiveScalarPlotWindow()
    try:
        for i in range(30):
            win._data.append({"shot_idx": i})
        assert [d["shot_idx"] for d in win._recent(5)] == [25, 26, 27, 28, 29]
        assert len(win._recent(100)) == 30
    finally:
        win.close()
        qt.delete_widgets(app, [win])


# ----------------------------------------------------------------------
# C4: the next run id is read off the GUI thread, one read at a time
# ----------------------------------------------------------------------

def test_background_poll_never_overlaps_and_delivers_the_result(app):
    from waxx.util.live_od.gui.main_window import BackgroundPoll
    gate = threading.Event()
    calls = []

    def slow_read():
        calls.append(threading.get_ident())
        gate.wait(10)                               # an SMB stall
        return 4242
    poll = BackgroundPoll(slow_read, name="lrs-test-poll")
    got = []
    poll.result.connect(got.append)
    t0 = time.monotonic()
    assert poll.poll() is True
    assert time.monotonic() - t0 < 0.5              # the caller is not held up
    assert _spin(app, lambda: len(calls) == 1, timeout=2.0)
    assert poll.poll() is False and poll.poll() is False and poll.skipped == 2
    assert calls[0] != threading.get_ident()
    gate.set()
    assert _spin(app, lambda: got == [4242])
    assert not poll.in_flight
    assert poll.poll() is True                      # free again
    assert _spin(app, lambda: got == [4242, 4242])
    assert len(calls) == 2


def test_background_poll_delivers_none_when_the_read_fails(app):
    from waxx.util.live_od.gui.main_window import BackgroundPoll

    def failing():
        raise OSError("network name no longer available")
    poll = BackgroundPoll(failing, name="lrs-test-poll")
    got = []
    poll.result.connect(got.append)
    assert poll.poll()
    assert _spin(app, lambda: got == [None])
    assert not poll.in_flight


# ----------------------------------------------------------------------
# C5: the image dispatcher blocks on its queue
# ----------------------------------------------------------------------

class _CountingQueue(queue.Queue):
    def __init__(self):
        super().__init__()
        self.gets = 0

    def get(self, *a, **k):
        self.gets += 1
        return super().get(*a, **k)


def _handler(monkeypatch, q, n_img=2):
    from waxx.util.live_od import config as live_od_config
    from waxx.util.live_od.camera_mother import DataHandler
    monkeypatch.setattr(live_od_config, "_active", live_od_config.LiveODConfig())
    h = DataHandler(q, "", save_data=False, imaging_type=0, n_img=n_img, n_shots=1,
                    n_pwa_per_shot=n_img)
    h.get_img_number(n_img, 1, n_img)
    return h


def test_the_dispatcher_waits_without_spinning_and_takes_frames_at_once(app, monkeypatch):
    q = _CountingQueue()
    h = _handler(monkeypatch, q)
    shown, done = [], threading.Event()
    h.got_image_from_queue.connect(lambda img: shown.append(time.monotonic()), )
    h.done_writing_signal.connect(done.set)
    h.start()
    try:
        time.sleep(0.5)
        assert q.gets <= 10                         # was ~500 with the 1 ms poll
        t_put = time.monotonic()
        q.put((np.zeros((2, 2), np.uint16), None, 0))
        assert _spin(app, lambda: len(shown) == 1, timeout=2.0)
        assert shown[0] - t_put < 0.1               # not held for the poll period
        q.put((np.zeros((2, 2), np.uint16), None, 1))
        assert _spin(app, done.is_set, timeout=2.0)
    finally:
        h.interrupted = True
        assert qt.join_or_keep(h, 5.0) == ""


@pytest.mark.parametrize("how", ["grab over", "interrupted"])
def test_the_dispatcher_stops_promptly(app, monkeypatch, how):
    q = queue.Queue()
    h = _handler(monkeypatch, q)
    done = threading.Event()
    h.done_writing_signal.connect(done.set)
    h.start()
    try:
        time.sleep(0.15)
        t0 = time.monotonic()
        if how == "grab over":
            h.grab_finished()
        else:
            h.interrupted = True
        assert h.wait(2000)
        from waxx.util.live_od.camera_mother import DISPATCH_POLL_S
        assert time.monotonic() - t0 < DISPATCH_POLL_S + 0.4
        assert _spin(app, done.is_set, timeout=2.0)
    finally:
        h.interrupted = True
        assert qt.join_or_keep(h, 5.0) == ""
