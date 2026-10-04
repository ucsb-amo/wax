"""FIX-B review findings on the liveOD window, built with a stand-in config and
never started (no socket, no beacon, no camera):

* m1  a camera thread of a replaced run cannot touch the new run, in the window
      between INIT_RUN (on the server's thread) and the queued spawn_baby (on
      the GUI thread) -- INIT_RUN runs on a real second thread here, so the
      new_run_signal really is queued;
* m4  shutdown closes the cameras before it waits for the image writer, inside
      the console handler's budget;
* m6  a stack-dump watchdog that cannot start does not stop the shutdown;
* m15 a camera thread stopped by the shutdown is told why;
* the live view: a camera nobody watches any more stops streaming.

Each test's window is torn down inside that test (liveod_qt_helpers). Every data
file lives in tmp_path.
"""
import logging
import os
import sys
import threading
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from live_od_data_fakes import FakeSaver, patch_payload_stash
import liveod_qt_helpers as qt


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def records():
    got = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            got.append(record.getMessage())
    handler = ListHandler(logging.WARNING)
    log = logging.getLogger("waxx.live_od")
    log.addHandler(handler)
    yield got
    log.removeHandler(handler)


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    import gc
    from waxx.util.live_od import config as live_od_config
    from waxx.util.live_od.data import run_file
    from waxx.util.live_od.gui import main_window as mw
    from waxx.util.live_od.gui import theme
    from waxx.util.live_od.gui.plotter import LiveODPlotter
    from waxx.util.live_od.live_od_broadcaster import LiveODBroadcaster
    from waxx.util.live_od.live_od_server import LiveODServer
    from PyQt6 import sip

    slot_errors = qt.SlotErrors()
    monkeypatch.setattr(sys, "excepthook", slot_errors)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    gc.collect()
    app.processEvents()
    n_before = len(slot_errors.seen)
    views_before = qt.pyqtgraph_views()
    threads_before = set(threading.enumerate())

    # never started: no socket, no beacon, no plotting thread
    for cls in (LiveODServer, LiveODBroadcaster, LiveODPlotter):
        monkeypatch.setattr(cls, "start", lambda self, *a: None)
    patch_payload_stash(monkeypatch, run_file, [])
    if theme._notifier is not None and sip.isdeleted(theme._notifier):
        monkeypatch.setattr(theme, "_notifier", None)
    spawned = []
    for name in ("CameraBaby", "DataHandler"):
        real = getattr(mw, name)
        monkeypatch.setattr(mw, name, lambda *a, _real=real, **k: spawned.append(_real(*a, **k)) or spawned[-1])

    cams = [types.SimpleNamespace(key=k, camera_type="fake", serial_no="") for k in ("cam_a", "cam_b")]
    config = live_od_config.LiveODConfig(
        data_saver=FakeSaver(tmp_path),
        run_id_source=types.SimpleNamespace(get_run_id=lambda: 1, update_run_id=lambda *a, **k: None,
                                            check_for_mapped_data_dir=lambda *a, **k: True),
        camera_params_list=cams)
    monkeypatch.setattr(live_od_config, "_active", None)
    win = mw.LiveODWindow(config, settings=None, log_dir=None)
    monkeypatch.setattr(sys, "excepthook", slot_errors)
    yield win

    problems = []
    for obj in spawned:
        if hasattr(obj, "request_stop"):
            obj.request_stop()
        if hasattr(obj, "grab_finished"):
            obj.grab_finished()
    # the run's writer is the server's and lives until END_RUN or shutdown:
    # finished here as shutdown does, so the dispatchers' join below sees it end
    win.live_od_server.wait_for_image_writer(0.0)
    for obj in spawned:
        problems.append(qt.join_or_keep(obj))
        writer = getattr(obj, "writer", None)
        if writer is not None and getattr(writer, "started", False):
            problems.append(qt.join_or_keep(writer._worker))
    win.shutdown("test teardown")
    win.run_id_timer.stop()
    win._camera_state_timer.stop()
    for t in qt.new_threads(threads_before):
        problems.append(qt.join_or_keep(t))
    qt.forget_views_since(views_before)
    qt.delete_widgets(app, [getattr(win, "live_scalar_plot_window", None),
                            getattr(win, "fk_tof_window", None),
                            getattr(win, "_adjust_panel", None), win])
    del win
    spawned.clear()
    qt.delete_widgets(app, [])
    problems = [p for p in problems if p]
    errors = slot_errors.seen[n_before:]
    assert not problems, problems
    assert not errors, f"exceptions in Qt slots while the window existed: {errors}"


class WaitingNanny:
    """The camera never becomes reachable: a camera thread waits in
    persistent_get_camera until its own break_check says stop."""

    def __init__(self):
        self.interrupted = False

    def persistent_get_camera(self, camera_params, break_check=None):
        while not break_check():
            time.sleep(0.01)
        from waxx.control.cameras import DummyCamera
        return DummyCamera()

    def update_params(self, camera, camera_params, report=None):
        return camera

    def get_camera(self, camera_params):
        from waxx.control.cameras import DummyCamera
        return DummyCamera()

    def close_all(self):
        return {}


def _camera_init(**kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": True, "camera_key": "cam_a",
           "camera_params": {"key": "cam_a", "camera_type": "fake"},
           "params": {"N_img": 3}, "N_shots_with_repeats": 1, "expt_class": "fixb_test"}
    msg.update(kw)
    return msg


def init_run_on_another_thread(srv, msg):
    """INIT_RUN as the server's own thread runs it: the window's slots are
    queued, so nothing of the new run reaches the GUI thread until it processes
    events. Returns the reply."""
    out = {}

    def run():
        out["reply"] = srv._handle_init_run(dict(msg))
    t = threading.Thread(target=run, name="fixb-server-thread")
    t.start()
    assert qt.join_or_keep(t) == ""
    return out["reply"]


def only_the_server_counts(window, handler):
    """The image dispatcher's display slots are not what these tests are about
    (and it has no image count before a grab starts): only the server's frame
    counter stays connected."""
    handler.got_image_from_queue.disconnect(window.analyzer.got_img)
    handler.got_image_from_queue.disconnect(window.count_images)


def emit_from_another_thread(fn):
    """A camera thread's emit, from a thread that is not the GUI thread (its
    DirectConnections run there, its queued ones wait for the GUI thread)."""
    t = threading.Thread(target=fn, name="fixb-camera-thread")
    t.start()
    assert qt.join_or_keep(t) == ""


# ----------------------------------------------------------------------
# m1: the queued-signal window
# ----------------------------------------------------------------------

def test_the_old_camera_thread_cannot_reach_a_run_the_window_has_not_spawned_yet(window, app, records):
    srv = window.live_od_server
    window.camera_nanny = WaitingNanny()

    first = init_run_on_another_thread(srv, _camera_init())
    app.processEvents()                                   # spawn_baby for run 1
    baby_a, handler_a = window._run_threads
    assert baby_a is not None and baby_a.isRunning()
    window.on_run_done()                                  # run 1 over for the window
    srv._run_in_progress = False
    only_the_server_counts(window, handler_a)

    second = init_run_on_another_thread(srv, _camera_init())
    assert second["run_token"] != first["run_token"]
    # new_run_signal is still queued: the window's latest camera run is run 1's
    assert window._run_threads == (baby_a, handler_a)

    # run 1's threads report now, in that window
    emit_from_another_thread(lambda: baby_a.cam_status_signal.emit(2))
    emit_from_another_thread(lambda: baby_a.grab_failed_signal.emit("camera timed out: late"))
    emit_from_another_thread(lambda: baby_a.camera_overrides_signal.emit("cam_a", {"gain": (300, 30)}))
    emit_from_another_thread(lambda: handler_a.got_image_from_queue.emit(np.zeros((2, 2))))
    assert not srv._cam_ready_event.is_set()
    assert srv._grab_failure == ""
    assert srv.camera_overrides_record() == {}
    assert srv._images_received_now() == 0 and srv._frame_times == []
    assert any("camera thread of an earlier run" in r for r in records)

    app.processEvents()                                   # spawn_baby for run 2
    baby_b = window._run_threads[0]
    assert baby_b is not baby_a and baby_b.run_token == second["run_token"]
    assert baby_a.run_token == first["run_token"]
    assert qt.join_or_keep(baby_a) == ""
    emit_from_another_thread(lambda: baby_b.cam_status_signal.emit(2))
    assert srv._cam_ready_event.is_set()
    handler_b = window._run_threads[1]
    only_the_server_counts(window, handler_b)
    emit_from_another_thread(lambda: handler_b.got_image_from_queue.emit(np.zeros((2, 2))))
    assert srv._images_received_now() == 1


def test_two_init_runs_before_the_window_catches_up(window, app):
    """Spawns come in INIT_RUN order, each with its own run's token: the thread
    spawned for the run that was replaced at once is not heard."""
    srv = window.live_od_server
    window.camera_nanny = WaitingNanny()
    spawned = []
    # connected after spawn_baby, so delivered after it: the thread it just made
    srv.new_run_signal.connect(lambda *a: spawned.append(window._run_threads[0]))
    r2 = init_run_on_another_thread(srv, _camera_init())
    r3 = init_run_on_another_thread(srv, _camera_init())
    app.processEvents()                                   # both spawns, in order
    baby_2, baby_3 = spawned
    assert baby_2.run_token == r2["run_token"] and baby_3.run_token == r3["run_token"]
    assert list(srv._spawn_tokens) == []                  # each spawn took its own
    assert qt.join_or_keep(baby_2) == ""                  # retired by the second spawn
    srv.on_cam_ready(run_token=baby_2.run_token)          # a report already in flight
    assert not srv._cam_ready_event.is_set()
    emit_from_another_thread(lambda: baby_3.cam_status_signal.emit(2))
    assert srv._cam_ready_event.is_set()


# ----------------------------------------------------------------------
# m4, m6, m15: shutdown
# ----------------------------------------------------------------------

class TimedCam:
    """An open camera that records when it was closed (the Andor's safe Close)."""

    def __init__(self):
        self.calls = []
        self.opened = True
        self.closed_at = None

    def is_opened(self):
        return self.opened

    def stop_grab(self):
        self.calls.append("stop_grab")

    def Close(self):
        self.closed_at = time.monotonic()
        self.calls.append("Close")
        self.opened = False


def slow_writer(record):
    """wait_for_image_writer that never sees the writer finish: it waits out its
    whole limit, as with a writer stuck on a slow data drive."""
    def wait(timeout):
        record.append(("start", time.monotonic(), timeout))
        time.sleep(timeout)
        record.append(("end", time.monotonic(), timeout))
        return False
    return wait


def test_the_cameras_are_closed_before_the_writer_is_waited_for(window, records):
    cam = TimedCam()
    window.camera_nanny.cam_a = cam
    writer = []
    window.live_od_server.wait_for_image_writer = slow_writer(writer)
    t0 = time.monotonic()
    window.shutdown("test")
    assert cam.calls == ["stop_grab", "Close"]
    (_, w_start, limit), (_, w_end, _) = writer
    assert cam.closed_at < w_end                          # not after the writer's wait
    assert cam.closed_at - t0 < 0.5                       # no camera thread to wait for
    assert any("image writer had not closed" in r for r in records)


def test_a_console_close_closes_the_camera_inside_its_budget(window):
    """The console handler's path: shutdown on the handler's own thread, bounded by
    console_guard's budget (Windows ends the process about then)."""
    from waxx.util.live_od import console_guard
    cam = TimedCam()
    window.camera_nanny.cam_a = cam
    window.live_od_server.wait_for_image_writer = slow_writer([])
    t0 = time.monotonic()
    console_guard._on_console_event(lambda name: window.shutdown(f"console: {name}"),
                                    console_guard.CTRL_CLOSE_EVENT, console_guard.DEFAULT_BUDGET_S)
    assert time.monotonic() - t0 < console_guard.DEFAULT_BUDGET_S + 0.5
    assert cam.calls == ["stop_grab", "Close"]
    assert cam.closed_at - t0 < 1.0


def test_a_camera_thread_that_is_slow_to_stop_delays_the_close_by_at_most_the_grab_wait(window, monkeypatch):
    from waxx.util.live_od.gui import main_window as mw

    class SlowThread:
        name = "Slowpoke"

        def __init__(self):
            self.stop_reasons = []
            self._running = True

        def request_stop(self, reason=None):
            self.stop_reasons.append(reason)

        def isRunning(self):
            return self._running

        def wait(self, ms):
            time.sleep(ms / 1000.0)
            return False
    slow = SlowThread()
    window._run_threads = (slow, None)
    cam = TimedCam()
    window.camera_nanny.cam_a = cam
    t0 = time.monotonic()
    window.shutdown("test")
    assert slow.stop_reasons == ["liveOD is shutting down"]
    assert mw.SHUTDOWN_GRAB_WAIT_S - 0.05 <= cam.closed_at - t0 < mw.SHUTDOWN_GRAB_WAIT_S + 0.5


def test_a_watchdog_that_cannot_start_does_not_stop_the_shutdown(window, monkeypatch, records):
    """No stderr (pythonw): faulthandler cannot arm its dump. The cameras are
    closed all the same, and the shutdown still happens only once."""
    from waxx.util.live_od.gui import main_window as mw

    def no_stderr(*a, **k):
        raise RuntimeError("sys.stderr is None")
    monkeypatch.setattr(mw, "faulthandler", types.SimpleNamespace(
        dump_traceback_later=no_stderr, cancel_dump_traceback_later=lambda: None))
    cam = TimedCam()
    window.camera_nanny.cam_a = cam
    window.shutdown("test")
    window.shutdown("again")
    assert cam.calls == ["stop_grab", "Close"]
    assert any("no stack-dump watchdog" in r for r in records)
    assert sum("shutting down (" in r for r in records) == 1


def test_a_camera_thread_stopped_by_the_shutdown_is_told_why(window, app, records):
    """m15: the thread's own last words say liveOD is shutting down, not that a
    newer run replaced its run."""
    from waxx.util.live_od.camera_mother import CameraBaby
    import inspect
    if "reason" not in inspect.signature(CameraBaby.request_stop).parameters:
        pytest.skip("waits for FIX-B edits.md: CameraBaby.request_stop(reason) (camera_mother)")
    srv = window.live_od_server
    window.camera_nanny = WaitingNanny()
    init_run_on_another_thread(srv, _camera_init())
    app.processEvents()
    baby = window._run_threads[0]
    window.shutdown("test")
    assert qt.join_or_keep(baby) == ""
    said = [r for r in records if baby.name in r and "grab ended" in r]
    assert said and "liveOD is shutting down" in said[-1]
    assert not any("replaced by a newer one" in r for r in said)


# ----------------------------------------------------------------------
# the live view: a stream nobody watches any more is stopped
# ----------------------------------------------------------------------

class StubHost:
    """The camera host's surface _check_live_streams uses, with a snapshot the
    test sets."""

    def __init__(self, **cam):
        self.cam = dict(cam)
        self.stopped = []

    def snapshot(self):
        return {"cameras": {"cam_a": dict(self.cam)}}

    def stop_stream(self, key):
        self.stopped.append(key)
        return None

    def shutdown(self, timeout_s):
        return {}


@pytest.fixture
def stubbed(window, monkeypatch):
    """The legacy window with a stub camera host and live window, for the rule
    alone (tests/test_fixb_live_view.py runs it on the real host)."""
    from waxx.util.live_od.gui import main_window as mw
    monkeypatch.setattr(mw, "LIVE_VIEW_STOP_WAIT_S", 0.4)
    host = StubHost(host_state="streaming", n_subs=1, locked=False)
    live = types.SimpleNamespace(keys=[], open_keys=lambda: list(live.keys))
    window.camera_host = host
    window.live_view_window = live
    window.camera_menu = types.SimpleNamespace(set_live_view_open=lambda *a: None)
    yield window, host, live
    window._live_stop_pending.clear()
    if window._live_stop_timer is not None:
        window._live_stop_timer.stop()
    window.camera_host = None
    window.live_view_window = None


def spin_until(app, pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_the_stream_stops_once_the_views_own_subscription_is_gone(stubbed, app, records):
    window, host, _ = stubbed
    window._on_live_view_closed("cam_a")
    assert host.stopped == [] and "cam_a" in window._live_stop_pending   # its view not detached yet
    host.cam["n_subs"] = 0                                               # now it is
    assert spin_until(app, lambda: host.stopped == ["cam_a"])
    assert window._live_stop_pending == {} and not window._live_stop_timer.isActive()


@pytest.mark.parametrize("change", [
    {"n_subs": 0, "host_state": "run_locked", "locked": True},           # a run holds it
    {"n_subs": 0, "host_state": "acquiring", "locked": False},           # a run's phase
    {"n_subs": 0, "host_state": "idle"},                                 # not streaming
])
def test_the_stream_is_left_alone_when_a_run_holds_it_or_it_does_not_stream(stubbed, app, change):
    window, host, _ = stubbed
    host.cam.update(change)
    window._on_live_view_closed("cam_a")
    spin_until(app, lambda: False, timeout=0.3)
    assert host.stopped == [] and window._live_stop_pending == {}


def test_a_view_opened_again_keeps_its_stream(stubbed, app):
    window, host, live = stubbed
    window._on_live_view_closed("cam_a")
    live.keys.append("cam_a")
    host.cam["n_subs"] = 0
    spin_until(app, lambda: False, timeout=0.3)
    assert host.stopped == [] and window._live_stop_pending == {}


def test_someone_else_still_watching_keeps_the_stream(stubbed, app):
    window, host, _ = stubbed
    window._on_live_view_closed("cam_a")                                 # n_subs stays 1
    assert spin_until(app, lambda: window._live_stop_pending == {}, timeout=2.0)
    assert host.stopped == []


def test_the_legacy_window_shows_the_same_camera_control(window):
    """Without the host the status row has the host's CameraControl too, minus ⚙
    and 🎥 (the settings dialog and the live view need the host)."""
    from waxx.util.live_od.gui.camera_control import CameraControl
    menu = window.camera_menu
    assert type(menu) is CameraControl
    assert not menu.cog_button.isVisibleTo(menu) and not menu.live_button.isVisibleTo(menu)
    assert window._live_view_button is None and window.live_view_window is None
