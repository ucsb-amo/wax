"""The liveOD window, built with a stand-in config and never started (no socket,
no beacon, no camera): shutdown (B12), the close-during-a-run question (D-e),
the local camera button under the run rule (B7), and a previous run's camera
thread cut off from the next run (B5). Every data file lives in tmp_path.

Each test's window is torn down inside that test (liveod_qt_helpers): every
thread it started is stopped and joined, and the window is deleted and collected
under a recording excepthook, so nothing is left for the garbage collector to
trip over in a later test.
"""
import logging
import os
import sys
import threading
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMessageBox

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


class FakeCam:
    """An open camera with the driver surface shutdown uses (no close_safely,
    like BaslerUSB)."""

    def __init__(self):
        self.calls = []
        self.opened = True

    def is_opened(self):
        return self.opened

    def stop_grab(self):
        self.calls.append("stop_grab")

    def Close(self):
        self.calls.append("Close")
        self.opened = False


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

    # Exceptions in slots are recorded, not fatal, for as long as the window
    # exists; an earlier test's garbage is collected now, not during this test.
    slot_errors = qt.SlotErrors()
    monkeypatch.setattr(sys, "excepthook", slot_errors)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    gc.collect()
    app.processEvents()
    n_before = len(slot_errors.seen)
    views_before = qt.pyqtgraph_views()
    threads_before = set(threading.enumerate())

    for cls in (LiveODServer, LiveODBroadcaster, LiveODPlotter):
        monkeypatch.setattr(cls, "start", lambda self, *a: None)
    patch_payload_stash(monkeypatch, run_file, [])
    # theme's module-level notifier may belong to a QApplication an earlier
    # module made and dropped; start from a live one
    if theme._notifier is not None and sip.isdeleted(theme._notifier):
        monkeypatch.setattr(theme, "_notifier", None)
    # every camera thread the window makes, so that all of them are joined
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
    monkeypatch.setattr(sys, "excepthook", slot_errors)        # the window installs its own
    yield win

    # --- teardown, all of it inside this test ---
    problems = []
    for obj in spawned:
        if hasattr(obj, "request_stop"):
            obj.request_stop()                   # a camera thread: stop quietly
        if hasattr(obj, "grab_finished"):
            obj.grab_finished()                  # an image dispatcher: drain and end
    for obj in spawned:
        problems.append(qt.join_or_keep(obj))
        writer = getattr(obj, "writer", None)
        if writer is not None and getattr(writer, "started", False):
            problems.append(qt.join_or_keep(writer._worker))
    win.shutdown("test teardown")                # idempotent; stops the timers
    win.run_id_timer.stop()
    win._camera_state_timer.stop()
    for t in qt.new_threads(threads_before):     # e.g. shutdown's camera closer
        problems.append(qt.join_or_keep(t))
    qt.forget_views_since(views_before)
    qt.delete_widgets(app, [getattr(win, "live_scalar_plot_window", None),
                            getattr(win, "fk_tof_window", None),
                            getattr(win, "_adjust_panel", None), win])
    del win
    spawned.clear()
    qt.delete_widgets(app, [])                   # one more collection, now without the window
    problems = [p for p in problems if p]
    errors = slot_errors.seen[n_before:]
    assert not problems, problems
    assert not errors, f"exceptions in Qt slots while the window existed: {errors}"


def _run_on(srv, camera_key="cam_a", run_id=81234):
    srv._run_in_progress = True
    srv._current_capture_images = True
    srv._current_camera_key = camera_key
    srv._current_run_id = run_id


# ----------------------------------------------------------------------
# shutdown
# ----------------------------------------------------------------------

def test_shutdown_closes_the_cameras_once_and_stops_serving(window, records):
    srv = window.live_od_server
    cam = FakeCam()
    window.camera_nanny.cam_a = cam                  # what the nanny holds once a camera is open
    btn = window.camera_conn_bar.get_button("cam_a")
    btn.camera = cam
    stopped = []
    srv.stop = lambda: stopped.append("server")
    window.broadcaster.stop = lambda: stopped.append("broadcaster")

    window.shutdown("test")
    window.shutdown("again")
    assert cam.calls == ["stop_grab", "Close"]
    assert stopped == ["server", "broadcaster"]
    assert not hasattr(window.camera_nanny, "cam_a")
    assert not btn.camera.is_opened()                # the button no longer holds the closed camera
    assert sum("shutting down" in r for r in records) == 1


def test_shutdown_from_another_thread(window):
    cam = FakeCam()
    window.camera_nanny.cam_b = cam
    t = threading.Thread(target=window.shutdown, args=("console: console window closed",))
    t.start()
    assert qt.join_or_keep(t) == ""
    assert cam.calls == ["stop_grab", "Close"]


def test_a_hung_camera_close_does_not_hang_shutdown(window, monkeypatch, records):
    from waxx.util.live_od.gui import main_window as mw
    release = threading.Event()

    class Hung(FakeCam):
        def Close(self):
            release.wait(10)
            super().Close()
    window.camera_nanny.cam_a = Hung()
    monkeypatch.setattr(mw, "SHUTDOWN_CLOSE_CAMERAS_S", 0.2)
    t0 = time.time()
    try:
        window.shutdown("test")
        assert time.time() - t0 < 5.0
    finally:
        release.set()                                # the closer thread ends; the fixture joins it
    assert any("did not finish within" in r for r in records)


def test_the_hooks_all_lead_to_one_shutdown(window, app, monkeypatch):
    from waxx.util.live_od.gui import main_window as mw
    registered, installed, posted, calls = [], [], [], []
    monkeypatch.setattr(mw.atexit, "register", lambda fn, *a: registered.append((fn, a)))
    monkeypatch.setattr(mw.console_guard, "install", lambda cb: installed.append(cb) or True)
    monkeypatch.setattr(mw.QMetaObject, "invokeMethod", lambda *a: posted.append(a))
    real = window.shutdown
    monkeypatch.setattr(window, "shutdown", lambda reason="": (calls.append(reason), real(reason)))
    window.install_shutdown_hooks(app)
    try:
        (fn, args), = registered
        fn(*args)                                    # interpreter exit
        installed[0]("console window closed")        # the console's X
        assert calls == ["interpreter exit", "console: console window closed"]
        assert window._shutdown_done and len(posted) == 1   # and the app is asked to quit
    finally:
        # connected as the window's own method (raises if it is not); the
        # application outlives the window, so its quit signal must not keep it
        app.aboutToQuit.disconnect(window._shutdown_on_quit)


def test_quitting_the_application_shuts_down(window):
    # the slot aboutToQuit is connected to (the test above checks the connection;
    # emitting the application's own signal would run every library's quit
    # handler, pyqtgraph's clean-up included)
    window._shutdown_on_quit()
    assert window._shutdown_done


def test_console_guard_runs_the_callback_within_its_budget():
    """The handler Windows would call, called directly (no console event is sent)."""
    from waxx.util.live_od import console_guard
    if sys.platform != "win32":
        assert console_guard.install(lambda name: None) is False
        return
    seen, release = [], threading.Event()
    before = set(threading.enumerate())
    try:
        assert console_guard.install(lambda name: seen.append(name), budget_s=2.0) is True
        assert console_guard._installed(console_guard.CTRL_CLOSE_EVENT) == 1
        assert seen == ["console window closed"]

        def slow(name):
            release.wait(10)
        assert console_guard.install(slow, budget_s=0.2) is True     # replaces the first
        t0 = time.time()
        assert console_guard._installed(console_guard.CTRL_LOGOFF_EVENT) == 1
        assert time.time() - t0 < 2.0                                   # returns inside the budget
    finally:
        console_guard.uninstall()
        release.set()
        for t in qt.new_threads(before):                               # the callback's helper thread
            assert qt.join_or_keep(t) == ""
    assert console_guard._installed is None


@pytest.mark.parametrize("event, name", [(0, "Ctrl+C"), (1, "Ctrl+Break")])
def test_console_interrupts_are_ignored_not_a_shutdown(records, event, name):
    """Ctrl+C in the liveOD console (e.g. copying a log line with nothing
    selected) must not end a run: no callback, handled (so no KeyboardInterrupt),
    one WARNING that says how to quit. Called directly: no console event is sent."""
    from waxx.util.live_od import console_guard
    assert event in console_guard.INTERRUPT_EVENTS
    called = []
    before = set(threading.enumerate())
    assert console_guard._on_console_event(called.append, event, 2.0) == 1
    assert called == []
    assert qt.new_threads(before) == []                                # no clean-up thread either
    assert [r for r in records if "ignored" in r] == [
        f"{name} ignored -- close the liveOD window to quit"]


def test_console_close_logoff_and_shutdown_still_shut_down():
    from waxx.util.live_od import console_guard
    called = []
    for event in (console_guard.CTRL_CLOSE_EVENT, console_guard.CTRL_LOGOFF_EVENT,
                  console_guard.CTRL_SHUTDOWN_EVENT):
        assert console_guard._on_console_event(called.append, event, 2.0) == 1
    assert called == ["console window closed", "user logging off", "system shutting down"]


def test_the_installed_handler_ignores_ctrl_c():
    """The handler Windows would call (not the helper), for Ctrl+C."""
    from waxx.util.live_od import console_guard
    if sys.platform != "win32":
        return
    called = []
    try:
        assert console_guard.install(called.append, budget_s=2.0) is True
        assert console_guard._installed(console_guard.CTRL_C_EVENT) == 1
        assert console_guard._installed(console_guard.CTRL_BREAK_EVENT) == 1
        assert called == []
    finally:
        console_guard.uninstall()


# ----------------------------------------------------------------------
# closing the window during a run (D-e)
# ----------------------------------------------------------------------

def test_closing_during_a_run_asks_first_and_no_keeps_liveod_running(window, monkeypatch):
    from waxx.util.live_od.gui import main_window as mw
    asked = []
    monkeypatch.setattr(mw.QMessageBox, "question",
                        lambda *a: asked.append(a) or QMessageBox.StandardButton.No)
    _run_on(window.live_od_server)
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted() and not window._shutdown_done
    parent, title, text, buttons, default = asked[0]
    assert "run 81234" in text and default == QMessageBox.StandardButton.No

    monkeypatch.setattr(mw.QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() and window._shutdown_done


def test_closing_with_no_run_does_not_ask(window, monkeypatch):
    from waxx.util.live_od.gui import main_window as mw

    def never(*a):
        raise AssertionError("asked without a run")
    monkeypatch.setattr(mw.QMessageBox, "question", never)
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() and window._shutdown_done


# ----------------------------------------------------------------------
# liveOD names itself in the camera locks (DRV's device_lock)
# ----------------------------------------------------------------------

def test_main_labels_the_camera_locks_as_liveods(monkeypatch):
    """main() without an application or a window: both are stand-ins."""
    import ctypes
    from waxx.control.cameras import device_lock
    from waxx.util.live_od import config as live_od_config
    from waxx.util.live_od.gui import main_window as mw
    labels, order = [], []
    monkeypatch.setattr(device_lock, "set_process_label", lambda label: labels.append(label))
    monkeypatch.setattr(live_od_config, "_active", None)
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(shell32=types.SimpleNamespace(
        SetCurrentProcessExplicitAppUserModelID=lambda *a: None)), raising=False)
    monkeypatch.setattr(mw, "QSettings", lambda *a: None)          # no registry writes

    class App:
        def __init__(self, argv):
            order.append("app")

        def exec(self):
            return 0

    class Win:
        def __init__(self, settings=None):
            order.append(("window", list(labels)))

        def install_shutdown_hooks(self, app):
            order.append("hooks")

        def setWindowTitle(self, t):
            pass

        def setWindowIcon(self, i):
            pass

        def style(self):
            return types.SimpleNamespace(standardIcon=lambda *a: None)

        def show(self):
            pass
    monkeypatch.setattr(mw, "QApplication", App)
    monkeypatch.setattr(mw, "LiveODWindow", Win)
    with pytest.raises(SystemExit):
        mw.main(live_od_config.LiveODConfig())
    assert labels == ["liveOD"]
    assert order == ["app", ("window", ["liveOD"]), "hooks"]       # labelled before any camera opens


# ----------------------------------------------------------------------
# the window's own camera button (B7)
# ----------------------------------------------------------------------

def test_local_toggle_follows_the_run_rule(window, records):
    srv = window.live_od_server
    pressed = []
    for key in ("cam_a", "cam_b"):
        btn = window.camera_conn_bar.get_button(key)
        btn.button_pressed = lambda k=key: pressed.append(k)
    window.camera_conn_bar.get_button("cam_a").camera = FakeCam()     # open
    _run_on(srv, "cam_a")

    window._on_camera_toggle_requested("cam_a")      # would close the run's camera
    window._on_camera_toggle_requested("cam_b")      # would open a camera during a run
    assert pressed == []
    assert any("Camera control rejected: run 81234 in progress (uses cam_a)" in r for r in records)

    window.camera_conn_bar.get_button("cam_b").camera = FakeCam()
    window._on_camera_toggle_requested("cam_b")      # closing a camera the run does not use
    srv._run_in_progress = False
    window._on_camera_toggle_requested("cam_a")      # no run: anything goes
    assert pressed == ["cam_b", "cam_a"]


# ----------------------------------------------------------------------
# a previous run's camera thread cannot reach the next run (B5)
# ----------------------------------------------------------------------

class WaitingNanny:
    """The camera never becomes reachable: a baby waits in persistent_get_camera
    until its own break_check says stop."""

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
           "params": {"N_img": 3}, "N_shots_with_repeats": 1, "expt_class": "b5_test"}
    msg.update(kw)
    return msg


def test_a_previous_runs_camera_thread_cannot_reach_the_next_run(window, app, records):
    srv = window.live_od_server
    window.camera_nanny = WaitingNanny()

    first = srv._handle_init_run(_camera_init())          # spawns baby A: waits for its camera
    baby_a = window.the_baby
    assert baby_a is not None and baby_a.isRunning()
    window.on_run_done()                                   # the run is over for liveOD (the_baby cleared)
    srv._run_in_progress = False

    srv._handle_init_run(_camera_init())                   # the next run
    baby_b = window.the_baby
    assert baby_b is not baby_a and window._run_threads[0] is baby_b
    assert qt.join_or_keep(baby_a) == ""                   # asked to stop, and it did
    assert os.path.exists(first["filepath"])               # quietly: the old run's file is untouched

    # whatever the old thread still says goes nowhere
    baby_a.grab_failed_signal.emit("camera timed out: late")
    baby_a.cam_status_signal.emit(2)
    baby_a.camera_overrides_signal.emit("cam_a", {"gain": (300, 30)})
    app.processEvents()
    assert srv._grab_failure == "" and not srv._cam_ready_event.is_set()
    assert srv.camera_overrides_record() == {}

    # a report already on its way is dropped by the check at the far end
    window._from_run_baby(baby_a, srv.on_grab_failed, "in flight")
    assert srv._grab_failure == ""
    assert any("its run has been replaced" in r for r in records)

    # the current run's thread does get through
    baby_b.cam_status_signal.emit(2)
    assert srv._cam_ready_event.is_set()
