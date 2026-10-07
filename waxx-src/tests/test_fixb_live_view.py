"""liveOD's live view on the real camera host (use_camera_host, beacon FakeBackend
cameras; serve=False and no beacons, see cam_host_helpers.quiet_network): when a
camera's last view closes, nothing else subscribes to it and no run holds it, the
window stops its live stream -- the Andor is not left streaming at live EM gain
for nobody. The viewers are stand-ins that attach when opened and let go a moment
after they are shut down, on another thread, as the real one's pool does. Data
files live in tmp_path.
"""
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging
import pytest
from PyQt6.QtWidgets import QWidget

from cam_host_helpers import wait_for
import liveod_qt_helpers as qt
from test_cam_host_window import _window


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def records():
    got = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            got.append(record.getMessage())
    handler = ListHandler(logging.INFO)
    log = logging.getLogger("waxx.live_od")
    log.addHandler(handler)
    yield got
    log.removeHandler(handler)


class LateViewer(QWidget):
    """The viewer surface LiveViewWindow uses: open_camera attaches its source
    (a subscriber, n_subs); shutdown detaches it DETACH_AFTER_S later on another
    thread."""
    DETACH_AFTER_S = 0.2

    def __init__(self, source, timers):
        super().__init__()
        self.source = source
        self._timers = timers

    def open_camera(self):
        self.source.open()

    def shutdown(self):
        t = threading.Timer(self.DETACH_AFTER_S, self.source.close)
        self._timers.append(t)
        t.start()


@pytest.fixture
def live(app, tmp_path, monkeypatch):
    from waxx.util.live_od.gui import live_view_window as lvw
    timers = []
    monkeypatch.setattr(lvw, "_default_viewer", lambda source: LateViewer(source, timers))
    win, fakes, hosts, teardown = _window(app, tmp_path, monkeypatch, True)
    try:
        yield win, hosts[0]
    finally:
        window = win.live_view_window
        if window is not None:
            for key in window.open_keys():
                window.close_camera(key)
        for t in timers:
            t.join(5.0)
        if window is not None:
            qt.delete_widgets(app, [window])
        teardown()


def cam_b(app, host):
    app.processEvents()
    return host.snapshot()["cameras"]["cam_b"]


def test_closing_the_last_view_stops_a_stream_nobody_watches(live, app, records):
    win, host = live
    win._on_live_view_requested("cam_b", True)            # the camera's movie-camera button
    wait_for(lambda: cam_b(app, host)["host_state"] == "streaming"
             and cam_b(app, host)["n_subs"] == 1, what="cam_b streaming for its view")
    win._on_live_view_requested("cam_b", False)
    assert "cam_b" in win._live_stop_pending               # its view has not let go yet
    wait_for(lambda: cam_b(app, host)["host_state"] != "streaming", timeout=5.0,
             what="cam_b's stream stopped")
    c = cam_b(app, host)
    assert c["host_state"] == "idle" and c["n_subs"] == 0
    assert win._live_stop_pending == {}
    assert any("nothing else subscribes to it: stopping its live stream" in r for r in records)


def test_another_subscriber_keeps_the_stream(live, app, monkeypatch, records):
    from waxx.util.live_od.gui import main_window as mw
    monkeypatch.setattr(mw, "LIVE_VIEW_STOP_WAIT_S", 0.8)
    win, host = live
    win._on_live_view_requested("cam_b", True)
    wait_for(lambda: cam_b(app, host)["host_state"] == "streaming", what="cam_b streaming")
    cid = host.spec("cam_b").camera_id
    host.core.attach(cid, "fixb:other", "inproc", label="another viewer")
    try:
        win._on_live_view_requested("cam_b", False)
        wait_for(lambda: (app.processEvents(), win._live_stop_pending == {})[1], timeout=5.0,
                 what="the pending check given up")
        assert cam_b(app, host)["host_state"] == "streaming"
        assert any("still watch it; its live stream goes on" in r for r in records)
    finally:
        host.core.detach(cid, "fixb:other")


def test_host_mode_has_the_camera_control_and_one_settings_dialog_per_camera(live, app):
    from waxx.util.live_od.gui.camera_control import CameraControl
    win, host = live
    assert isinstance(win.camera_menu, CameraControl)
    win.camera_menu.settings_requested.emit("cam_b")          # the cog
    dialog = win._camera_dialogs["cam_b"]
    win.camera_menu.settings_requested.emit("cam_b")
    assert win._camera_dialogs == {"cam_b": dialog}           # the same one again
    dialog.close()
    app.processEvents()
