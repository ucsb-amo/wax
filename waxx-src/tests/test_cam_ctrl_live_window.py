"""liveOD's live view window: the banner follows each frame's source (LIVE — not
recorded / RUN <id> — recorded, view only), frames reach the viewer read-only,
the display is capped at 10 Hz (2 Hz while a run is active), and closing a dock
ends only this window's view.

The viewer is beacon's real CameraViewerWidget on a fake CameraSource
(``FakeStream``, standing in for the host's LocalHostStream) over the FakeHost:
no camera, no socket.  The widget's state files go to a temporary BEACON_STATE_DIR.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cam_ctrl_fakes import FakeHost, make_stream  # noqa: E402
from liveod_qt_helpers import delete_widgets, join_or_keep, session_app  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return session_app()


@pytest.fixture(autouse=True)
def beacon_state(monkeypatch, tmp_path):
    monkeypatch.setenv("BEACON_STATE_DIR", str(tmp_path / "beacon_state"))


def pump(app, until, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        app.processEvents()
        if until():
            return True
        if time.monotonic() > deadline:
            raise AssertionError(f"not met within {timeout:g} s")
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------

def test_banner_text_follows_the_frame_source():
    from waxx.util.live_od.gui.live_view_window import LIVE_TEXT, banner_for, run_id_of
    assert banner_for("live") == ("live", LIVE_TEXT) == ("live", "LIVE — not recorded")
    assert banner_for("snap") == ("live", LIVE_TEXT)
    assert banner_for("run", "80713:1a2b3c4d") == ("run", "RUN 80713 — recorded, view only")
    assert banner_for("run", None, run_id=80714) == ("run", "RUN 80714 — recorded, view only")
    # save_data=False runs have id 0: their frames are not recorded, and the banner says so
    assert banner_for("run", "0:1a2b3c4d") == ("run", "RUN (unsaved) — not recorded, view only")
    # a tag that names no run id: shown as it is, with no claim about recording
    assert banner_for("run", "weird") == ("run", "RUN weird — view only")
    assert run_id_of("80713:ab") == 80713 and run_id_of(None) is None and run_id_of("x:1") is None


class _Inner:
    """A source that always has a new frame (a free-running camera)."""
    camera_id, category, serial, name, model = "basler_usb:1", "basler_usb", "1", "cam", "m"
    host, server_id, protocol, view_only = "h", "", "local", ""
    capabilities = frozenset({"settings"})
    display_name = "cam"
    reconnect_generation = 0

    def __init__(self):
        self.n = 0
        self.settings_calls = []
        self.image = np.zeros((4, 4), np.uint16)            # writeable, as a driver might hand it

    def next_frame(self, timeout_s):
        from beacon.camera.viewer.sources import ViewerFrame
        self.n += 1
        return ViewerFrame(self.image, source="live", seq=self.n)

    def set_settings(self, values, confirmed=frozenset()):
        self.settings_calls.append((values, confirmed))
        return {"ok": True}

    def open(self):
        return {"ok": True}

    close = open

    def status(self):
        return {}


def _paced(interval):
    from waxx.util.live_od.gui.live_view_window import PacedSource
    t = [0.0]
    seen = []

    def sleep(d):
        t[0] += d
    src = PacedSource(_Inner(), lambda: interval, seen.append, clock=lambda: t[0], sleep=sleep)
    return src, t, seen


@pytest.mark.parametrize("interval, cap_hz", [(0.1, 10.0), (0.5, 2.0)])
def test_the_display_rate_is_capped(interval, cap_hz):
    src, t, seen = _paced(interval)
    shown = []
    while t[0] < 3.0:
        f = src.next_frame(0.5)                  # the viewer's frame thread asks with 0.5 s
        if f is not None:
            shown.append(t[0])
    gaps = np.diff(shown)
    assert len(shown) >= 2 and gaps.min() >= interval - 1e-9
    assert len(shown) / 3.0 <= cap_hz + 1.0 / 3.0 + 1e-9
    assert len(seen) == len(shown)


def test_frames_are_handed_on_read_only():
    src, _t, seen = _paced(0.1)
    f = src.next_frame(0.5)
    assert not f.image.flags.writeable
    assert src.inner.image.flags.writeable           # the source's own array is untouched
    assert seen[0] is f
    src.set_settings({"gain": 150}, confirmed=frozenset({"gain"}))
    src.set_settings({"gain": 10})
    assert src.inner.settings_calls == [({"gain": 150}, frozenset({"gain"})),
                                        ({"gain": 10}, frozenset())]


def test_the_rate_follows_the_run_state(app):
    from waxx.util.live_od.gui.live_view_window import LiveViewWindow
    host = FakeHost()
    win = LiveViewWindow(host, stream_factory=make_stream)
    try:
        assert win.max_display_hz() == 10.0
        host.set_state("andor", "acquiring", run_id=80713)
        win.set_snapshot(host.snapshot())
        assert win.max_display_hz() == 2.0 and win.frame_interval_s() == 0.5
        host.set_state("andor", "idle", run_id=None)
        win.set_snapshot(host.snapshot())
        assert win.max_display_hz() == 10.0
        win.set_run_active(True)                     # liveOD's own run state
        assert win.max_display_hz() == 2.0
    finally:
        delete_widgets(app, [win])


# ---------------------------------------------------------------------------
# The window, with the real viewer on a fake source
# ---------------------------------------------------------------------------

def _close_all(app, win, viewers):
    from beacon.camera.viewer.widget import wait_pool
    workers = [v._frame_worker for v in viewers if getattr(v, "_frame_worker", None) is not None]
    for key in list(win.open_keys()):
        win.close_camera(key)
    left = []
    for w in workers:
        w.exited.wait(5.0)
        left.append(join_or_keep(w, 5.0))
    wait_pool(5.0)
    app.processEvents()
    return [x for x in left if x]


def test_banner_follows_each_frame_and_frames_render_read_only(app):
    from waxx.util.live_od.gui.live_view_window import LIVE_STYLE, LIVE_TEXT, LiveViewWindow
    host = FakeHost()
    win = LiveViewWindow(host, stream_factory=make_stream)
    opened, closed = [], []
    win.view_opened.connect(opened.append)
    win.view_closed.connect(closed.append)
    viewers = []
    try:
        win.show_camera("andor")
        viewers.append(win.viewer("andor"))
        assert opened == ["andor"] and win.open_keys() == ["andor"]
        assert host.calls_of("start_stream") == [("start_stream", "andor")]   # it was idle
        pump(app, lambda: host.cams["andor"]["n_subs"] == 1)                  # subscribed

        image = np.full((8, 10), 700, np.uint16)                             # writeable
        host.push_frame("andor", image, source="live")
        pump(app, lambda: win.banner("andor").text() == LIVE_TEXT)
        assert win.banner_kind("andor") == "live"
        assert win.banner("andor").styleSheet() == LIVE_STYLE
        viewer = win.viewer("andor")
        pump(app, lambda: viewer.last_image is not None)
        assert not viewer.last_image.flags.writeable                          # shown read-only
        assert image.flags.writeable                                          # not changed

        host.push_frame("andor", source="run", run_tag="80713:1a2b3c4d")
        pump(app, lambda: win.banner("andor").text() == "RUN 80713 — recorded, view only")
        assert win.banner_kind("andor") == "run"
        assert "00897b" in win.banner("andor").styleSheet()                   # teal
        assert win.last_source("andor") == ("run", "80713:1a2b3c4d")

        host.push_frame("andor", source="snap")
        pump(app, lambda: win.banner("andor").text() == LIVE_TEXT)

        host.push_frame("andor", source="run", run_tag="0:ffff0000")
        pump(app, lambda: "unsaved" in win.banner("andor").text())
        assert "not recorded" in win.banner("andor").text()
    finally:
        left = _close_all(app, win, viewers)
        delete_widgets(app, [win])
    assert closed == ["andor"]
    assert left == []


def test_closing_a_dock_ends_only_this_view(app):
    from waxx.util.live_od.gui.live_view_window import LiveViewWindow
    host = FakeHost()
    win = LiveViewWindow(host, stream_factory=make_stream)
    closed = []
    win.view_closed.connect(closed.append)
    viewers = []
    try:
        win.show_camera("andor")
        win.show_camera("xy_basler")
        viewers += [win.viewer("andor"), win.viewer("xy_basler")]
        pump(app, lambda: host.cams["andor"]["n_subs"] == 1 and host.cams["xy_basler"]["n_subs"] == 1)
        worker = viewers[1]._frame_worker
        # the dock's own close button
        win._docks["xy_basler"][0].close()
        app.processEvents()
        assert closed == ["xy_basler"] and win.open_keys() == ["andor"]
        if worker is not None:
            worker.exited.wait(5.0)
        from beacon.camera.viewer.widget import wait_pool
        wait_pool(5.0)
        pump(app, lambda: host.cams["xy_basler"]["n_subs"] == 0)            # detached
        assert host.calls_of("stop_stream") == []                           # never stopped
        assert host.cams["xy_basler"]["host_state"] == "streaming"
        # opening it again raises the same view instead of a second one
        win.show_camera("andor")
        assert win.open_keys() == ["andor"]
    finally:
        left = _close_all(app, win, viewers)
        delete_widgets(app, [win])
    assert left == []


def test_no_stream_is_started_for_a_camera_a_run_holds(app):
    from waxx.util.live_od.gui.live_view_window import LiveViewWindow
    host = FakeHost()
    host.set_state("andor", "acquiring", run_id=80713)
    win = LiveViewWindow(host, stream_factory=make_stream)
    viewers = []
    try:
        win.show_camera("andor")
        viewers.append(win.viewer("andor"))
        assert host.calls_of("start_stream") == []
        button = win._docks["andor"][1].stream_button
        assert not button.isEnabled()
        assert win.max_display_hz() == 2.0
        host.set_state("andor", "idle", run_id=None)
        win.set_snapshot(host.snapshot())
        assert button.isEnabled() and button.text() == "Start live"
        button.click()
        assert host.calls_of("start_stream") == [("start_stream", "andor")]
        win.set_snapshot(host.snapshot())
        assert button.text() == "Stop live"
        button.click()
        assert host.calls_of("stop_stream") == [("stop_stream", "andor")]
    finally:
        left = _close_all(app, win, viewers)
        delete_widgets(app, [win])
    assert left == []


def test_a_failed_stream_start_is_shown(app):
    from waxx.util.live_od.gui.live_view_window import LiveViewWindow
    from cam_ctrl_fakes import HostRefused, done_future
    host = FakeHost()
    host.start_stream = lambda key: done_future(exc=HostRefused(f"{key} is held elsewhere"))
    win = LiveViewWindow(host, stream_factory=make_stream)
    viewers = []
    try:
        win.show_camera("xy_basler")
        viewers.append(win.viewer("xy_basler"))
        note = win._docks["xy_basler"][1].note
        pump(app, lambda: not note.isHidden())
        assert "held elsewhere" in note.text()
    finally:
        left = _close_all(app, win, viewers)
        delete_widgets(app, [win])
    assert left == []


def test_fake_host_and_stream_match_the_real_interfaces():
    """The FakeHost's names are the real CameraHost's, and the FakeStream is a
    viewer source, so the GUI is built against the real interfaces."""
    from beacon.camera.viewer.sources import missing_members
    from cam_ctrl_fakes import HOST_METHODS, STREAM_METHODS
    from waxx.util.live_od.camera_host.host import CameraHost
    missing = [m for m in HOST_METHODS + STREAM_METHODS if not callable(getattr(CameraHost, m, None))]
    assert missing == []
    assert missing_members(make_stream(FakeHost(), "xy_basler")) == []
