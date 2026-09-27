"""liveOD's live view: a top-level window with one dock per camera (PLAN T13).

Opened from a camera's 🎥 in the status row.  Each dock holds beacon's
``CameraViewerWidget`` on ``LocalHostStream(host, key)`` -- the camera host's
frames in-process, no sockets -- under a banner that follows every frame:

    LIVE — not recorded                  (amber)  live and snap frames
    RUN 80713 — recorded, view only      (teal)   a run's frames (Q4: subscribers see them)
    RUN (unsaved) — not recorded, view only       a save_data=False run's frames

The run id comes from the frame's run tag (the host tags run frames
``"<run_id>:<token>"``), else from the snapshot.  (The widget's own banner shows
the raw tag and always says "recorded", so it is collapsed here.)

Frames are shown at most 10 per second, and 2 per second while any run is active
(the GUI thread is the contended one during a run; PLAN C3).  They are handed to
the widget read-only.  Opening a camera's view starts its live stream if the camera
is idle or closed (``host.start_stream``); each dock also has Start / Stop live.
Closing a dock only ends this window's view (the viewer detaches); the window
itself never stops the stream.  liveOD's main window does, a moment later, when
nothing else watches the camera and no run holds it.  Stop live releases only
liveOD's own request: a stream another program (the spot finder) asked for keeps
running until that program lets go.

Host methods used: ``start_stream``, ``stop_stream``, and whatever
``LocalHostStream`` uses (``spec``, ``core``, ``wait_frame``, ``describe``,
``set_live``, ``worker``, ``snapshot``).
"""

import dataclasses
import time
from typing import Callable, Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (QDockWidget, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from beacon.camera.viewer.sources import CameraSource

from waxx.util.live_od.gui.camera_control import HOST_STATES, RUN_PHASES, camera_entries

LIVE_TEXT = "LIVE — not recorded"
LIVE_STYLE = ("background: #ffb300; color: #1a1a1a; font-weight: bold; padding: 2px 8px;"
              " border-radius: 3px;")
RUN_STYLE = ("background: #00897b; color: #ffffff; font-weight: bold; padding: 2px 8px;"
             " border-radius: 3px;")
#: display rate caps (frames per second)
MAX_HZ = 10.0
MAX_HZ_DURING_RUN = 2.0


def run_id_of(run_tag) -> Optional[int]:
    """80713 from the host's run tag "80713:1a2b3c4d" (None if it is not one)."""
    head = str(run_tag or "").split(":", 1)[0].strip()
    return int(head) if head.isdigit() else None


def banner_for(source: str, run_tag=None, run_id=None) -> tuple:
    """``(kind, text)`` for a frame: kind "live" or "run".  ``run_id``: the
    snapshot's, used when the tag names none."""
    if source != "run":
        return "live", LIVE_TEXT
    rid = run_id_of(run_tag)
    if rid is None and run_id is not None:
        try:
            rid = int(run_id)
        except (TypeError, ValueError):
            rid = None
    if rid is None:
        return "run", f"RUN {run_tag or '?'} — view only"
    if rid == 0:
        return "run", "RUN (unsaved) — not recorded, view only"
    return "run", f"RUN {rid} — recorded, view only"


def readonly_frame(frame):
    """``frame`` with a read-only image (a read-only view if it was writeable)."""
    image = getattr(frame, "image", None)
    flags = getattr(image, "flags", None)
    if flags is None or not flags.writeable:
        return frame
    view = image.view()
    view.flags.writeable = False
    try:
        return dataclasses.replace(frame, image=view)
    except TypeError:
        return frame


class PacedSource(CameraSource):
    """A viewer source around another one: frames at most every ``interval_fn()``
    seconds, read-only, and each one reported to ``on_frame(frame)`` (on the
    viewer's frame thread).  Everything else is the inner source's."""

    def __init__(self, inner, interval_fn: Callable[[], float],
                 on_frame: Optional[Callable] = None, *, clock=time.monotonic,
                 sleep=time.sleep):
        super().__init__(inner.camera_id, category=inner.category, serial=inner.serial,
                         name=inner.name, model=inner.model, host=inner.host,
                         server_id=inner.server_id)
        self.inner = inner
        self.protocol = inner.protocol
        self.capabilities = inner.capabilities
        self.view_only = inner.view_only
        self._interval_fn = interval_fn
        self._on_frame = on_frame
        self._clock = clock
        self._sleep = sleep
        self._last_t: Optional[float] = None

    @property
    def display_name(self) -> str:
        return self.inner.display_name

    @property
    def reconnect_generation(self) -> int:
        return self.inner.reconnect_generation

    def open(self) -> dict:
        return self.inner.open()

    def close(self) -> dict:
        return self.inner.close()

    def next_frame(self, timeout_s: float):
        timeout_s = max(0.0, float(timeout_s))
        if self._last_t is not None:
            wait = self._last_t + max(0.0, float(self._interval_fn())) - self._clock()
            if wait > 0:
                if wait >= timeout_s:
                    self._sleep(timeout_s)
                    return None
                self._sleep(wait)
                timeout_s -= wait
        frame = self.inner.next_frame(timeout_s)
        if frame is None:
            return None
        self._last_t = self._clock()
        frame = readonly_frame(frame)
        if self._on_frame is not None:
            self._on_frame(frame)
        return frame

    def status(self) -> dict:
        return self.inner.status()

    def describe(self) -> dict:
        return self.inner.describe()

    def set_settings(self, values: dict, confirmed=frozenset()) -> dict:
        # ``confirmed`` only when there is one: a source without the keyword still works
        if confirmed:
            return self.inner.set_settings(values, confirmed=confirmed)
        return self.inner.set_settings(values)

    def get_defaults(self) -> dict:
        return self.inner.get_defaults()

    def save_defaults(self, settings=None, roi=None, norm_reference=None) -> dict:
        return self.inner.save_defaults(settings=settings, roi=roi, norm_reference=norm_reference)

    def rename(self, name: str) -> dict:
        return self.inner.rename(name)


class _Relay(QObject):
    """From the viewer's frame thread and the host's threads to the GUI thread."""
    seen = pyqtSignal(str, str, object)         # camera_key, frame source, run_tag
    note = pyqtSignal(str, str)                 # camera_key, message


class _Dock(QDockWidget):
    closing = pyqtSignal(str)

    def __init__(self, key: str, parent=None):
        super().__init__(key, parent)
        self.key = key
        self.setObjectName(f"live_view_{key}")

    def closeEvent(self, event):
        self.closing.emit(self.key)
        super().closeEvent(event)


class _ViewPanel(QWidget):
    """A dock's contents: the banner row (banner, Start/Stop live) over the viewer."""

    def __init__(self, viewer: QWidget, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.banner = QLabel("waiting for frames…")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.banner_kind = ""
        self.stream_button = QPushButton("Start live")
        self.stream_button.setFixedWidth(84)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.hide()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.banner, 1)
        row.addWidget(self.stream_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        layout.addLayout(row)
        layout.addWidget(self.note)
        layout.addWidget(viewer, 1)


def _default_stream(host, key):
    from waxx.util.live_od.camera_host.local_stream import LocalHostStream
    return LocalHostStream(host, key)


def _default_viewer(source):
    from beacon.camera.viewer.widget import CameraViewerWidget
    return CameraViewerWidget(source)


class LiveViewWindow(QMainWindow):
    """See the module docstring.

    ``LiveViewWindow(host, parent=None, *, stream_factory=None, viewer_factory=None,
    snapshot_signal=None, start_streams=True)``.  ``stream_factory(host, key)``
    makes a camera's source (default ``LocalHostStream``); ``viewer_factory(source)``
    its widget (default ``CameraViewerWidget``; it must have ``open_camera()`` and
    ``shutdown()``).  ``snapshot_signal``: the host's snapshots on the GUI thread
    (``HostQtBridge.snapshot_changed``); else the owner calls ``set_snapshot``.
    """

    view_opened = pyqtSignal(str)       # camera_key
    view_closed = pyqtSignal(str)       # camera_key

    def __init__(self, host, parent=None, *, stream_factory=None, viewer_factory=None,
                 snapshot_signal=None, start_streams: bool = True):
        super().__init__(parent)
        self.host = host
        self._stream_factory = stream_factory or _default_stream
        self._viewer_factory = viewer_factory or _default_viewer
        self._start_streams = bool(start_streams)
        self._docks: dict = {}              # camera_key -> (_Dock, _ViewPanel, PacedSource)
        self._entries: dict = {}
        self._run_active_explicit = False
        self._run_active_snapshot = False
        self._last_source: dict = {}        # camera_key -> (source, run_tag)
        self._relay = _Relay(self)
        self._relay.seen.connect(self._on_seen)
        self._relay.note.connect(self._on_note)

        self.setWindowTitle("liveOD — live view")
        self.setDockNestingEnabled(True)
        self._placeholder = QLabel("Open a camera's live view with its 🎥 button in liveOD.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self._placeholder)
        if viewer_factory is None:
            try:
                from beacon.camera.viewer import widget as _viewer_module
                self.setStyleSheet(getattr(_viewer_module, "DARK_STYLESHEET", ""))
            except Exception:
                pass
        self.resize(900, 700)
        try:
            self.set_snapshot(host.snapshot())
        except Exception:
            pass
        if snapshot_signal is not None:
            snapshot_signal.connect(self.set_snapshot)

    # ------------------------------------------------------------------
    # Rate
    # ------------------------------------------------------------------

    def run_active(self) -> bool:
        return self._run_active_explicit or self._run_active_snapshot

    def set_run_active(self, active: bool) -> None:
        """liveOD's own word that a run is on (its run state), on top of the snapshot's."""
        self._run_active_explicit = bool(active)

    def max_display_hz(self) -> float:
        return MAX_HZ_DURING_RUN if self.run_active() else MAX_HZ

    def frame_interval_s(self) -> float:
        return 1.0 / self.max_display_hz()

    # ------------------------------------------------------------------
    # Host state
    # ------------------------------------------------------------------

    def set_snapshot(self, snapshot) -> None:
        entries = camera_entries(snapshot)
        if not entries:
            return
        self._entries.update(entries)
        self._run_active_snapshot = any(
            str(e.get("host_state", "")) in RUN_PHASES or bool(e.get("locked"))
            for e in self._entries.values())
        for key in self._docks:
            self._render_stream_button(key)

    def _render_stream_button(self, key: str) -> None:
        _dock, panel, _src = self._docks[key]
        hs = str(self._entries.get(key, {}).get("host_state", ""))
        button = panel.stream_button
        if hs == "streaming":
            button.setText("Stop live")
            button.setEnabled(True)
            button.setToolTip(f"Give back liveOD's request for {key}'s live stream; it stops "
                              f"unless another program (e.g. the spot finder) still asks for it")
        elif hs in ("idle", "closed"):
            button.setText("Start live")
            button.setEnabled(True)
            button.setToolTip(f"Start {key}'s live stream")
        else:
            words = HOST_STATES.get(hs, ("", hs or "state unknown", ""))[1]
            button.setText("Start live")
            button.setEnabled(False)
            button.setToolTip(f"{key}: {words}")

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def open_keys(self) -> list:
        return list(self._docks)

    def viewer(self, key: str):
        return self._docks[key][1].viewer

    def banner(self, key: str) -> QLabel:
        return self._docks[key][1].banner

    def banner_kind(self, key: str) -> str:
        return self._docks[key][1].banner_kind

    def source(self, key: str) -> PacedSource:
        return self._docks[key][2]

    def show_camera(self, key: str) -> None:
        """Open (or raise) ``key``'s view; start its live stream if it is idle or closed."""
        if key in self._docks:
            dock = self._docks[key][0]
            dock.show()
            dock.raise_()
            self.show()
            self.raise_()
            return
        inner = self._stream_factory(self.host, key)
        source = PacedSource(inner, self.frame_interval_s,
                             lambda f, k=key: self._frame_seen(k, f))
        viewer = self._viewer_factory(source)
        # The window's banner replaces the widget's (which shows the raw run tag and
        # calls every run "recorded"): collapsed, not removed, since the widget shows it.
        own = getattr(viewer, "banner", None)
        if isinstance(own, QLabel):
            own.setMaximumHeight(0)
        panel = _ViewPanel(viewer)
        panel.stream_button.clicked.connect(lambda _=False, k=key: self._toggle_stream(k))
        dock = _Dock(key, self)
        dock.setWidget(panel)
        dock.closing.connect(self._on_dock_closing)
        others = [d for d, _p, _s in self._docks.values()]
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
        if others:
            self.splitDockWidget(others[-1], dock, Qt.Orientation.Horizontal)
        self._docks[key] = (dock, panel, source)
        self._placeholder.hide()
        self._render_stream_button(key)
        viewer.open_camera()
        hs = str(self._entries.get(key, {}).get("host_state", ""))
        if self._start_streams and hs in ("idle", "closed"):
            self._request_stream(key, start=True)
        self.show()
        self.raise_()
        self.view_opened.emit(key)

    def close_camera(self, key: str) -> None:
        """End this window's view of ``key`` (the stream is left to the host)."""
        item = self._docks.pop(key, None)
        if item is None:
            return
        dock, panel, _source = item
        try:
            dock.closing.disconnect(self._on_dock_closing)
        except (TypeError, RuntimeError):
            pass
        try:
            panel.viewer.shutdown()
        except Exception:
            pass
        self._last_source.pop(key, None)
        self.removeDockWidget(dock)
        dock.deleteLater()
        if not self._docks:
            self._placeholder.show()
        self.view_closed.emit(key)

    def _on_dock_closing(self, key: str) -> None:
        self.close_camera(key)

    def closeEvent(self, event):
        for key in list(self._docks):
            self.close_camera(key)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Streams
    # ------------------------------------------------------------------

    def _toggle_stream(self, key: str) -> None:
        hs = str(self._entries.get(key, {}).get("host_state", ""))
        self._request_stream(key, start=hs != "streaming")

    def _request_stream(self, key: str, start: bool) -> None:
        what = "start" if start else "stop"
        try:
            fut = (self.host.start_stream if start else self.host.stop_stream)(key)
        except Exception as exc:
            self._on_note(key, f"could not {what} the live stream: {exc}")
            return
        add = getattr(fut, "add_done_callback", None)
        if add is None:
            return

        def done(f, key=key, what=what):
            try:
                exc = f.exception()
            except Exception as e:           # cancelled
                exc = e
            if exc is not None:
                try:
                    self._relay.note.emit(key, f"could not {what} the live stream: {exc}")
                except RuntimeError:
                    pass                    # the window is gone
        add(done)

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _frame_seen(self, key: str, frame) -> None:
        """On the viewer's frame thread: pass the frame's provenance on."""
        try:
            self._relay.seen.emit(key, str(getattr(frame, "source", "live")),
                                  getattr(frame, "run_tag", None))
        except RuntimeError:
            pass

    def _on_seen(self, key: str, source: str, run_tag) -> None:
        item = self._docks.get(key)
        if item is None:
            return
        panel = item[1]
        self._last_source[key] = (source, run_tag)
        kind, text = banner_for(source, run_tag, self._entries.get(key, {}).get("run_id"))
        if (kind, text) != (panel.banner_kind, panel.banner.text()):
            panel.banner_kind = kind
            panel.banner.setText(text)
            panel.banner.setStyleSheet(RUN_STYLE if kind == "run" else LIVE_STYLE)
            panel.banner.setToolTip(f"run tag {run_tag}" if run_tag else
                                    "Live frames are not saved anywhere.")
        if panel.note.isVisible():
            panel.note.hide()

    def _on_note(self, key: str, text: str) -> None:
        item = self._docks.get(key)
        if item is None:
            return
        item[1].note.setText(f"{key}: {text}")
        item[1].note.setStyleSheet("color: #ff7070;")
        item[1].note.show()

    def last_source(self, key: str):
        """``(source, run_tag)`` of the last frame shown for ``key``, or None."""
        return self._last_source.get(key)


__all__ = ["LiveViewWindow", "PacedSource", "banner_for", "run_id_of", "readonly_frame",
           "LIVE_TEXT", "LIVE_STYLE", "RUN_STYLE", "MAX_HZ", "MAX_HZ_DURING_RUN"]
