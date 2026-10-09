import sys
import atexit
import faulthandler
import logging
import threading
from queue import Queue
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStyle,
                             QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QSettings, QMetaObject
import time
import names

from waxa import ROI

from waxx.util.live_od.config import get_config, set_config
from waxx.util.live_od.camera_mother import CameraMother, CameraBaby, DataHandler, CameraNanny
from waxx.util.live_od.camera_connection_widget import CamConnBar
from waxx.util.live_od.gui.camera_control import CameraControl
from waxx.util.live_od.gui.viewer import LiveODViewer
from waxx.util.live_od.gui.analyzer import Analyzer
from waxx.util.live_od.gui.plotter import LiveODPlotter
from waxx.util.live_od.live_od_server import LiveODServer
from waxx.util.live_od.live_od_broadcaster import LiveODBroadcaster
from waxx.util.live_od.gui.live_scalar_plot_window import LiveScalarPlotWindow
from waxx.util.live_od.gui.tool_windows import ToolLauncher
from waxx.util.live_od.gui.fk_tof_window import FkTofWindow
from waxx.util.live_od.gui.adjust_panel import AdjustPanel
from waxx.util.live_od.gui.status_strip import StatusStrip
from waxx.util.live_od.log import get_logger, setup_logging, install_excepthooks, DEFAULT_LOG_DIR
from waxx.util.live_od import console_guard
from waxx.control.cameras import DummyCamera

logger = get_logger("window")

# Bounds for shutdown() (s). A console close gives the handler about 4 s
# (console_guard.DEFAULT_BUDGET_S) before Windows ends the process, so the cameras
# are closed first: started within SHUTDOWN_GRAB_WAIT_S, finished (or given up on)
# SHUTDOWN_CLOSE_CAMERAS_S later -- 3.8 s at most. The image writer gets its time
# while the cameras close.
SHUTDOWN_SERVER_WAIT_S = 1.5        # the REP loop polls every 0.5 s
SHUTDOWN_GRAB_WAIT_S = 0.8          # a grab loop polls its stop every <= 0.2 s
SHUTDOWN_CLOSE_CAMERAS_S = 3.0      # Andor: stop, shutter closed, SDK close
SHUTDOWN_WRITER_WAIT_S = 3.0        # the image writer closing the run's file
SHUTDOWN_DUMP_AFTER_S = 20          # faulthandler dumps every thread's stack if still going
# The camera threads' own last words when the shutdown stops them.
SHUTDOWN_STOP_REASON = "liveOD is shutting down"
# A camera whose last live view closed stops streaming once nothing else subscribes
# to it. Its own view lets go a moment after it closes (on the viewer's pool); one
# that still has subscribers after LIVE_VIEW_STOP_WAIT_S streams on for them.
LIVE_VIEW_STOP_WAIT_S = 3.0
LIVE_VIEW_STOP_POLL_MS = 100


def _ms_left(deadline: float) -> int:
    """Milliseconds to ``deadline`` (time.monotonic()), for QThread.wait; >= 0."""
    return max(0, int((deadline - time.monotonic()) * 1000))


class BackgroundPoll(QObject):
    """Runs ``fn`` off the GUI thread and delivers its result as ``result``
    (queued to the receiver's thread). At most one call is in flight: ``poll``
    while one is running does nothing and returns False, so a call stuck on a
    stalled network drive never piles up threads behind it. An exception from
    ``fn`` is delivered as None.

    For the next-run-id label: server_talk.get_run_id reads a file on the data
    drive, and an SMB stall there used to freeze the window (2026-10-05)."""

    result = pyqtSignal(object)

    def __init__(self, fn, name: str = "liveod-poll", parent=None):
        super().__init__(parent)
        self._fn = fn
        self._name = name
        self._lock = threading.Lock()
        self._in_flight = False
        self.skipped = 0            # polls skipped because the last was still running

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def poll(self) -> bool:
        with self._lock:
            if self._in_flight:
                self.skipped += 1
                return False
            self._in_flight = True
        try:
            threading.Thread(target=self._work, name=self._name, daemon=True).start()
        except Exception:
            with self._lock:
                self._in_flight = False
            raise
        return True

    def _work(self):
        try:
            value = self._fn()
        except Exception:
            value = None
        with self._lock:
            self._in_flight = False
        try:
            self.result.emit(value)
        except RuntimeError:
            pass                    # the window (and this object) is gone


def _stop_camera_thread(thread, reason: str):
    """``thread.request_stop(reason=reason)``; a camera thread whose request_stop
    takes no reason is stopped all the same."""
    try:
        thread.request_stop(reason=reason)
    except TypeError:
        thread.request_stop()

class LiveODWindow(QWidget):
    interrupt = pyqtSignal()
    def __init__(self, config=None, settings=None, log_dir=DEFAULT_LOG_DIR):
        """``config``: the lab's LiveODConfig. Default: the active one
        (waxx.util.live_od.config.set_config, called by the lab's launcher).
        ``settings``: a QSettings to remember the layout in; None remembers nothing.
        ``log_dir``: where the rotating log file goes; None for no file."""

        # Checked before any Qt object exists, so a missing config is a readable
        # error rather than a half-built window.
        if config is not None:
            set_config(config)
        config = get_config()
        if config.data_saver is None or config.run_id_source is None:
            raise RuntimeError(
                "LiveODWindow needs a LiveODConfig with data_saver and run_id_source. "
                "Start liveOD through the lab's launcher (kexp: "
                "`python -m kexp.util.live_od.gui.main_window` or live_od.bat), which "
                "calls waxx.util.live_od.config.set_config first.")

        super().__init__()

        self.config = config
        self.server_talk = self.config.run_id_source
        self._settings = settings

        # Before anything that logs: the terminal, the log file and the GUI panel
        # all hang off this, and uncaught exceptions are routed into it too.
        self._qt_log_handler = setup_logging(log_dir)
        install_excepthooks()

        self.queue = Queue()
        self.camera_nanny = CameraNanny()
        # CameraMother is kept as a no-op stub for import compatibility.
        self.camera_mother = CameraMother(output_queue=self.queue,
                                          camera_nanny=self.camera_nanny,
                                          server_talk=self.server_talk)

        self.the_baby = None
        self.data_handler = None
        # The camera threads of the latest camera run. Unlike the two above, not
        # cleared when the run ends: the next spawn_baby uses it to cut the old
        # threads off from the new run (they may still be finishing).
        self._run_threads = (None, None)
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        self.last_camera = ""
        self.img_count = 0
        self.img_count_run = 0
        self._run_active = False   # True between INIT_RUN and END_RUN/reset
        self._run_was_reset = False  # True when reset() kills an active no-camera run
        # LiveODConfig.use_camera_host: a CameraHost owns the cameras (built here,
        # started once the server exists); None: CameraNanny, as before
        self.camera_host = self._build_camera_host()
        self._camera_host_bridge = None
        self.setup_widgets()
        self.setup_layout()
        self._qt_log_handler.record_signal.connect(self._on_log_record)

        # ZMQ REP server — drives all run lifecycle events.
        self.data_saver = self.config.data_saver
        self.live_od_server = LiveODServer(
            self.server_talk, self.data_saver,
            0,
        )
        self.live_od_server.new_run_signal.connect(self.spawn_baby)
        self.live_od_server.shot_progress_signal.connect(self.on_shot_progress)
        self.live_od_server.shot_progress_signal.connect(
            lambda _idx, _total, xvars: self.analyzer.set_xvar_values(xvars)
        )
        # after set_xvar_values (connection order = delivery order): the Analyzer
        # emits the shot's atom numbers now, with its field-dependent cross section
        self.live_od_server.shot_conditions_signal.connect(self.analyzer.set_shot_conditions)
        self.live_od_server.shot_timing_signal.connect(self.on_shot_timing)
        self.live_od_server.run_done_signal.connect(self.on_run_done)
        self.live_od_server.exited_run_signal.connect(self.on_run_exited)
        self.live_od_server.reset_signal.connect(self.reset)
        self.live_od_server.camera_control_signal.connect(self.on_remote_camera_control)
        self.live_od_server.run_state_signal.connect(self.on_run_state)
        self.live_od_server.set_camera_state_provider(self._camera_state_report)
        self.live_od_server.start()
        self._start_camera_host()

        # The numbers in the corner of the OD image need the per-shot scalars
        # whether or not a Live Plot window is open (~4 ms a shot).
        self.live_od_server.register_scalar_subscription('fits')
        self.analyzer.shot_scalars_signal.connect(self.viewer_window.on_shot_scalars)

        # Give the Analyzer a reference to the server for subscription queries
        self.analyzer.set_server(self.live_od_server)

        # ZMQ PUB broadcaster — forwards OD images and run events to remote viewers.
        self.broadcaster = LiveODBroadcaster()
        self.broadcaster.start()
        self.live_od_server.run_started_signal.connect(self.broadcaster.broadcast_run_started)
        self.live_od_server.shot_progress_signal.connect(self.broadcaster.broadcast_shot_progress)
        self.live_od_server.run_done_signal.connect(self.broadcaster.broadcast_run_done)
        self.live_od_server.run_state_signal.connect(self.broadcaster.broadcast_run_state)
        self.live_od_server.markers_changed_signal.connect(self._on_remote_markers_changed)
        self.viewer_window.markers_changed.connect(self._on_viewer_markers_changed)
        self.live_od_server.adjust_specs_signal.connect(self._on_adjust_specs)
        self.live_od_server.shot_adjust_values_signal.connect(self.broadcaster.broadcast_adjust_values)
        self.live_od_server.shot_adjust_values_signal.connect(self._adjust_panel.update_values)
        # a notice of each push during the run (diagnostic frames, scope
        # traces; no arrays) goes out to remote viewers whether the run saves
        # or not, straight from the server thread (no hop through this one)
        self.live_od_server.set_aux_notice_sink(self.broadcaster.broadcast_aux_notice)
        self.analyzer.broadcast_signal.connect(self.broadcaster.broadcast_od_image)
        self.analyzer.shot_scalars_signal.connect(self.live_scalar_plot_window.on_shot_scalars)
        self.analyzer.shot_scalars_signal.connect(self.broadcaster.broadcast_shot_scalars)
        self.analyzer.fk_tof_signal.connect(self.broadcaster.broadcast_fk_tof)
        self.live_scalar_plot_window.subscription_changed_signal.connect(
            self._on_scalar_subscription_changed
        )

        # Broadcast camera-button state changes so remote viewers can mirror
        # the open/closed/grabbing UI.
        for btn in self.camera_conn_bar.buttons:
            btn.state_changed.connect(self._broadcast_camera_states)
        # Periodic re-broadcast so late-joining remote viewers learn current
        # state without needing to click anything.
        self._camera_state_timer = QTimer(self)
        self._camera_state_timer.timeout.connect(self._broadcast_camera_states)
        self._camera_state_timer.timeout.connect(self._rebroadcast_markers)     # same reason
        self._camera_state_timer.start(2000)
        # Initial broadcast (will reach any already-subscribed viewers).
        QTimer.singleShot(500, self._broadcast_camera_states)

    def _broadcast_camera_states(self, *_):
        try:
            extra = {}
            host = getattr(self, 'camera_host', None)
            if host is not None:
                # additive: which cameras have Persist on (remote viewers ignore it
                # until they know it)
                extra["persist"] = {k: c["persist"] for k, c in host.snapshot()["cameras"].items()}
            self.broadcaster.broadcast_camera_state(
                self.camera_conn_bar.get_states(), **extra)
        except Exception as e:
            logger.debug(f"camera-state broadcast error: {e}")

    # ------------------------------------------------------------------
    # The camera host (LiveODConfig.use_camera_host)
    # ------------------------------------------------------------------

    def _build_camera_host(self):
        """The CameraHost when the lab turned it on, else None. Nothing is
        opened here: _start_camera_host starts it once the server exists."""
        if not getattr(self.config, 'use_camera_host', False):
            return None
        factory = getattr(self.config, 'camera_host_factory', None)
        if factory is None:
            from waxx.util.live_od.camera_host import CameraHost
            factory = CameraHost
        host = factory(self.config)
        logger.warning("liveOD camera host is ON (LiveODConfig.use_camera_host): liveOD owns "
                       "its cameras through one worker thread each and serves them to other "
                       "programs; set use_camera_host=False and restart liveOD to go back.")
        return host

    def _make_camera_bar(self):
        if self.camera_host is not None:
            from waxx.util.live_od.camera_host.bar import HostCameraBar
            return HostCameraBar(self.camera_host, self.output_window)
        return CamConnBar(self.camera_nanny, self.output_window)

    def _start_camera_host(self):
        host = self.camera_host
        if host is None:
            return
        self.live_od_server.set_camera_host(host)
        from waxx.util.live_od.camera_host.qt_bridge import HostQtBridge
        self._camera_host_bridge = HostQtBridge(host, parent=self)
        self._camera_host_bridge.snapshot_changed.connect(self.camera_conn_bar.apply_snapshot)
        self._camera_host_bridge.snapshot_changed.connect(self.camera_menu.set_snapshot)
        try:
            host.start()
        except Exception as exc:
            self.msg(f"The camera host did not start ({type(exc).__name__}: {exc}); runs that "
                     f"use a camera will be refused until liveOD is restarted.", logging.ERROR)
        snapshot = host.snapshot()
        self.camera_conn_bar.apply_snapshot(snapshot)
        self.camera_menu.set_snapshot(snapshot)

    def _nanny_for_run(self):
        """The camera thread's nanny for the next camera run spawned (host mode:
        that run's token, in INIT_RUN order), else the window's CameraNanny."""
        if getattr(self, 'camera_host', None) is None:
            return self.camera_nanny
        return self._nanny_for_token(self.live_od_server.take_spawn_token())

    def _nanny_for_token(self, run_token: str):
        """The camera thread's nanny: in host mode a HostNanny for the run with
        ``run_token``, else the window's CameraNanny."""
        host = getattr(self, 'camera_host', None)
        if host is None:
            return self.camera_nanny
        from waxx.util.live_od.camera_host.legacy import HostNanny
        return HostNanny(host, run_token)

    def _needs_grab_drain(self, camera_key: str) -> bool:
        return (getattr(self, 'camera_host', None) is None
                and self.config.camera_needs_grab_drain(camera_key))

    # ------------------------------------------------------------------
    # The camera control, its settings dialogs and the live view (host mode)
    # ------------------------------------------------------------------

    def _make_camera_control(self):
        """CameraControl fed by the camera host's snapshots, or (no host) by the
        CamConnBar's buttons' state words, without ⚙ and 🎥 (they need the host):
        the same button in both modes."""
        keys = [b.camera_name for b in self.camera_conn_bar.buttons]
        if self.camera_host is None:
            control = CameraControl(keys, glyphs=False)
            control.set_states(self.camera_conn_bar.get_states())   # those opened on start
            for btn in self.camera_conn_bar.buttons:
                btn.state_changed.connect(control.set_state)
            control.toggle_requested.connect(self._on_camera_toggle_requested)
            return control
        control = CameraControl(keys, expect_snapshots=True)
        control.action_requested.connect(self._on_camera_action_requested)
        control.settings_requested.connect(self._open_camera_settings)
        control.live_view_requested.connect(self._on_live_view_requested)
        return control

    def _on_camera_action_requested(self, camera_key: str, action: str):
        """CameraControl's main button (it has already asked where it must): a
        CameraHost.request, under the same rule as a remote CAMERA_CONTROL. The
        host logs a failure itself."""
        from waxx.util.live_od.gui.camera_control import HOST_REQUEST
        request = HOST_REQUEST.get(action)
        if request is None:
            self.msg(f"Camera {camera_key}: unknown action {action!r}", logging.WARNING)
            return
        server = getattr(self, 'live_od_server', None)
        if server is not None:
            ok, reason = server.camera_action_allowed(camera_key, request)
            if not ok:
                self.msg(f"Camera {camera_key}: {reason}", logging.WARNING)
                return
        self.msg(f"Camera {camera_key}: {action} (liveOD window)")
        try:
            self.camera_host.request(camera_key, request, origin=f"liveOD window: {action}")
        except Exception as exc:
            self.msg(f"Camera {camera_key}: {action} refused: {exc}", logging.WARNING)

    def _open_camera_settings(self, camera_key: str):
        """CameraControl's cog: the camera's settings, one non-modal dialog per camera
        (on its read-only Run tab while a run holds the camera)."""
        from waxx.util.live_od.gui.camera_settings_dialog import CameraSettingsDialog
        dialog = self._camera_dialogs.get(camera_key)
        if dialog is None:
            dialog = CameraSettingsDialog(
                camera_key, self.camera_host, self,
                snapshot_signal=self._camera_host_bridge.snapshot_changed)
            self._camera_dialogs[camera_key] = dialog
        elif dialog.is_locked():
            dialog.tabs.setCurrentIndex(dialog.run_tab_index)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _live_view(self):
        """The live view window, made on first use."""
        if self.live_view_window is None:
            from waxx.util.live_od.gui.live_view_window import LiveViewWindow
            from waxx.util.live_od.gui.status_strip import ACTIVE_STATES
            window = LiveViewWindow(self.camera_host,
                                    snapshot_signal=self._camera_host_bridge.snapshot_changed)
            window.view_opened.connect(lambda key: self.camera_menu.set_live_view_open(key, True))
            window.view_closed.connect(self._on_live_view_closed)
            # 2 Hz while a run is on (the GUI thread has the run's shots to draw)
            window.set_run_active(self._run_is_active())
            self.live_od_server.run_state_signal.connect(
                lambda state, _detail, w=window: w.set_run_active(state in ACTIVE_STATES))
            window.setWindowIcon(self.windowIcon())
            self.live_view_window = window
        return self.live_view_window

    def _on_live_view_requested(self, camera_key: str, open_: bool):
        """CameraControl's movie camera: open (or close) that camera's view."""
        window = self._live_view()
        if open_:
            window.show_camera(camera_key)
        else:
            window.close_camera(camera_key)

    def _on_live_view_closed(self, camera_key: str):
        """A camera's view in the live window closed. Once nothing else subscribes
        to the camera and no run holds it, its live stream is stopped
        (_check_live_streams): the camera -- the Andor, at live EM gain -- is not
        left streaming for nobody after the operator closes its view."""
        self.camera_menu.set_live_view_open(camera_key, False)
        self._live_stop_pending[camera_key] = time.monotonic() + LIVE_VIEW_STOP_WAIT_S
        self._check_live_streams()

    def _check_live_streams(self):
        """For each camera whose last view closed: stop its live stream when it
        streams with no subscriber (its own view lets go on the viewer's pool, a
        moment after the view closed, so this looks again every
        LIVE_VIEW_STOP_POLL_MS). Leave it be when a run holds it, it no longer
        streams, its view was opened again, or someone else still watches it
        after LIVE_VIEW_STOP_WAIT_S (then the stream is theirs)."""
        pending = self._live_stop_pending
        host = getattr(self, 'camera_host', None)
        if host is not None and pending:
            from waxx.util.live_od.gui.camera_control import RUN_PHASES
            try:
                cams = host.snapshot()["cameras"]
            except Exception as exc:
                logger.warning(f"live view: could not read the camera host's state ({exc}); "
                               f"streams of closed views are left running")
                cams = {}
                pending.clear()
            window = getattr(self, 'live_view_window', None)
            open_keys = set(window.open_keys()) if window is not None else set()
            now = time.monotonic()
            for key, deadline in list(pending.items()):
                c = cams.get(key) or {}
                hs = str(c.get("host_state", ""))
                n_subs = int(c.get("n_subs") or 0)
                if key in open_keys or c.get("locked") or hs in RUN_PHASES or hs != "streaming":
                    pending.pop(key, None)
                elif n_subs == 0:
                    pending.pop(key, None)
                    self._stop_unwatched_stream(key)
                elif now >= deadline:
                    pending.pop(key, None)
                    logger.info(f"{key}: its live view closed, but {n_subs} other "
                                f"subscriber(s) still watch it; its live stream goes on.")
        else:
            pending.clear()
        timer = self._live_stop_timer
        if pending and timer is None:
            timer = self._live_stop_timer = QTimer(self)
            timer.setInterval(LIVE_VIEW_STOP_POLL_MS)
            timer.timeout.connect(self._check_live_streams)
        if timer is not None:
            if pending and not timer.isActive():
                timer.start()
            elif not pending and timer.isActive():
                timer.stop()

    def _stop_unwatched_stream(self, camera_key: str):
        logger.info(f"{camera_key}: its last live view closed and nothing else subscribes "
                    f"to it: stopping its live stream.")
        try:
            fut = self.camera_host.stop_stream(camera_key)
        except Exception as exc:
            logger.warning(f"{camera_key}: could not stop its live stream: {exc}")
            return

        def done(f, key=camera_key):
            try:
                exc = f.exception()
            except Exception as e:          # cancelled
                exc = e
            if exc is not None:
                logger.warning(f"{key}: could not stop its live stream: {exc}")
        add = getattr(fut, "add_done_callback", None)
        if add is not None:
            add(done)

    def _shutdown_camera_host(self, timeout_s: float):
        bridge = getattr(self, '_camera_host_bridge', None)
        if bridge is not None:
            bridge.close()
        host = getattr(self, 'camera_host', None)
        if host is None:
            return
        report = host.shutdown(timeout_s)
        if not report.get("cameras_closed", True) or report.get("errors"):
            logger.warning(f"shutdown: the camera host did not close cleanly: {report}")

    def _camera_state_report(self) -> dict:
        """For the server's POLL reply: each camera's button state, plus what a
        remote tool needs to find the same device on its beacon server (type,
        serial).  Called from the server thread: plain attribute reads only.
        In host mode the host's own report (the same keys, plus host_state,
        persist, holder, n_subs)."""
        host = getattr(self, 'camera_host', None)
        if host is not None:
            return host.poll_cameras()
        out = {}
        for btn in self.camera_conn_bar.buttons:
            p = btn.camera_params
            ctype = getattr(p, "camera_type", "") or ""
            serial = getattr(p, "serial_no", "") or ""
            if isinstance(ctype, bytes):
                ctype = ctype.decode()
            if isinstance(serial, bytes):
                serial = serial.decode()
            out[btn.camera_name] = {"state": btn.state, "camera_type": str(ctype),
                                    "serial_no": str(serial)}
        return out

    def on_remote_camera_control(self, camera_key: str, action: str):
        """Slot for ``LiveODServer.camera_control_signal``.

        Routes the request to the matching ``CameraButton`` on the local
        ``CamConnBar``.  Runs on the GUI thread (PyQt widget state is not
        thread-safe).  During a run the server lets through only a *close* of
        a camera the run is not using (a Basler close is quick; the run's own
        camera and every open are refused there), so a blocking driver call
        here cannot stall SHOT_COMPLETE / RESET signals during acquisition.
        """
        btn = self.camera_conn_bar.get_button(camera_key)
        if btn is None:
            self.msg(f"Remote camera control: unknown camera {camera_key!r}")
            return
        try:
            if action == 'open':
                if not btn.camera.is_opened():
                    btn.open_camera()
            elif action == 'close':
                if btn.camera.is_opened():
                    btn.close_camera()
            else:  # toggle
                btn.button_pressed()
        except Exception as e:
            self.msg(f"Remote camera control error ({camera_key}/{action}): {e}", logging.ERROR)
        finally:
            self._broadcast_camera_states()

    def _on_camera_toggle_requested(self, camera_key: str):
        """The status row's camera button, or its drop-down: connect / disconnect.
        Under the same rule as a remote CAMERA_CONTROL (camera_action_allowed):
        during a run only a camera the run does not use may be closed."""
        btn = self.camera_conn_bar.get_button(camera_key)
        if btn is None:
            return
        try:
            action = "close" if btn.camera.is_opened() else "open"
        except Exception:
            action = "toggle"
        server = getattr(self, 'live_od_server', None)
        if server is not None:
            ok, reason = server.camera_action_allowed(camera_key, action)
            if not ok:
                self.msg(f"Camera {camera_key}: {reason}", logging.WARNING)
                return
        try:
            btn.button_pressed()
        except Exception as e:
            self.msg(f"Camera {camera_key}: {e}", logging.ERROR)

    def update_run_id_label(self):
        """Ask for the next run id in the background (BackgroundPoll); the label
        changes when the answer comes. Skipped while the last ask is still out."""
        poller = getattr(self, '_run_id_poll', None)
        if poller is None:
            poller = self._run_id_poll = BackgroundPoll(self.server_talk.get_run_id,
                                                        name="liveod-run-id", parent=self)
            poller.result.connect(self._on_run_id_polled)
        poller.poll()

    def _on_run_id_polled(self, rid):
        self.status_strip.set_next_run_id(rid)

    def setup_widgets(self):
        self.server_talk.check_for_mapped_data_dir()

        self.viewer_window = LiveODViewer()
        # the progress bar in the status strip carries the shot count here
        self.viewer_window.image_count_label.hide()
        self.setup_status_strip()
        self.setup_output_window()
        self.setup_run_buttons()
        # CamConnBar owns the cameras (a CameraButton each: open/close, state) but is
        # not shown: the camera button in the status row shows and drives it
        self.camera_conn_bar = self._make_camera_bar()
        self.camera_conn_bar.setParent(self)
        self.camera_conn_bar.hide()
        # the status row's camera button: CameraControl in both modes (settings, live
        # view and Persist only when the camera host owns the cameras)
        self.camera_menu = self._make_camera_control()
        self.status_strip.add_camera_widget(self.camera_menu)
        self.live_view_window = None
        self._camera_dialogs = {}
        # cameras whose last live view closed: camera_key -> until when (monotonic)
        # to wait for its other subscribers to go (_check_live_streams)
        self._live_stop_pending = {}
        self._live_stop_timer = None

        self.plotting_queue = Queue()
        self.analyzer = Analyzer(self.plotting_queue, self.viewer_window)
        self.plotter = LiveODPlotter(self.viewer_window, self.plotting_queue)
        self.plotter.start()

        # Scalar plot window — created once, shown on demand via Live Plot button
        self.live_scalar_plot_window = LiveScalarPlotWindow()
        self.viewer_window.live_plot_requested.connect(self._open_live_scalar_plot)

        self.fk_tof_window = FkTofWindow()
        self.viewer_window.fk_tof_requested.connect(self._open_fk_tof)
        self.analyzer.fk_tof_signal.connect(self.fk_tof_window.on_pwa_data)

        # Adjust panel
        self._adjust_panel = AdjustPanel()
        self._adjust_panel.value_changed_signal.connect(self._on_adjust_value_changed)
        self._adjust_panel.spec_updated_signal.connect(self._on_adjust_spec_updated)
        self._adjust_button = QPushButton("Adjust")
        self._adjust_button.setMinimumHeight(40)
        self._adjust_button.setEnabled(False)
        self._adjust_button.clicked.connect(self._open_adjust_panel)

    def setup_run_buttons(self):
        """The old 'Reset' button aborted the run if there was one and otherwise
        skipped a run ID. Skipping an ID by hand is not something anyone does, so the
        button only aborts. It is never disabled: if this window's idea of whether a
        run is active were ever wrong, a disabled abort button would be the worst way
        to find out. (A remote viewer's RESET still reaches reset() unchanged.)"""
        self.abort_button = QPushButton('Abort')
        self.abort_button.setToolTip(
            "Stop the run now and delete its data file; ignored while a run is being saved "
            "(it is complete then). No confirmation: it has to be fast.")
        self.abort_button.clicked.connect(self._on_abort_clicked)
        # fixed width: the button turns bold red during a run, and nothing that
        # happens when a run starts may change the window's width
        self.abort_button.setFixedWidth(56)
        self.fix_button = self.abort_button     # the name it had as 'Reset'
        self._update_run_buttons()

    def _run_is_active(self) -> bool:
        server = getattr(self, 'live_od_server', None)
        return (bool(getattr(self, '_run_active', False))
                or getattr(self, 'the_baby', None) is not None
                or bool(server is not None and server._run_in_progress))

    def _on_abort_clicked(self):
        if self._run_is_active():
            self.reset()
        else:
            self.msg("Abort: there is no run to abort.")

    def _update_run_buttons(self):
        self.abort_button.setStyleSheet(
            'background-color: #e53935; color: white; font-weight: bold;'
            if self._run_is_active() else '')

    def setup_output_window(self):
        self.output_window = self.viewer_window.output_window

    def setup_status_strip(self):
        self.status_strip = StatusStrip()
        self.status_strip.title_changed.connect(self._on_status_title)
        self.update_run_id_label()
        # Timer for periodic update
        self.run_id_timer = QTimer(self)
        self.run_id_timer.timeout.connect(self.update_run_id_label)
        self.run_id_timer.start(2000)  # 2 seconds

    def setup_layout(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 5, 6, 6)
        layout.setSpacing(3)
        status_row = QHBoxLayout()
        status_row.setSpacing(4)
        status_row.addWidget(self.status_strip, 1)
        status_row.addWidget(self.abort_button)
        layout.addLayout(status_row)
        self._adjust_button.setMinimumHeight(0)     # sized like its toolbar neighbours
        self._adjust_button.setFixedWidth(72)       # "Adjust (12)" fits: no growth at run start
        self.viewer_window.add_window_button(self._adjust_button)
        # the lab's tool windows (LiveODConfig.tool_windows), each its own process
        self._tool_launcher = ToolLauncher(get_config().tool_windows, self.msg)
        for button in self._tool_launcher.buttons():
            self.viewer_window.add_window_button(button)
        layout.addWidget(self.viewer_window, 1)
        self.setLayout(layout)
        if self._settings is not None:
            self.viewer_window.attach_settings(self._settings)
            geometry = self._settings.value("window/geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)

    # ------------------------------------------------------------------
    # Status, log
    # ------------------------------------------------------------------

    def on_run_state(self, state: str, detail: str):
        """Slot for ``LiveODServer.run_state_signal``."""
        self.status_strip.set_state(state, detail)
        if state in ("saved", "done", "error"):
            QApplication.alert(self, 0)     # flash the taskbar entry until focused

    # ------------------------------------------------------------------
    # Markers: drawn by the viewer, kept per camera by the server, shared with
    # the remote viewers
    # ------------------------------------------------------------------

    def _show_markers_for(self, camera_key: str):
        try:
            markers = self.live_od_server.markers.get(camera_key) if camera_key else []
        except Exception as exc:
            self.msg(f"Could not read the markers for {camera_key}: {exc}", logging.WARNING)
            markers = []
        self.viewer_window.set_markers(markers)
        self.broadcaster.broadcast_markers(camera_key, markers)

    def _on_viewer_markers_changed(self, camera_key: str, markers: list):
        """A pin was edited in this window."""
        if not camera_key:
            return      # no run yet, so no camera to keep them under
        try:
            stored = self.live_od_server.set_markers(camera_key, markers)
        except Exception as exc:
            self.msg(f"Could not store the markers for {camera_key}: {exc}", logging.WARNING)
            return
        self.broadcaster.broadcast_markers(camera_key, stored)

    def _on_remote_markers_changed(self, camera_key: str, markers: list):
        """A pin was edited in a remote viewer (the server has already stored it)."""
        if camera_key == self.viewer_window._camera_key:
            self.viewer_window.set_markers(markers)
        self.broadcaster.broadcast_markers(camera_key, markers)

    def _rebroadcast_markers(self):
        camera_key = self.viewer_window._camera_key
        if camera_key:
            self.broadcaster.broadcast_markers(camera_key, self.viewer_window.get_markers())

    def _on_status_title(self, text: str):
        base = self.config.window_title
        self.setWindowTitle(f"{base} — {text}" if text else base)

    def _on_log_record(self, levelno: int, text: str, created: float):
        """Every ``waxx.live_od`` record, on the GUI thread: into the panel, and out
        to the remote viewers."""
        try:
            self.output_window.append_record(levelno, text, created)
            if hasattr(self, 'broadcaster'):
                self.broadcaster.broadcast_log_msg(text, levelno)
            if levelno >= logging.ERROR:
                QApplication.alert(self, 0)
        except Exception as exc:
            # not through the logger: an error here would come straight back here
            print(f"[LiveODWindow] could not show a log record: {exc}", file=sys.stderr)

    def closeEvent(self, event):
        # D-e: closing during a run asks first (default: keep liveOD running)
        if not self._shutdown_done and self._run_is_active():
            server = getattr(self, 'live_od_server', None)
            run_id = getattr(server, '_current_run_id', 0) if server is not None else 0
            run_name = f"run {run_id}" if run_id else "an unsaved run (save_data=False)"
            answer = QMessageBox.question(
                self, "Close liveOD during a run?",
                f"liveOD is in the middle of {run_name}.\n\n"
                f"Closing liveOD now stops its camera and the server: the experiment's "
                f"next message to liveOD fails, and the run is not saved complete.\n\n"
                f"Close liveOD anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                self.msg(f"Close cancelled: liveOD is in the middle of {run_name}.")
                event.ignore()
                return
            self.msg(f"Closing liveOD in the middle of {run_name} (confirmed).", logging.WARNING)
        if self._settings is not None:
            self._settings.setValue("window/geometry", self.saveGeometry())
            self.viewer_window.save_settings()
        # the live view first: its viewers let go of the camera host before it stops
        for window in (getattr(self, 'live_view_window', None),
                       *getattr(self, '_camera_dialogs', {}).values()):
            try:
                if window is not None:
                    window.close()
            except Exception:
                pass
        self.shutdown("window closed")
        # liveOD's own windows go with it, so nothing keeps the process alive headless
        for window in (getattr(self, 'live_scalar_plot_window', None),
                       getattr(self, 'fk_tof_window', None),
                       getattr(self, '_adjust_panel', None)):
            try:
                if window is not None:
                    window.close()
            except Exception:
                pass
        super().closeEvent(event)

    def shutdown(self, reason: str = ""):
        """Stop serving and close the cameras, so that nothing is left holding
        them when liveOD exits. Idempotent, safe from any thread (the console
        handler calls it from its own), and bounded: each wait has a limit, and
        faulthandler dumps every thread's stack if the whole thing hangs.

        Order: the server and broadcaster stop (no new run can start, the
        experiment's next message fails); the camera threads are asked to stop
        (quietly: nothing is discarded) and given SHUTDOWN_GRAB_WAIT_S; then every
        camera is closed through its driver's safe Close() (Andor: acquisition
        stopped, shutter closed, SDK closed) on a helper thread, while the image
        dispatcher and writer get their time to close the run's file. The cameras
        come first: a console close leaves about 4 s before Windows ends the
        process, and the writer needs no camera. Each step is guarded on its own,
        so nothing before the camera close (a stack-dump watchdog with no stderr
        to write to, a thread that will not stop) can skip it.
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        # the window lives on the main thread; the console handler calls from its own
        on_gui_thread = threading.current_thread() is threading.main_thread()
        try:
            try:
                faulthandler.dump_traceback_later(SHUTDOWN_DUMP_AFTER_S, exit=False)
            except Exception as exc:
                # e.g. no stderr (pythonw): the shutdown goes on without the watchdog
                logger.warning(f"shutdown: no stack-dump watchdog ({type(exc).__name__}: {exc})")
            logger.warning(f"liveOD shutting down ({reason or 'no reason given'}).")
            server = getattr(self, 'live_od_server', None)
            broadcaster = getattr(self, 'broadcaster', None)
            handlers = self._shutdown_stop_threads(on_gui_thread, server, broadcaster)
            closer = self._start_closing_cameras(SHUTDOWN_CLOSE_CAMERAS_S)
            self._shutdown_wait_for_writer(server, handlers)
            self._finish_closing_cameras(closer)
            for thread in (server, broadcaster):
                try:
                    if thread is not None and thread.isRunning():
                        thread.wait(int(SHUTDOWN_SERVER_WAIT_S * 1000))
                except Exception:
                    pass
            logger.info("liveOD shut down.")
        except Exception as exc:
            logger.exception(f"shutdown: {exc}")
        finally:
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass

    def _shutdown_stop_threads(self, on_gui_thread: bool, server, broadcaster) -> set:
        """Shutdown, first step: the timers, the server and broadcaster, and the
        camera threads (asked to stop, quietly; waited for SHUTDOWN_GRAB_WAIT_S in
        all). Returns the image dispatchers, told the grab is over. Never raises."""
        if on_gui_thread:
            for timer in (getattr(self, 'run_id_timer', None),
                          getattr(self, '_camera_state_timer', None),
                          getattr(self, '_live_stop_timer', None)):
                try:
                    if timer is not None:
                        timer.stop()
                except Exception:
                    pass
        for thread in (server, broadcaster):
            try:
                if thread is not None:
                    thread.stop()
            except Exception as exc:
                logger.warning(f"shutdown: could not stop {type(thread).__name__}: {exc}")
        baby, handler = getattr(self, '_run_threads', (None, None))
        babies = {baby, getattr(self, 'the_baby', None)} - {None}
        handlers = {handler, getattr(self, 'data_handler', None)} - {None}
        for b in babies:
            try:
                _stop_camera_thread(b, SHUTDOWN_STOP_REASON)
            except Exception as exc:
                logger.warning(f"shutdown: could not stop camera thread: {exc}")
        for h in handlers:
            try:
                h.grab_finished()       # drain what is queued, then close
            except Exception:
                pass
        deadline = time.monotonic() + SHUTDOWN_GRAB_WAIT_S
        for b in babies:
            try:
                if b.isRunning() and not b.wait(_ms_left(deadline)):
                    logger.warning(f"shutdown: camera thread {getattr(b, 'name', '?')} still "
                                   f"running after {SHUTDOWN_GRAB_WAIT_S:g} s; closing the "
                                   f"cameras anyway")
            except Exception as exc:
                logger.warning(f"shutdown: camera thread: {exc}")
        return handlers

    def _shutdown_wait_for_writer(self, server, handlers):
        """Shutdown, while the cameras close: the image dispatchers and the
        writer get SHUTDOWN_WRITER_WAIT_S to close the run's file. Never raises."""
        deadline = time.monotonic() + SHUTDOWN_WRITER_WAIT_S
        for h in handlers:
            try:
                if h.isRunning() and not h.wait(_ms_left(deadline)):
                    logger.warning("shutdown: image dispatcher still running")
            except Exception:
                pass
        wait_writer = getattr(server, 'wait_for_image_writer', None)
        try:
            if callable(wait_writer) and not wait_writer(max(0.0, deadline - time.monotonic())):
                logger.warning(f"shutdown: the image writer had not closed the run's file "
                               f"after {SHUTDOWN_WRITER_WAIT_S:g} s")
        except Exception as exc:
            logger.warning(f"shutdown: waiting for the image writer: {exc}")

    def _start_closing_cameras(self, timeout_s: float):
        """Close every camera on a helper thread -- the camera host's (host mode)
        and the nanny's, each through its driver's safe Close() -- and return
        what _finish_closing_cameras waits for, at most ``timeout_s`` from now (a
        driver call that hangs must not hang the exit). Never raises."""
        host = getattr(self, 'camera_host', None)
        nanny = getattr(self, 'camera_nanny', None)
        results = {}

        def close():
            if host is not None:
                # host mode: the host closes its cameras (Andor: stop, shutter
                # closed, close) and gives borrowed Baslers back, bounded
                try:
                    self._shutdown_camera_host(timeout_s)
                except Exception as exc:
                    logger.warning(f"shutdown: camera host: {exc}")
            if nanny is not None:
                try:
                    results.update(nanny.close_all() or {})
                except Exception as exc:
                    logger.warning(f"shutdown: closing the cameras failed: {exc}")
        t = threading.Thread(target=close, name="liveOD-close-cameras", daemon=True)
        try:
            t.start()
        except Exception as exc:
            logger.warning(f"shutdown: could not start closing the cameras: {exc}")
            return None
        return t, results, nanny, time.monotonic() + timeout_s, timeout_s

    def _finish_closing_cameras(self, closer):
        """Wait for _start_closing_cameras' helper, until its deadline."""
        if closer is None:
            return
        t, results, nanny, deadline, timeout_s = closer
        t.join(max(0.0, deadline - time.monotonic()))
        if t.is_alive():
            logger.warning(f"shutdown: closing the cameras did not finish within {timeout_s:g} s")
            return
        failed = {k: v for k, v in results.items() if v}
        if failed:
            logger.warning(f"shutdown: cameras that did not close cleanly: {failed}")
        if nanny is None:
            return
        # the buttons of the cameras the nanny let go of no longer have a camera
        for btn in getattr(getattr(self, 'camera_conn_bar', None), 'buttons', []):
            if btn.camera_name in results and btn.camera_name not in vars(nanny):
                btn.camera = DummyCamera()

    def _close_cameras_bounded(self, timeout_s: float):
        """Close every camera (_start_closing_cameras) and wait, at most ``timeout_s``."""
        self._finish_closing_cameras(self._start_closing_cameras(timeout_s))

    def _shutdown_on_quit(self):
        self.shutdown("application quit")

    def install_shutdown_hooks(self, app):
        """Make shutdown() run however liveOD ends: Qt quitting, the interpreter
        exiting, or the console it runs in being closed (liveOD is started from
        a console, _bat/live_od.bat, not by the dashboard; a console close skips
        both of the others)."""
        # a bound method, not a lambda: the connection then ends with the window
        # instead of keeping it alive as long as the application
        app.aboutToQuit.connect(self._shutdown_on_quit)
        atexit.register(self.shutdown, "interpreter exit")

        def on_console_event(event_name):
            # console closed, logoff or shutdown (console_guard swallows Ctrl+C /
            # Ctrl+Break: a stray Ctrl+C must not end a run)
            self.shutdown(f"console: {event_name}")
            QMetaObject.invokeMethod(app, "quit", Qt.ConnectionType.QueuedConnection)
        if console_guard.install(on_console_event):
            logger.info("Console handler installed: closing this console closes the cameras.")

    def _open_live_scalar_plot(self):
        """Show the live scalar plot window, creating it if needed."""
        self.live_scalar_plot_window.show()
        self.live_scalar_plot_window.raise_()

    def _open_fk_tof(self):
        """Show the FK TOF window."""
        self.fk_tof_window.show()
        self.fk_tof_window.raise_()

    def _open_adjust_panel(self):
        """Show the adjust panel, or bring it to the front and focus it if already open."""
        panel = self._adjust_panel
        if panel.isMinimized():
            panel.setWindowState(panel.windowState() & ~Qt.WindowState.WindowMinimized)
        panel.show()
        panel.raise_()
        panel.activateWindow()

    def _on_adjust_specs(self, specs: list):
        """Called after INIT_RUN — repopulate with the new run's adjust params (may be empty)."""
        self._adjust_panel.populate(specs)
        count = len(specs)
        self._adjust_button.setText(f"Adjust ({count})" if count else "Adjust")
        self._adjust_button.setEnabled(bool(count))

    def _on_adjust_value_changed(self, key: str, value: float):
        """Forward spinbox changes to the server's live adjust dict."""
        self.live_od_server.update_adjust_value(key, value)

    def _on_adjust_spec_updated(self, key: str, min_val: float, max_val: float, step: float):
        """Sync new spec bounds to the server when the spec dialog is accepted."""
        self.live_od_server.update_adjust_spec(key, min_val, max_val, step)

    def _on_scalar_subscription_changed(self, old_tier, new_tier):
        """Update the server's subscription counters when the plot window changes metric or visibility."""
        if old_tier is not None:
            self.live_od_server.unregister_scalar_subscription(old_tier)
        if new_tier is not None:
            self.live_od_server.register_scalar_subscription(new_tier)

    def create_camera_baby(self, file, name):
        """Legacy stub — use spawn_baby via LiveODServer.new_run_signal."""
        self.spawn_baby(filepath=file, camera_key="",
                        capture_images=True, save_data=True,
                        imaging_type=0)

    def spawn_baby(self, filepath: str, camera_key: str,
                   capture_images: bool, save_data: bool,
                   imaging_type: int, n_img: int = 1,
                   n_shots: int = 1, n_pwa: int = 1,
                   camera_params: dict = None,
                   params_payload: dict = None,
                   run_info_payload: dict = None):
        """Spawn a new CameraBaby for an incoming run.

        Called from LiveODServer.new_run_signal (Qt queued connection so this
        always runs on the GUI thread).
        """
        # The token of the run this spawn is for: INIT_RUN noted it (camera runs
        # only), and spawns come in INIT_RUN order. Taken first, so that nothing
        # below can leave it to the next spawn.
        run_token = self.live_od_server.take_spawn_token() if capture_images else ""
        name = names.get_first_name()
        self._run_name = name
        self._run_capture_images = capture_images
        self._run_was_reset = False

        self._run_active = True

        # Propagate run metadata to Analyzer and scalar plot window
        self.analyzer.set_camera_params(camera_params or {})
        # object-plane pixel size (pixel / magnification), for the viewer's µm display
        try:
            self.viewer_window.set_pixel_size_m(
                float(camera_params['pixel_size_m']) / float(camera_params['magnification']))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            self.viewer_window.set_pixel_size_m(None)
        self.analyzer.reset()
        self.viewer_window.set_camera_key(camera_key)
        if capture_images and camera_key:
            self.camera_menu.set_current(camera_key)
        self.viewer_window.on_new_run()
        self.viewer_window.set_xvar_ranges(getattr(self.live_od_server, '_current_xvar_ranges', {}))
        run_id = self.live_od_server._current_run_id
        # the experiment file's name (the class name from an older experiment process)
        expt_class = (self.live_od_server._current_expt_name
                      or str((run_info_payload or {}).get('expt_class', '')))
        # this camera's pins, from the server's store
        self._show_markers_for(camera_key)
        self.status_strip.start_run(run_id, expt_class=expt_class,
                                    camera_key=camera_key if capture_images else "",
                                    save_data=save_data, n_shots=n_shots)
        self.output_window.append_separator(
            " · ".join(p for p in (f"Run {run_id}" if run_id else "Run (not saved)", expt_class) if p))
        self._update_run_buttons()
        xvarnames = list(run_info_payload.get('xvarnames', [])) if run_info_payload else []
        self.live_scalar_plot_window.on_new_run(self.live_od_server._current_run_id, xvarnames)
        self.fk_tof_window.on_new_run(
            self.live_od_server._current_run_id,
            dict(params_payload) if params_payload else {},
        )

        # Cut the previous camera run's threads off from this run before anything
        # new is wired: a late status 2, grab failure or frame from them must not
        # reach it, whether or not the previous run has ended.
        self._retire_previous_run_threads()

        # Interrupt any DataHandler left over from the previous run.
        # On long runs the DataHandler thread can still be draining the shared
        # queue when the next INIT_RUN arrives.  If not interrupted here, the
        # old DataHandler consumes images placed by the new CameraBaby before
        # the new DataHandler has started, so the display never updates.
        if self.data_handler is not None:
            if self.data_handler.isRunning():
                self.msg("Previous DataHandler still running — interrupting it.", logging.WARNING)
                self.data_handler.interrupted = True
                self.data_handler.wait(500)
            # Disconnect before replacing so a late SaveWorker from the
            # previous run can't falsely set _data_handler_done_event for
            # the new run, causing END_RUN to proceed before images are saved.
            try:
                self.data_handler.done_writing_signal.disconnect()   # only the server listens
            except Exception:
                pass
            # Likewise, a late save-failure from the previous run must not
            # abort the run that is starting now.
            try:
                self.data_handler.save_failed_signal.disconnect(self.on_save_failed)
            except Exception:
                pass
            # (its frames were cut off from the new run in _retire_previous_run_threads)
            self.data_handler = None

        # Interrupt any CameraBaby left over from the previous run.
        # This happens when images never arrived (e.g. camera not triggered by
        # the remote machine's hardware) and the grab-loop timed out slowly, or
        # when on_run_done() was called before the baby finished its grab.
        if self.the_baby is not None and self.the_baby.isRunning():
            self.msg("Previous CameraBaby still running — interrupting it.", logging.WARNING)
            try:
                self.the_baby.interrupted = True
            except Exception:
                pass
            self.the_baby = None
            self.data_handler = None

        if not capture_images:
            self.msg(f"{name}: I am born! (no camera, save_data={save_data})")
            self.the_baby = None
            self.data_handler = None
            return

        # Replace the shared queue with a fresh one for every camera run.
        # Without this, images left in the queue by an interrupted previous
        # DataHandler appear at the start of the new run — displayed as old-run
        # frames and (worse) written at indices 0, 1, … in the new HDF5 file.
        # Old objects (if still alive) keep their reference to the old queue.
        self.queue = Queue()

        self.data_handler = DataHandler(
            self.queue, data_filepath=filepath,
            save_data=save_data,
            imaging_type=imaging_type,
            camera_key=camera_key,
            camera_params=camera_params,
            params_payload=params_payload,
            run_info_payload=run_info_payload,
            n_img=n_img,
            n_shots=n_shots,
            n_pwa_per_shot=n_pwa,
        )
        self.the_baby = CameraBaby(self.data_handler, name, self.queue,
                                   self._nanny_for_token(run_token))
        self._run_threads = (self.the_baby, self.data_handler)
        baby = self.the_baby
        # Both threads carry the token of the run they work for, and the server
        # drops their reports once a later INIT_RUN has replaced that run -- also
        # before this window has caught up with it (new_run_signal is queued).
        self.the_baby.run_token = run_token
        self.data_handler.run_token = run_token

        # Standard data/image wiring
        self.data_handler.save_data_bool_signal.connect(
            self.data_handler.get_save_data_bool)
        self.data_handler.image_type_signal.connect(
            self.analyzer.get_analysis_type)
        self.data_handler.got_image_from_queue.connect(self.analyzer.got_img)
        self.data_handler.got_image_from_queue.connect(self.count_images)
        # The server counts frames against N_img at END_RUN, and a grab that
        # ends early tells it why; both from the camera threads directly.
        self.data_handler.got_image_from_queue.connect(
            lambda _img, t=run_token: self.live_od_server.on_image_received(run_token=t),
            Qt.ConnectionType.DirectConnection)
        # (these direct connections also check that the baby is still this run's;
        # the server checks the run token)
        self.the_baby.grab_failed_signal.connect(
            lambda reason, b=baby: self._from_run_baby(b, self.live_od_server.on_grab_failed, reason),
            Qt.ConnectionType.DirectConnection)
        self.the_baby.camera_overrides_signal.connect(
            lambda key, clamps, b=baby: self._from_run_baby(
                b, self.live_od_server.on_camera_overrides, key, clamps),
            Qt.ConnectionType.DirectConnection)

        self.the_baby.camera_connect.connect(self.check_new_camera)
        self.the_baby.camera_grab_start.connect(self.grab_start_msg)
        self.the_baby.camera_grab_start.connect(self.get_img_number)
        self.the_baby.camera_grab_start.connect(self.data_handler.get_img_number)
        self.the_baby.camera_grab_start.connect(self.viewer_window.get_img_number)
        self.the_baby.camera_grab_start.connect(self.analyzer.get_img_number)
        self.the_baby.camera_grab_start.connect(self.data_handler.start)
        self.the_baby.camera_grab_start.connect(self.reset_count)

        self.the_baby.honorable_death_signal.connect(
            lambda: self.msg(f'Run complete. {name} has died honorably.'))
        self.the_baby.dishonorable_death_signal.connect(
            lambda: self.msg(f'{name} died dishonorably.', logging.WARNING))

        self.the_baby.cam_status_signal.connect(
            lambda s, b=baby: self._from_run_baby(b, self.live_od_server.on_cam_ready) if s == 2 else None,
            Qt.ConnectionType.DirectConnection,
        )
        self.data_handler.done_writing_signal.connect(
            lambda t=run_token: self.live_od_server.on_data_handler_done(run_token=t),
            Qt.ConnectionType.DirectConnection,
        )
        # If the HDF5 file turns out to be unusable, abort the run right away
        # rather than acquiring a full scan whose images are all discarded.
        self.data_handler.save_failed_signal.connect(self.on_save_failed)
        # For Basler cameras: notify the server when this baby's grab loop has
        # fully exited so that WAIT_CAM_READY for the next run is not released
        # prematurely (while the old RetrieveResult() call is still blocking).
        if self._needs_grab_drain(camera_key):
            self.the_baby.done_signal.connect(
                self.live_od_server.on_basler_baby_done,
                Qt.ConnectionType.DirectConnection,
            )

        # Clear the nanny's interrupt flag before starting the new baby.
        # Without this, if spawn_baby fires during reset()'s processEvents() loop
        # (i.e. the experiment sends INIT_RUN before the old grab loop has died),
        # persistent_get_camera() would hit break_check() → True and return a
        # DummyCamera, causing "Camera not ready" on every subsequent run.
        self.camera_nanny.interrupted = False
        self.the_baby.start()
        self.msg(f"Baby {name} born — camera_key={camera_key}")

    def _from_run_baby(self, baby, fn, *args):
        """Pass a camera thread's report on to the server only if that thread
        belongs to the latest camera run this window has spawned, with the token
        of the run it was spawned for (the server drops it if a later INIT_RUN
        has replaced that run, which this window may not have caught up with
        yet). Runs on the camera thread (a DirectConnection); the retired
        threads' connections are cut in _retire_previous_run_threads, this
        catches one already in flight."""
        if baby is not self._run_threads[0]:
            logger.warning(f"Ignored {getattr(fn, '__name__', 'a report')} from camera thread "
                           f"{getattr(baby, 'name', '?')}: its run has been replaced.")
            return
        fn(*args, run_token=getattr(baby, 'run_token', None))

    def _retire_previous_run_threads(self):
        """A new run starts: the previous camera run's CameraBaby and DataHandler
        must no longer act on the server's run state (B5). An old status 2 set
        the new run's camera-ready event, an old grab failure marked the new run
        incomplete, an old camera_connect / grab-start reset the new run's
        counters. Their connections to the window and the server are cut here;
        what they need to finish on their own stays (the Basler grab-drain
        done_signal, their log messages).

        A previous baby that is still running although its run has ended (the
        window no longer holds it as the_baby: END_RUN or a reset came first) is
        asked to stop, quietly: its run may be saved, so it must not take the
        interrupt path, which discards the file. One still held as the_baby is
        interrupted by spawn_baby as before."""
        baby, handler = self._run_threads
        self._run_threads = (None, None)
        if baby is not None:
            for signal in (baby.grab_failed_signal, baby.cam_status_signal,
                           baby.camera_overrides_signal, baby.camera_connect,
                           baby.camera_grab_start):
                try:
                    signal.disconnect()
                except (TypeError, RuntimeError):
                    pass            # nothing connected
            if baby.isRunning() and baby is not self.the_baby:
                self.msg(f"Camera thread {baby.name} of the previous run is still running; "
                         f"stopping it.", logging.WARNING)
                baby.request_stop()
        if handler is not None:
            # its frames are not this run's: not counted, not shown
            try:
                handler.got_image_from_queue.disconnect()
            except (TypeError, RuntimeError):
                pass
            # nor its writer's end: a late "done" would open the new run's save
            # gate early (the server checks its run token as well), a late save
            # failure would abort the new run
            try:
                handler.done_writing_signal.disconnect()    # only the server listens
            except (TypeError, RuntimeError):
                pass            # already disconnected by spawn_baby
            try:
                handler.save_failed_signal.disconnect(self.on_save_failed)
            except (TypeError, RuntimeError):
                pass            # already disconnected by spawn_baby

    def on_save_failed(self, reason: str):
        """SaveWorker could not open/use the HDF5 file — abort the run.

        Without this the run continues to completion with every image silently
        dropped, and the failure only surfaces as a KeyError at END_RUN, by
        which point the data is unrecoverable.  reset() sets the server's
        _reset_requested flag, so the experiment aborts at the next shot
        boundary and the unusable file is deleted.
        """
        self.msg(f"Data file unusable ({reason}) — aborting run.", logging.ERROR)
        self.reset()

    def on_run_exited(self, how: str):
        """The server closed a run whose experiment exited with frames still due
        (``how``: why now). Its camera thread is asked to stop quietly: the frames
        that came stay in the file, which is neither saved nor deleted. The
        interrupt path (reset) would delete it. run_done_signal follows."""
        baby = self.the_baby
        if baby is not None and baby.isRunning():
            self.msg(f"The experiment exited; stopping camera thread {baby.name} ({how}). "
                     f"The run's file is kept as it is.", logging.WARNING)
            _stop_camera_thread(baby, f"its experiment exited ({how})")

    def on_run_done(self):
        """Called when the LiveODServer processes an END_RUN message."""
        self._run_active = False
        name = getattr(self, '_run_name', '?')
        if self.the_baby is None and not getattr(self, '_run_capture_images', True):
            # No-camera run — emit the honorable death message here.
            # (camera runs that were reset also have the_baby=None by this
            # point, so we guard with _run_capture_images to avoid printing
            # a spurious honorable-death after an aborted camera run.)
            if not self._run_was_reset:
                self.msg(f"{name} has died honorably.")
        # Camera runs emit their own honorable_death_signal message.
        self.the_baby = None
        self.data_handler = None
        # run_id is claimed + incremented atomically at INIT_RUN
        # (server_talk.claim_run_id), so there is nothing to increment here.
        self._run_was_reset = False
        self._update_run_buttons()

    def on_shot_timing(self, delta_t: float, eta_str: str):
        """Update Δt and the ETA in the status strip."""
        self.status_strip.set_timing(delta_t, eta_str)

    def on_shot_progress(self, shot_idx: int, N_total: int, xvar_values: object):
        """Update the GUI with per-shot progress from the ZMQ server."""
        self.status_strip.set_progress(shot_idx + 1, N_total)
        self.viewer_window.set_shot_xvars(shot_idx, xvar_values)
        self.update_image_count(shot_idx + 1, N_total)

    def restart_mother(self):
        """Legacy slot kept for compatibility — no-op in ZMQ mode."""
        pass

    def check_new_camera(self, camera_select):
        # Update button color immediately when camera connection changes
        if hasattr(self, 'camera_conn_bar'):
            for btn in self.camera_conn_bar.buttons:
                if hasattr(btn, 'camera_name') and btn.camera_name == camera_select:
                    btn._set_color_success()
                elif hasattr(btn, 'camera') and btn.camera is not None and not btn.camera.is_opened():
                    btn._set_color_closed()
        if self.last_camera != camera_select:
            self.clear_plots()
            self.last_camera = camera_select
            self.set_default_roi(camera_select)

    def set_default_roi(self, camera_select):
        # the lab's saved-ROI id for this camera, or None for no default
        key = self.config.default_roi_id_for(camera_select)
        if key:
            self.analyzer.roi = ROI(roi_id=key, use_saved_roi=False, printouts=False)

    def get_img_number(self, N_img, N_shots, N_pwa_per_shot):
        self.N_pwa_per_shot = N_pwa_per_shot

    def count_images(self):
        self.img_count += 1
        self.img_count_run += 1
        if self.img_count == self.N_pwa_per_shot:
            self.img_count = 0

    def reset_count(self):
        self.img_count = 0
        self.img_count_run = 0
        self.analyzer.imgs = []

    def msg(self, msg, level=logging.INFO):
        # to the terminal, the log file, the panel and the remote viewers: see _on_log_record
        logger.log(level, msg)

    def grab_start_msg(self, Nimg, *_):
        self.N_img = Nimg
        msg = f"Camera grabbing... Expecting {Nimg} images."
        self.msg(msg)

    def gotem_msg(self, count):
        msg = f"gotem (img {count}/{self.N_img})"
        self.msg(msg)

    def clear_plots(self):
        self.viewer_window.clear_plots()

    def update_image_count(self, count, total):
        self.viewer_window.update_image_count(count, total)

    def reset(self):
        # A run whose experiment is gone: its experiment exited with frames still
        # due (the server closes the run and keeps its file -- interrupting the
        # camera thread here would delete it), or an Abort nobody has answered
        # (pressed again: the server closes the aborted run now). The server
        # thread does the closing; the next pass of its loop, within 0.5 s.
        srv = getattr(self, 'live_od_server', None)
        if srv is not None:
            # A run being saved is complete (END_RUN came): the server ignores the
            # Reset (one WARNING), and nothing here is interrupted -- the camera
            # thread and data handler may still be finishing the file's write.
            if srv.reset_ignored_during_save():
                # visible on every press, not only in the log: the status
                # strip's notice (the server has logged the same text once
                # already -- no second log line here, unless there is no strip)
                text = srv.reset_ignored_text()
                strip = getattr(self, 'status_strip', None)
                if strip is not None:
                    strip.show_notice(text)
                else:
                    self.msg(text + ".")
                return
            if srv.exited_run_pending():
                srv.close_exited_run_now("Reset pressed")
                self.msg("Reset: the run's experiment has exited; closing the run "
                         "(its file is kept).", logging.WARNING)
                return
            if srv.abort_again():
                self.msg("Reset pressed again: the Abort has no answer from the experiment; "
                         "closing the aborted run now.", logging.WARNING)
                return
        # Guard against duplicate calls (e.g. local button + remote reset_signal
        # arriving close together).  If _reset_requested is already set and
        # there is no active run or camera baby, the reset was already handled.
        if (hasattr(self, 'live_od_server') and
                self.live_od_server._reset_requested and
                not getattr(self, '_run_active', False) and
                getattr(self, 'the_baby', None) is None):
            return
        # Ensure the ZMQ server flag is set regardless of whether this was
        # triggered by the local button or by the remote viewer (which goes
        # through _handle_reset first, but this is idempotent). request_reset
        # counts the press once (POLL's reset_count); from this window's own
        # button it is a person's.
        if hasattr(self, 'live_od_server'):
            srv = self.live_od_server
            if hasattr(srv, 'request_reset'):
                srv.request_reset("person")
            else:
                srv._reset_requested = True
                srv.note_reset_requested()
        if hasattr(self, 'camera_nanny'):
            try:
                self.camera_nanny.interrupted = True
            except Exception as e:
                logger.warning(f"reset: {e}")

        if hasattr(self, 'data_handler') and self.data_handler is not None:
            try:
                self.data_handler.interrupted = True
            except Exception as e:
                logger.warning(f"reset: {e}")

        if hasattr(self, 'the_baby') and self.the_baby is not None:
            try:
                self.the_baby.interrupted = True
                # self.the_baby.dishonorable_death()
                self.msg('Acquisition aborted, run ID advanced.', logging.WARNING)
            except Exception as e:
                logger.warning(f"reset: {e}")
        else:
            if getattr(self, '_run_active', False):
                # A no-camera run (setup_camera=False) is in progress.
                # The ZMQ poll mechanism will abort it at the next shot boundary.
                # The run_id was already claimed + incremented at INIT_RUN, so
                # do not increment again here.
                name = getattr(self, '_run_name', '?')
                self.msg(f'Run reset. {name} has died dishonorably.', logging.WARNING)
                self._run_active = False
                self._run_was_reset = True
            else:
                # No run was ever started for this id -> manually skip it.
                self.msg('No active run. Incrementing Run ID.')
                self.server_talk.update_run_id()
                self.update_run_id_label()

        if self.the_baby is not None:
            baby_to_wait_for = self.the_baby
            while not getattr(baby_to_wait_for, 'dead', False):
                if self.the_baby is not baby_to_wait_for:
                    # spawn_baby() fired during processEvents() — the experiment
                    # sent INIT_RUN before the old grab loop finished dying.
                    # The old baby will finish in the background; the new one
                    # already started.  Stop waiting here so camera_nanny.
                    # interrupted gets cleared below before the new baby's
                    # persistent_get_camera() runs.
                    break
                QApplication.processEvents()
                time.sleep(0.05)

        self.queue = Queue()
        # self.restart_mother()
        self.the_baby = None
        self.data_handler = None
        self.camera_nanny.interrupted = False
        self._update_run_buttons()

def main(config):
    """Run the liveOD acquisition window with the lab's LiveODConfig. Lab launchers
    call this (kexp: kexp/util/live_od/gui/main_window.py); waxx has no config of
    its own, so this module is not runnable by itself."""
    set_config(config)
    try:
        # A process that finds a camera busy is told liveOD holds it (the
        # drivers take an OS lock per device, labelled with this).
        from waxx.control.cameras.device_lock import set_process_label
        set_process_label("liveOD")
    except Exception as exc:
        logger.warning(f"could not label the camera locks as liveOD's ({exc}); a busy "
                       f"camera will name this process by its command line instead")
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(config.app_user_model_id)
    except Exception:
        pass        # not Windows: no taskbar identity to set
    from waxa.taskbar import group_console
    group_console()  # console joins the shared terminals taskbar button
    app = QApplication(sys.argv)
    # layout, toggles and per-camera OD levels are remembered between sessions
    win = LiveODWindow(settings=QSettings("waxx", "live_od"))
    # cameras are closed however liveOD ends, the console window's X included
    win.install_shutdown_hooks(app)
    win.setWindowTitle(config.window_title)
    win.setWindowIcon(win.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView))
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    sys.exit("waxx's liveOD window has no lab configuration of its own. Start it through "
             "the lab's launcher -- kexp: `python -m kexp.util.live_od.gui.main_window` "
             "or _bat/live_od.bat.")

