import sys
import logging
from queue import Queue
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStyle
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSettings
import time
import names

from waxa import ROI

from waxx.util.live_od.config import get_config, set_config
from waxx.util.live_od.camera_mother import CameraMother, CameraBaby, DataHandler, CameraNanny
from waxx.util.live_od.camera_connection_widget import CamConnBar
from waxx.util.live_od.gui.camera_menu import CameraMenuButton
from waxx.util.live_od.gui.viewer import LiveODViewer
from waxx.util.live_od.gui.analyzer import Analyzer
from waxx.util.live_od.gui.plotter import LiveODPlotter
from waxx.util.live_od.live_od_server import LiveODServer
from waxx.util.live_od.live_od_broadcaster import LiveODBroadcaster
from waxx.util.live_od.gui.live_scalar_plot_window import LiveScalarPlotWindow
from waxx.util.live_od.gui.fk_tof_window import FkTofWindow
from waxx.util.live_od.gui.adjust_panel import AdjustPanel
from waxx.util.live_od.gui.status_strip import StatusStrip
from waxx.util.live_od.log import get_logger, setup_logging, install_excepthooks, DEFAULT_LOG_DIR

logger = get_logger("window")

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
        self.last_camera = ""
        self.img_count = 0
        self.img_count_run = 0
        self._run_active = False   # True between INIT_RUN and END_RUN/reset
        self._run_was_reset = False  # True when reset() kills an active no-camera run
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
        self.live_od_server.reset_signal.connect(self.reset)
        self.live_od_server.camera_control_signal.connect(self.on_remote_camera_control)
        self.live_od_server.run_state_signal.connect(self.on_run_state)
        self.live_od_server.set_camera_state_provider(self._camera_state_report)
        self.live_od_server.start()

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
            self.broadcaster.broadcast_camera_state(
                self.camera_conn_bar.get_states())
        except Exception as e:
            logger.debug(f"camera-state broadcast error: {e}")

    def _camera_state_report(self) -> dict:
        """For the server's POLL reply: each camera's button state, plus what a
        remote tool needs to find the same device on its beacon server (type,
        serial).  Called from the server thread: plain attribute reads only."""
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
        """The status row's camera button, or its drop-down: connect / disconnect."""
        btn = self.camera_conn_bar.get_button(camera_key)
        if btn is None:
            return
        try:
            btn.button_pressed()
        except Exception as e:
            self.msg(f"Camera {camera_key}: {e}", logging.ERROR)

    def update_run_id_label(self):
        try:
            rid = self.server_talk.get_run_id()
        except Exception as e:
            rid = None
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
        self.camera_conn_bar = CamConnBar(self.camera_nanny, self.output_window)
        self.camera_conn_bar.setParent(self)
        self.camera_conn_bar.hide()
        self.camera_menu = CameraMenuButton([b.camera_name for b in self.camera_conn_bar.buttons])
        self.camera_menu.set_states(self.camera_conn_bar.get_states())   # those opened on start
        for btn in self.camera_conn_bar.buttons:
            btn.state_changed.connect(self.camera_menu.set_state)
        self.camera_menu.toggle_requested.connect(self._on_camera_toggle_requested)
        self.status_strip.add_camera_widget(self.camera_menu)

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
            "Stop the run now and delete its data file. No confirmation: it has to be fast.")
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
        if self._settings is not None:
            self._settings.setValue("window/geometry", self.saveGeometry())
            self.viewer_window.save_settings()
        super().closeEvent(event)

    def _open_live_scalar_plot(self):
        """Show the live scalar plot window, creating it if needed."""
        self.live_scalar_plot_window.show()
        self.live_scalar_plot_window.raise_()

    def _open_fk_tof(self):
        """Show the FK TOF window."""
        self.fk_tof_window.show()
        self.fk_tof_window.raise_()

    def _open_adjust_panel(self):
        """Show the adjust panel."""
        self._adjust_panel.show()
        self._adjust_panel.raise_()

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
                self.data_handler.done_writing_signal.disconnect(
                    self.live_od_server.on_data_handler_done)
            except Exception:
                pass
            # Likewise, a late save-failure from the previous run must not
            # abort the run that is starting now.
            try:
                self.data_handler.save_failed_signal.disconnect(self.on_save_failed)
            except Exception:
                pass
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
                                   self.camera_nanny)

        # Standard data/image wiring
        self.data_handler.save_data_bool_signal.connect(
            self.data_handler.get_save_data_bool)
        self.data_handler.image_type_signal.connect(
            self.analyzer.get_analysis_type)
        self.data_handler.got_image_from_queue.connect(self.analyzer.got_img)
        self.data_handler.got_image_from_queue.connect(self.count_images)

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
            lambda s: self.live_od_server.on_cam_ready() if s == 2 else None,
            Qt.ConnectionType.DirectConnection,
        )
        self.data_handler.done_writing_signal.connect(
            self.live_od_server.on_data_handler_done,
            Qt.ConnectionType.DirectConnection,
        )
        # If the HDF5 file turns out to be unusable, abort the run right away
        # rather than acquiring a full scan whose images are all discarded.
        self.data_handler.save_failed_signal.connect(self.on_save_failed)
        # For Basler cameras: notify the server when this baby's grab loop has
        # fully exited so that WAIT_CAM_READY for the next run is not released
        # prematurely (while the old RetrieveResult() call is still blocking).
        if self.config.camera_needs_grab_drain(camera_key):
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
        # through _handle_reset first, but this is idempotent).
        if hasattr(self, 'live_od_server'):
            self.live_od_server._reset_requested = True
            self.live_od_server.note_reset_requested()
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
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(config.app_user_model_id)
    except Exception:
        pass        # not Windows: no taskbar identity to set
    app = QApplication(sys.argv)
    # layout, toggles and per-camera OD levels are remembered between sessions
    win = LiveODWindow(settings=QSettings("waxx", "live_od"))
    win.setWindowTitle(config.window_title)
    win.setWindowIcon(win.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView))
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    sys.exit("waxx's liveOD window has no lab configuration of its own. Start it through "
             "the lab's launcher -- kexp: `python -m kexp.util.live_od.gui.main_window` "
             "or _bat/live_od.bat.")

