"""
Standalone LiveOD remote viewer window.

Subscribes to the LiveOD Server's ZMQ PUB broadcast and shows what the
acquisition window shows -- the status strip, the camera button, the person
hold, Abort, the OD images, the log -- without the data drive or the camera
hardware. A viewer that joins mid-run catches up by one POLL and GET_LOG.

Usage (run from any machine on the lab network)::

    python -m waxx.util.live_od.gui.remote_viewer_window
    python -m waxx.util.live_od.gui.remote_viewer_window 192.168.1.76
    python -m waxx.util.live_od.gui.remote_viewer_window 192.168.1.76 5561
"""

import logging
import pickle
import sys
import threading
import time
from queue import Queue

import zmq
from PyQt6.QtCore import QSettings, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from waxx.util.live_od.gui.plotter import LiveODPlotter, ShotPlotData
from waxx.util.live_od.gui.viewer import LiveODViewer
from waxx.util.live_od.gui.live_scalar_plot_window import LiveScalarPlotWindow
from waxx.util.live_od.gui.fk_tof_window import FkTofWindow
from waxx.util.live_od.gui.adjust_panel import AdjustPanel
from waxx.util.live_od.gui.camera_control import CameraControl
from waxx.util.live_od.gui.tool_windows import ToolLauncher
from waxx.util.live_od.gui.status_strip import ACTIVE_STATES, StatusStrip
from waxx.util.live_od.gui.hold_badge import HoldBadge
from waxx.util.live_od.gui.run_again_button import RunAgainButton


# ---------------------------------------------------------------------------
# ZMQ subscriber thread
# ---------------------------------------------------------------------------

class LiveODSubscriber(QThread):
    """Receives broadcast messages from ``LiveODBroadcaster`` and re-emits
    them as Qt signals so the GUI thread can process them safely.
    
    Periodically polls the server to verify connection status. If the
    server becomes unreachable, automatically attempts to rediscover it
    via UDP broadcast and reconnect.
    """

    od_image_signal = pyqtSignal(object)            # plot_data tuple
    od_frame_age_signal = pyqtSignal(float)         # end-to-end frame age in seconds
    shot_progress_signal = pyqtSignal(int, int, object)  # shot_idx, N_total, xvar_values
    run_started_signal = pyqtSignal(int, object)    # run_id, xvarnames
    run_done_signal = pyqtSignal()
    log_msg_signal = pyqtSignal(str, int)           # text, logging level number
    run_state_signal = pyqtSignal(str, str)         # state, detail (LiveODServer.run_state_signal)
    markers_signal = pyqtSignal(str, list)          # camera_key, markers (pins on the image)
    connection_status_signal = pyqtSignal(str)
    camera_state_signal = pyqtSignal(object)        # dict[camera_key -> state]
    camera_persist_signal = pyqtSignal(object)      # dict[camera_key -> Persist on] (camera host)
    shot_scalars_signal = pyqtSignal(object)        # per-shot scalar dict
    fk_tof_signal = pyqtSignal(object)              # per-shot FK TOF width dict
    adjust_values_signal = pyqtSignal(object)       # dict[key -> current_val]
    aux_notice_signal = pyqtSignal(object)          # AUX_NOTICE dict: run_id, t, items (no arrays)
    run_info_signal = pyqtSignal(object)            # RUN_STARTED's whole dict (its 'info' if sent)
    shot_timing_signal = pyqtSignal(float, str)     # delta_t s, ETA "HH:MM"
    next_run_id_signal = pyqtSignal(object)         # the next saved run's id, or None
    adjust_specs_signal = pyqtSignal(object)        # list of adjust specs ([] = none)
    connected_signal = pyqtSignal()                 # first message on a (re)connection

    # Reconnection configuration (seconds)
    RECONNECT_RETRY_INTERVAL = 2.0  # How often to retry server discovery
    DISCOVERY_TIMEOUT = 3.0      # Timeout for UDP broadcast discovery

    def __init__(self, ip: str, port: int):
        super().__init__()
        self._ip = ip
        self._port = port
        self._running = False
        self._connection_ok = True
        self._attempting_reconnect = False

    def run(self):
        """Main subscriber loop: connect and receive messages, with periodic polling."""
        context = zmq.Context()
        self._running = True

        while self._running:
            try:
                socket = context.socket(zmq.SUB)
                socket.setsockopt(zmq.RCVTIMEO, 100)        # 100 ms poll for graceful shutdown
                socket.setsockopt(zmq.SUBSCRIBE, b"")       # subscribe to all topics
                socket.setsockopt(zmq.RCVHWM, 8)            # drop oldest frames if backed up
                # TCP keepalive: OS detects dead connections without false positives
                socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
                socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 10)
                socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
                socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)
                socket.connect(f"tcp://{self._ip}:{self._port}")
                self._connection_ok = True
                self._attempting_reconnect = False
                self.connection_status_signal.emit(
                    f"Connecting to tcp://{self._ip}:{self._port}…"
                )
                print(f"[LiveODSubscriber] Connected to tcp://{self._ip}:{self._port}")
                first_message_received = False

                # Receive loop — zmq.Again just means the server is idle, not down
                while self._running:
                    try:
                        raw = socket.recv()
                    except zmq.Again:
                        continue  # no message yet; server may simply be idle

                    try:
                        msg = pickle.loads(raw)
                    except Exception as exc:
                        print(f"[LiveODSubscriber] deserialisation error: {exc}")
                        continue

                    tag = msg.get("tag", "")
                    if not first_message_received:
                        first_message_received = True
                        self.connection_status_signal.emit(
                            f"Connected to tcp://{self._ip}:{self._port}"
                        )
                        self.connected_signal.emit()
                    try:
                        if tag == "OD_IMAGE":
                            plot_data = ShotPlotData((
                                msg["img_atoms"], msg["img_light"], msg["img_dark"],
                                msg["od"], msg["sum_od_x"], msg["sum_od_y"],
                            ), msg.get("shot_idx"))     # None from a server that predates it
                            self.od_image_signal.emit(plot_data)
                            t_capture = msg.get("t_capture")
                            if t_capture is not None:
                                self.od_frame_age_signal.emit(time.time() - float(t_capture))
                        elif tag == "SHOT_PROGRESS":
                            self.shot_progress_signal.emit(
                                int(msg["shot_idx"]),
                                int(msg["N_total"]),
                                msg.get("xvar_values", {}),
                            )
                        elif tag == "RUN_STARTED":
                            self.run_started_signal.emit(
                                int(msg.get("run_id", 0)),
                                list(msg.get("xvarnames", [])),
                            )
                            self.run_info_signal.emit(dict(msg))
                        elif tag == "SHOT_TIMING":
                            self.shot_timing_signal.emit(float(msg.get("delta_t", 0.0)),
                                                         str(msg.get("eta", "--:--")))
                        elif tag == "NEXT_RUN_ID":
                            self.next_run_id_signal.emit(msg.get("run_id"))
                        elif tag == "ADJUST_SPECS":
                            self.adjust_specs_signal.emit(list(msg.get("specs") or []))
                        elif tag == "RUN_DONE":
                            self.run_done_signal.emit()
                        elif tag == "LOG_MSG":
                            # no "level": a server that predates it
                            self.log_msg_signal.emit(str(msg.get("text", "")),
                                                     int(msg.get("level", 20)))
                        elif tag == "MARKERS":
                            self.markers_signal.emit(str(msg.get("camera_key", "")),
                                                     list(msg.get("markers", [])))
                        elif tag == "RUN_STATE":
                            self.run_state_signal.emit(str(msg.get("state", "")),
                                                       str(msg.get("detail", "")))
                        elif tag == "CAMERA_STATE":
                            states = msg.get("states", {}) or {}
                            self.camera_state_signal.emit(dict(states))
                            # additive: only a camera-host liveOD sends it
                            if isinstance(msg.get("persist"), dict):
                                self.camera_persist_signal.emit(dict(msg["persist"]))
                        elif tag == "SHOT_SCALARS":
                            self.shot_scalars_signal.emit(dict(msg))
                        elif tag == "FK_TOF":
                            self.fk_tof_signal.emit(dict(msg))
                        elif tag == "ADJUST_VALUES":
                            self.adjust_values_signal.emit(dict(msg.get('values', {})))
                        elif tag == "AUX_NOTICE":
                            # what a PUT_DATA brought; the arrays are GET_AUX_DATA's
                            self.aux_notice_signal.emit(dict(msg))
                        elif tag == "HELLO":
                            pass  # heartbeat — already triggered connected status above
                    except Exception as exc:
                        print(f"[LiveODSubscriber] signal error ({tag}): {exc}")
            
            except Exception as exc:
                print(f"[LiveODSubscriber] Connection error: {exc}")
                self._on_connection_lost()
            finally:
                try:
                    socket.close()
                except:
                    pass
            
            # If we exited the receive loop but should still run, attempt reconnection
            if self._running:
                self._attempt_reconnection(context)
        
        context.term()

    def _on_connection_lost(self):
        """Handle connection loss by marking as disconnected."""
        if self._connection_ok:
            self._connection_ok = False
            self.connection_status_signal.emit(
                "Disconnected from server — attempting to rediscover…"
            )
            print(f"[LiveODSubscriber] Server not responding — marked as disconnected")

    def _attempt_reconnection(self, context: zmq.Context):
        """Attempt to rediscover the server via UDP broadcast and reconnect."""
        if not self._running:
            return
        
        if self._attempting_reconnect:
            print(f"[LiveODSubscriber] Reconnection already in progress, waiting…")
            time.sleep(self.RECONNECT_RETRY_INTERVAL)
            return
        
        self._attempting_reconnect = True
        print(f"[LiveODSubscriber] Attempting to rediscover server via UDP broadcast…")
        
        try:
            from waxx.util.comms_server.hardware_id import discover_scoped
            
            result = discover_scoped("live_od_broadcast", timeout=self.DISCOVERY_TIMEOUT)
            if result is not None:
                new_ip, new_port = result
                self._ip = new_ip
                self._port = new_port
                self._connection_ok = False
                self.connection_status_signal.emit(
                    f"Rediscovered server at tcp://{self._ip}:{self._port} — reconnecting…"
                )
                print(f"[LiveODSubscriber] Rediscovered server at tcp://{self._ip}:{self._port}")
            else:
                self.connection_status_signal.emit(
                    "Server not found via broadcast — retrying in a moment…"
                )
                print(f"[LiveODSubscriber] Server not found via broadcast — will retry")
                time.sleep(self.RECONNECT_RETRY_INTERVAL)
        except Exception as exc:
            print(f"[LiveODSubscriber] Rediscovery error: {exc}")
            time.sleep(self.RECONNECT_RETRY_INTERVAL)
        finally:
            self._attempting_reconnect = False

    def stop(self):
        self._running = False




# ---------------------------------------------------------------------------
# Remote viewer window
# ---------------------------------------------------------------------------

class RemoteViewerWindow(QWidget):
    """Standalone window that subscribes to a LiveODBroadcaster and shows what
    the acquisition window shows: the status strip (state pill, run label,
    progress, Δt / ETA, the next run id), the camera button, the person hold,
    Abort, the OD images and the log.

    A viewer that joins in the middle of a run asks the server once (POLL, and
    GET_LOG for the run so far) and fills the strip and the log from that.
    What only the acquisition window can do (the camera's settings and live
    view, the ROI the server's atom numbers use) stays there; the viewer's ROI
    crops only its own profiles."""

    _discovery_status_signal = pyqtSignal(str)
    _discovery_done_signal   = pyqtSignal(str, int)   # ip, port
    _camera_request_done_signal = pyqtSignal(str)    # camera_key (worker -> GUI re-enable)
    _adjust_state_fetched_signal = pyqtSignal(list, object)  # specs, values dict
    _reply_notice_signal = pyqtSignal(str)            # a request ignored/refused (worker -> GUI)
    _on_gui_signal = pyqtSignal(object)               # a callable to run on the GUI thread

    #: the window title, with the run's state after it
    TITLE = "LiveOD Viewer"

    def __init__(self, ip: str = None, port: int = None, settings=None):
        super().__init__()

        self._requested_ip   = ip
        self._requested_port = port
        self._ip   = ip   or "—"
        self._port = port or 0
        self._settings = settings

        # Viewer + plotter (independent of network)
        self.plotting_queue: Queue = Queue()
        self.viewer_window = LiveODViewer()
        # the status strip's progress bar carries the shot count, as in the acquisition window
        self.viewer_window.image_count_label.hide()
        self.plotter = LiveODPlotter(self.viewer_window, self.plotting_queue)
        self.plotter.start()

        self.subscriber: LiveODSubscriber | None = None
        self._current_run_id: int = 0
        self._run_in_progress = False
        self._marker_camera_key: str = ""   # the camera the server's markers belong to
        self.viewer_window.markers_changed.connect(self._on_viewer_markers_changed)

        # Scalar plot window
        self.live_scalar_plot_window = LiveScalarPlotWindow()
        self.live_scalar_plot_window.subscription_changed_signal.connect(
            self._on_scalar_subscription_changed
        )
        self.viewer_window.live_plot_requested.connect(self._open_live_scalar_plot)

        # FK TOF window
        self.fk_tof_window = FkTofWindow()
        self.viewer_window.fk_tof_requested.connect(self._open_fk_tof)

        # Adjust panel — connected once here, not per fetch: the specs are
        # re-fetched on every new run, and re-connecting each time would send one
        # extra SET request per fetch for every edit.
        self._adjust_panel = AdjustPanel()
        self._adjust_panel.setWindowTitle("Adjust Parameters (Remote)")
        self._adjust_panel.value_changed_signal.connect(self._on_remote_adjust_value_changed)
        self._adjust_panel.spec_updated_signal.connect(self._on_remote_spec_updated)
        self._adjust_button = QPushButton("Adjust")
        self._adjust_button.setEnabled(False)
        self._adjust_button.clicked.connect(self._open_adjust_panel)

        # Cached endpoint for the LiveOD REP server (RESET / CAMERA_CONTROL).
        # Populated lazily on the first request that succeeds; falls back to
        # UDP discover() if a request fails (server moved or restarted).
        self._req_ip: str | None = None
        self._req_port: int | None = None

        self._setup_layout()

        # Connect discovery signals (emitted from worker thread)
        self._discovery_status_signal.connect(self._on_connection_status)
        self._reply_notice_signal.connect(self._show_reply_notice)
        self._discovery_done_signal.connect(self._on_discovered)
        self._camera_request_done_signal.connect(self._on_camera_request_done)
        self._adjust_state_fetched_signal.connect(self._on_adjust_state_fetched)
        self._on_gui_signal.connect(lambda fn: fn())

        # Kick off discovery after the event loop starts
        QTimer.singleShot(0, self._start_discovery)

    def msg(self, text: str, level: int = logging.INFO):
        """A line in this window's log (not sent anywhere)."""
        self.viewer_window.output_window.append_message(str(text), int(level))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _start_discovery(self) -> None:
        """Show 'searching' and start background discovery thread."""
        if self._requested_ip is not None and self._requested_port is not None:
            # Both supplied explicitly — connect directly without UDP discovery.
            QTimer.singleShot(0, lambda: self._on_discovered(
                self._requested_ip, self._requested_port))
            return
        self._on_connection_status("Searching for LiveOD server…")
        threading.Thread(target=self._discover_worker, daemon=True).start()

    def _discover_worker(self) -> None:
        """Background thread: loops until the matching 'live_od_broadcast' is found."""
        from waxx.util.comms_server.hardware_id import discover_scoped
        while True:
            result = discover_scoped("live_od_broadcast", timeout=3.0)
            if result is not None:
                discovered_ip, discovered_port = result
                ip   = self._requested_ip   if self._requested_ip   is not None else discovered_ip
                port = self._requested_port if self._requested_port is not None else discovered_port
                self._discovery_done_signal.emit(ip, port)
                return
            self._discovery_status_signal.emit(
                "LiveOD server not found — retrying…"
            )
            time.sleep(1.0)

    def _on_discovered(self, ip: str, port: int) -> None:
        """Called on the main thread once the server is discovered."""
        self._ip   = ip
        self._port = port
        self.connection_label.setText(f"tcp://{ip}:{port} — connecting…")
        self.msg(f"Discovered LiveOD server at tcp://{ip}:{port}")
        self._start_subscriber(ip, port)

    def _start_subscriber(self, ip: str, port: int) -> None:
        if self.subscriber is not None:
            self.subscriber.stop()
            self.subscriber.wait(1000)
        self.subscriber = LiveODSubscriber(ip, port)
        self.subscriber.od_image_signal.connect(self._on_od_image)
        self.subscriber.od_frame_age_signal.connect(self._on_frame_age)
        self.subscriber.shot_progress_signal.connect(self._on_shot_progress)
        self.subscriber.run_info_signal.connect(self._on_run_started)
        self.subscriber.log_msg_signal.connect(
            self.viewer_window.output_window.append_message)
        self.subscriber.run_state_signal.connect(self._on_run_state)
        self.subscriber.markers_signal.connect(self._on_markers)
        self.subscriber.shot_scalars_signal.connect(self.viewer_window.on_shot_scalars)
        self.subscriber.connection_status_signal.connect(self._on_connection_status)
        self.subscriber.camera_state_signal.connect(self._on_camera_state)
        self.subscriber.camera_persist_signal.connect(self.camera_menu.set_persist)
        self.subscriber.shot_scalars_signal.connect(self.live_scalar_plot_window.on_shot_scalars)
        self.subscriber.fk_tof_signal.connect(self.fk_tof_window.on_pwa_data)
        self.subscriber.adjust_values_signal.connect(self._adjust_panel.update_values)
        self.subscriber.adjust_specs_signal.connect(
            lambda specs: self._on_adjust_state_fetched(list(specs), {}))
        self.subscriber.shot_timing_signal.connect(self.status_strip.set_timing)
        self.subscriber.next_run_id_signal.connect(self.status_strip.set_next_run_id)
        self.subscriber.connected_signal.connect(self._catch_up)
        self.subscriber.start()
        # Fetch current adjust state from server in background after subscriber starts
        threading.Thread(target=self._fetch_adjust_state, daemon=True).start()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _setup_layout(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 5, 6, 6)
        layout.setSpacing(3)

        # the acquisition window's status strip, fed from the broadcast
        self.status_strip = StatusStrip()
        self.status_strip.title_changed.connect(self._on_status_title)

        # The server's cameras: the one on the button is the server's current camera,
        # the arrow drops down the others. Filled in from the first CAMERA_STATE
        # broadcast, so the remote viewer needs no camera config (or ARTIQ / kexp).
        # The acquisition window's control, without ⚙ and 🎥 (they need the host).
        self.camera_menu = CameraControl(glyphs=False)
        self.camera_menu.toggle_requested.connect(self._on_camera_button_clicked)
        self.camera_menu.hide()
        self.status_strip.add_camera_widget(self.camera_menu)

        # the person hold (monitor server), shown only while it is on
        self.hold_badge = HoldBadge(log=self.msg)
        # the last run that came through the run queue, queued again as the next job
        self.run_again_button = RunAgainButton(log=self.msg)
        self.run_again_button.result_ready.connect(self._on_run_again_result)

        # Abort, as in the acquisition window: it aborts the run in progress and
        # nothing else (with no run, the server is asked first and nothing is sent)
        self.abort_button = QPushButton("Abort")
        self.abort_button.setFixedWidth(56)
        self.abort_button.setToolTip(
            "Stop the run now and delete its data file; ignored while a run is being saved "
            "(it is complete then). No confirmation: it has to be fast.")
        self.abort_button.clicked.connect(self._on_reset_clicked)
        self.reset_button = self.abort_button       # the name it had

        self.connection_label = QLabel("Searching for LiveOD server…")
        self.connection_label.setStyleSheet("color: #9a9a9a; font-size: 10px;")
        self.connection_label.setFixedWidth(170)
        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.clicked.connect(self._on_reconnect_clicked)

        status_row = QHBoxLayout()
        status_row.setSpacing(4)
        status_row.addWidget(self.status_strip, 1)
        status_row.addWidget(self.run_again_button)
        status_row.addSpacing(8)        # apart from Abort: a slip must not abort a run
        status_row.addWidget(self.hold_badge)
        status_row.addWidget(self.abort_button)
        status_row.addWidget(self.connection_label)
        status_row.addWidget(self.reconnect_button)
        layout.addLayout(status_row)

        # Live Plot is in the viewer's toolbar; Adjust joins it there.
        self.live_plot_button = self.viewer_window.live_plot_button
        self._adjust_button.setFixedWidth(72)
        self.viewer_window.add_window_button(self._adjust_button)
        # the lab's tool windows (LiveODConfig.tool_windows), each its own process
        from waxx.util.live_od.config import get_config
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
        self._update_abort_button()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_connection_status(self, msg: str):
        self.connection_label.setText(msg)
        self.connection_label.setToolTip(msg)
        self.msg(msg)

    def _on_status_title(self, text: str):
        self.setWindowTitle(f"{self.TITLE} — {text}" if text else self.TITLE)

    def _on_run_again_result(self, text: str, ok: bool):
        """A refusal must be seen, not only logged (the log has it either way)."""
        if not ok:
            self.status_strip.show_notice(text, 12.0)

    def _on_run_started(self, msg: dict):
        """RUN_STARTED. ``info`` (newer servers) carries what the acquisition
        window knows of the run; without it the viewer shows the run id only."""
        run_id = int(msg.get("run_id", 0) or 0)
        xvarnames = list(msg.get("xvarnames") or [])
        info = msg.get("info") if isinstance(msg.get("info"), dict) else {}
        self._current_run_id = run_id
        self._run_in_progress = True
        expt_name = str(info.get("expt_name") or "")
        camera_key = str(info.get("camera_key") or "")
        self.viewer_window.clear_plots()
        if info:
            # this camera's ROI and OD levels as this PC last left them (display only)
            self.viewer_window.set_camera_key(camera_key)
            self.viewer_window.set_pixel_size_m(info.get("pixel_size_m"))
            self.viewer_window.set_xvar_ranges(dict(info.get("xvar_ranges") or {}))
            if camera_key:
                self.camera_menu.set_current(camera_key)
        self.viewer_window.on_new_run()
        self.status_strip.start_run(run_id, expt_class=expt_name,
                                    camera_key=camera_key,
                                    save_data=bool(info.get("save_data", True)),
                                    n_shots=int(info.get("n_shots", 0) or 0))
        self.connection_label.setText(f"tcp://{self._ip}:{self._port}")
        self.viewer_window.output_window.append_separator(
            " · ".join(p for p in (f"Run {run_id}" if run_id else "Run (not saved)", expt_name)
                       if p))
        self.live_scalar_plot_window.on_new_run(run_id, xvarnames)
        self.fk_tof_window.on_new_run(run_id, dict(info.get("params") or {}))
        # the run queue's job ("" for a direct run; None: an older server did not say)
        self.run_again_button.note_run(
            run_id, expt_name,
            str(info.get("queue_job") or "") if "queue_job" in info else None,
            str(info.get("launcher") or ""))
        if "info" not in msg:
            # an older server sends no ADJUST_SPECS: ask for the run's adjust params
            threading.Thread(target=self._fetch_adjust_state, daemon=True).start()
        self._update_abort_button()

    def _on_shot_progress(self, shot_idx: int, N_total: int,
                          xvar_values: object):
        self.status_strip.set_progress(shot_idx + 1, N_total)
        self.viewer_window.update_image_count(shot_idx + 1, N_total)
        self.viewer_window.set_shot_xvars(shot_idx, xvar_values)
        # no log line here: the server's own (about one shot in twenty) arrives as LOG_MSG

    def _on_markers(self, camera_key: str, markers: list):
        """The server's pins for the camera it is on (sent on change and every 2 s)."""
        self._marker_camera_key = camera_key
        if camera_key:
            self.camera_menu.set_current(camera_key)
        self.viewer_window.set_markers(markers)     # a no-op when nothing changed

    def _on_viewer_markers_changed(self, _viewer_key: str, markers: list):
        """A pin was edited here: the server keeps them, under its current camera."""
        threading.Thread(
            target=self._send_req_to_server,
            args=({'tag': 'SET_MARKERS', 'camera_key': self._marker_camera_key,
                   'markers': list(markers)}, "SET_MARKERS"),
            daemon=True,
        ).start()

    def _send_req_to_server(self, payload: dict, label: str):
        self._request(payload, label)

    def _on_run_state(self, state: str, detail: str):
        """RUN_STATE: the pill, as in the acquisition window. RUN_DONE follows a
        run's last state and changes nothing here (it used to overwrite an
        Aborted / Exited / INCOMPLETE run with "complete")."""
        self.status_strip.set_state(state, detail)
        self._run_in_progress = state in ACTIVE_STATES
        self._update_abort_button()
        if state in ("saved", "done", "error"):
            QApplication.alert(self, 0)     # flash the taskbar entry until focused

    def _update_abort_button(self):
        self.abort_button.setStyleSheet(
            'background-color: #e53935; color: white; font-weight: bold;'
            if self._run_in_progress else '')

    def _on_od_image(self, plot_data: tuple):
        self.plotting_queue.put(plot_data)

    def _on_frame_age(self, age_s: float):
        self.connection_label.setText(
            f"tcp://{self._ip}:{self._port}  —  frame age {age_s*1e3:.0f} ms"
        )

    # ------------------------------------------------------------------
    # Catching up: a viewer that joins after the run started
    # ------------------------------------------------------------------

    def _catch_up(self):
        """On every (re)connection: POLL the server for the run in progress and
        GET_LOG its log so far (worker thread; read-only requests)."""
        def work():
            poll = self._request({'tag': 'POLL'}, "POLL", quiet=True)
            log = None
            if isinstance(poll, dict) and poll.get('ok'):
                log = self._request({'tag': 'GET_LOG', 'min_level': logging.INFO,
                                     'limit': 400}, "GET_LOG", quiet=True)
            self._on_gui_signal.emit(lambda: self._apply_catch_up(poll, log))
        threading.Thread(target=work, daemon=True).start()

    def _apply_catch_up(self, poll, log):
        """POLL's and GET_LOG's answers, on the GUI thread. A run the viewer
        already knows of (it saw RUN_STARTED) is left as it is."""
        if not (isinstance(poll, dict) and poll.get('ok')):
            return
        in_progress = bool(poll.get('run_in_progress'))
        if in_progress and self.status_strip._run_start is None:
            run_id = int(poll.get('run_id') or 0)
            self._current_run_id = run_id
            camera_key = str(poll.get('run_camera_key') or '')
            n_expected = int(poll.get('n_shots_expected') or 0)
            self.status_strip.start_run(
                run_id, expt_class=str(poll.get('expt_name') or ''), camera_key=camera_key,
                save_data=bool(poll.get('save_data', True)), n_shots=n_expected)
            if camera_key:
                self.camera_menu.set_current(camera_key)
            n_done = int(poll.get('n_shots') or 0)
            if n_done:
                self.status_strip.set_progress(n_done, n_expected)
        self.status_strip.set_state(str(poll.get('run_state') or 'idle'), "")
        self._run_in_progress = in_progress
        self._update_abort_button()
        # Run again next, for a viewer that saw no RUN_STARTED: POLL names the
        # current (or last) run's queue job; init_run_age_s is None before any run
        if (self.run_again_button.latest is None and 'queue_job' in poll
                and poll.get('init_run_age_s') is not None):
            self.run_again_button.note_run(poll.get('run_id'), str(poll.get('expt_name') or ''),
                                           str(poll.get('queue_job') or ''),
                                           str(poll.get('launcher') or ''))
        if isinstance(log, dict) and log.get('ok') and log.get('records'):
            run = log.get('run') or {}
            title = f"Run {run.get('run_id')}" if run.get('run_id') else "Run (not saved)"
            out = self.viewer_window.output_window
            out.append_separator(f"{title}: the server's log before this viewer joined")
            for record in log['records']:
                out.append_record(int(record.get('level', logging.INFO)),
                                  str(record.get('msg', '')), record.get('t'))
            out.append_separator("live from here")

    # ------------------------------------------------------------------
    # Reconnect
    # ------------------------------------------------------------------

    def _on_reconnect_clicked(self):
        self.msg("Reconnecting — searching for LiveOD server…")
        self._req_ip = None
        self._req_port = None
        if self.subscriber is not None:
            self.subscriber.stop()
            self.subscriber.wait(1000)
            self.subscriber = None
        threading.Thread(target=self._discover_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Abort
    # ------------------------------------------------------------------

    def _on_reset_clicked(self):
        threading.Thread(target=self._send_reset_to_server, daemon=True).start()

    def _send_reset_to_server(self):
        """Abort the run in progress, by its run id, so a late press never hits
        the next run. The server is asked first which run that is (this window
        may have missed a broadcast): with no run in progress nothing is sent --
        a RESET with no run would count as a person's Reset and put the hold on."""
        poll = self._request({'tag': 'POLL'}, "POLL", quiet=True)
        if not isinstance(poll, dict):
            self._on_gui_signal.emit(lambda: self.msg(
                "Abort: the liveOD server did not answer; nothing was sent.", logging.ERROR))
            return
        if not poll.get('run_in_progress'):
            self._on_gui_signal.emit(lambda: self.msg("Abort: there is no run to abort."))
            return
        run_id = int(poll.get('run_id') or 0)
        name = f"run {run_id}" if run_id else "the unsaved run"
        self._on_gui_signal.emit(lambda: self.msg(f"Abort: sending Reset for {name}…"))
        payload = {'tag': 'RESET', 'source': 'person'}
        if run_id:
            payload['run_id'] = run_id      # refused if another run is in progress by then
        self._request(payload, "Reset")

    # ------------------------------------------------------------------
    # Live scalar plot, FK TOF, Adjust
    # ------------------------------------------------------------------

    def _open_live_scalar_plot(self):
        self.live_scalar_plot_window.show()
        self.live_scalar_plot_window.raise_()

    def _open_fk_tof(self):
        self.fk_tof_window.show()
        self.fk_tof_window.raise_()

    def _open_adjust_panel(self):
        if self._adjust_panel.isMinimized():
            self._adjust_panel.showNormal()
        self._adjust_panel.show()
        self._adjust_panel.raise_()
        self._adjust_panel.activateWindow()

    def _fetch_adjust_state(self):
        """Worker thread: GET_ADJUST_VALUES from the REP server, then signal GUI thread."""
        reply = self._request({'tag': 'GET_ADJUST_VALUES'}, "GET_ADJUST_VALUES", quiet=True)
        if isinstance(reply, dict) and reply.get('ok'):     # empty specs too: the panel clears
            self._adjust_state_fetched_signal.emit(
                list(reply.get('specs') or []), dict(reply.get('values', {})))

    def _on_adjust_state_fetched(self, specs: list, values: object):
        """GET_ADJUST_VALUES's answer, or an ADJUST_SPECS broadcast (no values:
        they come with ADJUST_VALUES)."""
        self._adjust_panel.populate(specs)
        if values:
            self._adjust_panel.update_values(dict(values))
        count = len(specs)
        self._adjust_button.setText(f"Adjust ({count})" if count else "Adjust")
        self._adjust_button.setEnabled(bool(count))

    def _on_remote_adjust_value_changed(self, key: str, value: float):
        threading.Thread(
            target=self._request,
            args=({'tag': 'SET_ADJUST_VALUE', 'key': key, 'value': value},
                  f'SET_ADJUST_VALUE({key})'),
            daemon=True,
        ).start()

    def _on_remote_spec_updated(self, key: str, min_val: float, max_val: float, step: float):
        """Sync new bounds to the server when cog dialog is accepted in the remote viewer."""
        threading.Thread(
            target=self._request,
            args=({'tag': 'SET_ADJUST_SPEC', 'key': key, 'min_val': min_val,
                   'max_val': max_val, 'step': step}, f'SET_ADJUST_SPEC({key})'),
            daemon=True,
        ).start()

    def _on_scalar_subscription_changed(self, old_tier, new_tier):
        """Send SUBSCRIBE / UNSUBSCRIBE REQ messages when the plot window
        opens, closes, or switches metric tier."""
        for tag, tier in (("UNSUBSCRIBE_SCALARS", old_tier), ("SUBSCRIBE_SCALARS", new_tier)):
            if tier is not None:
                threading.Thread(target=self._request, args=({"tag": tag, "tier": tier}, tag),
                                 daemon=True).start()

    # ------------------------------------------------------------------
    # Camera control
    # ------------------------------------------------------------------

    def _on_camera_state(self, states: dict):
        self.camera_menu.set_states(states)
        self.camera_menu.setVisible(bool(states))
        # An incoming state update is the server's authoritative echo —
        # re-enable any camera that was greyed out after a click.
        for key in states:
            self.camera_menu.set_camera_enabled(key, True)

    def _on_camera_button_clicked(self, camera_key: str):
        # Disable until the server broadcasts the new CAMERA_STATE or the
        # worker thread's request fails.  Prevents queued duplicate REQs.
        self.camera_menu.set_camera_enabled(camera_key, False)
        self.msg(f"Toggling camera {camera_key}…")
        threading.Thread(
            target=self._send_camera_control,
            args=(camera_key, 'toggle'),
            daemon=True,
        ).start()

    def _on_camera_request_done(self, camera_key: str):
        """Called from the GUI thread when a CAMERA_CONTROL REQ finishes: the
        next CAMERA_STATE broadcast re-enables the camera; if none comes within
        5 s, this does."""
        QTimer.singleShot(5000, lambda k=camera_key: self.camera_menu.set_camera_enabled(k, True))

    def _send_camera_control(self, camera_key: str, action: str):
        self._request({'tag': 'CAMERA_CONTROL', 'camera_key': camera_key, 'action': action},
                      "Camera control")
        self._camera_request_done_signal.emit(camera_key)

    # ------------------------------------------------------------------
    # Requests to the server's REP port
    # ------------------------------------------------------------------

    def _resolve_req_endpoint(self) -> tuple[str | None, int | None]:
        """Return cached (ip, port) for the live_od REP server, discovering
        via UDP if no cached value exists."""
        if self._req_ip is not None and self._req_port is not None:
            return self._req_ip, self._req_port
        from waxx.util.comms_server.hardware_id import discover_scoped
        result = discover_scoped("live_od", timeout=3.0)
        if result is None:
            return None, None
        self._req_ip, self._req_port = result
        return self._req_ip, self._req_port

    def _request(self, payload: dict, label: str, quiet: bool = False):
        """``payload`` to the server (worker thread) -> its reply dict, or None.
        A failed send drops the cached endpoint and tries once more after a
        new discovery. A reply that refuses or ignores the request is shown
        (``_show_reply``) unless ``quiet``."""
        for _attempt in range(2):
            ip, port = self._resolve_req_endpoint()
            if ip is None:
                print(f"[RemoteViewer] {label} failed: live_od server not discovered")
                return None
            reply = self._send_req(ip, port, payload, label=label, quiet=quiet,
                                   return_reply=True)
            if reply is not None:
                return reply
            self._req_ip = None         # cached endpoint stale: rediscover once
            self._req_port = None
        return None

    def _send_req(self, ip: str, port: int, payload: dict, label: str = "REQ",
                  quiet: bool = False, return_reply: bool = False):
        """One REQ/REP round trip. True / False (sent or not), or with
        ``return_reply`` the reply dict / None."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDTIMEO, 3000)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(f"tcp://{ip}:{port}")
            sock.send(pickle.dumps(payload))
            reply = pickle.loads(sock.recv())
            if isinstance(reply, dict) and not reply.get('ok', True):
                print(f"[RemoteViewer] {label} reply: {reply}")
            if not quiet:
                self._show_reply(label, reply)
            return reply if return_reply else True
        except Exception as exc:
            print(f"[RemoteViewer] {label} failed: {exc}")
            return None if return_reply else False
        finally:
            sock.close()
            ctx.term()

    @staticmethod
    def reply_notice_text(label: str, reply) -> str:
        """The text for a reply that says the request was ignored or refused
        (a Reset while the run is being saved, or naming another run; any
        other request the server answered ``ok: False`` with an error, such as
        a camera action during a run) or that it closed a run whose process is
        gone, keeping its file; "" for any other reply."""
        if not isinstance(reply, dict):
            return ""
        if not (reply.get('ignored') or reply.get('saving') or reply.get('refused')
                or reply.get('kept')):
            if reply.get('ok', True) is False and reply.get('error'):
                return f"{label} refused: {reply['error']}"
            return ""
        if reply.get('kept') and reply.get('message'):
            return str(reply['message'])
        text = str(reply.get('message') or reply.get('error') or '')
        if not text:
            what = "ignored" if reply.get('ignored') else "refused"
            run = f"run {reply['run_id']} " if reply.get('run_id') else "the run "
            text = f"{label} {what}: {run}is being saved (it is complete)"
        return text

    def _show_reply(self, label: str, reply: dict):
        """Called on the request's (worker) thread: such a reply is shown on
        every press, in the log and in place of the run label, on the GUI
        thread (signal)."""
        text = self.reply_notice_text(label, reply)
        if text:
            print(f"[RemoteViewer] {text}")
            self._reply_notice_signal.emit(text)

    def _show_reply_notice(self, text: str):
        self.viewer_window.output_window.append_message(text, logging.WARNING)
        self.status_strip.show_notice(text)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._settings is not None:
            try:
                self._settings.setValue("window/geometry", self.saveGeometry())
                self.viewer_window.save_settings()
            except Exception as exc:
                print(f"[RemoteViewer] could not save the settings: {exc}")
        if self.subscriber is not None:
            self.subscriber.stop()
            self.subscriber.wait(2000)
        self.hold_badge.shutdown()
        self.run_again_button.shutdown()
        self.plotter.quit()
        self.plotter.wait(2000)
        # the viewer's own windows go with it, so nothing keeps the process alive
        for window in (self.live_scalar_plot_window, self.fk_tof_window, self._adjust_panel):
            try:
                window.close()
            except Exception:
                pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """``[ip] [port]``: the broadcast's address; by default it is discovered."""
    argv = list(sys.argv if argv is None else argv)
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "weldlab.kexp.gui.live_od_viewer"
        )
    except Exception:
        pass        # not Windows
    from waxa.taskbar import group_console
    group_console()  # console joins the shared terminals taskbar button

    ip = argv[1] if len(argv) > 1 else None
    port = int(argv[2]) if len(argv) > 2 else None

    app = QApplication(argv)
    # layout, toggles, theme and per-camera OD levels are remembered between
    # sessions (apart from the acquisition window's, which may be on this PC too)
    win = RemoteViewerWindow(ip=ip, port=port, settings=QSettings("waxx", "live_od_viewer"))
    win.setWindowTitle(RemoteViewerWindow.TITLE)
    win.setWindowIcon(
        win.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView)
    )
    win.show()
    win.hold_badge.start()      # the person hold, from the monitor server
    win.run_again_button.start()    # Run again next, through the run queue
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
