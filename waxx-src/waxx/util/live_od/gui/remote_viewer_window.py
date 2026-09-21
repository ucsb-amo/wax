"""
Standalone LiveOD remote viewer window.

Subscribes to the LiveOD Server's ZMQ PUB broadcast and displays
live OD images and shot progress without needing direct access to
the data drive or camera hardware.

Usage (run from any machine on the lab network)::

    python -m waxx.util.live_od.gui.remote_viewer_window
    python -m waxx.util.live_od.gui.remote_viewer_window 192.168.1.76
    python -m waxx.util.live_od.gui.remote_viewer_window 192.168.1.76 5561
"""

import pickle
import sys
import threading
import time
from queue import Queue

import zmq
from PyQt6.QtCore import QThread, Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
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
from waxx.util.live_od.gui.camera_menu import CameraMenuButton


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
    shot_scalars_signal = pyqtSignal(object)        # per-shot scalar dict
    fk_tof_signal = pyqtSignal(object)              # per-shot FK TOF width dict
    adjust_values_signal = pyqtSignal(object)       # dict[key -> current_val]

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
                        elif tag == "SHOT_SCALARS":
                            self.shot_scalars_signal.emit(dict(msg))
                        elif tag == "FK_TOF":
                            self.fk_tof_signal.emit(dict(msg))
                        elif tag == "ADJUST_VALUES":
                            self.adjust_values_signal.emit(dict(msg.get('values', {})))
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
    """Standalone window that subscribes to a LiveODBroadcaster and displays
    live OD images, sum-OD projections, and shot progress."""

    _discovery_status_signal = pyqtSignal(str)
    _discovery_done_signal   = pyqtSignal(str, int)   # ip, port
    _camera_request_done_signal = pyqtSignal(str)    # camera_key (worker -> GUI re-enable)
    _adjust_state_fetched_signal = pyqtSignal(list, object)  # specs, values dict

    def __init__(self, ip: str = None, port: int = None):
        super().__init__()

        self._requested_ip   = ip
        self._requested_port = port
        self._ip   = ip   or "—"
        self._port = port or 0

        # Viewer + plotter (independent of network)
        self.plotting_queue: Queue = Queue()
        self.viewer_window = LiveODViewer()
        self.plotter = LiveODPlotter(self.viewer_window, self.plotting_queue)
        self.plotter.start()

        self.subscriber: LiveODSubscriber | None = None
        self._current_run_id: int = 0
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
        self._adjust_button.setMinimumHeight(40)
        self._adjust_button.clicked.connect(self._open_adjust_panel)

        # Cached endpoint for the LiveOD REP server (RESET / CAMERA_CONTROL).
        # Populated lazily on the first request that succeeds; falls back to
        # UDP discover() if a request fails (server moved or restarted).
        self._req_ip: str | None = None
        self._req_port: int | None = None

        self._setup_layout()

        # Connect discovery signals (emitted from worker thread)
        self._discovery_status_signal.connect(self._on_connection_status)
        self._discovery_done_signal.connect(self._on_discovered)
        self._camera_request_done_signal.connect(self._on_camera_request_done)
        self._adjust_state_fetched_signal.connect(self._on_adjust_state_fetched)

        # Kick off discovery after the event loop starts
        QTimer.singleShot(0, self._start_discovery)

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
        self.viewer_window.output_window.appendPlainText(
            f"Discovered LiveOD server at tcp://{ip}:{port}"
        )
        self._start_subscriber(ip, port)

    def _start_subscriber(self, ip: str, port: int) -> None:
        if self.subscriber is not None:
            self.subscriber.stop()
            self.subscriber.wait(1000)
        self.subscriber = LiveODSubscriber(ip, port)
        self.subscriber.od_image_signal.connect(self._on_od_image)
        self.subscriber.od_frame_age_signal.connect(self._on_frame_age)
        self.subscriber.shot_progress_signal.connect(self._on_shot_progress)
        self.subscriber.run_started_signal.connect(self._on_run_started)
        self.subscriber.run_done_signal.connect(self._on_run_done)
        self.subscriber.log_msg_signal.connect(
            self.viewer_window.output_window.append_message)
        self.subscriber.run_state_signal.connect(self._on_run_state)
        self.subscriber.markers_signal.connect(self._on_markers)
        self.subscriber.shot_scalars_signal.connect(self.viewer_window.on_shot_scalars)
        self.subscriber.connection_status_signal.connect(self._on_connection_status)
        self.subscriber.camera_state_signal.connect(self._on_camera_state)
        self.subscriber.shot_scalars_signal.connect(self.live_scalar_plot_window.on_shot_scalars)
        self.subscriber.fk_tof_signal.connect(self.fk_tof_window.on_pwa_data)
        self.subscriber.adjust_values_signal.connect(self._adjust_panel.update_values)
        self.subscriber.start()
        # Fetch current adjust state from server in background after subscriber starts
        threading.Thread(target=self._fetch_adjust_state, daemon=True).start()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _setup_layout(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Status bar
        status_bar = QHBoxLayout()

        label_col = QVBoxLayout()
        label_col.setSpacing(2)

        self.run_id_label = QLabel("")
        self.run_id_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        run_id_font = QFont()
        run_id_font.setBold(True)
        run_id_font.setPointSize(13)
        self.run_id_label.setFont(run_id_font)

        self.connection_label = QLabel("Searching for LiveOD server…")
        self.connection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conn_font = QFont()
        conn_font.setPointSize(9)
        self.connection_label.setFont(conn_font)

        label_col.addWidget(self.run_id_label)
        label_col.addWidget(self.connection_label)
        status_bar.addLayout(label_col)

        # The server's cameras: the one on the button is the server's current camera,
        # the arrow drops down the others. Filled in from the first CAMERA_STATE
        # broadcast, so the remote viewer needs no camera config (or ARTIQ / kexp).
        self.camera_menu = CameraMenuButton()
        self.camera_menu.toggle_requested.connect(self._on_camera_button_clicked)
        self.camera_menu.hide()
        status_bar.addWidget(self.camera_menu)

        # The server decides what this does: abort the run if one is active,
        # otherwise skip a run ID (the acquisition window has a button for each).
        self.reset_button = QPushButton("Abort run / skip ID")
        self.reset_button.setStyleSheet(
            'background-color: #e53935; color: white; font-weight: bold; padding: 4px 14px;'
        )
        self.reset_button.setToolTip(
            "Abort the active run and delete its data file; with no run active, "
            "skip to the next run ID. No confirmation.")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        status_bar.addWidget(self.reset_button)

        # Live Plot is in the viewer's toolbar; Adjust joins it there.
        self.live_plot_button = self.viewer_window.live_plot_button
        self._adjust_button.setMinimumHeight(0)
        self.viewer_window.add_window_button(self._adjust_button)

        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.clicked.connect(self._on_reconnect_clicked)
        status_bar.addWidget(self.reconnect_button)

        layout.addLayout(status_bar)
        layout.addWidget(self.viewer_window)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_connection_status(self, msg: str):
        self.connection_label.setText(msg)
        self.viewer_window.output_window.appendPlainText(msg)

    def _on_run_started(self, run_id: int, xvarnames: object):
        self._current_run_id = run_id
        self.run_id_label.setText(f"Run {run_id} — in progress")
        self.connection_label.setText(f"tcp://{self._ip}:{self._port}")
        self.viewer_window.clear_plots()
        self.viewer_window.on_new_run()
        self.viewer_window.output_window.append_separator(
            f"Run {run_id}" if run_id else "Run (not saved)")
        self.live_scalar_plot_window.on_new_run(run_id, list(xvarnames) if xvarnames else [])
        # a new run may adjust a different set of params — refresh the panel
        threading.Thread(target=self._fetch_adjust_state, daemon=True).start()
        self.fk_tof_window.on_new_run(run_id, {})

    def _on_shot_progress(self, shot_idx: int, N_total: int,
                          xvar_values: object):
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
        ip, port = self._resolve_req_endpoint()
        if ip is None:
            print(f"[RemoteViewer] {label} failed: live_od server not discovered")
            return
        self._send_req(ip, port, payload, label=label)

    def _on_run_state(self, state: str, detail: str):
        from waxx.util.live_od.gui.status_strip import STATES
        text = STATES.get(state, (state,))[0]
        run = f"Run {self._current_run_id} — " if self._current_run_id else ""
        self.run_id_label.setText(f"{run}{text}")
        self.run_id_label.setToolTip(detail)

    def _on_od_image(self, plot_data: tuple):
        self.plotting_queue.put(plot_data)

    def _on_frame_age(self, age_s: float):
        self.connection_label.setText(
            f"tcp://{self._ip}:{self._port}  —  frame age {age_s*1e3:.0f} ms"
        )

    def _on_run_done(self):
        self.viewer_window.output_window.appendPlainText("Run complete.")
        run_id_str = f"Run {self._current_run_id} — " if self._current_run_id else ""
        self.run_id_label.setText(f"{run_id_str}complete")
        self.connection_label.setText(f"tcp://{self._ip}:{self._port}")

    # ------------------------------------------------------------------
    # Reconnect
    # ------------------------------------------------------------------

    def _on_reconnect_clicked(self):
        self.viewer_window.output_window.appendPlainText("Reconnecting — searching for LiveOD server…")
        self._req_ip = None
        self._req_port = None
        if self.subscriber is not None:
            self.subscriber.stop()
            self.subscriber.wait(1000)
            self.subscriber = None
        threading.Thread(target=self._discover_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _on_reset_clicked(self):
        self.viewer_window.output_window.appendPlainText("Sending Reset to server…")
        threading.Thread(target=self._send_reset_to_server, daemon=True).start()

    # ------------------------------------------------------------------
    # Live scalar plot
    # ------------------------------------------------------------------

    def _open_live_scalar_plot(self):
        self.live_scalar_plot_window.show()
        self.live_scalar_plot_window.raise_()

    def _open_fk_tof(self):
        self.fk_tof_window.show()
        self.fk_tof_window.raise_()

    def _open_adjust_panel(self):
        self._adjust_panel.show()
        self._adjust_panel.raise_()

    def _fetch_adjust_state(self):
        """Worker thread: GET_ADJUST_VALUES from the REP server, then signal GUI thread."""
        try:
            ip, port = self._resolve_req_endpoint()
            if ip is None:
                return
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.SNDTIMEO, 3000)
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            sock.setsockopt(zmq.LINGER, 0)
            try:
                import pickle
                sock.connect(f"tcp://{ip}:{port}")
                sock.send(pickle.dumps({'tag': 'GET_ADJUST_VALUES'}))
                reply = pickle.loads(sock.recv())
                if reply.get('ok'):          # empty specs too: the panel clears
                    self._adjust_state_fetched_signal.emit(
                        list(reply['specs']), dict(reply.get('values', {}))
                    )
            except Exception as exc:
                print(f"[RemoteViewer] GET_ADJUST_VALUES failed: {exc}")
            finally:
                sock.close()
                ctx.term()
        except Exception as exc:
            print(f"[RemoteViewer] _fetch_adjust_state error: {exc}")

    def _on_adjust_state_fetched(self, specs: list, values: object):
        """Called on GUI thread after successful GET_ADJUST_VALUES."""
        self._adjust_panel.populate(specs)
        self._adjust_panel.update_values(dict(values))
        count = len(specs)
        self._adjust_button.setText(f"Adjust ({count})" if count else "Adjust")

    def _on_remote_adjust_value_changed(self, key: str, value: float):
        threading.Thread(
            target=self._send_set_adjust_value, args=(key, value), daemon=True
        ).start()

    def _send_set_adjust_value(self, key: str, value: float):
        ip, port = self._resolve_req_endpoint()
        if ip is None:
            return
        self._send_req(ip, port, {'tag': 'SET_ADJUST_VALUE', 'key': key, 'value': value},
                       label=f'SET_ADJUST_VALUE({key})')

    def _on_remote_spec_updated(self, key: str, min_val: float, max_val: float, step: float):
        """Sync new bounds to the server when cog dialog is accepted in the remote viewer."""
        threading.Thread(
            target=self._send_set_adjust_spec,
            args=(key, min_val, max_val, step),
            daemon=True,
        ).start()

    def _send_set_adjust_spec(self, key: str, min_val: float, max_val: float, step: float):
        ip, port = self._resolve_req_endpoint()
        if ip is None:
            return
        self._send_req(
            ip, port,
            {'tag': 'SET_ADJUST_SPEC', 'key': key, 'min_val': min_val,
             'max_val': max_val, 'step': step},
            label=f'SET_ADJUST_SPEC({key})',
        )

    def _on_scalar_subscription_changed(self, old_tier, new_tier):
        """Send SUBSCRIBE / UNSUBSCRIBE REQ messages when the plot window
        opens, closes, or switches metric tier."""
        if old_tier is not None:
            threading.Thread(
                target=self._send_scalar_subscription_req,
                args=("UNSUBSCRIBE_SCALARS", old_tier),
                daemon=True,
            ).start()
        if new_tier is not None:
            threading.Thread(
                target=self._send_scalar_subscription_req,
                args=("SUBSCRIBE_SCALARS", new_tier),
                daemon=True,
            ).start()

    def _send_scalar_subscription_req(self, tag: str, tier: str):
        ip, port = self._resolve_req_endpoint()
        if ip is None:
            print(f"[RemoteViewer] {tag} failed: server not discovered")
            return
        self._send_req(ip, port, {"tag": tag, "tier": tier}, label=tag)

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
        self.viewer_window.output_window.appendPlainText(
            f"Toggling camera {camera_key}…"
        )
        threading.Thread(
            target=self._send_camera_control,
            args=(camera_key, 'toggle'),
            daemon=True,
        ).start()

    def _on_camera_request_done(self, camera_key: str):
        """Called from the GUI thread when a CAMERA_CONTROL REQ finishes.

        On success we keep the button disabled; the next CAMERA_STATE
        broadcast will re-enable it.  On failure we re-enable immediately so
        the user can retry.  Either way, schedule a fallback re-enable after
        a few seconds in case the broadcast never arrives.
        """
        # Fallback: if no CAMERA_STATE arrives within 5 s, re-enable the
        # camera so it doesn't stay greyed out forever.
        QTimer.singleShot(5000, lambda k=camera_key: self.camera_menu.set_camera_enabled(k, True))

    def _send_camera_control(self, camera_key: str, action: str):
        ip, port = self._resolve_req_endpoint()
        if ip is None:
            print("[RemoteViewer] Camera control failed: live_od server not discovered")
            self._camera_request_done_signal.emit(camera_key)
            return
        ok = self._send_req(
            ip, port,
            {'tag': 'CAMERA_CONTROL', 'camera_key': camera_key, 'action': action},
            label="Camera control",
        )
        if not ok:
            # Cached endpoint stale — rediscover once and retry.
            self._req_ip = None
            self._req_port = None
            ip, port = self._resolve_req_endpoint()
            if ip is not None:
                self._send_req(
                    ip, port,
                    {'tag': 'CAMERA_CONTROL', 'camera_key': camera_key, 'action': action},
                    label="Camera control (retry)",
                )
        self._camera_request_done_signal.emit(camera_key)

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

    def _send_req(self, ip: str, port: int, payload: dict, label: str = "REQ") -> bool:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDTIMEO, 3000)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(f"tcp://{ip}:{port}")
            sock.send(pickle.dumps(payload))
            reply = pickle.loads(sock.recv())
            if not reply.get('ok', True):
                print(f"[RemoteViewer] {label} reply: {reply}")
            return True
        except Exception as exc:
            print(f"[RemoteViewer] {label} failed: {exc}")
            return False
        finally:
            sock.close()
            ctx.term()

    def _send_reset_to_server(self):
        ip, port = self._resolve_req_endpoint()
        if ip is None:
            print("[RemoteViewer] Reset failed: live_od server not discovered")
            return
        ok = self._send_req(ip, port, {'tag': 'RESET'}, label="Reset")
        if not ok:
            # Cached endpoint stale — rediscover once and retry.
            self._req_ip = None
            self._req_port = None
            ip, port = self._resolve_req_endpoint()
            if ip is not None:
                self._send_req(ip, port, {'tag': 'RESET'}, label="Reset (retry)")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.subscriber is not None:
            self.subscriber.stop()
            self.subscriber.wait(2000)
        self.plotter.quit()
        self.plotter.wait(2000)
        self.live_scalar_plot_window.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "weldlab.kexp.gui.live_od_viewer"
    )

    ip = sys.argv[1] if len(sys.argv) > 1 else None
    port = int(sys.argv[2]) if len(sys.argv) > 2 else None

    app = QApplication(sys.argv)
    from waxx.util.live_od.gui import theme
    theme.apply_theme(True)     # the ⚙ button in the toolbar switches it for this session
    win = RemoteViewerWindow(ip=ip, port=port)
    win.setWindowTitle("LiveOD Viewer")
    win.setWindowIcon(
        win.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView)
    )
    # win.resize(1400, 900)
    win.show()
    sys.exit(app.exec())
