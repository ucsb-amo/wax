"""
Camera-server GUI window.

Provides ``StatusLight`` and ``CameraServerWindow`` — a standalone PyQt6
window that hosts a :class:`CameraServer`, displays live log output, and
shows status indicators for the server, camera, grab loop and connected
viewers.

The window is fully self-contained: pass *host*, *port*, and a
*camera_nanny* and it will create / manage the ``CameraServer`` internally.
"""

import sys
import threading
import time
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPlainTextEdit, QPushButton,
)
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtCore import Qt, QTimer

from waxx.util.live_od.camera_server import CameraServer
from waxx.util.live_od.camera_nanny import CameraNanny
from waxx.util.live_od.viewer_client import ViewerClient
from waxx.util.live_od.gui.viewer_window import XVarDisplay
from waxx.util.live_od.gui.shot_plot_window import ShotPlotWindow
from waxa.data.server_talk import server_talk as st


# ======================================================================
#  Status indicator widget
# ======================================================================

class StatusLight(QWidget):
    """Coloured dot + label."""

    def __init__(self, label_text):
        super().__init__()
        self._dot = QFrame()
        self._dot.setFixedSize(16, 16)
        self._label = QLabel(label_text)
        self._label.setFont(QFont("Segoe UI", 10))
        self._label.setStyleSheet("color: #e6f2ff; font-weight: 600;")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._dot)
        layout.addWidget(self._label)
        layout.addStretch()
        self.setLayout(layout)
        self.set_state("off")

    _COLOURS = {
        "off":     "gray",
        "ok":      "#22cc44",
        "active":  "#3399ff",
        "error":   "#ee3333",
        "waiting": "#eeaa00",
    }

    def set_state(self, state: str):
        colour = self._COLOURS.get(state, "gray")
        self._dot.setStyleSheet(
            f"background-color: {colour}; border-radius: 8px; "
            f"border: 1px solid #222;"
        )

    def set_label(self, text: str):
        self._label.setText(text)


# ======================================================================
#  Main server window
# ======================================================================

class CameraServerWindow(QWidget):
    """
    PyQt6 window that hosts a :class:`CameraServer`.

    Parameters
    ----------
    host : str
        IP address the server should bind to.
    port : int
        TCP port for experiment-client commands (viewer port = *port* + 1).
    camera_nanny : CameraNanny, optional
        Manages camera connections.  A default instance is created if *None*.
    """

    def __init__(self, host: str, port: int,
                camera_nanny=None,
                server_talk=None):
        super().__init__()
        self._host = host
        self._port = port
        self.setWindowTitle("Camera Server")

        if server_talk == None:
            server_talk = st()
        else:
            server_talk = server_talk
        self.server_talk = server_talk

        # ---- Server instance ----
        self.camera_nanny = camera_nanny or CameraNanny()
        self.server = CameraServer(
            host=self._host,
            port=self._port,
            camera_nanny=self.camera_nanny,
        )

        # ---- Widgets ----
        self.server_light = StatusLight(
            f"Server  {self._host}:{self._port}"
        )
        self.camera_light = StatusLight("Camera: –")
        self.grab_light = StatusLight("Grab loop: idle")
        self.viewer_light = StatusLight("Viewers: 0 connected")
        self.run_label = QLabel("Run ID: –")
        self.run_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.run_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.run_label.setStyleSheet("color: #f2f8ff;")

        self.xvar_display = XVarDisplay()

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setMaximumBlockCount(2000)
        self.log_box.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #0b1118;"
            "  color: #e6f2ff;"
            "  border: 1px solid #2d435a;"
            "  border-radius: 8px;"
            "}"
        )

        self.toggle_server_button = QPushButton("Start Server")
        self.toggle_server_button.setStyleSheet(
            "QPushButton {"
            "  background: #1e4f39; color: #e9fff2;"
            "  border: 1px solid #4b876a; border-radius: 8px;"
            "  font-size: 14px; font-weight: 700; padding: 8px 14px;"
            "}"
            "QPushButton:hover { background: #27684b; }"
        )
        self.toggle_server_button.clicked.connect(self._toggle_server)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet(
            "QPushButton {"
            "  background: #6e2a2f; color: #ffeef0;"
            "  border: 1px solid #9a4a50; border-radius: 8px;"
            "  font-size: 14px; font-weight: 700; padding: 8px 14px;"
            "}"
            "QPushButton:hover { background: #84343a; }"
        )
        self.reset_button.clicked.connect(self._send_reset)

        self.new_plot_button = QPushButton("New Plot")
        self.new_plot_button.setStyleSheet(
            "QPushButton {"
            "  background: #203754; color: #eaf5ff;"
            "  border: 1px solid #4d78a8; border-radius: 8px;"
            "  font-size: 14px; font-weight: 700; padding: 8px 14px;"
            "}"
            "QPushButton:hover { background: #29466a; }"
        )
        self.new_plot_button.clicked.connect(self._open_new_plot)

        self.setStyleSheet(
            "QWidget { background: #0b1118; color: #e6f2ff; }"
            "QGroupBox, QFrame { color: #e6f2ff; }"
        )

        # ---- Layout ----
        lights = QHBoxLayout()
        lights.addWidget(self.server_light)
        lights.addWidget(self.camera_light)
        lights.addWidget(self.grab_light)
        lights.addWidget(self.viewer_light)

        xvar_row = QHBoxLayout()
        xvar_row.addWidget(self.xvar_display)

        button_row = QHBoxLayout()
        button_row.addWidget(self.toggle_server_button)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.new_plot_button)
        self._button_row = button_row

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(lights)
        layout.addWidget(self.run_label)
        layout.addLayout(xvar_row)
        layout.addWidget(self.log_box, stretch=1)
        layout.addLayout(button_row)
        self.setLayout(layout)

        # ---- Connect server signals ----
        self._connect_server_signals()

        # ---- Periodic refreshes (every 1 s) ----
        self._viewer_timer = QTimer(self)
        self._viewer_timer.timeout.connect(self._update_viewer_count)
        self._viewer_timer.timeout.connect(self._update_run_id_label)
        self._viewer_timer.setSingleShot(False)
        self._viewer_timer.start(1000)

        self._server_thread = None
        self._img_count = 0
        self._N_img = 0
        self._active_run = False   # True while a run is in progress
        self._plot_windows: list[ShotPlotWindow] = []
        self._plot_counter = 0
        self._xvar_names: list[str] = []
        self._data_field_names: list[str] = []
        self._shot_index = 0
        self._shot_history: list[dict] = []

        self._update_run_id_label()
        self._toggle_server()
        QTimer.singleShot(0, self._apply_initial_window_size)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _connect_server_signals(self):
        self.server.log_signal.connect(self._on_log)
        self.server.camera_status_signal.connect(self._on_camera_status)
        self.server.run_started_signal.connect(self._on_run_started)
        self.server.run_completed_signal.connect(self._on_run_completed)
        self.server.image_grabbed_signal.connect(self._on_image_grabbed)
        self.server.xvars_signal.connect(self._on_xvars)

    # ------------------------------------------------------------------
    #  Signal handlers
    # ------------------------------------------------------------------

    def _on_log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{timestamp}] {msg}")

    def _on_camera_status(self, status: int):
        if status == -1:
            self._active_run = False
            self.camera_light.set_state("off")
            self.camera_light.set_label("Camera: –")
            self.grab_light.set_state("off")
            self.grab_light.set_label("Grab loop: idle")
        elif status == 0:
            self.camera_light.set_state("waiting")
            self.camera_light.set_label("Camera: connecting…")
        elif status == 1:
            self.camera_light.set_state("ok")
            # label updated in _on_run_started
        elif status == 3:
            self.grab_light.set_state("active")
            self.grab_light.set_label("Grab loop: running")

    def _on_run_started(self, info: dict):
        camera_key = info.get("camera_key", "?")
        run_id = info.get("run_id", "?")
        N_img = info.get("N_img", 0)
        self._shot_index = 0
        self._shot_history = []
        self._data_field_names = list(info.get("available_data_fields", []))
        self._N_img = N_img
        self._img_count = 0
        self._active_run = True
        self.camera_light.set_label(f"Camera: {camera_key}")
        self.run_label.setText(
            f"Run ID: {run_id}   |   Expecting {N_img} images"
        )
        for w in self._plot_windows:
            w.on_run_started()
            w.update_data_field_names(self._data_field_names)

    def _on_run_completed(self):
        self._active_run = False
        self.grab_light.set_state("ok")
        self.grab_light.set_label("Grab loop: complete ✓")
        self.run_label.setText(
            self.run_label.text().replace("Expecting", "Done —")
        )

    def _on_image_grabbed(self, _img, idx: int):
        self._img_count = idx + 1
        self.grab_light.set_label(
            f"Grab loop: {self._img_count}/{self._N_img}"
        )

    def _update_viewer_count(self):
        try:
            with self.server._viewer_lock:
                n = len(self.server._viewer_connections)
        except Exception:
            n = 0
        state = "ok" if n > 0 else "off"
        self.viewer_light.set_state(state)
        self.viewer_light.set_label(f"Viewers: {n} connected")

    def _apply_initial_window_size(self):
        self.layout().activate()
        self._button_row.activate()
        min_width = max(
            self.run_label.sizeHint().width(),
            self._button_row.sizeHint().width(),
            self.xvar_display.sizeHint().width(),
        )
        margins = self.layout().contentsMargins()
        width = min_width + margins.left() + margins.right()
        self.setMinimumWidth(width)
        self.resize(width, self.height())

    def _update_run_id_label(self):
        """Read run_id from disk and update the label (unless mid-run)."""
        if self._active_run:
            return
        try:
            rid = self.server_talk.get_run_id()
            self.run_label.setText(f"Run ID: {rid}")
        except Exception:
            pass

    def _on_xvars(self, xvars: dict):
        """Update the xvar display when xvars are forwarded."""
        self.xvar_display.update_xvars(xvars)

        scalar_names = [k for k, v in xvars.items() if np.isscalar(v)]
        if scalar_names != self._xvar_names:
            self._xvar_names = scalar_names
            for w in self._plot_windows:
                w.update_xvar_names(self._xvar_names)

        try:
            names = list(self.server._available_data_fields)
        except Exception:
            names = []
        if names != self._data_field_names:
            self._data_field_names = names
            for w in self._plot_windows:
                w.update_data_field_names(self._data_field_names)

        shot = {
            "shot_index": self._shot_index,
            "timestamp": time.time(),
            "xvars": dict(xvars),
        }
        try:
            current_data_fields = dict(getattr(self.server, "_last_data_fields", {}))
        except Exception:
            current_data_fields = {}
        if current_data_fields:
            shot["data_fields"] = current_data_fields
        self._shot_index += 1
        for key, value in xvars.items():
            shot[f"xvar.{key}"] = self._to_plot_scalar(value)
        for key, value in current_data_fields.items():
            shot[f"xvar.{key}"] = self._to_plot_scalar(value)

        self._shot_history.append(shot)

        if not self._plot_windows:
            return

        for w in self._plot_windows:
            w.on_new_shot(shot)

    def _open_new_plot(self):
        self._plot_counter += 1
        win = ShotPlotWindow(
            window_id=self._plot_counter,
            xvar_names=list(self._xvar_names),
            data_field_names=list(self._data_field_names),
        )
        win.closed.connect(self._on_plot_closed)
        self._plot_windows.append(win)
        if self._shot_history:
            for shot in self._shot_history:
                win.on_new_shot(shot)
        win.show()

    def _to_plot_scalar(self, value):
        try:
            if np.isscalar(value):
                return float(value)
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size == 0:
                return np.nan
            if arr.size == 1:
                return float(arr[0])
            return float(np.nanmean(arr))
        except Exception:
            return np.nan

    def _on_plot_closed(self, win: ShotPlotWindow):
        if win in self._plot_windows:
            self._plot_windows.remove(win)

    def _send_reset(self):
        """Send a reset command to the camera server from the server window."""
        self._on_log("Sending reset to camera server...")
        try:
            ViewerClient.send_reset(self._host, self._port)
        except Exception as e:
            self._on_log(f"Reset error: {e}")

    # ------------------------------------------------------------------
    #  Start / Stop
    # ------------------------------------------------------------------

    def _toggle_server(self):
        """Toggle server state: start if stopped, stop if running."""
        if self._server_thread is not None and self._server_thread.is_alive():
            # Server is running -> stop it
            self.server.stop()
            self.server_light.set_state("error")
            self.toggle_server_button.setText("Start Server")
            self.toggle_server_button.setStyleSheet(
                "QPushButton {"
                "  background: #1e4f39; color: #e9fff2;"
                "  border: 1px solid #4b876a; border-radius: 8px;"
                "  font-size: 14px; font-weight: 700; padding: 8px 14px;"
                "}"
                "QPushButton:hover { background: #27684b; }"
            )
            self._on_log("Server stopped by user.")
        else:
            # Server is not running -> start it
            # Re-create server so sockets are fresh after a stop
            self.server = CameraServer(
                host=self._host,
                port=self._port,
                camera_nanny=self.camera_nanny,
            )
            self._connect_server_signals()

            self._server_thread = threading.Thread(target=self.server.run, daemon=True)
            self._server_thread.start()
            self.server_light.set_state("ok")
            self.toggle_server_button.setText("Stop Server")
            self.toggle_server_button.setStyleSheet(
                "QPushButton {"
                "  background: #7b2f36; color: #ffeef0;"
                "  border: 1px solid #a3575d; border-radius: 8px;"
                "  font-size: 14px; font-weight: 700; padding: 8px 14px;"
                "}"
                "QPushButton:hover { background: #934048; }"
            )
            self._on_log("Server started.")
        
        # Ensure timer is always running
        if not self._viewer_timer.isActive():
            self._viewer_timer.start(1000)

    def closeEvent(self, event):
        for w in list(self._plot_windows):
            w.close()
        self._plot_windows.clear()
        self.server.stop()
        super().closeEvent(event)
