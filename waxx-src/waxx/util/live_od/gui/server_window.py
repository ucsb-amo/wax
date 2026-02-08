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

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPlainTextEdit, QPushButton,
)
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtCore import Qt, QTimer

from waxx.util.live_od.camera_server import CameraServer
from waxx.util.live_od.camera_nanny import CameraNanny
from waxa.data.increment_run_id import RUN_ID_PATH


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

    def __init__(self, host: str, port: int, camera_nanny=None):
        super().__init__()
        self._host = host
        self._port = port
        self.setWindowTitle("Camera Server")
        self.resize(700, 500)

        # ---- Server instance ----
        self.camera_nanny = camera_nanny or CameraNanny()
        print(self.camera_nanny)
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

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setMaximumBlockCount(2000)

        self.start_button = QPushButton("Start Server")
        self.start_button.setStyleSheet(
            "background-color: #ccffcc; font-size: 14px; font-weight: bold;"
        )
        self.start_button.clicked.connect(self._start_server)

        self.stop_button = QPushButton("Stop Server")
        self.stop_button.setStyleSheet(
            "background-color: #ffcccc; font-size: 14px; font-weight: bold;"
        )
        self.stop_button.clicked.connect(self._stop_server)
        self.stop_button.setEnabled(False)

        # ---- Layout ----
        lights = QHBoxLayout()
        lights.addWidget(self.server_light)
        lights.addWidget(self.camera_light)
        lights.addWidget(self.grab_light)
        lights.addWidget(self.viewer_light)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self.run_label)
        layout.addLayout(lights)
        layout.addWidget(self.log_box, stretch=1)
        layout.addLayout(button_row)
        self.setLayout(layout)

        # ---- Connect server signals ----
        self._connect_server_signals()

        # ---- Periodic refreshes (every 1 s) ----
        self._viewer_timer = QTimer(self)
        self._viewer_timer.timeout.connect(self._update_viewer_count)
        self._viewer_timer.timeout.connect(self._update_run_id_label)
        self._viewer_timer.start(1000)

        self._server_thread = None
        self._img_count = 0
        self._N_img = 0
        self._active_run = False   # True while a run is in progress

        self._update_run_id_label()
        self._start_server()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _connect_server_signals(self):
        self.server.log_signal.connect(self._on_log)
        self.server.camera_status_signal.connect(self._on_camera_status)
        self.server.run_started_signal.connect(self._on_run_started)
        self.server.run_completed_signal.connect(self._on_run_completed)
        self.server.image_grabbed_signal.connect(self._on_image_grabbed)

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
        self._N_img = N_img
        self._img_count = 0
        self._active_run = True
        self.camera_light.set_label(f"Camera: {camera_key}")
        self.run_label.setText(
            f"Run ID: {run_id}   |   Expecting {N_img} images"
        )

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

    def _update_run_id_label(self):
        """Read run_id from disk and update the label (unless mid-run)."""
        if self._active_run:
            return
        try:
            with open(RUN_ID_PATH, 'r') as f:
                rid = f.read().strip()
            self.run_label.setText(f"Run ID: {rid}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Start / Stop
    # ------------------------------------------------------------------

    def _start_server(self):
        if self._server_thread is not None and self._server_thread.is_alive():
            return
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
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._on_log("Server started.")

    def _stop_server(self):
        self.server.stop()
        self.server_light.set_state("error")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._on_log("Server stopped by user.")

    def closeEvent(self, event):
        self.server.stop()
        super().closeEvent(event)
