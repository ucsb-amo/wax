"""
Camera-server GUI window.

Provides ``StatusLight`` and ``CameraServerWindow`` — a standalone PyQt6
window that hosts a :class:`CameraServer`, displays live log output, and
shows status indicators for the server, camera, grab loop and connected
viewers.

The window is fully self-contained: pass *host*, *port*, and a
*camera_nanny* and it will create / manage the ``CameraServer`` internally.
"""

import threading
import time
import os
import subprocess

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPlainTextEdit, QPushButton, QDialog, QSizePolicy,
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QTimer

from waxx.util.live_od.camera_server import CameraServer
from waxx.util.live_od.camera_nanny import CameraNanny
from waxx.util.live_od.viewer_client import ViewerClient
from waxx.util.live_od.gui.viewer_window import XVarDisplay
from waxx.util.live_od.gui.log_panel import FilteredLogPanel, classify_log_level
from waxa.data.server_talk import server_talk as st


# ======================================================================
#  Status indicator widget
# ======================================================================

class StatusLight(QWidget):
    """Coloured dot + label."""

    def __init__(self, label_text):
        super().__init__()
        self._dot = QFrame()
        self._dot.setFixedSize(12, 12)
        self._label = QLabel(label_text)
        self._label.setFont(QFont("Segoe UI", 9))
        self._label.setStyleSheet("color: #e3eef9; font-weight: 600;")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
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
            f"background-color: {colour}; border-radius: 6px; "
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
        for indicator in (self.server_light, self.camera_light, self.grab_light, self.viewer_light):
            indicator.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.run_label = QLabel("Run ID: –")
        self.run_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.run_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.run_label.setStyleSheet("color: #eaf3ff;")
        self.run_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.xvar_display = XVarDisplay()
        self.xvar_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.log_box = FilteredLogPanel(
            title="Server Log",
            show_timestamps=True,
            max_entries=600,
            font_family="Consolas",
            font_size=9,
        )
        self.log_box.setMinimumHeight(125)
        self.log_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log_box.set_style_sheet(
            "QPlainTextEdit {"
            "  background: #0f1823;"
            "  color: #dbe9f8;"
            "  border: 1px solid #30465d;"
            "  border-radius: 7px;"
            "}"
        )

        self.toggle_server_button = QPushButton("Start Server")
        self.toggle_server_button.setStyleSheet(
            "QPushButton {"
            "  background: #1e4f39; color: #e9fff2;"
            "  border: 1px solid #4b876a; border-radius: 7px;"
            "  font-size: 12px; font-weight: 700; padding: 5px 10px;"
            "}"
            "QPushButton:hover { background: #27684b; }"
        )
        self.toggle_server_button.clicked.connect(self._toggle_server)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet(
            "QPushButton {"
            "  background: #6e2a2f; color: #ffeef0;"
            "  border: 1px solid #9a4a50; border-radius: 7px;"
            "  font-size: 12px; font-weight: 700; padding: 5px 10px;"
            "}"
            "QPushButton:hover { background: #84343a; }"
        )
        self.reset_button.clicked.connect(self._send_reset)

        self.log_button = QPushButton("Full Log")
        self.log_button.setStyleSheet(
            "QPushButton {"
            "  background: #3f5e1f; color: #f4ffe8;"
            "  border: 1px solid #7ea55a; border-radius: 7px;"
            "  font-size: 12px; font-weight: 700; padding: 5px 10px;"
            "}"
            "QPushButton:hover { background: #4f7527; }"
        )
        self.log_button.clicked.connect(self._open_full_log)

        self.open_viewer_button = QPushButton("Open Viewer")
        self.open_viewer_button.setStyleSheet(
            "QPushButton {"
            "  background: #203754; color: #eaf5ff;"
            "  border: 1px solid #4d78a8; border-radius: 7px;"
            "  font-size: 12px; font-weight: 700; padding: 5px 10px;"
            "}"
            "QPushButton:hover { background: #29466a; }"
        )
        self.open_viewer_button.clicked.connect(self._launch_viewer_window)

        for btn in (self.toggle_server_button, self.reset_button, self.log_button, self.open_viewer_button):
            btn.setFixedHeight(30)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.setStyleSheet(
            "QWidget { background: #0c131b; color: #e6f2ff; }"
            "QGroupBox, QFrame { color: #e6f2ff; }"
        )

        self.log_dialog = QDialog(self)
        self.log_dialog.setWindowTitle("Camera Server Log")
        self.log_dialog.setMinimumSize(620, 360)
        self.log_dialog.resize(940, 560)
        self.full_log_box = FilteredLogPanel(
            title="Server Log",
            show_timestamps=True,
            max_entries=50000,
            font_family="Consolas",
            font_size=9,
        )
        self.full_log_box.set_style_sheet(
            "QPlainTextEdit {"
            "  background: #0b1118;"
            "  color: #e6f2ff;"
            "  border: 1px solid #2d435a;"
            "  border-radius: 8px;"
            "}"
        )
        log_dialog_layout = QVBoxLayout()
        log_dialog_layout.setContentsMargins(8, 8, 8, 8)
        log_dialog_layout.addWidget(self.full_log_box)
        self.log_dialog.setLayout(log_dialog_layout)

        # ---- Layout ----
        left_lights = QVBoxLayout()
        left_lights.setContentsMargins(0, 0, 0, 0)
        left_lights.setSpacing(4)
        left_lights.addWidget(self.server_light)
        left_lights.addWidget(self.viewer_light)

        right_lights = QVBoxLayout()
        right_lights.setContentsMargins(0, 0, 0, 0)
        right_lights.setSpacing(4)
        right_lights.addWidget(self.camera_light)
        right_lights.addWidget(self.grab_light)

        lights = QHBoxLayout()
        lights.setContentsMargins(0, 0, 0, 0)
        lights.setSpacing(8)
        lights.addLayout(left_lights, stretch=1)
        lights.addLayout(right_lights, stretch=1)

        xvar_row = QHBoxLayout()
        xvar_row.setContentsMargins(0, 0, 0, 0)
        xvar_row.addWidget(self.xvar_display)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        button_row.addWidget(self.toggle_server_button)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.log_button)
        button_row.addWidget(self.open_viewer_button)
        self._button_row = button_row

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(lights, stretch=0)
        layout.addWidget(self.run_label, stretch=0)
        layout.addLayout(xvar_row, stretch=0)
        layout.addWidget(self.log_box, stretch=1)  # Log grows/shrinks on resize
        layout.addLayout(button_row, stretch=0)
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
        self.log_box.add_message(msg, level=classify_log_level(msg))

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

    def _open_full_log(self):
        """Fetch full timestamped server logs and show them in a larger window."""
        log_reply = ViewerClient.get_logs(
            self._host,
            self._port,
            since=0,
            limit=50000,
        )
        self.full_log_box.clear()
        if isinstance(log_reply, dict):
            entries = list(log_reply.get("entries", []))
            self.full_log_box.set_entries(entries)
        else:
            self.full_log_box.add_message("Could not fetch server logs.", level="normal")

        self.log_dialog.show()
        self.log_dialog.raise_()
        self.log_dialog.activateWindow()

    def _launch_viewer_window(self):
        """Launch a viewer window via the live_od batch file."""
        candidate_paths = []
        code_root = os.environ.get("code")
        if code_root:
            candidate_paths.append(
                os.path.join(code_root, "k-exp", "kexp", "_bat", "live_od.bat")
            )

        user_profile = os.environ.get("USERPROFILE", "")
        if user_profile:
            candidate_paths.append(
                os.path.join(user_profile, "code", "k-exp", "kexp", "_bat", "live_od.bat")
            )

        viewer_bat = next((p for p in candidate_paths if os.path.exists(p)), "")
        if not viewer_bat:
            self._on_log("Viewer launch error: could not find live_od.bat")
            return

        try:
            subprocess.Popen(
                ["cmd", "/c", viewer_bat],
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            self._on_log(f"Launched viewer via {viewer_bat}")
        except Exception as e:
            self._on_log(f"Viewer launch error: {e}")

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
                "  border: 1px solid #4b876a; border-radius: 7px;"
                "  font-size: 12px; font-weight: 700; padding: 5px 10px;"
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
                "  border: 1px solid #a3575d; border-radius: 7px;"
                "  font-size: 12px; font-weight: 700; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #934048; }"
            )
            self._on_log("Server started.")
        
        # Ensure timer is always running
        if not self._viewer_timer.isActive():
            self._viewer_timer.start(1000)

    def closeEvent(self, event):
        self.server.stop()
        super().closeEvent(event)
