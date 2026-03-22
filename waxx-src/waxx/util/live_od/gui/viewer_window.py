"""
Standalone LiveOD viewer window that runs as a client.

Connects to the camera server's viewer port, receives image broadcasts
and xvar updates, computes OD, and displays everything using the
``LiveODViewer`` widget.

Can be run from **any** computer on the network — only needs to know
the camera server IP/port (passed as constructor arguments).

Usage::

    from waxx.util.live_od.gui.viewer_window import LiveODClientWindow
    win = LiveODClientWindow(server_ip="192.168.1.76", server_port=7890)
"""

import sys
import time
import numpy as np
from queue import Queue
from types import SimpleNamespace

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSizePolicy, QStackedWidget, QGridLayout, QGroupBox,
)
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtCore import Qt, QTimer

from waxx.util.live_od.viewer_client import ViewerClient
from waxa import ROI

from waxx.util.live_od.gui.viewer import LiveODViewer
from waxx.util.live_od.gui.analyzer import Analyzer
from waxx.util.live_od.gui.plotter import LiveODPlotter
from waxx.util.live_od.gui.shot_plot_window import ShotPlotWindow


class ConnectionIndicator(QWidget):
    """Small coloured dot + label showing connection state."""

    def __init__(self, label_text="Server"):
        super().__init__()
        self._light = QFrame()
        self._light.setFixedSize(14, 14)
        self._set_color(False)
        self._label = QLabel(label_text)
        self._label.setStyleSheet("color: #e6f2ff; font-weight: 600;")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._light)
        if label_text:
            layout.addWidget(self._label)
        layout.addStretch()
        self.setLayout(layout)

    def _set_color(self, connected):
        color = "green" if connected else "gray"
        self._light.setStyleSheet(
            f"background-color: {color}; border-radius: 7px; border: 1px solid black;"
        )

    def set_connected(self, connected: bool):
        self._set_color(connected)


class XVarDisplay(QGroupBox):
    """Prominent styled panel showing the current xvar names and values."""

    def __init__(self):
        super().__init__("xvars")
        self.setStyleSheet(
            "QGroupBox {"
            "  background-color: #0f1722;"
            "  border: 1px solid #294056;"
            "  border-radius: 12px;"
            "  margin-top: 8px;"
            "  padding-top: 8px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 10px;"
            "  padding: 0 4px;"
            "  color: #9db6cc;"
            "  font-size: 8px;"
            "  font-weight: 700;"
            "}"
            "QFrame#xvarCard {"
            "  background: #162434;"
            "  border: 1px solid #31506b;"
            "  border-radius: 8px;"
            "}"
        )

        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(6)
        self._grid_widget = QWidget()
        self._grid_widget.setLayout(self._grid)

        self._placeholder = QLabel("–")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pf = QFont(); pf.setPointSize(10)
        self._placeholder.setFont(pf)
        self._placeholder.setStyleSheet("color: #6f8193; border: none;")

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(self._placeholder)
        layout.addWidget(self._grid_widget)
        self._grid_widget.hide()
        self.setLayout(layout)

    def _format_value(self, value):
        if isinstance(value, float):
            return f"{value:.6g}"
        text = str(value)
        if len(text) > 18:
            return text[:15] + "..."
        return text

    def update_xvars(self, xvars: dict):
        # clear old grid entries
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not xvars:
            self._placeholder.setText("–")
            self._placeholder.show()
            self._grid_widget.hide()
            return

        self._placeholder.hide()
        self._grid_widget.show()

        name_font = QFont(); name_font.setPointSize(8); name_font.setBold(True)
        val_font = QFont(); val_font.setPointSize(8); val_font.setBold(False)

        max_cols = 3
        for idx, (key, value) in enumerate(xvars.items()):
            card = QFrame()
            card.setObjectName("xvarCard")
            card_layout = QHBoxLayout()
            card_layout.setContentsMargins(8, 4, 8, 4)
            card_layout.setSpacing(6)

            name_lbl = QLabel(str(key))
            name_lbl.setFont(name_font)
            name_lbl.setStyleSheet("color: #8fb7d9; border: none;")

            val_lbl = QLabel(self._format_value(value))
            val_lbl.setFont(val_font)
            val_lbl.setStyleSheet("color: #f4f8ff; border: none;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            card_layout.addWidget(name_lbl)
            card_layout.addStretch()
            card_layout.addWidget(val_lbl)
            card.setLayout(card_layout)

            row = idx // max_cols
            col = idx % max_cols
            self._grid.addWidget(card, row, col)


class LiveODClientWindow(QWidget):
    """
    Standalone LiveOD viewer that connects to a remote camera server.

    Parameters
    ----------
    server_ip : str
        Camera server IP.
    server_port : int
        Camera server **command** port.  Viewer port = ``server_port + 1``.
    """

    def __init__(self, server_ip, server_port):
        super().__init__()
        self._server_ip = server_ip
        self._command_port = server_port
        self.viewer_port = server_port + 1

        # ---- viewer client (network) ----
        self.viewer_client = ViewerClient(server_ip, self.viewer_port)

        # ---- display widgets (reuse existing) ----
        self.viewer_window = LiveODViewer()
        self.plotting_queue = Queue()
        self.analyzer = Analyzer(self.plotting_queue, self.viewer_window)
        self.plotter = LiveODPlotter(self.viewer_window, self.plotting_queue)

        # ---- extra widgets ----
        self.xvar_display = XVarDisplay()
        self.xvar_display.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedHeight(28)
        self.reset_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.reset_button.setStyleSheet(
            "QPushButton {"
            "  background: #7f3038; color: #ffeef0;"
            "  border: 1px solid #ab5b61; border-radius: 8px;"
            "  font-size: 12px; font-weight: 700;"
            "}"
            "QPushButton:hover { background: #97424b; }"
        )
        self.reset_button.clicked.connect(self._send_reset)

        self.display_toggle_button = QPushButton("Show Plot")
        self.display_toggle_button.setFixedHeight(28)
        self.display_toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.display_toggle_button.setStyleSheet(
            "QPushButton {"
            "  background: #1d3550; color: #edf6ff;"
            "  border: 1px solid #4f7ca8; border-radius: 8px;"
            "  font-size: 12px; font-weight: 700;"
            "}"
            "QPushButton:hover { background: #26476a; }"
        )
        self.display_toggle_button.clicked.connect(self._toggle_main_display)

        self.viewer_window.new_plot_button.setText("New Plot 📈")
        self.viewer_window.new_plot_button.clicked.connect(self._open_new_plot)
        self.viewer_window.new_plot_button.setFixedHeight(28)
        self.viewer_window.new_plot_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.viewer_window.new_plot_button.setStyleSheet(
            "QPushButton { background: #203754; color: #eaf5ff; border: 1px solid #4d78a8; border-radius: 8px; font-weight: 700; font-size: 12px; }"
            "QPushButton:hover { background: #29466a; }"
        )
        self.viewer_window.log_button.setFixedHeight(28)
        self.viewer_window.log_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.viewer_window.log_button.setStyleSheet(
            "QPushButton { background: #2b3f55; color: #e9f4ff; border: 1px solid #49617a; border-radius: 8px; font-weight: 600; font-size: 12px; }"
            "QPushButton:hover { background: #35516e; }"
        )
        self.viewer_window.log_dialog.setMinimumSize(560, 320)
        self.viewer_window.log_dialog.resize(900, 520)
        self.viewer_window.log_dialog.setSizeGripEnabled(True)
        try:
            self.viewer_window.log_button.clicked.disconnect(self.viewer_window._show_log_dialog)
        except Exception:
            pass
        self.viewer_window.log_button.clicked.connect(self._show_server_log_dialog)
        self.viewer_window.counts_label.hide()

        self.main_plot_panel = ShotPlotWindow(
            window_id=0,
            xvar_names=[],
            data_field_names=[],
            camera_enabled=True,
            embedded=True,
        )
        self.analyzer.shot_result.connect(self.main_plot_panel.on_new_shot)

        self.main_display_stack = QStackedWidget()
        self.main_display_stack.addWidget(self.viewer_window)
        self.main_display_stack.addWidget(self.main_plot_panel)

        self.status_panel = QFrame()
        self.status_panel.setStyleSheet(
            "QFrame {"
            "  background: #102033; border: 1px solid #36506b; border-radius: 8px;"
            "}"
        )
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(8, 6, 8, 6)
        status_layout.setSpacing(1)

        self.status_run_value = QLabel("Run --")
        self.status_run_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        run_font = QFont(); run_font.setPointSize(8); run_font.setBold(True)
        self.status_run_value.setFont(run_font)
        self.status_run_value.setStyleSheet("color: #cde3f9;")

        self.status_shot_value = QLabel("Shots 0/0")
        self.status_shot_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        shot_font = QFont(); shot_font.setPointSize(8); shot_font.setBold(True)
        self.status_shot_value.setFont(shot_font)
        self.status_shot_value.setStyleSheet("color: #f5f9ff;")

        status_layout.addWidget(self.status_run_value)
        status_layout.addWidget(self.status_shot_value)
        self.status_panel.setLayout(status_layout)

        self.setStyleSheet(
            "QWidget { background: #0b1118; color: #e6f0fa; }"
            "QLabel { color: #e6f0fa; }"
        )

        # ---- pop-out plot windows ----
        self._plot_windows: list[ShotPlotWindow] = []
        self._plot_counter = 0
        self._xvar_names: list[str] = []
        self._data_field_names: list[str] = []
        self._shot_history: list[dict] = []

        # ---- state ----
        self._img_count = 0
        self._N_img = 0
        self._N_shots = 0
        self._N_pwa_per_shot = 0
        self._setup_camera = True
        self._main_display_mode = "images"
        self._current_run_id = None

        # ---- layout ----
        self._setup_layout()

        # ---- connect signals ----
        self.viewer_client.run_started.connect(self._on_run_started)
        self.viewer_client.image_received.connect(self._on_image_received)
        self.viewer_client.xvars_received.connect(self._on_xvars_received)
        self.viewer_client.available_data_fields_received.connect(
            self._on_available_data_fields_received
        )
        self.viewer_client.run_completed.connect(self._on_run_completed)
        self.viewer_client.reset_received.connect(self._on_reset_received)
        self.analyzer.shot_result.connect(self._cache_shot_result)

        self._run_id_timer = QTimer(self)
        self._run_id_timer.timeout.connect(self._update_run_id_from_server)
        self._run_id_timer.setSingleShot(False)
        self._run_id_timer.start(1000)

        # ---- start background threads ----
        self.plotter.start()
        self.viewer_client.start()
        QTimer.singleShot(0, self._apply_initial_window_size)

    # ------------------------------------------------------------------
    #  Layout
    # ------------------------------------------------------------------

    def _setup_layout(self):
        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(8)
        left_column.addWidget(self.status_panel)
        left_column.addWidget(self.xvar_display, stretch=1)

        stack_left = QVBoxLayout()
        stack_left.setContentsMargins(0, 0, 0, 0)
        stack_left.setSpacing(4)
        stack_left.addWidget(self.viewer_window.new_plot_button)
        stack_left.addWidget(self.display_toggle_button)

        stack_right = QVBoxLayout()
        stack_right.setContentsMargins(0, 0, 0, 0)
        stack_right.setSpacing(4)
        stack_right.addWidget(self.viewer_window.log_button)
        stack_right.addWidget(self.reset_button)

        button_columns = QHBoxLayout()
        button_columns.setContentsMargins(0, 0, 0, 0)
        button_columns.setSpacing(6)
        button_columns.addLayout(stack_left)
        button_columns.addLayout(stack_right)

        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.setSpacing(10)
        bar_row.addLayout(left_column, stretch=1)
        bar_row.addLayout(button_columns)

        top_bar = QFrame()
        top_bar.setObjectName("viewerTopBar")
        top_bar.setStyleSheet(
            "QFrame#viewerTopBar {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #111a24, stop:1 #152230);"
            "  border: 1px solid #294056; border-radius: 12px;"
            "}"
        )
        top_bar_layout = QVBoxLayout()
        top_bar_layout.setContentsMargins(10, 10, 10, 10)
        top_bar_layout.setSpacing(0)
        top_bar_layout.addLayout(bar_row)
        top_bar.setLayout(top_bar_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(top_bar)
        layout.addWidget(self.main_display_stack, stretch=1)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    #  Slots
    # ------------------------------------------------------------------

    def _on_run_started(self, info: dict):
        N_img = info["N_img"]
        N_shots = info["N_shots"]
        N_pwa_per_shot = info["N_pwa_per_shot"]
        camera_key = info.get("camera_key", "")
        self._setup_camera = bool(info.get("setup_camera", True))
        imaging_type = info.get("imaging_type", False)
        run_id = info.get("run_id", 0)
        self._data_field_names = list(info.get("available_data_fields", []))

        self._N_img = N_img
        self._N_shots = N_shots
        self._N_pwa_per_shot = N_pwa_per_shot
        self._img_count = 0
        self._xvar_names = []
        self._shot_history = []
        self.main_plot_panel.on_run_started()
        self.main_plot_panel.update_xvar_names([])
        self.main_plot_panel.update_data_field_names(self._data_field_names)
        self.main_plot_panel.set_camera_enabled(self._setup_camera)

        self.analyzer.get_img_number(N_img, N_shots, N_pwa_per_shot)
        self.analyzer.get_analysis_type(imaging_type)

        # Give the analyzer camera pixel info for Gaussian fitting
        px_size = info.get("pixel_size_m", 0.0)
        mag = info.get("magnification", 1.0)
        if px_size > 0:
            self.analyzer.camera_params = SimpleNamespace(
                pixel_size_m=px_size, magnification=mag
            )
        else:
            self.analyzer.camera_params = None

        self.viewer_window.get_img_number(N_img, N_shots, N_pwa_per_shot, run_id)
        self.viewer_window.clear_plots()
        self.viewer_window.update_image_count(0, N_img)
        self.viewer_window.update_shot_count(0, N_shots)
        self._current_run_id = run_id
        self._refresh_status_summary()
        self._set_main_display_mode("plot" if not self._setup_camera else "images")

        # Set a sensible default ROI for the camera
        self._set_default_roi(camera_key)

        # Notify pop-out plot windows
        for w in self._plot_windows:
            w.on_run_started()
            w.update_xvar_names([])
            w.update_data_field_names(self._data_field_names)
            w.set_camera_enabled(self._setup_camera)

        self.viewer_window.output_window.appendPlainText(
            f"Run {run_id} started — camera: {camera_key if self._setup_camera else 'disabled'}, "
            f"expecting {N_img} images."
        )

    def _on_image_received(self, image: np.ndarray, index: int):
        self._img_count += 1
        self.analyzer.got_img(image)
        self.viewer_window.update_image_count(self._img_count, self._N_img)
        denom = self._N_pwa_per_shot + 2
        if denom > 0:
            shot_count = self._img_count // denom
            self.viewer_window.update_shot_count(shot_count, self._N_shots)

    def _on_run_completed(self):
        self.viewer_window.output_window.appendPlainText("Run complete.")

    def _show_server_log_dialog(self):
        """Fetch full timestamped server logs and open a resizable log window."""
        log_reply = ViewerClient.get_logs(self._server_ip, self._command_port, since=0, limit=50000)
        if isinstance(log_reply, dict):
            entries = list(log_reply.get("entries", []))
            self.viewer_window.output_window.clear()
            for entry in entries:
                ts = float(entry.get("timestamp", 0.0))
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts > 0 else "--"
                msg = str(entry.get("message", ""))
                self.viewer_window.output_window.appendPlainText(f"[{stamp}] {msg}")
        else:
            self.viewer_window.output_window.appendPlainText("Could not fetch server logs.")

        self.viewer_window.log_dialog.show()
        self.viewer_window.log_dialog.raise_()
        self.viewer_window.log_dialog.activateWindow()

    def _send_reset(self):
        """Send a reset command to the camera server."""
        self.viewer_window.output_window.appendPlainText(
            "Sending reset to camera server..."
        )
        ViewerClient.send_reset(self._server_ip, self._command_port)

    def _on_reset_received(self):
        """Server confirmed the reset."""
        self.viewer_window.output_window.appendPlainText(
            "Server reset: grab stopped, data file deleted."
        )

    def _on_xvars_received(self, xvars: dict):
        """Update the xvar display, analyzer state, and pop-out windows."""
        data_field_set = set(self._data_field_names)
        scan_xvars = {k: v for k, v in xvars.items() if k not in data_field_set}

        self.xvar_display.update_xvars(scan_xvars)
        self.analyzer.set_xvars(xvars)

        # No-camera runs still need per-shot plot points (e.g. APD-derived fields).
        if not self._setup_camera:
            self.analyzer.emit_xvar_only_shot_result()

        new_names = [
            k for k, v in scan_xvars.items()
            if np.isscalar(v)
        ]
        if new_names != self._xvar_names:
            self._xvar_names = new_names
            self.main_plot_panel.update_xvar_names(new_names)
            for w in self._plot_windows:
                w.update_xvar_names(new_names)

    def _on_available_data_fields_received(self, field_names: list):
        names = [str(name) for name in field_names]
        if names == self._data_field_names:
            return
        self._data_field_names = names
        self.main_plot_panel.update_data_field_names(self._data_field_names)
        for w in self._plot_windows:
            w.update_data_field_names(self._data_field_names)

    def _cache_shot_result(self, shot: dict):
        self._shot_history.append(dict(shot))
        self.viewer_window.update_shot_count(len(self._shot_history), self._N_shots)
        self._refresh_status_summary()

    # ------------------------------------------------------------------
    #  Pop-out plot windows
    # ------------------------------------------------------------------

    def _open_new_plot(self):
        """Create a new pop-out ShotPlotWindow."""
        self._plot_counter += 1
        win = ShotPlotWindow(
            window_id=self._plot_counter,
            xvar_names=list(self._xvar_names),
            data_field_names=list(self._data_field_names),
            camera_enabled=self._setup_camera,
        )
        win.closed.connect(self._on_plot_closed)
        self.analyzer.shot_result.connect(win.on_new_shot)
        self._plot_windows.append(win)
        if self._shot_history:
            for shot in self._shot_history:
                win.on_new_shot(shot)
        win.show()

    def _on_plot_closed(self, win: ShotPlotWindow):
        """De-register a closed pop-out window."""
        try:
            self.analyzer.shot_result.disconnect(win.on_new_shot)
        except (TypeError, RuntimeError):
            pass
        if win in self._plot_windows:
            self._plot_windows.remove(win)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _set_default_roi(self, camera_key: str):
        if "andor" in camera_key:
            key = "andor_all"
        elif "basler" in camera_key:
            key = "basler_all"
        else:
            return
        try:
            self.analyzer.roi = ROI(roi_id=key, use_saved_roi=False, printouts=False)
        except Exception:
            pass

    def _refresh_status_summary(self):
        run_text = "--" if self._current_run_id is None else str(self._current_run_id)
        self.status_run_value.setText(f"Run {run_text}")
        self.status_shot_value.setText(f"Shots {len(self._shot_history)}/{self._N_shots}")

    def _apply_initial_window_size(self):
        min_width = max(self.layout().itemAt(0).widget().sizeHint().width(), self.viewer_window.minimumWidth())
        margins = self.layout().contentsMargins()
        width = min_width + margins.left() + margins.right()
        self.setMinimumWidth(width)
        self.resize(width, self.height())

    def _set_main_display_mode(self, mode: str):
        self._main_display_mode = mode
        if mode == "plot":
            self.main_display_stack.setCurrentWidget(self.main_plot_panel)
            self.display_toggle_button.setText("Show Images")
        else:
            self.main_display_stack.setCurrentWidget(self.viewer_window)
            self.display_toggle_button.setText("Show Plot")

    def _toggle_main_display(self):
        if self._main_display_mode == "images":
            self._set_main_display_mode("plot")
            return
        self._set_main_display_mode("images")

    def _update_run_id_from_server(self):
        status = ViewerClient.get_status(self._server_ip, self._command_port)
        if not status:
            return
        run_id = status.get("run_id", None)
        if run_id is None:
            return
        self._current_run_id = run_id
        self._refresh_status_summary()

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        # Close any open pop-out plot windows
        for w in list(self._plot_windows):
            w.close()
        self._plot_windows.clear()
        self.viewer_client.stop()
        super().closeEvent(event)
