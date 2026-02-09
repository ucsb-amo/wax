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
import numpy as np
from queue import Queue
from types import SimpleNamespace

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QGridLayout,
)
from PyQt6.QtGui import QFont, QGuiApplication, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from waxx.util.live_od.viewer_client import ViewerClient
from waxa.image_processing import compute_OD, process_ODs
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
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._light)
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


class XVarDisplay(QFrame):
    """Prominent styled panel showing the current xvar names and values."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "XVarDisplay {"
            "  background-color: #1e1e2e;"
            "  border: 2px solid #555;"
            "  border-radius: 8px;"
            "}"
        )

        self._header = QLabel("Scan Variables")
        hfont = QFont()
        hfont.setPointSize(10)
        hfont.setBold(True)
        self._header.setFont(hfont)
        self._header.setStyleSheet("color: #aaa; border: none;")
        self._header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._grid = QHBoxLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(16)
        self._grid_widget = QWidget()
        self._grid_widget.setLayout(self._grid)

        self._placeholder = QLabel("–")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pf = QFont(); pf.setPointSize(13)
        self._placeholder.setFont(pf)
        self._placeholder.setStyleSheet("color: #777; border: none;")

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 6, 14, 8)
        layout.addWidget(self._header)
        layout.addWidget(self._placeholder)
        layout.addWidget(self._grid_widget)
        self._grid_widget.hide()
        self.setLayout(layout)

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

        name_font = QFont(); name_font.setPointSize(11); name_font.setBold(True)
        val_font = QFont(); val_font.setPointSize(14); val_font.setBold(True)

        first = True
        for k, v in xvars.items():
            if not first:
                sep = QLabel("  │  ")
                sep.setStyleSheet("color: #555; border: none;")
                self._grid.addWidget(sep)
            first = False
            
            name_lbl = QLabel(f"{k}")
            name_lbl.setFont(name_font)
            name_lbl.setStyleSheet("color: #8888cc; border: none;")

            eq_lbl = QLabel("=")
            eq_lbl.setStyleSheet("color: #999; border: none;")

            val_lbl = QLabel(f"{v}")
            val_lbl.setFont(val_font)
            val_lbl.setStyleSheet("color: #e0e0e0; border: none;")

            self._grid.addWidget(name_lbl)
            self._grid.addWidget(eq_lbl)
            self._grid.addWidget(val_lbl)
        
        self._grid.addStretch()


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
        self.conn_indicator = ConnectionIndicator(
            f"Camera server @ {server_ip}:{server_port}"
        )
        self.xvar_display = XVarDisplay()
        self.screenshot_button = QPushButton("📷 Screenshot 📷")
        self.screenshot_button.setStyleSheet(
            "background-color: #3464eb; font-size: 16px; color: #f2f2f2; font-weight: bold;"
        )
        self.screenshot_button.clicked.connect(self._copy_screenshot)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.setStyleSheet(
            "background-color: #ffcccc; font-size: 20px; font-weight: bold;"
        )
        self.reset_button.clicked.connect(self._send_reset)

        self.new_plot_button = QPushButton("📊 New Plot")
        self.new_plot_button.setMinimumHeight(40)
        self.new_plot_button.setStyleSheet(
            "background-color: #2e7d32; font-size: 16px; color: #f2f2f2; font-weight: bold;"
            " border-radius: 6px;"
        )
        self.new_plot_button.clicked.connect(self._open_new_plot)

        self.run_id_label = QLabel("Run ID: –")
        self.run_id_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(); font.setPointSize(12); font.setBold(True)
        self.run_id_label.setFont(font)

        # ---- pop-out plot windows ----
        self._plot_windows: list[ShotPlotWindow] = []
        self._plot_counter = 0
        self._xvar_names: list[str] = []

        # ---- state ----
        self._img_count = 0
        self._N_img = 0
        self._N_pwa_per_shot = 0

        # ---- layout ----
        self._setup_layout()

        # ---- connect signals ----
        self.viewer_client.connection_status.connect(self.conn_indicator.set_connected)
        self.viewer_client.run_started.connect(self._on_run_started)
        self.viewer_client.image_received.connect(self._on_image_received)
        self.viewer_client.xvars_received.connect(self._on_xvars_received)
        self.viewer_client.run_completed.connect(self._on_run_completed)
        self.viewer_client.reset_received.connect(self._on_reset_received)

        # ---- start background threads ----
        self.plotter.start()
        self.viewer_client.start()

    # ------------------------------------------------------------------
    #  Layout
    # ------------------------------------------------------------------

    def _setup_layout(self):
        # ---- top row: left info | centre xvars | right buttons ----
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        # Left group: run ID + connection
        left_group = QVBoxLayout()
        left_group.setSpacing(2)
        left_group.addWidget(self.run_id_label)
        left_group.addWidget(self.conn_indicator)
        top_bar.addLayout(left_group)

        top_bar.addStretch()

        # Centre: xvar display
        top_bar.addWidget(self.xvar_display)

        top_bar.addStretch()

        # Right group: action buttons
        btn_group = QHBoxLayout()
        btn_group.setSpacing(8)
        btn_group.addWidget(self.new_plot_button)
        btn_group.addWidget(self.screenshot_button)
        btn_group.addWidget(self.reset_button)
        top_bar.addLayout(btn_group)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.viewer_window, stretch=1)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    #  Slots
    # ------------------------------------------------------------------

    def _on_run_started(self, info: dict):
        N_img = info["N_img"]
        N_shots = info["N_shots"]
        N_pwa_per_shot = info["N_pwa_per_shot"]
        camera_key = info.get("camera_key", "")
        imaging_type = info.get("imaging_type", False)
        run_id = info.get("run_id", 0)

        self._N_img = N_img
        self._N_pwa_per_shot = N_pwa_per_shot
        self._img_count = 0

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
        self.run_id_label.setText(f"Run ID: {run_id}")

        # Set a sensible default ROI for the camera
        self._set_default_roi(camera_key)

        # Notify pop-out plot windows
        for w in self._plot_windows:
            w.on_run_started()

        self.viewer_window.output_window.appendPlainText(
            f"Run {run_id} started — camera: {camera_key}, "
            f"expecting {N_img} images."
        )

    def _on_image_received(self, image: np.ndarray, index: int):
        self._img_count += 1
        self.analyzer.got_img(image)
        self.viewer_window.update_image_count(self._img_count, self._N_img)

    def _on_run_completed(self):
        self.viewer_window.output_window.appendPlainText("Run complete.")

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
        self.xvar_display.update_xvars(xvars)
        self.analyzer.set_xvars(xvars)

        new_names = list(xvars.keys())
        if new_names != self._xvar_names:
            self._xvar_names = new_names
            for w in self._plot_windows:
                w.update_xvar_names(new_names)

    # ------------------------------------------------------------------
    #  Pop-out plot windows
    # ------------------------------------------------------------------

    def _open_new_plot(self):
        """Create a new pop-out ShotPlotWindow."""
        self._plot_counter += 1
        win = ShotPlotWindow(
            window_id=self._plot_counter,
            xvar_names=list(self._xvar_names),
        )
        win.closed.connect(self._on_plot_closed)
        self.analyzer.shot_result.connect(win.on_new_shot)
        self._plot_windows.append(win)
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

    def _copy_screenshot(self):
        pixmap = self.grab()
        clipboard = QGuiApplication.clipboard()
        clipboard.setPixmap(pixmap)
        self.viewer_window.output_window.appendPlainText(
            "Screenshot copied to clipboard."
        )

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
