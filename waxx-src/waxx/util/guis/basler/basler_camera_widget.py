"""BaslerCameraWidget — self-contained PyQt6 widget for one remote Basler camera.

Embeds a live image display and a pixel-counts-over-time plot.
All camera I/O goes through a ``BaslerCameraClient``; no direct pypylon calls.

State (ROI rectangle + normalisation reference) is persisted per camera serial
to ``~/.waxx/basler_<serial>_state.json``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from waxx.util.guis.basler.basler_camera_client import BaslerCameraClient

_STATE_DIR = os.path.join(os.path.expanduser("~"), ".waxx")

# ---------------------------------------------------------------------------
# Shared dark stylesheet — applied once at the main-window level so all
# BaslerCameraWidgets and CountsPanels inherit it automatically.
# ---------------------------------------------------------------------------
DARK_STYLESHEET = """
* {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11px;
    color: #d0d4e8;
}
QMainWindow, QWidget {
    background: #1a1b2e;
}
/* Camera-panel card */
QFrame#cameraCard {
    background: #22233a;
    border: 1px solid #3a3c5a;
    border-radius: 6px;
}
/* Header bar inside each card */
QWidget#camHeader {
    background: #2d2f4a;
    border-radius: 4px;
}
QLabel#camTitle {
    font-size: 13px;
    font-weight: 600;
    color: #c8cce8;
}
QLabel#camSub {
    font-size: 10px;
    color: #7880a8;
}
/* Toolbar strip */
QWidget#camToolbar {
    background: #1e1f34;
    border-top: 1px solid #30314e;
    border-bottom: 1px solid #30314e;
}
/* Buttons */
QPushButton {
    background: #2d2f4a;
    border: 1px solid #4a4d72;
    border-radius: 4px;
    padding: 2px 8px;
    color: #c0c4de;
}
QPushButton:hover { background: #383a60; border-color: #7880c8; }
QPushButton:pressed { background: #22243e; }
QPushButton:checked {
    background: #1e4d2a;
    border-color: #3fad5a;
    color: #7fdd9a;
}
QPushButton#openBtn:!checked {
    background: #1c3a20;
    border-color: #2e6638;
    color: #72b07e;
}
QPushButton#openBtn:checked {
    background: #3a1c1c;
    border-color: #7a3030;
    color: #c07878;
}
/* Spinboxes and checkboxes */
QDoubleSpinBox, QSpinBox {
    background: #2a2c44;
    border: 1px solid #4a4d72;
    border-radius: 3px;
    padding: 1px 3px;
    color: #c0c4de;
    selection-background-color: #4a5080;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {
    background: #35375a;
    border: none;
    width: 14px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #484b78; }
QCheckBox { spacing: 4px; color: #a0a4c0; }
QCheckBox::indicator {
    width: 12px; height: 12px;
    border: 1px solid #4a4d72;
    border-radius: 2px;
    background: #2a2c44;
}
QCheckBox::indicator:checked {
    background: #3a7f50;
    border-color: #4fbd74;
}
/* Toolbar separators */
QFrame#toolSep {
    background: #404368;
    max-width: 1px;
}
/* Image area */
QLabel#imageLabel {
    background: #0d0e1a;
    border: 1px solid #2a2c44;
    border-radius: 3px;
    color: #60638a;
    font-size: 12px;
}
/* Status bar */
QStatusBar { background: #14152a; color: #60638a; font-size: 10px; }
/* Scroll area */
QScrollArea { background: #1a1b2e; border: none; }
QScrollBar:vertical {
    background: #1a1b2e; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #3a3c5a; border-radius: 4px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal {
    background: #1a1b2e; height: 8px; border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #3a3c5a; border-radius: 4px; min-width: 20px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
/* Main window toolbar */
QToolBar {
    background: #14152a;
    border-bottom: 1px solid #2a2c44;
    spacing: 4px;
    padding: 2px 6px;
}
QToolBar QToolButton {
    background: #2d2f4a;
    border: 1px solid #4a4d72;
    border-radius: 4px;
    padding: 3px 10px;
    color: #c0c4de;
}
QToolBar QToolButton:hover { background: #383a60; border-color: #7880c8; }
QToolBar QToolButton:pressed { background: #22243e; }
"""

# LED colour constants
_LED_CLOSED  = "#5a5e7a"   # muted blue-grey
_LED_OPENING = "#c8a820"   # amber
_LED_OPEN    = "#3fad5a"   # green
_LED_ERROR   = "#bf4040"   # red


def _led_style(color: str) -> str:
    return (
        f"background: {color};"
        " border-radius: 5px;"
        f" border: 1px solid {color}88;"
    )


# ---------------------------------------------------------------------------
# Pixel-counts-over-time panel (ported from the original mot_viewer.py)
# ---------------------------------------------------------------------------

class CountsPanel(QWidget):
    """PyQtGraph plot of summed pixel counts inside the ROI vs time."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setContentsMargins(0, 0, 0, 0)
        self.plot_widget.setLabel("bottom", "seconds ago")
        self.plot_widget.setTitle("Summed Pixel Counts vs Time")
        layout.addWidget(self.plot_widget)

        self.plot_item = self.plot_widget.getPlotItem()
        self.main_curve = self.plot_item.plot(pen=pg.mkPen("g", width=2))

        # Second ViewBox for normalised right axis.
        self.vb2 = pg.ViewBox()
        self.plot_item.scene().addItem(self.vb2)
        self.plot_item.getAxis("right").linkToView(self.vb2)
        self.vb2.setXLink(self.plot_item)
        self.plot_item.vb.sigResized.connect(self._sync_vb2)
        self.plot_item.hideAxis("right")

        self.norm_ref_line = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen("g", style=Qt.PenStyle.DashLine, width=1)
        )
        self.plot_item.addItem(self.norm_ref_line)
        self.norm_ref_line.hide()

        self.norm_line = pg.InfiniteLine(
            pos=1, angle=0,
            pen=pg.mkPen("w", style=pg.QtCore.Qt.PenStyle.DashLine, width=1),
        )
        self.vb2.addItem(self.norm_line)
        self.norm_line.hide()

        self.timestamps: list[float] = []
        self.counts: list[float] = []
        self.start_time: Optional[datetime] = None

        self.fixed_interval: bool = True
        self.time_window: int = 30
        self.normalize: bool = True
        self.norm_reference: Optional[float] = None
        self.auto_rescale: bool = True
        self.show_norm_reference_line: bool = True

        # ``viewer`` is the containing BaslerCameraWidget; used to sync the
        # normalize checkbox state when the user clicks the plot.
        self.viewer: Optional[QWidget] = None

        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)

    # ------------------------------------------------------------------ #

    def _sync_vb2(self) -> None:
        self.vb2.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.vb2.linkedViewChanged(self.plot_item.vb, self.vb2.XAxis)

    def _on_plot_clicked(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.timestamps:
            pos = event.scenePos()
            pt = self.plot_item.vb.mapSceneToView(pos)
            self.norm_reference = pt.y()
            self.normalize = True
            if self.viewer is not None:
                self.viewer.norm_action.blockSignals(True)
                self.viewer.norm_action.setChecked(True)
                self.viewer.norm_action.blockSignals(False)
            self.update_plot()

    def clear_normalization(self) -> None:
        self.norm_reference = None
        self.normalize = False
        if self.viewer is not None:
            self.viewer.norm_action.blockSignals(True)
            self.viewer.norm_action.setChecked(False)
            self.viewer.norm_action.blockSignals(False)
        self.update_plot()

    def add_count(self, count: float) -> None:
        if self.start_time is None:
            self.start_time = datetime.now()
        self.timestamps.append((datetime.now() - self.start_time).total_seconds())
        self.counts.append(count)
        self.update_plot()

    def update_plot(self) -> None:
        if not self.timestamps:
            self.main_curve.setData([], [])
            return

        current = self.timestamps[-1]
        t_ago = [t - current for t in self.timestamps]

        if self.fixed_interval:
            start = next((i for i, t in enumerate(t_ago) if t >= -self.time_window), 0)
            pt = t_ago[start:]
            pc = self.counts[start:]
            self.plot_item.enableAutoRange(axis="x", enable=False)
            self.plot_item.setXRange(-self.time_window, 0, padding=0)
        else:
            pt = t_ago
            pc = self.counts
            self.plot_item.enableAutoRange(axis="x", enable=True)

        self.main_curve.setData(pt, pc)
        self.plot_item.enableAutoRange(axis="y", enable=self.auto_rescale)

        if self.normalize and pc:
            if self.norm_reference is None:
                self.norm_reference = float(np.mean(pc))
            ref = self.norm_reference
            if ref != 0:
                self.vb2.setYRange(min(pc) / ref, max(pc) / ref, padding=0.1)
            self.plot_item.showAxis("right")
            self.plot_item.getAxis("right").setLabel("Normalised")
            self.norm_ref_line.setValue(self.norm_reference)
            if self.show_norm_reference_line:
                self.norm_ref_line.show()
            else:
                self.norm_ref_line.hide()
        else:
            self.plot_item.hideAxis("right")
            self.norm_ref_line.hide()

    def clear_data(self) -> None:
        self.timestamps = []
        self.counts = []
        self.start_time = None
        self.update_plot()


# ---------------------------------------------------------------------------
# Per-camera widget
# ---------------------------------------------------------------------------

class BaslerCameraWidget(QFrame):
    """Live camera viewer widget backed by a remote ``BaslerCameraClient``.

    Displays a header bar (name / serial / host), a compact toolbar row, and
    a horizontal splitter containing the live image and the pixel-counts plot.
    All camera I/O goes through the provided ``BaslerCameraClient``.
    """

    def __init__(self, client: BaslerCameraClient, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraCard")
        self.client = client

        self._state_file = self._make_state_path()
        self._saved_norm_reference: Optional[float] = None
        self._saved_gain: Optional[float] = None
        self._saved_exposure: Optional[float] = None
        self._saved_trigger_mode: Optional[str] = None
        self._load_state()

        self.last_image: Optional[np.ndarray] = None
        self.show_rectangle: bool = True
        self.rect_start: Optional[QPoint] = None
        self.rect_end: Optional[QPoint] = None
        self.current_rect: Optional[QRect] = None
        self.saturation_in_box_only: bool = True
        self.max_pixel_value: int = 255

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._update_frame)

        self._build_ui()

    # ------------------------------------------------------------------ #
    # State persistence
    # ------------------------------------------------------------------ #

    def _make_state_path(self) -> str:
        os.makedirs(_STATE_DIR, exist_ok=True)
        return os.path.join(_STATE_DIR, f"basler_{self.client.serial}_state.json")

    def _load_state(self) -> None:
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file) as f:
                data = json.load(f)
            if "rect" in data:
                x1, y1, x2, y2 = data["rect"]
                self.current_rect = QRect(x1, y1, x2 - x1, y2 - y1)
            self._saved_norm_reference = data.get("norm_reference")
            self._saved_gain = data.get("gain")
            self._saved_exposure = data.get("exposure")
            self._saved_trigger_mode = data.get("trigger_mode")
        except Exception:
            pass

    def _save_state(self) -> None:
        data: dict = {}
        if self.current_rect and self.current_rect.width() > 0:
            x = self.current_rect.x()
            y = self.current_rect.y()
            data["rect"] = [x, y, x + self.current_rect.width(), y + self.current_rect.height()]
        if self.counts_panel.norm_reference is not None:
            data["norm_reference"] = self.counts_panel.norm_reference
        if self._saved_gain is not None:
            data["gain"] = self._saved_gain
        if self._saved_exposure is not None:
            data["exposure"] = self._saved_exposure
        if self._saved_trigger_mode is not None:
            data["trigger_mode"] = self._saved_trigger_mode
        try:
            with open(self._state_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(0)

        # ---- Header bar ------------------------------------------------
        header = QWidget()
        header.setObjectName("camHeader")
        hdr_layout = QHBoxLayout(header)
        hdr_layout.setContentsMargins(8, 6, 8, 6)
        hdr_layout.setSpacing(8)

        # Status LED (10x10 dot)
        self._status_dot = QLabel()
        self._status_dot.setFixedSize(10, 10)
        self._status_dot.setStyleSheet(_led_style(_LED_CLOSED))
        hdr_layout.addWidget(self._status_dot)

        # Camera name block
        name_block = QVBoxLayout()
        name_block.setSpacing(0)
        name_block.setContentsMargins(0, 0, 0, 0)

        display = self.client.user_id if self.client.user_id else self.client.serial
        self._title_label = QLabel(display)
        self._title_label.setObjectName("camTitle")
        name_block.addWidget(self._title_label)

        # Note: model / S/N / host info is shown in the surrounding
        # QDockWidget's title-bar (see ``BaslerCamerasMainWindow``), so we
        # deliberately omit the sub-label here to avoid duplication.

        hdr_layout.addLayout(name_block)
        hdr_layout.addStretch()

        self.open_btn = QPushButton("Open")
        self.open_btn.setObjectName("openBtn")
        self.open_btn.setCheckable(True)
        self.open_btn.setChecked(False)
        self.open_btn.setFixedWidth(60)
        self.open_btn.toggled.connect(self._on_open_toggled)
        hdr_layout.addWidget(self.open_btn)

        root.addWidget(header)

        # ---- Compact toolbar -------------------------------------------
        toolbar = QWidget()
        toolbar.setObjectName("camToolbar")
        tbar = QHBoxLayout(toolbar)
        tbar.setContentsMargins(6, 3, 6, 3)
        tbar.setSpacing(5)

        tbar.addWidget(_tlabel("Gain"))
        self.gain_spinbox = QDoubleSpinBox()
        self.gain_spinbox.setRange(0.0, 100.0)
        self.gain_spinbox.setDecimals(2)
        self.gain_spinbox.setEnabled(False)
        self.gain_spinbox.setFixedWidth(70)
        self.gain_spinbox.valueChanged.connect(self._on_gain_changed)
        tbar.addWidget(self.gain_spinbox)

        tbar.addWidget(_tsep())
        tbar.addWidget(_tlabel("Exp µs"))
        self.exposure_spinbox = QDoubleSpinBox()
        self.exposure_spinbox.setRange(0.0, 1_000_000.0)
        self.exposure_spinbox.setDecimals(1)
        self.exposure_spinbox.setEnabled(False)
        self.exposure_spinbox.setFixedWidth(82)
        self.exposure_spinbox.valueChanged.connect(self._on_exposure_changed)
        tbar.addWidget(self.exposure_spinbox)

        tbar.addWidget(_tsep())
        tbar.addWidget(_tlabel("Trig"))
        self.trigger_combo = QComboBox()
        self.trigger_combo.setEnabled(False)
        self.trigger_combo.setFixedWidth(72)
        self.trigger_combo.currentTextChanged.connect(self._on_trigger_mode_changed)
        tbar.addWidget(self.trigger_combo)

        tbar.addWidget(_tsep())

        # ---- Options dropdown (sat / norm / autoY / win / clear) ------
        self._opt_menu = QMenu(self)

        self.sat_action = QAction("Saturation warning", self)
        self.sat_action.setCheckable(True)
        self.sat_action.setChecked(True)
        self.sat_action.triggered.connect(lambda v: setattr(self, "saturation_in_box_only", v))
        self._opt_menu.addAction(self.sat_action)

        self._opt_menu.addSeparator()

        self.norm_action = QAction("Normalize counts", self)
        self.norm_action.setCheckable(True)
        self.norm_action.setChecked(True)
        self.norm_action.toggled.connect(self._on_normalize_toggled)
        self._opt_menu.addAction(self.norm_action)

        self.auto_rescale_action = QAction("Auto-scale Y", self)
        self.auto_rescale_action.setCheckable(True)
        self.auto_rescale_action.setChecked(True)
        self.auto_rescale_action.toggled.connect(self._on_auto_rescale_toggled)
        self._opt_menu.addAction(self.auto_rescale_action)

        self._opt_menu.addSeparator()

        _win_host = QWidget()
        _win_layout = QHBoxLayout(_win_host)
        _win_layout.setContentsMargins(8, 4, 8, 4)
        _win_layout.setSpacing(4)
        _win_layout.addWidget(_tlabel("Window"))
        self.time_window_spinbox = QSpinBox()
        self.time_window_spinbox.setRange(1, 300)
        self.time_window_spinbox.setValue(30)
        self.time_window_spinbox.setSuffix(" s")
        self.time_window_spinbox.setFixedWidth(60)
        self.time_window_spinbox.valueChanged.connect(self._on_time_window_changed)
        _win_layout.addWidget(self.time_window_spinbox)
        self.fixed_interval_btn = QPushButton("Fix")
        self.fixed_interval_btn.setCheckable(True)
        self.fixed_interval_btn.setChecked(True)
        self.fixed_interval_btn.setToolTip("Fixed time window")
        self.fixed_interval_btn.toggled.connect(self._on_fixed_interval_toggled)
        self.fixed_interval_btn.setFixedWidth(36)
        _win_layout.addWidget(self.fixed_interval_btn)
        _win_action = QWidgetAction(self)
        _win_action.setDefaultWidget(_win_host)
        self._opt_menu.addAction(_win_action)

        self._opt_menu.addSeparator()

        _clear_action = QAction("Clear history", self)
        _clear_action.triggered.connect(self._on_clear_counts)
        self._opt_menu.addAction(_clear_action)

        _opt_btn = QToolButton()
        _opt_btn.setText("Options")
        _opt_btn.setMenu(self._opt_menu)
        _opt_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tbar.addWidget(_opt_btn)

        # ---- Save Defaults --------------------------------------------
        # Snapshot the current ROI / gain / exposure / trigger to the
        # per-camera state file so the next time the camera is opened it
        # comes up with these values.
        self.save_defaults_btn = QPushButton("Save Defaults")
        self.save_defaults_btn.setToolTip(
            "Save the current ROI, gain, exposure, and trigger mode as the\n"
            "default values for this camera (used when re-opening it)."
        )
        self.save_defaults_btn.clicked.connect(self._on_save_defaults)
        tbar.addWidget(self.save_defaults_btn)

        tbar.addStretch()

        # ---- Counts panel toggle (far right) --------------------------
        self.counts_toggle_btn = QPushButton("Counts ▶")
        self.counts_toggle_btn.setCheckable(True)
        self.counts_toggle_btn.setChecked(False)
        self.counts_toggle_btn.setToolTip("Show / hide the pixel counts plot")
        self.counts_toggle_btn.toggled.connect(self._on_toggle_counts)
        tbar.addWidget(self.counts_toggle_btn)

        root.addWidget(toolbar)

        # ---- Main splitter: image | counts ------------------------------
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter = self._splitter
        splitter.setHandleWidth(3)

        self.image_label = QLabel()
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("camera closed")
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_label.mousePressEvent = self.on_image_mouse_press
        self.image_label.mouseMoveEvent = self.on_image_mouse_move
        self.image_label.mouseReleaseEvent = self.on_image_mouse_release
        splitter.addWidget(self.image_label)

        self.counts_panel = CountsPanel(self)
        self.counts_panel.viewer = self
        self.counts_panel.hide()   # collapsed by default
        if self._saved_norm_reference is not None:
            self.counts_panel.norm_reference = self._saved_norm_reference
        splitter.addWidget(self.counts_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------ #
    # Open / close
    # ------------------------------------------------------------------ #

    def open_camera(self) -> None:
        """Open the remote camera and start the frame-polling timer."""
        self.open_btn.blockSignals(True)
        self.open_btn.setChecked(True)
        self.open_btn.blockSignals(False)
        self._do_open()

    def close_camera(self) -> None:
        """Stop polling and close the remote camera."""
        self.open_btn.blockSignals(True)
        self.open_btn.setChecked(False)
        self.open_btn.blockSignals(False)
        self._do_close()

    def _on_open_toggled(self, checked: bool) -> None:
        if checked:
            self._do_open()
        else:
            self._do_close()

    def _do_open(self) -> None:
        self.open_btn.setText("…")
        self.open_btn.setEnabled(False)
        self._status_dot.setStyleSheet(_led_style(_LED_OPENING))
        try:
            resp = self.client.open()
            if not resp.get("ok", False):
                self._set_error(resp.get("error", "Failed to open"))
                self.open_btn.setChecked(False)
                self.open_btn.setText("Open")
                self.open_btn.setEnabled(True)
                self._status_dot.setStyleSheet(_led_style(_LED_ERROR))
                return
            self._refresh_settings()
            self.open_btn.setText("Close")
            self.open_btn.setEnabled(True)
            self.image_label.setText("")
            self._status_dot.setStyleSheet(_led_style(_LED_OPEN))
            self.poll_timer.start(30)
        except Exception as exc:
            self._set_error(str(exc))
            self.open_btn.setChecked(False)
            self.open_btn.setText("Open")
            self.open_btn.setEnabled(True)
            self._status_dot.setStyleSheet(_led_style(_LED_ERROR))

    def _do_close(self) -> None:
        self.poll_timer.stop()
        try:
            self.client.close()
        except Exception:
            pass
        self.gain_spinbox.setEnabled(False)
        self.exposure_spinbox.setEnabled(False)
        self.trigger_combo.setEnabled(False)
        self.open_btn.setText("Open")
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("camera closed")
        self._status_dot.setStyleSheet(_led_style(_LED_CLOSED))

    def _refresh_settings(self) -> None:
        """Query the server for gain/exposure/trigger and update widgets.

        If the per-camera state file has saved defaults for gain, exposure
        or trigger_mode, push those to the camera *before* reading back so
        each camera comes up with its remembered configuration.
        """
        try:
            gr = self.client.get_gain_range()
            er = self.client.get_exposure_range()
            # Push persisted defaults first (silently).
            if self._saved_gain is not None:
                try:
                    self.client.set_gain(float(self._saved_gain))
                except Exception:
                    pass
            if self._saved_exposure is not None:
                try:
                    self.client.set_exposure(float(self._saved_exposure))
                except Exception:
                    pass
            if self._saved_trigger_mode is not None:
                try:
                    self.client.set_trigger_mode(str(self._saved_trigger_mode))
                except Exception:
                    pass

            g = self.client.get_gain()
            e = self.client.get_exposure()
            tm = self.client.get_trigger_mode()
            tmo = self.client.get_trigger_mode_options()

            if gr.get("ok"):
                self.gain_spinbox.blockSignals(True)
                self.gain_spinbox.setRange(*gr["result"])
                self.gain_spinbox.blockSignals(False)
            if er.get("ok"):
                self.exposure_spinbox.blockSignals(True)
                self.exposure_spinbox.setRange(*er["result"])
                self.exposure_spinbox.blockSignals(False)
            if g.get("ok"):
                self.gain_spinbox.blockSignals(True)
                self.gain_spinbox.setValue(g["result"])
                self.gain_spinbox.blockSignals(False)
                self._saved_gain = float(g["result"])
            if e.get("ok"):
                self.exposure_spinbox.blockSignals(True)
                self.exposure_spinbox.setValue(e["result"])
                self.exposure_spinbox.blockSignals(False)
                self._saved_exposure = float(e["result"])
            # Populate trigger combo from the camera's allowed values.
            if tmo.get("ok"):
                opts = list(tmo["result"])
                self.trigger_combo.blockSignals(True)
                self.trigger_combo.clear()
                self.trigger_combo.addItems(opts)
                self.trigger_combo.blockSignals(False)
            if tm.get("ok"):
                self.trigger_combo.blockSignals(True)
                idx = self.trigger_combo.findText(str(tm["result"]))
                if idx >= 0:
                    self.trigger_combo.setCurrentIndex(idx)
                self.trigger_combo.blockSignals(False)
                self._saved_trigger_mode = str(tm["result"])

            self.gain_spinbox.setEnabled(True)
            self.exposure_spinbox.setEnabled(True)
            self.trigger_combo.setEnabled(True)
            # Persist any newly-discovered defaults.
            self._save_state()
        except Exception as exc:
            print(f"[BaslerWidget] Could not read camera settings: {exc}")

    def _set_error(self, msg: str) -> None:
        self.image_label.setText(f"Error: {msg}")

    # ------------------------------------------------------------------ #
    # Toolbar callbacks
    # ------------------------------------------------------------------ #

    def _on_gain_changed(self, value: float) -> None:
        try:
            self.client.set_gain(value)
            self._saved_gain = float(value)
            self._save_state()
        except Exception as exc:
            print(f"[BaslerWidget] set_gain error: {exc}")

    def _on_exposure_changed(self, value: float) -> None:
        try:
            self.client.set_exposure(value)
            self._saved_exposure = float(value)
            self._save_state()
        except Exception as exc:
            print(f"[BaslerWidget] set_exposure error: {exc}")

    def _on_trigger_mode_changed(self, value: str) -> None:
        if not value:
            return
        try:
            resp = self.client.set_trigger_mode(value)
            if resp.get("ok"):
                self._saved_trigger_mode = str(value)
                self._save_state()
            else:
                print(f"[BaslerWidget] set_trigger_mode failed: {resp.get('error')}")
        except Exception as exc:
            print(f"[BaslerWidget] set_trigger_mode error: {exc}")

    def _on_normalize_toggled(self, checked: bool) -> None:
        if checked:
            self.counts_panel.normalize = True
            if self.counts_panel.counts:
                self.counts_panel.norm_reference = float(np.mean(self.counts_panel.counts))
        else:
            self.counts_panel.norm_reference = None
            self.counts_panel.normalize = False
        self.counts_panel.update_plot()

    # keep a legacy alias so any code that used norm_btn still works
    @property
    def norm_btn(self):
        return self.norm_action

    def _on_toggle_counts(self, checked: bool) -> None:
        self.counts_panel.setVisible(checked)
        self.counts_toggle_btn.setText("Counts ▼" if checked else "Counts ▶")
        if checked:
            total = self._splitter.width()
            third = max(total // 3, 200)
            self._splitter.setSizes([total - third, third])
            # ROI is required for pixel counts to be meaningful; warn the
            # user if one hasn't been drawn yet.
            self._check_roi_for_counts()

    def _has_valid_roi(self) -> bool:
        r = self.current_rect
        return bool(r and r.width() > 0 and r.height() > 0)

    def _check_roi_for_counts(self) -> None:
        """When the counts panel is opened, make sure an ROI exists.

        If no ROI rectangle is set we set the plot title to an explicit
        instruction so it's obvious why no counts are appearing.  As soon
        as the user draws an ROI the title is restored.
        """
        if self._has_valid_roi():
            self.counts_panel.plot_widget.setTitle("Summed Pixel Counts vs Time")
        else:
            self.counts_panel.plot_widget.setTitle(
                "No ROI set \u2014 drag on the image to define one"
            )

    def _on_save_defaults(self) -> None:
        """Snapshot current settings to disk as this camera's defaults."""
        # Pull live values out of the spinboxes/combo so we save exactly
        # what the user sees, even if the camera is currently closed.
        if self.gain_spinbox.isEnabled():
            self._saved_gain = float(self.gain_spinbox.value())
        if self.exposure_spinbox.isEnabled():
            self._saved_exposure = float(self.exposure_spinbox.value())
        if self.trigger_combo.isEnabled() and self.trigger_combo.currentText():
            self._saved_trigger_mode = self.trigger_combo.currentText()
        self._save_state()

        # Give the user feedback with a brief button-text flash.
        original = self.save_defaults_btn.text()
        self.save_defaults_btn.setText("Saved \u2713")
        self.save_defaults_btn.setEnabled(False)
        QTimer.singleShot(
            1200,
            lambda: (
                self.save_defaults_btn.setText(original),
                self.save_defaults_btn.setEnabled(True),
            ),
        )

    def _on_auto_rescale_toggled(self, checked: bool) -> None:
        self.counts_panel.auto_rescale = checked
        self.counts_panel.update_plot()

    def _on_time_window_changed(self, value: int) -> None:
        self.counts_panel.time_window = value
        self.counts_panel.update_plot()

    def _on_fixed_interval_toggled(self, checked: bool) -> None:
        self.fixed_interval_btn.setText("Fixed" if checked else "All")
        self.counts_panel.fixed_interval = checked
        self.counts_panel.update_plot()

    def _on_clear_counts(self) -> None:
        self.counts_panel.clear_data()

    # ------------------------------------------------------------------ #
    # Frame polling
    # ------------------------------------------------------------------ #

    def _update_frame(self) -> None:
        try:
            data = self.client.get_frame()
        except Exception as exc:
            print(f"[BaslerWidget] get_frame error: {exc}")
            return
        if not data.get("ok"):
            return

        frame: np.ndarray = data["frame"]
        self.last_image = frame
        self.max_pixel_value = data.get("max_pixel_value", 255)

        # Convert to RGB.
        if frame.ndim == 2:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ROI pixel counts.
        if self.current_rect and self.current_rect.width() > 0 and self.current_rect.height() > 0:
            x1 = max(0, self.current_rect.x())
            y1 = max(0, self.current_rect.y())
            x2 = min(frame.shape[1], self.current_rect.x() + self.current_rect.width())
            y2 = min(frame.shape[0], self.current_rect.y() + self.current_rect.height())
            if x2 > x1 and y2 > y1:
                self.counts_panel.add_count(float(np.sum(frame[y1:y2, x1:x2])))

        h, w, ch = image_rgb.shape
        qt_image = QImage(image_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

        # ROI rectangle overlay.
        if self.show_rectangle:
            qt_image = qt_image.copy()
            p = QPainter(qt_image)
            p.setPen(QPen(QColor(0, 255, 0), 3))
            if self.rect_start and self.rect_end:
                rx1 = min(self.rect_start.x(), self.rect_end.x())
                ry1 = min(self.rect_start.y(), self.rect_end.y())
                rx2 = max(self.rect_start.x(), self.rect_end.x())
                ry2 = max(self.rect_start.y(), self.rect_end.y())
                p.drawRect(rx1, ry1, rx2 - rx1, ry2 - ry1)
            if self.current_rect and self.current_rect.width() > 0:
                p.drawRect(self.current_rect)
            p.end()

        # Saturation check.
        if (
            self.saturation_in_box_only
            and self.current_rect
            and self.current_rect.width() > 0
        ):
            sx1 = max(0, self.current_rect.x())
            sy1 = max(0, self.current_rect.y())
            sx2 = min(frame.shape[1], self.current_rect.x() + self.current_rect.width())
            sy2 = min(frame.shape[0], self.current_rect.y() + self.current_rect.height())
            check = frame[sy1:sy2, sx1:sx2] if sx2 > sx1 and sy2 > sy1 else frame
        else:
            check = frame

        if np.any(check >= self.max_pixel_value):
            qt_image = qt_image.copy()
            p = QPainter(qt_image)
            p.setFont(QFont("Times", 100))
            p.setPen(QPen(QColor(255, 0, 0), 10))
            p.drawText(10, 170, "⚠ SATURATION WARNING ⚠")
            p.end()

        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    # ------------------------------------------------------------------ #
    # ROI mouse interaction
    # ------------------------------------------------------------------ #

    def _label_to_image_coords(self, pos: QPoint) -> QPoint:
        if self.last_image is None:
            return pos
        pix = self.image_label.pixmap()
        if pix is None or pix.isNull():
            return pos
        ls = self.image_label.size()
        ps = pix.size()
        ox = (ls.width() - ps.width()) / 2
        oy = (ls.height() - ps.height()) / 2
        px = max(0.0, min(float(pos.x()) - ox, float(ps.width())))
        py = max(0.0, min(float(pos.y()) - oy, float(ps.height())))
        sx = self.last_image.shape[1] / ps.width()
        sy = self.last_image.shape[0] / ps.height()
        return QPoint(int(px * sx), int(py * sy))

    def on_image_mouse_press(self, event) -> None:
        if self.show_rectangle:
            self.rect_start = self._label_to_image_coords(event.pos())
            self.rect_end = QPoint(self.rect_start)

    def on_image_mouse_move(self, event) -> None:
        if self.show_rectangle and self.rect_start is not None:
            self.rect_end = self._label_to_image_coords(event.pos())

    def on_image_mouse_release(self, event) -> None:
        if self.show_rectangle and self.rect_start is not None:
            self.rect_end = self._label_to_image_coords(event.pos())
            x1 = min(self.rect_start.x(), self.rect_end.x())
            y1 = min(self.rect_start.y(), self.rect_end.y())
            x2 = max(self.rect_start.x(), self.rect_end.x())
            y2 = max(self.rect_start.y(), self.rect_end.y())
            self.current_rect = QRect(x1, y1, x2 - x1, y2 - y1)
            self._save_state()
            # If the counts panel is open, refresh the title now that we
            # have a valid ROI (clears the "No ROI set..." instruction).
            if self.counts_panel.isVisible():
                self._check_roi_for_counts()

    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:
        self._do_close()
        self._save_state()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _tsep() -> QFrame:
    """Thin vertical separator for the compact toolbar."""
    sep = QFrame()
    sep.setObjectName("toolSep")
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    sep.setFixedWidth(1)
    return sep


def _tlabel(text: str) -> QLabel:
    """Muted label for toolbar prefixes."""
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #60638a; font-size: 10px;")
    return lbl
