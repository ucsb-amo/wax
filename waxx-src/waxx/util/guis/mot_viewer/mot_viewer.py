#!/usr/bin/env python3
"""
Lightweight Basler USB camera stream viewer with PyQt6 GUI
Connects to camera with serial number and displays live stream
"""

import sys
import cv2
import numpy as np
import math
import json
import os
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
                             QDialog, QHBoxLayout, QSlider, QSpinBox, QDoubleSpinBox, QFormLayout,
                             QMenuBar, QMenu, QSplitter, QWidgetAction)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PyQt6.QtCore import QTimer, Qt, QRect, QPoint
from pypylon import pylon
import pyqtgraph as pg


class CameraSettingsDialog(QDialog):
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.setWindowTitle("Camera Settings")
        
        # Position to the right of parent window
        if parent:
            parent_geometry = parent.geometry()
            x = parent_geometry.right() + 10
            y = parent_geometry.top()
            self.setGeometry(x, y, 320, 140)
        else:
            self.setGeometry(200, 200, 320, 140)
        
        layout = QFormLayout()
        
        # Get camera limits
        try:
            gain_min = self.camera.Gain.GetMin()
            gain_max = self.camera.Gain.GetMax()
        except:
            gain_min, gain_max = 0, 100
        
        try:
            exposure_min = self.camera.ExposureTime.GetMin()
            exposure_max = self.camera.ExposureTime.GetMax()
        except:
            exposure_min, exposure_max = 0, 1000000
        
        self.exposure_min = exposure_min
        self.exposure_max = exposure_max
        
        # Gain setting with slider and spinbox
        gain_layout = QHBoxLayout()
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(int(gain_min * 100), int(gain_max * 100))
        self.gain_spinbox = QDoubleSpinBox()
        self.gain_spinbox.setRange(gain_min, gain_max)
        self.gain_spinbox.setDecimals(2)
        try:
            current_gain = self.camera.Gain.GetValue()
            self.gain_spinbox.setValue(current_gain)
            self.gain_slider.setValue(int(current_gain * 100))
        except:
            self.gain_spinbox.setValue(gain_min)
            self.gain_slider.setValue(int(gain_min * 100))
        
        self.gain_slider.valueChanged.connect(self.on_gain_slider_changed)
        self.gain_spinbox.valueChanged.connect(self.on_gain_spinbox_changed)
        
        gain_layout.addWidget(self.gain_slider)
        gain_layout.addWidget(self.gain_spinbox)
        layout.addRow("Gain (dB):", gain_layout)
        
        # Exposure time setting with slider and spinbox
        exposure_layout = QHBoxLayout()
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(0, 1000)  # Linear slider for logarithmic mapping
        self.exposure_spinbox = QDoubleSpinBox()
        self.exposure_spinbox.setRange(exposure_min, exposure_max)
        self.exposure_spinbox.setDecimals(2)
        self.exposure_spinbox.setSuffix(" µs")
        try:
            current_exposure = self.camera.ExposureTime.GetValue()
            self.exposure_spinbox.setValue(current_exposure)
            # Convert exposure time to logarithmic slider value
            log_slider_value = int(1000 * math.log(current_exposure / exposure_min) / math.log(exposure_max / exposure_min))
            self.exposure_slider.setValue(log_slider_value)
        except:
            self.exposure_spinbox.setValue(exposure_min)
            self.exposure_slider.setValue(0)
        
        self.exposure_slider.valueChanged.connect(self.on_exposure_slider_changed)
        self.exposure_spinbox.valueChanged.connect(self.on_exposure_spinbox_changed)
        
        exposure_layout.addWidget(self.exposure_slider)
        exposure_layout.addWidget(self.exposure_spinbox)
        layout.addRow("Exposure Time:", exposure_layout)

        self.setLayout(layout)
    
    def on_gain_slider_changed(self, value):
        self.gain_spinbox.blockSignals(True)
        self.gain_spinbox.setValue(value / 100.0)
        self.gain_spinbox.blockSignals(False)
        self.apply_gain()
    
    def on_gain_spinbox_changed(self, value):
        self.gain_slider.blockSignals(True)
        self.gain_slider.setValue(int(value * 100))
        self.gain_slider.blockSignals(False)
        self.apply_gain()
    
    def apply_gain(self):
        try:
            self.camera.Gain.SetValue(self.gain_spinbox.value())
        except Exception as e:
            print(f"Error setting gain: {e}")
    
    def on_exposure_slider_changed(self, value):
        self.exposure_spinbox.blockSignals(True)
        # Convert logarithmic slider value to exposure time
        if self.exposure_max > self.exposure_min:
            exposure_value = self.exposure_min * math.exp(value / 1000.0 * math.log(self.exposure_max / self.exposure_min))
            self.exposure_spinbox.setValue(exposure_value)
        self.exposure_spinbox.blockSignals(False)
        self.apply_exposure()
    
    def on_exposure_spinbox_changed(self, value):
        self.exposure_slider.blockSignals(True)
        # Convert exposure time to logarithmic slider value
        if self.exposure_max > self.exposure_min and value > self.exposure_min:
            log_slider_value = int(1000 * math.log(value / self.exposure_min) / math.log(self.exposure_max / self.exposure_min))
            self.exposure_slider.setValue(log_slider_value)
        self.exposure_slider.blockSignals(False)
        self.apply_exposure()
    
    def apply_exposure(self):
        try:
            self.camera.ExposureTime.SetValue(self.exposure_spinbox.value())
        except Exception as e:
            print(f"Error setting exposure time: {e}")



class CountsSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Counts Settings")
        self.parent_counts = parent
        self.setGeometry(400, 300, 350, 150)
        
        layout = QFormLayout()
        
        # Fixed interval checkbox
        self.fixed_interval_checkbox = QPushButton("Fixed Interval")
        self.fixed_interval_checkbox.setCheckable(True)
        self.fixed_interval_checkbox.setChecked(True)
        self.fixed_interval_checkbox.toggled.connect(self.on_fixed_interval_toggled)
        layout.addRow("Display Mode:", self.fixed_interval_checkbox)
        
        # Time window slider and spinbox
        time_window_layout = QHBoxLayout()
        self.time_window_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_window_slider.setRange(1, 300)  # 1 to 300 seconds
        self.time_window_slider.setValue(30)
        self.time_window_spinbox = QSpinBox()
        self.time_window_spinbox.setRange(1, 300)
        self.time_window_spinbox.setValue(30)
        self.time_window_spinbox.setSuffix(" s")
        
        self.time_window_slider.valueChanged.connect(self.on_slider_changed)
        self.time_window_spinbox.valueChanged.connect(self.on_spinbox_changed)
        
        time_window_layout.addWidget(self.time_window_slider)
        time_window_layout.addWidget(self.time_window_spinbox)
        layout.addRow("Time Window:", time_window_layout)
        
        self.setLayout(layout)
    
    def on_slider_changed(self, value):
        self.time_window_spinbox.blockSignals(True)
        self.time_window_spinbox.setValue(value)
        self.time_window_spinbox.blockSignals(False)
        self.apply_settings()
    
    def on_spinbox_changed(self, value):
        self.time_window_slider.blockSignals(True)
        self.time_window_slider.setValue(value)
        self.time_window_slider.blockSignals(False)
        self.apply_settings()
    
    def on_fixed_interval_toggled(self, checked):
        self.apply_settings()
    
    def apply_settings(self):
        if self.parent_counts:
            self.parent_counts.fixed_interval = self.fixed_interval_checkbox.isChecked()
            self.parent_counts.time_window = self.time_window_spinbox.value()


class CountsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # PyQtGraph plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setContentsMargins(0, 0, 0, 0)
        self.plot_widget.setLabel('bottom', 'seconds ago')
        self.plot_widget.setTitle("Summed Pixel Counts vs Time")
        layout.addWidget(self.plot_widget)

        # Persistent main curve (green)
        self.plot_item = self.plot_widget.getPlotItem()
        self.main_curve = self.plot_item.plot(pen=pg.mkPen('g', width=2))

        # Second ViewBox for normalized right y-axis
        self.vb2 = pg.ViewBox()
        self.plot_item.scene().addItem(self.vb2)
        self.plot_item.getAxis('right').linkToView(self.vb2)
        self.vb2.setXLink(self.plot_item)
        self.plot_item.vb.sigResized.connect(self._sync_vb2_geometry)
        self.plot_item.hideAxis('right')

        # Dotted reference line at the raw norm_reference value on the left axis
        self.norm_ref_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine, width=1))
        self.plot_item.addItem(self.norm_ref_line)
        self.norm_ref_line.hide()

        self.timestamps = []
        self.counts = []
        self.start_time = None
        self.fixed_interval = True
        self.time_window = 30
        self.normalize = True
        self.norm_reference = None
        self.auto_rescale = True
        self.show_norm_reference_line = True
        self.settings_dialog = None
        self.viewer = None

        # Add horizontal reference line at y=1 for normalized plot
        self.norm_line = pg.InfiniteLine(pos=1, angle=0, pen=pg.mkPen('w', style=pg.QtCore.Qt.PenStyle.DashLine, width=1))
        self.vb2.addItem(self.norm_line)
        self.norm_line.hide()

        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)
    
    def add_count(self, count):
        if self.start_time is None:
            self.start_time = datetime.now()
        
        self.timestamps.append((datetime.now() - self.start_time).total_seconds())
        self.counts.append(count)
        self.update_plot()
    
    def _on_plot_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.timestamps:
            pos = event.scenePos()
            mouse_point = self.plot_item.vb.mapSceneToView(pos)
            self.norm_reference = mouse_point.y()
            self.normalize = True
            if self.viewer:
                self.viewer.normalize_action.blockSignals(True)
                self.viewer.normalize_action.setChecked(True)
                self.viewer.normalize_action.blockSignals(False)
            self.update_plot()

    def clear_normalization(self):
        self.norm_reference = None
        self.normalize = False
        if self.viewer:
            self.viewer.normalize_action.blockSignals(True)
            self.viewer.normalize_action.setChecked(False)
            self.viewer.normalize_action.blockSignals(False)
        self.update_plot()

    def _sync_vb2_geometry(self):
        self.vb2.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.vb2.linkedViewChanged(self.plot_item.vb, self.vb2.XAxis)

    def update_plot(self):
        plot_item = self.plot_item

        if len(self.timestamps) == 0:
            self.main_curve.setData([], [])
            return

        # Transform timestamps to "seconds ago" (0 = now, negative = older)
        current_time = self.timestamps[-1]
        t_ago = [t - current_time for t in self.timestamps]

        if self.fixed_interval:
            start_idx = 0
            for i, t in enumerate(t_ago):
                if t >= -self.time_window:
                    start_idx = i
                    break
            plot_timestamps = t_ago[start_idx:]
            plot_counts = self.counts[start_idx:]
            plot_item.enableAutoRange(axis='x', enable=False)
            plot_item.setXRange(-self.time_window, 0, padding=0)
        else:
            plot_timestamps = t_ago
            plot_counts = self.counts
            plot_item.enableAutoRange(axis='x', enable=True)

        # Update main curve
        self.main_curve.setData(plot_timestamps, plot_counts)

        # Left Y auto-range
        plot_item.enableAutoRange(axis='y', enable=self.auto_rescale)

        # Normalized right axis
        if self.normalize and len(plot_counts) > 0:
            if self.norm_reference is None:
                self.norm_reference = float(np.mean(plot_counts))
            ref = self.norm_reference
            if ref != 0:
                norm_max = max(plot_counts) / ref
                norm_min = min(plot_counts) / ref
                if self.auto_rescale:
                    self.vb2.setYRange(norm_min, norm_max, padding=0.1)
            plot_item.showAxis('right')
            plot_item.getAxis('right').setLabel('Normalized')
            self.norm_ref_line.setValue(self.norm_reference)
            self.norm_ref_line.show()
        else:
            plot_item.hideAxis('right')
            self.norm_ref_line.hide()

    def show_settings(self):
        if self.settings_dialog is None:
            self.settings_dialog = CountsSettingsDialog(self)
            self.settings_dialog.show()
        else:
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()
    
    def clear_data(self):
        self.timestamps = []
        self.counts = []
        self.start_time = None
        self.update_plot()
    
class BaslerCameraViewer(QMainWindow):
    def __init__(self,basler_serial="40277706"):
        super().__init__()
        self.SERIAL_NUMBER = basler_serial
        self.camera = None
        self.is_streaming = False
        self.connecting = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Retry timer for reconnect attempts on failure
        self.retry_timer = QTimer(self)
        self.retry_timer.setInterval(2000)
        self.retry_timer.timeout.connect(self.connect_camera)
        
        # Rectangle for pixel counting
        self.rect_start = None
        self.rect_end = None
        self.current_rect = None
        self.show_rectangle = True
        self.rect_file = "camera_rect.json"
        self.last_image = None
        self.current_pixmap = None
        self.settings_dialog = None
        self.saturation_in_box_only = True
        self.saved_norm_reference = None
        self.load_rectangle()
        
        self.initUI()
        self.connect_camera()
    
    def initUI(self):
        self.setWindowTitle("Basler Stream")
        self.setGeometry(100, 100, 1280, 500)

        # Create menu bar
        menubar = self.menuBar()

        # Camera menu with inline gain / exposure spinboxes
        cam_menu = menubar.addMenu("Camera")

        def _make_row(menu, label_text, widget, label_width=90):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(8, 2, 8, 2)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(label_width)
            layout.addWidget(lbl)
            layout.addWidget(widget)
            wa = QWidgetAction(menu)
            wa.setDefaultWidget(row)
            menu.addAction(wa)

        self.gain_spinbox = QDoubleSpinBox()
        self.gain_spinbox.setRange(0, 100)
        self.gain_spinbox.setDecimals(2)
        self.gain_spinbox.setEnabled(False)
        self.gain_spinbox.valueChanged.connect(self._on_gain_changed)
        _make_row(cam_menu, "Gain (dB):", self.gain_spinbox)

        self.exposure_spinbox = QDoubleSpinBox()
        self.exposure_spinbox.setRange(0, 1000000)
        self.exposure_spinbox.setDecimals(2)
        self.exposure_spinbox.setSuffix(" µs")
        self.exposure_spinbox.setEnabled(False)
        self.exposure_spinbox.valueChanged.connect(self._on_exposure_changed)
        _make_row(cam_menu, "Exposure:", self.exposure_spinbox)

        cam_menu.addSeparator()
        self.sat_in_box_action = cam_menu.addAction("Sat. warning in box")
        self.sat_in_box_action.setCheckable(True)
        self.sat_in_box_action.setChecked(self.saturation_in_box_only)
        self.sat_in_box_action.toggled.connect(lambda checked: setattr(self, 'saturation_in_box_only', checked))

        counts_menu = menubar.addMenu("Counts")
        clear_action = counts_menu.addAction("Clear")
        clear_action.triggered.connect(lambda: self.counts_panel.clear_data())
        counts_menu.addSeparator()

        # Inline fixed-interval toggle
        self.fixed_interval_btn = QPushButton("On")
        self.fixed_interval_btn.setCheckable(True)
        self.fixed_interval_btn.setChecked(True)
        self.fixed_interval_btn.setFixedWidth(44)
        self.fixed_interval_btn.toggled.connect(self._on_fixed_interval_menu_toggled)
        _make_row(counts_menu, "Fixed interval:", self.fixed_interval_btn, label_width=100)

        # Inline time-window spinbox
        self.time_window_menu_spinbox = QSpinBox()
        self.time_window_menu_spinbox.setRange(1, 300)
        self.time_window_menu_spinbox.setValue(30)
        self.time_window_menu_spinbox.setSuffix(" s")
        self.time_window_menu_spinbox.valueChanged.connect(self._on_time_window_menu_changed)
        _make_row(counts_menu, "Time window:", self.time_window_menu_spinbox, label_width=100)

        counts_menu.addSeparator()
        self.normalize_action = counts_menu.addAction("Normalize")
        self.normalize_action.setCheckable(True)
        self.normalize_action.setChecked(True)
        self.normalize_action.toggled.connect(self._on_normalize_toggled)
        self.norm_ref_line_action = counts_menu.addAction("Reference line at y=1")
        self.norm_ref_line_action.setCheckable(True)
        self.norm_ref_line_action.setChecked(True)
        self.norm_ref_line_action.toggled.connect(self._on_norm_ref_line_toggled)
        clear_norm_action = counts_menu.addAction("Clear Normalization")
        clear_norm_action.triggered.connect(lambda: self.counts_panel.clear_normalization())
        self.auto_rescale_action = counts_menu.addAction("Auto-rescale Y")
        self.auto_rescale_action.setCheckable(True)
        self.auto_rescale_action.setChecked(True)
        self.auto_rescale_action.toggled.connect(self._on_auto_rescale_toggled)

        # Stream toggle action directly on menu bar
        self.stream_action = menubar.addAction("Stop Stream")
        self.stream_action.triggered.connect(self.on_toggle_stream)

        # Central widget - horizontal split: camera left, counts right
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_layout.addWidget(splitter)

        # Left side: camera image + toggle button
        camera_widget = QWidget()
        camera_layout = QVBoxLayout(camera_widget)
        camera_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.mousePressEvent = self.on_image_mouse_press
        self.image_label.mouseMoveEvent = self.on_image_mouse_move
        self.image_label.mouseReleaseEvent = self.on_image_mouse_release
        camera_layout.addWidget(self.image_label)

        splitter.addWidget(camera_widget)
        splitter.setStretchFactor(0, 2)

        # Right side: counts panel (always visible)
        self.counts_panel = CountsPanel()
        self.counts_panel.viewer = self
        if self.saved_norm_reference is not None:
            self.counts_panel.norm_reference = self.saved_norm_reference
        splitter.addWidget(self.counts_panel)
        splitter.setStretchFactor(1, 3)
        splitter.setHandleWidth(2)
    
    def on_toggle_stream(self):
        if self.is_streaming:
            self.timer.stop()
            self.camera.StopGrabbing()
            self.camera.Close()
            self.is_streaming = False
            self.stream_action.setText("Start Stream")
            self.image_label.setText("Stream stopped")
        else:
            self.connect_camera()
    
    def show_settings_dialog(self):
        if self.camera is None:
            return
        if self.settings_dialog is None:
            self.settings_dialog = CameraSettingsDialog(self.camera, self)
            self.settings_dialog.show()
        else:
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()

    def _update_camera_menu_spinboxes(self):
        """Sync the inline camera menu spinboxes to the current camera state."""
        try:
            self.gain_spinbox.setRange(self.camera.Gain.GetMin(), self.camera.Gain.GetMax())
            self.gain_spinbox.blockSignals(True)
            self.gain_spinbox.setValue(self.camera.Gain.GetValue())
            self.gain_spinbox.blockSignals(False)
            self.gain_spinbox.setEnabled(True)
        except Exception:
            pass
        try:
            exp_min = self.camera.ExposureTime.GetMin()
            exp_max = self.camera.ExposureTime.GetMax()
            self.exposure_spinbox.setRange(exp_min, exp_max)
            self.exposure_spinbox.blockSignals(True)
            self.exposure_spinbox.setValue(self.camera.ExposureTime.GetValue())
            self.exposure_spinbox.blockSignals(False)
            self.exposure_spinbox.setEnabled(True)
        except Exception:
            pass

    def _on_gain_changed(self, value):
        if self.camera is not None:
            try:
                self.camera.Gain.SetValue(value)
            except Exception as e:
                print(f"Error setting gain: {e}")

    def _on_exposure_changed(self, value):
        if self.camera is not None:
            try:
                self.camera.ExposureTime.SetValue(value)
            except Exception as e:
                print(f"Error setting exposure time: {e}")

    def _on_fixed_interval_menu_toggled(self, checked):
        self.fixed_interval_btn.setText("On" if checked else "Off")
        self.counts_panel.fixed_interval = checked
        self.counts_panel.update_plot()

    def _on_time_window_menu_changed(self, value):
        self.counts_panel.time_window = value
        self.counts_panel.update_plot()

    def _on_norm_ref_line_toggled(self, checked):
        self.counts_panel.show_norm_reference_line = checked
        self.counts_panel.update_plot()

    def _on_normalize_toggled(self, checked):
        self.counts_panel.normalize = checked
        if checked and self.counts_panel.counts:
            self.counts_panel.norm_reference = float(np.mean(self.counts_panel.counts))
        self.counts_panel.update_plot()

    def _on_auto_rescale_toggled(self, checked):
        self.counts_panel.auto_rescale = checked
        self.counts_panel.update_plot()

    def label_to_image_coords(self, label_pos):
        """Convert label coordinates to original image coordinates"""
        if self.last_image is None:
            return label_pos
        
        label = self.image_label
        pixmap = label.pixmap()
        if not pixmap:
            return label_pos
        
        # Get the actual displayed pixmap size
        displayed_pixmap_size = pixmap.size()
        label_size = label.size()
        
        # Calculate offset due to centering
        offset_x = (label_size.width() - displayed_pixmap_size.width()) / 2
        offset_y = (label_size.height() - displayed_pixmap_size.height()) / 2
        
        # Convert label position to pixmap position
        pixmap_x = label_pos.x() - offset_x
        pixmap_y = label_pos.y() - offset_y
        
        # Clip to pixmap bounds
        pixmap_x = max(0, min(pixmap_x, displayed_pixmap_size.width()))
        pixmap_y = max(0, min(pixmap_y, displayed_pixmap_size.height()))
        
        # Scale from pixmap coordinates to original image coordinates
        scale_x = self.last_image.shape[1] / displayed_pixmap_size.width()
        scale_y = self.last_image.shape[0] / displayed_pixmap_size.height()
        
        image_x = int(pixmap_x * scale_x)
        image_y = int(pixmap_y * scale_y)
        
        return QPoint(image_x, image_y)
    
    def on_image_mouse_press(self, event):
        if self.show_rectangle:
            self.rect_start = self.label_to_image_coords(event.pos())
            self.rect_end = QPoint(self.rect_start)
    
    def on_image_mouse_move(self, event):
        if self.show_rectangle and self.rect_start:
            self.rect_end = self.label_to_image_coords(event.pos())
    
    def on_image_mouse_release(self, event):
        if self.show_rectangle and self.rect_start:
            self.rect_end = self.label_to_image_coords(event.pos())
            self.save_rectangle()
    
    def load_rectangle(self):
        if os.path.exists(self.rect_file):
            try:
                with open(self.rect_file, 'r') as f:
                    data = json.load(f)
                    self.current_rect = QRect(data['x1'], data['y1'], data['x2'] - data['x1'], data['y2'] - data['y1'])
                    if 'norm_reference' in data and data['norm_reference'] is not None:
                        self.saved_norm_reference = data['norm_reference']
            except:
                self.current_rect = None
    
    def save_rectangle(self):
        if self.rect_start and self.rect_end:
            x1 = min(self.rect_start.x(), self.rect_end.x())
            y1 = min(self.rect_start.y(), self.rect_end.y())
            x2 = max(self.rect_start.x(), self.rect_end.x())
            y2 = max(self.rect_start.y(), self.rect_end.y())
            self.current_rect = QRect(x1, y1, x2 - x1, y2 - y1)
            
            data = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'norm_reference': self.counts_panel.norm_reference}
            with open(self.rect_file, 'w') as f:
                json.dump(data, f)
    
    def connect_camera(self):
        if self.connecting:
            return
        self.connecting = True
        try:
            # Create camera factory
            tlf = pylon.TlFactory.GetInstance()
            devices = tlf.EnumerateDevices()
            
            if not devices:
                self.image_label.setText("No camera devices found!")
                self.set_failed_state()
                return
            
            # Find camera by serial number
            for device in devices:
                if device.GetSerialNumber() == self.SERIAL_NUMBER:
                    self.camera = pylon.InstantCamera(tlf.CreateDevice(device))
                    break
            
            if self.camera is None:
                self.image_label.setText(f"Camera with serial number {self.SERIAL_NUMBER} not found!")
                self.set_failed_state()
                return
            
            # Open camera
            self.camera.Open()
            model_name = self.camera.GetDeviceInfo().GetModelName()
            self.setWindowTitle(f"Basler Camera Stream - {model_name} ({self.SERIAL_NUMBER})")
            
            # Get maximum pixel value from camera
            try:
                # Try to get the max pixel value directly from the camera
                if hasattr(self.camera, 'PixelDynamicRangeMax'):
                    self.max_pixel_value = int(self.camera.PixelDynamicRangeMax.GetValue())
                else:
                    # Fallback to 8-bit (255)
                    self.max_pixel_value = 255
                # print(f"Max pixel value: {self.max_pixel_value}")
            except Exception as e:
                # Fallback to 8-bit (255)
                self.max_pixel_value = 255
                print(f"Could not read max pixel value from camera, using default 8-bit: {e}")
            
            # Set default values
            try:
                self.camera.Gain.SetValue(12)
            except Exception as e:
                print(f"Could not set default gain: {e}")
            
            try:
                self.camera.ExposureTime.SetValue(300)
            except Exception as e:
                print(f"Could not set default exposure time: {e}")

            # Sync camera menu spinboxes to the current camera state
            self._update_camera_menu_spinboxes()

            # Start grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_streaming = True
            # Connected: stop retrying, reset button UI
            if self.retry_timer.isActive():
                self.retry_timer.stop()
            self.stream_action.setText("Stop Stream")
            
            # Start timer for frame updates
            self.timer.start(30)  # Update every 30ms (~33fps)
            
        except Exception as e:
            self.image_label.setText(f"Error connecting to camera: {str(e)}")
            self.set_failed_state()
        finally:
            self.connecting = False

    def set_failed_state(self):
        self.is_streaming = False
        self.stream_action.setText("Could not open camera — retrying")
        if not self.retry_timer.isActive():
            self.retry_timer.start()
    
    def update_frame(self):
        if self.camera is None or not self.camera.IsGrabbing():
            return
        
        try:
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grabResult.GrabSucceeded():
                image = grabResult.Array
                self.last_image = image.copy()
                
                # Convert BGR to RGB for display
                if len(image.shape) == 2:  # Grayscale
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:  # Color
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Calculate pixel counts if rectangle is defined
                if self.current_rect:
                    x1 = max(0, self.current_rect.x())
                    y1 = max(0, self.current_rect.y())
                    x2 = min(image.shape[1], self.current_rect.x() + self.current_rect.width())
                    y2 = min(image.shape[0], self.current_rect.y() + self.current_rect.height())

                    if x2 > x1 and y2 > y1:
                        roi = image[y1:y2, x1:x2]
                        pixel_sum = np.sum(roi)
                        self.counts_panel.add_count(pixel_sum)
                
                # Convert to QImage
                h, w, ch = image_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Draw rectangle if needed
                if self.show_rectangle:
                    qt_image = qt_image.copy()
                    painter = QPainter(qt_image)
                    painter.setPen(QPen(QColor(0, 255, 0), 3))
                    
                    # Draw current rectangle being drawn
                    if self.rect_start and self.rect_end:
                        x1 = min(self.rect_start.x(), self.rect_end.x())
                        y1 = min(self.rect_start.y(), self.rect_end.y())
                        x2 = max(self.rect_start.x(), self.rect_end.x())
                        y2 = max(self.rect_start.y(), self.rect_end.y())
                        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                    
                    # Draw saved rectangle
                    if self.current_rect and self.current_rect.width() > 0 and self.current_rect.height() > 0:
                        painter.drawRect(self.current_rect)
                    
                    painter.end()
                
                # Check for saturated pixels
                max_pixel_value = self.max_pixel_value if hasattr(self, 'max_pixel_value') else 255
                if self.saturation_in_box_only and self.current_rect and self.current_rect.width() > 0 and self.current_rect.height() > 0:
                    sx1 = max(0, self.current_rect.x())
                    sy1 = max(0, self.current_rect.y())
                    sx2 = min(image.shape[1], self.current_rect.x() + self.current_rect.width())
                    sy2 = min(image.shape[0], self.current_rect.y() + self.current_rect.height())
                    check_region = image[sy1:sy2, sx1:sx2] if sx2 > sx1 and sy2 > sy1 else image
                else:
                    check_region = image
                is_saturated = np.any(check_region == max_pixel_value)
                
                # Draw saturation warning if needed
                if is_saturated:
                    qt_image = qt_image.copy()
                    painter = QPainter(qt_image)
                    painter.setPen(QPen(QColor(255, 0, 0), 10))
                    painter.setFont(painter.font())
                    font_large = QFont("Times", 100)
                    # Apply the new font to the painter
                    painter.setFont(font_large)
                    painter.drawText(10, 170, "⚠ SATURATION WARNING ⚠")
                    painter.end()
                
                # Scale to fit label
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.current_pixmap = pixmap
            
            grabResult.Release()
        
        except Exception as e:
            print(f"Error updating frame: {str(e)}")
    
    def closeEvent(self, event):
        if self.retry_timer.isActive():
            self.retry_timer.stop()
        if self.settings_dialog:
            self.settings_dialog.close()
        if self.camera:
            self.timer.stop()
            self.camera.StopGrabbing()
            self.camera.Close()
        super().closeEvent(event)

def main():
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('kexp.mot_viewer')
    except Exception:
        pass

    app = QApplication(sys.argv)

    # Render red ball emoji as window/taskbar icon
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setFont(QFont("Segoe UI Emoji", 48))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "\U0001f534")
    painter.end()
    app.setWindowIcon(QIcon(pixmap))

    viewer = BaslerCameraViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
