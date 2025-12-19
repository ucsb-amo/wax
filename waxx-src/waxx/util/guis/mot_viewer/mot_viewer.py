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
                             QMenuBar, QMenu)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
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


class CountsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pixel Counts")
        self.parent_viewer = parent
        
        # Position to the right of parent window
        if parent:
            parent_geometry = parent.geometry()
            x = parent_geometry.right() + 10
            y = parent_geometry.top()
            self.setGeometry(x, y, 500, 400)
        else:
            self.setGeometry(200, 200, 500, 400)
        
        layout = QVBoxLayout()
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_data)
        button_layout.addWidget(clear_button)
        
        # Settings button
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.show_settings)
        button_layout.addWidget(settings_button)
        
        layout.addLayout(button_layout)
        
        # PyQtGraph plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Pixel Counts')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setTitle("Summed Pixel Counts vs Time")
        self.plot_widget.enableAutoRange()
        layout.addWidget(self.plot_widget)
        
        self.setLayout(layout)
        
        self.timestamps = []
        self.counts = []
        self.start_time = None
        self.fixed_interval = True
        self.time_window = 30
        self.settings_dialog = None
    
    def add_count(self, count):
        if self.start_time is None:
            self.start_time = datetime.now()
        
        self.timestamps.append((datetime.now() - self.start_time).total_seconds())
        self.counts.append(count)
        self.update_plot()
    
    def update_plot(self):
        self.plot_widget.clear()
        plot_item = self.plot_widget.getPlotItem()
        
        # Determine which data to plot
        if self.fixed_interval and len(self.timestamps) > 0:
            current_time = self.timestamps[-1]
            # Keep a full-width window even if not enough data yet
            if current_time < self.time_window:
                start_time = 0
                end_time = self.time_window
            else:
                start_time = current_time - self.time_window
                end_time = current_time
            
            # Slice data within window
            start_idx = 0
            for i, t in enumerate(self.timestamps):
                if t >= start_time:
                    start_idx = i
                    break
            plot_timestamps = self.timestamps[start_idx:]
            plot_counts = self.counts[start_idx:]
            
            # Lock X range to fixed window, keep Y auto
            plot_item.enableAutoRange(axis='x', enable=False)
            plot_item.enableAutoRange(axis='y', enable=True)
            plot_item.setXRange(start_time, end_time, padding=0)
        else:
            plot_timestamps = self.timestamps
            plot_counts = self.counts
            # Auto-range both axes when not fixed interval or no data
            plot_item.enableAutoRange(axis='x', enable=True)
            plot_item.enableAutoRange(axis='y', enable=True)
        
        # Draw data if any points exist (green line, no points)
        if len(plot_timestamps) > 0:
            self.plot_widget.plot(plot_timestamps, plot_counts, pen=pg.mkPen('g', width=2))
    
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
    
    def closeEvent(self, event):
        if self.settings_dialog:
            self.settings_dialog.close()
        if self.parent_viewer:
            self.parent_viewer.counts_window = None
            self.parent_viewer.show_rectangle = False
        super().closeEvent(event)

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
        self.show_rectangle = False
        self.counts_window = None
        self.rect_file = "camera_rect.json"
        self.last_image = None
        self.current_pixmap = None
        self.settings_dialog = None
        self.load_rectangle()
        
        self.initUI()
        self.connect_camera()
    
    def initUI(self):
        self.setWindowTitle("Basler Camera Stream")
        self.setGeometry(100, 100, 768, 432)
        
        # Create menu bar
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")
        settings_action = settings_menu.addAction("Camera Settings")
        settings_action.triggered.connect(self.show_settings_dialog)
        
        # Add a direct Counts action on the menu bar
        counts_action = menubar.addAction("Counts")
        counts_action.triggered.connect(self.show_counts_window)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.mousePressEvent = self.on_image_mouse_press
        self.image_label.mouseMoveEvent = self.on_image_mouse_move
        self.image_label.mouseReleaseEvent = self.on_image_mouse_release
        layout.addWidget(self.image_label)
        
        # Toggle button
        self.toggle_button = QPushButton("Stop")
        self.toggle_button.clicked.connect(self.on_toggle_stream)
        layout.addWidget(self.toggle_button)
        
        central_widget.setLayout(layout)
    
    def on_toggle_stream(self):
        if self.is_streaming:
            # Stop streaming and close camera
            self.timer.stop()
            self.camera.StopGrabbing()
            self.camera.Close()
            self.is_streaming = False
            self.toggle_button.setText("Start")
            self.toggle_button.setStyleSheet("")
            self.image_label.setText("Stream stopped")
        else:
            # Attempt to (re)connect and start streaming
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
    
    def show_counts_window(self):
        if self.counts_window is None:
            self.counts_window = CountsWindow(self)
            self.counts_window.show()
            self.show_rectangle = True
        else:
            self.counts_window.raise_()
            self.counts_window.activateWindow()
    
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
            except:
                self.current_rect = None
    
    def save_rectangle(self):
        if self.rect_start and self.rect_end:
            x1 = min(self.rect_start.x(), self.rect_end.x())
            y1 = min(self.rect_start.y(), self.rect_end.y())
            x2 = max(self.rect_start.x(), self.rect_end.x())
            y2 = max(self.rect_start.y(), self.rect_end.y())
            self.current_rect = QRect(x1, y1, x2 - x1, y2 - y1)
            
            data = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
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
            self.setWindowTitle(f"Basler Camera Stream - {model_name}")
            
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
                self.camera.Gain.SetValue(5)
            except Exception as e:
                print(f"Could not set default gain: {e}")
            
            try:
                self.camera.ExposureTime.SetValue(1000)
            except Exception as e:
                print(f"Could not set default exposure time: {e}")
            
            # Start grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_streaming = True
            # Connected: stop retrying, reset button UI
            if self.retry_timer.isActive():
                self.retry_timer.stop()
            self.toggle_button.setText("Stop")
            self.toggle_button.setStyleSheet("")
            
            # Start timer for frame updates
            self.timer.start(30)  # Update every 30ms (~33fps)
            
        except Exception as e:
            self.image_label.setText(f"Error connecting to camera: {str(e)}")
            self.set_failed_state()
        finally:
            self.connecting = False

    def set_failed_state(self):
        # Indicate failure and schedule retry
        self.is_streaming = False
        self.toggle_button.setText("Could not open camera")
        self.toggle_button.setStyleSheet("background-color: orange; color: black;")
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
                
                # Calculate pixel counts if rectangle is defined and counts window is open
                if self.counts_window and self.current_rect:
                    x1 = max(0, self.current_rect.x())
                    y1 = max(0, self.current_rect.y())
                    x2 = min(image.shape[1], self.current_rect.x() + self.current_rect.width())
                    y2 = min(image.shape[0], self.current_rect.y() + self.current_rect.height())
                    
                    if x2 > x1 and y2 > y1:
                        roi = image[y1:y2, x1:x2]
                        pixel_sum = np.sum(roi)
                        self.counts_window.add_count(pixel_sum)
                
                # Convert to QImage
                h, w, ch = image_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Draw rectangle if needed
                if self.show_rectangle:
                    qt_image = qt_image.copy()
                    painter = QPainter(qt_image)
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    
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
                is_saturated = np.any(image == max_pixel_value)
                
                # Draw saturation warning if needed
                if is_saturated:
                    qt_image = qt_image.copy()
                    painter = QPainter(qt_image)
                    painter.setPen(QPen(QColor(255, 0, 0), 10))
                    painter.setFont(painter.font())
                    painter.drawText(10, 30, "⚠ SATURATION WARNING ⚠")
                    painter.end()
                
                # Scale to fit label
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaledToWidth(self.image_label.width(), Qt.TransformationMode.SmoothTransformation)
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
        if self.counts_window:
            self.counts_window.close()
        if self.camera:
            self.timer.stop()
            self.camera.StopGrabbing()
            self.camera.Close()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    viewer = BaslerCameraViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
