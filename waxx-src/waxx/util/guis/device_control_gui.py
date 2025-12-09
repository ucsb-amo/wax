import sys
import json
import os
import importlib.util
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox, QPushButton,
    QCheckBox, QComboBox, QLineEdit, QGroupBox, QMessageBox, QStackedWidget
)
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QFont

# from waxx.config.dds_id import dds_frame
from waxx.util.import_module_from_file import load_module_from_file

PX_WIDTH_PER_COLUMN = 100
STATE_BUTTON_ON_COLOR = "green"

class DeviceWidget(QWidget):
    """Base class for device control widgets"""
    value_changed = pyqtSignal(str, str, dict)
    
    def __init__(self, device_name: str, device_config: Dict[str, Any]):
        super().__init__()
        self.device_name = device_name
        self.device_config = device_config
        self.setFont(QFont("Arial", 9))
        
    def get_updated_config(self) -> Dict[str, Any]:
        """Return the updated configuration for this device"""
        raise NotImplementedError
        
    def update_from_config(self, config: Dict[str, Any]):
        """Update widget values from configuration"""
        raise NotImplementedError

class DDSWidget(DeviceWidget):
    """Widget for controlling DDS devices"""
    
    def __init__(self, device_name: str, device_config: Dict[str, Any], dds_frame_obj=None):
        super().__init__(device_name, device_config)
        self.dds_frame_obj = dds_frame_obj
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        label = QLineEdit(self.device_name)
        label.setReadOnly(True)
        label.setToolTip(self.device_name)
        layout.addWidget(label)
        
        # Frequency controls
        freq_layout = QHBoxLayout()
        # freq_layout.addWidget(QLabel("Frequency:"))
        
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setSingleStep(0.1)
        self.freq_spinbox.setDecimals(4)
        self.freq_spinbox.setValue(self.device_config["frequency"] / 1e6)  # Convert Hz to MHz
        self.freq_spinbox.setMinimum(0.)
        self.freq_spinbox.setMaximum(400.)
        self.freq_spinbox.editingFinished.connect(self.on_update_clicked)
        self.freq_spinbox.valueChanged.connect(self.on_update_clicked)
        freq_layout.addWidget(self.freq_spinbox)
        
        # Frequency unit selector (MHz/Γ) if transition is not None
        self.freq_unit_combo = QComboBox()
        self.freq_unit_combo.addItem("MHz")
        if self.device_config.get("transition", "None") != "None":
            self.freq_unit_combo.addItem("Γ")
        self.freq_unit_combo.currentTextChanged.connect(self.on_freq_unit_changed)
        
        freq_layout.addWidget(self.freq_unit_combo)
            
        layout.addLayout(freq_layout)
        
        # Amplitude controls
        amp_layout = QHBoxLayout()
        
        self.amp_spinbox = QDoubleSpinBox()
        self.amp_spinbox.setRange(0, 1)
        self.amp_spinbox.setDecimals(3)
        self.amp_spinbox.setSingleStep(0.005)
        self.amp_spinbox.setValue(self.device_config["amplitude"])
        self.amp_spinbox.editingFinished.connect(self.on_update_clicked)
        self.amp_spinbox.valueChanged.connect(self.on_update_clicked)

        self.vpd_spinbox = QDoubleSpinBox()
        self.vpd_spinbox.setRange(0, 10)
        self.vpd_spinbox.setDecimals(2)
        self.vpd_spinbox.setSingleStep(0.05)
        self.vpd_spinbox.setValue(self.device_config.get("v_pd", 5.0))
        self.vpd_spinbox.editingFinished.connect(self.on_update_clicked)
        self.vpd_spinbox.valueChanged.connect(self.on_update_clicked)

        self.power_control_widget = QHBoxLayout()
        self.power_control_widget.addWidget(self.amp_spinbox)
        self.power_control_widget.addWidget(self.vpd_spinbox)
        
        amp_layout.addLayout(self.power_control_widget)
        
        # Amplitude unit selector (Amp/V)
        self.amp_unit_combo = QComboBox()
        self.amp_unit_combo.addItems(["Amp"])
        start_unit = "amp"
        if self.device_config.get("dac_ch", -1) != -1:
            self.amp_unit_combo.addItem("V")
            start_unit = "V"
            self.amp_unit_combo.setCurrentIndex(1)
        self.amp_unit_combo.currentTextChanged.connect(self.on_amp_unit_changed)
        amp_layout.addWidget(self.amp_unit_combo)
        layout.addLayout(amp_layout)
        
        # Update button

        self.state_button = QPushButton("Off")
        self.state_button.setCheckable(True)
        self.state_button.toggled.connect(self.on_state_button_toggled)
        state_button_row = QHBoxLayout()
        # state_button_row.addWidget(QLabel("sw state:"))
        state_button_row.addWidget(self.state_button)

        layout.addLayout(state_button_row)
        
        self.setLayout(layout)
        self.update_from_config(self.device_config)
        self.update_from_config(self.device_config)

        self.on_amp_unit_changed(start_unit)
        
    def on_state_button_toggled(self, checked):
        if checked:
            self.state_button.setText("On")
            self.state_button.setStyleSheet(f"background-color: {STATE_BUTTON_ON_COLOR}")
        else:
            self.state_button.setText("Off")
            self.state_button.setStyleSheet("")
        self.on_update_clicked()
        
    def on_freq_unit_changed(self, unit):
        """Handle frequency unit change between MHz and Γ"""
        current_value = self.freq_spinbox.value()

        if unit == "Γ":
            # Convert MHz to Γ
            if self.dds_frame_obj:
                try:
                    uru_idx = self.device_config["urukul_idx"]
                    ch = self.device_config["ch"]
                    dds_obj = self.dds_frame_obj.dds_array[uru_idx][ch]
                    freq_hz = current_value * 1e6
                    Γ_value = dds_obj.frequency_to_detuning(freq_hz)
                    self.freq_spinbox.setMinimum(-100.)
                    self.freq_spinbox.setMaximum(100.)
                    self.freq_spinbox.setValue(Γ_value)
                except Exception as e:
                    print(e)
        elif unit == "MHz":
            # Convert Γ to MHz
            if self.dds_frame_obj:
                try:
                    uru_idx = self.device_config["urukul_idx"]
                    ch = self.device_config["ch"]
                    dds_obj = self.dds_frame_obj.dds_array[uru_idx][ch]
                    freq_hz = dds_obj.detuning_to_frequency(current_value)
                    self.freq_spinbox.setMinimum(0.)
                    self.freq_spinbox.setMaximum(400.)
                    self.freq_spinbox.setValue(freq_hz / 1e6)
                except Exception as e:
                    print(e)
                    
    def on_amp_unit_changed(self, unit):
        """Handle amplitude unit change between Amp and V"""
        if self.device_config.get("dac_ch", -1) == -1:
            self.vpd_spinbox.setVisible(False)
            return
        if unit == "V":
            self.amp_spinbox.setVisible(False)
            self.vpd_spinbox.setVisible(True)
        else:
            self.amp_spinbox.setVisible(True)
            self.vpd_spinbox.setVisible(False)
            
    def on_update_clicked(self):
        """Handle update button click"""
        updated_config = self.get_updated_config()
        self.value_changed.emit("dds", self.device_name, updated_config)
        
    def get_updated_config(self) -> Dict[str, Any]:
        """Return the updated configuration for this DDS device"""
        config = self.device_config.copy()
        
        # Update frequency
        freq_value = self.freq_spinbox.value()
        if self.freq_unit_combo.currentText() == "Γ":
            try:
                uru_idx = self.device_config["urukul_idx"]
                ch = self.device_config["ch"]
                dds_obj = self.dds_frame_obj.dds_array[uru_idx][ch]
                freq_hz = dds_obj.detuning_to_frequency(freq_value)
                
                config["frequency"] = freq_hz
            except:
                config["frequency"] = freq_value * 1e6  # Fallback to MHz conversion
        else:
            config["frequency"] = freq_value * 1e6  # Convert MHz to Hz
            
        # Update amplitude
        # if self.amp_unit_combo.currentText() == "V":
        config["v_pd"] = self.vpd_spinbox.value()
        # else:
        config["amplitude"] = self.amp_spinbox.value()

        # Update sw state
        config["sw_state"] = int(self.state_button.isChecked())
            
        return config
        
    def update_from_config(self, config: Dict[str, Any]):
        """Update widget values from configuration"""
        self.device_config = config
        self.freq_spinbox.setValue(config["frequency"] / 1e6)
        self.amp_spinbox.setValue(config["amplitude"])
        if "v_pd" in config:
            self.vpd_spinbox.setValue(config["v_pd"])
        self.state_button.setChecked(bool(config["sw_state"]))

class DACWidget(DeviceWidget):
    """Widget for controlling DAC devices"""

    def __init__(self, device_name: str, device_config: Dict[str, Any]):
        super().__init__(device_name, device_config)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()

        label = QLineEdit(self.device_name)
        label.setReadOnly(True)
        label.setToolTip(self.device_name)
        layout.addWidget(label)
        
        # Voltage control
        voltage_layout = QHBoxLayout()
        # voltage_layout.addWidget(QLabel("Voltage:"))
        
        self.voltage_spinbox = QDoubleSpinBox()
        self.voltage_spinbox.setRange(-9.999, 9.999)
        self.voltage_spinbox.setDecimals(3)
        self.voltage_spinbox.setSuffix(" V")
        self.voltage_spinbox.setValue(self.device_config["voltage"])
        self.voltage_spinbox.editingFinished.connect(self.on_update_clicked)
        self.voltage_spinbox.valueChanged.connect(self.on_update_clicked)
        voltage_layout.addWidget(self.voltage_spinbox)
        
        layout.addLayout(voltage_layout)
        
        self.setLayout(layout)
        
    def on_update_clicked(self):
        """Handle update button click"""
        updated_config = self.get_updated_config()
        self.value_changed.emit("dac", self.device_name, updated_config)
        
    def get_updated_config(self) -> Dict[str, Any]:
        """Return the updated configuration for this DAC device"""
        config = self.device_config.copy()
        config["voltage"] = self.voltage_spinbox.value()
        return config
        
    def update_from_config(self, config: Dict[str, Any]):
        """Update widget values from configuration"""
        self.device_config = config
        self.voltage_spinbox.setValue(config["voltage"])


class TTLWidget(DeviceWidget):
    """Widget for controlling TTL devices"""

    def __init__(self, device_name: str, device_config: Dict[str, Any]):
        super().__init__(device_name, device_config)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()

        label = QLineEdit(self.device_name)
        label.setReadOnly(True)
        label.setToolTip(self.device_name)
        layout.addWidget(label)
        
        # State control
        state_layout = QHBoxLayout()
        # state_layout.addWidget(QLabel("State:"))
        
        self.state_button = QPushButton("Off")
        self.state_button.setCheckable(True)
        self.state_button.toggled.connect(self.on_state_button_toggled)
        state_layout.addWidget(self.state_button)
        
        layout.addLayout(state_layout)
        
        self.setLayout(layout)
        self.update_from_config(self.device_config)

    def on_state_button_toggled(self, checked):
        if checked:
            self.state_button.setText("On")
            self.state_button.setStyleSheet("background-color: lightgreen")
        else:
            self.state_button.setText("Off")
            self.state_button.setStyleSheet("")
        self.on_update_clicked()
        
    def on_update_clicked(self):
        """Handle update button click"""
        updated_config = self.get_updated_config()
        self.value_changed.emit("ttl", self.device_name, updated_config)
        
    def on_pulse_clicked(self):
        """Handle pulse button click"""
        try:
            pulse_time = float(self.pulse_time_edit.text())
            # Here you would implement the actual pulse functionality
            # For now, just show a message
            QMessageBox.information(self, "Pulse", f"Pulse command sent for {pulse_time} seconds")
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid pulse time format")
        
    def get_updated_config(self) -> Dict[str, Any]:
        """Return the updated configuration for this TTL device"""
        config = self.device_config.copy()
        config["ttl_state"] = int(self.state_button.isChecked())
        return config
        
    def update_from_config(self, config: Dict[str, Any]):
        """Update widget values from configuration"""
        self.device_config = config
        self.state_button.setChecked(bool(config["ttl_state"]))


class DeviceStateGUI(QMainWindow):
    """Main GUI application for device state management"""
    
    def __init__(self,
                  monitor_server_ip,
                  device_state_json_path,
                  dds_frame):
        super().__init__()
        self.config_file = device_state_json_path
        self.config_data = {}
        self.device_widgets = {}
        
        self.dds_frame_obj = dds_frame

        self.setup_ui()
        self.load_config()
        self.setup_timer()
        
    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("Device State Control")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        central_widget_layout = QVBoxLayout()
        central_widget_layout.addWidget(self.tab_widget)
        central_widget.setLayout(central_widget_layout)
        
        # Create tabs
        self.dds_tab = QWidget()
        self.dac_tab = QWidget()
        self.ttl_tab = QWidget()
        
        self.tab_widget.addTab(self.dds_tab, "DDS")
        self.tab_widget.addTab(self.dac_tab, "DAC")
        self.tab_widget.addTab(self.ttl_tab, "TTL")
        
        # Setup tab layouts
        self.dds_layout = QGridLayout()
        self.dac_layout = QGridLayout()
        self.ttl_layout = QGridLayout()
        
        self.dds_tab.setLayout(self.dds_layout)
        self.dac_tab.setLayout(self.dac_layout)
        self.ttl_tab.setLayout(self.ttl_layout)
        
    def setup_timer(self):
        """Setup timer for periodic config file checking"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_config_changes)
        self.timer.start(1000)  # Check every 1 second
        
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                new_config = json.load(f)
                
            # Check if config has changed
            if new_config != self.config_data:
                self.config_data = new_config
                self.update_device_widgets()
                
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Configuration file not found: {self.config_file}")
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Error", f"Invalid JSON in configuration file: {e}")
            
    def check_config_changes(self):
        """Check for changes in the config file"""
        self.load_config()
        
    def update_device_widgets(self):
        """Update device widgets based on current configuration"""
        # Clear existing widgets
        self.clear_layouts()
        self.device_widgets.clear()
        
        # Add DDS widgets organized by urukul_idx (columns) and ch (rows)
        if "dds" in self.config_data:
            for device_name, device_config in self.config_data["dds"].items():
                # Add urukul_idx and ch to config for DDS widgets
                if "urukul_idx" not in device_config:
                    device_config["urukul_idx"] = device_config.get("urukul_idx", 0)
                if "ch" not in device_config:
                    device_config["ch"] = device_config.get("ch", 0)
                    
                widget = DDSWidget(device_name, device_config, self.dds_frame_obj)
                widget.value_changed.connect(self.on_device_value_changed)
                
                # Position by urukul_idx (column) and ch (row)
                row = device_config["ch"]
                col = device_config["urukul_idx"]
                
                self.dds_layout.addWidget(widget, row, col)
                self.device_widgets[f"dds.{device_name}"] = widget
                    
        # Add DAC widgets grouped into columns of 8
        if "dac" in self.config_data:
            for device_name, device_config in self.config_data["dac"].items():
                widget = DACWidget(device_name, device_config)
                widget.value_changed.connect(self.on_device_value_changed)
                
                # Extract channel number from device config or name
                ch = device_config.get("ch", 0)
                if ch == 0 and device_name.startswith("dac_ch"):
                    try:
                        ch = int(device_name.split("dac_ch")[1])
                    except (ValueError, IndexError):
                        ch = 0
                
                # Position: column groups of 8, row within group
                col = ch // 8
                row = ch % 8
                
                self.dac_layout.addWidget(widget, row, col)
                self.device_widgets[f"dac.{device_name}"] = widget
                    
        # Add TTL widgets grouped into columns of 8 (0-7, 8-15, etc.)
        if "ttl" in self.config_data:
            ttl_devices = self.config_data["ttl"].items()
            
            # Pre-process to get channel numbers and find max channel
            processed_ttls = []
            max_ch = -1
            for device_name, device_config in ttl_devices:
                ch = device_config.get("ch", 0)
                if ch > max_ch:
                    max_ch = ch
                processed_ttls.append((device_name, device_config))

            num_cols = (max_ch // 8) + 1 if max_ch != -1 else 0

            # Create a grid of widgets to place, initialized with placeholders
            widget_grid = [[QWidget() for _ in range(8)] for _ in range(num_cols)]

            for device_name, device_config in processed_ttls:
                ch = device_config["ch"]
                col = ch // 8
                row = ch % 8
                
                if 0 <= col < num_cols:
                    widget = TTLWidget(device_name, device_config)
                    widget.value_changed.connect(self.on_device_value_changed)
                    widget_grid[col][row] = widget
                    self.device_widgets[f"ttl.{device_name}"] = widget

            # Add widgets to layout
            for col_idx, col_widgets in enumerate(widget_grid):
                for row_idx, widget in enumerate(col_widgets):
                    self.ttl_layout.addWidget(widget, row_idx, col_idx)
        
        self.adjust_window_width()
                    
        # Adjust window width based on the number of columns
        self.adjust_window_width()
        
    def adjust_window_width(self):
        """Adjust the window width based on the tab with the most columns."""
        max_columns = 0
        for layout in [self.dds_layout, self.dac_layout, self.ttl_layout]:
            max_columns = max(max_columns, layout.columnCount())
        
        if max_columns > 0:
            # Add a buffer column for aesthetics
            new_width = (max_columns + 1) * PX_WIDTH_PER_COLUMN
            self.resize(new_width, self.height())
        
    def clear_layouts(self):
        """Clear all device widgets from layouts"""
        for layout in [self.dds_layout, self.dac_layout, self.ttl_layout]:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                    
    def on_device_value_changed(self, device_type: str, device_name: str, updated_config: Dict[str, Any]):
        """Handle device value changes"""
        # Determine device type and update config
        if device_name in self.config_data.get("dds", {}) and device_type=="dds":
            self.config_data["dds"][device_name].update(updated_config)
        elif device_name in self.config_data.get("dac", {}) and device_type=="dac":
            self.config_data["dac"][device_name].update(updated_config)
        elif device_name in self.config_data.get("ttl", {}) and device_type=="ttl":
            print(updated_config)
            self.config_data["ttl"][device_name].update(updated_config)
            
        # Save updated config to file
        self.save_config()
        
    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config_data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")