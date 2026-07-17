import sys
import json
import os
import threading
import importlib.util
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox, QPushButton,
    QCheckBox, QComboBox, QLineEdit, QGroupBox, QMessageBox, QStackedWidget,
    QSizePolicy
)
from PyQt6.QtCore import QTimer, pyqtSignal, QThread, QSignalBlocker, QEvent
from PyQt6.QtGui import QFont, QIcon, QPainter, QPixmap

from PyQt6.QtCore import Qt

import time

from waxx.util.comms_server.comm_client import MonitorClient
from waxx.util.comms_server.comm_server import STATES
from waxx.util.comms_server.state_broadcast import StateListener
from waxa.browser.browser_window import (
    parse_name_search_terms,
    name_matches_all_terms,
)

PX_WIDTH_PER_COLUMN = 100
STATE_BUTTON_ON_COLOR = "green"
DEFAULT_BUTTON_COLOR = "#363636FF"  # Dark gray
UNDO_BUTTON_COLOR = "orange"

# --- Compact / responsive mode -------------------------------------------
COMPACT_PX_WIDTH_PER_COLUMN = 44   # narrower per-column budget when compact
COMPACT_HYSTERESIS_PX = 60         # extra room required to leave compact mode
HOVER_DWELL_MS = 300               # hover time before the settings popup opens
POPUP_HIDE_GRACE_MS = 300          # delay before the popup hides after leaving
SEARCH_OUTLINE_STYLE = "border: 2px solid #4da6ff;"  # search highlight (compact)
DAC_NONZERO_NAME_COLOR = "#5a5a5a"  # brighter gray when |voltage| > 0


class ScrollableButton(QLineEdit):
    """A compact, button-like control whose (possibly long) name text scrolls
    horizontally by dragging instead of forcing the column wider.

    Implemented as a read-only, frameless ``QLineEdit`` (which scrolls long
    text natively) that behaves like a button: it emits ``clicked`` on a
    press-release that did not drag, and optionally supports a checkable
    on/off state via ``toggled``.
    """

    clicked = pyqtSignal()
    toggled = pyqtSignal(bool)

    def __init__(self, text: str = "", checkable: bool = False, parent=None):
        super().__init__(text, parent)
        self.setReadOnly(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursorPosition(0)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        # Allow the button to shrink well below its text width so long names
        # scroll rather than widening the column.
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._checkable = checkable
        self._checked = False
        self._press_pos = None
        self._dragged = False
        self.set_button_style()

    # --- button-like API --------------------------------------------------
    def isCheckable(self) -> bool:
        return self._checkable

    def setCheckable(self, value: bool) -> None:
        self._checkable = bool(value)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, value: bool) -> None:
        value = bool(value)
        if value != self._checked:
            self._checked = value
            self.toggled.emit(value)

    def toggle(self) -> None:
        self.setChecked(not self._checked)

    def set_button_style(self, background: str = None, outline: bool = False, padding: str = "0px 1px") -> None:
        """Compose the tight button look: minimal padding, optional background
        colour and an optional search-highlight outline.
        
        Parameters
        ----------
        background : str, optional
            Background color name or hex value.
        outline : bool
            If True, apply search-highlight outline style.
        padding : str
            CSS padding string (e.g. "0px 0px" for zero padding, "0px 1px" for minimal).
        """
        border = SEARCH_OUTLINE_STYLE if outline else "border: 1px solid #5a5a5a;"
        bg = f"background-color: {background};" if background else ""
        self.setStyleSheet(
            f"QLineEdit {{ border-radius: 3px; padding: {padding}; "
            f"{border} {bg} }}"
        )

    # --- click vs. drag detection -----------------------------------------
    def mousePressEvent(self, event):
        self._press_pos = event.position()
        self._dragged = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is not None:
            delta = event.position() - self._press_pos
            if abs(delta.x()) + abs(delta.y()) > 4:
                self._dragged = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # Only toggle and emit clicked on left-click; right-click is handled separately.
        if (self._press_pos is not None and not self._dragged and 
            event.button() == Qt.MouseButton.LeftButton):
            if self._checkable:
                self.toggle()
            self.clicked.emit()
        self._press_pos = None
        self.setCursorPosition(0)


class SettingsPopup(QWidget):
    """Frameless hover/right-click popup that hosts a compact device widget's
    detailed controls. One instance is created per compact device widget.

    The widget's ``controls_container`` is re-parented into this popup while
    compact mode is active, so there is never any duplicated state.
    """

    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("SettingsPopup")
        self.setStyleSheet(
            "QWidget#SettingsPopup { background-color: #2b2b2b; "
            "border: 1px solid #5a5a5a; border-radius: 4px; }"
        )
        self._vbox = QVBoxLayout(self)
        self._vbox.setContentsMargins(6, 6, 6, 6)
        self._vbox.setSpacing(4)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

    def host(self, content: QWidget) -> None:
        """Re-parent *content* into this popup and show it."""
        self._vbox.addWidget(content)
        content.setVisible(True)

    def take(self, content: QWidget) -> None:
        """Release *content* from this popup's layout (caller re-parents it)."""
        self._vbox.removeWidget(content)

    def show_below(self, anchor: QWidget) -> None:
        self._hide_timer.stop()
        self.adjustSize()
        self.move(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        self.show()
        self.raise_()

    def schedule_hide(self) -> None:
        self._hide_timer.start(POPUP_HIDE_GRACE_MS)

    def cancel_hide(self) -> None:
        self._hide_timer.stop()

    def enterEvent(self, event):  # noqa: N802 (Qt API)
        self.cancel_hide()
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802 (Qt API)
        self.schedule_hide()
        super().leaveEvent(event)


class DeviceWidget(QWidget):
    """Base class for device control widgets"""
    value_changed = pyqtSignal(str, str, dict)
    
    def __init__(self, device_name: str, device_config: Dict[str, Any]):
        super().__init__()
        self.device_name = device_name
        self.device_config = device_config
        self.setFont(QFont("Arial", 9))

        # --- compact-mode state (populated by subclasses in setup_ui) ---
        self._compact = False
        self._search_matched = False
        self.name_button = None            # compact name/on-off button
        self.controls_container = None     # detailed controls (move to popup)
        self._controls_home_layout = None  # inline layout owning the container
        self._popup = None                 # lazily created SettingsPopup
        self._dwell_timer = None           # hover dwell -> open popup
        
    def get_updated_config(self) -> Dict[str, Any]:
        """Return the updated configuration for this device"""
        raise NotImplementedError
        
    def update_from_config(self, config: Dict[str, Any]):
        """Update widget values from configuration"""
        raise NotImplementedError

    def set_search_highlight(self, matched: bool) -> None:
        """Highlight the matched device.

        In expanded mode the device-name label gets a light-blue background.
        In compact mode the name button gets a blue *outline* instead, so the
        state-indicating background colour (green on / DAC gray) stays visible.
        """
        self._search_matched = matched
        if self._compact:
            self._refresh_compact_style()
        else:
            lbl = getattr(self, "device_label", None)
            if lbl is not None:
                lbl.setStyleSheet("QLineEdit { background-color: #1a4f72; color: #e8e8e8; }" if matched else "")

    # --- compact-mode shared helpers -------------------------------------

    def set_compact(self, compact: bool) -> None:
        """Switch this widget between expanded and compact layouts.

        Default implementation only records the flag; subclasses override to
        actually rearrange their widgets.
        """
        self._compact = compact

    def _refresh_compact_style(self) -> None:
        """Recompose the compact button style (state colour + search outline).

        Overridden by subclasses that have a compact name button.
        """

    def _ensure_popup(self) -> "SettingsPopup":
        if self._popup is None:
            self._popup = SettingsPopup()
        return self._popup

    def _wire_name_button_triggers(self) -> None:
        """Open the settings popup on 300 ms hover dwell or right-click."""
        if self.name_button is None:
            return
        self.name_button.installEventFilter(self)
        self._dwell_timer = QTimer(self)
        self._dwell_timer.setSingleShot(True)
        self._dwell_timer.timeout.connect(self._open_settings_popup)

    def _open_settings_popup(self) -> None:
        if not self._compact or self.controls_container is None:
            return
        self._ensure_popup().show_below(self.name_button)

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if obj is self.name_button and self._compact:
            et = event.type()
            if et == QEvent.Type.Enter:
                if self._dwell_timer is not None:
                    self._dwell_timer.start(HOVER_DWELL_MS)
            elif et == QEvent.Type.Leave:
                if self._dwell_timer is not None:
                    self._dwell_timer.stop()
                if self._popup is not None and self._popup.isVisible():
                    self._popup.schedule_hide()
            elif et == QEvent.Type.ContextMenu:
                self._open_settings_popup()
                return True
        return super().eventFilter(obj, event)

class DDSWidget(DeviceWidget):
    """Widget for controlling DDS devices"""
    
    def __init__(self, device_name: str, device_config: Dict[str, Any],
                dds_frame_obj=None,
                step_size_controller=None):
        super().__init__(device_name, device_config)
        self.dds_frame_obj = dds_frame_obj
        self.step_size_controller = step_size_controller  # Reference to shared step size controls
        self.has_unsaved_changes = False
        self.instant_apply = False
        self.device_label = None  # Will store reference to label for tooltip update
        self._force_update_pending = False
        # Store previous values for undo functionality
        self.prev_freq = None
        self.prev_freq_unit = None
        self.prev_amp = None
        self.prev_vpd = None
        self.prev_sw_state = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)  # Add 2px padding around widget
        layout.setSpacing(2)

        # Compact name / on-off button (hidden in expanded mode).  A
        # ScrollableButton so a long name scrolls instead of widening the column.
        self.name_button = ScrollableButton(self.device_name)
        self.name_button.setToolTip(self.device_name)
        self.name_button.setVisible(False)
        self.name_button.clicked.connect(self._on_name_button_clicked)
        layout.addWidget(self.name_button)

        self.device_label = QLineEdit(self.device_name)
        self.device_label.setCursorPosition(0)
        self.device_label.setReadOnly(True)
        self.device_label.setToolTip(self.device_name)
        layout.addWidget(self.device_label)

        # All detailed controls live in a container so they can be moved into a
        # hover/right-click popup in compact mode.
        self.controls_container = QWidget()
        controls_layout = QVBoxLayout(self.controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(2)

        # Frequency controls
        freq_layout = QHBoxLayout()
        # freq_layout.addWidget(QLabel("Frequency:"))
        
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setSingleStep(0.1)
        self.freq_spinbox.setDecimals(4)
        self.freq_spinbox.setValue(self.device_config["frequency"] / 1e6)  # Convert Hz to MHz
        self.freq_spinbox.setMinimum(0.)
        self.freq_spinbox.setMaximum(400.)
        self.freq_spinbox.lineEdit().returnPressed.connect(self.on_update_clicked)
        self.freq_spinbox.valueChanged.connect(self.on_freq_spinbox_value_changed)
        freq_layout.addWidget(self.freq_spinbox)
        
        # Frequency unit selector (MHz/Γ) if transition is not None
        self.freq_unit_combo = QComboBox()
        self.freq_unit_combo.addItem("MHz")
        if self.device_config.get("transition", "None") != "None":
            self.freq_unit_combo.addItem("Γ")
        self.freq_unit_combo.currentTextChanged.connect(self.on_freq_unit_changed)
        
        freq_layout.addWidget(self.freq_unit_combo)
            
        controls_layout.addLayout(freq_layout)
        
        # Amplitude controls
        amp_layout = QHBoxLayout()
        
        self.amp_spinbox = QDoubleSpinBox()
        self.amp_spinbox.setRange(0, 1)
        self.amp_spinbox.setDecimals(3)
        self.amp_spinbox.setSingleStep(0.005)
        self.amp_spinbox.setValue(self.device_config["amplitude"])
        self.amp_spinbox.lineEdit().returnPressed.connect(self.on_update_clicked)
        self.amp_spinbox.valueChanged.connect(self.on_amp_spinbox_value_changed)

        self.vpd_spinbox = QDoubleSpinBox()
        self.vpd_spinbox.setRange(0, 10)
        self.vpd_spinbox.setDecimals(2)
        self.vpd_spinbox.setSingleStep(0.05)
        self.vpd_spinbox.setValue(self.device_config.get("v_pd", 5.0))
        self.vpd_spinbox.lineEdit().returnPressed.connect(self.on_update_clicked)
        self.vpd_spinbox.valueChanged.connect(self.on_vpd_spinbox_value_changed)

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
        controls_layout.addLayout(amp_layout)
        
        # Update button

        self.state_button = QPushButton("Off")
        self.state_button.setCheckable(True)
        self.state_button.toggled.connect(self.on_state_button_toggled)
        state_button_row = QHBoxLayout()
        # state_button_row.addWidget(QLabel("sw state:"))
        state_button_row.addWidget(self.state_button)
        
        self.default_button = QPushButton("default")
        self.default_button.clicked.connect(self.on_default_undo_clicked)
        self.default_button.setStyleSheet(f"background-color: {DEFAULT_BUTTON_COLOR}")
        state_button_row.addWidget(self.default_button)

        controls_layout.addLayout(state_button_row)

        # Inline home for the controls container (expanded mode).
        self._controls_home_layout = layout
        layout.addWidget(self.controls_container)
        
        self.setLayout(layout)
        self.update_from_config(self.device_config)

        self.on_amp_unit_changed(start_unit)
        self._wire_name_button_triggers()

    def _on_name_button_clicked(self):
        """Compact-mode left click toggles the DDS sw state."""
        self.state_button.toggle()

    def set_compact(self, compact: bool) -> None:
        self._compact = compact
        if compact:
            self.device_label.setVisible(False)
            self._controls_home_layout.removeWidget(self.controls_container)
            self._ensure_popup().host(self.controls_container)
            self.name_button.setVisible(True)
            self._refresh_compact_style()
        else:
            if self._popup is not None:
                self._popup.hide()
                self._popup.take(self.controls_container)
            self._controls_home_layout.addWidget(self.controls_container)
            self.controls_container.setVisible(True)
            self.name_button.setVisible(False)
            self.device_label.setVisible(True)
            self.device_label.setStyleSheet(
                "QLineEdit { background-color: #1a4f72; color: #e8e8e8; }"
                if self._search_matched else ""
            )

    def _refresh_compact_style(self) -> None:
        if self.name_button is None:
            return
        self.name_button.setText(self.device_name)
        self.name_button.setCursorPosition(0)
        bg = STATE_BUTTON_ON_COLOR if self.state_button.isChecked() else None
        self.name_button.set_button_style(bg, outline=self._search_matched)
    
    def on_default_undo_clicked(self):
        """Handle default/undo button click"""
        if self.has_unsaved_changes:
            # Undo: restore previous values. Restore the frequency unit first so
            # the spinbox range matches the stored value's unit before writing it
            # (otherwise an MHz value gets clamped into a Γ-ranged spinbox).
            if self.prev_freq_unit is not None:
                self.freq_unit_combo.setCurrentText(self.prev_freq_unit)
            with QSignalBlocker(self.freq_spinbox):
                self.freq_spinbox.setValue(self.prev_freq)
            with QSignalBlocker(self.amp_spinbox):
                self.amp_spinbox.setValue(self.prev_amp)
            with QSignalBlocker(self.vpd_spinbox):
                self.vpd_spinbox.setValue(self.prev_vpd)
            with QSignalBlocker(self.state_button):
                self.state_button.setChecked(self.prev_sw_state)
                # Manually update state button style since signal is blocked
                if self.prev_sw_state:
                    self.state_button.setText("On")
                    self.state_button.setStyleSheet(f"background-color: {STATE_BUTTON_ON_COLOR}")
                else:
                    self.state_button.setText("Off")
                    self.state_button.setStyleSheet("")
            self.has_unsaved_changes = False
            self.update_default_button_state()
        else:
            # Reset to default values — mark as pending until Enter is pressed
            if hasattr(self.dds_frame_obj, self.device_name):
                # Snapshot the pre-default state so undo restores exactly what
                # was on screen (including the current frequency unit) before
                # "default" switched the unit and wrote the default value.
                self.prev_freq_unit = self.freq_unit_combo.currentText()
                self.prev_freq = self.freq_spinbox.value()
                self.prev_amp = self.amp_spinbox.value()
                self.prev_vpd = self.vpd_spinbox.value()
                self.prev_sw_state = self.state_button.isChecked()
                dds = vars(self.dds_frame_obj)[self.device_name]
                # If this channel is defined by detuning (has a transition),
                # show the default in Γ units; otherwise show it in MHz.
                if getattr(dds, "transition", "None") != "None":
                    self.freq_unit_combo.setCurrentText("Γ")
                    self.freq_spinbox.setValue(dds.frequency_to_detuning(dds.frequency))
                else:
                    self.freq_unit_combo.setCurrentText("MHz")
                    self.freq_spinbox.setValue(dds.frequency/1.e6)
                self.amp_spinbox.setValue(dds.amplitude)
                # self.state_button.setChecked(dds.sw_state)
                self.vpd_spinbox.setValue(dds.v_pd)
                # Stage the change: show "undo" and require Enter to confirm,
                # even if a value happens to equal the current one.
                self._force_update_pending = True
                self.has_unsaved_changes = True
                print(f"[GUI] DDS {self.device_name}: Default button pressed - set _force_update_pending=True")
                print(f"[GUI] DDS {self.device_name}: Loading default values: freq={dds.frequency}, amp={dds.amplitude}, v_pd={dds.v_pd}")
                self.update_default_button_state()
                self.freq_spinbox.setFocus()
                self.freq_spinbox.selectAll()
        
    def on_state_button_toggled(self, checked):
        if checked:
            self.state_button.setText("On")
            self.state_button.setStyleSheet(f"background-color: {STATE_BUTTON_ON_COLOR}")
        else:
            self.state_button.setText("Off")
            self.state_button.setStyleSheet("")
        self._refresh_compact_style()
        self.on_update_clicked()
        
    def on_instant_apply_toggled(self, checked):
        """Handle instant apply checkbox toggle"""
        self.instant_apply = checked
        
    def setup_step_sizes(self):
        """Setup step sizes from the shared step size controller"""
        if self.step_size_controller:
            self.freq_spinbox.setSingleStep(self.step_size_controller.freq_step_spinbox.value())
            self.amp_spinbox.setSingleStep(self.step_size_controller.amp_step_spinbox.value())
            self.vpd_spinbox.setSingleStep(self.step_size_controller.vpd_step_spinbox.value())
            self.instant_apply = self.step_size_controller.instant_apply_button.isChecked()
    
    def set_tooltip(self, urukul_idx: int, ch: int):
        """Set tooltip to show device name and urukul/channel"""
        if self.device_label:
            self.device_label.setToolTip(f"{self.device_name}\nurukul{urukul_idx}_ch{ch}")
        
    def on_freq_spinbox_value_changed(self):
        """Handle frequency spinbox value change"""
        self.on_value_changed()
        if self.instant_apply:
            self.on_update_clicked()
            
    def on_amp_spinbox_value_changed(self):
        """Handle amplitude spinbox value change"""
        self.on_value_changed()
        if self.instant_apply:
            self.on_update_clicked()
            
    def on_vpd_spinbox_value_changed(self):
        """Handle VPD spinbox value change"""
        self.on_value_changed()
        if self.instant_apply:
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

    def on_value_changed(self):
        """Mark that values have changed but not yet submitted"""
        self.has_unsaved_changes = True
        self.update_default_button_state()

    def highlight_unsaved(self):
        """Highlight spinboxes orange when they have unsaved changes"""
        if self.has_unsaved_changes:
            self.freq_spinbox.setStyleSheet("QDoubleSpinBox { background-color: orange; }")
            self.amp_spinbox.setStyleSheet("QDoubleSpinBox { background-color: orange; }")
            self.vpd_spinbox.setStyleSheet("QDoubleSpinBox { background-color: orange; }")
        else:
            self.freq_spinbox.setStyleSheet("")
            self.amp_spinbox.setStyleSheet("")
            self.vpd_spinbox.setStyleSheet("")
    
    def update_default_button_state(self):
        """Update button text and style based on unsaved changes"""
        if self.has_unsaved_changes:
            self.default_button.setText("undo")
            self.default_button.setStyleSheet(f"background-color: {UNDO_BUTTON_COLOR}")
        else:
            self.default_button.setText("default")
            self.default_button.setStyleSheet(f"background-color: {DEFAULT_BUTTON_COLOR}")
        self.highlight_unsaved()
            
    def on_update_clicked(self):
        """Handle update button click (triggered by editingFinished)"""
        # Store current values as previous for next undo
        self.prev_freq = self.freq_spinbox.value()
        self.prev_freq_unit = self.freq_unit_combo.currentText()
        self.prev_amp = self.amp_spinbox.value()
        self.prev_vpd = self.vpd_spinbox.value()
        self.prev_sw_state = self.state_button.isChecked()
        self.has_unsaved_changes = False
        self.update_default_button_state()
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

        if self._force_update_pending:
            config["force_update_counter"] = config.get("force_update_counter", 0) + 1
            self._force_update_pending = False
            print(f"[GUI] DDS {self.device_name}: Incremented force_update_counter to {config['force_update_counter']}")
            print(f"[GUI] DDS {self.device_name}: Config = freq={config.get('frequency')}, amp={config.get('amplitude')}, v_pd={config.get('v_pd')}, sw_state={config.get('sw_state')}")

        return config

    def _freq_hz_to_display(self, freq_hz: float) -> float:
        """Convert a frequency in Hz to the value shown in the freq spinbox.

        Honors the current unit selection: returns detuning (Γ) when the unit
        combo is set to "Γ", otherwise MHz.  This prevents writing an MHz value
        into a Γ-ranged spinbox (range [-100, 100]), which would otherwise
        clamp the display to the max and get "stuck" there.
        """
        if self.freq_unit_combo.currentText() == "Γ" and self.dds_frame_obj:
            try:
                uru_idx = self.device_config["urukul_idx"]
                ch = self.device_config["ch"]
                dds_obj = self.dds_frame_obj.dds_array[uru_idx][ch]
                return dds_obj.frequency_to_detuning(freq_hz)
            except Exception:
                pass
        return freq_hz / 1e6

    def update_from_config(self, config: Dict[str, Any]):
        """Update widget values from configuration"""
        self.device_config = config
        self.has_unsaved_changes = False
        self.highlight_unsaved()
        with QSignalBlocker(self.freq_spinbox), QSignalBlocker(self.amp_spinbox), QSignalBlocker(self.vpd_spinbox):
            # Update main spinbox values (respecting the current MHz/Γ unit so
            # a Γ-mode spinbox is never fed an out-of-range MHz value).
            self.freq_spinbox.setValue(self._freq_hz_to_display(config["frequency"]))
            self.amp_spinbox.setValue(config["amplitude"])
            if "v_pd" in config:
                self.vpd_spinbox.setValue(config["v_pd"])
            
            # Store as previous values for undo
            self.prev_freq = self.freq_spinbox.value()
            self.prev_freq_unit = self.freq_unit_combo.currentText()
            self.prev_amp = self.amp_spinbox.value()
            self.prev_vpd = self.vpd_spinbox.value()
            
            # Update step sizes from shared controller
            self.setup_step_sizes()
            
        with QSignalBlocker(self.state_button):
            self.state_button.setChecked(bool(config["sw_state"]))
            self.prev_sw_state = self.state_button.isChecked()
        if config["sw_state"]:
            self.state_button.setText("On")
            self.state_button.setStyleSheet(f"background-color: {STATE_BUTTON_ON_COLOR}")
        else:
            self.state_button.setText("Off")
            self.state_button.setStyleSheet("")
        self._refresh_compact_style()

class DACWidget(DeviceWidget):
    """Widget for controlling DAC devices"""

    def __init__(self, device_name: str, device_config: Dict[str, Any],
                  step_size_controller=None,
                  dac_frame_obj=None):
        super().__init__(device_name, device_config)
        self.dac_frame_obj = dac_frame_obj
        self.step_size_controller = step_size_controller  # Reference to shared step size controls
        self.has_unsaved_changes = False
        self._force_update_pending = False
        # Store previous value for undo functionality
        self.prev_voltage = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)  # Add 2px padding around widget
        layout.setSpacing(2)

        # Compact name button (hidden in expanded mode).  A ScrollableButton so
        # a long name scrolls instead of widening the column.
        self.name_button = ScrollableButton(self.device_name)
        self.name_button.setToolTip(self.device_name)
        self.name_button.setVisible(False)
        self.name_button.clicked.connect(self._on_name_button_clicked)
        layout.addWidget(self.name_button)

        self.device_label = QLineEdit(self.device_name)
        self.device_label.setCursorPosition(0)
        self.device_label.setReadOnly(True)
        self.device_label.setToolTip(self.device_name)
        layout.addWidget(self.device_label)

        # Detailed controls live in a container so they can move to a popup.
        self.controls_container = QWidget()
        controls_layout = QVBoxLayout(self.controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(2)

        # Voltage control
        voltage_layout = QHBoxLayout()
        # voltage_layout.addWidget(QLabel("Voltage:"))
        
        self.voltage_spinbox = QDoubleSpinBox()
        self.voltage_spinbox.setRange(-9.999, 9.999)
        self.voltage_spinbox.setDecimals(3)
        self.voltage_spinbox.setSuffix(" V")
        self.voltage_spinbox.setValue(self.device_config["voltage"])
        self.voltage_spinbox.lineEdit().returnPressed.connect(self.on_update_clicked)
        self.voltage_spinbox.valueChanged.connect(self.on_value_changed)
        voltage_layout.addWidget(self.voltage_spinbox)
        
        self.default_button = QPushButton("default")
        self.default_button.clicked.connect(self.on_default_undo_clicked)
        self.default_button.setStyleSheet(f"background-color: {DEFAULT_BUTTON_COLOR}")
        voltage_layout.addWidget(self.default_button)
        
        controls_layout.addLayout(voltage_layout)

        self._controls_home_layout = layout
        layout.addWidget(self.controls_container)
        
        self.setLayout(layout)
        self._refresh_compact_style()
        self._wire_name_button_triggers()

    def _on_name_button_clicked(self):
        """Compact-mode left click opens the voltage settings popup."""
        self._open_settings_popup()

    def set_compact(self, compact: bool) -> None:
        self._compact = compact
        if compact:
            self.device_label.setVisible(False)
            self._controls_home_layout.removeWidget(self.controls_container)
            self._ensure_popup().host(self.controls_container)
            self.name_button.setVisible(True)
            self._refresh_compact_style()
        else:
            if self._popup is not None:
                self._popup.hide()
                self._popup.take(self.controls_container)
            self._controls_home_layout.addWidget(self.controls_container)
            self.controls_container.setVisible(True)
            self.name_button.setVisible(False)
            self.device_label.setVisible(True)
            self.device_label.setStyleSheet(
                "QLineEdit { background-color: #1a4f72; color: #e8e8e8; }"
                if self._search_matched else ""
            )

    def _refresh_compact_style(self) -> None:
        if self.name_button is None:
            return
        self.name_button.setText(self.device_name)
        self.name_button.setCursorPosition(0)
        bg = DAC_NONZERO_NAME_COLOR if abs(self.voltage_spinbox.value()) > 1e-9 else None
        self.name_button.set_button_style(bg, outline=self._search_matched)

    def on_default_undo_clicked(self):
        """Handle default/undo button click"""
        if self.has_unsaved_changes:
            # Undo: restore previous value
            with QSignalBlocker(self.voltage_spinbox):
                self.voltage_spinbox.setValue(self.prev_voltage)
            self.has_unsaved_changes = False
            self.update_default_button_state()
        else:
            # Reset to default value — mark as pending until Enter is pressed
            if hasattr(self.dac_frame_obj, self.device_name):
                dac = vars(self.dac_frame_obj)[self.device_name]
                self.voltage_spinbox.setValue(dac.v)
                # Stage the change: show "undo" and require Enter to confirm,
                # even if the value happens to equal the current one.
                self._force_update_pending = True
                self.has_unsaved_changes = True
                self.update_default_button_state()
                self.voltage_spinbox.setFocus()
                self.voltage_spinbox.selectAll()

    def on_value_changed(self):
        """Mark that values have changed but not yet submitted"""
        self.has_unsaved_changes = True
        self.update_default_button_state()

    def setup_step_sizes(self):
        """Setup step sizes from the shared step size controller"""
        if self.step_size_controller:
            self.voltage_spinbox.setSingleStep(self.step_size_controller.dac_voltage_step_spinbox.value())

    def highlight_unsaved(self):
        """Highlight spinbox orange when it has unsaved changes"""
        if self.has_unsaved_changes:
            self.voltage_spinbox.setStyleSheet("QDoubleSpinBox { background-color: orange; }")
        else:
            self.voltage_spinbox.setStyleSheet("")
    
    def update_default_button_state(self):
        """Update button text and style based on unsaved changes"""
        if self.has_unsaved_changes:
            self.default_button.setText("undo")
            self.default_button.setStyleSheet(f"background-color: {UNDO_BUTTON_COLOR}")
        else:
            self.default_button.setText("default")
            self.default_button.setStyleSheet(f"background-color: {DEFAULT_BUTTON_COLOR}")
        self.highlight_unsaved()
        
    def on_update_clicked(self):
        """Handle update button click (triggered by editingFinished)"""
        # Store current value as previous for next undo
        self.prev_voltage = self.voltage_spinbox.value()
        self.has_unsaved_changes = False
        self.update_default_button_state()
        updated_config = self.get_updated_config()
        self.value_changed.emit("dac", self.device_name, updated_config)
        
    def get_updated_config(self) -> Dict[str, Any]:
        """Return the updated configuration for this DAC device"""
        config = self.device_config.copy()
        config["voltage"] = self.voltage_spinbox.value()
        if self._force_update_pending:
            config["force_update_counter"] = config.get("force_update_counter", 0) + 1
            self._force_update_pending = False
        return config
        
    def update_from_config(self, config: Dict[str, Any]):
        """Update widget values from configuration"""
        self.device_config = config
        self.has_unsaved_changes = False
        self.highlight_unsaved()
        with QSignalBlocker(self.voltage_spinbox):
            self.voltage_spinbox.setValue(config["voltage"])
            # Store as previous value for undo
            self.prev_voltage = self.voltage_spinbox.value()
            # Update step size from shared controller
            self.setup_step_sizes()
        self._refresh_compact_style()


class TTLWidget(DeviceWidget):
    """Widget for controlling TTL devices"""

    def __init__(self, device_name: str, device_config: Dict[str, Any]):
        super().__init__(device_name, device_config)
        self.device_label = None  # Will store reference to label for tooltip update
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 4)  # Padding above label and below button to group them
        layout.setSpacing(-2)  # Negative spacing to bring button closer to label

        self.device_label = QLineEdit(self.device_name)
        self.device_label.setCursorPosition(0)
        self.device_label.setReadOnly(True)
        self.device_label.setToolTip(self.device_name)
        layout.addWidget(self.device_label)
        
        # State control
        state_layout = QHBoxLayout()
        state_layout.setContentsMargins(0, 0, 0, 0)
        # state_layout.addWidget(QLabel("State:"))
        
        self.state_button = ScrollableButton("Off", checkable=True)
        self.state_button.toggled.connect(self.on_state_button_toggled)
        state_layout.addWidget(self.state_button)
        
        layout.addLayout(state_layout)
        
        self.setLayout(layout)
        self.update_from_config(self.device_config)

    def on_state_button_toggled(self, checked):
        self.state_button.setText(self.device_name if self._compact else ("On" if checked else "Off"))
        # Match DDS OFF color styling: use empty stylesheet for off state
        if not self._compact:
            if checked:
                self.state_button.setStyleSheet(f"background-color: {STATE_BUTTON_ON_COLOR}")
            else:
                self.state_button.setStyleSheet("")
        self._refresh_compact_style()
        self.on_update_clicked()

    def set_compact(self, compact: bool) -> None:
        self._compact = compact
        if compact:
            self.device_label.setVisible(False)
        else:
            self.device_label.setVisible(True)
            self.device_label.setStyleSheet(
                "QLineEdit { background-color: #1a4f72; color: #e8e8e8; }"
                if self._search_matched else ""
            )
        self.state_button.setText(
            self.device_name if compact
            else ("On" if self.state_button.isChecked() else "Off")
        )
        self._refresh_compact_style()

    def _refresh_compact_style(self) -> None:
        """Compose state colour (green on) + search outline on the button.

        The TTL state button *is* the compact name button, so search highlight
        is drawn as an outline only while compact.  Use minimal/negative padding
        to allow the button to fill the grid cell without wasting space.
        TTL on/off colors match DDS: green when on, no background when off.
        """
        bg = STATE_BUTTON_ON_COLOR if self.state_button.isChecked() else None
        outline = self._compact and self._search_matched
        self.state_button.set_button_style(bg, outline=outline, padding="-1px 0px")
        self.state_button.setCursorPosition(0)
        
    def set_tooltip(self, ch: int):
        """Set tooltip to show device name and channel"""
        if self.device_label:
            self.device_label.setToolTip(f"{self.device_name}\nttl{ch}")
        self.state_button.setToolTip(f"{self.device_name}\nttl{ch}")
        
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
        with QSignalBlocker(self.state_button):
            self.state_button.setChecked(bool(config["ttl_state"]))
        self.state_button.setText(
            self.device_name if self._compact
            else ("On" if config["ttl_state"] else "Off")
        )
        self._refresh_compact_style()


class _UpdateSender(QThread):
    """Sends per-device deltas to the monitor server off the GUI thread.

    Edits are coalesced *last-wins per device*: while a send is queued, newer
    changes to the same device merge into the pending payload, so rapid spins
    of a single spinbox collapse to one network round-trip carrying the latest
    value.  The server is the sole writer of the JSON, so this never races with
    other clients.
    """

    ack = pyqtSignal(str, str, dict)        # device_type, device_name, ack
    send_failed = pyqtSignal(str, str)      # device_type, device_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._pending: Dict[tuple, Dict[str, Any]] = {}
        self._running = True
        self._client: MonitorClient | None = None

    def enqueue(self, device_type: str, device_name: str, changes: Dict[str, Any]) -> None:
        with self._cond:
            key = (device_type, device_name)
            if key in self._pending:
                self._pending[key].update(changes)
            else:
                self._pending[key] = dict(changes)
            self._cond.notify()

    def run(self):
        while True:
            with self._cond:
                while self._running and not self._pending:
                    self._cond.wait(0.5)
                if not self._running:
                    return
                key = next(iter(self._pending))
                changes = self._pending.pop(key)
            dtype, name = key
            if self._client is None:
                try:
                    self._client = MonitorClient(discovery_timeout=0.5)
                except Exception:
                    self.send_failed.emit(dtype, name)
                    continue
            ack = self._client.send_update(dtype, name, changes)
            if ack is None:
                # Lost connection — force rediscovery next time.
                self._client = None
                self.send_failed.emit(dtype, name)
            elif ack.get("status") == "ok":
                self.ack.emit(dtype, name, ack)
            else:
                self.send_failed.emit(dtype, name)

    def stop(self):
        with self._cond:
            self._running = False
            self._cond.notify_all()


class _StateRequestWorker(QThread):
    """Fetches the full device-state snapshot from the server (get_state).

    An existing ``MonitorClient`` may be supplied as ``initial_client`` to
    avoid repeated service-discovery overhead.  If none is supplied (or if the
    call fails), a fresh client is constructed so the parent can cache it for
    subsequent requests.  The ``_rediscover`` path inside ``send_message``
    handles server restarts transparently — if the cached client's address
    goes stale, it self-heals after one failed attempt.  Only if all retries
    fail is ``state_failed`` emitted to force the parent to discard the client
    and rebuild from scratch on the next call.
    """

    state_loaded = pyqtSignal(dict)   # {"version": int, "config": dict}
    state_failed = pyqtSignal()
    # Emitted when this worker had to create a new MonitorClient so the parent
    # can cache it for future requests.
    client_ready = pyqtSignal(object)

    def __init__(self, initial_client=None, parent=None):
        super().__init__(parent)
        self._initial_client = initial_client

    def run(self):
        client = self._initial_client
        created_new = False
        if client is None:
            try:
                client = MonitorClient(discovery_timeout=0.5)
                created_new = True
            except Exception:
                self.state_failed.emit()
                return
        state = client.get_state()
        if state and state.get("status") == "ok":
            if created_new:
                # Let the parent cache this client before state_loaded fires.
                self.client_ready.emit(client)
            self.state_loaded.emit({
                "version": state.get("version"),
                "config": state.get("config", {}) or {},
            })
        else:
            # Signal failure so the parent discards the cached client and
            # forces a fresh discovery (+ construction) on the next request.
            self.state_failed.emit()


class _ResetWorker(QThread):
    """Sends reset message to monitor server off the GUI thread."""

    succeeded = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, monitor_client, parent=None):
        super().__init__(parent)
        self._client = monitor_client

    def run(self):
        try:
            self._client.send_reset()
            self.succeeded.emit()
        except Exception as e:
            self.failed.emit(str(e))


class MonitorStatusChecker(QThread):
    """Thread that periodically checks the monitor server status"""
    status_updated = pyqtSignal(int)
    connection_failed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.monitor_client: MonitorClient | None = None
        self.status_checker = None
        self.running = True
        self.retry_connection = False

    def run(self):
        while self.running:
            if self.monitor_client is None:
                try:
                    self.monitor_client = MonitorClient(discovery_timeout=0.5)
                except RuntimeError:
                    # Discovery failed: surface via signal only (red
                    # indicator); do not spam stdout.
                    self.connection_failed.emit()
                    time.sleep(2.0)
                    continue
            try:
                status = self.monitor_client.check_status()
                if status is not None:
                    self.status_updated.emit(int(status))
                    time.sleep(0.5)
                else:
                    # send_message() swallows exceptions and returns None on
                    # failure — treat this as a lost connection and force
                    # re-discovery on the next iteration.
                    self.monitor_client = None
                    self.connection_failed.emit()
                    time.sleep(2.0)
            except Exception:
                # Reset client so the next iteration re-runs service discovery.
                # This handles server restarts (new dynamic port) cleanly.
                self.monitor_client = None
                self.connection_failed.emit()
                time.sleep(2.0)
            
    def stop(self):
        self.running = False
        
    def retry(self):
        self.retry_connection = True

class DeviceStateGUI(QMainWindow):
    """Main GUI application for device state management"""
    
    def __init__(self,
                  dds_frame=None,
                  dac_frame=None):
        super().__init__()
        self.config_data = {}
        self.device_widgets = {}
        
        self.dds_frame_obj = dds_frame
        self.dac_frame_obj = dac_frame

        self.connection_failed = False

        # Server-pushed state tracking.  ``_version`` is the last device-state
        # version we have applied; ``_pending`` maps (device_type, device_name)
        # to the timestamp of a local edit awaiting the server's echo (used to
        # avoid clobbering an in-flight edit with an incoming broadcast).
        self._version = None
        self._pending: Dict[tuple, float] = {}
        # Cached MonitorClient reused across state-request calls to avoid
        # repeated service-discovery overhead.  Cleared on failure so the next
        # request triggers a fresh discovery (handles server restarts).
        self._state_client: "MonitorClient | None" = None

        # Compact layout is suppressed; always expanded (never auto-collapse).
        self._compact_mode = False
        self._compact_override = False  # Force expanded mode always
        self._in_recompute = False
        self._scroll_viewport = None  # discovered on first showEvent

        self.setup_ui()
        self._setup_update_sender()
        self._setup_state_listener()
        self.request_state()        # initial async snapshot load
        self.setup_timer()          # periodic safety reconcile
        self.setup_status_checker()
        self.running = False
        
    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("Device State Control")
        self.setGeometry(100, 100, 1200, 800)
        self._set_window_icon()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget_layout = QVBoxLayout()
        
        # Create status button at the top
        self.status_button = QPushButton("Trying to connect to monitor server...")
        self.status_button.clicked.connect(self.on_status_button_clicked)
        font = QFont()
        font.setPointSize(14)
        font.setBold(False)
        self.status_button.setFont(font)
        self.status_button.setMinimumHeight(25)
        # self.status_button.setFixedWidth(1000)

        # Compact-mode toggle is suppressed; always expanded.
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(self.status_button, 1)
        central_widget_layout.addLayout(status_row)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        central_widget_layout.addWidget(self.tab_widget)
        central_widget.setLayout(central_widget_layout)

        # Shared search bar placed inline with the tab bar (corner widget).
        # Content is preserved when switching tabs; filtering is re-applied
        # whenever the text changes OR the active tab changes.
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search channels  (Ctrl+F)")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.setMinimumWidth(220)
        self.search_bar.setMaximumHeight(22)
        self.search_bar.textChanged.connect(self._apply_active_search)
        self.tab_widget.setCornerWidget(self.search_bar, Qt.Corner.TopRightCorner)
        self.tab_widget.currentChanged.connect(
            lambda _: self._apply_active_search(self.search_bar.text())
        )
        
        # Create tabs
        self.dds_tab = QWidget()
        self.dac_tab = QWidget()
        self.ttl_tab = QWidget()
        
        self.tab_widget.addTab(self.dds_tab, "DDS")
        self.tab_widget.addTab(self.dac_tab, "DAC")
        self.tab_widget.addTab(self.ttl_tab, "TTL")

        # Setup DDS tab with step size controls at top
        dds_tab_layout = QVBoxLayout()
        dds_tab_layout.setContentsMargins(8, 8, 8, 8)  # Increased padding around the tab
        dds_tab_layout.setSpacing(12)  # Increased space between step controls and device grid
        
        # Step size controls panel for DDS
        dds_step_layout = QHBoxLayout()
        dds_step_layout.setContentsMargins(0, 0, 0, 0)
        dds_step_layout.setSpacing(5)
        
        # Bolded title
        title_label = QLabel("Step Settings")
        title_font = title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(9)
        title_label.setFont(title_font)
        dds_step_layout.addWidget(title_label)
        
        dds_step_layout.addSpacing(10)
        dds_step_layout.addWidget(QLabel("Freq:"))
        
        self.freq_step_spinbox = QDoubleSpinBox()
        self.freq_step_spinbox.setRange(0.001, 100)
        self.freq_step_spinbox.setDecimals(3)
        self.freq_step_spinbox.setSingleStep(0.01)
        self.freq_step_spinbox.setValue(0.1)
        self.freq_step_spinbox.setSuffix(" MHz")
        self.freq_step_spinbox.setMaximumHeight(20)
        dds_step_layout.addWidget(self.freq_step_spinbox)
        
        dds_step_layout.addSpacing(10)
        dds_step_layout.addWidget(QLabel("Amp:"))
        
        self.amp_step_spinbox = QDoubleSpinBox()
        self.amp_step_spinbox.setRange(0.001, 1)
        self.amp_step_spinbox.setDecimals(3)
        self.amp_step_spinbox.setSingleStep(0.001)
        self.amp_step_spinbox.setValue(0.005)
        self.amp_step_spinbox.setMaximumHeight(20)
        dds_step_layout.addWidget(self.amp_step_spinbox)
        
        dds_step_layout.addSpacing(10)
        dds_step_layout.addWidget(QLabel("V:"))
        
        self.vpd_step_spinbox = QDoubleSpinBox()
        self.vpd_step_spinbox.setRange(0.001, 10)
        self.vpd_step_spinbox.setDecimals(3)
        self.vpd_step_spinbox.setSingleStep(0.01)
        self.vpd_step_spinbox.setValue(0.05)
        self.vpd_step_spinbox.setSuffix(" V")
        self.vpd_step_spinbox.setMaximumHeight(20)
        dds_step_layout.addWidget(self.vpd_step_spinbox)
        
        dds_step_layout.addStretch()
        
        self.instant_apply_button = QPushButton("turn on instant apply")
        self.instant_apply_button.setCheckable(True)
        self.instant_apply_button.setChecked(False)
        self.instant_apply_button.setStyleSheet("background-color: orange; color: black; font-weight: bold;")
        self.instant_apply_button.setMaximumHeight(20)
        self.instant_apply_button.toggled.connect(self.on_instant_apply_toggled)
        dds_step_layout.addWidget(self.instant_apply_button)

        dds_tab_layout.addLayout(dds_step_layout)

        # DDS devices grid layout (wrapped in container for border)
        self.dds_container = QWidget()
        self.dds_layout = QGridLayout()
        self.dds_layout.setHorizontalSpacing(2)
        self.dds_layout.setContentsMargins(8, 8, 8, 8)  # Increased internal padding
        self.dds_container.setLayout(self.dds_layout)
        self.dds_container.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
        dds_tab_layout.addWidget(self.dds_container)
        self.dds_tab.setLayout(dds_tab_layout)
        
        # Connect DDS step size controls to update all DDS widgets
        self.freq_step_spinbox.valueChanged.connect(self.on_dds_step_size_changed)
        self.amp_step_spinbox.valueChanged.connect(self.on_dds_step_size_changed)
        self.vpd_step_spinbox.valueChanged.connect(self.on_dds_step_size_changed)
        
        # Setup DAC tab with step size controls at top
        dac_tab_layout = QVBoxLayout()
        dac_tab_layout.setContentsMargins(8, 8, 8, 8)  # Increased padding around the tab
        dac_tab_layout.setSpacing(12)  # Increased space between step controls and device grid
        
        # Step size controls panel for DAC
        dac_step_layout = QHBoxLayout()
        dac_step_layout.setContentsMargins(0, 0, 0, 0)
        dac_step_layout.setSpacing(5)
        
        # Bolded title
        dac_title_label = QLabel("Step Settings")
        dac_title_font = dac_title_label.font()
        dac_title_font.setBold(True)
        dac_title_font.setPointSize(9)
        dac_title_label.setFont(dac_title_font)
        dac_step_layout.addWidget(dac_title_label)
        
        dac_step_layout.addSpacing(10)
        dac_step_layout.addWidget(QLabel("Voltage:"))
        
        self.dac_voltage_step_spinbox = QDoubleSpinBox()
        self.dac_voltage_step_spinbox.setRange(0.001, 9.999)
        self.dac_voltage_step_spinbox.setDecimals(3)
        self.dac_voltage_step_spinbox.setSingleStep(0.001)
        self.dac_voltage_step_spinbox.setValue(0.01)
        self.dac_voltage_step_spinbox.setSuffix(" V")
        self.dac_voltage_step_spinbox.setMaximumHeight(20)
        dac_step_layout.addWidget(self.dac_voltage_step_spinbox)

        dac_tab_layout.addLayout(dac_step_layout)

        # DAC devices grid layout (wrapped in container for border)
        self.dac_container = QWidget()
        self.dac_layout = QGridLayout()
        self.dac_layout.setHorizontalSpacing(2)
        self.dac_layout.setContentsMargins(8, 8, 8, 8)  # Increased internal padding
        self.dac_container.setLayout(self.dac_layout)
        self.dac_container.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
        dac_tab_layout.addWidget(self.dac_container)
        self.dac_tab.setLayout(dac_tab_layout)
        
        # Connect DAC step size controls to update all DAC widgets
        self.dac_voltage_step_spinbox.valueChanged.connect(self.on_dac_step_size_changed)
        
        # Setup TTL tab
        ttl_tab_layout = QVBoxLayout()
        ttl_tab_layout.setContentsMargins(8, 8, 8, 8)  # Increased padding around the tab
        ttl_tab_layout.setSpacing(12)  # Increased space for consistency with other tabs

        # TTL devices grid layout (wrapped in container for border)
        self.ttl_container = QWidget()
        self.ttl_layout = QGridLayout()
        self.ttl_layout.setHorizontalSpacing(1)
        self.ttl_layout.setVerticalSpacing(0)  # No vertical padding between rows
        self.ttl_layout.setContentsMargins(8, 8, 8, 8)  # Increased internal padding
        self.ttl_container.setLayout(self.ttl_layout)
        self.ttl_container.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
        ttl_tab_layout.addWidget(self.ttl_container)
        self.ttl_tab.setLayout(ttl_tab_layout)

        # Ctrl+F focuses the shared search bar.
        # Must be on the central widget (not self) so the shortcut fires when
        # DeviceStateGUI is embedded inside a dashboard panel (the QMainWindow
        # itself is hidden by embed_main_window; shortcuts on hidden widgets
        # do not fire).
        from PyQt6.QtGui import QKeySequence, QShortcut  # noqa: PLC0415
        _ctrlf = QShortcut(QKeySequence("Ctrl+F"), central_widget)
        _ctrlf.setContext(Qt.ShortcutContext.WindowShortcut)
        _ctrlf.activated.connect(self._focus_active_search_bar)

        # Ctrl+Tab / Ctrl+Shift+Tab cycle between the DDS/DAC/TTL tabs.
        # Bound on the central widget with WindowShortcut context (same reason
        # as Ctrl+F above) so they fire when the panel is docked, floated, or
        # popped out of the dashboard. Explicit shortcuts are needed because
        # QTabWidget's built-in Ctrl+Tab handling only works while the tab bar
        # itself has focus, which it rarely does inside an embedded panel.
        _next_tab = QShortcut(QKeySequence("Ctrl+Tab"), central_widget)
        _next_tab.setContext(Qt.ShortcutContext.WindowShortcut)
        _next_tab.activated.connect(lambda: self._cycle_tab(1))
        _prev_tab = QShortcut(QKeySequence("Ctrl+Shift+Tab"), central_widget)
        _prev_tab.setContext(Qt.ShortcutContext.WindowShortcut)
        _prev_tab.activated.connect(lambda: self._cycle_tab(-1))

    def _cycle_tab(self, step: int) -> None:
        """Advance the active tab by *step* (wraps around)."""
        count = self.tab_widget.count()
        if count == 0:
            return
        self.tab_widget.setCurrentIndex(
            (self.tab_widget.currentIndex() + step) % count
        )

    def _set_window_icon(self):
        """Set a game-controller emoji icon for the window and taskbar."""
        icon = QIcon()

        for size in (16, 24, 32, 48, 64, 128):
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            font = QFont("Segoe UI Emoji")
            font.setPixelSize(int(size * 0.8))
            painter.setFont(font)
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🎮")
            painter.end()

            icon.addPixmap(pixmap)

        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def setup_timer(self):
        """Periodic safety reconcile against the server.

        UDP broadcasts are the primary update path; this slow timer just
        catches any missed datagram by re-fetching the snapshot.  It runs at
        a relaxed cadence so it never causes UI churn.
        """
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_config_changes)
        self.timer.start(1000)  # reconcile every 1 second

    def _setup_update_sender(self):
        """Start the background sender that pushes deltas to the server."""
        self._update_sender = _UpdateSender(self)
        self._update_sender.ack.connect(self._on_update_ack)
        self._update_sender.send_failed.connect(self._on_update_failed)
        self._update_sender.start()

    def _setup_state_listener(self):
        """Start the UDP listener that receives server state broadcasts."""
        self._state_listener = StateListener(parent=self)
        self._state_listener.state_received.connect(self._on_state_broadcast)
        self._state_listener.start()
        
    def setup_status_checker(self):
        """Setup the status checker thread"""
        self.status_checker = MonitorStatusChecker()
        self.status_checker.status_updated.connect(self.update_status_button)
        self.status_checker.connection_failed.connect(self.on_connection_failed)
        self.status_checker.start()
        
    def on_status_button_clicked(self):
        """Handle status button click"""
        if self.connection_failed:
            # Retry connection
            self.status_button.setText("Connecting...")
            self.status_button.setStyleSheet("background-color: gray; color: white;")
            self.connection_failed = False
            self.status_checker.retry()
        else:
            # Normal reset operation
            reply = QMessageBox.question(
                self, 
                'Reset Server',
                "Are you sure you want to send a reset message to the server?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                client = getattr(self.status_checker, 'monitor_client', None)
                if client is None:
                    QMessageBox.warning(self, "Not Connected",
                        "Monitor client is not yet connected.")
                    return
                self.status_button.setText("Sending reset…")
                self.status_button.setEnabled(False)
                worker = _ResetWorker(client, self)
                worker.succeeded.connect(self._on_reset_succeeded)
                worker.failed.connect(self._on_reset_failed)
                worker.finished.connect(worker.deleteLater)
                worker.start()
                self._reset_worker = worker
                
    def _on_reset_succeeded(self) -> None:
        self.status_button.setEnabled(True)

    def _on_reset_failed(self, msg: str) -> None:
        self.status_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to send reset message: {msg}")

    def update_status_button(self, status):
        """Update the status button based on the server status"""
        self.connection_failed = False
        if status == STATES.READY:
            self.status_button.setText("monitor ready")
            self.status_button.setStyleSheet("background-color: green; color: white;")
        elif status == STATES.LOADING:
            self.status_button.setText("monitor getting ready...")
            self.status_button.setStyleSheet("background-color: orange; color: white;")
        else:  # STATES.NOT_READY
            self.status_button.setText("monitor not ready - click to restart")
            self.status_button.setStyleSheet("background-color: #c46666; color: white;")
            
    def on_connection_failed(self):
        """Handle connection failure"""
        self.connection_failed = True
        self.status_button.setText("Server connection failed")
        self.status_button.setStyleSheet("background-color: #b22222; color: white;")
        
    def on_instant_apply_toggled(self, checked):
        """Handle instant apply button toggle and update color and text"""
        if checked:
            self.instant_apply_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
            self.instant_apply_button.setText("turn off instant apply")
        else:
            self.instant_apply_button.setStyleSheet("background-color: orange; color: black; font-weight: bold;")
            self.instant_apply_button.setText("turn on instant apply")
        self.on_dds_step_size_changed()
        
    def on_instant_apply_toggled_dac(self, checked):
        """Handle DAC instant apply (placeholder for future use)"""
        pass
    
    def on_dac_step_size_changed(self):
        """Update all DAC widgets when step size changes"""
        for widget_key, widget in self.device_widgets.items():
            if widget_key.startswith("dac."):
                widget.setup_step_sizes()
    
    def on_dds_step_size_changed(self):
        """Update all DDS widgets when step size or instant apply changes"""
        for widget_key, widget in self.device_widgets.items():
            if widget_key.startswith("dds."):
                widget.setup_step_sizes()

    # ------------------------------------------------------------------
    # Channel search
    # ------------------------------------------------------------------

    def _focus_active_search_bar(self) -> None:
        """Ctrl+F: focus the shared search bar and select all text."""
        self.search_bar.setFocus()
        self.search_bar.selectAll()

    def _apply_search(self, query: str, device_prefix: str) -> None:
        """Highlight device labels that match *query* for the given prefix (dds/dac/ttl)."""
        terms = parse_name_search_terms(query)
        prefix = device_prefix + "."
        for key, widget in self.device_widgets.items():
            if not key.startswith(prefix):
                continue
            device_name = key[len(prefix):]
            widget.set_search_highlight(bool(terms) and name_matches_all_terms(device_name, terms))

    def _apply_active_search(self, query: str) -> None:
        """Apply *query* to the currently visible tab's devices only."""
        prefixes = ["dds", "dac", "ttl"]
        idx = self.tab_widget.currentIndex()
        if 0 <= idx < len(prefixes):
            self._apply_search(query, prefixes[idx])

    def request_state(self):
        """Fetch the full device-state snapshot from the server (async)."""
        if getattr(self, '_state_req_in_progress', False):
            return
        self._state_req_in_progress = True
        worker = _StateRequestWorker(initial_client=self._state_client, parent=self)
        worker.state_loaded.connect(self._on_state_loaded)
        worker.state_failed.connect(self._on_state_failed)
        worker.state_failed.connect(self._on_state_client_failed)
        worker.client_ready.connect(self._on_state_client_ready)
        worker.finished.connect(lambda: setattr(self, '_state_req_in_progress', False))
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._state_req_worker = worker  # keep alive until finished

    def _on_state_loaded(self, state: dict) -> None:
        """Apply a freshly fetched snapshot on the main thread."""
        self._version = state.get("version")
        new_config = state.get("config", {}) or {}
        if not self.device_widgets:
            # First load → build everything.
            self.config_data = new_config
            self.update_device_widgets()
        else:
            # Reconcile incrementally so we never clobber busy widgets.
            self._reconcile_config(new_config)

    def _on_state_failed(self) -> None:
        """Snapshot fetch failed — surface as a connection problem."""
        self.on_connection_failed()

    def _on_state_client_ready(self, client) -> None:
        """Cache a MonitorClient created by a state-request worker."""
        self._state_client = client

    def _on_state_client_failed(self) -> None:
        """Discard the cached MonitorClient so the next request rebuilds it.

        ``send_message`` already attempts ``_rediscover()`` internally, so
        transient server restarts are handled transparently without reaching
        here.  This path fires only when all retries are exhausted, meaning
        the server is truly unreachable and the cached address is stale.
        """
        self._state_client = None

    def _device_keys(self, config: dict) -> set:
        keys = set()
        for dtype in ("dds", "dac", "ttl"):
            for name in config.get(dtype, {}):
                keys.add(f"{dtype}.{name}")
        return keys

    def _reconcile_config(self, new_config: dict) -> None:
        """Update widgets to match *new_config*, skipping busy widgets."""
        if self._device_keys(new_config) != self._device_keys(self.config_data):
            # Device set changed (rare) → full rebuild.
            self.config_data = new_config
            self.update_device_widgets()
            return
        for dtype in ("dds", "dac", "ttl"):
            section = self.config_data.setdefault(dtype, {})
            for name, cfg in new_config.get(dtype, {}).items():
                if cfg == section.get(name):
                    continue
                section[name] = cfg
                key = (dtype, name)
                widget = self.device_widgets.get(f"{dtype}.{name}")
                if widget is not None and not self._widget_busy(widget, key):
                    widget.update_from_config(cfg)

    def _on_state_broadcast(self, payload: dict) -> None:
        """Handle a UDP ``state_update`` pushed by the server."""
        if payload.get("type") != "state_update":
            return
        version = payload.get("version")
        if version is None:
            return
        if self._version is not None and version <= self._version:
            # Already applied (covers the echo of our own update).
            return
        expected = None if self._version is None else self._version + 1
        if expected is not None and version != expected:
            # Missed one or more broadcasts → full resync over TCP.
            self.request_state()
            return
        self._version = version
        dtype = payload.get("device_type")
        name = payload.get("device_name")
        changes = payload.get("changes", {}) or {}
        self._apply_single_change(dtype, name, changes)

    def _apply_single_change(self, dtype: str, name: str, changes: dict) -> None:
        section = self.config_data.setdefault(dtype, {})
        dev = section.setdefault(name, {})
        dev.update(changes)
        widget = self.device_widgets.get(f"{dtype}.{name}")
        if widget is None:
            # Unknown device → rebuild so a widget gets created.
            self.update_device_widgets()
            return
        if self._widget_busy(widget, (dtype, name)):
            return
        widget.update_from_config(dev)

    @staticmethod
    def _is_descendant(child, parent) -> bool:
        w = child
        while w is not None:
            if w is parent:
                return True
            w = w.parentWidget()
        return False

    def _widget_busy(self, widget, key: tuple) -> bool:
        """True if a widget should not be overwritten by an incoming update."""
        ts = self._pending.get(key)
        if ts is not None:
            if time.time() - ts < 5.0:
                return True
            # Stale pending (ack/echo never arrived) → stop blocking updates.
            self._pending.pop(key, None)
        if getattr(widget, "has_unsaved_changes", False):
            return True
        fw = QApplication.focusWidget()
        if fw is not None and self._is_descendant(fw, widget):
            return True
        return False

    def check_config_changes(self):
        """Safety reconcile (called by the slow timer)."""
        self.request_state()
        
    def update_device_widgets(self):
        """Update device widgets based on current configuration"""
        # Clear existing widgets
        self.clear_layouts()
        self.device_widgets.clear()
        
        # Add DDS widgets organized by urukul_idx (columns) and ch (rows)
        if "dds" in self.config_data:
            # Collect all DDS devices
            for device_name, device_config in self.config_data["dds"].items():
                # Add urukul_idx and ch to config for DDS widgets
                if "urukul_idx" not in device_config:
                    device_config["urukul_idx"] = device_config.get("urukul_idx", 0)
                if "ch" not in device_config:
                    device_config["ch"] = device_config.get("ch", 0)
                
                urukul_idx = device_config["urukul_idx"]
                ch = device_config["ch"]
                
                widget = DDSWidget(device_name, device_config, self.dds_frame_obj, self)
                widget.value_changed.connect(self.on_device_value_changed)
                widget.setup_step_sizes()
                widget.set_tooltip(urukul_idx, ch)
                
                # Position by urukul_idx (column) and ch (row)
                self.dds_layout.addWidget(widget, ch, urukul_idx)
                self.device_widgets[f"dds.{device_name}"] = widget
                    
        # Add DAC widgets grouped into columns of 8
        if "dac" in self.config_data:
            for device_name, device_config in self.config_data["dac"].items():
                widget = DACWidget(device_name, device_config, step_size_controller=self, dac_frame_obj=self.dac_frame_obj)
                widget.value_changed.connect(self.on_device_value_changed)
                widget.setup_step_sizes()
                
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
                    widget.set_tooltip(ch)
                    widget_grid[col][row] = widget
                    self.device_widgets[f"ttl.{device_name}"] = widget

                
            # Add widgets to layout
            for col_idx, col_widgets in enumerate(widget_grid):
                for row_idx, widget in enumerate(col_widgets):
                    self.ttl_layout.addWidget(widget, row_idx, col_idx)

        # Apply the current compact mode to every freshly built widget.
        for widget in self.device_widgets.values():
            widget.set_compact(self._compact_mode)

        self.adjust_window_width()

        # Re-apply active search so highlights survive config reloads.
        self._apply_active_search(self.search_bar.text())

        # Re-evaluate compact mode now that the column count is known.
        self._recompute_compact()

    def closeEvent(self, event):
        """Handle window close event"""
        if self.status_checker:
            self.status_checker.stop()
            self.status_checker.wait()
        listener = getattr(self, "_state_listener", None)
        if listener is not None:
            listener.stop()
            listener.wait()
        sender = getattr(self, "_update_sender", None)
        if sender is not None:
            sender.stop()
            sender.wait()
        event.accept()
                    
        # Adjust window width based on the number of columns
        self.adjust_window_width()
        
    def adjust_window_width(self):
        """Adjust the window width based on the tab with the most columns."""
        max_columns = 0
        for layout in [self.dds_layout, self.dac_layout, self.ttl_layout]:
            max_columns = max(max_columns, layout.columnCount())
        
        if max_columns > 0:
            px = COMPACT_PX_WIDTH_PER_COLUMN if self._compact_mode else PX_WIDTH_PER_COLUMN
            # Add a buffer column for aesthetics
            new_width = (max_columns + 1) * px
            self.resize(new_width, self.height())

    # --- compact / responsive mode --------------------------------------

    def _expanded_required_width(self) -> int:
        """Pixel width the expanded layout needs for its widest tab."""
        max_columns = 1
        for layout in [self.dds_layout, self.dac_layout, self.ttl_layout]:
            max_columns = max(max_columns, layout.columnCount())
        return max_columns * PX_WIDTH_PER_COLUMN

    def _available_width(self) -> int:
        """Width available to the layout.

        When embedded in a dashboard the GUI lives inside a ``QScrollArea``;
        we measure the *viewport* width so we can collapse BEFORE the inner
        widget would overflow and trigger horizontal scrollbars.
        """
        from PyQt6.QtWidgets import QScrollArea  # noqa: PLC0415
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QScrollArea):
                return w.viewport().width()
            w = w.parentWidget()
        cw = self.centralWidget()
        return cw.width() if cw is not None else self.width()

    def _recompute_compact(self):
        """Decide compact vs expanded from override or available width."""
        if self._in_recompute:
            return
        if self._compact_override is not None:
            target = self._compact_override
        else:
            avail = self._available_width()
            if avail <= 0:
                return
            required = self._expanded_required_width()
            if self._compact_mode:
                # Leave compact only once there is comfortably enough room.
                target = not (avail >= required + COMPACT_HYSTERESIS_PX)
            else:
                target = avail < required
        if target != self._compact_mode:
            self.set_compact_mode(target)

    def set_compact_mode(self, compact: bool):
        """Switch every device widget into/out of compact layout."""
        self._in_recompute = True
        try:
            self._compact_mode = compact
            for widget in self.device_widgets.values():
                widget.set_compact(compact)
            # Tighten column spacing in compact mode; restore to normal in expanded.
            # TTL channels use tighter spacing even in expanded mode (1 instead of 2).
            if compact:
                spacing = 0
            else:
                spacing = 2
            self.dds_layout.setHorizontalSpacing(spacing * 2)  # Double the spacing
            self.dac_layout.setHorizontalSpacing(spacing * 2)  # Double the spacing
            self.ttl_layout.setHorizontalSpacing((spacing * 2) if compact else 2)  # Double the spacing
            self.ttl_layout.setVerticalSpacing(0)  # Keep TTL vertical spacing tight always
            # Show box outline only in expanded view
            border_style = "border: none;" if compact else "border: 1px solid #555; border-radius: 4px;"
            self.dds_container.setStyleSheet(border_style)
            self.dac_container.setStyleSheet(border_style)
            self.ttl_container.setStyleSheet(border_style)
            self.adjust_window_width()
        finally:
            self._in_recompute = False

    def showEvent(self, event):  # noqa: N802 (Qt API)
        super().showEvent(event)
        # Discover the enclosing scroll viewport (embedded case) once and watch
        # it for resizes so we collapse based on the available viewport width.
        if self._scroll_viewport is None:
            from PyQt6.QtWidgets import QScrollArea  # noqa: PLC0415
            w = self.parentWidget()
            while w is not None:
                if isinstance(w, QScrollArea):
                    self._scroll_viewport = w.viewport()
                    self._scroll_viewport.installEventFilter(self)
                    break
                w = w.parentWidget()
        self._recompute_compact()

    def resizeEvent(self, event):  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._recompute_compact()

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if obj is self._scroll_viewport and event.type() == QEvent.Type.Resize:
            self._recompute_compact()
        return super().eventFilter(obj, event)
        
    def clear_layouts(self):
        """Clear all device widgets from layouts"""
        for layout in [self.dds_layout, self.dac_layout, self.ttl_layout]:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                    
    def on_device_value_changed(self, device_type: str, device_name: str, updated_config: Dict[str, Any]):
        """Handle a local device edit: update optimistically and push to server."""
        # Optimistic local update so the UI stays responsive even if the
        # network round-trip is slow.
        section = self.config_data.setdefault(device_type, {})
        if device_name in section:
            section[device_name].update(updated_config)
        else:
            section[device_name] = dict(updated_config)

        # Mark this device as having an in-flight edit so an incoming broadcast
        # (including our own echo) does not clobber the spinbox mid-interaction.
        self._pending[(device_type, device_name)] = time.time()

       if device_type == "dds" and "force_update_counter" in updated_config:
           print(f"[GUI] on_device_value_changed: {device_type} {device_name}, force_update_counter={updated_config['force_update_counter']}")

       # Hand off to the background sender (coalesces rapid same-device edits).
       self._update_sender.enqueue(device_type, device_name, updated_config)

    def _on_update_ack(self, device_type: str, device_name: str, ack: dict) -> None:
        """Server accepted our delta."""
        self._version = ack.get("version", self._version)
        self._pending.pop((device_type, device_name), None)

    def _on_update_failed(self, device_type: str, device_name: str) -> None:
        """Server unreachable / rejected our delta.

        Keep the optimistic local value (so the user's intent is preserved on
        screen) and flag the connection problem with a red status indicator.
        """
        self._pending.pop((device_type, device_name), None)
        self.on_connection_failed()
