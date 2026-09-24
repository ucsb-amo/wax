import logging
import threading
from collections import deque
from typing import Dict, Any
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox, QPushButton,
    QLineEdit, QMessageBox, QSizePolicy, QMenu, QListWidget,
    QScrollArea, QGraphicsOpacityEffect
)
from PyQt6.QtCore import QTimer, pyqtSignal, QThread, QSignalBlocker, QSettings, QByteArray
from PyQt6.QtGui import QFont, QIcon, QPainter, QPixmap, QColor, QKeySequence, QShortcut

from PyQt6.QtCore import Qt

import time

from waxx.util.comms_server.comm_client import MonitorClient
from waxx.util.comms_server.comm_server import STATES
from waxx.util.comms_server.state_broadcast import StateListener
from waxx.util.dashboard import theme
from waxa.helper.name_search import (
    parse_name_search_terms,
    name_matches_all_terms,
)

_LOG = logging.getLogger("waxx.device_control")

PX_WIDTH_PER_COLUMN = 100
STATE_BUTTON_ON_COLOR = theme.OK
DEFAULT_BUTTON_COLOR = theme.BG_BUTTON
UNDO_BUTTON_COLOR = theme.PENDING
SEARCH_LABEL_BG = "#1a4f72"          # dark blue label background for search matches
UNREACHABLE_COLOR = "#7a1616"        # darker than theme.ERR: the server itself is gone
SEARCH_DIM_OPACITY = 0.4             # non-matching widgets while a search is active
DAC_TINT_MAX_ALPHA = 0.35            # label tint at |voltage| = 10 V
DAC_TINT_FULL_SCALE_V = 10.0
CHANGES_LOG_MAX_ROWS = 500           # ring buffer behind the pop-out changes log
GRID_SPACING = 4                     # px between channel cards in the DDS/DAC/TTL grids
CHANNELS_PER_COLUMN = 8              # DAC / TTL channels per grid column

# --- TTL pulse mode -------------------------------------------------------
# The Monitor kernel re-reads the device-state JSON once per loop iteration
# (``waxx.base.monitor.T_MONITOR_UPDATE_INTERVAL``, 0.1 s plus RPC time).  A
# pulse must leave "on" in the JSON long enough for at least one of those reads
# to see it before "off" replaces it.  Two guarantees are stacked:
#   1. "off" is only sent after the server has *acked* "on" (the server writes
#      the JSON atomically before it replies), so the two edits can never be
#      coalesced by the sender or reordered.
#   2. after that ack, "on" is held for TTL_PULSE_HOLD_S — three monitor poll
#      periods, i.e. one extra beyond the two needed even if the ack lands just
#      after a read started.
T_MONITOR_POLL_S = 0.1               # mirror of monitor.T_MONITOR_UPDATE_INTERVAL
TTL_PULSE_HOLD_S = 3 * T_MONITOR_POLL_S
TTL_PULSE_ACK_TIMEOUT_S = 2.0        # give up waiting for the "on" ack, send "off"

# --- polling cadences -----------------------------------------------------
STATUS_POLL_S = 1.0                  # monitor status poll
STATUS_RETRY_S = 2.0                 # back-off after a failed status poll
RECONCILE_MS = 10000                 # periodic full-snapshot safety reconcile

_SETTINGS_ORG = "waxx"
_SETTINGS_APP = "device_control_gui"


def _gui_settings() -> QSettings:
    """Per-user persistent GUI preferences (e.g. which TTL buttons pulse)."""
    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


def _setting(key: str, default, type_=None):
    """Read one preference, swallowing any QSettings error."""
    try:
        if type_ is None:
            return _gui_settings().value(key, default)
        return _gui_settings().value(key, default, type=type_)
    except Exception:
        return default


def _save_setting(key: str, value) -> None:
    try:
        _gui_settings().setValue(key, value)
    except Exception:
        pass


def _fmt_duration(seconds: float) -> str:
    """Human-readable 'held for' duration: 12 s, 3 min, 2 h 05 min, 1 d 3 h."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, m = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} h {m:02d} min"
    days, h = divmod(hours, 24)
    return f"{days} d {h} h"


def _muted_label(text: str, disabled: bool = False) -> QLabel:
    """Small muted placeholder label for an unassigned grid slot."""
    lbl = QLabel(text)
    color = theme.FG_DISABLED if disabled else theme.FG_MUTED
    lbl.setStyleSheet(f"QLabel {{ color: {color}; font-size: 8pt; border: none; }}")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if disabled:
        lbl.setEnabled(False)
    return lbl


class ScrollableButton(QLineEdit):
    """A button-like control whose (possibly long) name text scrolls
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

    def set_button_style(self, background: str = None, padding: str = "0px 1px") -> None:
        """Compose the tight button look: minimal padding and an optional
        background colour.

        Parameters
        ----------
        background : str, optional
            Background color name or hex value.
        padding : str
            CSS padding string (e.g. "0px 0px" for zero padding, "0px 1px" for minimal).
        """
        bg = f"background-color: {background};" if background else ""
        self.setStyleSheet(
            f"QLineEdit {{ border-radius: 3px; padding: {padding}; "
            f"border: 1px solid {theme.BORDER_STRONG}; {bg} }}"
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


class DeviceWidget(QWidget):
    """Base class for device control widgets"""
    value_changed = pyqtSignal(str, str, dict)

    def __init__(self, device_name: str, device_config: Dict[str, Any]):
        super().__init__()
        self.device_name = device_name
        self.device_config = device_config
        self.setFont(QFont("Arial", 9))
        self._search_matched = False
        self._opacity_effect = None        # lazily created for search dimming
        self.device_label = None           # set by subclasses in setup_ui
        # Each channel is drawn as its own card (raised background + 1 px
        # frame) so neighbouring cells do not run together.  The object-name
        # selector keeps the rule from cascading into the child spinboxes and
        # buttons; WA_StyledBackground makes a plain QWidget paint it.
        self.setObjectName("device_cell")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#device_cell {{ background-color: {theme.BG_CARD}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 3px; }}")

    def get_updated_config(self) -> Dict[str, Any]:
        """Return the updated configuration for this device"""
        raise NotImplementedError

    def update_from_config(self, config: Dict[str, Any]):
        """Update widget values from configuration"""
        raise NotImplementedError

    # --- search -----------------------------------------------------------

    def set_search_highlight(self, matched: bool) -> None:
        """Highlight the matched device: the device-name label gets a
        light-blue background."""
        self._search_matched = matched
        self._refresh_label_style()

    def _refresh_label_style(self) -> None:
        """Recompose the device label style.  Subclasses that tint the label
        for other reasons (DAC nonzero cue) override and call the base logic
        only when not search-matched."""
        lbl = self.device_label
        if lbl is None:
            return
        lbl.setStyleSheet(
            f"QLineEdit {{ background-color: {SEARCH_LABEL_BG}; color: {theme.FG_STRONG}; }}"
            if self._search_matched else ""
        )

    def set_search_dimmed(self, dimmed: bool) -> None:
        """Fade the whole widget while a search is active and it does not
        match.  Opacity only — the controls stay fully functional."""
        if dimmed:
            if self._opacity_effect is None:
                self._opacity_effect = QGraphicsOpacityEffect(self)
                self.setGraphicsEffect(self._opacity_effect)
            self._opacity_effect.setOpacity(SEARCH_DIM_OPACITY)
            self._opacity_effect.setEnabled(True)
        elif self._opacity_effect is not None:
            self._opacity_effect.setOpacity(1.0)
            # Disabled effects cost nothing to paint.
            self._opacity_effect.setEnabled(False)

    # --- Esc = undo -------------------------------------------------------

    def _install_escape_undo(self) -> None:
        """Esc while focus is inside this widget undoes staged edits (same
        path as the undo button).  ``WidgetWithChildrenShortcut`` covers every
        spinbox line-edit at once."""
        sc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc.activated.connect(self._on_escape)

    def _on_escape(self) -> None:
        if getattr(self, "has_unsaved_changes", False):
            handler = getattr(self, "on_default_undo_clicked", None)
            if handler is not None:
                handler()


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
        self._force_update_pending = False
        # Display units.  ``_freq_unit`` is "MHz" or "Γ"; ``_amp_unit`` is
        # "Amp" or "V" (the two replace the old unit combo boxes).
        self._freq_unit = "MHz"
        self._amp_unit = "Amp"
        # Store previous values for undo functionality
        self.prev_freq = None
        self.prev_freq_unit = None
        self.prev_amp = None
        self.prev_vpd = None
        self.prev_sw_state = None
        self.setup_ui()

    # --- unit availability ------------------------------------------------

    def _has_transition(self) -> bool:
        return self.device_config.get("transition", "None") != "None"

    def _has_dac(self) -> bool:
        return self.device_config.get("dac_ch", -1) != -1

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 3, 4, 3)  # inset from the card frame
        layout.setSpacing(2)

        self.device_label = QLineEdit(self.device_name)
        self.device_label.setCursorPosition(0)
        self.device_label.setReadOnly(True)
        self.device_label.setToolTip(self.device_name)
        layout.addWidget(self.device_label)

        # Frequency controls.  Unit (MHz / Γ) is chosen from a right-click
        # menu on the spinbox and shown as the spinbox suffix.
        freq_layout = QHBoxLayout()

        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setSingleStep(0.1)
        self.freq_spinbox.setDecimals(4)
        self.freq_spinbox.setValue(self.device_config["frequency"] / 1e6)  # Convert Hz to MHz
        self.freq_spinbox.setMinimum(0.)
        self.freq_spinbox.setMaximum(400.)
        self.freq_spinbox.setSuffix(" MHz")
        self.freq_spinbox.lineEdit().returnPressed.connect(self.on_update_clicked)
        self.freq_spinbox.valueChanged.connect(self.on_freq_spinbox_value_changed)
        self._install_context_menu(self.freq_spinbox, self._show_freq_unit_menu)
        freq_layout.addWidget(self.freq_spinbox)
        layout.addLayout(freq_layout)

        # Amplitude controls (amp spinbox or v_pd spinbox, one visible at a
        # time; right-click chooses).
        amp_layout = QHBoxLayout()

        self.amp_spinbox = QDoubleSpinBox()
        self.amp_spinbox.setRange(0, 1)
        self.amp_spinbox.setDecimals(3)
        self.amp_spinbox.setSingleStep(0.005)
        self.amp_spinbox.setSuffix(" amp")
        self.amp_spinbox.setValue(self.device_config["amplitude"])
        self.amp_spinbox.lineEdit().returnPressed.connect(self.on_update_clicked)
        self.amp_spinbox.valueChanged.connect(self.on_amp_spinbox_value_changed)
        self._install_context_menu(self.amp_spinbox, self._show_amp_unit_menu)

        self.vpd_spinbox = QDoubleSpinBox()
        self.vpd_spinbox.setRange(0, 10)
        self.vpd_spinbox.setDecimals(2)
        self.vpd_spinbox.setSingleStep(0.05)
        self.vpd_spinbox.setSuffix(" V")
        self.vpd_spinbox.setValue(self.device_config.get("v_pd", 5.0))
        self.vpd_spinbox.lineEdit().returnPressed.connect(self.on_update_clicked)
        self.vpd_spinbox.valueChanged.connect(self.on_vpd_spinbox_value_changed)
        self._install_context_menu(self.vpd_spinbox, self._show_amp_unit_menu)

        self.power_control_widget = QHBoxLayout()
        self.power_control_widget.addWidget(self.amp_spinbox)
        self.power_control_widget.addWidget(self.vpd_spinbox)
        amp_layout.addLayout(self.power_control_widget)
        layout.addLayout(amp_layout)

        # sw state + default/undo row
        self.state_button = QPushButton("Off")
        self.state_button.setCheckable(True)
        self.state_button.toggled.connect(self.on_state_button_toggled)
        state_button_row = QHBoxLayout()
        state_button_row.addWidget(self.state_button)

        self.default_button = QPushButton("default")
        self.default_button.clicked.connect(self.on_default_undo_clicked)
        self.default_button.setStyleSheet(f"background-color: {DEFAULT_BUTTON_COLOR}")
        state_button_row.addWidget(self.default_button)

        layout.addLayout(state_button_row)

        self.setLayout(layout)
        self.update_from_config(self.device_config)

        # Restore per-device unit preferences (only if still valid for this
        # device's config), else the historical defaults: MHz, and V when the
        # channel has a VVA/PID DAC.
        start_amp_unit = "V" if self._has_dac() else "Amp"
        saved_amp = _setting(f"amp_unit/{self.device_name}", start_amp_unit, str)
        if saved_amp == "V" and not self._has_dac():
            saved_amp = "Amp"
        self.on_amp_unit_changed(saved_amp)
        saved_freq = _setting(f"freq_unit/{self.device_name}", "MHz", str)
        if saved_freq == "Γ" and self._has_transition():
            self.on_freq_unit_changed("Γ")

        self._refresh_default_tooltip()
        self._install_escape_undo()

    # --- context menus -------------------------------------------------------

    @staticmethod
    def _install_context_menu(spinbox: QDoubleSpinBox, handler) -> None:
        """Route right-clicks on *spinbox* (frame or line-edit) to *handler*,
        which receives a global QPoint."""
        for w in (spinbox, spinbox.lineEdit()):
            w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            w.customContextMenuRequested.connect(
                lambda pos, w=w: handler(w.mapToGlobal(pos)))

    def _show_freq_unit_menu(self, global_pos) -> None:
        menu = QMenu(self)
        actions = {}
        units = ["MHz"] + (["Γ"] if self._has_transition() else [])
        for unit in units:
            act = menu.addAction(unit)
            act.setCheckable(True)
            act.setChecked(unit == self._freq_unit)
            actions[act] = unit
        chosen = menu.exec(global_pos)
        if chosen is not None and actions[chosen] != self._freq_unit:
            self.on_freq_unit_changed(actions[chosen])
            _save_setting(f"freq_unit/{self.device_name}", self._freq_unit)

    def _show_amp_unit_menu(self, global_pos) -> None:
        menu = QMenu(self)
        actions = {}
        choices = [("Amplitude", "Amp")] + ([("Voltage", "V")] if self._has_dac() else [])
        for label, unit in choices:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(unit == self._amp_unit)
            actions[act] = unit
        chosen = menu.exec(global_pos)
        if chosen is not None and actions[chosen] != self._amp_unit:
            self.on_amp_unit_changed(actions[chosen])
            _save_setting(f"amp_unit/{self.device_name}", self._amp_unit)

    # --- default / undo ------------------------------------------------------

    def _default_dds(self):
        """The ``_id`` frame object for this channel, or None."""
        if self.dds_frame_obj is not None and hasattr(self.dds_frame_obj, self.device_name):
            return vars(self.dds_frame_obj)[self.device_name]
        return None

    def _refresh_default_tooltip(self) -> None:
        hint = "Enter applies · Esc undoes"
        dds = self._default_dds()
        if dds is not None:
            try:
                hint = (f"Default: {dds.frequency / 1e6:.3f} MHz, amp {dds.amplitude:.3f}, "
                        f"{dds.v_pd:.2f} V\n" + hint)
            except Exception:
                pass
        self.default_button.setToolTip(hint)

    def on_default_undo_clicked(self):
        """Handle default/undo button click"""
        if self.has_unsaved_changes:
            # Undo: restore previous values. Restore the frequency unit first so
            # the spinbox range matches the stored value's unit before writing it
            # (otherwise an MHz value gets clamped into a Γ-ranged spinbox).
            if self.prev_freq_unit is not None and self.prev_freq_unit != self._freq_unit:
                self.on_freq_unit_changed(self.prev_freq_unit)
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
            dds = self._default_dds()
            if dds is not None:
                # Snapshot the pre-default state so undo restores exactly what
                # was on screen (including the current frequency unit) before
                # "default" switched the unit and wrote the default value.
                self.prev_freq_unit = self._freq_unit
                self.prev_freq = self.freq_spinbox.value()
                self.prev_amp = self.amp_spinbox.value()
                self.prev_vpd = self.vpd_spinbox.value()
                self.prev_sw_state = self.state_button.isChecked()
                # If this channel is defined by detuning (has a transition),
                # show the default in Γ units; otherwise show it in MHz.
                if getattr(dds, "transition", "None") != "None":
                    self.on_freq_unit_changed("Γ")
                    self.freq_spinbox.setValue(dds.frequency_to_detuning(dds.frequency))
                else:
                    self.on_freq_unit_changed("MHz")
                    self.freq_spinbox.setValue(dds.frequency/1.e6)
                self.amp_spinbox.setValue(dds.amplitude)
                # self.state_button.setChecked(dds.sw_state)
                self.vpd_spinbox.setValue(dds.v_pd)
                # Stage the change: show "undo" and require Enter to confirm,
                # even if a value happens to equal the current one.
                self._force_update_pending = True
                self.has_unsaved_changes = True
                _LOG.debug("DDS %s: default pressed; staged freq=%s amp=%s v_pd=%s",
                           self.device_name, dds.frequency, dds.amplitude, dds.v_pd)
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
        """Handle frequency unit change between MHz and Γ.

        Converts the displayed value in place; a unit change is not a value
        change, so the spinbox signals are blocked while converting.
        """
        suffix = " Γ" if unit == "Γ" else " MHz"
        if unit == self._freq_unit:
            # Already displaying this unit — nothing to convert.  (The old
            # combo box did not emit on a same-text setCurrentText either.)
            self.freq_spinbox.setSuffix(suffix)
            return
        current_value = self.freq_spinbox.value()
        self._freq_unit = unit
        self.freq_spinbox.setSuffix(suffix)

        if unit == "Γ":
            # Convert MHz to Γ
            if self.dds_frame_obj:
                try:
                    uru_idx = self.device_config["urukul_idx"]
                    ch = self.device_config["ch"]
                    dds_obj = self.dds_frame_obj.dds_array[uru_idx][ch]
                    freq_hz = current_value * 1e6
                    Γ_value = dds_obj.frequency_to_detuning(freq_hz)
                    with QSignalBlocker(self.freq_spinbox):
                        self.freq_spinbox.setMinimum(-100.)
                        self.freq_spinbox.setMaximum(100.)
                        self.freq_spinbox.setValue(Γ_value)
                except Exception as e:
                    _LOG.warning("DDS %s: MHz→Γ conversion failed: %s", self.device_name, e)
        elif unit == "MHz":
            # Convert Γ to MHz
            if self.dds_frame_obj:
                try:
                    uru_idx = self.device_config["urukul_idx"]
                    ch = self.device_config["ch"]
                    dds_obj = self.dds_frame_obj.dds_array[uru_idx][ch]
                    freq_hz = dds_obj.detuning_to_frequency(current_value)
                    with QSignalBlocker(self.freq_spinbox):
                        self.freq_spinbox.setMinimum(0.)
                        self.freq_spinbox.setMaximum(400.)
                        self.freq_spinbox.setValue(freq_hz / 1e6)
                except Exception as e:
                    _LOG.warning("DDS %s: Γ→MHz conversion failed: %s", self.device_name, e)

    def on_amp_unit_changed(self, unit):
        """Handle amplitude unit change between Amp and V"""
        if not self._has_dac():
            self._amp_unit = "Amp"
            self.amp_spinbox.setVisible(True)
            self.vpd_spinbox.setVisible(False)
            return
        self._amp_unit = "V" if unit == "V" else "Amp"
        if self._amp_unit == "V":
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
            css = f"QDoubleSpinBox {{ background-color: {theme.PENDING}; }}"
            self.freq_spinbox.setStyleSheet(css)
            self.amp_spinbox.setStyleSheet(css)
            self.vpd_spinbox.setStyleSheet(css)
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
        self.prev_freq_unit = self._freq_unit
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
        if self._freq_unit == "Γ":
            try:
                uru_idx = self.device_config["urukul_idx"]
                ch = self.device_config["ch"]
                dds_obj = self.dds_frame_obj.dds_array[uru_idx][ch]
                freq_hz = dds_obj.detuning_to_frequency(freq_value)

                config["frequency"] = freq_hz
            except Exception:
                config["frequency"] = freq_value * 1e6  # Fallback to MHz conversion
        else:
            config["frequency"] = freq_value * 1e6  # Convert MHz to Hz

        # Update amplitude (both the amp and v_pd values are always sent)
        config["v_pd"] = self.vpd_spinbox.value()
        config["amplitude"] = self.amp_spinbox.value()

        # Update sw state
        config["sw_state"] = int(self.state_button.isChecked())

        if self._force_update_pending:
            config["force_update_counter"] = config.get("force_update_counter", 0) + 1
            self._force_update_pending = False
            _LOG.debug("DDS %s: force_update_counter -> %s (freq=%s amp=%s v_pd=%s sw=%s)",
                       self.device_name, config["force_update_counter"],
                       config.get("frequency"), config.get("amplitude"),
                       config.get("v_pd"), config.get("sw_state"))

        return config

    def _freq_hz_to_display(self, freq_hz: float) -> float:
        """Convert a frequency in Hz to the value shown in the freq spinbox.

        Honors the current unit selection: returns detuning (Γ) when the unit
        is "Γ", otherwise MHz.  This prevents writing an MHz value into a
        Γ-ranged spinbox (range [-100, 100]), which would otherwise clamp the
        display to the max and get "stuck" there.
        """
        if self._freq_unit == "Γ" and self.dds_frame_obj:
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
            self.prev_freq_unit = self._freq_unit
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


class DACWidget(DeviceWidget):
    """Widget for controlling DAC devices"""

    def __init__(self, device_name: str, device_config: Dict[str, Any],
                  step_size_controller=None,
                  dac_frame_obj=None):
        super().__init__(device_name, device_config)
        self.dac_frame_obj = dac_frame_obj
        self.step_size_controller = step_size_controller  # Reference to shared step size controls
        self.has_unsaved_changes = False
        self.instant_apply = False
        self._force_update_pending = False
        # Store previous value for undo functionality
        self.prev_voltage = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 3, 4, 3)  # inset from the card frame
        layout.setSpacing(2)

        self.device_label = QLineEdit(self.device_name)
        self.device_label.setCursorPosition(0)
        self.device_label.setReadOnly(True)
        self.device_label.setToolTip(self.device_name)
        layout.addWidget(self.device_label)

        # Voltage control
        voltage_layout = QHBoxLayout()

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

        layout.addLayout(voltage_layout)

        self.setLayout(layout)
        self._refresh_label_style()
        self._refresh_default_tooltip()
        self._install_escape_undo()

    # --- label tint (nonzero cue) -------------------------------------------

    def _refresh_label_style(self) -> None:
        """Search highlight wins; otherwise tint the label background with
        theme.ACCENT at an alpha proportional to |voltage| / 10 V (max ~35 %)."""
        if self._search_matched or self.device_label is None:
            super()._refresh_label_style()
            return
        v = abs(self.voltage_spinbox.value()) if hasattr(self, "voltage_spinbox") else 0.0
        frac = min(v / DAC_TINT_FULL_SCALE_V, 1.0)
        if frac <= 1e-9:
            self.device_label.setStyleSheet("")
            return
        c = QColor(theme.ACCENT)
        alpha = int(round(255 * DAC_TINT_MAX_ALPHA * frac))
        self.device_label.setStyleSheet(
            f"QLineEdit {{ background-color: rgba({c.red()}, {c.green()}, {c.blue()}, {alpha}); }}"
        )

    # --- default / undo ------------------------------------------------------

    def _default_dac(self):
        if self.dac_frame_obj is not None and hasattr(self.dac_frame_obj, self.device_name):
            return vars(self.dac_frame_obj)[self.device_name]
        return None

    def _refresh_default_tooltip(self) -> None:
        hint = "Enter applies · Esc undoes"
        dac = self._default_dac()
        if dac is not None:
            try:
                hint = f"Default: {dac.v:.3f} V\n" + hint
            except Exception:
                pass
        self.default_button.setToolTip(hint)

    def on_default_undo_clicked(self):
        """Handle default/undo button click"""
        if self.has_unsaved_changes:
            # Undo: restore previous value
            with QSignalBlocker(self.voltage_spinbox):
                self.voltage_spinbox.setValue(self.prev_voltage)
            self.has_unsaved_changes = False
            self.update_default_button_state()
            self._refresh_label_style()
        else:
            # Reset to default value — mark as pending until Enter is pressed
            dac = self._default_dac()
            if dac is not None:
                self.voltage_spinbox.setValue(dac.v)
                # Stage the change: show "undo" and require Enter to confirm,
                # even if the value happens to equal the current one.
                self._force_update_pending = True
                self.has_unsaved_changes = True
                self.update_default_button_state()
                self.voltage_spinbox.setFocus()
                self.voltage_spinbox.selectAll()

    def on_value_changed(self):
        """Mark that values have changed but not yet submitted; apply at
        once when instant apply is on for the DAC tab."""
        self.has_unsaved_changes = True
        self.update_default_button_state()
        self._refresh_label_style()
        if self.instant_apply:
            self.on_update_clicked()

    def on_instant_apply_toggled(self, checked):
        self.instant_apply = checked

    def setup_step_sizes(self):
        """Setup step sizes from the shared step size controller"""
        if self.step_size_controller:
            self.voltage_spinbox.setSingleStep(self.step_size_controller.dac_voltage_step_spinbox.value())
            btn = getattr(self.step_size_controller, "dac_instant_apply_button", None)
            if btn is not None:
                self.instant_apply = btn.isChecked()

    def highlight_unsaved(self):
        """Highlight spinbox orange when it has unsaved changes"""
        if self.has_unsaved_changes:
            self.voltage_spinbox.setStyleSheet(f"QDoubleSpinBox {{ background-color: {theme.PENDING}; }}")
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
        self._refresh_label_style()


class TTLWidget(DeviceWidget):
    """Widget for controlling TTL devices"""

    # QSettings key prefix for the per-device "pulse mode" preference.
    _PULSE_SETTING_PREFIX = "ttl_pulse_mode/"

    def __init__(self, device_name: str, device_config: Dict[str, Any]):
        super().__init__(device_name, device_config)
        # --- pulse mode ---------------------------------------------------
        # When enabled, a left click sends "on", waits for the server ack,
        # holds for TTL_PULSE_HOLD_S, then sends "off" (see module constants).
        self._pulse_mode = self._load_pulse_mode()
        self._pulse_active = False       # a pulse is in flight
        self._pulse_waiting_ack = False  # "on" sent, ack not yet received
        self._pulse_hold_timer = QTimer(self)
        self._pulse_hold_timer.setSingleShot(True)
        self._pulse_hold_timer.timeout.connect(self._finish_pulse)
        self._pulse_ack_timer = QTimer(self)
        self._pulse_ack_timer.setSingleShot(True)
        self._pulse_ack_timer.timeout.connect(self._on_pulse_ack_timeout)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(3, 2, 3, 2)  # inset from the card frame
        layout.setSpacing(1)

        self.device_label = QLineEdit(self.device_name)
        self.device_label.setCursorPosition(0)
        self.device_label.setReadOnly(True)
        self.device_label.setToolTip(self.device_name)
        layout.addWidget(self.device_label)

        # State control
        state_layout = QHBoxLayout()
        state_layout.setContentsMargins(0, 0, 0, 0)

        self.state_button = ScrollableButton("Off", checkable=not self._pulse_mode)
        self.state_button.toggled.connect(self.on_state_button_toggled)
        self.state_button.clicked.connect(self._on_state_button_clicked)
        # Right-click → mode menu (toggle vs. pulse).
        self.state_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.state_button.customContextMenuRequested.connect(self._show_state_button_menu)
        state_layout.addWidget(self.state_button)

        layout.addLayout(state_layout)

        self.setLayout(layout)
        self.update_from_config(self.device_config)

    # --- pulse mode: preference + menu -----------------------------------

    def _load_pulse_mode(self) -> bool:
        return bool(_setting(self._PULSE_SETTING_PREFIX + self.device_name, False, bool))

    def _save_pulse_mode(self) -> None:
        _save_setting(self._PULSE_SETTING_PREFIX + self.device_name, bool(self._pulse_mode))

    def _show_state_button_menu(self, pos) -> None:
        menu = QMenu(self)
        toggle_action = menu.addAction("On / Off button")
        toggle_action.setCheckable(True)
        toggle_action.setChecked(not self._pulse_mode)
        pulse_action = menu.addAction("Pulse mode")
        pulse_action.setCheckable(True)
        pulse_action.setChecked(self._pulse_mode)
        chosen = menu.exec(self.state_button.mapToGlobal(pos))
        if chosen is pulse_action:
            self.set_pulse_mode(True)
        elif chosen is toggle_action:
            self.set_pulse_mode(False)

    def set_pulse_mode(self, enabled: bool) -> None:
        """Switch the state button between toggle and pulse behaviour."""
        enabled = bool(enabled)
        if enabled == self._pulse_mode:
            return
        self._pulse_mode = enabled
        self._save_pulse_mode()
        # In pulse mode a left click must not toggle the cached state; the
        # pulse sequence drives ``setChecked`` itself.
        self.state_button.setCheckable(not enabled)
        self._refresh_state_text()
        self._refresh_button_style()
        self._refresh_tooltip()

    # --- pulse mode: sequencing ------------------------------------------

    def _on_state_button_clicked(self) -> None:
        # In toggle mode the ``toggled`` signal already did the work.
        if self._pulse_mode:
            self._start_pulse()

    def _start_pulse(self) -> None:
        if self._pulse_active:
            return
        self._pulse_active = True
        self._pulse_waiting_ack = True
        self._send_state(True)
        self._pulse_ack_timer.start(int(TTL_PULSE_ACK_TIMEOUT_S * 1e3))

    def on_update_acked(self, ok: bool) -> None:
        """Called by the main window when the server acks (or fails) our edit.

        The first ack after the "on" was sent starts the hold; the "off" ack
        is ignored because the pulse is no longer active by then.
        """
        if not (self._pulse_active and self._pulse_waiting_ack):
            return
        self._pulse_waiting_ack = False
        self._pulse_ack_timer.stop()
        if ok:
            self._pulse_hold_timer.start(int(TTL_PULSE_HOLD_S * 1e3))
        else:
            # "on" never reached the server; send "off" anyway so the GUI and
            # the JSON cannot be left disagreeing if it did partially land.
            self._finish_pulse()

    def _on_pulse_ack_timeout(self) -> None:
        if self._pulse_active and self._pulse_waiting_ack:
            _LOG.warning("TTL %s: no ack for pulse 'on' after %.1f s; sending 'off'.",
                         self.device_name, TTL_PULSE_ACK_TIMEOUT_S)
            self._pulse_waiting_ack = False
            self._finish_pulse()

    def _finish_pulse(self) -> None:
        self._pulse_hold_timer.stop()
        self._send_state(False)
        self._pulse_active = False
        self._pulse_waiting_ack = False

    def _send_state(self, on: bool) -> None:
        """Set the cached state without re-entering the toggle handler, refresh
        the visuals, and push the delta to the server."""
        with QSignalBlocker(self.state_button):
            self.state_button.setChecked(on)
        self._refresh_state_text()
        self._refresh_button_style()
        self.on_update_clicked()

    # --- visuals ----------------------------------------------------------

    def _state_text(self) -> str:
        if self._pulse_mode:
            return "Pulse"
        return "On" if self.state_button.isChecked() else "Off"

    def _refresh_state_text(self) -> None:
        self.state_button.setText(self._state_text())
        self.state_button.setCursorPosition(0)

    def _refresh_tooltip(self) -> None:
        ch = self.device_config.get("ch")
        base = f"{self.device_name}\nttl{ch}" if ch is not None else self.device_name
        if self._pulse_mode:
            base += "\n[pulse mode: click = on → off]"
        base += "\n(right-click to change mode)"
        if self.device_label:
            self.device_label.setToolTip(base)
        self.state_button.setToolTip(base)

    def on_state_button_toggled(self, checked):
        self._refresh_state_text()
        self._refresh_button_style()
        self.on_update_clicked()

    def _refresh_button_style(self) -> None:
        """State colour on the button: green when on, no background when off
        (matches DDS).  Minimal/negative padding lets the button fill the grid
        cell without wasting space."""
        bg = STATE_BUTTON_ON_COLOR if self.state_button.isChecked() else None
        self.state_button.set_button_style(bg, padding="-1px 0px")
        self.state_button.setCursorPosition(0)

    def set_tooltip(self, ch: int):
        """Set tooltip to show device name and channel"""
        self.device_config["ch"] = ch
        self._refresh_tooltip()

    def on_update_clicked(self):
        """Handle update button click"""
        updated_config = self.get_updated_config()
        self.value_changed.emit("ttl", self.device_name, updated_config)

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
        self._refresh_state_text()
        self._refresh_button_style()


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
    """Long-lived thread that fetches the full device-state snapshot from the
    server (``get_state``) on demand.

    ``request()`` wakes the thread; a request arriving while a fetch is in
    flight is dropped (the in-flight one will deliver an equally fresh
    snapshot).  The worker owns its ``MonitorClient`` and keeps it across
    requests to avoid repeated service-discovery overhead; it is discarded
    after a failed call so the next request re-runs discovery (handles server
    restarts).  ``send_message`` already attempts ``_rediscover()`` internally,
    so ``state_failed`` only fires when the server is truly unreachable.
    """

    state_loaded = pyqtSignal(dict)   # {"version": int, "config": dict}
    state_failed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._busy = False
        self._running = True
        self._client: MonitorClient | None = None

    def request(self) -> bool:
        """Ask for a snapshot.  Returns False if one is already in flight."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self._wake.set()
        return True

    def run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if not self._running:
                return
            try:
                self._fetch_once()
            finally:
                with self._lock:
                    self._busy = False

    def _fetch_once(self) -> None:
        if self._client is None:
            try:
                self._client = MonitorClient(discovery_timeout=0.5)
            except Exception:
                self.state_failed.emit()
                return
        state = self._client.get_state()
        if state and state.get("status") == "ok":
            self.state_loaded.emit({
                "version": state.get("version"),
                "config": state.get("config", {}) or {},
            })
        else:
            # Force a fresh discovery (+ construction) on the next request.
            self._client = None
            self.state_failed.emit()

    def stop(self):
        self._running = False
        self._wake.set()


class _MonitorCommandWorker(QThread):
    """Sends a one-shot monitor command ("reset" or "stop") off the GUI thread.

    A fresh ``MonitorClient`` is created in the worker so the status checker's
    client is never used from two threads at once.
    """

    succeeded = pyqtSignal(str)     # command
    failed = pyqtSignal(str, str)   # command, message

    def __init__(self, command: str, parent=None):
        super().__init__(parent)
        self._command = command

    def run(self):
        try:
            client = MonitorClient(discovery_timeout=1.0)
            if self._command == "stop":
                reply = client.send_stop()
                if reply is None:
                    raise RuntimeError("no reply from monitor server")
            else:
                client.send_reset()
            self.succeeded.emit(self._command)
        except Exception as e:
            self.failed.emit(self._command, str(e))


class MonitorStatusChecker(QThread):
    """Thread that periodically checks the monitor server status.

    Prefers the structured ``get_status()`` (emits ``status_detail``); falls
    back to the legacy ``check_status()`` integer (emits ``status_updated``)
    while talking to an older server.
    """
    status_updated = pyqtSignal(int)
    status_detail = pyqtSignal(dict)
    connection_failed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.monitor_client: MonitorClient | None = None
        self.running = True
        self._wake = threading.Event()

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep so ``wake()``/``stop()`` take effect at once."""
        self._wake.wait(seconds)
        self._wake.clear()

    def wake(self) -> None:
        """Poll again now (used by the status pill's retry click)."""
        self._wake.set()

    def run(self):
        while self.running:
            if self.monitor_client is None:
                try:
                    self.monitor_client = MonitorClient(discovery_timeout=0.5)
                except RuntimeError:
                    # Discovery failed: surface via signal only (red
                    # indicator); do not spam the log.
                    self.connection_failed.emit()
                    self._sleep(STATUS_RETRY_S)
                    continue
            try:
                detail = self.monitor_client.get_status()
                if detail is not None:
                    self.status_detail.emit(detail)
                    self._sleep(STATUS_POLL_S)
                    continue
                status = self.monitor_client.check_status()
                if status is not None:
                    self.status_updated.emit(int(status))
                    self._sleep(STATUS_POLL_S)
                else:
                    # send_message() swallows exceptions and returns None on
                    # failure — treat this as a lost connection and force
                    # re-discovery on the next iteration.
                    self.monitor_client = None
                    self.connection_failed.emit()
                    self._sleep(STATUS_RETRY_S)
            except Exception as e:
                # Reset client so the next iteration re-runs service discovery.
                # This handles server restarts (new dynamic port) cleanly.
                _LOG.warning("monitor status poll failed: %s", e)
                self.monitor_client = None
                self.connection_failed.emit()
                self._sleep(STATUS_RETRY_S)

    def stop(self):
        self.running = False
        self._wake.set()


class _ClickableLabel(QLabel):
    """QLabel that emits ``clicked`` on a left-button release."""

    clicked = pyqtSignal()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


# Human-readable text for the machine-readable ``sub_state`` values.
_SUB_STATE_TEXT = {
    "running": "running",
    "starting": "starting up",
    "never_started": "never started",
    "interrupted_by_run": "interrupted by an experiment run",
    "exited": "exited",
    "failed": "failed",
    "preflight_failed": "preflight failed",
    "stopped_on_request": "stopped on request",
}


class ChangesLogWindow(QWidget):
    """Pop-out window listing the device changes this GUI has seen.

    Parentless on purpose (own taskbar entry): the control GUI is usually
    embedded in a dashboard panel, and a window owned by a panel gets lost
    behind the dashboard.  The lines themselves live in
    ``DeviceStateGUI._changes`` so closing this window loses nothing; it is
    deleted on close and rebuilt from that buffer next time.
    """

    closed = pyqtSignal()
    clear_requested = pyqtSignal()

    _GEOMETRY_KEY = "ui/changes_geometry"

    def __init__(self, lines):
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("Device changes")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setStyleSheet(f"background: {theme.BG};")

        box = QVBoxLayout(self)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(4)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.count_label = QLabel()
        self.count_label.setStyleSheet(f"color: {theme.FG_MUTED};")
        head.addWidget(self.count_label, 1)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setToolTip("Copy every line to the clipboard.")
        self.copy_button.clicked.connect(self._copy_all)
        head.addWidget(self.copy_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Forget the recorded changes (the devices are not touched).")
        self.clear_button.clicked.connect(self._clear)
        head.addWidget(self.clear_button)
        box.addLayout(head)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.list.setFont(mono)
        self.list.setStyleSheet(
            f"QListWidget {{ color: {theme.FG}; background: {theme.BG_SUNKEN}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 4px; }}")
        box.addWidget(self.list, 1)

        for line in lines:
            self.list.addItem(line)
        self._refresh_count()
        self.list.scrollToBottom()

        self.resize(640, 360)
        geom = _setting(self._GEOMETRY_KEY, None)
        if isinstance(geom, QByteArray) and not geom.isEmpty():
            try:
                self.restoreGeometry(geom)
            except Exception:
                pass

    def append_line(self, line: str) -> None:
        self.list.addItem(line)
        while self.list.count() > CHANGES_LOG_MAX_ROWS:
            self.list.takeItem(0)
        self._refresh_count()
        self.list.scrollToBottom()

    def _refresh_count(self) -> None:
        n = self.list.count()
        self.count_label.setText(
            f"{n} change{'s' if n != 1 else ''}  (last {CHANGES_LOG_MAX_ROWS} kept)")
        self.copy_button.setEnabled(n > 0)
        self.clear_button.setEnabled(n > 0)

    def _copy_all(self) -> None:
        text = "\n".join(self.list.item(i).text() for i in range(self.list.count()))
        app = QApplication.instance()
        if app is not None:
            app.clipboard().setText(text)

    def _clear(self) -> None:
        self.list.clear()
        self._refresh_count()
        self.clear_requested.emit()

    def closeEvent(self, event):
        _save_setting(self._GEOMETRY_KEY, self.saveGeometry())
        self.closed.emit()
        super().closeEvent(event)


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
        # Last known monitor state (STATES.*), structured status dict (if the
        # server supports it) and when the current state began.
        self._monitor_state: int | None = None
        self._monitor_status: dict | None = None
        self._state_since: float | None = None
        self._command_in_flight = False

        # Server-pushed state tracking.  ``_version`` is the last device-state
        # version we have applied; ``_pending`` maps (device_type, device_name)
        # to the timestamp of a local edit awaiting the server's echo (used to
        # avoid clobbering an in-flight edit with an incoming broadcast).
        self._version = None
        self._pending: Dict[tuple, float] = {}
        # (device_type, device_name) -> (old_config, sent_changes) for edits
        # made in this GUI, so the recent-changes strip can log old → new once
        # the server confirms them.
        self._own_edits: Dict[tuple, tuple] = {}
        # Log of confirmed device changes (newest last) shown by the pop-out
        # changes window; kept here so it survives the window being closed.
        self._changes: deque = deque(maxlen=CHANGES_LOG_MAX_ROWS)
        self._changes_window: ChangesLogWindow | None = None

        self.setup_ui()
        self._setup_update_sender()
        self._setup_state_listener()
        self._setup_state_worker()
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

        central_widget_layout.addLayout(self._build_status_row())

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
        self.freq_step_spinbox.setValue(float(_setting("steps/dds_freq", 0.1, float)))
        self.freq_step_spinbox.setSuffix(" MHz")
        self.freq_step_spinbox.setMaximumHeight(20)
        dds_step_layout.addWidget(self.freq_step_spinbox)

        dds_step_layout.addSpacing(10)
        dds_step_layout.addWidget(QLabel("Amp:"))

        self.amp_step_spinbox = QDoubleSpinBox()
        self.amp_step_spinbox.setRange(0.001, 1)
        self.amp_step_spinbox.setDecimals(3)
        self.amp_step_spinbox.setSingleStep(0.001)
        self.amp_step_spinbox.setValue(float(_setting("steps/dds_amp", 0.005, float)))
        self.amp_step_spinbox.setMaximumHeight(20)
        dds_step_layout.addWidget(self.amp_step_spinbox)

        dds_step_layout.addSpacing(10)
        dds_step_layout.addWidget(QLabel("V:"))

        self.vpd_step_spinbox = QDoubleSpinBox()
        self.vpd_step_spinbox.setRange(0.001, 10)
        self.vpd_step_spinbox.setDecimals(3)
        self.vpd_step_spinbox.setSingleStep(0.01)
        self.vpd_step_spinbox.setValue(float(_setting("steps/dds_vpd", 0.05, float)))
        self.vpd_step_spinbox.setSuffix(" V")
        self.vpd_step_spinbox.setMaximumHeight(20)
        dds_step_layout.addWidget(self.vpd_step_spinbox)

        dds_step_layout.addStretch()

        self.instant_apply_button = self._make_instant_apply_button(
            bool(_setting("instant_apply/dds", False, bool)))
        self.instant_apply_button.toggled.connect(self.on_instant_apply_toggled)
        dds_step_layout.addWidget(self.instant_apply_button)

        dds_tab_layout.addLayout(dds_step_layout)

        # DDS devices grid layout (wrapped in container for border)
        self.dds_container = QWidget()
        self.dds_layout = QGridLayout()
        self.dds_layout.setHorizontalSpacing(GRID_SPACING)
        self.dds_layout.setVerticalSpacing(GRID_SPACING)
        self.dds_layout.setContentsMargins(6, 6, 6, 6)
        self.dds_container.setLayout(self.dds_layout)
        self._style_grid_container(self.dds_container, "dds_container")
        dds_tab_layout.addWidget(self.dds_container)
        self.dds_tab.setLayout(dds_tab_layout)

        # Connect DDS step size controls to update all DDS widgets (+ persist)
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
        self.dac_voltage_step_spinbox.setValue(float(_setting("steps/dac_voltage", 0.01, float)))
        self.dac_voltage_step_spinbox.setSuffix(" V")
        self.dac_voltage_step_spinbox.setMaximumHeight(20)
        dac_step_layout.addWidget(self.dac_voltage_step_spinbox)

        dac_step_layout.addStretch()

        # DAC tab has its own instant-apply toggle, mirroring the DDS one.
        self.dac_instant_apply_button = self._make_instant_apply_button(
            bool(_setting("instant_apply/dac", False, bool)))
        self.dac_instant_apply_button.toggled.connect(self.on_instant_apply_toggled_dac)
        dac_step_layout.addWidget(self.dac_instant_apply_button)

        dac_tab_layout.addLayout(dac_step_layout)

        # DAC devices grid layout (wrapped in container for border)
        self.dac_container = QWidget()
        self.dac_layout = QGridLayout()
        self.dac_layout.setHorizontalSpacing(GRID_SPACING)
        self.dac_layout.setVerticalSpacing(GRID_SPACING)
        self.dac_layout.setContentsMargins(6, 6, 6, 6)
        self.dac_container.setLayout(self.dac_layout)
        self._style_grid_container(self.dac_container, "dac_container")
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
        self.ttl_layout.setHorizontalSpacing(GRID_SPACING)
        self.ttl_layout.setVerticalSpacing(GRID_SPACING - 1)
        self.ttl_layout.setContentsMargins(6, 6, 6, 6)
        self.ttl_container.setLayout(self.ttl_layout)
        self._style_grid_container(self.ttl_container, "ttl_container")
        ttl_tab_layout.addWidget(self.ttl_container)
        self.ttl_tab.setLayout(ttl_tab_layout)

        # Restore + persist the active tab.
        saved_tab = int(_setting("ui/active_tab", 0, int))
        if 0 <= saved_tab < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(saved_tab)
        self.tab_widget.currentChanged.connect(
            lambda idx: _save_setting("ui/active_tab", int(idx)))

        # Ctrl+F focuses the shared search bar.
        # Must be on the central widget (not self) so the shortcut fires when
        # DeviceStateGUI is embedded inside a dashboard panel (the QMainWindow
        # itself is hidden by embed_main_window; shortcuts on hidden widgets
        # do not fire).
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

    # ------------------------------------------------------------------
    # Status row (monitor state + Start / Restart / Stop)
    # ------------------------------------------------------------------

    def _build_status_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.status_pill = _ClickableLabel("Connecting to monitor server…")
        pill_font = QFont()
        pill_font.setPointSize(11)
        pill_font.setBold(True)
        self.status_pill.setFont(pill_font)
        self.status_pill.setMinimumHeight(25)
        self.status_pill.clicked.connect(self.on_status_pill_clicked)
        self._style_pill(theme.OFF)
        row.addWidget(self.status_pill)

        self.status_detail_label = QLabel("")
        self.status_detail_label.setStyleSheet(f"color: {theme.FG_MUTED};")
        row.addWidget(self.status_detail_label, 1)

        # The changes log lives in its own pop-out window (see
        # ChangesLogWindow) instead of a strip under the grids.
        self.changes_button = QPushButton("Changes")
        self.changes_button.setMaximumHeight(25)
        self.changes_button.clicked.connect(self.show_changes_log)
        self._refresh_changes_button()
        row.addWidget(self.changes_button)

        self.start_button = QPushButton("Start monitor")
        self.start_button.setToolTip("Start the monitor experiment (it is not running).")
        self.start_button.clicked.connect(self.on_start_clicked)
        self.restart_button = QPushButton("Restart")
        self.restart_button.setToolTip("Restart the monitor experiment.")
        self.restart_button.clicked.connect(self.on_restart_clicked)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip("Stop the monitor experiment and leave it stopped.")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        for btn in (self.start_button, self.restart_button, self.stop_button):
            btn.setEnabled(False)
            btn.setMaximumHeight(25)
            row.addWidget(btn)
        return row

    def _style_pill(self, bg: str) -> None:
        self.status_pill.setStyleSheet(
            f"QLabel {{ background: {bg}; color: white; border-radius: 6px; padding: 2px 10px; }}"
        )

    def _refresh_status_buttons(self) -> None:
        """Enable Start / Restart / Stop according to the monitor state."""
        reachable = not self.connection_failed and self._monitor_state is not None
        busy = self._command_in_flight
        st = self._monitor_state
        self.start_button.setEnabled(reachable and not busy and st == STATES.NOT_READY)
        live = reachable and not busy and st in (STATES.READY, STATES.LOADING)
        self.restart_button.setEnabled(live)
        self.stop_button.setEnabled(live)

    def _set_monitor_state(self, state: int) -> None:
        """Apply a monitor state (STATES.*) to the pill; track when it began."""
        self.connection_failed = False
        if state != self._monitor_state:
            self._monitor_state = state
            self._state_since = time.time()
        if state == STATES.READY:
            self.status_pill.setText("Monitor ready")
            self._style_pill(theme.OK)
        elif state == STATES.LOADING:
            self.status_pill.setText("Monitor starting…")
            self._style_pill(theme.WARN)
        else:  # STATES.NOT_READY
            self.status_pill.setText("Monitor not running")
            self._style_pill(theme.ERR)
        self._refresh_status_buttons()

    def _on_status_updated(self, status: int) -> None:
        """Legacy integer status from an older server (no detail available)."""
        self._monitor_status = None
        self._set_monitor_state(int(status))
        held = _fmt_duration(time.time() - self._state_since) if self._state_since else ""
        self.status_detail_label.setText(f"for {held}" if held else "")

    def _on_status_detail(self, detail: dict) -> None:
        """Structured status from ``get_status()``."""
        self._monitor_status = detail
        try:
            state = int(detail.get("state", STATES.NOT_READY))
        except (TypeError, ValueError):
            state = STATES.NOT_READY
        self._set_monitor_state(state)
        since = detail.get("since")
        try:
            since = float(since) if since is not None else None
        except (TypeError, ValueError):
            since = None
        if since is not None:
            self._state_since = since
        sub = str(detail.get("sub_state") or "")
        reason = str(detail.get("reason") or "").strip()
        parts = []
        text = reason or _SUB_STATE_TEXT.get(sub, sub.replace("_", " "))
        if text:
            parts.append(text)
        if self._state_since:
            parts.append(f"for {_fmt_duration(time.time() - self._state_since)}")
        self.status_detail_label.setText("  ·  ".join(parts))
        tip = []
        if detail.get("pid") is not None:
            tip.append(f"pid {detail['pid']}")
        if detail.get("expt_path"):
            tip.append(str(detail["expt_path"]))
        self.status_detail_label.setToolTip("\n".join(tip))

    def on_connection_failed(self):
        """Handle connection failure"""
        self.connection_failed = True
        self._monitor_state = None
        self.status_pill.setText("Monitor server unreachable")
        self._style_pill(UNREACHABLE_COLOR)
        self.status_detail_label.setText("click the status to retry")
        self._refresh_status_buttons()

    def on_status_pill_clicked(self):
        """Clicking the pill does nothing unless the server is unreachable,
        in which case it retries at once."""
        if not self.connection_failed:
            return
        self.status_pill.setText("Connecting…")
        self._style_pill(theme.OFF)
        self.status_detail_label.setText("")
        self.connection_failed = False
        checker = getattr(self, "status_checker", None)
        if checker is not None:
            checker.wake()
        self.request_state()

    def on_start_clicked(self):
        self._send_monitor_command("reset")

    def on_restart_clicked(self):
        reply = QMessageBox.question(
            self, "Restart monitor",
            "Restart the monitor experiment? If another experiment is currently "
            "using the core device this will interrupt it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._send_monitor_command("reset")

    def on_stop_clicked(self):
        reply = QMessageBox.question(
            self, "Stop monitor",
            "Stop the monitor experiment? Hardware will no longer be held in "
            "the idle state until it is started again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._send_monitor_command("stop")

    def _send_monitor_command(self, command: str) -> None:
        """Run ``send_reset`` / ``send_stop`` off the GUI thread."""
        if self._command_in_flight:
            return
        self._command_in_flight = True
        self._refresh_status_buttons()
        self.status_detail_label.setText("sending stop…" if command == "stop" else "sending restart…")
        worker = _MonitorCommandWorker(command, self)
        worker.succeeded.connect(self._on_command_succeeded)
        worker.failed.connect(self._on_command_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._command_worker = worker

    def _on_command_succeeded(self, command: str) -> None:
        self._command_in_flight = False
        self._refresh_status_buttons()
        _LOG.info("monitor %s sent", command)
        checker = getattr(self, "status_checker", None)
        if checker is not None:
            checker.wake()

    def _on_command_failed(self, command: str, msg: str) -> None:
        self._command_in_flight = False
        self._refresh_status_buttons()
        _LOG.warning("monitor %s failed: %s", command, msg)
        QMessageBox.critical(self, "Error", f"Failed to send {command} to the monitor server: {msg}")

    # ------------------------------------------------------------------
    # Changes log (pop-out window)
    # ------------------------------------------------------------------

    @staticmethod
    def _style_grid_container(container: QWidget, object_name: str) -> None:
        """Frame around one device grid.  The rule is scoped by object name
        so it does not cascade into every child widget (a bare ``border:``
        rule on the container used to restyle every spinbox and button)."""
        container.setObjectName(object_name)
        container.setStyleSheet(
            f"QWidget#{object_name} {{ border: 1px solid {theme.BORDER}; border-radius: 4px; }}")

    def show_changes_log(self) -> None:
        """Open (or raise) the pop-out window listing device changes."""
        if self._changes_window is None:
            win = ChangesLogWindow(list(self._changes))
            win.setWindowIcon(self.windowIcon())
            win.clear_requested.connect(self._clear_changes)
            win.closed.connect(self._on_changes_window_closed)
            self._changes_window = win
        self._changes_window.show()
        self._changes_window.raise_()
        self._changes_window.activateWindow()

    def _on_changes_window_closed(self) -> None:
        self._changes_window = None

    def _clear_changes(self) -> None:
        self._changes.clear()
        self._refresh_changes_button()

    def _refresh_changes_button(self) -> None:
        n = len(self._changes)
        self.changes_button.setText(f"Changes ({n})" if n else "Changes")
        tip = "Open the log of device changes in its own window."
        if n:
            tip += f"\nLast: {self._changes[-1]}"
        self.changes_button.setToolTip(tip)

    @staticmethod
    def _describe_change(dtype: str, old: dict, changes: dict) -> str:
        """One-line ``key old → new`` summary of *changes* against *old*.
        Returns "" when nothing user-visible changed."""
        def onoff(v):
            return "on" if bool(v) else "off"

        parts = []
        if dtype == "ttl":
            if "ttl_state" in changes and bool(changes["ttl_state"]) != bool(old.get("ttl_state")):
                parts.append(f"{onoff(old.get('ttl_state'))} → {onoff(changes['ttl_state'])}")
        elif dtype == "dac":
            if "voltage" in changes and changes["voltage"] != old.get("voltage"):
                o = old.get("voltage")
                o = f"{o:.3f} V" if isinstance(o, (int, float)) else "?"
                parts.append(f"{o} → {changes['voltage']:.3f} V")
        elif dtype == "dds":
            fmt = {
                "frequency": ("freq", lambda v: f"{v / 1e6:.3f}", " MHz"),
                "amplitude": ("amp", lambda v: f"{v:.3f}", ""),
                "v_pd": ("v_pd", lambda v: f"{v:.2f}", " V"),
                "sw_state": ("sw", onoff, ""),
            }
            for key, (label, f, unit) in fmt.items():
                if key not in changes or changes[key] == old.get(key):
                    continue
                o = old.get(key)
                try:
                    o_s = f(o) if o is not None else "?"
                    n_s = f(changes[key])
                except Exception:
                    o_s, n_s = str(o), str(changes[key])
                parts.append(f"{label} {o_s} → {n_s}{unit}")
        if not parts and "force_update_counter" in changes \
                and changes["force_update_counter"] != old.get("force_update_counter"):
            parts.append("force update")
        return ", ".join(parts)

    def _record_change(self, dtype: str, name: str, old: dict, changes: dict) -> None:
        text = self._describe_change(dtype, old, changes)
        if not text:
            return
        line = f"{time.strftime('%H:%M:%S')}  {dtype} {name}  {text}"
        self._changes.append(line)
        self._refresh_changes_button()
        if self._changes_window is not None:
            self._changes_window.append_line(line)

    # ------------------------------------------------------------------

    @staticmethod
    def _make_instant_apply_button(checked: bool) -> QPushButton:
        btn = QPushButton()
        btn.setCheckable(True)
        btn.setMaximumHeight(20)
        btn.setChecked(checked)
        DeviceStateGUI._style_instant_apply_button(btn, checked)
        return btn

    @staticmethod
    def _style_instant_apply_button(btn: QPushButton, checked: bool) -> None:
        if checked:
            btn.setStyleSheet(f"background-color: {theme.ERR}; color: white; font-weight: bold;")
            btn.setText("turn off instant apply")
        else:
            btn.setStyleSheet(f"background-color: {theme.PENDING}; color: black; font-weight: bold;")
            btn.setText("turn on instant apply")

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
        self.timer.timeout.connect(self._periodic_reconcile)
        self.timer.start(RECONCILE_MS)

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

    def _setup_state_worker(self):
        """Start the long-lived snapshot fetcher (serves ``request_state``)."""
        self._state_worker = _StateRequestWorker(parent=self)
        self._state_worker.state_loaded.connect(self._on_state_loaded)
        self._state_worker.state_failed.connect(self._on_state_failed)
        self._state_worker.start()

    def setup_status_checker(self):
        """Setup the status checker thread"""
        self.status_checker = MonitorStatusChecker()
        self.status_checker.status_updated.connect(self._on_status_updated)
        self.status_checker.status_detail.connect(self._on_status_detail)
        self.status_checker.connection_failed.connect(self.on_connection_failed)
        self.status_checker.start()

    def on_instant_apply_toggled(self, checked):
        """DDS instant apply toggled: restyle, persist, push to widgets."""
        self._style_instant_apply_button(self.instant_apply_button, checked)
        _save_setting("instant_apply/dds", bool(checked))
        self.on_dds_step_size_changed()

    def on_instant_apply_toggled_dac(self, checked):
        """DAC instant apply toggled: restyle, persist, push to widgets."""
        self._style_instant_apply_button(self.dac_instant_apply_button, checked)
        _save_setting("instant_apply/dac", bool(checked))
        self.on_dac_step_size_changed()

    def on_dac_step_size_changed(self):
        """Update all DAC widgets when step size / instant apply changes"""
        _save_setting("steps/dac_voltage", self.dac_voltage_step_spinbox.value())
        for widget_key, widget in self.device_widgets.items():
            if widget_key.startswith("dac."):
                widget.setup_step_sizes()

    def on_dds_step_size_changed(self):
        """Update all DDS widgets when step size or instant apply changes"""
        _save_setting("steps/dds_freq", self.freq_step_spinbox.value())
        _save_setting("steps/dds_amp", self.amp_step_spinbox.value())
        _save_setting("steps/dds_vpd", self.vpd_step_spinbox.value())
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

    def _enclosing_scroll_area(self):
        """The QScrollArea this window is embedded in (dashboard), or None."""
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QScrollArea):
                return w
            w = w.parentWidget()
        return None

    def _apply_search(self, query: str, device_prefix: str) -> None:
        """Highlight matching devices, dim the rest, and scroll the first
        match into view, for the given prefix (dds/dac/ttl)."""
        terms = parse_name_search_terms(query)
        prefix = device_prefix + "."
        first_match = None
        for key, widget in self.device_widgets.items():
            if not key.startswith(prefix):
                continue
            device_name = key[len(prefix):]
            matched = bool(terms) and name_matches_all_terms(device_name, terms)
            widget.set_search_highlight(matched)
            widget.set_search_dimmed(bool(terms) and not matched)
            if matched and first_match is None:
                first_match = widget
        if first_match is not None:
            scroll = self._enclosing_scroll_area()
            if scroll is not None:
                scroll.ensureWidgetVisible(first_match)

    def _apply_active_search(self, query: str) -> None:
        """Apply *query* to the currently visible tab's devices only."""
        prefixes = ["dds", "dac", "ttl"]
        idx = self.tab_widget.currentIndex()
        if 0 <= idx < len(prefixes):
            self._apply_search(query, prefixes[idx])

    def request_state(self):
        """Fetch the full device-state snapshot from the server (async).

        Served by the long-lived ``_StateRequestWorker``; a request while one
        is in flight is dropped.
        """
        worker = getattr(self, "_state_worker", None)
        if worker is not None:
            worker.request()

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
        # Recent-changes strip: old → new.  If this is the echo of an edit
        # made here, log it against the pre-edit snapshot (config_data was
        # already updated optimistically) and drop the ack-time entry so it
        # is not logged twice.
        own = self._own_edits.get((dtype, name))
        if own is not None and all(dev.get(k) == v for k, v in changes.items()):
            old, sent = self._own_edits.pop((dtype, name))
            self._record_change(dtype, name, old, sent)
        else:
            self._record_change(dtype, name, dict(dev), changes)
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

    def _periodic_reconcile(self):
        """Safety reconcile (called by the slow timer)."""
        self.request_state()

    def update_device_widgets(self):
        """Update device widgets based on current configuration"""
        # Clear existing widgets
        self.clear_layouts()
        self.device_widgets.clear()

        # Add DDS widgets organized by urukul_idx (columns) and ch (rows).
        # No row/column headers: the urukul/channel is in each card's tooltip.
        if "dds" in self.config_data:
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

        # Add DAC widgets grouped into columns of 8; empty slots get a dimmed
        # "dac N" placeholder so the spatial map stays readable.
        if "dac" in self.config_data:
            placed = {}
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
                col, row = divmod(ch, CHANNELS_PER_COLUMN)
                self.dac_layout.addWidget(widget, row, col)
                self.device_widgets[f"dac.{device_name}"] = widget
                placed[ch] = widget

            num_cols = (max(placed) // CHANNELS_PER_COLUMN) + 1 if placed else 0
            for col in range(num_cols):
                for row in range(CHANNELS_PER_COLUMN):
                    ch = col * CHANNELS_PER_COLUMN + row
                    if ch not in placed:
                        self.dac_layout.addWidget(
                            _muted_label(f"dac {ch}", disabled=True), row, col)

        # Add TTL widgets grouped into columns of 8 (0-7, 8-15, etc.).  A
        # column with no output channel at all is skipped: the state file
        # only lists TTL outputs, so a bank of inputs (or an unused bank)
        # would otherwise show up as a column of empty placeholders.
        if "ttl" in self.config_data:
            by_col: Dict[int, list] = {}
            for device_name, device_config in self.config_data["ttl"].items():
                ch = int(device_config.get("ch", 0))
                by_col.setdefault(ch // CHANNELS_PER_COLUMN, []).append(
                    (ch, device_name, device_config))

            for col_idx, col in enumerate(sorted(by_col)):
                in_col = {}
                for ch, device_name, device_config in by_col[col]:
                    widget = TTLWidget(device_name, device_config)
                    widget.value_changed.connect(self.on_device_value_changed)
                    widget.set_tooltip(ch)
                    in_col[ch % CHANNELS_PER_COLUMN] = widget
                    self.device_widgets[f"ttl.{device_name}"] = widget
                for row in range(CHANNELS_PER_COLUMN):
                    ch = col * CHANNELS_PER_COLUMN + row
                    widget = in_col.get(row) or _muted_label(f"ttl {ch}", disabled=True)
                    self.ttl_layout.addWidget(widget, row, col_idx)

        self.adjust_window_width()

        # Re-apply active search so highlights survive config reloads.
        self._apply_active_search(self.search_bar.text())

    def closeEvent(self, event):
        """Handle window close event"""
        win = self._changes_window
        if win is not None:
            self._changes_window = None
            win.close()
        checker = getattr(self, "status_checker", None)
        if checker is not None:
            checker.stop()
            checker.wait()
        for attr in ("_state_listener", "_update_sender", "_state_worker"):
            thread = getattr(self, attr, None)
            if thread is not None:
                thread.stop()
                thread.wait()
        event.accept()

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
        """Handle a local device edit: update optimistically and push to server."""
        section = self.config_data.setdefault(device_type, {})
        key = (device_type, device_name)
        # Remember the pre-edit snapshot for the recent-changes strip.  Rapid
        # edits to one device coalesce (like the sender): keep the earliest
        # "old", merge the newest "changes".
        if key in self._own_edits:
            old, sent = self._own_edits[key]
            sent.update(updated_config)
        else:
            self._own_edits[key] = (dict(section.get(device_name, {})), dict(updated_config))

        # Optimistic local update so the UI stays responsive even if the
        # network round-trip is slow.
        if device_name in section:
            section[device_name].update(updated_config)
        else:
            section[device_name] = dict(updated_config)

        # Mark this device as having an in-flight edit so an incoming broadcast
        # (including our own echo) does not clobber the spinbox mid-interaction.
        self._pending[key] = time.time()

        if device_type == "dds" and "force_update_counter" in updated_config:
            _LOG.debug("on_device_value_changed: %s %s force_update_counter=%s",
                       device_type, device_name, updated_config["force_update_counter"])

        # Hand off to the background sender (coalesces rapid same-device edits).
        self._update_sender.enqueue(device_type, device_name, updated_config)

    def _on_update_ack(self, device_type: str, device_name: str, ack: dict) -> None:
        """Server accepted our delta."""
        self._version = ack.get("version", self._version)
        self._pending.pop((device_type, device_name), None)
        own = self._own_edits.pop((device_type, device_name), None)
        if own is not None:
            # Not already logged via the broadcast echo → log now.
            self._record_change(device_type, device_name, own[0], own[1])
        self._notify_widget_ack(device_type, device_name, ok=True)

    def _on_update_failed(self, device_type: str, device_name: str) -> None:
        """Server unreachable / rejected our delta.

        Keep the optimistic local value (so the user's intent is preserved on
        screen) and flag the connection problem with a red status indicator.
        """
        self._pending.pop((device_type, device_name), None)
        self._own_edits.pop((device_type, device_name), None)
        _LOG.warning("update to %s %s was not acknowledged by the monitor server",
                     device_type, device_name)
        self.on_connection_failed()
        self._notify_widget_ack(device_type, device_name, ok=False)

    def _notify_widget_ack(self, device_type: str, device_name: str, ok: bool) -> None:
        """Forward a per-device ack/failure to the widget that sent the edit.

        Only widgets that opt in (define ``on_update_acked``) are called; the
        TTL widget uses it to sequence the on → off steps of a pulse.
        """
        widget = self.device_widgets.get(f"{device_type}.{device_name}")
        handler = getattr(widget, "on_update_acked", None)
        if handler is not None:
            handler(ok)
