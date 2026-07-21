"""TpiDeviceWidget — PyQt6 control panel for one remote TPI-1005-A.

Shows live RF state (switch, frequency, level) received over the server's ZMQ
PUB socket and provides controls to change each parameter via ZMQ REQ/REP.
"""
from __future__ import annotations

import time
from typing import Optional

from PyQt6.QtCore import Qt, QSignalBlocker, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from waxx.util.guis.tpi.tpi_client import TpiDeviceClient, TpiStateSubscriber


# ---------------------------------------------------------------------------
# Background thread wrapping TpiStateSubscriber
# ---------------------------------------------------------------------------

class _StateThread(QThread):
    """Runs TpiStateSubscriber in a background thread and re-emits updates."""

    state_received = pyqtSignal(dict)

    def __init__(self, client: TpiDeviceClient, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._sub: Optional[TpiStateSubscriber] = None

    def run(self) -> None:
        self._sub = TpiStateSubscriber(self._client.connection)
        self._sub.start(callback=self.state_received.emit)
        self.exec()  # enter Qt event loop so the thread stays alive
        self._sub.stop()

    def stop_subscriber(self) -> None:
        if self._sub:
            self._sub.stop()
        self.quit()
        self.wait(2000)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

# Pending-update model colours (mirrors device_control_gui).
DEFAULT_BUTTON_COLOR = "#363636"
UNDO_BUTTON_COLOR = "orange"

DARK_STYLE = """
QWidget {
    background: #1a1b2e;
    color: #d0d4e8;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12px;
}
QFrame#card {
    background: #22233a;
    border: 1px solid #3a3c5a;
    border-radius: 6px;
}
QLabel#stale {
    font-size: 10px;
    color: #7a7e9a;
}
QPushButton#rfOn {
    background: #1b4d2e;
    color: #4caf50;
    border: 1px solid #4caf50;
    border-radius: 4px;
    font-size: 14px;
    font-weight: bold;
    padding: 5px 10px;
}
QPushButton#rfOff {
    background: #4d1b1b;
    color: #ef5350;
    border: 1px solid #ef5350;
    border-radius: 4px;
    font-size: 14px;
    font-weight: bold;
    padding: 5px 10px;
}
QPushButton#lock {
    background: transparent;
    border: none;
    font-size: 14px;
    padding: 0px;
}
QPushButton#lock:hover {
    background: #2a2d4a;
    border-radius: 4px;
}
QPushButton#default {
    border: 1px solid #3a3c5a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #d0d4e8;
}
QPushButton#default:disabled {
    color: #5a5d78;
    border-color: #2a2c44;
}
QDoubleSpinBox, QSpinBox {
    background: #12132a;
    color: #e8eaf6;
    border: 1px solid #3a3c5a;
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 15px;
    font-weight: bold;
}
QDoubleSpinBox:read-only, QSpinBox:read-only {
    background: #191a30;
    color: #d0d4e8;
    font-weight: normal;
}
"""


class TpiDeviceWidget(QWidget):
    """Self-contained control panel for one TPI-1005-A device.

    Subscribes to the server's PUB socket for live state and sends commands
    via the device client's REQ/REP socket.
    """

    def __init__(self, client: TpiDeviceClient, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._client = client
        self._last_update: float = 0.0
        self._rf_on: Optional[bool] = None

        # Pending-update model: an edit stages values (orange) until Enter
        # commits them.  ``_prev_*`` hold the last committed/known-good values
        # so "undo" can restore them.
        self._has_unsaved: bool = False
        self._prev_freq: Optional[float] = None
        self._prev_level: Optional[int] = None

        self.setStyleSheet(DARK_STYLE)
        self._build_ui()

        # Subscribe to live state
        self._thread = _StateThread(client, self)
        self._thread.state_received.connect(self._on_state)
        self._thread.start()

        # Stale indicator — update every second
        self._stale_timer = QTimer(self)
        self._stale_timer.setInterval(1000)
        self._stale_timer.timeout.connect(self._update_stale)
        self._stale_timer.start()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # Device spec lives in the dock title bar, so no header here.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        card = QFrame()
        card.setObjectName("card")
        row = QHBoxLayout(card)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        # --- RF switch (instant; independent of the lock) ----------------
        self._rf_btn = QPushButton("OFF")
        self._rf_btn.setObjectName("rfOff")
        self._rf_btn.setFixedWidth(64)
        self._rf_btn.clicked.connect(self._toggle_rf)
        row.addWidget(self._rf_btn)

        # --- Frequency (merged live display + editable spinbox) ----------
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(35.0, 4400.0)
        self._freq_spin.setDecimals(2)
        self._freq_spin.setSingleStep(1.0)
        self._freq_spin.setSuffix(" MHz")
        self._freq_spin.setFixedWidth(105)
        self._freq_spin.valueChanged.connect(self._mark_unsaved)
        self._freq_spin.lineEdit().returnPressed.connect(self._commit)
        row.addWidget(self._freq_spin)

        # --- Level (merged live display + editable spinbox) --------------
        self._level_spin = QSpinBox()
        self._level_spin.setRange(-90, 10)
        self._level_spin.setSuffix(" dBm")
        self._level_spin.setFixedWidth(90)
        self._level_spin.valueChanged.connect(self._mark_unsaved)
        self._level_spin.lineEdit().returnPressed.connect(self._commit)
        row.addWidget(self._level_spin)

        # --- default / undo (pending-update model) -----------------------
        self._default_btn = QPushButton("default")
        self._default_btn.setObjectName("default")
        self._default_btn.setFixedWidth(64)
        self._default_btn.clicked.connect(self._on_default_undo_clicked)
        row.addWidget(self._default_btn)

        # --- single lock for the whole device ----------------------------
        self._lock_btn = QPushButton("🔒")
        self._lock_btn.setObjectName("lock")
        self._lock_btn.setCheckable(True)
        self._lock_btn.setChecked(True)  # start locked (read-only, tracks device)
        self._lock_btn.setFixedWidth(28)
        self._lock_btn.toggled.connect(self._on_lock_toggled)
        row.addWidget(self._lock_btn)

        row.addStretch()

        self._stale_label = QLabel("Waiting…")
        self._stale_label.setObjectName("stale")
        self._stale_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(self._stale_label)

        outer.addWidget(card)
        outer.addStretch()

        self._apply_lock(True)
        self._refresh_default_button()

    # ------------------------------------------------------------------ #
    # Lock toggle — purely local: gates whether the spinboxes are editable.
    # Locked = read-only, tracks the live device value.  Never sends
    # anything to the device.  One lock covers the whole device.
    # ------------------------------------------------------------------ #

    def _on_lock_toggled(self, locked: bool) -> None:
        # Re-locking discards any staged (uncommitted) edit so the display
        # snaps back to the device's actual value.
        if locked and self._has_unsaved:
            self._revert_pending()
        self._apply_lock(locked)

    def _apply_lock(self, locked: bool) -> None:
        for spin in (self._freq_spin, self._level_spin):
            spin.setReadOnly(locked)
            spin.setButtonSymbols(
                QAbstractSpinBox.ButtonSymbols.NoButtons
                if locked
                else QAbstractSpinBox.ButtonSymbols.UpDownArrows
            )
        self._default_btn.setEnabled(not locked)
        self._lock_btn.setText("🔒" if locked else "🔓")
        self._lock_btn.setToolTip(
            "Locked — tracks device, editing disabled. Click to edit."
            if locked
            else "Unlocked — edit and press Enter to apply; lock again to discard."
        )
        if not locked:
            self._freq_spin.setFocus()
            self._freq_spin.selectAll()

    # ------------------------------------------------------------------ #
    # Pending-update model
    # ------------------------------------------------------------------ #

    def _mark_unsaved(self, *_args) -> None:
        """A spinbox value changed by the user — stage it until Enter."""
        self._has_unsaved = True
        self._refresh_default_button()
        self._highlight_unsaved()

    def _highlight_unsaved(self) -> None:
        if self._has_unsaved:
            self._freq_spin.setStyleSheet("QDoubleSpinBox { background-color: orange; color: black; }")
            self._level_spin.setStyleSheet("QSpinBox { background-color: orange; color: black; }")
        else:
            self._freq_spin.setStyleSheet("")
            self._level_spin.setStyleSheet("")

    def _refresh_default_button(self) -> None:
        if self._has_unsaved:
            self._default_btn.setText("undo")
            self._default_btn.setStyleSheet(f"background-color: {UNDO_BUTTON_COLOR}; color: black;")
            self._default_btn.setToolTip("Discard staged edit")
        else:
            self._default_btn.setText("default")
            self._default_btn.setStyleSheet(f"background-color: {DEFAULT_BUTTON_COLOR};")
            if self._has_defaults():
                self._default_btn.setToolTip("Load default settings (press Enter to apply)")
            else:
                self._default_btn.setToolTip("No default configured for this consultant")

    def _has_defaults(self) -> bool:
        return (self._client.default_freq_mhz is not None
                or self._client.default_level_dbm is not None)

    def _commit(self) -> None:
        """Enter pressed — send the staged values to the device."""
        if not self._has_unsaved:
            return
        freq = self._freq_spin.value()
        level = self._level_spin.value()
        self._prev_freq = freq
        self._prev_level = level
        self._has_unsaved = False
        self._refresh_default_button()
        self._highlight_unsaved()
        self._client.set_freq(freq)
        self._client.set_level(level)

    def _revert_pending(self) -> None:
        """Discard a staged edit and restore the last known-good values."""
        with QSignalBlocker(self._freq_spin), QSignalBlocker(self._level_spin):
            if self._prev_freq is not None:
                self._freq_spin.setValue(self._prev_freq)
            if self._prev_level is not None:
                self._level_spin.setValue(self._prev_level)
        self._has_unsaved = False
        self._refresh_default_button()
        self._highlight_unsaved()

    def _on_default_undo_clicked(self) -> None:
        if self._has_unsaved:
            self._revert_pending()
            return
        # Stage the configured defaults; require Enter to actually apply them.
        if not self._has_defaults():
            return
        if self._client.default_freq_mhz is not None:
            self._freq_spin.setValue(float(self._client.default_freq_mhz))
        if self._client.default_level_dbm is not None:
            self._level_spin.setValue(int(self._client.default_level_dbm))
        # Force the staged state even if a default equals the current value.
        self._has_unsaved = True
        self._refresh_default_button()
        self._highlight_unsaved()
        self._freq_spin.setFocus()
        self._freq_spin.selectAll()

    # ------------------------------------------------------------------ #
    # State updates from PUB socket
    # ------------------------------------------------------------------ #

    def _on_state(self, msg: dict) -> None:
        if msg.get("serial") != self._client.serial:
            return
        self._last_update = time.monotonic()
        rf_on = msg.get("rf_on", False)
        freq = msg.get("freq_mhz", 0.0)
        level = msg.get("level_dbm", 0)

        self._rf_on = rf_on
        self._rf_btn.setText("ON" if rf_on else "OFF")
        self._rf_btn.setObjectName("rfOn" if rf_on else "rfOff")
        self.style().unpolish(self._rf_btn)
        self.style().polish(self._rf_btn)

        # Never clobber a staged edit; otherwise mirror the live device value.
        # Signals are blocked so mirroring does not itself mark the row unsaved.
        if not self._has_unsaved:
            with QSignalBlocker(self._freq_spin), QSignalBlocker(self._level_spin):
                self._freq_spin.setValue(freq)
                self._level_spin.setValue(int(level))
            self._prev_freq = self._freq_spin.value()
            self._prev_level = self._level_spin.value()

    def _update_stale(self) -> None:
        if self._last_update == 0.0:
            return
        age = time.monotonic() - self._last_update
        self._stale_label.setText(f"Last update: {age:.1f}s ago")

    # ------------------------------------------------------------------ #
    # Controls → REQ/REP commands
    # ------------------------------------------------------------------ #

    def _toggle_rf(self) -> None:
        new_state = not bool(self._rf_on)
        self._client.set_rf(new_state)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        self._stale_timer.stop()
        self._thread.stop_subscriber()
