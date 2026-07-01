"""TpiDeviceWidget — PyQt6 control panel for one remote TPI-1005-A.

Shows live RF state (switch, frequency, level) received over the server's ZMQ
PUB socket and provides controls to change each parameter via ZMQ REQ/REP.
"""
from __future__ import annotations

import time
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
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
QLabel#header {
    font-size: 13px;
    font-weight: bold;
    color: #a0a8d0;
}
QLabel#value {
    font-size: 22px;
    font-weight: bold;
    color: #e8eaf6;
}
QLabel#unit {
    font-size: 12px;
    color: #7a7e9a;
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
    padding: 6px 18px;
}
QPushButton#rfOff {
    background: #4d1b1b;
    color: #ef5350;
    border: 1px solid #ef5350;
    border-radius: 4px;
    font-size: 14px;
    font-weight: bold;
    padding: 6px 18px;
}
QPushButton#set {
    background: #2a2d4a;
    color: #90caf9;
    border: 1px solid #5c7aaa;
    border-radius: 4px;
    padding: 4px 10px;
}
QPushButton#set:hover { background: #353860; }
QDoubleSpinBox, QSpinBox {
    background: #12132a;
    color: #d0d4e8;
    border: 1px solid #3a3c5a;
    border-radius: 3px;
    padding: 3px 6px;
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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Header
        header = QLabel(self._client.display_name)
        header.setObjectName("header")
        outer.addWidget(header)

        card = QFrame()
        card.setObjectName("card")
        grid = QGridLayout(card)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setSpacing(10)

        # --- RF switch ---------------------------------------------------
        grid.addWidget(QLabel("RF output"), 0, 0)

        self._rf_btn = QPushButton("OFF")
        self._rf_btn.setObjectName("rfOff")
        self._rf_btn.setFixedWidth(90)
        self._rf_btn.clicked.connect(self._toggle_rf)
        grid.addWidget(self._rf_btn, 0, 1, 1, 2)

        # --- Frequency ---------------------------------------------------
        grid.addWidget(QLabel("Frequency"), 1, 0)

        self._freq_val = QLabel("—")
        self._freq_val.setObjectName("value")
        grid.addWidget(self._freq_val, 1, 1)

        grid.addWidget(QLabel("MHz"), 1, 2)

        freq_row = QHBoxLayout()
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(35.0, 4400.0)
        self._freq_spin.setDecimals(3)
        self._freq_spin.setSingleStep(1.0)
        self._freq_spin.setFixedWidth(110)
        freq_row.addWidget(self._freq_spin)

        freq_set = QPushButton("Set")
        freq_set.setObjectName("set")
        freq_set.clicked.connect(self._set_freq)
        freq_row.addWidget(freq_set)
        freq_row.addStretch()
        grid.addLayout(freq_row, 2, 1, 1, 2)

        # --- Level -------------------------------------------------------
        grid.addWidget(QLabel("Level"), 3, 0)

        self._level_val = QLabel("—")
        self._level_val.setObjectName("value")
        grid.addWidget(self._level_val, 3, 1)

        grid.addWidget(QLabel("dBm"), 3, 2)

        level_row = QHBoxLayout()
        self._level_spin = QSpinBox()
        self._level_spin.setRange(-90, 10)
        self._level_spin.setFixedWidth(80)
        level_row.addWidget(self._level_spin)

        level_set = QPushButton("Set")
        level_set.setObjectName("set")
        level_set.clicked.connect(self._set_level)
        level_row.addWidget(level_set)
        level_row.addStretch()
        grid.addLayout(level_row, 4, 1, 1, 2)

        grid.setColumnStretch(1, 1)
        outer.addWidget(card)

        self._stale_label = QLabel("Waiting for data…")
        self._stale_label.setObjectName("stale")
        self._stale_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        outer.addWidget(self._stale_label)

        outer.addStretch()

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
        self._rf_btn.setStyleSheet(self._rf_btn.styleSheet())  # force re-polish
        self.style().unpolish(self._rf_btn)
        self.style().polish(self._rf_btn)

        self._freq_val.setText(f"{freq:.3f}")
        if not self._freq_spin.hasFocus():
            self._freq_spin.setValue(freq)

        self._level_val.setText(str(level))
        if not self._level_spin.hasFocus():
            self._level_spin.setValue(level)

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

    def _set_freq(self) -> None:
        self._client.set_freq(self._freq_spin.value())

    def _set_level(self) -> None:
        self._client.set_level(self._level_spin.value())

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        self._stale_timer.stop()
        self._thread.stop_subscriber()
