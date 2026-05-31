"""Client-side Keysight monitor GUI.

This widget talks ONLY to ``KeysightServer`` over TCP — it never opens a
direct VXI11 connection to the supplies.  Multiple dashboards can run
this widget concurrently without spamming the hardware.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from waxx.util.guis.keysight.keysight_client import KeysightClient

T_UPDATE_MS = 500
FONTSIZE_PT = 14

# Per-supply over-current alert thresholds (A), keyed by ``max_current``.
ALERT_THRESHOLDS: dict[int, float] = {500: 100, 170: 50}


class _StatusDecoder:
    """Decode the QUEStionable condition register into a short label."""

    _BITS = {
        0: "OV", 1: "OC", 2: "PF", 3: "CP", 4: "OT",
        5: "MSP", 6: "", 7: "", 8: "", 9: "INH", 10: "UNR",
    }

    def decode(self, status: int) -> str:
        out = []
        for bit, name in self._BITS.items():
            if name and ((status >> bit) & 1):
                out.append(name)
        return " ".join(out)


class _SupplyRow(QWidget):
    """One row: label + value/action button.

    The button text and click handler change with the supply state — it
    doubles as the status indicator and the "fix it" action button.
    """

    def __init__(self, client: KeysightClient, ip: str, max_current: int,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._client = client
        self._ip = ip
        self._max_current = int(max_current)
        self._alert_threshold = ALERT_THRESHOLDS.get(self._max_current)
        self._decoder = _StatusDecoder()
        self._connected = False
        self._output_on: Optional[bool] = None
        self._status = 0
        self._err_str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        self.value_btn = QPushButton("…")
        self.value_btn.clicked.connect(self._on_click)
        font = QFont()
        font.setPointSize(FONTSIZE_PT)
        font.setBold(True)
        fixed_w = QFontMetrics(font).horizontalAdvance("000.00 A") + 20
        self.value_btn.setFixedWidth(fixed_w)

        text_label = QLabel(f"{self._max_current} A supply current = ")
        text_label.setStyleSheet(f"font-size: {FONTSIZE_PT}pt;")

        self.layout = QHBoxLayout()
        self.layout.addWidget(text_label)
        self.layout.addWidget(self.value_btn)

    # ------------------------------------------------------------------ #

    def apply_snapshot(self, snap: dict) -> None:
        self._connected = bool(snap.get("connected"))
        self._output_on = snap.get("output_on")
        self._status = int(snap.get("status") or 0)
        current = snap.get("current_a")

        if not self._connected:
            self._set_value("CXN_ERR", "orange")
            return
        if self._status:
            self._err_str = self._decoder.decode(self._status)
            self._set_value(self._err_str or f"STAT 0x{self._status:X}", "")
            return
        if self._output_on is False:
            self._set_value("OFF", "orange")
            return
        if current is None:
            self._set_value("…", "")
            return
        alert = (
            self._alert_threshold is not None
            and float(current) > self._alert_threshold
        )
        self._set_value(f"{float(current):1.2f} A", "red" if alert else "")

    def _set_value(self, text: str, bg: str) -> None:
        self.value_btn.setText(text)
        self.value_btn.setStyleSheet(
            f"font-weight: bold; font-size: {FONTSIZE_PT}pt; "
            f"text-align: right; padding-right: 10px; "
            f"background-color: {bg};"
        )

    def _on_click(self) -> None:
        try:
            if not self._connected:
                self._client.reconnect(self._ip)
            elif self._output_on is False:
                self._client.turn_on(self._ip)
            else:
                # Connected + on: assume the click is to clear protection.
                self._client.clear_protect(self._ip)
        except Exception as exc:
            print(f"[Keysight] RPC failed for {self._ip}: {exc}")


class KeysightClientWindow(QWidget):
    """Client GUI: discovers ``KeysightServer`` and renders one row per supply."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._client: Optional[KeysightClient] = None
        self._rows: dict[str, _SupplyRow] = {}
        self._error_label = QLabel("Connecting to keysight server…")
        self._error_label.setStyleSheet(
            f"font-size: {FONTSIZE_PT}pt; color: #b22222; font-weight: bold;"
        )

        self._root = QVBoxLayout(self)
        self._root.addWidget(self._error_label)
        self._error_label.hide()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(T_UPDATE_MS)
        # Kick once at startup so the user sees data quickly.
        QTimer.singleShot(50, self._refresh)

    # ------------------------------------------------------------------ #

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            self._client = KeysightClient(timeout_s=2.0, discovery_timeout=0.5)
            return True
        except Exception as exc:
            self._show_error(f"keysight server not found: {exc}")
            return False

    def _refresh(self) -> None:
        if not self._ensure_client():
            return
        try:
            snapshot = self._client.get_snapshot()
        except Exception as exc:
            # Lost server — force rediscovery on next tick.
            self._client = None
            self._show_error(f"keysight server unreachable: {exc}")
            return
        self._hide_error()
        self._apply(snapshot)

    def _apply(self, snapshot: list[dict]) -> None:
        # Lazily build a row per supply on first snapshot.
        for snap in snapshot:
            ip = str(snap.get("ip"))
            row = self._rows.get(ip)
            if row is None:
                row = _SupplyRow(self._client, ip, int(snap.get("max_current", 0)), self)
                self._rows[ip] = row
                self._root.addLayout(row.layout)
            row.apply_snapshot(snap)

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.show()

    def _hide_error(self) -> None:
        if self._error_label.isVisible():
            self._error_label.hide()

    # ------------------------------------------------------------------ #

    def closeEvent(self, event):  # noqa: N802 - Qt-style
        self._timer.stop()
        event.accept()


__all__ = ["KeysightClientWindow"]
