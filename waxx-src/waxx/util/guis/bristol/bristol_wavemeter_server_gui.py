"""Server-side Qt6 GUI for the Bristol wavemeter server.

Also exports ``BristolDetuningDisplay``, a shared compact widget (f₀
spinbox + live Δ label) used by both the server GUI and the client GUI.
"""
from __future__ import annotations

import sys
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from waxx.util.guis.bristol.bristol_wavemeter_server import BristolWavemeterServer

_POLL_MS = 200
_F0_DEFAULT_THZ = 389.28617  # K-39 D1 line

_DARK_BG     = "#1e1e1e"
_DARK_WIDGET = "#2d2d2d"
_DARK_BORDER = "#444444"

DARK_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {_DARK_BG};
    color: #e0e0e0;
}}
QLabel {{
    color: #e0e0e0;
}}
QPushButton {{
    background-color: {_DARK_WIDGET};
    color: #e0e0e0;
    border: 1px solid {_DARK_BORDER};
    border-radius: 4px;
    padding: 2px 8px;
}}
QPushButton:hover {{
    background-color: #3a3a3a;
}}
QDoubleSpinBox, QSpinBox {{
    background-color: {_DARK_WIDGET};
    color: #e0e0e0;
    border: 1px solid {_DARK_BORDER};
    border-radius: 3px;
    padding: 2px 4px;
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{
    background-color: #3a3a3a;
    border: none;
    width: 16px;
}}
"""


def apply_dark_palette(app: QApplication) -> None:
    """Apply a dark QPalette to *app* (covers native widgets the stylesheet misses)."""
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(_DARK_BG))
    p.setColor(QPalette.ColorRole.WindowText,      QColor("#e0e0e0"))
    p.setColor(QPalette.ColorRole.Base,            QColor(_DARK_WIDGET))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor("#252525"))
    p.setColor(QPalette.ColorRole.Text,            QColor("#e0e0e0"))
    p.setColor(QPalette.ColorRole.Button,          QColor(_DARK_WIDGET))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor("#e0e0e0"))
    p.setColor(QPalette.ColorRole.Highlight,       QColor("#2980b9"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(p)


# ---------------------------------------------------------------------------
# Shared detuning-display widget
# ---------------------------------------------------------------------------

class BristolDetuningDisplay(QWidget):
    """Compact f₀ selector + live detuning label.

    Call ``update_frequency(freq_thz)`` each refresh cycle.  Pass ``None``
    to show dashes.  The widget is self-contained — no external timer needed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        f0_lbl = QLabel("f₀:")
        f0_lbl.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(f0_lbl)

        self._f0_spin = QDoubleSpinBox()
        self._f0_spin.setDecimals(6)
        self._f0_spin.setRange(100.0, 1000.0)
        self._f0_spin.setValue(_F0_DEFAULT_THZ)
        self._f0_spin.setSingleStep(0.001)
        self._f0_spin.setSuffix(" THz")
        self._f0_spin.setFont(QFont("Monospace", 10))
        self._f0_spin.setFixedWidth(175)
        layout.addWidget(self._f0_spin)

        self._det_lbl = QLabel("Δ = — GHz")
        self._det_lbl.setFont(QFont("Monospace", 18, QFont.Weight.Bold))
        self._det_lbl.setStyleSheet("color: #ff6464;")
        self._det_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._det_lbl, 1)

    @property
    def f0_thz(self) -> float:
        return self._f0_spin.value()

    def update_frequency(self, freq_thz: float | None) -> None:
        if freq_thz is not None:
            det = (freq_thz - self.f0_thz) * 1e3
            self._det_lbl.setText(f"Δ = {det:+.3f} GHz")
        else:
            self._det_lbl.setText("Δ = — GHz")

    def update_detuning(
        self,
        mean_ghz: float | None,
        std_mhz: float | None = None,
    ) -> None:
        """Set the detuning label directly with mean ± σ.

        Used by clients that already compute a running average and want
        a one-line "Δ = X ± Y MHz" readout in place of the per-sample
        display.  ``mean_ghz`` is in GHz; ``std_mhz`` in MHz.
        """
        if mean_ghz is None or not (mean_ghz == mean_ghz):  # NaN guard
            self._det_lbl.setText("Δ = — GHz")
            return
        if std_mhz is None or not (std_mhz == std_mhz):
            self._det_lbl.setText(f"Δ = {mean_ghz:+.3f} GHz")
        else:
            self._det_lbl.setText(
                f"Δ = {mean_ghz:+.3f} GHz ± {std_mhz:.2f} MHz"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine_icon() -> QIcon:
    """Draw a simple red sine-wave icon at multiple sizes."""
    import math
    icon = QIcon()
    for size in (16, 24, 32, 48, 64):
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        painter = QPainter(px)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = painter.pen()
        pen.setColor(QColor("#e74c3c"))
        pen.setWidthF(max(1.5, size / 16))
        painter.setPen(pen)
        margin = size * 0.1
        w = size - 2 * margin
        h = size * 0.35
        cy = size / 2
        steps = max(60, size * 2)
        pts = [
            (margin + w * i / steps,
             cy - h * math.sin(2 * math.pi * i / steps))
            for i in range(steps + 1)
        ]
        for i in range(len(pts) - 1):
            painter.drawLine(
                int(pts[i][0]), int(pts[i][1]),
                int(pts[i + 1][0]), int(pts[i + 1][1]),
            )
        painter.end()
        icon.addPixmap(px)
    return icon


class BristolServerGUI(QMainWindow):
    def __init__(self, server: BristolWavemeterServer):
        super().__init__()
        self._server = server
        self.setWindowTitle("Bristol Wavemeter")
        self.setWindowIcon(_make_sine_icon())
        self.setStyleSheet(DARK_STYLESHEET)
        self.setFixedWidth(420)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 6, 10, 6)

        # ── Row 1: status | port | reconnect ──────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self._status_lbl = QLabel("● DISCONNECTED")
        self._status_lbl.setFont(QFont("Monospace", 10))
        self._status_lbl.setStyleSheet("color: #e74c3c; font-weight: bold;")
        row1.addWidget(self._status_lbl)
        row1.addStretch()
        self._port_lbl = QLabel("port: —")
        self._port_lbl.setStyleSheet("color: #555555; font-size: 10px;")
        row1.addWidget(self._port_lbl)
        btn = QPushButton("↺")
        btn.setFixedSize(24, 24)
        btn.setToolTip("Reconnect to wavemeter")
        btn.clicked.connect(self._on_reconnect)
        row1.addWidget(btn)
        layout.addLayout(row1)

        # ── Row 2: f (live) | f₀ spinbox ──────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self._freq_lbl = QLabel("f = — THz")
        self._freq_lbl.setFont(QFont("Monospace", 10))
        self._freq_lbl.setStyleSheet("color: #7aaadd;")
        row2.addWidget(self._freq_lbl)
        row2.addStretch()
        f0_lbl = QLabel("f₀:")
        f0_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        row2.addWidget(f0_lbl)
        self._f0_spin = QDoubleSpinBox()
        self._f0_spin.setDecimals(6)
        self._f0_spin.setRange(100.0, 1000.0)
        self._f0_spin.setValue(_F0_DEFAULT_THZ)
        self._f0_spin.setSingleStep(0.001)
        self._f0_spin.setSuffix(" THz")
        self._f0_spin.setFont(QFont("Monospace", 9))
        self._f0_spin.setFixedWidth(170)
        row2.addWidget(self._f0_spin)
        layout.addLayout(row2)

        # ── Row 3: detuning ───────────────────────────────────────────
        self._det_lbl = QLabel("Δ = — GHz")
        self._det_lbl.setFont(QFont("Monospace", 20, QFont.Weight.Bold))
        self._det_lbl.setStyleSheet("color: #ff6464;")
        self._det_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._det_lbl)

        # ── Footer: age / error ───────────────────────────────────────
        self._footer_lbl = QLabel("")
        self._footer_lbl.setStyleSheet("color: #555555; font-size: 10px;")
        self._footer_lbl.setWordWrap(True)
        layout.addWidget(self._footer_lbl)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(_POLL_MS)
        self._refresh()

    def _refresh(self) -> None:
        reading   = self._server.get_reading()
        status    = self._server.get_status()
        connected = reading.get("connected", False)

        if connected:
            self._status_lbl.setText("● CONNECTED")
            self._status_lbl.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            self._status_lbl.setText("● DISCONNECTED")
            self._status_lbl.setStyleSheet("color: #e74c3c; font-weight: bold;")

        self._port_lbl.setText(f"port: {self._server._waxx_port}")

        freq = reading.get("frequency_thz")
        ts   = reading.get("timestamp")

        self._freq_lbl.setText(f"f = {freq:.6f} THz" if freq is not None else "f = — THz")

        if freq is not None:
            det = (freq - self._f0_spin.value()) * 1e3
            self._det_lbl.setText(f"Δ = {det:+.3f} GHz")
        else:
            self._det_lbl.setText("Δ = — GHz")

        age = f"  {(time.time()-ts)*1000:.0f} ms ago" if ts is not None else ""
        err = status.get("error") or ""
        self._footer_lbl.setText((age + ("  " + err if err else "")).strip())

    def _on_reconnect(self) -> None:
        self._server._disconnect_wavemeter()

    def closeEvent(self, event):
        self._server.stop()
        event.accept()


def main(wavemeter_host: str = "192.168.1.105") -> None:
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "weldlab.kexp.gui.bristol_wavemeter_server"
        )
    except Exception:
        pass

    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_palette(app)

    server = BristolWavemeterServer(wavemeter_host=wavemeter_host)
    server.start()

    gui = BristolServerGUI(server)
    gui.show()

    exit_code = app.exec()
    server.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
