"""Client-side Qt6 GUI for the Bristol wavemeter server.

Compact dark-mode detuning plotter.  Imports ``BristolDetuningDisplay``,
``DARK_STYLESHEET``, and ``apply_dark_palette`` from the server GUI module
to avoid duplicating the shared f\u2080 / \u0394 widget.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from waxx.util.guis.bristol.bristol_wavemeter_client import BristolWavemeterGuiClient
from waxx.util.guis.bristol.bristol_wavemeter_server_gui import (
    DARK_STYLESHEET,
    BristolDetuningDisplay,
    _make_sine_icon,
    apply_dark_palette,
)

_POLL_MS = 100
_MAX_HISTORY_S = 120


class BristolDetuningWidget(QWidget):
    """Compact detuning plotter — controls along the top, plot in centre,
    shared detuning display at the bottom."""

    def __init__(self, client: BristolWavemeterGuiClient):
        super().__init__()
        self._client = client
        self._times: list[float] = []
        self._detunings_ghz: list[float] = []
        self._start_time = time.time()
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(_POLL_MS)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        # \u2500\u2500 Top control bar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n        top = QHBoxLayout()\n        top.setSpacing(6)

        self._wm_lbl = QLabel("f = \u2014 THz")
        self._wm_lbl.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        self._wm_lbl.setStyleSheet("color: #44aaff;")
        top.addWidget(self._wm_lbl)

        top.addStretch()

        n_lbl = QLabel("N:")
        n_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        top.addWidget(n_lbl)
        self._n_spin = QSpinBox()
        self._n_spin.setRange(5, 1000)
        self._n_spin.setValue(100)
        self._n_spin.setFixedWidth(70)
        self._n_spin.setFont(QFont("Monospace", 10))
        top.addWidget(self._n_spin)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(26)
        clear_btn.clicked.connect(self._clear)
        top.addWidget(clear_btn)

        self._status_lbl = QLabel("\u25cf")
        self._status_lbl.setStyleSheet("color: #555555; font-size: 10px;")
        top.addWidget(self._status_lbl)

        root.addLayout(top)

        # \u2500\u2500 Plot \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", "\u0394f", units="GHz")
        self._plot.setLabel("bottom", "t", units="s")
        self._plot.getAxis("left").setStyle(tickFont=pg.Qt.QtGui.QFont("Monospace", 9))
        self._plot.getAxis("bottom").setStyle(tickFont=pg.Qt.QtGui.QFont("Monospace", 9))
        self._curve = self._plot.plot(pen=pg.mkPen("#ffaa00", width=1.5))
        root.addWidget(self._plot, 1)

        # \u2500\u2500 Avg / std bar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n        self._avg_lbl = QLabel("avg: \u2014")
        self._avg_lbl.setFont(QFont("Monospace", 11, QFont.Weight.Bold))
        self._avg_lbl.setStyleSheet("color: #ff8888;")
        self._avg_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        root.addWidget(self._avg_lbl)

        # \u2500\u2500 Shared f\u2080 + \u0394 display \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n        self._det_display = BristolDetuningDisplay()
        root.addWidget(self._det_display)

    def _update(self) -> None:
        t = time.time() - self._start_time
        freq_thz = None

        try:
            reading = self._client.get_reading()
            if reading.get("connected") and reading.get("frequency_thz") is not None:
                freq_thz = reading["frequency_thz"]
                self._wm_lbl.setText(f"f = {freq_thz:.6f} THz")
                self._status_lbl.setText("\u25cf")
                self._status_lbl.setStyleSheet("color: #2ecc71; font-size: 10px;")
            else:
                self._wm_lbl.setText("f = \u2014 THz")
                self._status_lbl.setText("\u25cf")
                self._status_lbl.setStyleSheet("color: #e67e22; font-size: 10px;")
        except Exception:
            self._wm_lbl.setText("f = \u2014 THz")
            self._status_lbl.setText("\u25cf")
            self._status_lbl.setStyleSheet("color: #e74c3c; font-size: 10px;")

        self._det_display.update_frequency(freq_thz)

        f0 = self._det_display.f0_thz
        det = (freq_thz - f0) * 1e3 if freq_thz is not None else float("nan")
        self._times.append(t)
        self._detunings_ghz.append(det)

        cutoff = t - _MAX_HISTORY_S
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)
            self._detunings_ghz.pop(0)

        self._curve.setData(self._times, self._detunings_ghz)

        N = self._n_spin.value()
        recent = [v for v in self._detunings_ghz[-N:] if not np.isnan(v)]
        if recent:
            avg = np.mean(recent)
            std = np.std(recent, ddof=1) if len(recent) > 1 else 0.0
            self._avg_lbl.setText(f"\u0394\u0304={avg:+.4f} GHz  \u03c3={std*1e3:.2f} MHz")
        else:
            self._avg_lbl.setText("avg: \u2014")

    def _clear(self) -> None:
        self._times.clear()
        self._detunings_ghz.clear()
        self._start_time = time.time()
        self._curve.clear()
        self._plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self._avg_lbl.setText("avg: \u2014")


class BristolClientWindow(QMainWindow):
    def __init__(self, client: BristolWavemeterGuiClient):
        super().__init__()
        self.setWindowTitle("Bristol Wavemeter — Detuning")
        self.setWindowIcon(_make_sine_icon())
        self.setStyleSheet(DARK_STYLESHEET)
        self.setMinimumSize(500, 380)
        self.setCentralWidget(BristolDetuningWidget(client))


def main() -> None:
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "weldlab.kexp.gui.bristol_wavemeter_client"
        )
    except Exception:
        pass

    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_palette(app)
    client = BristolWavemeterGuiClient()
    win = BristolClientWindow(client)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

