"""Client-side Qt6 GUI for the Bristol wavemeter server.

Compact dark-mode detuning plotter.  Imports ``BristolDetuningDisplay``,
``DARK_STYLESHEET``, and ``apply_dark_palette`` from the server GUI module
to avoid duplicating the shared f\u2080 / \u0394 widget.
"""
from __future__ import annotations

import collections
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
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

from waxx.util.guis.bristol.bristol_wavemeter_client import BristolWavemeterGuiClient
from waxx.util.guis.bristol.bristol_wavemeter_server_gui import (
    DARK_STYLESHEET,
    _F0_DEFAULT_THZ,
    _make_sine_icon,
    apply_dark_palette,
)

_POLL_MS = 100
_MAX_HISTORY_S = 120


class BristolDetuningWidget(QWidget):
    """Compact detuning plotter — controls along the top, plot in centre,
    shared detuning display at the bottom."""

    def __init__(self):
        super().__init__()
        self._client: BristolWavemeterGuiClient | None = None
        self._connecting = False
        # Bounded to _MAX_HISTORY_S worth of data; deque auto-evicts oldest
        # so the O(n) pop(0) trim loop is no longer needed.
        self._times: collections.deque = collections.deque(maxlen=_MAX_HISTORY_S * (1000 // _POLL_MS))
        self._detunings_ghz: collections.deque = collections.deque(maxlen=_MAX_HISTORY_S * (1000 // _POLL_MS))
        self._start_time = time.time()
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(_POLL_MS)
        self._try_connect()

    # ------------------------------------------------------------------
    # Deferred connection
    # ------------------------------------------------------------------

    def _try_connect(self) -> None:
        """Spawn a background thread to discover and connect to the server."""
        if self._connecting:
            return
        self._connecting = True
        self._status_lbl.setText("◌")
        self._status_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self) -> None:
        try:
            self._client = BristolWavemeterGuiClient()
        except RuntimeError:
            pass  # _update will retry via _try_connect
        finally:
            self._connecting = False

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        # ── Row 1: live frequency + status + f₀ reference spinbox ────
        # All on one row so the panel can stay narrow; the (taller)
        # detuning readout drops below on its own line.
        top = QHBoxLayout()
        top.setSpacing(8)

        self._wm_lbl = QLabel("f = \u2014 THz")
        self._wm_lbl.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        self._wm_lbl.setStyleSheet("color: #44aaff;")
        top.addWidget(self._wm_lbl)

        self._status_lbl = QLabel("\u25cf")
        self._status_lbl.setStyleSheet("color: #555555; font-size: 10px;")
        top.addWidget(self._status_lbl)

        top.addStretch(1)

        f0_lbl = QLabel("f\u2080:")
        f0_lbl.setStyleSheet("color: #888888; font-size: 11px;")
        top.addWidget(f0_lbl)

        self._f0_spin = QDoubleSpinBox()
        self._f0_spin.setDecimals(6)
        self._f0_spin.setRange(100.0, 1000.0)
        self._f0_spin.setValue(_F0_DEFAULT_THZ)
        self._f0_spin.setSingleStep(0.001)
        self._f0_spin.setSuffix(" THz")
        self._f0_spin.setFont(QFont("Monospace", 10))
        self._f0_spin.setFixedWidth(150)
        top.addWidget(self._f0_spin)

        root.addLayout(top)

        # ── Row 2: detuning readout on its own line (large) ──────────
        self._det_lbl = QLabel("\u0394 = \u2014 GHz")
        self._det_lbl.setFont(QFont("Monospace", 13, QFont.Weight.Bold))
        self._det_lbl.setStyleSheet("color: #ff6464;")
        self._det_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._det_lbl)

        # ── Collapsible plot + averaging controls ────────────────────
        try:
            from waxx.util.dashboard.widgets import CollapsibleGroupBox  # noqa: PLC0415
            plot_box = CollapsibleGroupBox("Plot", expanded=False)
        except Exception:
            plot_box = QWidget()
            QVBoxLayout(plot_box)

        ctl_row = QHBoxLayout()
        ctl_row.setSpacing(6)
        n_lbl = QLabel("N:")
        n_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        ctl_row.addWidget(n_lbl)
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 1000)
        self._n_spin.setValue(1)
        self._n_spin.setFixedWidth(70)
        self._n_spin.setFont(QFont("Monospace", 10))
        ctl_row.addWidget(self._n_spin)
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(22)
        clear_btn.clicked.connect(self._clear)
        ctl_row.addWidget(clear_btn)
        ctl_row.addStretch(1)
        ctl_wrap = QWidget()
        ctl_wrap.setLayout(ctl_row)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", "\u0394f", units="GHz")
        self._plot.setLabel("bottom", "t", units="s")
        self._plot.getAxis("left").setStyle(tickFont=pg.Qt.QtGui.QFont("Monospace", 9))
        self._plot.getAxis("bottom").setStyle(tickFont=pg.Qt.QtGui.QFont("Monospace", 9))
        self._curve = self._plot.plot(pen=pg.mkPen("#ffaa00", width=1.5))
        # Add Δ = 0 reference line (horizontal, only visible if in y-range)
        zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color="#666666", style=pg.Qt.PenStyle.DashLine, width=1))
        self._plot.addItem(zero_line)
        self._plot.setMinimumHeight(180)

        if hasattr(plot_box, "addWidget"):
            plot_box.addWidget(ctl_wrap)
            plot_box.addWidget(self._plot)
        else:
            plot_box.layout().addWidget(ctl_wrap)
            plot_box.layout().addWidget(self._plot)

        root.addWidget(plot_box, 1)

        # Hidden average label kept for back-compat with old _clear() code path.
        self._avg_lbl = QLabel("")
        self._avg_lbl.setVisible(False)

    def _update(self) -> None:
        if self._client is None:
            if not self._connecting:
                self._try_connect()
            self._wm_lbl.setText("f = \u2014 THz")
            return

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

        # NOTE: detuning label is updated below from the running-average
        # block as "Δ = mean ± σ".

        f0 = self._f0_spin.value()
        det = (freq_thz - f0) * 1e3 if freq_thz is not None else float("nan")
        self._times.append(t)
        self._detunings_ghz.append(det)

        self._curve.setData(list(self._times), list(self._detunings_ghz))

        N = self._n_spin.value()
        recent = [v for v in list(self._detunings_ghz)[-N:] if not np.isnan(v)]
        if recent:
            avg = float(np.mean(recent))
            std = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
            std_mhz = std * 1e3
            self._det_lbl.setText(
                f"\u0394 = {avg:+.3f} GHz \u00b1 {std_mhz:.2f} MHz"
            )
            self._avg_lbl.setText(f"\u0394\u0304={avg:+.3f} GHz  \u03c3={std_mhz:.2f} MHz")
        else:
            self._det_lbl.setText("\u0394 = \u2014 GHz")
            self._avg_lbl.setText("avg: \u2014")

    def _clear(self) -> None:
        self._times.clear()
        self._detunings_ghz.clear()
        self._start_time = time.time()  # reset so deque maxlen stays consistent
        self._curve.clear()
        self._plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self._avg_lbl.setText("avg: \u2014")


class BristolClientWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bristol Wavemeter — Detuning")
        self.setWindowIcon(_make_sine_icon())
        self.setStyleSheet(DARK_STYLESHEET)
        self.setMinimumSize(320, 200)
        self.setCentralWidget(BristolDetuningWidget())


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
    win = BristolClientWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

