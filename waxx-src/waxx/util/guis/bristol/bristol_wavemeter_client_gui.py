"""Client-side Qt6 GUI for the Bristol wavemeter server.

Compact dark-mode detuning readout.  The \u0394-vs-t history plot sits below
the readout in a collapsible section: the \u25b8/\u25be arrow shows it inline
(collapsed by default, so the panel stays a few lines tall inside the
dashboard), and the \u29c9 button moves it into its own window
(:class:`BristolPlotWindow`).  Closing that window, or \u29c9 again, puts it back.
Imports ``DARK_STYLESHEET`` and ``apply_dark_palette`` from the server GUI
module to avoid duplicating the shared f\u2080 / \u0394 styling.
"""
from __future__ import annotations

import collections
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QToolButton,
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
_PLOT_WINDOW_APP_ID = "weldlab.kexp.gui.bristol_wavemeter_plot"
_POPOUT_TIP = "Pop the Δ-vs-t plot out into its own window (own taskbar entry)"


class BristolPlotWindow(QMainWindow):
    """Parentless top-level window hosting the detuning history plot.

    Parentless on purpose, for the same reasons as the dashboard's
    ``PanelWindow``: a child ``Qt.WindowType.Window`` would be a *tool*
    window of the dashboard (hidden whenever the dashboard is minimised,
    no taskbar button, easy to lose behind it).  This window gets its own
    taskbar entry and Alt-Tab slot and can sit on another monitor.

    The window does not own the plot: :meth:`take` / :meth:`give_back` move
    the owner's plot body in and out, so the same plot (and its history) is
    either embedded in the panel or shown here.  Closing the window only
    **hides** it and emits :attr:`closed_by_user`; the owning
    :class:`BristolDetuningWidget` then docks the plot back and re-shows the
    same window (same geometry) on the next pop-out.  :meth:`shutdown`
    closes it for real when the owner is torn down.
    """

    closed_by_user = pyqtSignal()

    def __init__(self, title: str):
        super().__init__(None)  # parentless: own taskbar entry
        self._shutting_down = False
        self.setWindowTitle(title)
        self.setWindowIcon(_make_sine_icon())
        self.setStyleSheet(DARK_STYLESHEET)
        # A fixed container as the central widget: setCentralWidget() would
        # delete a previous central widget, and the body has to survive
        # moving back to the panel.
        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(container)
        self.resize(640, 360)
        try:
            from waxx.util.dashboard.panel_window import set_window_app_id  # noqa: PLC0415
            set_window_app_id(self, _PLOT_WINDOW_APP_ID)  # best effort, Windows only
        except Exception:  # noqa: BLE001 - purely cosmetic
            pass

    def take(self, body: QWidget) -> None:
        """Host ``body`` (reparents it into this window)."""
        self._layout.addWidget(body)
        body.show()

    def give_back(self, body: QWidget) -> None:
        """Release ``body`` so the owner can re-embed it."""
        self._layout.removeWidget(body)
        body.setParent(None)

    def shutdown(self) -> None:
        """Close for real (owner is going away)."""
        self._shutting_down = True
        self.close()
        self.deleteLater()

    def closeEvent(self, event):  # noqa: N802 (Qt override)
        if self._shutting_down:
            super().closeEvent(event)
            return
        # User closed it: keep the window (and the plot history) around,
        # just hide, and let the owner dock the plot back into the panel.
        self.hide()
        event.ignore()
        self.closed_by_user.emit()


class BristolDetuningWidget(QWidget):
    """Compact detuning plotter — controls along the top, plot in centre,
    shared detuning display at the bottom."""

    def __init__(self):
        super().__init__()
        self._client: BristolWavemeterGuiClient | None = None
        # Bounded to _MAX_HISTORY_S worth of data; deque auto-evicts oldest
        # so the O(n) pop(0) trim loop is no longer needed.
        self._times: collections.deque = collections.deque(maxlen=_MAX_HISTORY_S * (1000 // _POLL_MS))
        self._detunings_ghz: collections.deque = collections.deque(maxlen=_MAX_HISTORY_S * (1000 // _POLL_MS))
        self._start_time = time.time()

        # --- Shared state written by the background poller, read by the GUI ---
        # All socket I/O happens on the poller thread so that a dead/stopped
        # server can never block the Qt event loop (which would freeze the
        # whole dashboard).  The GUI timer only ever reads these cached values.
        self._state_lock = threading.Lock()
        self._latest_freq_thz: float | None = None  # last good reading, or None
        self._server_reachable = False              # did the last poll succeed?
        self._stop_event = threading.Event()

        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(_POLL_MS)

        # Background poller: discovery + blocking reads live entirely here.
        self._poller = threading.Thread(
            target=self._poll_loop, name="BristolPoller", daemon=True
        )
        self._poller.start()

    # ------------------------------------------------------------------
    # Background poller (all network I/O — never touches Qt widgets)
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Discover the server and poll it in a loop, off the GUI thread.

        Every blocking operation (discovery, socket connect, read) runs here,
        so if the server is stopped the GUI stays perfectly responsive — the
        worst that happens is this thread waits on a timeout while the UI keeps
        rendering the last-known state and a red status dot.
        """
        while not self._stop_event.is_set():
            client = self._client
            if client is None:
                try:
                    # Short discovery timeout so we retry reasonably often
                    # without spinning; this only blocks the poller thread.
                    self._client = BristolWavemeterGuiClient(discovery_timeout=2.0)
                except RuntimeError:
                    self._set_state(None, reachable=False)
                    self._stop_event.wait(1.0)  # back off before re-discovering
                    continue
                client = self._client

            try:
                reading = client.get_reading()
                if reading.get("connected") and reading.get("frequency_thz") is not None:
                    self._set_state(reading["frequency_thz"], reachable=True)
                else:
                    # Server answered but the wavemeter itself has no reading.
                    self._set_state(None, reachable=True)
            except Exception:
                # Server went away mid-session — drop the client so the next
                # iteration re-discovers it, and mark the link as down.
                self._client = None
                self._set_state(None, reachable=False)

            self._stop_event.wait(_POLL_MS / 1000.0)

    def _set_state(self, freq_thz: float | None, reachable: bool) -> None:
        with self._state_lock:
            self._latest_freq_thz = freq_thz
            self._server_reachable = reachable

    def stop(self) -> None:
        """Signal the poller thread to exit and close the plot pop-out.

        Called on widget/window close and from the dashboard panel's
        ``cleanup()``; the pop-out is parentless, so nothing else would
        close it when this widget goes away.
        """
        self._stop_event.set()
        win = self._plot_win
        if win is not None:
            if self._popped_out:
                # Take the plot back first so deleting the window does not
                # delete it out from under _update().
                win.give_back(self._plot_body)
                self._embed_layout.addWidget(self._plot_body)
                self._popped_out = False
            self._plot_win = None
            win.shutdown()

    def closeEvent(self, event):  # noqa: N802 (Qt override)
        self.stop()
        super().closeEvent(event)

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

        # ── Row 3: plot header - ▸/▾ embeds inline, ⧉ pops out ───────
        plot_hdr = QHBoxLayout()
        plot_hdr.setSpacing(4)
        self._plot_toggle = QToolButton()
        self._plot_toggle.setText("Plot")
        self._plot_toggle.setCheckable(True)
        self._plot_toggle.setChecked(False)
        self._plot_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._plot_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._plot_toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: 600; padding: 2px 4px; }"
        )
        self._plot_toggle.setToolTip("Show / hide the Δ-vs-t history plot in this panel")
        self._plot_toggle.clicked.connect(self._on_plot_toggle)
        plot_hdr.addWidget(self._plot_toggle)
        plot_hdr.addStretch(1)
        self._popout_btn = QToolButton()
        self._popout_btn.setText("⧉")
        self._popout_btn.setAutoRaise(True)
        self._popout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._popout_btn.setFixedSize(22, 22)
        self._popout_btn.setToolTip(_POPOUT_TIP)
        self._popout_btn.clicked.connect(self._on_popout_clicked)
        plot_hdr.addWidget(self._popout_btn)
        root.addLayout(plot_hdr)

        # Inline host for the plot body.  Stretch 1 so an expanded plot
        # takes the spare height; while it is hidden the trailing stretch
        # (factor 0) keeps the rows packed at the top.
        self._embed_host = QWidget()
        self._embed_layout = QVBoxLayout(self._embed_host)
        self._embed_layout.setContentsMargins(0, 0, 0, 0)
        self._embed_host.setVisible(False)
        root.addWidget(self._embed_host, 1)
        root.addStretch(0)

        # ── Plot body: averaging controls + plot ─────────────────────
        # Built once, up front, so the N spinbox exists for the readout
        # even before the plot has ever been shown.  It lives either in
        # _embed_host or in the pop-out window, never both.  History is
        # collected regardless; the curve is only redrawn while visible.
        plot_body = QWidget()
        plot_layout = QVBoxLayout(plot_body)
        plot_layout.setContentsMargins(8, 6, 8, 6)
        plot_layout.setSpacing(4)

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
        # Add Δ = 0 reference line (horizontal, only visible if in y-range).
        # ignoreBounds keeps it out of autorange so it never pins the view to 0.
        zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color="#666666", style=pg.QtCore.Qt.PenStyle.DashLine, width=1))
        self._plot.addItem(zero_line, ignoreBounds=True)
        self._plot.setMinimumHeight(180)

        plot_layout.addWidget(ctl_wrap)
        plot_layout.addWidget(self._plot, 1)
        self._plot_body = plot_body
        self._embed_layout.addWidget(plot_body)
        self._popped_out = False

        self._plot_win: BristolPlotWindow | None = BristolPlotWindow(
            "Bristol Wavemeter — Δ history"
        )
        self._plot_win.closed_by_user.connect(self.dock_plot)

        # Hidden average label kept for back-compat with old _clear() code path.
        self._avg_lbl = QLabel("")
        self._avg_lbl.setVisible(False)

    # ------------------------------------------------------------------
    # Plot placement: embedded (▸/▾) or popped out (⧉)
    # ------------------------------------------------------------------

    def set_plot_expanded(self, expanded: bool) -> None:
        """Show / hide the plot inline.  No-op while it is popped out."""
        if self._popped_out:
            return
        expanded = bool(expanded)
        self._plot_toggle.setChecked(expanded)
        self._plot_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._embed_host.setVisible(expanded)
        if expanded:
            self._redraw_curve()

    def _on_plot_toggle(self) -> None:
        self.set_plot_expanded(self._plot_toggle.isChecked())

    def _on_popout_clicked(self) -> None:
        if self._popped_out:
            self.dock_plot()
        else:
            self.show_plot_window()

    def show_plot_window(self) -> None:
        """Move the plot into the pop-out window and show / raise it."""
        win = self._plot_win
        if win is None:  # after stop()
            return
        if not self._popped_out:
            self._embed_layout.removeWidget(self._plot_body)
            win.take(self._plot_body)
            self._popped_out = True
            self._embed_host.setVisible(False)
            self._plot_toggle.setEnabled(False)
            self._plot_toggle.setArrowType(Qt.ArrowType.RightArrow)
            self._plot_toggle.setText("Plot (popped out)")
            self._popout_btn.setToolTip("Return the plot to this panel")
        self._redraw_curve()
        win.show()
        win.raise_()
        win.activateWindow()

    def dock_plot(self) -> None:
        """Move the plot back into the panel, shown inline.

        Called by ⧉ while popped out and when the user closes the pop-out
        window; the window itself is kept (hidden) for the next pop-out.
        """
        win = self._plot_win
        if not self._popped_out or win is None:
            return
        win.hide()
        win.give_back(self._plot_body)
        self._embed_layout.addWidget(self._plot_body)
        self._plot_body.show()
        self._popped_out = False
        self._plot_toggle.setEnabled(True)
        self._plot_toggle.setText("Plot")
        self._popout_btn.setToolTip(_POPOUT_TIP)
        self.set_plot_expanded(True)

    def _redraw_curve(self) -> None:
        self._curve.setData(list(self._times), list(self._detunings_ghz))

    def _update(self) -> None:
        # Non-blocking: read only the cached values published by the poller
        # thread.  This runs on the Qt event loop, so it must never do socket
        # I/O - otherwise a stopped server would freeze the whole dashboard.
        with self._state_lock:
            freq_thz = self._latest_freq_thz
            reachable = self._server_reachable

        t = time.time() - self._start_time

        if freq_thz is not None:
            self._wm_lbl.setText(f"f = {freq_thz:.6f} THz")
            self._status_lbl.setText("\u25cf")
            self._status_lbl.setStyleSheet("color: #2ecc71; font-size: 10px;")
        elif reachable:
            # Server reachable but the wavemeter itself has no reading.
            self._wm_lbl.setText("f = \u2014 THz")
            self._status_lbl.setText("\u25cf")
            self._status_lbl.setStyleSheet("color: #e67e22; font-size: 10px;")
        else:
            # Server unreachable / stopped.
            self._wm_lbl.setText("f = \u2014 THz")
            self._status_lbl.setText("\u25cf")
            self._status_lbl.setStyleSheet("color: #e74c3c; font-size: 10px;")

        # NOTE: detuning label is updated below from the running-average
        # block as "Δ = mean ± σ".

        f0 = self._f0_spin.value()
        det = (freq_thz - f0) * 1e3 if freq_thz is not None else float("nan")
        self._times.append(t)
        self._detunings_ghz.append(det)

        # History is always recorded; only redraw while someone can see it
        # (inline and expanded, or in the shown pop-out window).
        if self._plot_body.isVisible():
            self._redraw_curve()

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
        self._widget = BristolDetuningWidget()
        self.setCentralWidget(self._widget)

    def closeEvent(self, event):  # noqa: N802 (Qt override)
        # Child widgets don't receive closeEvent, so stop the poller here.
        self._widget.stop()
        super().closeEvent(event)


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

