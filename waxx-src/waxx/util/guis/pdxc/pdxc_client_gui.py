"""Client GUI for the PDXC piezo stage controller server.

Shows a labelled panel with "In" / "Out" buttons and a step-size spin box
under a "beamsplitter position" heading.  The step size is stored on the
server, so it survives GUI and server restarts and is shared by every
client; the spin box is seeded from the server on connect.

Network I/O is dispatched to ``QThreadPool`` so the GUI thread never
stalls.  Controls grey out while a move is in progress and the status
line updates accordingly.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
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

from waxx.control.misc.pdxc import MAX_PULSES, MIN_PULSES, PDXC_Client

FONTSIZE_PT = 13
STARTUP_GRACE_S = 10.0
RECONNECT_INTERVAL_S = 5.0


# ---------------------------------------------------------------------------
# Background worker helpers
# ---------------------------------------------------------------------------

class _Signals(QObject):
    done = pyqtSignal(object)    # result (any type)
    failed = pyqtSignal(str)     # error message


class _BgCall(QRunnable):
    """Run ``func()`` on a QThreadPool thread; emit signals for GUI-thread pickup."""

    def __init__(self, func) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._func = func
        self.signals = _Signals()

    def run(self) -> None:
        try:
            self.signals.done.emit(self._func())
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class PDXCClientWidget(QWidget):
    """Panel: 'beamsplitter position' label + In / Out buttons + status line."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._client: Optional[PDXC_Client] = None
        self._start_time = time.monotonic()
        self._last_connect_attempt = float("-inf")
        self._connect_in_flight = False
        self._suppress_step_signal = False
        self._build_ui()

        # Poll for the server until it appears.  _try_connect() self-throttles
        # to RECONNECT_INTERVAL_S, so a short tick just keeps retrying when the
        # server starts (or restarts) after the GUI does.
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setInterval(1000)
        self._reconnect_timer.timeout.connect(self._try_connect)
        self._reconnect_timer.start()
        self._try_connect()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title_font = QFont()
        title_font.setPointSize(FONTSIZE_PT)
        title_font.setBold(True)

        btn_font = QFont()
        btn_font.setPointSize(FONTSIZE_PT)

        status_font = QFont()
        status_font.setPointSize(FONTSIZE_PT - 2)

        lbl = QLabel("beamsplitter position")
        lbl.setFont(title_font)
        root.addWidget(lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_in = QPushButton("In")
        self._btn_in.setFont(btn_font)
        self._btn_in.setFixedHeight(36)
        self._btn_in.clicked.connect(self._on_in)
        btn_row.addWidget(self._btn_in)

        self._btn_out = QPushButton("Out")
        self._btn_out.setFont(btn_font)
        self._btn_out.setFixedHeight(36)
        self._btn_out.clicked.connect(self._on_out)
        btn_row.addWidget(self._btn_out)

        root.addLayout(btn_row)

        step_row = QHBoxLayout()
        step_row.setSpacing(10)

        step_lbl = QLabel("step size")
        step_lbl.setFont(btn_font)
        step_row.addWidget(step_lbl)

        self._step_spin = QSpinBox()
        self._step_spin.setFont(btn_font)
        self._step_spin.setRange(MIN_PULSES, MAX_PULSES)
        self._step_spin.setSingleStep(1000)
        self._step_spin.setSuffix(" pulses")
        self._step_spin.setValue(MAX_PULSES)
        self._step_spin.setKeyboardTracking(False)   # commit on Enter / focus-out
        self._step_spin.valueChanged.connect(self._on_step_changed)
        step_row.addWidget(self._step_spin)

        root.addLayout(step_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setFont(status_font)
        self._status_lbl.setStyleSheet("color: gray;")
        root.addWidget(self._status_lbl)

        self._set_buttons_enabled(False)
        self._set_status("connecting...", "gray")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _try_connect(self) -> None:
        now = time.monotonic()
        if self._connect_in_flight:
            return
        if (now - self._last_connect_attempt) < RECONNECT_INTERVAL_S:
            return
        self._last_connect_attempt = now
        self._connect_in_flight = True

        worker = _BgCall(lambda: PDXC_Client(discovery_timeout=4.0))
        worker.signals.done.connect(self._on_client_ready)
        worker.signals.failed.connect(self._on_connect_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_client_ready(self, client) -> None:
        self._connect_in_flight = False
        self._reconnect_timer.stop()
        self._client = client
        self._set_buttons_enabled(True)
        self._set_status(f"ready  ({client.host}:{client.port})", "green")

        worker = _BgCall(client.get_step_size)
        worker.signals.done.connect(self._on_step_loaded)
        QThreadPool.globalInstance().start(worker)

    def _on_step_loaded(self, value) -> None:
        """Show the server-side step size without echoing it straight back."""
        self._suppress_step_signal = True
        try:
            self._step_spin.setValue(int(value))
        finally:
            self._suppress_step_signal = False

    def _on_step_changed(self, value: int) -> None:
        """Persist the new step size on the server."""
        if self._client is None or self._suppress_step_signal:
            return
        client = self._client
        worker = _BgCall(lambda: client.set_step_size(value))
        worker.signals.failed.connect(
            lambda err: self._set_status(f"step size not saved: {err}", "red")
        )
        QThreadPool.globalInstance().start(worker)

    def _on_connect_failed(self, err: str) -> None:
        self._connect_in_flight = False
        elapsed = time.monotonic() - self._start_time
        if elapsed > STARTUP_GRACE_S:
            self._set_status("server not found — retrying...", "orange")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._btn_in.setEnabled(enabled)
        self._btn_out.setEnabled(enabled)
        self._step_spin.setEnabled(enabled)

    def _set_status(self, text: str, color: str) -> None:
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"color: {color};")

    def _on_in(self) -> None:
        self._start_move("in")

    def _on_out(self) -> None:
        self._start_move("out")

    def _start_move(self, direction: str) -> None:
        if self._client is None:
            self._set_status("not connected", "red")
            return
        self._set_buttons_enabled(False)
        self._set_status(f"moving {direction}...", "#c8a000")

        client = self._client
        steps = self._step_spin.value()
        move = client.move_in if direction == "in" else client.move_out
        worker = _BgCall(lambda: move(steps))
        worker.signals.done.connect(self._on_move_done)
        worker.signals.failed.connect(self._on_move_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_move_done(self, result) -> None:
        self._set_buttons_enabled(True)
        self._set_status("done", "green")

    def _on_move_failed(self, err: str) -> None:
        self._set_buttons_enabled(True)
        self._set_status(f"error: {err}", "red")
        if "unreachable" in err:
            # server went away mid-move: drop the client and resume polling
            self._client = None
            self._set_buttons_enabled(False)
            self._reconnect_timer.start()


# ---------------------------------------------------------------------------
# Standalone launcher
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("PDXC Beamsplitter Control")
    win.setCentralWidget(PDXCClientWidget())
    win.resize(320, 140)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
