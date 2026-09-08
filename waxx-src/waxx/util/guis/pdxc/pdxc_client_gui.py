"""Client GUI for the PDXC piezo stage controller server.

A compact panel for the beamsplitter stage: stacked buttons that drive it to
either end stop -- the one outlined in green is where the stage currently is
-- plus a settings popover for the throw parameters.

The stage is open-loop, so "where it is" is whatever the server last
commanded it to; the server persists that across restarts and every client
sees the same value.  Jogging the stage invalidates it back to "unknown"
until the next full throw.

Network I/O is dispatched to ``QThreadPool`` so the GUI thread never stalls.
Controls grey out while a move is in progress and the status line updates
accordingly.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt6.QtCore import Qt, QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from waxx.control.misc.pdxc import (
    MAX_PULSES,
    MIN_PULSES,
    POSITION_IN,
    POSITION_OUT,
    POSITION_UNKNOWN,
    PDXC_Client,
)

FONTSIZE_PT = 10
STARTUP_GRACE_S = 10.0
RECONNECT_INTERVAL_S = 5.0
POSITION_POLL_S = 5.0        # pick up moves made by experiments / other clients
MAX_THROW_MOVES = 10         # mirrors the server-side cap

# in = beamsplitter inserted (APD), out = retracted (Andor camera)
_LABELS = {POSITION_IN: "in / APD",
           POSITION_OUT: "out / Andor",
           POSITION_UNKNOWN: "unknown"}
_ACTIVE_COLOR = "#2e7d32"    # green outline marks where the stage is


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
# Settings popover
# ---------------------------------------------------------------------------

class _SettingsDialog(QDialog):
    """Throw parameters: pulses per move, and moves per direction.

    Edits apply immediately -- the server persists them -- so there is no OK
    button to forget to press.
    """

    def __init__(self, parent: QWidget, settings: dict, apply_fn) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stage settings")
        self._apply = apply_fn
        self._loading = True

        form = QFormLayout(self)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(6)

        self._pulses = QSpinBox()
        self._pulses.setRange(MIN_PULSES, MAX_PULSES)
        self._pulses.setSingleStep(1000)
        self._pulses.setSuffix(" pulses")
        self._pulses.setKeyboardTracking(False)
        self._pulses.setValue(int(settings.get("throw_pulses", MAX_PULSES)))
        self._pulses.setToolTip("Pulses per move, used for both the in and out "
                                "throws.")
        self._pulses.valueChanged.connect(lambda v: self._push("throw_pulses", v))
        form.addRow("step size", self._pulses)

        self._moves_in = self._moves_box(settings.get("moves_in", 1))
        self._moves_in.setToolTip("Moves of that size issued to reach in / APD.")
        self._moves_in.valueChanged.connect(lambda v: self._push("moves_in", v))
        form.addRow("moves in", self._moves_in)

        self._moves_out = self._moves_box(settings.get("moves_out", 1))
        self._moves_out.setToolTip("Moves of that size issued to reach out / Andor.")
        self._moves_out.valueChanged.connect(lambda v: self._push("moves_out", v))
        form.addRow("moves out", self._moves_out)

        note = QLabel("Each throw drives this many moves of this size into the "
                      "end stop.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        form.addRow(close)

        self._loading = False

    @staticmethod
    def _moves_box(value) -> QSpinBox:
        box = QSpinBox()
        box.setRange(1, MAX_THROW_MOVES)
        box.setKeyboardTracking(False)
        box.setValue(int(value))
        return box

    def _push(self, key: str, value: int) -> None:
        if not self._loading:
            self._apply(key, value)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class PDXCClientWidget(QWidget):
    """Compact beamsplitter panel: position readout + In / Out + settings."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._client: Optional[PDXC_Client] = None
        self._start_time = time.monotonic()
        self._last_connect_attempt = float("-inf")
        self._connect_in_flight = False
        self._move_in_flight = False
        self._position = POSITION_UNKNOWN
        self._settings: dict = {}
        self._build_ui()

        # Poll for the server until it appears.  _try_connect() self-throttles
        # to RECONNECT_INTERVAL_S, so a short tick just keeps retrying when the
        # server starts (or restarts) after the GUI does.
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setInterval(1000)
        self._reconnect_timer.timeout.connect(self._try_connect)
        self._reconnect_timer.start()

        # Someone else (an experiment, another GUI) may move the stage.
        self._position_timer = QTimer(self)
        self._position_timer.setInterval(int(POSITION_POLL_S * 1000))
        self._position_timer.timeout.connect(self._refresh_position)

        self._try_connect()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 5, 6, 5)
        root.setSpacing(4)

        title_font = QFont()
        title_font.setPointSize(FONTSIZE_PT)
        title_font.setBold(True)

        btn_font = QFont()
        btn_font.setPointSize(FONTSIZE_PT)

        small_font = QFont()
        small_font.setPointSize(FONTSIZE_PT - 2)

        # header: title + settings gear
        header = QHBoxLayout()
        header.setSpacing(4)
        title = QLabel("beamsplitter")
        title.setFont(title_font)
        header.addWidget(title)
        header.addStretch(1)

        self._btn_settings = QToolButton()
        self._btn_settings.setText("⚙")
        self._btn_settings.setToolTip("Stage settings")
        self._btn_settings.setAutoRaise(True)
        self._btn_settings.setFixedSize(20, 20)
        self._btn_settings.clicked.connect(self._open_settings)
        header.addWidget(self._btn_settings)
        root.addLayout(header)

        # stacked destination buttons; the outlined one is where the stage is
        self._btn_in = QPushButton(_LABELS[POSITION_IN])
        self._btn_in.setFont(btn_font)
        self._btn_in.setFixedHeight(26)
        self._btn_in.clicked.connect(lambda: self._start_move(POSITION_IN))
        root.addWidget(self._btn_in)

        self._btn_out = QPushButton(_LABELS[POSITION_OUT])
        self._btn_out.setFont(btn_font)
        self._btn_out.setFixedHeight(26)
        self._btn_out.clicked.connect(lambda: self._start_move(POSITION_OUT))
        root.addWidget(self._btn_out)

        self._status_lbl = QLabel("")
        self._status_lbl.setFont(small_font)
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setFixedHeight(14)
        root.addWidget(self._status_lbl)

        self._render_position()
        self._set_controls_enabled(False)
        self._set_status("connecting...", "gray")

    def _render_position(self) -> None:
        """Outline the button matching where the stage currently is.

        That outline is the whole position indicator; an unknown position
        simply leaves both buttons plain.
        """
        for state, button in ((POSITION_IN, self._btn_in),
                              (POSITION_OUT, self._btn_out)):
            if state == self._position:
                button.setStyleSheet(
                    f"QPushButton {{ border: 2px solid {_ACTIVE_COLOR}; "
                    f"font-weight: bold; }}")
            else:
                button.setStyleSheet("")

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._btn_in.setEnabled(enabled)
        self._btn_out.setEnabled(enabled)
        self._btn_settings.setEnabled(enabled)

    def _set_status(self, text: str, color: str) -> None:
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"color: {color};")

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
        self._set_controls_enabled(True)
        self._set_status(f"{client.host}:{client.port}", "green")
        self._refresh_position()
        self._refresh_settings()
        self._position_timer.start()

    def _on_connect_failed(self, err: str) -> None:
        self._connect_in_flight = False
        elapsed = time.monotonic() - self._start_time
        if elapsed > STARTUP_GRACE_S:
            self._set_status("server not found - retrying...", "orange")

    def _drop_client(self) -> None:
        """Server went away: stop polling and resume discovery."""
        self._client = None
        self._position_timer.stop()
        self._set_controls_enabled(False)
        self._reconnect_timer.start()

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def _refresh_position(self) -> None:
        if self._client is None or self._move_in_flight:
            return
        client = self._client
        worker = _BgCall(client.get_position_state)
        worker.signals.done.connect(self._on_position)
        QThreadPool.globalInstance().start(worker)

    def _on_position(self, state) -> None:
        state = str(state)
        if state not in _LABELS:
            state = POSITION_UNKNOWN
        if state != self._position:
            self._position = state
            self._render_position()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _refresh_settings(self) -> None:
        """Cache the throw parameters so the settings popover opens instantly."""
        if self._client is None:
            return
        client = self._client

        def fetch():
            return {"throw_pulses": client.get_throw_pulses(),
                    "moves_in": client.get_throw_moves(POSITION_IN),
                    "moves_out": client.get_throw_moves(POSITION_OUT)}

        worker = _BgCall(fetch)
        worker.signals.done.connect(self._on_settings)
        QThreadPool.globalInstance().start(worker)

    def _on_settings(self, settings) -> None:
        self._settings = dict(settings)

    def _open_settings(self) -> None:
        if self._client is None:
            return
        _SettingsDialog(self, self._settings, self._apply_setting).exec()

    def _apply_setting(self, key: str, value: int) -> None:
        """Push one settings change to the server."""
        client = self._client
        if client is None:
            return
        if key == "throw_pulses":
            call = lambda: client.set_throw_pulses(value)
        elif key == "moves_in":
            call = lambda: client.set_throw_moves(POSITION_IN, value)
        elif key == "moves_out":
            call = lambda: client.set_throw_moves(POSITION_OUT, value)
        else:
            return
        self._settings[key] = value

        worker = _BgCall(call)
        worker.signals.failed.connect(
            lambda err: self._set_status(f"not saved: {err}", "red"))
        QThreadPool.globalInstance().start(worker)

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _start_move(self, state: str) -> None:
        if self._client is None:
            self._set_status("not connected", "red")
            return
        self._move_in_flight = True
        self._set_controls_enabled(False)
        self._set_status(f"moving {_LABELS[state]}...", "#c8a000")

        client = self._client
        # force: a click means "go there", even if the server already thinks
        # the stage is there -- this is how you re-seat a stage by hand.
        worker = _BgCall(lambda: client.move_to(state, force=True))
        worker.signals.done.connect(lambda _r, s=state: self._on_move_done(s))
        worker.signals.failed.connect(self._on_move_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_move_done(self, state: str) -> None:
        self._move_in_flight = False
        self._set_controls_enabled(True)
        self._set_status("done", "green")
        self._position = state
        self._render_position()

    def _on_move_failed(self, err: str) -> None:
        self._move_in_flight = False
        self._set_controls_enabled(True)
        self._set_status(f"error: {err}", "red")
        self._position = POSITION_UNKNOWN
        self._render_position()
        if "unreachable" in err:
            self._drop_client()


# ---------------------------------------------------------------------------
# Standalone launcher
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("PDXC Beamsplitter Control")
    win.setCentralWidget(PDXCClientWidget())
    win.resize(200, 130)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
