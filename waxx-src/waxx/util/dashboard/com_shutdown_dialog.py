"""Modal dialog shown while COM-managing servers shut down.

When the dashboard window is closed, supervisors whose ``ServerSpec`` has a
non-empty ``com_label`` are prioritized: we send each a CTRL_BREAK (so the
child's ``SIGINT`` handler runs and the COM port releases cleanly) and then
block the close event behind this dialog.

The dialog:

* Lists each COM server in a small grid: ``label | COM port | status | elapsed``.
* Polls each supervisor's ``is_running()`` every 200 ms.
* Tails the last few lines from each supervisor's ``log_line`` signal so
  the user can see exactly what each child is doing.
* Offers ``Force kill remaining`` and ``Cancel close`` buttons.
* Resolves with one of :data:`RESULT_ALL_CLOSED`, :data:`RESULT_USER_FORCED`,
  :data:`RESULT_USER_CANCELLED`.

The dialog blocks indefinitely by default — the user explicitly drives any
escalation.  This mirrors the lab requirement: never give up on a clean COM
release without user consent.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Iterable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


_LOG = logging.getLogger("waxx.dashboard.com_shutdown_dialog")

RESULT_ALL_CLOSED = "all_closed"
RESULT_USER_FORCED = "user_forced"
RESULT_USER_CANCELLED = "user_cancelled"


_STATUS_TERMINATING = "terminating…"
_STATUS_CLOSED = "closed"
_STATUS_FORCED = "force-killed"


class ComShutdownDialog(QDialog):
    """Modal dialog that blocks dashboard close until COM servers exit."""

    POLL_INTERVAL_MS = 200
    LOG_TAIL_LINES = 200

    def __init__(self, supervisors: Iterable[tuple[str, str, object]], parent=None):
        """
        Parameters
        ----------
        supervisors:
            Iterable of ``(label, com_port, supervisor)`` triples — one per
            COM-managing server still alive at close.  ``supervisor`` must
            expose ``is_running()``, ``force_kill()``, and a ``log_line``
            Qt signal that emits ``str``.
        """
        super().__init__(parent)
        self.setWindowTitle("Closing COM devices…")
        self.setModal(True)
        self.setMinimumWidth(560)
        # Disable the system close button: the user must use one of the
        # explicit buttons so we always know what they intended.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self._result = RESULT_ALL_CLOSED  # default if all exit cleanly
        self._sups: list[tuple[str, str, object]] = list(supervisors)
        self._status_labels: dict[int, QLabel] = {}
        self._elapsed_labels: dict[int, QLabel] = {}
        self._final_status: dict[int, str] = {}
        self._t0 = time.monotonic()
        self._log_tail: deque[str] = deque(maxlen=self.LOG_TAIL_LINES)

        self._build_ui()
        self._wire_signals()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self.POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()
        # Run one poll immediately in case the supervisors already exited
        # before the dialog managed to show.
        QTimer.singleShot(0, self._poll)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def result_reason(self) -> str:
        """One of ``RESULT_*`` constants once :meth:`exec` returns."""
        return self._result

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        header = QLabel(
            "Closing COM-managing servers — do not unplug devices.\n"
            "Window will close automatically when each server has released "
            "its serial port."
        )
        header.setWordWrap(True)
        root.addWidget(header)

        grid_host = QVBoxLayout()
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        # Header row
        for col, text in enumerate(("Server", "COM", "Status", "Elapsed")):
            lbl = QLabel(f"<b>{text}</b>")
            grid.addWidget(lbl, 0, col)
        for row, (label, com_port, _sup) in enumerate(self._sups, start=1):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(QLabel(com_port or "—"), row, 1)
            status_lbl = QLabel(_STATUS_TERMINATING)
            elapsed_lbl = QLabel("0.0 s")
            grid.addWidget(status_lbl, row, 2)
            grid.addWidget(elapsed_lbl, row, 3)
            self._status_labels[row - 1] = status_lbl
            self._elapsed_labels[row - 1] = elapsed_lbl
        grid_host.addLayout(grid)
        root.addLayout(grid_host)

        # Live log tail.
        self._log_view = QPlainTextEdit(self)
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(self.LOG_TAIL_LINES)
        self._log_view.setPlaceholderText("Waiting for server output…")
        self._log_view.setMinimumHeight(140)
        root.addWidget(self._log_view, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel close")
        self._cancel_btn.setToolTip(
            "Keep the dashboard open.  Note: any server that already received "
            "the shutdown signal will stay stopped — restart manually if "
            "needed."
        )
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        self._force_btn = QPushButton("Force kill remaining")
        self._force_btn.setToolTip(
            "Hard-kill any COM server that hasn't exited yet.  May leave a "
            "serial port held until Windows releases it."
        )
        self._force_btn.clicked.connect(self._on_force_kill)
        btn_row.addWidget(self._force_btn)
        root.addLayout(btn_row)

    def _wire_signals(self) -> None:
        for _label, _com, sup in self._sups:
            sig = getattr(sup, "log_line", None)
            if sig is None:
                continue
            try:
                sig.connect(self._on_log_line)
            except Exception:
                _LOG.exception("could not connect log_line for %r", getattr(sup, "server_id", "?"))

    # ------------------------------------------------------------------
    # Polling / state
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        elapsed_total = time.monotonic() - self._t0
        all_done = True
        for idx, (_label, _com, sup) in enumerate(self._sups):
            running = True
            try:
                running = bool(sup.is_running())
            except Exception:
                running = False
            elapsed_lbl = self._elapsed_labels.get(idx)
            if elapsed_lbl is not None and idx not in self._final_status:
                elapsed_lbl.setText(f"{elapsed_total:.1f} s")
            if running:
                all_done = False
            else:
                if idx not in self._final_status:
                    self._final_status[idx] = _STATUS_CLOSED
                    status_lbl = self._status_labels.get(idx)
                    if status_lbl is not None:
                        status_lbl.setText(_STATUS_CLOSED)
                        status_lbl.setStyleSheet("color: #4caf50; font-weight: 600;")
        if all_done:
            self._poll_timer.stop()
            self._result = RESULT_ALL_CLOSED
            self.accept()

    def _on_log_line(self, line: str) -> None:
        try:
            self._log_view.appendPlainText(line)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_force_kill(self) -> None:
        for idx, (_label, _com, sup) in enumerate(self._sups):
            try:
                if sup.is_running():
                    sup.force_kill(wait_ms=200)
                    self._final_status[idx] = _STATUS_FORCED
                    status_lbl = self._status_labels.get(idx)
                    if status_lbl is not None:
                        status_lbl.setText(_STATUS_FORCED)
                        status_lbl.setStyleSheet("color: #ff7043; font-weight: 600;")
            except Exception:
                _LOG.exception("force_kill failed for %r", getattr(sup, "server_id", "?"))
        self._poll_timer.stop()
        self._result = RESULT_USER_FORCED
        self.accept()

    def _on_cancel(self) -> None:
        self._poll_timer.stop()
        self._result = RESULT_USER_CANCELLED
        self.reject()

    def closeEvent(self, ev: QCloseEvent) -> None:  # noqa: N802 - Qt API
        # Map any stray close (e.g. Alt+F4) to "cancel" so callers never
        # mistake an X-close for a successful shutdown.
        if self._result == RESULT_ALL_CLOSED and any(
            idx not in self._final_status for idx in range(len(self._sups))
        ):
            self._result = RESULT_USER_CANCELLED
        super().closeEvent(ev)


__all__ = [
    "ComShutdownDialog",
    "RESULT_ALL_CLOSED",
    "RESULT_USER_FORCED",
    "RESULT_USER_CANCELLED",
]
