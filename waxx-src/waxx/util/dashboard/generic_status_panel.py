"""Generic key/value status panel used when a server has no bespoke UI.

Many lab servers only need to show "is it alive? what's the latest snapshot?"
This widget provides exactly that:

* a key/value table of snapshot fields
* an optional "last log line" preview
* a "View full log" link that opens the dashboard log folder
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


_LOG = logging.getLogger("waxx.dashboard.generic")


class GenericServerStatusPanel(QWidget):
    """Generic snapshot table + last log line."""

    def __init__(self, server_id: str, log_dir: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._server_id = server_id
        self._log_dir = log_dir

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Key", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table, 1)

        self._last_log = QLabel("(no log lines yet)", self)
        self._last_log.setStyleSheet("QLabel { color: #555; font-family: Consolas, monospace; font-size: 10px; }")
        self._last_log.setWordWrap(True)
        layout.addWidget(self._last_log)

        bottom = QHBoxLayout()
        self._open_logs = QPushButton("View full log", self)
        self._open_logs.clicked.connect(self._open_log_dir)
        self._open_logs.setEnabled(bool(self._log_dir))
        bottom.addStretch(1)
        bottom.addWidget(self._open_logs)
        layout.addLayout(bottom)

    def set_snapshot(self, snapshot: Optional[dict]) -> None:
        if not snapshot:
            self._table.setRowCount(1)
            self._table.setItem(0, 0, QTableWidgetItem("(snapshot)"))
            self._table.setItem(0, 1, QTableWidgetItem("not available"))
            return
        rows = sorted(snapshot.items())
        self._table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(str(k)))
            self._table.setItem(i, 1, QTableWidgetItem(str(v)))

    def set_last_log_line(self, line: str) -> None:
        # Truncate so a runaway line doesn't blow up the layout.
        if len(line) > 400:
            line = line[:400] + "..."
        self._last_log.setText(line)

    def _open_log_dir(self) -> None:
        if not self._log_dir:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._log_dir))


__all__ = ["GenericServerStatusPanel"]
