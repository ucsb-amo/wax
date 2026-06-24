"""Capped log-tail view used by every server panel.

A thin wrapper around :class:`QPlainTextEdit` that:

* limits buffered lines via ``setMaximumBlockCount`` (older lines drop)
* exposes "copy all" / "clear" / "save to file" actions
* color-tags stderr lines red, INFO lines green-ish, etc.

Designed to be cheap so the supervisor can always drain the subprocess
stdout even when the panel is hidden.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QTextCharFormat, QColor, QTextCursor
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")
_LEVEL_COLORS = {
    "DEBUG": QColor("#888"),
    "INFO": QColor("#205a96"),
    "WARNING": QColor("#a06000"),
    "ERROR": QColor("#b22222"),
    "CRITICAL": QColor("#b22222"),
}


class LogTailView(QWidget):
    """Compact log-tail viewer with copy/clear/save toolbar."""

    def __init__(self, max_lines: int = 5000, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._editor = QPlainTextEdit(self)
        self._editor.setReadOnly(True)
        self._editor.setMaximumBlockCount(int(max_lines))
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.setStyleSheet(
            "QPlainTextEdit { font-family: Consolas, 'Cascadia Mono', monospace;"
            " font-size: 10px; background: #fafafa; color: #222; }"
        )

        self._copy_btn = QPushButton("Copy")
        self._clear_btn = QPushButton("Clear")
        self._save_btn = QPushButton("Save\u2026")
        self._copy_btn.clicked.connect(self._copy_all)
        self._clear_btn.clicked.connect(self._editor.clear)
        self._save_btn.clicked.connect(self._save_to_file)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.addWidget(self._copy_btn)
        toolbar.addWidget(self._clear_btn)
        toolbar.addWidget(self._save_btn)
        toolbar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addLayout(toolbar)
        layout.addWidget(self._editor, 1)

    def append(self, line: str) -> None:
        """Append a single line to the log tail.

        Line color is auto-derived from the level token if present.
        """
        color = QColor("#222")
        m = _LEVEL_RE.search(line)
        if m:
            color = _LEVEL_COLORS.get(m.group(1), color)
        elif "[ERR]" in line:
            color = _LEVEL_COLORS["ERROR"]

        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor.setCharFormat(fmt)
        cursor.insertText(line + "\n")
        # Auto-scroll only if user was already at the bottom.
        sb = self._editor.verticalScrollBar()
        if sb.value() >= sb.maximum() - 4:
            sb.setValue(sb.maximum())

    def _copy_all(self) -> None:
        self._editor.selectAll()
        self._editor.copy()
        cur = self._editor.textCursor()
        cur.clearSelection()
        self._editor.setTextCursor(cur)

    def _save_to_file(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        default = f"dashboard_log_{ts}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Save log tail", default, "Text (*.txt)")
        if not path:
            return
        try:
            Path(path).write_text(self._editor.toPlainText(), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save:\n{exc!r}")


__all__ = ["LogTailView"]
