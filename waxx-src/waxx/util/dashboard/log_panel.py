"""Log dock: every supervised server's stdout/stderr plus the dashboard's own
log records, in one filterable tail.

Filters: source (All / one server / dashboard), minimum level
(Everything / Warnings and errors), and a substring search.  Lines are kept
in a ring buffer so changing a filter re-renders the recent history instead
of starting from an empty view.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from typing import Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.dashboard.log_tail import LogTailView


_LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")
_LEVEL_RANK = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

DASHBOARD_SOURCE = "dashboard"


def _line_level(line: str) -> int:
    m = _LEVEL_RE.search(line)
    if m:
        return _LEVEL_RANK[m.group(1)]
    if "[ERR]" in line:
        return 30  # stderr output: treat as warning-grade
    return 20


class _QtLogHandler(QObject, logging.Handler):
    """Root-logger handler that forwards records to the panel via a signal
    (thread-safe: records may come from worker threads).

    ``QObject`` must be the first base or PyQt does not bind the signal.
    """

    record_ready = pyqtSignal(str, str)  # (source, line)

    def __init__(self, level: int = logging.INFO):
        QObject.__init__(self)
        logging.Handler.__init__(self, level)
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.record_ready.emit(DASHBOARD_SOURCE, self.format(record))
        except Exception:
            pass


class LogPanel(QWidget):
    """Filterable tail of server output + dashboard log records."""

    MAX_LINES = 5000

    def __init__(self, sources: list[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._buffer: deque[tuple[str, int, str]] = deque(maxlen=self.MAX_LINES)
        self._sources = list(sources)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self._source = QComboBox(self)
        self._source.addItem("All sources", None)
        self._source.addItem("Dashboard", DASHBOARD_SOURCE)
        for s in self._sources:
            self._source.addItem(s, s)
        self._source.currentIndexChanged.connect(self._rerender)
        bar.addWidget(QLabel("Source", self))
        bar.addWidget(self._source)

        self._level = QComboBox(self)
        self._level.addItem("Everything", 0)
        self._level.addItem("Warnings and errors", 30)
        self._level.addItem("Errors only", 40)
        self._level.currentIndexChanged.connect(self._rerender)
        bar.addWidget(QLabel("Level", self))
        bar.addWidget(self._level)

        self._search = QLineEdit(self)
        self._search.setPlaceholderText("filter text…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._rerender)
        bar.addWidget(self._search, 1)
        layout.addLayout(bar)

        self._view = LogTailView(max_lines=self.MAX_LINES, parent=self)
        layout.addWidget(self._view, 1)

        self._count = QLabel("", self)
        self._count.setStyleSheet(f"QLabel {{ color: {theme.FG_MUTED}; font-size: 10px; }}")
        self._count.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._count)

        self._handler: Optional[_QtLogHandler] = None

    # ------------------------------------------------------------------
    # Feeding
    # ------------------------------------------------------------------

    def append(self, source: str, line: str) -> None:
        """Add one line from *source* (a server id or ``DASHBOARD_SOURCE``)."""
        if source not in self._sources and source != DASHBOARD_SOURCE:
            self._sources.append(source)
            self._source.addItem(source, source)
        level = _line_level(line)
        if source != DASHBOARD_SOURCE:
            stamp = time.strftime("%H:%M:%S")
            line = f"{stamp} [{source}] {line}"
        self._buffer.append((source, level, line))
        if self._passes(source, level, line):
            self._view.append(line)
        self._update_count()

    def make_appender(self, source: str):
        """Return a one-argument slot suitable for ``supervisor.log_line``."""
        return lambda line, _s=source: self.append(_s, line)

    def attach_root_logging(self, level: int = logging.INFO) -> None:
        """Mirror the process's own log records (>= level) into the panel."""
        if self._handler is not None:
            return
        self._handler = _QtLogHandler(level)
        self._handler.record_ready.connect(self.append)
        logging.getLogger().addHandler(self._handler)

    def show_errors_only(self) -> None:
        """Preset used by the toolbar's Errors action."""
        self._source.setCurrentIndex(0)
        self._level.setCurrentIndex(1)

    def cleanup(self) -> None:
        if self._handler is not None:
            try:
                logging.getLogger().removeHandler(self._handler)
            except Exception:
                pass
            self._handler = None

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _passes(self, source: str, level: int, line: str) -> bool:
        want_src = self._source.currentData()
        if want_src is not None and source != want_src:
            return False
        if level < int(self._level.currentData() or 0):
            return False
        needle = self._search.text().strip().lower()
        return not needle or needle in line.lower()

    def _rerender(self, *_args) -> None:
        self._view._editor.clear()
        for source, level, line in self._buffer:
            if self._passes(source, level, line):
                self._view.append(line)
        self._update_count()

    def _update_count(self) -> None:
        self._count.setText(f"{len(self._buffer)} lines buffered")


__all__ = ["LogPanel", "DASHBOARD_SOURCE"]
