"""Running-servers overview: one compact status row per supervisor.

Each row is ``LED  label  state``; the Start / Stop / Restart controls live
in every panel header and in the Servers menu, so they are not repeated
here.  Rows react to ``state_changed`` in real time and use the same
colours as the headers (:mod:`waxx.util.dashboard.theme`).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from waxx.util.dashboard import theme


_LOG = logging.getLogger("waxx.dashboard.running_servers")


def _state_name(state) -> str:
    try:
        return str(getattr(state, "name", state)).upper()
    except Exception:
        return "UNKNOWN"


class _ServerRow:
    """The three widgets making up one row; kept in a grid by the panel."""

    def __init__(self, server_id: str, label: str, supervisor, parent: QWidget):
        self.server_id = server_id
        self.led = QLabel("●", parent)
        self.led.setFixedWidth(14)
        self.title = QLabel(label, parent)
        self.title.setStyleSheet(f"QLabel {{ color: {theme.FG_STRONG}; }}")
        self.title.setToolTip(server_id)
        self.state = QLabel("idle", parent)
        self.state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.state.setStyleSheet(
            f"QLabel {{ color: {theme.FG_MUTED}; font-family: Consolas, monospace; font-size: 10px; }}"
        )
        self._on_state(getattr(supervisor, "state", None))
        if supervisor is not None:
            try:
                supervisor.state_changed.connect(self._on_state)
            except Exception:
                _LOG.exception("could not connect state_changed for %s", server_id)

    def _on_state(self, state) -> None:
        name = _state_name(state)
        color = theme.state_color(name)
        self.led.setStyleSheet(f"QLabel {{ color: {color}; font-size: 14px; }}")
        self.state.setText(name.lower())
        self.state.setStyleSheet(
            f"QLabel {{ color: {color}; font-family: Consolas, monospace; font-size: 10px; }}"
        )


class RunningServersPanel(QWidget):
    """Compact list of every registered supervisor and its live state."""

    def __init__(
        self,
        entries: Iterable[tuple[str, str, object]],
        *,
        columns: int = 1,  # kept for API compatibility; rows are single-column
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        inner = QWidget(scroll)
        grid = QGridLayout(inner)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._rows: dict[str, _ServerRow] = {}
        for r, (sid, label, sup) in enumerate(entries):
            row = _ServerRow(sid, label, sup, inner)
            grid.addWidget(row.led, r, 0)
            grid.addWidget(row.title, r, 1)
            grid.addWidget(row.state, r, 2)
            self._rows[sid] = row
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(len(self._rows), 1)
        if not self._rows:
            empty = QLabel("(no supervised servers)", inner)
            empty.setStyleSheet(f"QLabel {{ color: {theme.FG_MUTED}; }}")
            grid.addWidget(empty, 0, 0, 1, 3)
        scroll.setWidget(inner)


__all__ = ["RunningServersPanel"]
