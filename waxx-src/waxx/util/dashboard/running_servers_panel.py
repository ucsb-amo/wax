"""Running-servers overview panel.

A single QWidget that displays a compact tile per registered
``ServerSupervisor`` (state LED + label + Start/Stop/Restart actions).
Suitable for embedding as a regular dashboard panel.

Each tile reacts to the supervisor's ``state_changed`` signal in real time.
For servers whose body panel already lives elsewhere in the dashboard, the
tile is a lightweight status summary -- the heavy GUI stays where the user
docked it.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


_LOG = logging.getLogger("waxx.dashboard.running_servers")


_STATE_COLORS = {
    "stopped": "#666",
    "starting": "#cc8800",
    "running": "#2e8b57",
    "crashed": "#b22222",
    "stopping": "#cc8800",
}


def _state_name(state) -> str:
    # SupervisorState may be an enum; fall back to str() otherwise.
    try:
        return str(getattr(state, "name", state)).lower()
    except Exception:
        return "unknown"


class _ServerTile(QFrame):
    """Compact tile: LED + label + state + Start/Stop/Restart."""

    def __init__(self, server_id: str, label: str, supervisor, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._sup = supervisor
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet(
            "QFrame { background-color: #2b2b2b; border: 1px solid #444; border-radius: 4px; }"
            "QLabel { color: #ddd; }"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        row1 = QHBoxLayout()
        self._led = QLabel("●", self)
        self._led.setStyleSheet("QLabel { color: #666; font-size: 16px; }")
        row1.addWidget(self._led)
        self._title = QLabel(f"<b>{label}</b>", self)
        row1.addWidget(self._title)
        row1.addStretch(1)
        self._state_lbl = QLabel("stopped", self)
        self._state_lbl.setStyleSheet("QLabel { color: #888; font-family: Consolas, monospace; font-size: 10px; }")
        row1.addWidget(self._state_lbl)
        outer.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self._start_btn = QPushButton("Start", self)
        self._stop_btn = QPushButton("Stop", self)
        self._restart_btn = QPushButton("Restart", self)
        for b in (self._start_btn, self._stop_btn, self._restart_btn):
            b.setStyleSheet("QPushButton { padding: 2px 8px; }")
            row2.addWidget(b)
        row2.addStretch(1)
        outer.addLayout(row2)

        if supervisor is not None:
            self._start_btn.clicked.connect(supervisor.start)
            self._stop_btn.clicked.connect(supervisor.stop)
            self._restart_btn.clicked.connect(supervisor.restart)
            try:
                supervisor.state_changed.connect(self._on_state)
            except Exception:
                _LOG.exception("could not connect state_changed for %s", server_id)
        else:
            for b in (self._start_btn, self._stop_btn, self._restart_btn):
                b.setEnabled(False)

    def _on_state(self, state) -> None:
        name = _state_name(state)
        color = _STATE_COLORS.get(name, "#888")
        self._led.setStyleSheet(f"QLabel {{ color: {color}; font-size: 16px; }}")
        self._state_lbl.setText(name)
        self._state_lbl.setStyleSheet(
            f"QLabel {{ color: {color}; font-family: Consolas, monospace; font-size: 10px; }}"
        )


class RunningServersPanel(QWidget):
    """Grid of :class:`_ServerTile`, one per registered supervisor.

    Layout is responsive: the number of columns shrinks to match the
    available width so tiles never extend past the panel.
    """

    TILE_MIN_W = 220   # below this width, drop to one column
    TILE_MAX_W = 360   # tiles never get wider than this

    def __init__(
        self,
        entries: Iterable[tuple[str, str, object]],
        *,
        columns: int = 2,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Never grow horizontally beyond the dock; vertical scrollbar handles
        # overflow.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        self._scroll = scroll
        self._max_cols = max(1, int(columns))
        self._current_cols = self._max_cols

        inner = QWidget(scroll)
        self._inner = inner
        grid = QGridLayout(inner)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._grid = grid

        self._tile_list: list[tuple[str, _ServerTile]] = []
        self._tiles: dict[str, _ServerTile] = {}
        for sid, label, sup in entries:
            tile = _ServerTile(sid, label, sup, inner)
            tile.setMaximumWidth(self.TILE_MAX_W)
            tile.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            self._tile_list.append((sid, tile))
            self._tiles[sid] = tile

        scroll.setWidget(inner)
        self._reflow_tiles(self._max_cols)

    def _reflow_tiles(self, cols: int) -> None:
        cols = max(1, int(cols))
        if cols == self._current_cols and self._grid.count():
            return
        # Detach all tiles, then re-add in new column count.
        for i in reversed(range(self._grid.count())):
            item = self._grid.takeAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.setParent(self._inner)
        for idx, (_sid, tile) in enumerate(self._tile_list):
            row, col = divmod(idx, cols)
            self._grid.addWidget(tile, row, col)
        # Reset and re-apply stretches so trailing space doesn't smear tiles.
        for c in range(max(cols, self._current_cols) + 1):
            self._grid.setColumnStretch(c, 0)
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)
        self._grid.setRowStretch(self._grid.rowCount(), 1)
        self._current_cols = cols

    def resizeEvent(self, ev) -> None:  # noqa: N802 (Qt API)
        try:
            avail = self._scroll.viewport().width()
        except Exception:
            avail = self.width()
        cols = max(1, min(self._max_cols, avail // self.TILE_MIN_W))
        self._reflow_tiles(cols)
        super().resizeEvent(ev)


__all__ = ["RunningServersPanel"]
