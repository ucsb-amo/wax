"""Running-servers overview: one compact status row per supervisor.

Each row is ``LED  label  state  [start/stop] [restart]``.  The first button
shows a play triangle while the server is down and a square while it runs;
the circular arrow restarts it.  Rows react to ``state_changed`` in real time
and use the same colours as the headers (:mod:`waxx.util.dashboard.theme`).

An EXTERNAL server is running but was not started by this dashboard (its
discovery beacon is on the subnet), so it cannot be stopped from here; its
stop button stays disabled and the supervisor drops the row back to idle once
that instance stops beaconing.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Iterable, Optional

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from waxx.util.dashboard import theme


_LOG = logging.getLogger("waxx.dashboard.running_servers")

#: ``confirm(panel, server_id, action) -> bool`` with action "stop" or
#: "restart"; asked only when a live process would be killed.
ConfirmFn = Callable[[QWidget, str, str], bool]

_START_STATES = ("IDLE", "CRASHED", "FAILED")
_STOP_STATES = ("RUNNING", "STARTING")
_RESTART_STATES = ("RUNNING", "CRASHED", "IDLE")

_TIPS = {
    "IDLE": "Start",
    "CRASHED": "Start (exited with an error)",
    "FAILED": "Start (clears the crash-restart limit)",
    "RUNNING": "Stop",
    "STARTING": "Stop",
    "STOPPING": "Stopping…",
    "EXTERNAL": "Running outside this dashboard (another process or PC beacons it) "
                "— stop it where it was started",
}

_STATE_TIPS = {
    "EXTERNAL": "Its discovery beacon is on the subnet, but this dashboard did not start it. "
                "Re-checked every 2 s; the row goes idle when it stops beaconing.",
}


def _state_name(state) -> str:
    try:
        return str(getattr(state, "name", state)).upper()
    except Exception:
        return "UNKNOWN"


def _paint_glyph(kind: str, color: str, size: int = 12) -> QPixmap:
    """A play triangle, stop square or restart arrow, drawn so it never
    falls back to a colour-emoji font."""
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color)
    s = float(size)
    if kind == "play":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawPolygon(QPolygonF([QPointF(s * 0.22, s * 0.12), QPointF(s * 0.88, s * 0.5),
                                 QPointF(s * 0.22, s * 0.88)]))
    elif kind == "stop":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawRoundedRect(QRectF(s * 0.18, s * 0.18, s * 0.64, s * 0.64), 1.0, 1.0)
    else:  # restart: clockwise arc with the head at its upper-right end
        pen = QPen(c, s * 0.13)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        cx = cy = s * 0.5
        r = s * 0.32
        head_deg = 60.0
        p.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r), int(head_deg * 16), int(285 * 16))
        th = math.radians(head_deg)
        tip_x, tip_y = cx + r * math.cos(th), cy - r * math.sin(th)
        dx, dy = math.sin(th), math.cos(th)  # clockwise tangent, screen coords
        nx, ny = -dy, dx
        h, w = s * 0.30, s * 0.22
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawPolygon(QPolygonF([
            QPointF(tip_x + dx * h * 0.6, tip_y + dy * h * 0.6),
            QPointF(tip_x - dx * h * 0.4 + nx * w, tip_y - dy * h * 0.4 + ny * w),
            QPointF(tip_x - dx * h * 0.4 - nx * w, tip_y - dy * h * 0.4 - ny * w),
        ]))
    p.end()
    return pm


def _glyph_icon(kind: str) -> QIcon:
    icon = QIcon()
    icon.addPixmap(_paint_glyph(kind, theme.FG), QIcon.Mode.Normal)
    icon.addPixmap(_paint_glyph(kind, theme.FG_STRONG), QIcon.Mode.Active)
    icon.addPixmap(_paint_glyph(kind, "#555555"), QIcon.Mode.Disabled)
    return icon


def _icon_button(parent: QWidget) -> QToolButton:
    btn = QToolButton(parent)
    btn.setAutoRaise(True)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedSize(18, 18)
    btn.setIconSize(QSize(10, 10))
    btn.setStyleSheet(
        "QToolButton { border: 0; border-radius: 3px; padding: 0; }"
        f"QToolButton:hover {{ background: {theme.BG_BUTTON_HOVER}; }}"
    )
    return btn


class _ServerRow:
    """The widgets making up one row; kept in a grid by the panel."""

    def __init__(self, server_id: str, label: str, supervisor, parent: QWidget,
                 confirm: Optional[ConfirmFn] = None, panel: Optional[QWidget] = None):
        self.server_id = server_id
        self._sup = supervisor
        self._panel = panel if panel is not None else parent  # dialog parent for confirm
        self._confirm = confirm
        self._name = "UNKNOWN"
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
        self.toggle = _icon_button(parent)
        self.toggle.clicked.connect(self._on_toggle)
        self.restart = _icon_button(parent)
        self.restart.setIcon(_glyph_icon("restart"))
        self.restart.setToolTip("Restart")
        self.restart.clicked.connect(self._on_restart)
        self._on_state(getattr(supervisor, "state", None))
        if supervisor is not None:
            try:
                supervisor.state_changed.connect(self._on_state)
            except Exception:
                _LOG.exception("could not connect state_changed for %s", server_id)

    def _on_state(self, state) -> None:
        name = _state_name(state)
        self._name = name
        color = theme.state_color(name)
        self.led.setStyleSheet(f"QLabel {{ color: {color}; font-size: 14px; }}")
        self.state.setText(name.lower())
        self.state.setToolTip(_STATE_TIPS.get(name, ""))
        self.state.setStyleSheet(
            f"QLabel {{ color: {color}; font-family: Consolas, monospace; font-size: 10px; }}"
        )
        has_sup = self._sup is not None
        self.toggle.setIcon(_glyph_icon("play" if name in _START_STATES else "stop"))
        self.toggle.setToolTip(_TIPS.get(name, name.lower()))
        self.toggle.setEnabled(has_sup and name in _START_STATES + _STOP_STATES)
        self.restart.setEnabled(has_sup and name in _RESTART_STATES)

    def _confirmed(self, action: str) -> bool:
        if self._confirm is None or self._name not in _STOP_STATES:
            return True
        try:
            return bool(self._confirm(self._panel, self.server_id, action))
        except Exception:
            _LOG.exception("confirm(%s, %s) raised; not proceeding", self.server_id, action)
            return False

    def _on_toggle(self) -> None:
        sup, name = self._sup, self._name
        if sup is None:
            return
        if name in ("CRASHED", "FAILED"):
            sup.reset_and_start()
        elif name == "IDLE":
            sup.start()
        elif name in _STOP_STATES and self._confirmed("stop"):
            sup.stop()

    def _on_restart(self) -> None:
        if self._sup is not None and self._confirmed("restart"):
            self._sup.restart()


class RunningServersPanel(QWidget):
    """Compact list of every registered supervisor, its live state and
    start / stop / restart buttons.

    *confirm* (optional) is asked before a stop or restart that would kill a
    live process: ``confirm(panel, server_id, action) -> bool``.
    """

    def __init__(
        self,
        entries: Iterable[tuple[str, str, object]],
        *,
        columns: int = 1,  # kept for API compatibility; rows are single-column
        confirm: Optional[ConfirmFn] = None,
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
        grid.setContentsMargins(8, 6, 6, 6)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._rows: dict[str, _ServerRow] = {}
        for r, (sid, label, sup) in enumerate(entries):
            row = _ServerRow(sid, label, sup, inner, confirm, panel=self)
            grid.addWidget(row.led, r, 0)
            grid.addWidget(row.title, r, 1)
            grid.addWidget(row.state, r, 2)
            grid.addWidget(row.toggle, r, 3)
            grid.addWidget(row.restart, r, 4)
            self._rows[sid] = row
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(len(self._rows), 1)
        if not self._rows:
            empty = QLabel("(no supervised servers)", inner)
            empty.setStyleSheet(f"QLabel {{ color: {theme.FG_MUTED}; }}")
            grid.addWidget(empty, 0, 0, 1, 5)
        scroll.setWidget(inner)


__all__ = ["RunningServersPanel"]
