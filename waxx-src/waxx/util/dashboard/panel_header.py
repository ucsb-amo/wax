"""Panel header bar used as the dock's title-bar widget.

Lays out (left-to-right) on a single row:

* per-panel emoji icon
* Panel title
* Server-only "Server" pill: its dot shows the local supervisor's state
  (running / starting / crashed / idle), turned amber / red while a running
  server is still connecting / not answering; its drop-down holds
  Start / Stop / Restart (which act on the local supervisor)
* Conn badge (snapshot poller status; client panels, or a server panel whose
  pill is hidden)
* Optional :class:`ComStatusButton` driven by the snapshot's ``"com"`` key
* Pop-out button (moves the panel into an independent top-level window)

Qt's built-in QDockWidget close button is *not* drawn when a custom
title bar widget is installed, so the header carries its own close button
as well.  There is no in-dashboard float toggle: panels are either docked
or popped out into a real top-level window.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.dashboard.restyle import set_style, set_tooltip
from waxx.util.dashboard.server_supervisor import SupervisorState
from waxx.util.dashboard.widgets import ComStatusButton


def _dot_icon(css_color: str, d: int = 11) -> QIcon:
    pm = QPixmap(d, d)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor("#1a1a1a"), 1))
    p.setBrush(QBrush(QColor(css_color)))
    p.drawEllipse(QPointF(d / 2, d / 2), d / 2 - 1, d / 2 - 1)
    p.end()
    return QIcon(pm)


class _ServerPill(QToolButton):
    """"Server" pill: a connection dot plus a Start / Stop / Restart drop-down."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setText("Server")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setIconSize(QSize(11, 11))
        self.setStyleSheet(
            f"QToolButton {{ color: {theme.FG}; padding: 1px 6px; font-size: 11px;"
            f" border: 1px solid {theme.BORDER}; border-radius: 8px;"
            f" background-color: {theme.BG_BUTTON}; }}"
            f"QToolButton:hover {{ background-color: {theme.BG_BUTTON_HOVER}; }}"
            "QToolButton::menu-indicator { image: none; width: 0; }"
        )
        self.menu_ = QMenu(self)
        self.setMenu(self.menu_)
        self.set_dot(theme.OFF)

    def set_dot(self, css_color: str) -> None:
        self.setIcon(_dot_icon(css_color))


def _glyph_button(text: str, tooltip: str, parent: QWidget) -> QToolButton:
    btn = QToolButton(parent)
    btn.setText(text)
    btn.setToolTip(tooltip)
    btn.setAutoRaise(True)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedSize(18, 18)
    btn.setStyleSheet(
        f"QToolButton {{ color: {theme.FG_MUTED}; border: 0; border-radius: 3px;"
        " font-size: 12px; padding: 0; }"
        f"QToolButton:hover {{ color: {theme.FG_STRONG}; background: {theme.BG_BUTTON_HOVER}; }}"
    )
    return btn


class PanelHeaderBar(QWidget):
    """Compact, single-row title bar suitable for ``QDockWidget.setTitleBarWidget``.

    Server panels show title + Server pill (dot + Start/Stop/Restart menu) + COM.
    Client panels show just title + conn + COM (no supervisor controls,
    no LED).  Every panel gets pop-out / close glyphs on the right.
    """

    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    restart_clicked = pyqtSignal()
    popout_clicked = pyqtSignal()
    close_clicked = pyqtSignal()

    def __init__(
        self,
        label: str,
        *,
        is_server: bool = True,
        com_label: Optional[str] = None,
        icon: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setObjectName("PanelHeaderBar")
        # QWidget needs WA_StyledBackground for QSS borders to render.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._active = False
        self._apply_frame_style()

        layout = QHBoxLayout(self)
        # Top/bottom margins must be >= the border width so child widgets
        # don't paint over the top border.
        layout.setContentsMargins(7, 2, 4, 2)
        layout.setSpacing(4)
        self._layout = layout

        self._icon_label: Optional[QLabel] = None
        if icon:
            self._icon_label = QLabel(icon, self)
            self._icon_label.setStyleSheet(
                "QLabel { font-family: 'Segoe UI Emoji', 'Apple Color Emoji',"
                " 'Noto Color Emoji', sans-serif; font-size: 13px;"
                " padding: 0 2px 0 0; }"
            )
            layout.addWidget(self._icon_label)

        self._title = QLabel(label, self)
        self._title.setStyleSheet(f"QLabel {{ color: {theme.FG_STRONG}; font-weight: 600; }}")
        layout.addWidget(self._title)

        # Server pill (connection dot + Start/Stop/Restart menu) right after the title.
        self._pill: Optional[_ServerPill] = None
        self._start_btn: Optional[QAction] = None
        self._stop_btn: Optional[QAction] = None
        self._restart_btn: Optional[QAction] = None
        self._supervisor_state = SupervisorState.IDLE
        self._conn_status = "disconnected"
        self._conn_detail = ""
        self._conn_seen = False
        if is_server:
            layout.addSpacing(6)
            self._pill = _ServerPill(self)
            menu = self._pill.menu_
            self._start_btn = menu.addAction("Start")
            self._stop_btn = menu.addAction("Stop")
            self._restart_btn = menu.addAction("Restart")
            self._start_btn.triggered.connect(lambda _c=False: self.start_clicked.emit())
            self._stop_btn.triggered.connect(lambda _c=False: self.stop_clicked.emit())
            self._restart_btn.triggered.connect(lambda _c=False: self.restart_clicked.emit())
            layout.addWidget(self._pill)
            self.set_state(SupervisorState.IDLE)
            self._refresh_pill()

        layout.addStretch(1)

        # Conn badge - small pill, hidden until the first poll result.
        self._conn_badge = QLabel("—", self)
        self._conn_badge.setVisible(False)
        self._conn_badge.setStyleSheet(f"QLabel {{ {theme.pill_css(theme.OFF)} }}")
        layout.addWidget(self._conn_badge)

        # COM button - only present when the spec declares com_label.
        self._com_btn: Optional[ComStatusButton] = None
        if com_label:
            self._com_btn = ComStatusButton(com_label, parent=self)
            self._com_btn.setToolTip(f"{com_label} — waiting for the server's first snapshot")
            layout.addWidget(self._com_btn)

        # Window-ish glyphs: pop out, close.
        layout.addSpacing(2)
        self._popout_btn = _glyph_button("⧉", "Pop out into its own window (own taskbar entry)", self)
        self._popout_btn.clicked.connect(self.popout_clicked)
        layout.addWidget(self._popout_btn)
        self._close_btn = _glyph_button("✕", "Hide panel (re-open from the Panels menu)", self)
        self._close_btn.clicked.connect(self.close_clicked)
        layout.addWidget(self._close_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def title(self) -> str:
        return self._title.text()

    def com_button(self) -> Optional[ComStatusButton]:
        return self._com_btn

    def set_active(self, active: bool) -> None:
        """Accent the frame when this panel holds keyboard focus."""
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        self._apply_frame_style()

    def set_server_controls_visible(self, visible: bool) -> None:
        """Hide the Server pill for in-process servers (no subprocess); the
        plain conn badge then carries the connection state."""
        if self._pill is not None:
            self._pill.setVisible(bool(visible))
        self._sync_conn_badge()

    def set_window_glyphs_visible(self, popout: bool = True, close: bool = True) -> None:
        self._popout_btn.setVisible(popout)
        self._close_btn.setVisible(close)

    def set_state(self, state: SupervisorState) -> None:
        """Supervisor state: colours the pill's dot and enables the menu entries."""
        self._supervisor_state = state
        if self._start_btn is not None:
            self._start_btn.setEnabled(state in (SupervisorState.IDLE, SupervisorState.CRASHED,
                                                 SupervisorState.EXTERNAL))
        if self._stop_btn is not None:
            self._stop_btn.setEnabled(state in (SupervisorState.RUNNING, SupervisorState.STARTING))
        if self._restart_btn is not None:
            self._restart_btn.setEnabled(
                state in (SupervisorState.RUNNING, SupervisorState.CRASHED, SupervisorState.IDLE)
            )
        self._refresh_pill()

    def set_conn(self, status: str, detail: str = "") -> None:
        status = str(status).lower()
        self._conn_status = status
        self._conn_detail = detail
        self._conn_seen = True
        text = {"connected": "OK", "disconnected": "--", "connecting": "…"}.get(status, "ERR")
        self._conn_badge.setText(text)
        set_style(self._conn_badge, f"QLabel {{ {theme.pill_css(theme.conn_color(status))} }}")
        set_tooltip(self._conn_badge, detail or f"snapshot poll: {status}")
        self._sync_conn_badge()
        self._refresh_pill()

    def _sync_conn_badge(self) -> None:
        """The badge shows only where there is no pill to carry the dot."""
        pill_shown = self._pill is not None and not self._pill.isHidden()
        self._conn_badge.setVisible(self._conn_seen and not pill_shown)

    def _refresh_pill(self) -> None:
        if self._pill is None:
            return
        color = theme.state_color(self._supervisor_state)
        if (self._supervisor_state in (SupervisorState.RUNNING, SupervisorState.EXTERNAL)
                and self._conn_seen):
            # Running but the panel can't talk to it: don't show a healthy green.
            if self._conn_status == "connecting":
                color = theme.WARN
            elif self._conn_status not in ("connected", "disconnected"):
                color = theme.ERR
        self._pill.set_dot(color)
        sup = getattr(self._supervisor_state, "name", str(self._supervisor_state)).title()
        self._pill.setToolTip(
            f"Panel client: {self._conn_status}"
            + (f" ({self._conn_detail})" if self._conn_detail else "")
            + f"\nServer process on this PC: {sup}"
        )

    def set_com(self, com: Optional[dict]) -> None:
        """Drive the COM pill from a snapshot's ``"com"`` dict (``SerialSnapshot.as_dict()`` shape)."""
        if self._com_btn is None:
            return
        if not isinstance(com, dict):
            self._com_btn.set_status("error", detail="snapshot has no 'com' key")
            return
        parts = []
        if com.get("baud"):
            parts.append(f"{com['baud']} baud")
        rx = com.get("last_rx_seconds_ago")
        if isinstance(rx, (int, float)):
            parts.append(f"last rx {rx:.1f} s ago")
        if com.get("reconnect_attempts"):
            parts.append(f"{com['reconnect_attempts']} reconnects")
        if isinstance(com.get("next_retry_s"), (int, float)):
            parts.append(f"next try in {com['next_retry_s']:.0f} s")
        if com.get("last_error"):
            parts.append(f"error: {com['last_error']}")
        self._com_btn.set_status(
            str(com.get("status", "error")),
            port=com.get("port") or None,
            detail="\n".join(parts),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_frame_style(self) -> None:
        border = theme.ACCENT if self._active else theme.BORDER
        self.setStyleSheet(
            f"QWidget#PanelHeaderBar {{ background-color: {theme.BG_RAISED};"
            f" border-top: 1px solid {border};"
            f" border-left: 1px solid {border};"
            f" border-right: 1px solid {border};"
            f" border-bottom: 1px solid {theme.BORDER_STRONG};"
            " border-top-left-radius: 3px;"
            " border-top-right-radius: 3px; }"
            f"QWidget#PanelHeaderBar QLabel {{ color: {theme.FG}; }}"
        )


__all__ = ["PanelHeaderBar"]
