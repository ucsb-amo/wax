"""Panel header bar used as the dock's title-bar widget.

Lays out (left-to-right) on a single row:

* per-panel emoji icon
* LED indicator for supervisor state (server panels only)
* Panel title
* Server-only Start / Stop / Restart buttons
* Conn badge (snapshot poller status)
* Optional :class:`ComStatusButton` driven by the snapshot's ``"com"`` key
* Pop-out button (moves the panel into an independent top-level window)

Qt's built-in QDockWidget float / close buttons are *not* drawn when a
custom title bar widget is installed, so the header carries its own
float-toggle and close buttons as well.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.dashboard.server_supervisor import SupervisorState
from waxx.util.dashboard.widgets import ComStatusButton


class _LedDot(QLabel):
    """Tiny circular LED rendered via stylesheet."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(11, 11)
        self._color = theme.OFF
        self._refresh()

    def set_color(self, css_color: str) -> None:
        self._color = css_color
        self._refresh()

    def _refresh(self) -> None:
        self.setStyleSheet(
            f"QLabel {{ background-color: {self._color}; border-radius: 5px;"
            " border: 1px solid #1a1a1a; }"
        )


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

    Server panels show LED + title + Start/Stop/Restart + conn + COM.
    Client panels show just title + conn + COM (no supervisor controls,
    no LED).  Every panel gets pop-out / float / close glyphs on the right.
    """

    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    restart_clicked = pyqtSignal()
    popout_clicked = pyqtSignal()
    float_clicked = pyqtSignal()
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

        self._led: Optional[_LedDot] = None
        if is_server:
            self._led = _LedDot(self)
            layout.addWidget(self._led)

        self._title = QLabel(label, self)
        self._title.setStyleSheet(f"QLabel {{ color: {theme.FG_STRONG}; font-weight: 600; }}")
        layout.addWidget(self._title)

        # Server controls (Start/Stop/Restart) come right after the title.
        self._start_btn: Optional[QPushButton] = None
        self._stop_btn: Optional[QPushButton] = None
        self._restart_btn: Optional[QPushButton] = None
        if is_server:
            layout.addSpacing(6)
            self._start_btn = self._mk_text_button("Start", self.start_clicked)
            self._stop_btn = self._mk_text_button("Stop", self.stop_clicked)
            self._restart_btn = self._mk_text_button("Restart", self.restart_clicked)
            layout.addWidget(self._start_btn)
            layout.addWidget(self._stop_btn)
            layout.addWidget(self._restart_btn)
            self.set_state(SupervisorState.IDLE)

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

        # Window-ish glyphs: pop out, float/dock, close.
        layout.addSpacing(2)
        self._popout_btn = _glyph_button("⧉", "Pop out into its own window (own taskbar entry)", self)
        self._popout_btn.clicked.connect(self.popout_clicked)
        layout.addWidget(self._popout_btn)
        self._float_btn = _glyph_button("❐", "Float / re-dock inside the dashboard", self)
        self._float_btn.clicked.connect(self.float_clicked)
        layout.addWidget(self._float_btn)
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
        """Hide Start/Stop/Restart + LED for in-process servers (no subprocess)."""
        for w in (self._start_btn, self._stop_btn, self._restart_btn, self._led):
            if w is not None:
                w.setVisible(bool(visible))

    def set_window_glyphs_visible(self, popout: bool = True, float_: bool = True, close: bool = True) -> None:
        self._popout_btn.setVisible(popout)
        self._float_btn.setVisible(float_)
        self._close_btn.setVisible(close)

    def set_state(self, state: SupervisorState) -> None:
        if self._led is not None:
            self._led.set_color(theme.state_color(state))
            self._led.setToolTip(getattr(state, "name", str(state)).title())
        if self._start_btn is not None:
            self._start_btn.setEnabled(state in (SupervisorState.IDLE, SupervisorState.CRASHED,
                                                 SupervisorState.EXTERNAL))
        if self._stop_btn is not None:
            self._stop_btn.setEnabled(state in (SupervisorState.RUNNING, SupervisorState.STARTING))
        if self._restart_btn is not None:
            self._restart_btn.setEnabled(
                state in (SupervisorState.RUNNING, SupervisorState.CRASHED, SupervisorState.IDLE)
            )

    def set_conn(self, status: str, detail: str = "") -> None:
        status = str(status).lower()
        text = {"connected": "OK", "disconnected": "--", "connecting": "…"}.get(status, "ERR")
        self._conn_badge.setVisible(True)
        self._conn_badge.setText(text)
        self._conn_badge.setStyleSheet(f"QLabel {{ {theme.pill_css(theme.conn_color(status))} }}")
        self._conn_badge.setToolTip(detail or f"snapshot poll: {status}")

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

    def _mk_text_button(self, text: str, signal) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setFlat(True)
        btn.setStyleSheet(theme.header_button_css())
        btn.clicked.connect(signal)
        return btn


__all__ = ["PanelHeaderBar"]
