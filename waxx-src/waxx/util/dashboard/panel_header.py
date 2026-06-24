"""Panel header bar used as the dock's title-bar widget.

Lays out (left-to-right) on a single row:

* LED indicator for supervisor state (server panels only)
* Panel title
* Server-only Start / Stop / Restart buttons
* Conn badge (snapshot poller status)
* Optional :class:`ComStatusButton`

Qt's built-in QDockWidget float / close buttons are reused (the panel
container places this header as the dock's title-bar widget, so Qt still
draws its native dock controls on the right).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from waxx.util.dashboard.server_supervisor import SupervisorState
from waxx.util.dashboard.widgets import ComStatusButton


_STATE_COLOR = {
    SupervisorState.IDLE: "#777",
    SupervisorState.STARTING: "#d4a017",
    SupervisorState.RUNNING: "#2e8b57",
    SupervisorState.STOPPING: "#d4a017",
    SupervisorState.CRASHED: "#b22222",
    SupervisorState.FAILED: "#b22222",
    SupervisorState.EXTERNAL: "#2e8b57",
}

_CONN_COLOR = {
    "connected": "#2e8b57",
    "disconnected": "#555",
    "error": "#b22222",
}


class _LedDot(QLabel):
    """Tiny circular LED rendered via stylesheet."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._color = "#777"
        self._refresh()

    def set_color(self, css_color: str) -> None:
        self._color = css_color
        self._refresh()

    def _refresh(self) -> None:
        self.setStyleSheet(
            f"QLabel {{ background-color: {self._color}; border-radius: 6px;"
            " border: 1px solid #222; }"
        )


class PanelHeaderBar(QWidget):
    """Compact, single-row title bar suitable for ``QDockWidget.setTitleBarWidget``.

    Server panels show LED + title + Start/Stop/Restart + conn + COM.
    Client panels show just title + conn + COM (no supervisor controls,
    no LED).
    """

    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    restart_clicked = pyqtSignal()

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
        # Dark, compact title bar styling.  The top/left/right borders
        # match the body frame so the header reads as part of the same
        # bordered panel.
        self.setStyleSheet(
            "QWidget#PanelHeaderBar { background-color: #2b2b2b;"
            " border-top: 2px solid #7a7a7a;"
            " border-left: 2px solid #7a7a7a;"
            " border-right: 2px solid #7a7a7a;"
            " border-bottom: 1px solid #1a1a1a;"
            " border-top-left-radius: 4px;"
            " border-top-right-radius: 4px; }"
            "QWidget#PanelHeaderBar QLabel { color: #e0e0e0; }"
        )

        layout = QHBoxLayout(self)
        # Top/bottom margins must be >= the border width (2 px) so child
        # widgets don't paint over the top border.
        layout.setContentsMargins(8, 3, 6, 3)
        layout.setSpacing(4)

        # Icon + LED + title.  The per-panel emoji comes first so users
        # can spot a panel at a glance, and the supervisor LED sits just
        # to its right (between the icon and the title) so it reads as
        # part of the panel's identity rather than a stray dot.
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

        title = QLabel(label, self)
        title.setStyleSheet("QLabel { color: #e8e8e8; font-weight: 600; }")
        layout.addWidget(title)

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

        layout.addStretch(1)

        # Conn badge - small pill, hidden by default.
        self._conn_badge = QLabel("\u2014", self)
        self._conn_badge.setVisible(False)
        self._conn_badge.setStyleSheet(
            "QLabel { background: #555; color: white; border-radius: 6px;"
            " padding: 1px 6px; font-size: 10px; font-weight: 600; }"
        )
        layout.addWidget(self._conn_badge)

        # COM button - only present when the spec declares com_label.
        self._com_btn: Optional[ComStatusButton] = None
        if com_label:
            self._com_btn = ComStatusButton(com_label, parent=self)
            layout.addWidget(self._com_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def com_button(self) -> Optional[ComStatusButton]:
        return self._com_btn

    def set_state(self, state: SupervisorState) -> None:
        if self._led is not None:
            self._led.set_color(_STATE_COLOR.get(state, "#777"))
        if self._start_btn is not None:
            self._start_btn.setEnabled(state in (SupervisorState.IDLE, SupervisorState.CRASHED))
        if self._stop_btn is not None:
            self._stop_btn.setEnabled(state in (SupervisorState.RUNNING, SupervisorState.STARTING))
        if self._restart_btn is not None:
            self._restart_btn.setEnabled(
                state in (SupervisorState.RUNNING, SupervisorState.CRASHED, SupervisorState.IDLE)
            )

    def set_conn(self, status: str, detail: str = "") -> None:
        if status not in _CONN_COLOR:
            status = "error"
        bg = _CONN_COLOR[status]
        self._conn_badge.setVisible(True)
        self._conn_badge.setText({"connected": "OK", "disconnected": "--", "error": "ERR"}[status])
        self._conn_badge.setStyleSheet(
            f"QLabel {{ background: {bg}; color: white; border-radius: 6px;"
            " padding: 1px 6px; font-size: 10px; font-weight: 600; }"
        )
        if detail:
            self._conn_badge.setToolTip(detail)

    def _mk_text_button(self, text: str, signal) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setFlat(True)
        btn.setStyleSheet(
            "QPushButton { color: #ddd; padding: 1px 8px; font-size: 11px;"
            " border: 1px solid #444; border-radius: 3px;"
            " background-color: #3a3a3a; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
            "QPushButton:disabled { color: #666; border-color: #333;"
            " background-color: #2f2f2f; }"
        )
        btn.clicked.connect(signal)
        return btn


__all__ = ["PanelHeaderBar"]
