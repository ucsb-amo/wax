"""Reusable Qt widgets for the kexp dashboard framework.

Three small widgets that several panels reuse:

* :class:`ComStatusButton` - red/yellow/green pill in panel headers for any
  server that owns a serial COM port; click toggles disconnect/reconnect.
* :class:`CollapsibleGroupBox` - drop-in replacement for ``QGroupBox`` whose
  contents can be collapsed via a chevron toggle, letting dashboard docks
  shrink to fit available screen real estate.
* :class:`ErrorBodyWidget` - red banner + traceback shown in place of a panel
  body when its ``body_factory`` raised (Threat 1 in plan).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# Match SerialConnection status strings (kept as plain str to avoid Qt depending
# on the serial helper module).
_COM_STATUS_CONNECTED = "connected"
_COM_STATUS_CONNECTING = "connecting"
_COM_STATUS_DISCONNECTED = "disconnected"
_COM_STATUS_ERROR = "error"

_COM_PALETTE = {
    _COM_STATUS_CONNECTED: ("#2e8b57", "white", "\u2713"),
    _COM_STATUS_CONNECTING: ("#d4a017", "white", "\u22ef"),
    _COM_STATUS_DISCONNECTED: ("#888", "white", "\u00b7"),
    _COM_STATUS_ERROR: ("#b22222", "white", "\u2717"),
}


class ComStatusButton(QToolButton):
    """Clickable pill showing serial-port connection status.

    The button is *driven* by the panel's snapshot poller via
    :meth:`set_status`.  Clicking emits :attr:`disconnect_requested` (when
    currently connected) or :attr:`reconnect_requested` (when currently
    disconnected or in error).

    Click is debounced for ``debounce_ms`` after every emit to prevent rapid
    toggle storms (e.g. accidental double-clicks).  The button is non-clickable
    in the ``connecting`` state.
    """

    disconnect_requested = pyqtSignal()
    reconnect_requested = pyqtSignal()

    def __init__(
        self,
        label: str = "COM",
        debounce_ms: int = 1000,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._label = label
        self._status = _COM_STATUS_DISCONNECTED
        self._tooltip_detail = ""
        self._debounce_ms = int(debounce_ms)
        self._debounce_active = False

        self.setAutoRaise(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.clicked.connect(self._on_clicked)
        self._apply_style()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(
        self,
        status: str,
        *,
        port: Optional[str] = None,
        detail: str = "",
    ) -> None:
        """Update status from snapshot.

        Parameters
        ----------
        status:
            One of ``"connected"``, ``"connecting"``, ``"disconnected"``, ``"error"``.
        port:
            Optional port name to use as the button label.
        detail:
            Tooltip detail text (baud, latency, last error).
        """
        if status not in _COM_PALETTE:
            status = _COM_STATUS_ERROR
        self._status = status
        if port:
            self._label = port
        self._tooltip_detail = detail
        self._apply_style()

    def status(self) -> str:
        return self._status

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        bg, fg, glyph = _COM_PALETTE[self._status]
        self.setText(f"{self._label} {glyph}")
        self.setStyleSheet(
            "QToolButton {"
            f" background-color: {bg};"
            f" color: {fg};"
            " border: none;"
            " border-radius: 8px;"
            " padding: 2px 8px;"
            " font-weight: 600;"
            " font-size: 11px;"
            "}"
            "QToolButton:disabled { background-color: #aaa; }"
            "QToolButton:hover { padding: 2px 9px; }"
        )
        tip = f"{self._label} - {self._status}"
        if self._tooltip_detail:
            tip = f"{tip}\n{self._tooltip_detail}"
        self.setToolTip(tip)
        # Connecting state ignores clicks.
        self.setEnabled(self._status != _COM_STATUS_CONNECTING and not self._debounce_active)

    def _on_clicked(self) -> None:
        if self._debounce_active:
            return
        if self._status == _COM_STATUS_CONNECTED:
            reply = QMessageBox.question(
                self,
                "Disconnect COM port?",
                f"Disconnect {self._label}?\n"
                "Hardware will stop responding until reconnect.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.disconnect_requested.emit()
        else:
            # disconnected or error
            self.reconnect_requested.emit()

        self._debounce_active = True
        self._apply_style()
        QTimer.singleShot(self._debounce_ms, self._end_debounce)

    def _end_debounce(self) -> None:
        self._debounce_active = False
        self._apply_style()


class CollapsibleGroupBox(QWidget):
    """A QGroupBox-like container whose contents can be collapsed.

    Use in place of ``QGroupBox`` for any section of a panel that is rarely
    touched, so the user can shrink the dock to just the essentials.  The
    title bar shows a chevron (\u25b8 / \u25be) that toggles the body.
    """

    toggled = pyqtSignal(bool)  # True when expanded

    def __init__(
        self,
        title: str = "",
        *,
        expanded: bool = True,
        scrollable: bool = False,
        max_expanded_height: int = 240,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._expanded = bool(expanded)
        self._scrollable = bool(scrollable)
        self._max_expanded_height = int(max_expanded_height)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(self._expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: 600; padding: 2px 4px; }"
        )
        self._toggle.clicked.connect(self._on_toggle)
        outer.addWidget(self._toggle)

        self._content = QFrame(self)
        self._content.setFrameShape(QFrame.Shape.NoFrame)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 0, 4, 4)

        if self._scrollable:
            # Wrap the content in a QScrollArea so expanding the box does
            # not push the parent panel taller than ``max_expanded_height``.
            self._scroll: Optional[QScrollArea] = QScrollArea(self)
            self._scroll.setWidgetResizable(True)
            self._scroll.setFrameShape(QFrame.Shape.NoFrame)
            self._scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._scroll.setMaximumHeight(self._max_expanded_height)
            self._scroll.setWidget(self._content)
            outer.addWidget(self._scroll)
            self._scroll.setVisible(self._expanded)
        else:
            self._scroll = None
            outer.addWidget(self._content)
            self._content.setVisible(self._expanded)

    def setContentLayout(self, layout) -> None:  # noqa: N802 - Qt-style
        """Replace the content layout with a caller-provided layout."""
        # Re-parent any existing children safely.
        old = self._content.layout()
        if old is not None:
            QWidget().setLayout(old)
        self._content.setLayout(layout)
        self._content_layout = layout

    def addWidget(self, w: QWidget) -> None:  # noqa: N802 - Qt-style
        self._content_layout.addWidget(w)

    def isExpanded(self) -> bool:  # noqa: N802
        return self._expanded

    def setExpanded(self, expanded: bool) -> None:  # noqa: N802
        if expanded == self._expanded:
            return
        self._expanded = bool(expanded)
        self._toggle.setChecked(self._expanded)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )
        if self._scroll is not None:
            self._scroll.setVisible(self._expanded)
        else:
            self._content.setVisible(self._expanded)
        self.toggled.emit(self._expanded)

    def _on_toggle(self) -> None:
        self.setExpanded(self._toggle.isChecked())


class ErrorBodyWidget(QWidget):
    """Fallback body shown when a panel's ``body_factory`` raised.

    Displays a red banner, the exception type + message, an expandable
    traceback view, and a Retry button.  Construction never raises; this
    widget is the last line of defense against startup failures.
    """

    retry_requested = pyqtSignal()

    def __init__(
        self,
        panel_id: str,
        exception: BaseException,
        traceback_text: str,
        *,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        header = QLabel(
            f"<b>Panel '{panel_id}' failed to load.</b><br>"
            f"<span style='color:#b22222'>{type(exception).__name__}: {exception}</span>"
        )
        header.setWordWrap(True)
        header.setStyleSheet(
            "QLabel { background-color: #ffe4e1; color: #4a0000; padding: 6px; border: 1px solid #b22222; border-radius: 4px; }"
        )
        layout.addWidget(header)

        tb_view = QPlainTextEdit(self)
        tb_view.setReadOnly(True)
        tb_view.setPlainText(traceback_text)
        tb_view.setStyleSheet("QPlainTextEdit { font-family: Consolas, monospace; font-size: 10px; }")
        layout.addWidget(tb_view, 1)

        retry = QPushButton("Retry", self)
        retry.clicked.connect(self.retry_requested.emit)
        layout.addWidget(retry)


__all__ = [
    "ComStatusButton",
    "CollapsibleGroupBox",
    "ErrorBodyWidget",
]
