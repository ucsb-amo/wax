"""Dock-widget wrappers for the dashboard.

Each panel is a :class:`QDockWidget` whose **title bar widget** is a
:class:`PanelHeaderBar` (LED + title + start/stop/restart + conn/COM badges +
pop-out / float / close glyphs).

Bodies are lazy: the dock starts with a :class:`PlaceholderBody` and the
real widget is built by :meth:`realize_body`, which the dashboard calls when
the panel first becomes visible (or eagerly for in-process servers).

Pop-out support: :meth:`detach_for_popout` hands the header and body to an
independent :class:`~waxx.util.dashboard.panel_window.PanelWindow` and
leaves the dock hidden with placeholders; :meth:`reattach` puts them back.
"""

from __future__ import annotations

import logging
import traceback
from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDockWidget, QFrame, QLabel, QVBoxLayout, QWidget

from waxx.util.dashboard import theme
from waxx.util.dashboard.embed_helpers import lint_panel
from waxx.util.dashboard.panel_header import PanelHeaderBar
from waxx.util.dashboard.placeholder_body import PlaceholderBody
from waxx.util.dashboard.widgets import ErrorBodyWidget


_LOG = logging.getLogger("waxx.dashboard.panel")


def _body_frame_css(active: bool) -> str:
    border = theme.ACCENT if active else theme.BORDER
    return (
        "QFrame#PanelBodyFrame {"
        f" border-left: 1px solid {border};"
        f" border-right: 1px solid {border};"
        f" border-bottom: 1px solid {border};"
        " border-top: none;"
        " border-bottom-left-radius: 3px;"
        " border-bottom-right-radius: 3px;"
        " background: transparent; }"
    )


class _PanelDockBase(QDockWidget):
    """Common machinery for server + client dock panels."""

    #: Emitted after the real body has been constructed (or an ErrorBodyWidget swapped in).
    body_realized = pyqtSignal(object)  # the panel

    def __init__(
        self,
        panel_id: str,
        label: str,
        body_factory: Optional[Callable[[], QWidget]],
        *,
        com_label: Optional[str] = None,
        is_server: bool = False,
        icon: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(label, parent)
        self.setObjectName(f"PanelDock::{panel_id}")
        self.panel_id = panel_id
        self.label = label
        self.icon = icon
        self._body_factory = body_factory
        self._active = False
        # Allow shrinking; embedded GUIs sometimes carry oversized minimums.
        self.setMinimumSize(0, 0)
        # The header carries its own float/close glyphs.
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        # Body slot - the only widget the dock holds.  Starts as a placeholder
        # so the dashboard renders before any factory runs.  A QFrame with a
        # unique object name so a per-instance stylesheet draws a persistent
        # border (Qt drops dock chrome QSS as soon as the panel is tabified,
        # floated, or restored from saved state).
        self._body_slot = QFrame(self)
        self._body_slot.setObjectName("PanelBodyFrame")
        self._body_slot.setFrameShape(QFrame.Shape.NoFrame)
        self._body_slot.setStyleSheet(_body_frame_css(False))
        self._body_slot.setMinimumSize(0, 0)
        body_layout = QVBoxLayout(self._body_slot)
        body_layout.setContentsMargins(1, 1, 1, 1)
        body_layout.setSpacing(0)
        self._placeholder = PlaceholderBody("Initializing", parent=self._body_slot)
        body_layout.addWidget(self._placeholder)
        self.setWidget(self._body_slot)

        self._header = PanelHeaderBar(
            label, is_server=is_server, com_label=com_label, icon=icon,
        )
        self.setTitleBarWidget(self._header)
        self._header.float_clicked.connect(lambda: self.setFloating(not self.isFloating()))
        self._header.close_clicked.connect(self.close)
        self._body_widget: Optional[QWidget] = None
        self._popped_out = False

    # ------------------------------------------------------------------
    # Lazy body construction
    # ------------------------------------------------------------------

    def is_realized(self) -> bool:
        return self._body_widget is not None

    def has_body_factory(self) -> bool:
        return self._body_factory is not None

    def realize_body(self) -> None:
        """Replace the placeholder with the real body widget.

        Any factory exception surfaces as an :class:`ErrorBodyWidget` rather
        than propagating.  Idempotent.
        """
        if self._body_widget is not None or self._body_factory is None:
            return
        try:
            widget = self._body_factory()
            if widget is None:
                raise RuntimeError(f"body_factory for '{self.panel_id}' returned None")
        except Exception as exc:
            _LOG.error("Panel %s body_factory failed", self.panel_id, exc_info=True)
            widget = ErrorBodyWidget(self.panel_id, exc, traceback.format_exc())
            widget.retry_requested.connect(self._retry_body)

        old_layout = self._body_slot.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()
        widget.setMinimumSize(0, 0)
        old_layout.addWidget(widget)
        self._body_widget = widget
        for warn in lint_panel(widget, self.panel_id):
            _LOG.warning(warn)
        self.body_realized.emit(self)

    def _retry_body(self) -> None:
        self._body_widget = None
        self.realize_body()

    def body_widget(self) -> Optional[QWidget]:
        return self._body_widget

    def header(self) -> PanelHeaderBar:
        return self._header

    # ------------------------------------------------------------------
    # Focus accent
    # ------------------------------------------------------------------

    def set_active(self, active: bool) -> None:
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        self._body_slot.setStyleSheet(_body_frame_css(active))
        self._header.set_active(active)

    # ------------------------------------------------------------------
    # Pop-out support
    # ------------------------------------------------------------------

    def is_popped_out(self) -> bool:
        return self._popped_out

    def detach_for_popout(self) -> tuple[PanelHeaderBar, QFrame]:
        """Hand the header + body to a pop-out window; leave placeholders here.

        The dock is hidden by the caller.  Returns ``(header, body_slot)``.
        """
        if self._popped_out:
            return self._header, self._body_slot
        self._popped_out = True
        # Placeholder title bar so Qt keeps treating this as a custom-titled dock.
        stub_title = QLabel(f"{self.label} (popped out)", self)
        stub_title.setStyleSheet(f"QLabel {{ color: {theme.FG_MUTED}; padding: 2px 6px; }}")
        self.setTitleBarWidget(stub_title)
        stub_body = QLabel("Popped out — use the Panels menu to bring it back.", self)
        stub_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stub_body.setStyleSheet(f"QLabel {{ color: {theme.FG_MUTED}; }}")
        self.setWidget(stub_body)
        self._body_slot.setParent(None)
        self._header.setParent(None)
        return self._header, self._body_slot

    def reattach(self) -> None:
        """Take the header + body back from the pop-out window."""
        if not self._popped_out:
            return
        self._popped_out = False
        old_title = self.titleBarWidget()
        old_body = self.widget()
        self._header.setParent(self)
        self.setTitleBarWidget(self._header)
        self._body_slot.setParent(self)
        self.setWidget(self._body_slot)
        self._body_slot.show()
        self._header.show()
        for w in (old_title, old_body):
            if w is not None:
                w.setParent(None)
                w.deleteLater()


class ServerPanel(_PanelDockBase):
    """Dock panel for a supervised server."""

    def __init__(
        self,
        panel_id: str,
        label: str,
        body_factory: Optional[Callable[[], QWidget]],
        *,
        com_label: Optional[str] = None,
        icon: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            panel_id, label, body_factory,
            com_label=com_label,
            is_server=True,
            icon=icon,
            parent=parent,
        )


class ClientPanel(_PanelDockBase):
    """Dock panel for a client tool (no Start/Stop)."""

    def __init__(
        self,
        panel_id: str,
        label: str,
        body_factory: Optional[Callable[[], QWidget]],
        *,
        com_label: Optional[str] = None,
        icon: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            panel_id, label, body_factory,
            com_label=com_label,
            is_server=False,
            icon=icon,
            parent=parent,
        )


__all__ = ["ServerPanel", "ClientPanel"]
