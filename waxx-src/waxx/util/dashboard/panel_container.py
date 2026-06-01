"""Dock-widget wrappers for the dashboard.

Each panel is a :class:`QDockWidget` whose **title bar widget** is a
:class:`PanelHeaderBar` (LED + title + start/stop/restart + conn/COM badges).
Qt's built-in float / close buttons in the dock title bar are reused, so
there's no separate pop-out button.
"""

from __future__ import annotations

import logging
import traceback
from typing import Callable, Optional

from PyQt6.QtWidgets import QDockWidget, QFrame, QVBoxLayout, QWidget

from waxx.util.dashboard.embed_helpers import lint_panel
from waxx.util.dashboard.panel_header import PanelHeaderBar
from waxx.util.dashboard.placeholder_body import PlaceholderBody
from waxx.util.dashboard.widgets import ErrorBodyWidget


_LOG = logging.getLogger("waxx.dashboard.panel")


class _PanelDockBase(QDockWidget):
    """Common machinery for server + client dock panels."""

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
        self._body_factory = body_factory
        # Allow shrinking; embedded GUIs sometimes carry oversized minimums.
        self.setMinimumSize(0, 0)

        # Body slot - the only widget the dock holds.  Starts as a placeholder
        # so the dashboard renders before any factory runs.
        #
        # The slot is a QFrame with a unique object name so a per-instance
        # stylesheet draws a persistent border around the panel body.  We
        # paint the border *here* (not via a global QSS rule on QDockWidget)
        # because Qt drops dock chrome QSS as soon as the panel is tabified,
        # floated, or restored from saved state.
        self._body_slot = QFrame(self)
        self._body_slot.setObjectName("PanelBodyFrame")
        self._body_slot.setFrameShape(QFrame.Shape.NoFrame)
        # Visibly distinct border so panels are easy to tell apart even
        # when several are docked side-by-side.  Bumped to 2 px and a
        # lighter colour from the original 1 px / #5a5a5a.
        # Border wraps left/right/bottom only — the header bar paints the
        # top and side borders so that the title bar visually sits inside
        # the same frame as the body.
        self._body_slot.setStyleSheet(
            "QFrame#PanelBodyFrame {"
            " border-left: 2px solid #7a7a7a;"
            " border-right: 2px solid #7a7a7a;"
            " border-bottom: 2px solid #7a7a7a;"
            " border-top: none;"
            " border-bottom-left-radius: 4px;"
            " border-bottom-right-radius: 4px;"
            " background: transparent; }"
        )
        self._body_slot.setMinimumSize(0, 0)
        body_layout = QVBoxLayout(self._body_slot)
        body_layout.setContentsMargins(2, 2, 2, 2)
        body_layout.setSpacing(0)
        self._placeholder = PlaceholderBody("Initializing", parent=self._body_slot)
        body_layout.addWidget(self._placeholder)
        self.setWidget(self._body_slot)

        # Custom title bar widget hosts the LED + buttons.  Qt still draws
        # its built-in float/close icons on the right side.
        self._header = PanelHeaderBar(
            label, is_server=is_server, com_label=com_label, icon=icon,
        )
        self.setTitleBarWidget(self._header)
        self._body_widget: Optional[QWidget] = None

    # ------------------------------------------------------------------
    # Lazy body construction
    # ------------------------------------------------------------------

    def realize_body(self) -> None:
        """Replace the placeholder with the real body widget.

        Called by the dashboard window after the main window is shown, so
        any factory exception surfaces as an :class:`ErrorBodyWidget` rather
        than blocking startup.
        """
        if self._body_widget is not None:
            return
        if self._body_factory is None:
            return
        try:
            widget = self._body_factory()
            if widget is None:
                raise RuntimeError(f"body_factory for '{self.panel_id}' returned None")
        except Exception as exc:
            _LOG.error("Panel %s body_factory failed", self.panel_id, exc_info=True)
            tb = traceback.format_exc()
            widget = ErrorBodyWidget(self.panel_id, exc, tb)
            widget.retry_requested.connect(self._retry_body)

        # Swap in the real body.
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

    def _retry_body(self) -> None:
        self._body_widget = None
        self.realize_body()

    def body_widget(self) -> Optional[QWidget]:
        return self._body_widget

    def header(self) -> PanelHeaderBar:
        return self._header


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
