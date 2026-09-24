"""Independent top-level window hosting one popped-out dashboard panel.

A floating ``QDockWidget`` is a Qt *tool* window owned by the dashboard:
Windows hides it whenever the dashboard is minimized and never gives it a
taskbar button.  :class:`PanelWindow` is parentless, so it gets its own
taskbar entry and Alt-Tab slot, stays up while the dashboard is minimized,
and can sit on another monitor.

Mechanics: the dock hands over its header bar and body frame
(:meth:`_PanelDockBase.detach_for_popout`); this window shows them stacked.
The header keeps every wire it had (supervisor LED, Start/Stop, COM pill),
so nothing needs re-plumbing.  Closing the window, or clicking the header's
pop-out glyph again, returns the panel to the dashboard.

On Windows each popped-out window is given its own AppUserModelID so the
taskbar does not stack it under the dashboard's button (best effort, needs
pywin32; silently skipped otherwise).
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from PyQt6.QtCore import QByteArray, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from waxx.util.dashboard import theme


_LOG = logging.getLogger("waxx.dashboard.panel_window")


def emoji_icon(glyph: Optional[str], fallback: Optional[QIcon] = None) -> QIcon:
    """Render a single emoji into a multi-size QIcon (taskbar + title bar)."""
    if not glyph:
        return fallback or QIcon()
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            font = QFont("Segoe UI Emoji")
            font.setPixelSize(int(size * 0.8))
            p.setFont(font)
            p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
        finally:
            p.end()
        icon.addPixmap(pm)
    return icon


def set_window_app_id(widget: QWidget, app_id: str) -> bool:
    """Give one top-level window its own Windows taskbar identity.

    Windows groups taskbar buttons by AppUserModelID, which Qt sets once per
    process.  Setting a distinct id on the popped-out window's HWND makes it
    its own taskbar button with its own icon.  Returns True on success;
    False (never raises) off Windows or without pywin32.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        import pythoncom  # noqa: PLC0415
        from win32com.propsys import propsys, pscon  # noqa: PLC0415

        hwnd = int(widget.winId())
        store = propsys.SHGetPropertyStoreForWindow(hwnd, propsys.IID_IPropertyStore)
        store.SetValue(
            pscon.PKEY_AppUserModel_ID,
            propsys.PROPVARIANTType(app_id, pythoncom.VT_LPWSTR),
        )
        store.Commit()
        return True
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("set_window_app_id(%r) failed: %r", app_id, exc)
        return False


class PanelWindow(QMainWindow):
    """Parentless window showing one panel's header + body.

    Signals
    -------
    return_requested(panel)
        The user closed the window or clicked the header glyph; the dashboard
        should call :meth:`give_back` and re-dock the panel.
    """

    return_requested = pyqtSignal(object)

    def __init__(self, panel, *, app_id: Optional[str] = None, title_prefix: str = ""):
        super().__init__(None)  # parentless on purpose: own taskbar entry
        self._panel = panel
        self._returning = False
        self.setObjectName(f"PanelWindow::{panel.panel_id}")
        self.setWindowTitle(f"{title_prefix}{panel.label}")
        self.setWindowIcon(emoji_icon(panel.icon, fallback=self.windowIcon()))
        self.setStyleSheet(f"QMainWindow {{ background: {theme.BG}; }}")

        header, body = panel.detach_for_popout()
        header.set_window_glyphs_visible(popout=True, float_=False, close=False)
        header._popout_btn.setToolTip("Return this panel to the dashboard")

        container = QWidget(self)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(0)
        header.setParent(container)
        body.setParent(container)
        vbox.addWidget(header)
        vbox.addWidget(body, 1)
        header.show()
        body.show()
        self.setCentralWidget(container)
        self._header = header
        self._body = body

        self.resize(max(420, body.sizeHint().width() + 16),
                    max(300, body.sizeHint().height() + header.sizeHint().height() + 16))
        if app_id:
            # winId() forces native window creation, which is what we want:
            # the property store needs a real HWND.
            set_window_app_id(self, app_id)

    @property
    def panel(self):
        return self._panel

    def give_back(self) -> None:
        """Detach header + body so the dock can :meth:`reattach` them."""
        self._header.set_window_glyphs_visible(popout=True, float_=True, close=True)
        self._header._popout_btn.setToolTip("Pop out into its own window (own taskbar entry)")
        self._header.setParent(None)
        self._body.setParent(None)

    def saved_geometry(self) -> QByteArray:
        return self.saveGeometry()

    def closeEvent(self, ev):  # noqa: N802 - Qt API
        if not self._returning:
            self._returning = True
            # Let the dashboard re-dock; it destroys this window afterwards.
            self.return_requested.emit(self._panel)
            ev.ignore()
            return
        super().closeEvent(ev)

    def finish_close(self) -> None:
        """Called by the dashboard after the panel has been re-docked."""
        self._returning = True
        self.close()
        self.deleteLater()


__all__ = ["PanelWindow", "emoji_icon", "set_window_app_id"]
