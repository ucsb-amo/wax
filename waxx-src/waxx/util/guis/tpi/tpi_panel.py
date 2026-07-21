"""TPI signal-generator panel - embeds the TpiDevicesMainWindow.

The main window already provides ZMQ discovery + one QDockWidget per
discovered TPI-1005-A device; we just embed it.  Because it hosts nested
QDockWidgets inside its own QMainWindow, the dashboard embeds the whole
QMainWindow visibly so those inner docks remain functional (same pattern
as the Basler multi-camera panel).
"""

from __future__ import annotations

from waxx.util.dashboard.embed_helpers import WidgetPanelBase, embed_main_window


class TpiServerPanel(WidgetPanelBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        from waxx.util.guis.tpi.tpi_gui import TpiDevicesMainWindow  # noqa: PLC0415

        self._gui = TpiDevicesMainWindow()
        embed_main_window(self, self._gui, embed_as_window=True)

    def cleanup(self) -> None:
        # Forward the dashboard's panel-cleanup hook to the embedded GUI so
        # its discovery thread / rescan timer get stopped cleanly.
        gui_cleanup = getattr(self._gui, "cleanup", None)
        if callable(gui_cleanup):
            try:
                gui_cleanup()
            except Exception:
                pass


# The GUI is a pure ZMQ client, so the same widget works on either dashboard side.
TpiClientPanel = TpiServerPanel


__all__ = ["TpiServerPanel", "TpiClientPanel"]
