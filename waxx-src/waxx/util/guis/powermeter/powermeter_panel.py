"""Power meter panels - embed beacon's PowerMeterViewerMainWindow in a dashboard.

The main window already provides discovery of ``powermeter_server:*`` servers
and one QDockWidget per Thorlabs console; we just embed it (the Camera Viewer
panel's pattern).  Each side keeps its own saved layout (``layout_key``).

Machine-agnostic: only beacon and the waxx dashboard helpers are imported.
"""

from __future__ import annotations

from waxx.util.dashboard.embed_helpers import WidgetPanelBase, embed_main_window


class PowerMeterServerPanel(WidgetPanelBase):
    #: Names the saved layout: <beacon state dir>/powermeter_viewer_layout_<key>.json.
    LAYOUT_KEY = "dashboard_server"

    def __init__(self, parent=None):
        super().__init__(parent)
        from beacon.powermeter.viewer.main_window import PowerMeterViewerMainWindow  # noqa: PLC0415

        self._gui = PowerMeterViewerMainWindow(layout_key=self.LAYOUT_KEY)
        embed_main_window(self, self._gui, embed_as_window=True)

    def cleanup(self) -> None:
        # The dashboard delivers no closeEvent to an embedded GUI: stop its
        # discovery thread, rescan timer and subscriptions here.
        gui_cleanup = getattr(self._gui, "cleanup", None)
        if callable(gui_cleanup):
            try:
                gui_cleanup()
            except Exception:
                pass


class PowerMeterClientPanel(PowerMeterServerPanel):
    """The same viewer on the client dashboard (its own saved layout)."""

    LAYOUT_KEY = "dashboard_client"


__all__ = ["PowerMeterServerPanel", "PowerMeterClientPanel"]
