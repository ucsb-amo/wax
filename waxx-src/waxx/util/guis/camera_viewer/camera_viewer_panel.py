"""Camera Viewer panels - embed beacon's CameraViewerMainWindow in a dashboard.

The main window already provides discovery (v1 Basler servers, v2 camera
servers, liveOD's cameras) + one QDockWidget per camera; we just embed it.
Because it hosts nested QDockWidgets inside its own QMainWindow, the dashboard
embeds the whole QMainWindow visibly so those inner docks remain functional
(same pattern as the TPI panel).

Machine-agnostic: only beacon and the waxx dashboard helpers are imported.
Each side keeps its own saved layout (``layout_key``), so the server and
client dashboards do not overwrite each other's dock arrangement.
"""

from __future__ import annotations

from waxx.util.dashboard.embed_helpers import WidgetPanelBase, embed_main_window


class CameraViewerServerPanel(WidgetPanelBase):
    #: Names the saved layout: <beacon state dir>/camera_viewer_layout_<key>.json.
    LAYOUT_KEY = "dashboard_server"

    def __init__(self, parent=None):
        super().__init__(parent)
        from beacon.camera.viewer.main_window import CameraViewerMainWindow  # noqa: PLC0415

        self._gui = CameraViewerMainWindow(auto_open=False, layout_key=self.LAYOUT_KEY)
        embed_main_window(self, self._gui, embed_as_window=True)

    def cleanup(self) -> None:
        # Forward the dashboard's panel-cleanup hook to the embedded GUI so its
        # discovery thread, rescan timer and camera connections are closed.
        gui_cleanup = getattr(self._gui, "cleanup", None)
        if callable(gui_cleanup):
            try:
                gui_cleanup()
            except Exception:
                pass


class CameraViewerClientPanel(CameraViewerServerPanel):
    """The same viewer on the client dashboard (its own saved layout)."""

    LAYOUT_KEY = "dashboard_client"


__all__ = ["CameraViewerServerPanel", "CameraViewerClientPanel"]
