"""Dashboard main window: lays out panels, persists layout, hosts pop-outs.

The same window class is used for both the server and client dashboards;
the difference is just which panels are added.

Features
--------
* Default layout honours each panel's ``placement`` / ``tab_group``: "tab"
  panels sharing a group in one area are stacked, everything else gets its
  own dock split from its neighbours.  ``Reset to default layout`` re-applies
  that live; ``Revert to saved layout`` goes back to what was loaded.
* QSettings layout persistence (per-host key), saved one second after any
  dock move as well as at close, with a screen-bounds clamp on restore.
* Lazy body realization: a panel's body is built when the panel first
  becomes visible (eager for in-process servers), with a progress line in
  the status bar; heavy imports can be warmed on a background thread.
* Pop-out: any panel can move into an independent top-level window with
  its own taskbar button (:mod:`panel_window`); popped-out panels are
  remembered and restored on the next launch.
* Panels menu (Ctrl+1..9 raises), Layout menu, live Servers menu, Tools
  menu + toolbar (Logs, Errors, Revert/Reset layout), Log dock, persistent
  status-bar summary (running / crashed / idle counts, data dir, host, log).
* Close: graceful shutdown requests fan out to every server, COM servers
  are watched in a modal until their ports are released, survivors are
  hard-killed with a single taskkill.

The window is intentionally permissive: no panel construction is allowed to
block startup.  All errors land in their panel's ErrorBodyWidget plus the
shared log file.
"""

from __future__ import annotations

import importlib
import json
import logging
import platform
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PyQt6.QtCore import QByteArray, QPointF, QRectF, QSettings, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QToolButton,
    QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.dashboard.logging_setup import active_log_dir, pop_boot_warnings
from waxx.util.dashboard.panel_container import ClientPanel, ServerPanel, _PanelDockBase
from waxx.util.dashboard.panel_window import PanelWindow
from waxx.util.dashboard.server_supervisor import SupervisorState, shutdown_all


_LOG = logging.getLogger("waxx.dashboard.window")

_DEFAULT_ORG = "waxx"

_AREA_MAP = {
    "left": Qt.DockWidgetArea.LeftDockWidgetArea,
    "right": Qt.DockWidgetArea.RightDockWidgetArea,
    "top": Qt.DockWidgetArea.TopDockWidgetArea,
    "bottom": Qt.DockWidgetArea.BottomDockWidgetArea,
}


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------

def _make_satellite_dish_icon(size: int = 64) -> QIcon:
    """Rotating radar-style dish on a tripod with a sweep arc."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    try:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fg = QColor(220, 220, 220)
        accent = QColor(120, 220, 140)
        pen = QPen(fg, max(1.0, size * 0.055))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        base_y = size * 0.92
        apex = QPointF(size * 0.50, size * 0.62)
        p.drawLine(apex, QPointF(size * 0.25, base_y))
        p.drawLine(apex, QPointF(size * 0.50, base_y))
        p.drawLine(apex, QPointF(size * 0.75, base_y))
        hub = QPointF(size * 0.50, size * 0.48)
        p.drawLine(apex, hub)
        p.save()
        p.translate(hub)
        p.rotate(-18.0)
        dish_w = size * 0.78
        dish_h = size * 0.16
        dish_rect = QRectF(-dish_w / 2, -dish_h / 2, dish_w, dish_h)
        p.setBrush(QBrush(QColor(90, 90, 90)))
        p.setPen(QPen(fg, max(1.0, size * 0.045)))
        p.drawEllipse(dish_rect)
        p.drawLine(QPointF(-dish_w / 2, 0.0), QPointF(dish_w / 2, 0.0))
        p.restore()
        sweep_pen = QPen(accent, max(1.0, size * 0.05))
        sweep_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(sweep_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for r_frac in (0.18, 0.28, 0.38):
            r = size * r_frac
            p.drawArc(QRectF(hub.x() - r, hub.y() - r, 2 * r, 2 * r), 30 * 16, 120 * 16)
    finally:
        p.end()
    return QIcon(pm)


def _make_remote_icon(size: int = 64) -> QIcon:
    """Hand-held remote with an IR beam."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    try:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fg = QColor(220, 220, 220)
        accent = QColor(255, 200, 90)
        p.save()
        p.translate(size * 0.50, size * 0.55)
        p.rotate(-20.0)
        body_w = size * 0.36
        body_h = size * 0.72
        body_rect = QRectF(-body_w / 2, -body_h / 2, body_w, body_h)
        p.setPen(QPen(fg, max(1.0, size * 0.05)))
        p.setBrush(QBrush(QColor(70, 70, 70)))
        p.drawRoundedRect(body_rect, size * 0.07, size * 0.07)
        p.setBrush(QBrush(accent))
        p.setPen(QPen(accent.darker(150), max(1.0, size * 0.03)))
        p.drawEllipse(QPointF(0.0, -body_h / 2 + size * 0.06), size * 0.045, size * 0.045)
        screen_rect = QRectF(-body_w / 2 + size * 0.06, -body_h / 2 + size * 0.14,
                             body_w - size * 0.12, size * 0.14)
        p.setBrush(QBrush(QColor(120, 200, 255)))
        p.setPen(QPen(QColor(120, 200, 255).darker(150), max(1.0, size * 0.025)))
        p.drawRoundedRect(screen_rect, size * 0.02, size * 0.02)
        p.setBrush(QBrush(fg))
        p.setPen(Qt.PenStyle.NoPen)
        btn_r = size * 0.035
        y0 = -body_h / 2 + size * 0.36
        dx = body_w * 0.25
        dy = size * 0.10
        for row in range(3):
            for col in (-1, 1):
                p.drawEllipse(QPointF(col * dx, y0 + row * dy), btn_r, btn_r)
        p.restore()
        beam_pen = QPen(accent, max(1.0, size * 0.05))
        beam_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(beam_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        origin = QPointF(size * 0.72, size * 0.22)
        for r_frac in (0.10, 0.18, 0.26):
            r = size * r_frac
            p.drawArc(QRectF(origin.x() - r, origin.y() - r, 2 * r, 2 * r), 200 * 16, 80 * 16)
    finally:
        p.end()
    return QIcon(pm)


def dashboard_icon(kind: str) -> QIcon:
    """Return the standard dashboard icon for ``kind`` ('server' or 'client')."""
    return _make_satellite_dish_icon() if kind == "server" else _make_remote_icon()


# Backwards-compatible alias; the palette + QSS now live in ``theme``.
_apply_dark_theme = theme.apply_dark_theme


# ---------------------------------------------------------------------------
# Placement description
# ---------------------------------------------------------------------------

@dataclass
class PanelPlacement:
    """Where a panel goes in the default layout."""

    panel: _PanelDockBase
    area: str = "right"
    placement: str = "dock"      # "dock" | "tab"
    tab_group: str = "main"
    default_visible: bool = True
    realize_eagerly: bool = False
    warm_imports: tuple[str, ...] = ()

    @classmethod
    def coerce(cls, item) -> "PanelPlacement":
        if isinstance(item, PanelPlacement):
            return item
        if len(item) == 2:
            panel, area = item
            return cls(panel, area)
        if len(item) == 3:
            panel, area, _page = item   # page (multi-page mode) is retired
            return cls(panel, area)
        if len(item) == 5:
            panel, area, _page, placement, tab_group = item
            return cls(panel, area, placement or "dock", tab_group or "main")
        raise ValueError("panels entries must be PanelPlacement or (panel, area[, page[, placement, tab_group]])")


class _ImportWarmer(QThread):
    """Import modules on a background thread so the main-thread factories find them cached."""

    progress = pyqtSignal(str)

    def __init__(self, modules: list[str], parent=None):
        super().__init__(parent)
        self._modules = list(dict.fromkeys(modules))

    def run(self) -> None:
        for name in self._modules:
            if self.isInterruptionRequested():
                return
            try:
                importlib.import_module(name)
                self.progress.emit(name)
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("warm import of %s failed: %r", name, exc)


class _DataDirProbe(QThread):
    """Cheap existence check of the shared data dir, off the GUI thread."""

    result = pyqtSignal(object)  # True / False / None

    def run(self) -> None:
        try:
            from waxx.util.dashboard import data_dir_guard  # noqa: PLC0415
            self.result.emit(data_dir_guard.data_dir_present())
        except Exception:
            self.result.emit(None)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class DashboardMainWindow(QMainWindow):
    """Shared main window for server + client dashboards.

    Parameters
    ----------
    kind:
        ``"server"`` or ``"client"`` (QSettings prefix + window title).
    title:
        Visible window title.
    panels:
        Iterable of :class:`PanelPlacement` (or legacy ``(panel, area)``
        tuples).  Declaration order is the default dock order.
    host_ip:
        Used for the per-host QSettings key.
    settings_org:
        QSettings organisation.
    app_id:
        Base AppUserModelID; popped-out windows get ``<app_id>.<panel_id>``.
    """

    #: Delay between two body realizations at startup (ms).
    REALIZE_STAGGER_MS = 30

    def __init__(
        self,
        kind: str,
        title: str,
        panels: Iterable,
        *,
        host_ip: Optional[str] = None,
        settings_org: str = _DEFAULT_ORG,
        app_id: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        _app = QApplication.instance()
        if _app is not None:
            theme.apply_dark_theme(_app)
        self._kind = kind
        self._host_ip = host_ip or "unknown"
        self._app_id = app_id or f"waxx.{kind}Dashboard"
        self.setWindowTitle(title)
        _icon = dashboard_icon(kind)
        self.setWindowIcon(_icon)
        if _app is not None:
            try:
                _app.setWindowIcon(_icon)
            except Exception:
                pass
        self.setDockNestingEnabled(True)
        self.setAnimated(False)
        self.setObjectName(f"DashboardMainWindow::{kind}")
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks | QMainWindow.DockOption.AllowTabbedDocks
        )

        # Central widget: an empty strip so the docks have something to anchor
        # on and can take the whole window.
        center = QWidget(self)
        center.setMinimumSize(0, 0)
        center.setMaximumSize(0, 0)
        self.setCentralWidget(center)

        self._placements: list[PanelPlacement] = [PanelPlacement.coerce(p) for p in panels]
        self._panels: list[_PanelDockBase] = [pl.panel for pl in self._placements]
        self._by_id: dict[str, _PanelDockBase] = {p.panel_id: p for p in self._panels}
        self._placement_of: dict[str, PanelPlacement] = {pl.panel.panel_id: pl for pl in self._placements}

        # Supervisors / links registered by the app.
        self._supervisors: dict[str, object] = {}
        self._supervisor_labels: dict[str, str] = {}
        self._com_ids: set[str] = set()
        self._links: list = []
        self._autostart_ids: list[str] = []
        self._log_panel = None
        self._panel_actions: dict[str, QAction] = {}

        # Pop-out bookkeeping.
        self._popped: dict[str, PanelWindow] = {}

        # Layout bookkeeping.
        self._settings = QSettings(settings_org, "dashboard")
        self._last_loaded_layout: Optional[dict] = None
        self._layout_ready = False   # suppress dirty-saves during startup
        self._layout_save_timer = QTimer(self)
        self._layout_save_timer.setSingleShot(True)
        self._layout_save_timer.setInterval(1000)
        self._layout_save_timer.timeout.connect(self._save_layout)

        # Realization bookkeeping.
        self._realize_queue: deque[_PanelDockBase] = deque()
        self._realize_total = 0
        self._realize_done = 0
        self._startup_realization = True
        self._warmer: Optional[_ImportWarmer] = None

        self._build_status_bar()
        self._place_default_layout()
        for panel in self._panels:
            self._wire_panel(panel)
        self._build_menu()
        self._build_toolbar()
        self._restore_layout()

        warnings = pop_boot_warnings()
        if warnings:
            self.statusBar().showMessage("; ".join(warnings)[:300], 10000)

        if _app is not None:
            _app.focusChanged.connect(self._on_focus_changed)

        # Realize after the window paints once.
        QTimer.singleShot(0, self._begin_realization)

    # ------------------------------------------------------------------
    # Registration API used by the apps
    # ------------------------------------------------------------------

    def register_supervisors(
        self,
        supervisors: dict[str, object],
        com_ids: "set[str] | None" = None,
        labels: "dict[str, str] | None" = None,
    ) -> None:
        """Register ``ServerSupervisor`` instances (stopped gracefully on close).

        ``com_ids`` lists the subset whose specs declared a ``com_label``;
        those are watched in a modal at close so serial ports release
        cleanly.  ``labels`` maps id -> human label for menus and the log.
        """
        self._supervisors.update(supervisors)
        if com_ids:
            self._com_ids.update(com_ids)
        if labels:
            self._supervisor_labels.update(labels)
        for sid, sup in supervisors.items():
            try:
                sup.state_changed.connect(lambda _s: self._refresh_counts())
                if self._log_panel is not None:
                    sup.log_line.connect(self._log_panel.make_appender(sid))
            except Exception:
                pass
        self._refresh_counts()

    def register_links(self, links: Iterable) -> None:
        """Register ``ServerLink`` objects so they are stopped at close."""
        self._links.extend(links)

    def set_autostart(self, ids: Iterable[str]) -> None:
        self._autostart_ids = list(ids)

    def attach_log_panel(self, log_panel) -> None:
        """Feed every registered supervisor's output into *log_panel*."""
        self._log_panel = log_panel
        for sid, sup in self._supervisors.items():
            try:
                sup.log_line.connect(log_panel.make_appender(sid))
            except Exception:
                pass

    def panel(self, panel_id: str) -> Optional[_PanelDockBase]:
        return self._by_id.get(panel_id)

    def schedule_autostart(self, delay_ms: int = 1200) -> None:
        """Start the autostart set after *delay_ms*.

        The delay lets the discovery registry collect a beacon round
        (servers advertise every 0.5 s) so an instance somebody launched by
        hand is detected and marked EXTERNAL instead of being double-started.
        """
        QTimer.singleShot(int(delay_ms), self._do_autostart)

    def _do_autostart(self) -> None:
        for sid in self._autostart_ids:
            sup = self._supervisors.get(sid)
            if sup is None:
                _LOG.warning("autostart: no supervisor for id=%s", sid)
                continue
            try:
                sup.start()
            except Exception:
                _LOG.exception("autostart: start() failed for %s", sid)

    # ------------------------------------------------------------------
    # Default layout
    # ------------------------------------------------------------------

    def _place_default_layout(self) -> None:
        """Add every dock according to its placement spec.

        Per area, in declaration order: each "dock" panel and each distinct
        "tab" group becomes its own dock, split from the previous one in that
        area; members of a tab group are stacked onto the group's first
        panel.
        """
        for area_name in ("left", "right", "top", "bottom"):
            items = [pl for pl in self._placements if (pl.area if pl.area in _AREA_MAP else "right") == area_name]
            if not items:
                continue
            qarea = _AREA_MAP[area_name]
            group_first: dict[tuple, _PanelDockBase] = {}
            for pl in items:
                key = ("tab", pl.tab_group) if pl.placement == "tab" else ("dock", pl.panel.panel_id)
                first = group_first.get(key)
                if first is not None:
                    self.tabifyDockWidget(first, pl.panel)
                    continue
                # Successive addDockWidget() calls into one area stack the
                # docks (vertically for the side areas, horizontally for
                # top/bottom).  splitDockWidget() is deliberately not used:
                # splitting from a dock that is already tabified adds the new
                # dock to that tab stack instead of next to it.
                self.addDockWidget(qarea, pl.panel)
                group_first[key] = pl.panel
            # Raise the first member of every tab stack.
            for first in group_first.values():
                first.raise_()
        for pl in self._placements:
            pl.panel.setVisible(pl.default_visible)

    def _reset_to_default_layout(self) -> None:
        """Live re-application of the placement spec (no restart needed)."""
        for pid in list(self._popped):
            self.return_panel(self._by_id[pid])
        for panel in self._panels:
            try:
                panel.setFloating(False)
                self.removeDockWidget(panel)
            except Exception:
                pass
        self._place_default_layout()
        for panel in self._panels:
            if not panel.isHidden():
                self._ensure_realized(panel)
        self._settings.remove(self._layout_key("geometry"))
        self._settings.remove(self._layout_key("state"))
        self._last_loaded_layout = None
        self._mark_layout_dirty()
        self.statusBar().showMessage("Layout reset to defaults", 3000)

    # ------------------------------------------------------------------
    # Per-panel wiring
    # ------------------------------------------------------------------

    def _wire_panel(self, panel: _PanelDockBase) -> None:
        hdr = panel.header()
        hdr.popout_clicked.connect(lambda _c=False, p=panel: self.toggle_popout(p))
        panel.visibilityChanged.connect(lambda vis, p=panel: self._on_panel_visibility(p, vis))
        panel.dockLocationChanged.connect(lambda _a: self._mark_layout_dirty())
        panel.topLevelChanged.connect(lambda _f: self._mark_layout_dirty())

    def _on_panel_visibility(self, panel: _PanelDockBase, visible: bool) -> None:
        if visible and not panel.is_popped_out():
            self._ensure_realized(panel)
        act = self._panel_actions.get(panel.panel_id)
        if act is not None:
            act.setChecked(not panel.isHidden())
        self._mark_layout_dirty()

    def _on_focus_changed(self, _old: Optional[QWidget], new: Optional[QWidget]) -> None:
        active = self._panel_for_widget(new)
        for panel in self._panels:
            panel.set_active(panel is active)

    def _panel_for_widget(self, w: Optional[QWidget]) -> Optional[_PanelDockBase]:
        while w is not None:
            if isinstance(w, _PanelDockBase):
                return w
            if isinstance(w, PanelWindow):
                return w.panel
            w = w.parentWidget()
        return None

    # ------------------------------------------------------------------
    # Menu / toolbar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # --- Panels -----------------------------------------------------
        panels_menu = mb.addMenu("&Panels")
        for i, panel in enumerate(self._panels):
            act = QAction(f"{panel.icon + '  ' if panel.icon else ''}{panel.label}", self)
            act.setCheckable(True)
            act.setChecked(not panel.isHidden())
            if i < 9:
                act.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            act.triggered.connect(lambda checked, p=panel: self._panel_action_triggered(p, checked))
            panels_menu.addAction(act)
            self._panel_actions[panel.panel_id] = act
        panels_menu.addSeparator()
        self._popped_menu = panels_menu.addMenu("Popped-out windows")
        self._popped_menu.aboutToShow.connect(self._rebuild_popped_menu)
        show_all = QAction("Show all panels", self)
        show_all.triggered.connect(self._show_all_panels)
        panels_menu.addAction(show_all)

        # --- Layout -----------------------------------------------------
        layout_menu = mb.addMenu("&Layout")
        self._revert_act = QAction("Revert to saved layout", self)
        self._revert_act.setToolTip("Go back to the layout that was loaded at startup (or last loaded from a file)")
        self._revert_act.triggered.connect(self._revert_layout)
        layout_menu.addAction(self._revert_act)
        reset = QAction("Reset to default layout", self)
        reset.setToolTip("Re-apply the default dock placement live and forget the saved layout")
        reset.triggered.connect(self._reset_to_default_layout)
        layout_menu.addAction(reset)
        layout_menu.addSeparator()
        save_act = QAction("Save layout to file…", self)
        save_act.triggered.connect(self._save_layout_to_file)
        layout_menu.addAction(save_act)
        load_act = QAction("Load layout from file…", self)
        load_act.triggered.connect(self._load_layout_from_file)
        layout_menu.addAction(load_act)

        # --- Servers (rebuilt on open so it reflects live state) ---------
        self._servers_menu = mb.addMenu("&Servers")
        self._servers_menu.aboutToShow.connect(self._rebuild_servers_menu)

        # --- Tools ------------------------------------------------------
        tools_menu = mb.addMenu("&Tools")
        self._logs_act = QAction("Open log folder", self)
        self._logs_act.setShortcut(QKeySequence("Ctrl+L"))
        self._logs_act.triggered.connect(self._open_log_folder)
        tools_menu.addAction(self._logs_act)
        self._errors_act = QAction("Show recent errors", self)
        self._errors_act.setShortcut(QKeySequence("Ctrl+E"))
        self._errors_act.triggered.connect(self._show_recent_errors)
        tools_menu.addAction(self._errors_act)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setObjectName("DashboardToolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        tb.addAction(self._logs_act)
        tb.addAction(self._errors_act)
        tb.addSeparator()
        tb.addAction(self._revert_act)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self._toolbar = tb

    def _panel_action_triggered(self, panel: _PanelDockBase, checked: bool) -> None:
        if panel.is_popped_out():
            win = self._popped.get(panel.panel_id)
            if win is not None:
                win.showNormal()
                win.raise_()
                win.activateWindow()
            act = self._panel_actions.get(panel.panel_id)
            if act is not None:
                act.setChecked(True)
            return
        if checked:
            panel.show()
            panel.raise_()
            self._ensure_realized(panel)
            body = panel.body_widget()
            if body is not None:
                body.setFocus()
        else:
            panel.hide()

    def _show_all_panels(self) -> None:
        for panel in self._panels:
            if not panel.is_popped_out():
                panel.show()

    def _rebuild_popped_menu(self) -> None:
        menu = self._popped_menu
        menu.clear()
        if not self._popped:
            empty = QAction("(none)", self)
            empty.setEnabled(False)
            menu.addAction(empty)
            return
        for pid, win in sorted(self._popped.items()):
            act = QAction(f"Return '{win.panel.label}' to dashboard", self)
            act.triggered.connect(lambda _c=False, p=win.panel: self.return_panel(p))
            menu.addAction(act)
        menu.addSeparator()
        all_act = QAction("Return all", self)
        all_act.triggered.connect(lambda: [self.return_panel(self._by_id[pid]) for pid in list(self._popped)])
        menu.addAction(all_act)

    def _rebuild_servers_menu(self) -> None:
        menu = self._servers_menu
        menu.clear()
        if not self._supervisors:
            empty = QAction("(no servers registered)", self)
            empty.setEnabled(False)
            menu.addAction(empty)
            return
        _DOT = {
            "RUNNING": "\U0001f7e2", "STARTING": "\U0001f7e1", "STOPPING": "\U0001f7e1",
            "IDLE": "⚪", "CRASHED": "\U0001f534", "FAILED": "\U0001f534",
            "EXTERNAL": "\U0001f7e0",
        }
        for sid in sorted(self._supervisors):
            sup = self._supervisors[sid]
            state_name = getattr(getattr(sup, "state", None), "name", "?")
            label = self._supervisor_labels.get(sid, sid)
            sub = menu.addMenu(f"{_DOT.get(state_name, chr(0x26AA))}  {label}  —  {state_name.lower()}")
            start = QAction("Start", self)
            start.setEnabled(state_name in ("IDLE", "CRASHED", "EXTERNAL"))
            start.triggered.connect(lambda _c=False, s=sup: s.reset_and_start() if getattr(s, "state", None) == SupervisorState.CRASHED else s.start())
            sub.addAction(start)
            stop = QAction("Stop", self)
            stop.setEnabled(state_name in ("RUNNING", "STARTING"))
            stop.triggered.connect(lambda _c=False, s=sup: s.stop())
            sub.addAction(stop)
            restart = QAction("Restart", self)
            restart.setEnabled(state_name in ("RUNNING", "CRASHED", "IDLE"))
            restart.triggered.connect(lambda _c=False, s=sup: s.restart())
            sub.addAction(restart)
            if state_name == "FAILED":
                reset = QAction("Reset failure and start", self)
                reset.triggered.connect(lambda _c=False, s=sup: s.reset_and_start())
                sub.addAction(reset)
            pnl = self._by_id.get(sid)
            if pnl is not None:
                sub.addSeparator()
                show = QAction("Show panel", self)
                show.triggered.connect(lambda _c=False, p=pnl: self._panel_action_triggered(p, True))
                sub.addAction(show)
        menu.addSeparator()
        if self._autostart_ids:
            start_auto = QAction(f"Start autostart set ({len(self._autostart_ids)})", self)
            start_auto.triggered.connect(self._do_autostart)
            menu.addAction(start_auto)
        stop_all = QAction("Stop all running servers", self)
        stop_all.triggered.connect(self._stop_all_servers)
        menu.addAction(stop_all)

    def _stop_all_servers(self) -> None:
        running = [s for s in self._supervisors.values() if getattr(s, "is_alive", lambda: False)()]
        if not running:
            return
        reply = QMessageBox.question(
            self, "Stop all servers?",
            f"Stop {len(running)} running server(s)?  Hardware behind them stops responding until restarted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        for s in running:
            try:
                s.stop()
            except Exception:
                _LOG.exception("stop() failed for %s", getattr(s, "server_id", "?"))

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> None:
        sb = QStatusBar(self)
        sb.setSizeGripEnabled(True)
        self.setStatusBar(sb)
        sb.showMessage("Loading…")

        def _perm(text: str = "", tip: str = "") -> QLabel:
            lbl = QLabel(text, sb)
            lbl.setStyleSheet(f"QLabel {{ color: {theme.FG_MUTED}; padding: 0 8px; }}")
            lbl.setToolTip(tip)
            sb.addPermanentWidget(lbl)
            return lbl

        self._counts_btn = QToolButton(sb)
        self._counts_btn.setAutoRaise(True)
        self._counts_btn.setToolTip("Server states — click for the Servers menu")
        self._counts_btn.setStyleSheet(f"QToolButton {{ color: {theme.FG}; padding: 0 8px; border: 0; }}")
        self._counts_btn.clicked.connect(self._popup_servers_menu)
        sb.addPermanentWidget(self._counts_btn)

        self._datadir_lbl = _perm("", "Shared data directory reachability")
        self._host_lbl = _perm(f"{platform.node()}  {self._host_ip}", "This PC and its lab-subnet IP")

        self._log_btn = QToolButton(sb)
        self._log_btn.setAutoRaise(True)
        self._log_btn.setText("log")
        self._log_btn.setStyleSheet(f"QToolButton {{ color: {theme.FG_MUTED}; padding: 0 8px; border: 0; }}")
        log_dir = active_log_dir()
        self._log_btn.setToolTip(f"Open log folder\n{log_dir}" if log_dir else "Log folder not configured")
        self._log_btn.clicked.connect(self._open_log_folder)
        sb.addPermanentWidget(self._log_btn)

        self._datadir_timer = QTimer(self)
        self._datadir_timer.setInterval(30000)
        self._datadir_timer.timeout.connect(self._probe_data_dir)
        self._datadir_timer.start()
        self._datadir_probe: Optional[_DataDirProbe] = None
        QTimer.singleShot(500, self._probe_data_dir)

    def _popup_servers_menu(self) -> None:
        self._rebuild_servers_menu()
        self._servers_menu.exec(self._counts_btn.mapToGlobal(self._counts_btn.rect().topLeft()))

    def _refresh_counts(self) -> None:
        if not self._supervisors:
            self._counts_btn.setVisible(False)
            return
        tally: dict[str, int] = {}
        for sup in self._supervisors.values():
            name = getattr(getattr(sup, "state", None), "name", "?")
            tally[name] = tally.get(name, 0) + 1
        running = tally.get("RUNNING", 0) + tally.get("EXTERNAL", 0)
        bad = tally.get("CRASHED", 0) + tally.get("FAILED", 0)
        busy = tally.get("STARTING", 0) + tally.get("STOPPING", 0)
        idle = tally.get("IDLE", 0)
        parts = [f"<span style='color:{theme.OK}'>●</span> {running} running"]
        if busy:
            parts.append(f"<span style='color:{theme.WARN}'>●</span> {busy} busy")
        if bad:
            parts.append(f"<span style='color:{theme.ERR}'>●</span> {bad} crashed")
        parts.append(f"<span style='color:{theme.OFF}'>○</span> {idle} idle")
        # QToolButton has no rich text; use a QLabel-like plain rendering.
        plain = f"● {running} running" + (f"   ● {busy} busy" if busy else "") \
            + (f"   ● {bad} crashed" if bad else "") + f"   ○ {idle} idle"
        color = theme.ERR if bad else (theme.WARN if busy else theme.FG)
        self._counts_btn.setText(plain)
        self._counts_btn.setStyleSheet(f"QToolButton {{ color: {color}; padding: 0 8px; border: 0; }}")
        self._counts_btn.setVisible(True)

    def _probe_data_dir(self) -> None:
        if self._datadir_probe is not None and self._datadir_probe.isRunning():
            return
        self._datadir_probe = _DataDirProbe(self)
        self._datadir_probe.result.connect(self._on_data_dir_probe)
        self._datadir_probe.finished.connect(self._datadir_probe.deleteLater)
        self._datadir_probe.start()

    def _on_data_dir_probe(self, present) -> None:
        if present is None:
            self._datadir_lbl.setText("")
            return
        if present:
            self._datadir_lbl.setText("data dir OK")
            self._datadir_lbl.setStyleSheet(f"QLabel {{ color: {theme.FG_MUTED}; padding: 0 8px; }}")
        else:
            self._datadir_lbl.setText("data dir UNREACHABLE")
            self._datadir_lbl.setStyleSheet(f"QLabel {{ color: {theme.ERR}; padding: 0 8px; font-weight: 600; }}")

    # ------------------------------------------------------------------
    # Layout persistence
    # ------------------------------------------------------------------

    def _layout_key(self, which: str) -> str:
        # Per-host so different lab PCs can keep their own arrangement.
        return f"dashboard/{self._kind}/{self._host_ip}/{which}"

    def _mark_layout_dirty(self) -> None:
        if self._layout_ready:
            self._layout_save_timer.start()

    def _restore_layout(self) -> None:
        geo = self._settings.value(self._layout_key("geometry"))
        state = self._settings.value(self._layout_key("state"))
        try:
            if geo is not None:
                self.restoreGeometry(geo)
            if state is not None:
                self.restoreState(state)
        except Exception as exc:
            _LOG.warning("Failed to restore layout, falling back to defaults: %r", exc)
        self._clamp_to_screen()
        if geo is not None or state is not None:
            self._last_loaded_layout = {"geometry": geo, "state": state}

    def _clamp_to_screen(self) -> None:
        """Keep the window on a connected screen (an unplugged monitor must not strand it)."""
        app = QApplication.instance()
        if app is None:
            return
        try:
            frame = self.frameGeometry()
            if app.screenAt(frame.center()) is not None:
                return
            screen = app.primaryScreen()
            if screen is None:
                return
            avail = screen.availableGeometry()
            w = min(frame.width(), avail.width())
            h = min(frame.height(), avail.height())
            self.resize(w, h)
            self.move(avail.left() + (avail.width() - w) // 2, avail.top() + (avail.height() - h) // 2)
            _LOG.info("window geometry was off-screen; moved onto the primary screen")
        except Exception:
            pass

    def _save_layout(self) -> None:
        try:
            self._settings.setValue(self._layout_key("geometry"), self.saveGeometry())
            self._settings.setValue(self._layout_key("state"), self.saveState())
            self._settings.setValue(self._layout_key("popped"), json.dumps(sorted(self._popped)))
            for pid, win in self._popped.items():
                self._settings.setValue(self._layout_key(f"popped_geometry/{pid}"), win.saved_geometry())
            self._settings.sync()
        except Exception as exc:
            _LOG.warning("Failed to save layout: %r", exc)

    def _apply_layout_snapshot(self, snap: Optional[dict]) -> bool:
        if not snap:
            return False
        try:
            geo = snap.get("geometry")
            state = snap.get("state")
            if geo is not None:
                self.restoreGeometry(geo)
            if state is not None:
                self.restoreState(state)
            self._clamp_to_screen()
            return True
        except Exception as exc:
            _LOG.warning("Failed to apply layout snapshot: %r", exc)
            return False

    def _revert_layout(self) -> None:
        if self._apply_layout_snapshot(self._last_loaded_layout):
            self.statusBar().showMessage("Reverted to the saved layout", 3000)
            self._mark_layout_dirty()
        else:
            self.statusBar().showMessage("No saved layout to revert to — use Reset to default layout", 4000)

    # Backwards-compatible names.
    _snap_default_layout = _revert_layout
    _reset_layout = _reset_to_default_layout

    def _save_layout_to_file(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save layout", f"dashboard_layout_{self._kind}.json", "Layout JSON (*.json)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            geom_ba = self.saveGeometry()
            state_ba = self.saveState()
            payload = {
                "kind": self._kind,
                "host_ip": self._host_ip,
                "geometry_hex": bytes(geom_ba).hex(),
                "state_hex": bytes(state_ba).hex(),
                "popped": sorted(self._popped),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._last_loaded_layout = {"geometry": geom_ba, "state": state_ba}
            self._save_layout()
            self.statusBar().showMessage(f"Layout saved to {path}", 5000)
        except Exception as exc:
            _LOG.exception("Save layout failed")
            QMessageBox.warning(self, "Save failed", f"Could not write {path}:\n{exc!r}")

    def _load_layout_from_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load layout", "", "Layout JSON (*.json)")
        if not path_str:
            return
        path = Path(path_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            geom = QByteArray(bytes.fromhex(payload["geometry_hex"]))
            state = QByteArray(bytes.fromhex(payload["state_hex"]))
            for pid in list(self._popped):
                self.return_panel(self._by_id[pid])
            self._apply_layout_snapshot({"geometry": geom, "state": state})
            for pid in payload.get("popped", []):
                if pid in self._by_id:
                    self.pop_out(self._by_id[pid])
            self._last_loaded_layout = {"geometry": geom, "state": state}
            self._save_layout()
            self.statusBar().showMessage(f"Layout loaded from {path}", 5000)
        except Exception as exc:
            _LOG.exception("Load layout failed")
            QMessageBox.warning(self, "Load failed", f"Could not load {path}:\n{exc!r}")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _open_log_folder(self) -> None:
        log_dir = active_log_dir()
        if log_dir is None:
            QMessageBox.warning(self, "No log folder", "Log folder is not configured.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))

    def _show_recent_errors(self) -> None:
        """Raise the Log panel with the warnings-and-errors preset, or fall back to a dialog."""
        log_panel_dock = self._by_id.get("_log")
        if self._log_panel is not None and log_panel_dock is not None:
            self._panel_action_triggered(log_panel_dock, True)
            try:
                self._log_panel.show_errors_only()
            except Exception:
                pass
            return
        log_dir = active_log_dir()
        if log_dir is None:
            QMessageBox.warning(self, "No log folder", "Log folder is not configured.")
            return
        candidates = list(Path(log_dir).glob("*.log"))
        if not candidates:
            QMessageBox.information(self, "Recent errors", "No log files found.")
            return
        target = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            QMessageBox.warning(self, "Read failed", f"Could not read {target}:\n{exc!r}")
            return
        lines = [ln for ln in text.splitlines() if any(lvl in ln for lvl in ("WARNING", "ERROR", "CRITICAL"))]
        msg = QMessageBox(self)
        msg.setWindowTitle(f"Recent errors - {target.name}")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(f"Last warning/error/critical lines in {target.name}:")
        msg.setDetailedText("\n".join(lines[-80:]) or "(no warnings/errors)")
        msg.exec()

    # ------------------------------------------------------------------
    # Body realization
    # ------------------------------------------------------------------

    def _begin_realization(self) -> None:
        # 1. Warm heavy imports on a background thread.
        modules: list[str] = []
        for pl in self._placements:
            modules.extend(pl.warm_imports)
        if modules:
            self._warmer = _ImportWarmer(modules, parent=self)
            self._warmer.start()

        # 2. Queue the bodies that must exist now: eager ones, then everything
        #    currently visible on screen.  Hidden / tabbed-behind panels wait
        #    for their first visibilityChanged(True).
        eager = [pl.panel for pl in self._placements if pl.realize_eagerly]
        visible = [p for p in self._panels
                   if p not in eager and not p.isHidden() and not p.visibleRegion().isEmpty()]
        for p in eager + visible:
            if not p.is_realized() and p.has_body_factory() and p not in self._realize_queue:
                self._realize_queue.append(p)
        self._realize_total = len(self._realize_queue)
        self._realize_done = 0
        if not self._realize_queue:
            self._finish_startup_realization()
            return
        QTimer.singleShot(0, self._realize_next)

    def _realize_next(self) -> None:
        if not self._realize_queue:
            if self._startup_realization:
                self._finish_startup_realization()
            return
        panel = self._realize_queue.popleft()
        self._realize_done += 1
        if self._startup_realization:
            self.statusBar().showMessage(
                f"Loading {self._realize_done}/{self._realize_total}: {panel.label}…")
        try:
            panel.realize_body()
        except Exception:
            _LOG.exception("Unexpected error realizing panel %s", panel.panel_id)
        QTimer.singleShot(self.REALIZE_STAGGER_MS, self._realize_next)

    def _ensure_realized(self, panel: _PanelDockBase) -> None:
        if panel.is_realized() or not panel.has_body_factory():
            return
        if panel in self._realize_queue:
            return
        self._realize_queue.append(panel)
        if not self._startup_realization:
            QTimer.singleShot(0, self._realize_next)

    def _finish_startup_realization(self) -> None:
        self._startup_realization = False
        # The initial restoreState() ran before bodies existed; realizing
        # them resizes docks.  Re-apply the snapshot once so the saved
        # layout actually sticks, then restore popped-out panels.
        if self._last_loaded_layout:
            self._apply_layout_snapshot(self._last_loaded_layout)
        self._restore_popped_panels()
        self._layout_ready = True
        n = sum(1 for p in self._panels if p.is_realized())
        self.statusBar().showMessage(f"Ready — {n} panel(s) loaded", 5000)
        _LOG.info("startup realization complete: %d panel(s) built eagerly", n)

    # ------------------------------------------------------------------
    # Pop-out
    # ------------------------------------------------------------------

    def toggle_popout(self, panel: _PanelDockBase) -> None:
        if panel.is_popped_out():
            self.return_panel(panel)
        else:
            self.pop_out(panel)

    def pop_out(self, panel: _PanelDockBase) -> None:
        """Move *panel* into its own top-level window (own taskbar button)."""
        if panel.is_popped_out():
            return
        self._ensure_realized(panel)
        if not panel.is_realized() and panel.has_body_factory():
            # Realize synchronously: a pop-out of a placeholder is pointless.
            try:
                if panel in self._realize_queue:
                    self._realize_queue.remove(panel)
                panel.realize_body()
            except Exception:
                _LOG.exception("realize before pop-out failed for %s", panel.panel_id)
        panel.setFloating(False)
        win = PanelWindow(panel, app_id=f"{self._app_id}.{panel.panel_id}",
                          title_prefix=f"{self.windowTitle().split(' - ')[0]} — ")
        geo = self._settings.value(self._layout_key(f"popped_geometry/{panel.panel_id}"))
        if geo is not None:
            try:
                win.restoreGeometry(geo)
            except Exception:
                pass
        win.return_requested.connect(self.return_panel)
        self._popped[panel.panel_id] = win
        panel.hide()
        win.show()
        win.raise_()
        win.activateWindow()
        act = self._panel_actions.get(panel.panel_id)
        if act is not None:
            act.setChecked(True)
            act.setText(f"{panel.icon + '  ' if panel.icon else ''}{panel.label}  (popped out)")
        self._mark_layout_dirty()

    def return_panel(self, panel: _PanelDockBase) -> None:
        win = self._popped.pop(panel.panel_id, None)
        if win is None:
            return
        try:
            self._settings.setValue(self._layout_key(f"popped_geometry/{panel.panel_id}"), win.saved_geometry())
        except Exception:
            pass
        win.give_back()
        panel.reattach()
        panel.show()
        panel.raise_()
        win.finish_close()
        act = self._panel_actions.get(panel.panel_id)
        if act is not None:
            act.setText(f"{panel.icon + '  ' if panel.icon else ''}{panel.label}")
            act.setChecked(True)
        self._mark_layout_dirty()

    def _restore_popped_panels(self) -> None:
        raw = self._settings.value(self._layout_key("popped"))
        if not raw:
            return
        try:
            ids = json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            return
        for pid in ids:
            panel = self._by_id.get(pid)
            if panel is not None:
                try:
                    self.pop_out(panel)
                except Exception:
                    _LOG.exception("could not restore popped-out panel %s", pid)

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, ev):  # noqa: N802 - Qt-style
        # Persist layout (including which panels are popped out) first.
        self._layout_ready = False
        self._save_layout()

        # Phase 1: ask every server to shut down cleanly, all at once.
        sups = list(self._supervisors.items())
        for sid, sup in sups:
            try:
                sup.suppress_restart()
                sup.request_terminate()
            except Exception:
                _LOG.exception("request_terminate() failed for id=%s", sid)

        # Phase 2: COM-owning servers are watched in a modal until each has
        # exited (so Windows releases the serial port) or the user decides
        # to force-kill / cancel.
        com_entries: list[tuple[str, str, object]] = []
        for sid in sorted(self._com_ids):
            sup = self._supervisors.get(sid)
            if sup is None:
                continue
            try:
                alive = bool(sup.is_running())
            except Exception:
                alive = False
            if not alive:
                continue
            com_port = ""
            pnl = self._by_id.get(sid)
            if pnl is not None and pnl.header().com_button() is not None:
                com_port = pnl.header().com_button()._label
            com_entries.append((self._supervisor_labels.get(sid, sid), com_port, sup))
        if com_entries:
            try:
                from waxx.util.dashboard.com_shutdown_dialog import (  # noqa: PLC0415
                    ComShutdownDialog,
                    RESULT_USER_CANCELLED,
                )
                dlg = ComShutdownDialog(com_entries, parent=self)
                dlg.exec()
                if dlg.result_reason() == RESULT_USER_CANCELLED:
                    # Keep the dashboard open.  Servers that already honoured
                    # the shutdown request are stopped; restart from the headers.
                    self._layout_ready = True
                    ev.ignore()
                    return
            except Exception:
                _LOG.exception("ComShutdownDialog failed; falling back to fast shutdown")

        # Phase 3: bounded collective wait, then one taskkill for survivors.
        grace = max([1500] + [int(getattr(s, "graceful_stop_timeout_s", 1.5) * 1000) for _, s in sups])
        killed = shutdown_all([s for _, s in sups], grace_ms=grace)
        if killed:
            _LOG.warning("hard-killed at close: %s", killed)

        # Phase 4: popped-out windows, links, panel bodies.
        for pid, win in list(self._popped.items()):
            try:
                self._settings.setValue(self._layout_key(f"popped_geometry/{pid}"), win.saved_geometry())
                win.finish_close()
            except Exception:
                pass
        for link in self._links:
            try:
                link.stop()
            except Exception:
                pass
        if self._warmer is not None and self._warmer.isRunning():
            self._warmer.requestInterruption()
        for panel in self._panels:
            body = panel.body_widget()
            cleanup = getattr(body, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    _LOG.exception("cleanup() failed for panel %s", panel.panel_id)
        try:
            self._settings.sync()
        except Exception:
            pass
        super().closeEvent(ev)


__all__ = ["DashboardMainWindow", "PanelPlacement", "dashboard_icon"]
