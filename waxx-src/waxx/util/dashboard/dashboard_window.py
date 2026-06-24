"""Dashboard main window: lays out panels and persists layout.

The same window class is used for both the server and client dashboards
- the difference is just which panels are added.

Features
--------
* QSettings-based layout persistence (per-host key) with try/except fallback
  to default layout when state is corrupt
* View menu listing every panel for show/hide
* Toolbar with: 'Open log folder', 'Recent errors', 'Reset layout'
* Boot-warning surface (warnings collected during logging setup)
* Lazy body realization so the window appears immediately even if a panel
  body's constructor is slow

The window is intentionally permissive: no panel construction is allowed to
block startup.  All errors land in their panel's ErrorBodyWidget plus the
shared log file.
"""

from __future__ import annotations

import base64
import json
import logging
import platform
from pathlib import Path
from typing import Iterable, Optional

from PyQt6.QtCore import Qt, QSettings, QTimer, QUrl, QByteArray, QPointF, QRectF
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QDesktopServices,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from waxx.util.dashboard.logging_setup import active_log_dir, pop_boot_warnings
from waxx.util.dashboard.panel_container import ClientPanel, ServerPanel, _PanelDockBase


_LOG = logging.getLogger("waxx.dashboard.window")

# QSettings keys.
_DEFAULT_ORG = "waxx"
_SETTINGS_GEOMETRY = "dashboard/{kind}/geometry"
_SETTINGS_STATE = "dashboard/{kind}/state"


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

        # Tripod legs
        base_y = size * 0.92
        apex = QPointF(size * 0.50, size * 0.62)
        p.drawLine(apex, QPointF(size * 0.25, base_y))
        p.drawLine(apex, QPointF(size * 0.50, base_y))
        p.drawLine(apex, QPointF(size * 0.75, base_y))

        # Vertical mast from tripod apex up to dish hub
        hub = QPointF(size * 0.50, size * 0.48)
        p.drawLine(apex, hub)

        # Radar dish: an elongated tilted ellipse (the classic side-on
        # rotating-radar silhouette).
        p.save()
        p.translate(hub)
        p.rotate(-18.0)
        dish_w = size * 0.78
        dish_h = size * 0.16
        dish_rect = QRectF(-dish_w / 2, -dish_h / 2, dish_w, dish_h)
        p.setBrush(QBrush(QColor(90, 90, 90)))
        p.setPen(QPen(fg, max(1.0, size * 0.045)))
        p.drawEllipse(dish_rect)
        # Center spine across the dish for a hint of structure.
        p.drawLine(QPointF(-dish_w / 2, 0.0), QPointF(dish_w / 2, 0.0))
        p.restore()

        # Sweep arc (radar return) above the dish.
        sweep_pen = QPen(accent, max(1.0, size * 0.05))
        sweep_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(sweep_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for r_frac in (0.18, 0.28, 0.38):
            r = size * r_frac
            arc_rect = QRectF(hub.x() - r, hub.y() - r, 2 * r, 2 * r)
            p.drawArc(arc_rect, 30 * 16, 120 * 16)
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

        # Remote body (rounded rect, tilted).
        p.save()
        p.translate(size * 0.50, size * 0.55)
        p.rotate(-20.0)
        body_w = size * 0.36
        body_h = size * 0.72
        body_rect = QRectF(-body_w / 2, -body_h / 2, body_w, body_h)
        p.setPen(QPen(fg, max(1.0, size * 0.05)))
        p.setBrush(QBrush(QColor(70, 70, 70)))
        p.drawRoundedRect(body_rect, size * 0.07, size * 0.07)

        # IR emitter at the top.
        p.setBrush(QBrush(accent))
        p.setPen(QPen(accent.darker(150), max(1.0, size * 0.03)))
        p.drawEllipse(QPointF(0.0, -body_h / 2 + size * 0.06),
                      size * 0.045, size * 0.045)

        # Screen / display.
        screen_rect = QRectF(-body_w / 2 + size * 0.06,
                             -body_h / 2 + size * 0.14,
                             body_w - size * 0.12,
                             size * 0.14)
        p.setBrush(QBrush(QColor(120, 200, 255)))
        p.setPen(QPen(QColor(120, 200, 255).darker(150), max(1.0, size * 0.025)))
        p.drawRoundedRect(screen_rect, size * 0.02, size * 0.02)

        # Three button rows (2 per row).
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

        # IR beam waves emanating from emitter (upper-right).
        beam_pen = QPen(accent, max(1.0, size * 0.05))
        beam_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(beam_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        origin = QPointF(size * 0.72, size * 0.22)
        for r_frac in (0.10, 0.18, 0.26):
            r = size * r_frac
            beam_rect = QRectF(origin.x() - r, origin.y() - r, 2 * r, 2 * r)
            p.drawArc(beam_rect, 200 * 16, 80 * 16)
    finally:
        p.end()
    return QIcon(pm)


def dashboard_icon(kind: str) -> QIcon:
    """Return the standard dashboard icon for ``kind`` ('server' or 'client')."""
    if kind == "server":
        return _make_satellite_dish_icon()
    return _make_remote_icon()


def _apply_dark_theme(app: QApplication) -> None:
    """Install a Fusion dark palette + QSS so every embedded panel inherits dark colors."""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(55, 55, 55))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Link, QColor(80, 160, 240))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(60, 110, 180))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(120, 120, 120))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(120, 120, 120))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(120, 120, 120))
    app.setPalette(palette)
    app.setStyleSheet(
        "QToolTip { color: #ddd; background-color: #2b2b2b; border: 1px solid #555; }"
        " QDockWidget { border: 2px solid #7a7a7a; }"
        " QDockWidget::title { background: #2b2b2b; color: #ddd; padding: 2px;"
        " border-bottom: 1px solid #5a5a5a; }"
        # Persistent body border on every panel: targeted by object name so
        # the QSS specificity stays high enough that re-layouts don't drop it.
        " QFrame#PanelBodyFrame { border: 2px solid #7a7a7a; border-radius: 4px;"
        " background: transparent; }"
        " QMainWindow::separator { background: #555; width: 3px; height: 3px; }"
        " QMainWindow::separator:hover { background: #888; }"
        " QTabBar::tab { background: #3a3a3a; color: #ddd; padding: 4px 10px;"
        " border: 1px solid #2a2a2a; }"
        " QTabBar::tab:selected { background: #4a4a4a; }"
        " QMenuBar { background: #2b2b2b; color: #ddd; }"
        " QMenuBar::item:selected { background: #444; }"
        " QMenu { background: #2b2b2b; color: #ddd; border: 1px solid #444; }"
        " QMenu::item:selected { background: #444; }"
        " QStatusBar { background: #2b2b2b; color: #aaa; }"
        " QToolBar { background: #2b2b2b; border: 0; spacing: 4px; }"
    )


class DashboardMainWindow(QMainWindow):
    """Shared main window for server + client dashboards.

    Parameters
    ----------
    kind:
        ``"server"`` or ``"client"`` (used as the QSettings prefix and the
        window title).
    title:
        Visible window title.
    panels:
        Iterable of (panel, dock_area) tuples.  Order matters: first panel
        in a given dock area becomes the "anchor" panel; subsequent panels
        in the same area are tabified onto it.
    """

    def __init__(
        self,
        kind: str,
        title: str,
        panels: Iterable[tuple],
        *,
        host_ip: Optional[str] = None,
        settings_org: str = _DEFAULT_ORG,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        # Dark theme covers the whole app + every embedded panel.
        _app = QApplication.instance()
        if _app is not None:
            _apply_dark_theme(_app)
        self._kind = kind
        self._host_ip = host_ip or "unknown"
        self.setWindowTitle(title)
        # Set both window and (best-effort) application icon so the taskbar
        # entry uses the dashboard-flavored glyph as well.
        _icon = dashboard_icon(kind)
        self.setWindowIcon(_icon)
        if _app is not None:
            try:
                _app.setWindowIcon(_icon)
            except Exception:
                pass
        self.setDockNestingEnabled(True)
        self.setAnimated(False)  # disable dock-snap animations for snappier drag/drop
        self.setObjectName(f"DashboardMainWindow::{kind}")

        # Initial status bar message; replaced after we know how many panels loaded.
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Loading...")

        # Build dock area map.
        area_map = {
            "left": Qt.DockWidgetArea.LeftDockWidgetArea,
            "right": Qt.DockWidgetArea.RightDockWidgetArea,
            "top": Qt.DockWidgetArea.TopDockWidgetArea,
            "bottom": Qt.DockWidgetArea.BottomDockWidgetArea,
        }
        self._area_map = area_map

        # Normalize panels to (panel, area, page).  page defaults to "main".
        normalized: list[tuple[_PanelDockBase, str, str]] = []
        page_order: list[str] = []
        for item in panels:
            if len(item) == 2:
                pnl, area_name = item
                page = "main"
            elif len(item) == 3:
                pnl, area_name, page = item
                page = page or "main"
            else:
                raise ValueError(
                    "panels entries must be (panel, area) or (panel, area, page)"
                )
            normalized.append((pnl, area_name, page))
            if page not in page_order:
                page_order.append(page)

        self._normalized_panels = normalized

        self._panels: list[_PanelDockBase] = []
        self._page_windows: dict[str, QMainWindow] = {}
        self._tabs: Optional[QTabWidget] = None

        if len(page_order) <= 1:
            # Single-page: docks belong directly to this QMainWindow (the
            # historical behavior).
            anchors: dict[str, _PanelDockBase] = {}
            self._page_windows[page_order[0] if page_order else "main"] = self
            for panel, area_name, _page in normalized:
                qarea = area_map.get(area_name, Qt.DockWidgetArea.RightDockWidgetArea)
                if area_name in anchors:
                    self.tabifyDockWidget(anchors[area_name], panel)
                else:
                    self.addDockWidget(qarea, panel)
                    anchors[area_name] = panel
                self._panels.append(panel)
        else:
            # Multi-page: outer QTabWidget where each tab hosts an inner
            # QMainWindow with its own docks.  This is the only way Qt lets
            # us host QDockWidgets on more than one "page" cleanly.
            self._tabs = QTabWidget(self)
            self._tabs.setDocumentMode(True)
            self._tabs.setTabPosition(QTabWidget.TabPosition.North)
            self._tabs.setMovable(False)
            self.setCentralWidget(self._tabs)

            page_anchors: dict[str, dict[str, _PanelDockBase]] = {}
            for page in page_order:
                inner = QMainWindow(self)
                inner.setObjectName(f"DashboardPageWindow::{kind}::{page}")
                inner.setDockNestingEnabled(True)
                inner.setAnimated(False)  # match outer window: no snap animations
                # Inner needs a (hidden) central widget so docks anchor sanely.
                center = QWidget(inner)
                center.setMinimumSize(0, 0)
                inner.setCentralWidget(center)
                self._tabs.addTab(inner, page.replace("_", " ").title())
                self._page_windows[page] = inner
                page_anchors[page] = {}

            for panel, area_name, page in normalized:
                inner = self._page_windows[page]
                anchors = page_anchors[page]
                qarea = area_map.get(area_name, Qt.DockWidgetArea.RightDockWidgetArea)
                if area_name in anchors:
                    inner.tabifyDockWidget(anchors[area_name], panel)
                else:
                    inner.addDockWidget(qarea, panel)
                    anchors[area_name] = panel
                self._panels.append(panel)

        # Optional registry of ServerSupervisors so we can stop them
        # gracefully (which releases COM ports etc.) on window close.
        self._supervisors: dict[str, object] = {}
        # Subset of ``self._supervisors`` whose specs declared a COM port.
        # These are shut down first at close time with a blocking modal.
        self._com_ids: set[str] = set()

        # View menu + (no toolbar - tools live in the menu bar now).
        self._build_menu()

        # Restore prior layout if available.
        self._settings = QSettings(settings_org, "dashboard")
        self._restore_layout()

        # Surface any boot warnings (e.g. log dir fallback).
        warnings = pop_boot_warnings()
        if warnings:
            self.statusBar().showMessage("; ".join(warnings)[:300], 10000)

        # Schedule body realization after the window paints once.
        QTimer.singleShot(0, self._realize_all_bodies)

    # ------------------------------------------------------------------
    # Menu / toolbar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        panels_menu = self.menuBar().addMenu("&Panels")
        for panel in self._panels:
            act = panel.toggleViewAction()
            panels_menu.addAction(act)
        panels_menu.addSeparator()
        save_act = QAction("Save Layout to File...", self)
        save_act.triggered.connect(self._save_layout_to_file)
        panels_menu.addAction(save_act)
        load_act = QAction("Load Layout from File...", self)
        load_act.triggered.connect(self._load_layout_from_file)
        panels_menu.addAction(load_act)
        panels_menu.addSeparator()
        reset = QAction("Reset Layout", self)
        reset.triggered.connect(self._reset_layout)
        panels_menu.addAction(reset)

        # Servers menu: lists every registered supervisor with its current
        # state and per-server Start/Stop/Restart actions. Rebuilt every
        # time the menu is opened so it reflects live state.
        self._servers_menu = self.menuBar().addMenu("&Servers")
        self._servers_menu.aboutToShow.connect(self._rebuild_servers_menu)

        tools_menu = self.menuBar().addMenu("&Tools")
        open_logs = QAction("\U0001f4c2 Logs", self)
        open_logs.setToolTip("Open the dashboard log folder")
        open_logs.triggered.connect(self._open_log_folder)
        tools_menu.addAction(open_logs)

        recent_errors = QAction("\U0001f50d Recent errors", self)
        recent_errors.setToolTip("Show recent WARNING/ERROR/CRITICAL log lines")
        recent_errors.triggered.connect(self._show_recent_errors)
        tools_menu.addAction(recent_errors)

        # Also surface the same actions as top-level menubar buttons (no
        # dropdown) so they're one click away.
        logs_top = QAction("\U0001f4c2 Logs", self)
        logs_top.triggered.connect(self._open_log_folder)
        self.menuBar().addAction(logs_top)

        errors_top = QAction("\U0001f50d Errors", self)
        errors_top.triggered.connect(self._show_recent_errors)
        self.menuBar().addAction(errors_top)

        # Snap all windows back to their default positions immediately
        # (no restart required, unlike the destructive "Reset Layout"
        # entry in the View menu).
        snap = QAction("\U0001f9f2 Snap to Default", self)
        snap.setToolTip("Re-apply the default dock placement for every panel")
        snap.triggered.connect(self._snap_default_layout)
        self.menuBar().addAction(snap)

    def _build_toolbar(self) -> None:
        # Tools moved to the menu bar; the toolbar is intentionally a no-op
        # but kept for backwards-compatible API.
        return

    # ------------------------------------------------------------------
    # Servers menu
    # ------------------------------------------------------------------

    def _rebuild_servers_menu(self) -> None:
        """Repopulate the Servers menu with current supervisor states.

        Called on ``aboutToShow`` so the menu is always live without
        needing signal wiring per supervisor.
        """
        menu = self._servers_menu
        menu.clear()
        if not self._supervisors:
            empty = QAction("(no servers registered)", self)
            empty.setEnabled(False)
            menu.addAction(empty)
            return

        # State -> Unicode dot color, for at-a-glance status.
        _DOT = {
            "RUNNING":  "\U0001f7e2",  # green circle
            "STARTING": "\U0001f7e1",  # yellow circle
            "STOPPING": "\U0001f7e1",
            "IDLE":     "\u26AA",      # white circle
            "CRASHED":  "\U0001f534",  # red circle
            "FAILED":   "\U0001f534",
            "EXTERNAL": "\U0001f7e0",  # orange circle
        }
        for sid in sorted(self._supervisors):
            sup = self._supervisors[sid]
            state = getattr(sup, "state", None)
            state_name = getattr(state, "name", None) or getattr(state, "value", None) or "?"
            dot = _DOT.get(str(state_name), "\u26AA")
            sub = menu.addMenu(f"{dot}  {sid}  \u2014  {state_name}")

            start = QAction("Start", self)
            start_fn = getattr(sup, "start", None)
            if callable(start_fn) and str(state_name) not in ("RUNNING", "STARTING"):
                start.triggered.connect(start_fn)
            else:
                start.setEnabled(False)
            sub.addAction(start)

            stop = QAction("Stop", self)
            stop_fn = getattr(sup, "stop", None)
            if callable(stop_fn) and str(state_name) in ("RUNNING", "STARTING"):
                stop.triggered.connect(lambda _checked=False, fn=stop_fn: fn())
            else:
                stop.setEnabled(False)
            sub.addAction(stop)

            restart = QAction("Restart", self)
            restart_fn = getattr(sup, "restart", None)
            if callable(restart_fn):
                restart.triggered.connect(restart_fn)
            else:
                restart.setEnabled(False)
            sub.addAction(restart)

    # ------------------------------------------------------------------
    # Layout persistence
    # ------------------------------------------------------------------

    def _layout_key(self, which: str) -> str:
        # Per-host so different lab PCs can keep their own arrangement.
        return f"dashboard/{self._kind}/{self._host_ip}/{which}"

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
        # Inner page windows have their own dock arrangement.
        page_states: dict[str, object] = {}
        for page, inner in self._page_windows.items():
            if inner is self:
                continue
            try:
                ps = self._settings.value(self._layout_key(f"page_state/{page}"))
                if ps is not None:
                    inner.restoreState(ps)
                    page_states[page] = ps
            except Exception as exc:
                _LOG.warning("Failed to restore page %r layout: %r", page, exc)
        # Capture the just-restored layout as the "last loaded" snapshot so
        # that ``Snap to Default`` brings the user back to what was on screen
        # at startup (overridden later by any ``Load Layout...`` action).
        if geo is not None or state is not None or page_states:
            self._last_loaded_layout = {
                "geometry": geo,
                "state": state,
                "page_states": page_states,
            }

    def _save_layout(self) -> None:
        try:
            self._settings.setValue(self._layout_key("geometry"), self.saveGeometry())
            self._settings.setValue(self._layout_key("state"), self.saveState())
            for page, inner in self._page_windows.items():
                if inner is self:
                    continue
                self._settings.setValue(
                    self._layout_key(f"page_state/{page}"), inner.saveState()
                )
        except Exception as exc:
            _LOG.warning("Failed to save layout: %r", exc)

    def _reset_layout(self) -> None:
        reply = QMessageBox.question(
            self, "Reset layout?",
            "Discard saved layout and restore defaults?\n"
            "(The dashboard will need to restart.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._settings.remove(self._layout_key("geometry"))
        self._settings.remove(self._layout_key("state"))
        QMessageBox.information(self, "Reset layout", "Layout cleared. Restart the dashboard to apply.")

    def _apply_layout_snapshot(self, snap: dict) -> bool:
        """Apply a previously captured layout snapshot.

        Returns ``True`` if the snapshot was applied successfully.
        """
        if not snap:
            return False
        try:
            geo = snap.get("geometry")
            state = snap.get("state")
            if geo is not None:
                self.restoreGeometry(geo)
            if state is not None:
                self.restoreState(state)
            for page, ps in (snap.get("page_states") or {}).items():
                inner = self._page_windows.get(page)
                if inner is not None and inner is not self and ps is not None:
                    inner.restoreState(ps)
            return True
        except Exception as exc:
            _LOG.warning("Failed to apply layout snapshot: %r", exc)
            return False

    def _snap_default_layout(self) -> None:
        """Snap back to the last loaded layout.

        "Last loaded" means either the layout restored from QSettings at
        startup, or the most recent ``Load Layout...`` file the user
        applied.  If neither exists (first-ever run), fall back to the
        original placement spec.
        """
        snap = getattr(self, "_last_loaded_layout", None)
        if snap and self._apply_layout_snapshot(snap):
            self.statusBar().showMessage("Snapped to last loaded layout", 3000)
            return
        # Fall back: re-apply the original placement spec live.
        for panel in self._panels:
            try:
                panel.setFloating(False)
                host = self._page_windows.get("main", self)
                # Find which inner window currently owns this panel.
                for pg, inner in self._page_windows.items():
                    if panel.parent() is inner or panel.parentWidget() is inner:
                        host = inner
                        break
                host.removeDockWidget(panel)
            except Exception:
                pass

        page_anchors: dict[str, dict[str, _PanelDockBase]] = {
            pg: {} for pg in self._page_windows.keys()
        }
        for panel, area_name, page in self._normalized_panels:
            inner = self._page_windows.get(page) or self._page_windows.get("main") or self
            qarea = self._area_map.get(area_name, Qt.DockWidgetArea.RightDockWidgetArea)
            anchors = page_anchors.setdefault(page, {})
            if area_name in anchors:
                inner.tabifyDockWidget(anchors[area_name], panel)
            else:
                inner.addDockWidget(qarea, panel)
                anchors[area_name] = panel
            panel.setVisible(True)

    # ------------------------------------------------------------------
    # Save/load layout to/from a file (portable config dump)
    # ------------------------------------------------------------------

    def _save_layout_to_file(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save layout", f"dashboard_layout_{self._kind}.json",
            "Layout JSON (*.json)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            geom = bytes(self.saveGeometry()).hex()
            state = bytes(self.saveState()).hex()
            payload = {
                "kind": self._kind,
                "host_ip": self._host_ip,
                "geometry_hex": geom,
                "state_hex": state,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.statusBar().showMessage(f"Layout saved to {path}", 5000)
        except Exception as exc:
            _LOG.exception("Save layout failed")
            QMessageBox.warning(self, "Save failed", f"Could not write {path}:\n{exc!r}")

    def _load_layout_from_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load layout", "", "Layout JSON (*.json)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            geom = QByteArray(bytes.fromhex(payload["geometry_hex"]))
            state = QByteArray(bytes.fromhex(payload["state_hex"]))
            self.restoreGeometry(geom)
            self.restoreState(state)
            # Also persist to QSettings so it survives a restart.
            self._settings.setValue(self._layout_key("geometry"), geom)
            self._settings.setValue(self._layout_key("state"), state)
            # Remember this as the new "snap to default" target.
            self._last_loaded_layout = {
                "geometry": geom,
                "state": state,
                "page_states": {},
            }
            self.statusBar().showMessage(f"Layout loaded from {path}", 5000)
        except Exception as exc:
            _LOG.exception("Load layout failed")
            QMessageBox.warning(self, "Load failed", f"Could not load {path}:\n{exc!r}")

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _open_log_folder(self) -> None:
        log_dir = active_log_dir()
        if log_dir is None:
            QMessageBox.warning(self, "No log folder", "Log folder is not configured.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))

    def _show_recent_errors(self) -> None:
        log_dir = active_log_dir()
        if log_dir is None:
            QMessageBox.warning(self, "No log folder", "Log folder is not configured.")
            return
        # Scan the dashboard's own log file for recent WARNING/ERROR/CRITICAL.
        hostname = platform.node()
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
        recent = "\n".join(lines[-80:]) or "(no warnings/errors)"
        msg = QMessageBox(self)
        msg.setWindowTitle(f"Recent errors - {target.name}")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(f"Last warning/error/critical lines in {target.name}:")
        msg.setDetailedText(recent)
        msg.exec()

    # ------------------------------------------------------------------
    # Body realization
    # ------------------------------------------------------------------

    def _realize_all_bodies(self) -> None:
        # Stagger panel realization so the event loop stays alive between
        # factories.  Heavy factories (device_control, camera widgets) that
        # do module imports or file I/O run on the main thread, so spacing
        # them 60 ms apart keeps the window responsive and avoids the
        # appearance of a complete freeze during startup.
        self._realize_pending = len(self._panels)
        self._realize_done = 0
        for i, panel in enumerate(self._panels):
            QTimer.singleShot(i * 60, lambda p=panel: self._realize_one_body(p))

    def _realize_one_body(self, panel) -> None:
        try:
            panel.realize_body()
        except Exception:
            _LOG.exception("Unexpected error realizing panel %s", panel.panel_id)
        self._realize_done = getattr(self, '_realize_done', 0) + 1
        pending = getattr(self, '_realize_pending', 0)
        if self._realize_done >= pending:
            # Re-apply the layout captured at startup.  The initial
            # restoreState() in __init__ runs before panel bodies exist, so
            # realizing the bodies resizes/re-docks panels and clobbers the
            # restored arrangement.  Now that every body is realized, apply
            # the snapshot once more so the saved layout actually sticks.
            snap = getattr(self, "_last_loaded_layout", None)
            if snap:
                self._apply_layout_snapshot(snap)
            self.statusBar().showMessage(
                f"Ready - {self._realize_done} panel(s) loaded", 5000)

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, ev):  # noqa: N802 - Qt-style
        self._save_layout()

        # Phase 0: COM-priority shutdown.  Any supervisor whose spec
        # declared a COM port is shut down FIRST and the dashboard close
        # is blocked behind a modal dialog until each one has either
        # exited cleanly (so the OS releases the serial port) or the user
        # explicitly force-kills / cancels.
        com_sup_entries: list[tuple[str, str, object]] = []
        for sid in sorted(self._com_ids):
            sup = self._supervisors.get(sid)
            if sup is None:
                continue
            # Best-effort label + COM port; fall back to the supervisor id.
            label = getattr(sup, "server_id", sid)
            com_port = ""
            spec = getattr(sup, "spec", None)
            if spec is not None:
                label = getattr(spec, "label", label)
                com_port = getattr(spec, "com_label", "") or ""
            is_running = getattr(sup, "is_running", None) or getattr(sup, "is_alive", None)
            try:
                alive = bool(is_running()) if callable(is_running) else False
            except Exception:
                alive = False
            if not alive:
                continue
            com_sup_entries.append((label, com_port, sup))

        if com_sup_entries:
            # Stop the crash-restart loop fighting our clean shutdown.
            for _label, _com, sup in com_sup_entries:
                suppress = getattr(sup, "suppress_restart", None)
                if callable(suppress):
                    try:
                        suppress()
                    except Exception:
                        pass
                req = getattr(sup, "request_terminate", None)
                if callable(req):
                    try:
                        req()
                    except Exception:
                        _LOG.exception(
                            "request_terminate() failed during COM shutdown for id=%s",
                            getattr(sup, "server_id", "?"),
                        )

            try:
                from waxx.util.dashboard.com_shutdown_dialog import (  # noqa: PLC0415
                    ComShutdownDialog,
                    RESULT_USER_CANCELLED,
                )
                dlg = ComShutdownDialog(com_sup_entries, parent=self)
                dlg.exec()
                if dlg.result_reason() == RESULT_USER_CANCELLED:
                    # User aborted close — keep dashboard open.  Servers
                    # that already received CTRL_BREAK will have stopped;
                    # the user can restart them from each panel header.
                    ev.ignore()
                    return
            except Exception:
                _LOG.exception("ComShutdownDialog failed; falling back to fast shutdown")

        # Stop registered server subprocesses in parallel so the GUI
        # thread doesn't block for ``N * graceful_stop_timeout`` seconds
        # while each headless server times out one-by-one.
        #
        # Strategy:
        #   1. Issue ``terminate()`` to *every* supervisor up front.
        #   2. Wait up to ~500 ms total for any cooperative exits.
        #   3. Hard-kill anything still alive (fast).
        #
        # ``QProcess.terminate()`` on Windows is a no-op for headless
        # Python console apps (no top-level windows to receive
        # ``WM_CLOSE``), so step 1 effectively buys ~0 cooperative exits
        # today.  We still do it in case a future server installs a real
        # SIGTERM handler.
        sups = list(self._supervisors.items())

        # Step 1: fan out terminate() requests.
        for sid, sup in sups:
            req = getattr(sup, "request_terminate", None)
            if callable(req):
                try:
                    req()
                except Exception:
                    _LOG.exception("request_terminate() failed for id=%s", sid)
                continue
            # Fallback for older supervisors without the fast path.
            stop = getattr(sup, "stop", None)
            if callable(stop):
                try:
                    stop()  # non-blocking
                except Exception:
                    _LOG.exception("supervisor stop() failed for id=%s", sid)

        # Step 2: short collective grace window.
        from PyQt6.QtCore import QDeadlineTimer  # noqa: PLC0415

        deadline = QDeadlineTimer(500)  # ms total across all supervisors
        for sid, sup in sups:
            wait = getattr(sup, "wait_for_finished", None)
            if not callable(wait):
                continue
            remaining = deadline.remainingTime()
            if remaining <= 0:
                break
            try:
                wait(remaining)
            except Exception:
                _LOG.exception("wait_for_finished() failed for id=%s", sid)

        # Step 3: hard-kill survivors in parallel.
        for sid, sup in sups:
            killer = getattr(sup, "force_kill", None)
            if callable(killer):
                try:
                    killer(wait_ms=200)
                except Exception:
                    _LOG.exception("force_kill() failed for id=%s", sid)

        # Give panels a chance to clean up.
        for panel in self._panels:
            body = panel.body_widget()
            cleanup = getattr(body, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    _LOG.exception("cleanup() failed for panel %s", panel.panel_id)
        super().closeEvent(ev)

    def register_supervisors(
        self,
        supervisors: dict[str, object],
        com_ids: "set[str] | None" = None,
    ) -> None:
        """Register ``ServerSupervisor`` instances so they are stopped
        gracefully when the dashboard window is closed.

        Pass a mapping of ``server_id -> supervisor``.  Any object with a
        ``.stop()`` method is accepted.  ``com_ids`` lists the subset whose
        specs declared a ``com_label``; those supervisors are shut down
        first (with a blocking modal dialog) on window close so the serial
        ports release cleanly.
        """
        self._supervisors.update(supervisors)
        if com_ids:
            self._com_ids.update(com_ids)


__all__ = ["DashboardMainWindow"]
