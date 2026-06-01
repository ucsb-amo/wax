"""BaslerCamerasMainWindow - dock-based GUI for discovered Basler cameras.

Each camera gets a ``QDockWidget`` whose title-bar carries the full camera
identification (user_id, model, S/N, host) plus a built-in close (X)
button.  Docks are movable, floatable, and can be tabified by dragging
them on top of each other or split side-by-side.

The window starts with a central "Add Camera" placeholder.  Discovery
runs in the background so the Add Camera dialog is populated as soon as
servers respond.  Double-clicking a row in the Add Camera dialog
immediately accepts that row, so adding a single camera is a one-click
gesture.

Usage::

    win = BaslerCamerasMainWindow()                          # all cameras
    win = BaslerCamerasMainWindow(serial_filter=["40277706"], auto_open=True)
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from waxx.util.guis.basler.basler_camera_client import (
    BaslerCameraClient,
    discover_all_basler_cameras,
)
from waxx.util.guis.basler.basler_camera_widget import DARK_STYLESHEET, BaslerCameraWidget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dock widget with close signal
# ---------------------------------------------------------------------------

class _CameraDock(QDockWidget):
    """QDockWidget that emits ``closed(serial)`` when its X is clicked."""

    closed = pyqtSignal(str)

    def __init__(self, serial: str, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self._serial = serial

    def closeEvent(self, event) -> None:
        self.closed.emit(self._serial)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Background discovery worker
# ---------------------------------------------------------------------------

class _DiscoveryWorker(QThread):
    cameras_found = pyqtSignal(list)  # list[BaslerCameraClient]

    def __init__(
        self,
        serial_filter: Optional[list[str]],
        collect_for: float,
        parent: Optional[QObject] = None,
        quiet: bool = False,
    ) -> None:
        super().__init__(parent)
        self.serial_filter = serial_filter
        self.collect_for = collect_for
        self.quiet = quiet

    def run(self) -> None:
        clients = discover_all_basler_cameras(
            collect_for=self.collect_for, quiet=self.quiet,
        )
        if self.serial_filter:
            clients = [c for c in clients if c.serial in self.serial_filter]
        self.cameras_found.emit(clients)


# ---------------------------------------------------------------------------
# "Add Camera" dialog
# ---------------------------------------------------------------------------

class _AddCameraDialog(QDialog):
    """Multi-select list of cameras not currently shown.

    Double-clicking a row accepts the dialog with that single row
    selected, which is the fast path for adding one camera at a time.
    """

    def __init__(
        self,
        available: list[BaslerCameraClient],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Camera")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        if not available:
            layout.addWidget(QLabel("No additional cameras found on the network."))
        else:
            layout.addWidget(QLabel("Select cameras to add (double-click a row to add immediately):"))
            self._list = QListWidget()
            self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            for client in available:
                sub = f"{client.model}  \u00b7  S/N {client.serial}  \u00b7  @{client.hostname}"
                item = QListWidgetItem(f"{client.display_name}\n{sub}")
                item.setData(Qt.ItemDataRole.UserRole, client)
                self._list.addItem(item)
            self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
            layout.addWidget(self._list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        # Make sure the double-clicked row is part of the current selection,
        # then accept.  This turns double-click into a one-shot "add this
        # camera" gesture even when the user hasn't ticked it first.
        if not item.isSelected():
            item.setSelected(True)
        self.accept()

    def selected_clients(self) -> list[BaslerCameraClient]:
        if not hasattr(self, "_list"):
            return []
        return [item.data(Qt.ItemDataRole.UserRole) for item in self._list.selectedItems()]


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BaslerCamerasMainWindow(QMainWindow):
    """QMainWindow with one ``_CameraDock`` per camera.

    Args:
        serial_filter: Only show cameras whose serials are in this list.
            ``None`` shows all cameras.
        auto_open: Open each camera stream automatically once discovered.
        collect_for: Seconds to collect UDP beacons before querying cameras.
    """

    def __init__(
        self,
        serial_filter: Optional[list[str]] = None,
        auto_open: bool = False,
        collect_for: float = 3.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.serial_filter = serial_filter
        self.auto_open = auto_open
        self._camera_widgets: dict[str, BaslerCameraWidget] = {}
        self._docks: dict[str, _CameraDock] = {}
        self._all_clients: list[BaslerCameraClient] = []
        self._first_dock: Optional[_CameraDock] = None

        self.setWindowTitle("Basler Cameras")
        self.setMinimumSize(700, 500)
        self.resize(1300, 780)
        self.setStyleSheet(DARK_STYLESHEET)
        # Allow tabified docks, side-by-side splits, and grouped dragging
        # so users can freely rearrange cameras.
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.GroupedDragging
        )
        self.setAnimated(False)  # snappier dock drag/drop

        # ---- Toolbar ---------------------------------------------------
        tb = QToolBar("Controls", self)
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        self._refresh_action = tb.addAction("\u27f3  Refresh")
        self._refresh_action.triggered.connect(self._on_refresh)
        tb.addSeparator()
        self._add_cam_action = tb.addAction("\uff0b  Add Camera")
        self._add_cam_action.triggered.connect(self._on_add_camera)

        # ---- Central placeholder (shown when no docks are open) --------
        self._placeholder = QWidget()
        ph_layout = QVBoxLayout(self._placeholder)
        ph_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ph_lbl = QLabel("No cameras added yet.")
        ph_lbl.setStyleSheet("color: #60638a; font-size: 16px;")
        ph_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_layout.addWidget(ph_lbl)

        ph_btn = QPushButton("\uff0b  Add Camera")
        ph_btn.setFixedSize(160, 40)
        ph_btn.clicked.connect(self._on_add_camera)
        ph_layout.addWidget(ph_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(self._placeholder)

        self.statusBar().showMessage("Scanning for Basler servers\u2026")

        # ---- Start background discovery --------------------------------
        self._worker = _DiscoveryWorker(serial_filter, collect_for, self)
        self._worker.cameras_found.connect(self._on_cameras_found)
        self._worker.start()

        # ---- Periodic re-scan so newly-spawned camera servers appear
        #      without the user needing to click "Refresh".  The worker
        #      checks ``isRunning()`` itself, so we just kick it.
        self._rescan_timer = QTimer(self)
        self._rescan_timer.setInterval(15_000)  # 15 s
        self._rescan_timer.timeout.connect(self._on_periodic_rescan)
        self._rescan_timer.start()

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _on_cameras_found(self, clients: list[BaslerCameraClient]) -> None:
        existing = {c.serial for c in self._all_clients}
        new_clients: list[BaslerCameraClient] = []
        for c in clients:
            if c.serial not in existing:
                self._all_clients.append(c)
                existing.add(c.serial)
                new_clients.append(c)

        n_total = len(self._all_clients)
        if new_clients:
            n_new = len(new_clients)
            self.statusBar().showMessage(
                f"Discovered {n_new} new camera{'s' if n_new != 1 else ''} "
                f"({n_total} total).  Use \u201c\uff0b Add Camera\u201d to display."
            )
        elif n_total == 0:
            self.statusBar().showMessage("No cameras found on network.")

        if self.auto_open:
            for client in new_clients:
                self._add_camera_tile(client)

    # ------------------------------------------------------------------ #
    # Tile management
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dock_title(client: BaslerCameraClient) -> str:
        name = client.user_id if client.user_id else client.display_name
        return (
            f"{name}  \u00b7  {client.model}"
            f"  \u00b7  S/N {client.serial}"
            f"  \u00b7  @{client.hostname}"
        )

    def _add_camera_tile(self, client: BaslerCameraClient) -> None:
        if client.serial in self._camera_widgets:
            return

        widget = BaslerCameraWidget(client)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        widget.setMinimumSize(420, 360)

        dock = _CameraDock(client.serial, self._dock_title(client), self)
        dock.setObjectName(f"basler_dock_{client.serial}")
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        dock.closed.connect(self._on_dock_closed)

        # Hide the placeholder on the first camera.
        if not self._camera_widgets:
            self._placeholder.hide()

        # Tile: expand horizontally up to 3, then tabify the rest.
        if self._first_dock is None:
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            self._first_dock = dock
        elif len(self._camera_widgets) < 3:
            last = list(self._docks.values())[-1]
            self.splitDockWidget(last, dock, Qt.Orientation.Horizontal)
        else:
            self.tabifyDockWidget(self._first_dock, dock)

        self._camera_widgets[client.serial] = widget
        self._docks[client.serial] = dock

        if self.auto_open:
            widget.open_btn.setChecked(True)

        self._update_status()

    def _on_dock_closed(self, serial: str) -> None:
        widget = self._camera_widgets.pop(serial, None)
        if widget:
            try:
                widget._do_close()
            except Exception:
                pass

        closed_dock = self._docks.pop(serial, None)
        if closed_dock is self._first_dock:
            self._first_dock = next(iter(self._docks.values()), None)

        if not self._camera_widgets:
            self._placeholder.show()

        self._update_status()

    def _update_status(self) -> None:
        n = len(self._camera_widgets)
        if n:
            self.statusBar().showMessage(f"{n} camera{'s' if n != 1 else ''} displayed.")
        else:
            self.statusBar().showMessage("No cameras displayed.")

    # ------------------------------------------------------------------ #
    # Toolbar actions
    # ------------------------------------------------------------------ #

    def _on_refresh(self) -> None:
        if self._worker.isRunning():
            return
        self.statusBar().showMessage("Re-scanning for Basler servers\u2026")
        self._worker = _DiscoveryWorker(self.serial_filter, 3.0, self)
        self._worker.cameras_found.connect(self._on_cameras_found)
        self._worker.start()

    def _on_periodic_rescan(self) -> None:
        """Quietly poll for newly-spawned camera servers in the background.

        Does not change the status-bar message unless new cameras are found
        (that update happens inside :meth:`_on_cameras_found`).  If the
        previous worker is still running, this call is a no-op.
        """
        if self._worker.isRunning():
            return
        # Use a short collect window for periodic polls so we don't waste
        # time blocking on UDP receives when nothing new is on the network.
        # ``quiet=True`` keeps unchanged-state results out of the log; only
        # genuine topology changes still surface at INFO.
        self._worker = _DiscoveryWorker(
            self.serial_filter, 1.5, self, quiet=True,
        )
        self._worker.cameras_found.connect(self._on_cameras_found)
        self._worker.start()

    def _on_add_camera(self) -> None:
        not_shown = [c for c in self._all_clients if c.serial not in self._camera_widgets]
        dlg = _AddCameraDialog(not_shown, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            for client in dlg.selected_clients():
                self._add_camera_tile(client)

    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:
        try:
            self._rescan_timer.stop()
        except Exception:
            pass
        for widget in self._camera_widgets.values():
            try:
                widget._do_close()
            except Exception:
                pass
        # Ask the discovery worker to quit and wait briefly for it to
        # actually exit.  Without this Qt destroys the QThread Python
        # wrapper while the underlying OS thread is still blocked inside
        # a UDP recv, which prints "QThread: Destroyed while thread is
        # still running" on stderr.  The worker's UDP collect window is
        # at most ~3 s, so a 3.5 s wait reliably catches it.
        if self._worker.isRunning():
            try:
                self._worker.quit()
                self._worker.wait(3500)
            except Exception:
                pass
        super().closeEvent(event)

    def cleanup(self) -> None:  # called by the dashboard close path
        """Tear down threads/timers so the embedded GUI can be destroyed.

        The dashboard does not deliver Qt closeEvents to embedded GUIs, so
        the same shutdown work the standalone ``closeEvent`` would do is
        exposed here for the panel wrapper to call.
        """
        self.closeEvent(None)
