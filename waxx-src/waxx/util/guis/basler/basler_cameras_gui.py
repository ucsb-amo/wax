"""BaslerCamerasMainWindow — dock-based GUI for discovered Basler cameras.

Each camera gets a ``QDockWidget`` with a built-in title-bar close (✕) button.
Docks are movable, floatable, and can be tabified by dragging.

The window starts empty with a central "＋ Add Camera" placeholder.  Discovery
runs in the background so the Add Camera dialog is populated immediately.

Usage::

    win = BaslerCamerasMainWindow()                          # all cameras
    win = BaslerCamerasMainWindow(serial_filter=["40277706"], auto_open=True)
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
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
    """QDockWidget that emits ``closed(serial)`` when its ✕ is clicked."""

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
    ) -> None:
        super().__init__(parent)
        self.serial_filter = serial_filter
        self.collect_for = collect_for

    def run(self) -> None:
        clients = discover_all_basler_cameras(collect_for=self.collect_for)
        if self.serial_filter:
            clients = [c for c in clients if c.serial in self.serial_filter]
        self.cameras_found.emit(clients)


# ---------------------------------------------------------------------------
# "Add Camera" dialog
# ---------------------------------------------------------------------------

class _AddCameraDialog(QDialog):
    """Multi-select list of cameras not currently shown."""

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
            layout.addWidget(QLabel("Select cameras to add:"))
            self._list = QListWidget()
            self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            for client in available:
                sub = f"{client.model}  ·  S/N {client.serial}  ·  @{client.hostname}"
                item = QListWidgetItem(f"{client.display_name}\n{sub}")
                item.setData(Qt.ItemDataRole.UserRole, client)
                self._list.addItem(item)
            layout.addWidget(self._list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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
        self.setDockOptions(
            QMainWindow.DockOption.AllowTabbedDocks |
            QMainWindow.DockOption.AnimatedDocks
        )
        self.setAnimated(False)  # skip animation during dock drag for better responsiveness

        # ---- Toolbar ---------------------------------------------------
        tb = QToolBar("Controls", self)
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        self._refresh_action = tb.addAction("⟳  Refresh")
        self._refresh_action.triggered.connect(self._on_refresh)
        tb.addSeparator()
        self._add_cam_action = tb.addAction("＋  Add Camera")
        self._add_cam_action.triggered.connect(self._on_add_camera)

        # ---- Central placeholder (shown when no docks are open) --------
        self._placeholder = QWidget()
        ph_layout = QVBoxLayout(self._placeholder)
        ph_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ph_lbl = QLabel("No cameras added yet.")
        ph_lbl.setStyleSheet("color: #60638a; font-size: 16px;")
        ph_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_layout.addWidget(ph_lbl)

        ph_btn = QPushButton("＋  Add Camera")
        ph_btn.setFixedSize(160, 40)
        ph_btn.clicked.connect(self._on_add_camera)
        ph_layout.addWidget(ph_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(self._placeholder)

        self.statusBar().showMessage("Scanning for Basler servers…")

        # ---- Start background discovery --------------------------------
        self._worker = _DiscoveryWorker(serial_filter, collect_for, self)
        self._worker.cameras_found.connect(self._on_cameras_found)
        self._worker.start()

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _on_cameras_found(self, clients: list[BaslerCameraClient]) -> None:
        existing = {c.serial for c in self._all_clients}
        for c in clients:
            if c.serial not in existing:
                self._all_clients.append(c)
                existing.add(c.serial)

        n = len(clients)
        if n:
            self.statusBar().showMessage(
                f"Found {n} camera{'s' if n != 1 else ''} on network. "
                "Use \u201c\uff0b Add Camera\u201d to display."
            )
        else:
            self.statusBar().showMessage("No cameras found on network.")

        if self.auto_open:
            for client in clients:
                self._add_camera_tile(client)

    # ------------------------------------------------------------------ #
    # Tile management
    # ------------------------------------------------------------------ #

    def _add_camera_tile(self, client: BaslerCameraClient) -> None:
        if client.serial in self._camera_widgets:
            return

        widget = BaslerCameraWidget(client)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        widget.setMinimumSize(420, 360)

        dock = _CameraDock(client.serial, client.display_name, self)
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        dock.closed.connect(self._on_dock_closed)

        # Hide the placeholder on the first camera.
        if not self._camera_widgets:
            self._placeholder.hide()

        # Tile: expand horizontally up to 3, then tabify.
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
        self.statusBar().showMessage("Re-scanning for Basler servers…")
        self._worker = _DiscoveryWorker(self.serial_filter, 3.0, self)
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
        for widget in self._camera_widgets.values():
            try:
                widget._do_close()
            except Exception:
                pass
        if self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        super().closeEvent(event)
