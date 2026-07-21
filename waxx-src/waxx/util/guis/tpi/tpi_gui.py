"""TpiDevicesMainWindow — dock-based GUI for all discovered TPI-1005-A devices.

Each device gets a QDockWidget with a TpiDeviceWidget inside.  Discovery runs
in a background QThread so the window appears immediately.

Usage::

    python -m waxx.util.guis.tpi.tpi_gui
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from waxx.util.guis.tpi.tpi_client import TpiDeviceClient, discover_all_tpi_devices
from waxx.util.guis.tpi.tpi_device_widget import DARK_STYLE, TpiDeviceWidget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background discovery worker
# ---------------------------------------------------------------------------

class _DiscoveryWorker(QThread):
    devices_found = pyqtSignal(list)  # list[TpiDeviceClient]

    def __init__(self, collect_for: float = 3.0, quiet: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._collect_for = collect_for
        self._quiet = quiet

    def run(self) -> None:
        clients = discover_all_tpi_devices(collect_for=self._collect_for, quiet=self._quiet)
        self.devices_found.emit(clients)


# ---------------------------------------------------------------------------
# Dock with close signal
# ---------------------------------------------------------------------------

class _DeviceDock(QDockWidget):
    closed = pyqtSignal(str)  # serial

    def __init__(self, serial: str, title: str, parent=None) -> None:
        super().__init__(title, parent)
        self._serial = serial

    def closeEvent(self, event) -> None:
        self.closed.emit(self._serial)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class TpiDevicesMainWindow(QMainWindow):
    """QMainWindow with one dock per discovered TPI-1005-A device."""

    def __init__(self, collect_for: float = 3.0, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._collect_for = collect_for
        self._widgets: dict[str, TpiDeviceWidget] = {}
        self._docks: dict[str, _DeviceDock] = {}
        self._seen_serials: set[str] = set()
        self._first_dock: Optional[_DeviceDock] = None

        self.setWindowTitle("RF Consultants")
        self.setMinimumSize(400, 300)
        self.resize(700, 400)
        self.setStyleSheet(DARK_STYLE)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )

        # Toolbar
        tb = self.addToolBar("Controls")
        tb.setMovable(False)
        tb.addAction("⟳  Refresh", self._on_refresh)

        # Placeholder shown when no devices are open
        placeholder = QWidget()
        lay = QVBoxLayout(placeholder)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel("Scanning for TPI servers…")
        lbl.setStyleSheet("color: #60638a; font-size: 15px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        btn = QPushButton("⟳  Refresh")
        btn.setFixedSize(130, 36)
        btn.clicked.connect(self._on_refresh)
        lay.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self._placeholder = placeholder
        self._placeholder_label = lbl
        self.setCentralWidget(placeholder)

        self.statusBar().showMessage("Scanning for TPI servers…")
        self._start_discovery(self._collect_for, quiet=False)

        # Periodic background rescan
        self._rescan_timer = QTimer(self)
        self._rescan_timer.setInterval(15_000)
        self._rescan_timer.timeout.connect(self._on_periodic_rescan)
        self._rescan_timer.start()

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _start_discovery(self, collect_for: float, quiet: bool = False) -> None:
        self._worker = _DiscoveryWorker(collect_for, quiet, self)
        self._worker.devices_found.connect(self._on_devices_found)
        self._worker.start()

    def _on_devices_found(self, clients: list[TpiDeviceClient]) -> None:
        new = [c for c in clients if c.serial not in self._seen_serials]
        for client in new:
            self._seen_serials.add(client.serial)
            self._add_dock(client)

        if not self._widgets:
            self._placeholder_label.setText("No TPI devices found.\nConnect a device and click Refresh.")
            self.statusBar().showMessage("No TPI devices found.")
        else:
            n = len(self._widgets)
            self.statusBar().showMessage(f"{n} device{'s' if n != 1 else ''} connected.")

    def _on_refresh(self) -> None:
        if self._worker.isRunning():
            return
        self.statusBar().showMessage("Scanning for TPI servers…")
        self._start_discovery(3.0, quiet=False)

    def _on_periodic_rescan(self) -> None:
        if self._worker.isRunning():
            return
        self._start_discovery(1.5, quiet=True)

    # ------------------------------------------------------------------ #
    # Dock management
    # ------------------------------------------------------------------ #

    def _add_dock(self, client: TpiDeviceClient) -> None:
        if client.serial in self._widgets:
            return

        widget = TpiDeviceWidget(client)
        key_prefix = f"{client.key}  ·  " if client.key else ""
        title = f"{key_prefix}{client.model}  ·  S/N {client.serial}  ·  @{client.hostname}"
        dock = _DeviceDock(client.serial, title, self)
        dock.setObjectName(f"tpi_dock_{client.serial}")
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.closed.connect(self._on_dock_closed)

        if self._first_dock is None:
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            self._first_dock = dock
        else:
            self.splitDockWidget(self._first_dock, dock, Qt.Orientation.Horizontal)

        self._widgets[client.serial] = widget
        self._docks[client.serial] = dock
        self._placeholder.hide()

    def _on_dock_closed(self, serial: str) -> None:
        widget = self._widgets.pop(serial, None)
        if widget:
            widget.cleanup()
        self._docks.pop(serial, None)
        if serial == getattr(self._first_dock, '_serial', None):
            self._first_dock = next(iter(self._docks.values()), None)
        if not self._widgets:
            self._placeholder.show()

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:
        self._rescan_timer.stop()
        for widget in self._widgets.values():
            widget.cleanup()
        if self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(3500)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TpiDevicesMainWindow()
    win.show()
    sys.exit(app.exec())
