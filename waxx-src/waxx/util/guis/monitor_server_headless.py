"""Headless monitor server.

Runs the same UDP responder + MonitorManager as :mod:`monitor_server_gui`
but with no visible window — the dashboard already shows the
ready / not-ready state through the device control GUI, so a second Qt
window for this is just clutter.

A QApplication is still required because the UDP server and the
MonitorManager rely on Qt signals/threads internally; we just never
create or show any widgets.
"""
from __future__ import annotations

import sys
import logging

from PyQt6.QtCore import QObject, QTimer, QCoreApplication

from waxx.util.device_state.monitor_manager import MonitorManager
from waxx.util.guis.monitor_server_gui import MonitorUDPServer, Status
from waxx.util.comms_server.comm_server import STATES
from PyQt6.QtCore import QThread


log = logging.getLogger(__name__)


class HeadlessMonitorServer(QObject):
    def __init__(self, monitor_expt_path: str, config_file_path: str | None = None):
        super().__init__()
        self.config_file_path = config_file_path
        self.monitor_manager = MonitorManager(monitor_expt_path)
        self.monitor_manager.msg.connect(lambda m: log.info("monitor: %s", m))
        self.status = Status()

        self._setup_udp_server()
        self._set_status(STATES.NOT_READY)

        self._timer = QTimer(self)
        self._timer.setInterval(125)
        self._timer.timeout.connect(self._check_status)
        self._timer.start()

        # Do NOT auto-start the monitor experiment on launch.  Starting
        # automatically would interrupt any experiment already running on
        # the hardware when the dashboard is (re)started.  The monitor
        # can be started manually from the Device Control panel, or it
        # will be triggered automatically when the previous experiment
        # finishes (via the UDP reset signal).

    def _setup_udp_server(self) -> None:
        self.server_thread = QThread()
        self.udp_server = MonitorUDPServer(config_file_path=self.config_file_path)
        self.udp_server.moveToThread(self.server_thread)
        self.udp_server.reset_signal.connect(self._restart_monitor)
        self.udp_server.message_received.connect(self._handle_message)
        self.server_thread.started.connect(self.udp_server.run)
        self.server_thread.start()

    def _restart_monitor(self) -> None:
        if self.monitor_manager.isRunning():
            self.monitor_manager.terminate()
            self.monitor_manager.wait(500)
        self.monitor_manager.start()
        self._set_status(STATES.LOADING)

    def _set_status(self, status) -> None:
        self.status.state = status
        self.udp_server.status.state = status

    def _check_status(self) -> None:
        if self.monitor_manager.isRunning() and self.status.state != STATES.READY:
            self._set_status(STATES.LOADING)
        elif not self.monitor_manager.isRunning():
            self._set_status(STATES.NOT_READY)
        else:
            self._set_status(STATES.READY)

    def _handle_message(self, message: str) -> None:
        log.info("msg: %s", message)
        if "run complete" in message:
            self._restart_monitor()
        elif "monitor ready" in message:
            self._set_status(STATES.READY)

    def shutdown(self) -> None:
        try:
            self.udp_server.stop()
        except Exception:
            pass
        self.server_thread.quit()
        self.server_thread.wait()
        try:
            self.monitor_manager.terminate()
            self.monitor_manager.wait()
        except Exception:
            pass


def run(monitor_expt_path: str, config_file_path: str | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    server = HeadlessMonitorServer(monitor_expt_path, config_file_path=config_file_path)

    def _on_quit():
        server.shutdown()

    app.aboutToQuit.connect(_on_quit)
    return app.exec()
