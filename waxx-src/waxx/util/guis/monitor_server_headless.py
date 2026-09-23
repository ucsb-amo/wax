"""Headless monitor server.

Runs the same UDP responder + MonitorManager as :mod:`monitor_server_gui`
but with no visible window — the dashboard already shows the
ready / not-ready state through the device control GUI, so a second Qt
window for this is just clutter.

A QApplication is still required because the UDP server and the
MonitorManager rely on Qt signals/threads internally; we just never
create or show any widgets.

This process is a child of the server dashboard, which tags and shows every
line it writes to stdout/stderr.  Everything below therefore goes through
``logging`` (stderr, line buffered) so the dashboard terminal is the primary
place monitor problems are reported.
"""
from __future__ import annotations

import sys
import logging

from PyQt6.QtCore import QObject, QTimer, QCoreApplication

from waxx.util.device_state.monitor_manager import MonitorManager
from waxx.util.guis.monitor_server_gui import MonitorUDPServer, Status
from waxx.util.comms_server.comm_server import STATES
from waxx.util.comms_server.hardware_id import monitor_server_id
from beacon.discovery.client import discover
from PyQt6.QtCore import QThread


log = logging.getLogger(__name__)


_STATE_NAMES = {STATES.READY: "READY", STATES.LOADING: "LOADING",
                STATES.NOT_READY: "NOT_READY"}


def _state_name(state) -> str:
    return _STATE_NAMES.get(state, str(state))


class HeadlessMonitorServer(QObject):
    def __init__(self, monitor_expt_path: str, config_file_path: str | None = None):
        super().__init__()
        self.config_file_path = config_file_path
        self.monitor_expt_path = monitor_expt_path
        self.monitor_manager = MonitorManager(monitor_expt_path)
        self.monitor_manager.msg.connect(lambda m: log.info("monitor: %s", m))
        self.monitor_manager.monitor_stopped.connect(self._on_monitor_stopped)
        self.status = Status()

        log.info("monitor experiment: %s", monitor_expt_path)
        log.info("device state file: %s", config_file_path)
        if config_file_path is None:
            log.error(
                "No device-state config path was passed: state reads/writes from "
                "the Device Control GUI will all fail with 'no config path'. The "
                "launcher could not resolve it (usually an unset env var or an "
                "unmapped drive)."
            )
        # Report an unusable experiment path now, at dashboard startup, rather
        # than only when someone first tries to start the monitor.
        problems = self.monitor_manager.preflight_problems()
        for problem in problems:
            log.error("Monitor cannot be started as configured: %s", problem)

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

    def _on_monitor_stopped(self, reason: str) -> None:
        log.warning("Monitor is not running: %s", reason)

    def _restart_monitor(self) -> None:
        if getattr(self, "_restarting", False):
            log.info("Monitor restart ignored: a restart is already in progress.")
            return
        self._restarting = True
        try:
            if self.monitor_manager.isRunning():
                log.info("Restarting monitor experiment...")
                self.monitor_manager.stop()
            else:
                log.info("Starting monitor experiment...")
            self.monitor_manager.start()
            if self.monitor_manager.isRunning():
                self._set_status(STATES.LOADING)
            else:
                # start() refused (pre-flight failure); it has already said why.
                self._set_status(STATES.NOT_READY)
        finally:
            self._restarting = False

    def _set_status(self, status) -> None:
        # Logged only on change: _check_status runs at 8 Hz.
        previous = getattr(self, "_logged_state", None)
        if status != previous:
            if previous is None:
                log.info("state: %s", _state_name(status))
            else:
                log.info("state: %s -> %s", _state_name(previous), _state_name(status))
            self._logged_state = status
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
            log.exception("Error while stopping the monitor UDP server")
        self.server_thread.quit()
        self.server_thread.wait()
        try:
            self.monitor_manager.stop()
        except Exception:
            log.exception("Error while stopping the monitor experiment")


def run(monitor_expt_path: str, config_file_path: str | None = None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")

    # Refuse to start a second monitor server for the same hardware: if a beacon
    # for our hardware-scoped id is already on the subnet, two servers would
    # compete for signals / ports / broadcasts.
    server_id = monitor_server_id()
    existing = discover(server_id, timeout=1.5)
    if existing is not None:
        ip, port = existing
        log.error(
            "A monitor server for '%s' is already running at %s:%d. "
            "Refusing to start a second server for the same hardware.",
            server_id, ip, port,
        )
        return 1

    log.info("Starting monitor server '%s'.", server_id)
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    try:
        server = HeadlessMonitorServer(monitor_expt_path, config_file_path=config_file_path)
    except Exception:
        log.exception("Monitor server failed to start")
        return 1

    def _on_quit():
        server.shutdown()

    app.aboutToQuit.connect(_on_quit)
    return app.exec()
