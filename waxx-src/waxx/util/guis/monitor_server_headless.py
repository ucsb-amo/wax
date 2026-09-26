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
from waxx.util.guis.monitor_server_gui import MonitorUDPServer, MonitorStatus, _state_name
from waxx.util.comms_server.comm_server import STATES
from waxx.util.comms_server.hardware_id import monitor_server_id
from beacon.discovery.client import discover
from PyQt6.QtCore import QThread


log = logging.getLogger(__name__)

# Re-exported for callers that imported the old name from here.
Status = MonitorStatus


class HeadlessMonitorServer(QObject):
    """Owns the MonitorManager and the TCP responder; drives the status.

    State machine as served by ``status`` / ``status_json``:

    * LOADING ``starting``  -- the experiment process is up, ``monitor ready``
      not yet received.
    * READY ``running``     -- the monitor loop is applying changes.
    * NOT_READY             -- ``never_started`` before the first start,
      ``stopped_on_request`` after a ``stop`` command, otherwise the manager's
      classification of the exit (``interrupted_by_run``, ``exited``,
      ``failed``, ``preflight_failed``).
    """

    def __init__(self, monitor_expt_path: str, config_file_path: str | None = None,
                 journal_dir: str | None = None):
        super().__init__()
        self.config_file_path = config_file_path
        self.journal_dir = journal_dir
        self.monitor_expt_path = monitor_expt_path
        self.monitor_manager = MonitorManager(monitor_expt_path)
        self.monitor_manager.msg.connect(lambda m: log.info("monitor: %s", m))
        self.monitor_manager.monitor_stopped.connect(self._on_monitor_stopped)

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
        # One status object, shared with the TCP responder so status_json
        # always serves exactly what this process believes.
        self.status = self.udp_server.status
        self.status.expt_path = str(monitor_expt_path)
        self._set_status(STATES.NOT_READY, "never_started", "")

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
        self.udp_server = MonitorUDPServer(config_file_path=self.config_file_path,
                                           journal_dir=self.journal_dir)
        log.info("ops journal: %s", self.journal_dir or "in memory only (no directory given)")
        self.udp_server.moveToThread(self.server_thread)
        self.udp_server.reset_signal.connect(self._restart_monitor)
        self.udp_server.stop_signal.connect(self._stop_monitor)
        self.udp_server.message_received.connect(self._handle_message)
        self.server_thread.started.connect(self.udp_server.run)
        self.server_thread.start()

    def _on_monitor_stopped(self, reason: str) -> None:
        log.warning("Monitor is not running: %s", reason)
        # The signal is queued from the manager thread and may land a tick
        # before the thread has fully exited; the 8 Hz check picks the same
        # detail up from the manager as soon as isRunning() drops.
        if not self.monitor_manager.isRunning():
            self._set_status(STATES.NOT_READY,
                             self.monitor_manager.last_stop_kind or "failed", reason)

    def _not_ready_detail(self) -> tuple[str, str]:
        """(sub_state, reason) for NOT_READY when nothing more specific is known.

        A specific NOT_READY sub_state already recorded (by the stop handler or
        by ``_on_monitor_stopped``) is kept; otherwise it is the manager's
        classification of the last exit, or ``never_started``.
        """
        if self.status.state == STATES.NOT_READY and self.status.sub_state:
            return self.status.sub_state, self.status.reason
        kind = self.monitor_manager.last_stop_kind
        if kind is None:
            return "never_started", ""
        return kind, self.monitor_manager.last_stop_reason or ""

    def _stop_monitor(self) -> None:
        """``stop`` command: stop the monitor and leave it stopped."""
        if self.monitor_manager.isRunning():
            log.info("Stopping monitor experiment on request...")
            self.monitor_manager.stop()
        else:
            log.info("Stop requested; the monitor experiment is not running.")
        self._set_status(STATES.NOT_READY, "stopped_on_request",
                         "stopped by a client request")

    def _restart_monitor(self) -> None:
        if getattr(self, "_restarting", False):
            log.info("Monitor restart ignored: a restart is already in progress.")
            return
        self._restarting = True
        self.udp_server.retire_ops("monitor restarting")
        try:
            if self.monitor_manager.isRunning():
                log.info("Restarting monitor experiment...")
                self.monitor_manager.stop()
            else:
                log.info("Starting monitor experiment...")
            self.monitor_manager.start()
            if self.monitor_manager.isRunning():
                self._set_status(STATES.LOADING, "starting", "")
            else:
                # start() refused (pre-flight failure); it has already said why
                # and _on_monitor_stopped has recorded preflight_failed.
                self._set_status(STATES.NOT_READY, *self._not_ready_detail())
        finally:
            self._restarting = False

    def _set_status(self, status, sub_state: str | None = None,
                    reason: str | None = None) -> None:
        changed = self.status.set_state(status, sub_state, reason)
        self.status.pid = self.monitor_manager.pid
        # NOT_READY retires the composite-op registration.
        self.udp_server.on_monitor_state(status, self.status.sub_state)
        # Logged only on change: _check_status runs at 8 Hz.
        previous = getattr(self, "_logged_state", None)
        if changed or previous is None:
            current = f"{_state_name(status)} ({self.status.sub_state})"
            if previous is None:
                log.info("state: %s", current)
            else:
                log.info("state: %s -> %s", previous, current)
            self._logged_state = current

    def _check_status(self) -> None:
        running = self.monitor_manager.isRunning()
        if running and self.status.state != STATES.READY:
            self._set_status(STATES.LOADING, "starting", "")
        elif not running:
            self._set_status(STATES.NOT_READY, *self._not_ready_detail())
        else:
            self._set_status(STATES.READY, "running", "")

    def _handle_message(self, message: str) -> None:
        log.info("msg: %s", message)
        if "run complete" in message:
            self._restart_monitor()
        elif "monitor ready" in message:
            self._set_status(STATES.READY, "running", "")

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


def run(monitor_expt_path: str, config_file_path: str | None = None,
        journal_dir: str | None = None) -> int:
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
        server = HeadlessMonitorServer(monitor_expt_path, config_file_path=config_file_path,
                                       journal_dir=journal_dir)
    except Exception:
        log.exception("Monitor server failed to start")
        return 1

    def _on_quit():
        server.shutdown()

    app.aboutToQuit.connect(_on_quit)
    return app.exec()
