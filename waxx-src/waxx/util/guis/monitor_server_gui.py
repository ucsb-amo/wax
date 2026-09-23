import socket
import json
import logging
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPainter

from waxx.util.device_state.monitor_manager import MonitorManager
from waxx.util.comms_server.comm_server import UdpServer, STATES, ReadyBit
from waxx.util.comms_server.state_broadcast import StateBroadcaster
from waxx.util.comms_server.hardware_id import monitor_server_id
from beacon.discovery.client import discover
from waxx.util.device_state.state_file_io import read_state, apply_delta

log = logging.getLogger(__name__)

_STATE_NAMES = {STATES.READY: "READY", STATES.LOADING: "LOADING",
                STATES.NOT_READY: "NOT_READY"}


def _state_name(state) -> str:
    return _STATE_NAMES.get(state, str(state))


class Status:
    def __init__(self,state=False):
        self.state = state

class MonitorUDPServer(UdpServer):
    """TCP responder + sole writer of the device-state JSON.

    Besides the legacy string commands (``status``/``reset``/``run complete``/
    ``monitor ready``) it handles structured JSON requests from clients:

    * ``{"type": "update", "device_type", "device_name", "changes"}`` — merge a
      delta into the JSON atomically, bump the version, broadcast the change.
    * ``{"type": "get_state"}`` — return the full snapshot + current version.

    The version starts from the current epoch seconds so that a server restart
    always yields versions higher than any value a client still holds (forcing
    a clean resync rather than ignoring "older" updates).
    """

    reset_signal = pyqtSignal()

    def __init__(self, config_file_path=None):
        super().__init__(host="0.0.0.0", port=0, server_id=monitor_server_id())

        self.status = Status()
        self._print_connections_bool = False

        self.config_file_path = config_file_path
        self._version = int(time.time())
        self._broadcaster = StateBroadcaster()

    def on_message_received(self,message):
        m = message.strip()
        if m.startswith("{"):
            # Structured (JSON) requests are fully handled in generate_reply.
            return
        if m == 'reset':
            self.reset_signal.emit()
        if m == 'status':
            return
        self.message_received.emit(message)

    def generate_reply(self, message):
        m = message.strip()
        if m.startswith("{"):
            return self._handle_structured(m)
        return str(int(self.status.state))

    def _handle_structured(self, raw):
        try:
            obj = json.loads(raw)
        except Exception:
            return json.dumps({"status": "error", "msg": "invalid json"})
        mtype = obj.get("type")
        if mtype == "get_state":
            return self._reply_get_state()
        if mtype == "update":
            return self._reply_update(obj)
        return json.dumps({"status": "error", "msg": f"unknown type {mtype}"})

    def _reply_get_state(self):
        if not self.config_file_path:
            return json.dumps({"status": "error", "msg": "no config path"})
        try:
            cfg = read_state(self.config_file_path)
        except FileNotFoundError:
            cfg = {}
        except Exception as e:
            return json.dumps({"status": "error", "msg": str(e)})
        return json.dumps({"status": "ok", "version": self._version, "config": cfg})

    def _reply_update(self, obj):
        if not self.config_file_path:
            return json.dumps({"status": "error", "msg": "no config path"})
        dtype = obj.get("device_type")
        name = obj.get("device_name")
        changes = obj.get("changes")
        if dtype not in ("dds", "dac", "ttl") or not name or not isinstance(changes, dict):
            return json.dumps({"status": "error", "msg": "bad update"})
        try:
            apply_delta(self.config_file_path, dtype, name, changes)
        except Exception as e:
            return json.dumps({"status": "error", "msg": str(e)})
        self._log_update(dtype, name, changes)
        self._version += 1
        version = self._version
        self._broadcaster.send({
            "type": "state_update",
            "version": version,
            "device_type": dtype,
            "device_name": name,
            "changes": changes,
        })
        # Keep linked DDS v_pd and DAC voltage in sync in both directions.
        self._propagate_linked_vpd(dtype, name, changes)
        return json.dumps({"status": "ok", "version": version})

    def _log_update(self, dtype: str, name: str, changes: dict) -> None:
        """Print a formatted confirmation of an accepted device-state update."""
        parts = []
        if dtype == "dds":
            if "frequency" in changes:
                parts.append(f"freq {changes['frequency'] / 1e6:.3f} MHz")
            if "amplitude" in changes:
                parts.append(f"amp {changes['amplitude']:.3f}")
            if "v_pd" in changes:
                parts.append(f"v_pd {changes['v_pd']:.3f} V")
            if "sw_state" in changes:
                parts.append("sw " + ("on" if changes["sw_state"] else "off"))
        elif dtype == "dac":
            if "voltage" in changes:
                parts.append(f"{changes['voltage']:.3f} V")
        elif dtype == "ttl":
            if "ttl_state" in changes:
                parts.append("on" if changes["ttl_state"] else "off")
        if parts:
            log.info("[%s] %s -> %s", dtype.upper(), name, ", ".join(parts))

    def _propagate_linked_vpd(self, dtype: str, name: str, changes: dict) -> None:
        """Cross-propagate v_pd <-> voltage for DDS/DAC pairs sharing a channel.

        The link is stored as ``dac_ch_key`` in every DDS config entry that has
        DAC control (written by ``generate_state_file.Generator``).  When either
        side changes the voltage the other side is updated atomically and a
        broadcast is sent so GUI widgets on both tabs stay in sync.
        
        Also propagates force_update_counter to ensure linked devices are
        force-updated together.
        """
        try:
            cfg = read_state(self.config_file_path)
        except Exception:
            return

        if dtype == "dds" and ("v_pd" in changes or "force_update_counter" in changes):
            dac_key = cfg.get("dds", {}).get(name, {}).get("dac_ch_key", "")
            if not dac_key:
                return
            linked = {}
            if "v_pd" in changes:
                linked["voltage"] = changes["v_pd"]
            if "force_update_counter" in changes:
                linked["force_update_counter"] = changes["force_update_counter"]
            if not linked:
                return
            try:
                apply_delta(self.config_file_path, "dac", dac_key, linked)
            except Exception:
                return
            self._version += 1
            self._broadcaster.send({
                "type": "state_update",
                "version": self._version,
                "device_type": "dac",
                "device_name": dac_key,
                "changes": linked,
            })

        elif dtype == "dac" and ("voltage" in changes or "force_update_counter" in changes):
            for dds_name, dds_cfg in cfg.get("dds", {}).items():
                if dds_cfg.get("dac_ch_key", "") != name:
                    continue
                linked = {}
                if "voltage" in changes:
                    linked["v_pd"] = changes["voltage"]
                if "force_update_counter" in changes:
                    linked["force_update_counter"] = changes["force_update_counter"]
                if not linked:
                    continue
                try:
                    apply_delta(self.config_file_path, "dds", dds_name, linked)
                except Exception:
                    continue
                self._version += 1
                self._broadcaster.send({
                    "type": "state_update",
                    "version": self._version,
                    "device_type": "dds",
                    "device_name": dds_name,
                    "changes": linked,
                })

    def stop(self):
        try:
            self._broadcaster.close()
        except Exception:
            pass
        super().stop()


class MonitorServerGUI(QWidget):
    def __init__(self,
                monitor_expt_path,
                config_file_path=None):
        super().__init__()

        self.config_file_path = config_file_path

        # Refuse to start a second monitor server for the same hardware.
        server_id = monitor_server_id()
        existing = discover(server_id, timeout=1.5)
        if existing is not None:
            ip, port = existing
            QMessageBox.critical(
                self,
                "Monitor server already running",
                f"A monitor server for '{server_id}' is already running at "
                f"{ip}:{port}.\n\nRefusing to start a second server for the same "
                "hardware.",
            )
            self._aborted = True
            QTimer.singleShot(0, self.close)
            return
        self._aborted = False

        self.setWindowTitle("Monitor Server")
        eye_icon = self._create_eye_icon()
        self.setWindowIcon(eye_icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(eye_icon)
        self.setGeometry(100, 100, 250, 80)

        # Everything the monitor reports goes through logging (stderr, line
        # buffered) rather than print, so it shows up promptly in the terminal
        # that launched this GUI as well as in the dashboard log.
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s %(levelname)s %(message)s")

        self.monitor_manager = MonitorManager(monitor_expt_path)
        self.monitor_manager.msg.connect(lambda m: log.info("monitor: %s", m))
        self.monitor_manager.monitor_stopped.connect(self._on_monitor_stopped)

        log.info("monitor experiment: %s", monitor_expt_path)
        log.info("device state file: %s", config_file_path)
        if config_file_path is None:
            log.error(
                "No device-state config path was passed: state reads/writes will "
                "fail with 'no config path'. The launcher could not resolve it "
                "(usually an unset env var or an unmapped drive)."
            )
        for problem in self.monitor_manager.preflight_problems():
            log.error("Monitor cannot be started as configured: %s", problem)

        self.status = Status()

        self.setup_ui()
        self.setup_udp_server()

        # Initial status is "not ready".  (This used to pass False, which equals
        # STATES.READY == 0 -- so the server briefly advertised READY, and the
        # button flashed green, while nothing was running.)
        self.set_status(STATES.NOT_READY)

        self.monitor_check_timer = QTimer(self)
        self.monitor_check_timer.setInterval(125)
        self.monitor_check_timer.timeout.connect(self.check_monitor_status)
        self.monitor_check_timer.start()

    @staticmethod
    def _create_eye_icon(size=64):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        font = QFont("Segoe UI Emoji")
        font.setPixelSize(int(size * 0.8))
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "👁")
        painter.end()

        return QIcon(pixmap)

    def setup_ui(self):
        layout = QVBoxLayout()
        self.status_indicator = QPushButton("NOT READY")
        self.status_indicator.clicked.connect(self.on_button_clicked)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.status_indicator.setFont(font)
        layout.addWidget(self.status_indicator)
        self.setLayout(layout)

    def setup_udp_server(self):
        self.server_thread = QThread()
        
        self.udp_server = MonitorUDPServer(config_file_path=self.config_file_path)
        self.udp_server.moveToThread(self.server_thread)

        self.udp_server.reset_signal.connect(self.restart_monitor)
        self.server_thread.started.connect(self.udp_server.run)
        self.udp_server.message_received.connect(self.handle_message)
        
        self.server_thread.start()

    def on_button_clicked(self):
        if self.status.state == STATES.READY:
            reply = QMessageBox.question(self, 'Restart Monitor',
                                         "Are you sure you'd like to restart the monitor experiment?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                log.info("Manual monitor restart triggered.")
                self.restart_monitor()
        elif self.status.state == STATES.NOT_READY:
            log.info("Manual monitor start triggered.")
            self.monitor_manager.start()

    def _on_monitor_stopped(self, reason: str) -> None:
        """Surface the monitor's own failure reason in the terminal and the UI."""
        log.warning("Monitor is not running: %s", reason)
        self.status_indicator.setToolTip(f"Monitor is not running: {reason}")

    def restart_monitor(self):
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
                self.set_status(STATES.LOADING)
            else:
                # start() refused (pre-flight failure); it has already said why.
                self.set_status(STATES.NOT_READY)
        finally:
            self._restarting = False

    def set_status(self, status):
        # Logged only on change: check_monitor_status runs at 8 Hz.
        previous = getattr(self, "_logged_state", None)
        if status != previous:
            if previous is None:
                log.info("state: %s", _state_name(status))
            else:
                log.info("state: %s -> %s", _state_name(previous), _state_name(status))
            self._logged_state = status

        if status == STATES.READY:
            self.status_indicator.setText("READY")
            self.status_indicator.setStyleSheet("background-color: green; color: white;")
        elif status == STATES.NOT_READY:
            self.status_indicator.setText("NOT READY")
            self.status_indicator.setStyleSheet("background-color: #c46666; color: white;")
        else:
            self.status_indicator.setText("Loading...")
            self.status_indicator.setStyleSheet("background-color: orange; color: white;")

        self.status.state = status
        self.udp_server.status.state = status

    def check_monitor_status(self):
        if self.monitor_manager.isRunning() and self.status_indicator.text() != "READY":
            self.set_status(STATES.LOADING)
        elif not self.monitor_manager.isRunning():
            self.set_status(STATES.NOT_READY)
        else:
            self.set_status(STATES.READY)

    def handle_message(self, message):
        log.info("msg: %s", message)
        if "run complete" in message:
            log.info("Run complete message received. Restarting monitor.")
            self.restart_monitor()
        elif "monitor ready" in message:
            log.info("Monitor ready message received.")
            self.set_status(STATES.READY)
        
    def closeEvent(self, event):
        # Guarded with getattr: when __init__ aborted early (another monitor
        # server was already running) none of these exist, and an
        # AttributeError traceback here would bury the message saying why.
        log.info("Closing monitor server GUI...")
        udp_server = getattr(self, "udp_server", None)
        if udp_server is not None:
            udp_server.stop()
        server_thread = getattr(self, "server_thread", None)
        if server_thread is not None:
            server_thread.quit()
            server_thread.wait()
        monitor_manager = getattr(self, "monitor_manager", None)
        if monitor_manager is not None:
            monitor_manager.stop()
        event.accept()