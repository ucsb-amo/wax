import socket
import json
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPainter

from waxx.util.device_state.monitor_manager import MonitorManager
from waxx.util.comms_server.comm_server import UdpServer, STATES, ReadyBit
from waxx.util.comms_server.state_broadcast import StateBroadcaster
from waxx.util.comms_server.hardware_id import monitor_server_id
from waxx.util.comms_server.waxx_client import discover
from waxx.util.device_state.state_file_io import read_state, apply_delta

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
            print(f"[{dtype.upper()}] {name} -> {', '.join(parts)}")

    def _propagate_linked_vpd(self, dtype: str, name: str, changes: dict) -> None:
        """Cross-propagate v_pd <-> voltage for DDS/DAC pairs sharing a channel.

        The link is stored as ``dac_ch_key`` in every DDS config entry that has
        DAC control (written by ``generate_state_file.Generator``).  When either
        side changes the voltage the other side is updated atomically and a
        broadcast is sent so GUI widgets on both tabs stay in sync.
        """
        try:
            cfg = read_state(self.config_file_path)
        except Exception:
            return

        if dtype == "dds" and "v_pd" in changes:
            dac_key = cfg.get("dds", {}).get(name, {}).get("dac_ch_key", "")
            if not dac_key:
                return
            linked = {"voltage": changes["v_pd"]}
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

        elif dtype == "dac" and "voltage" in changes:
            for dds_name, dds_cfg in cfg.get("dds", {}).items():
                if dds_cfg.get("dac_ch_key", "") != name:
                    continue
                linked = {"v_pd": changes["voltage"]}
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

        self.monitor_manager = MonitorManager(monitor_expt_path)
        self.monitor_manager.msg.connect(print) # For debugging

        self.status = Status()

        self.setup_ui()
        self.setup_udp_server()

        self.set_status(False) # Initial status is "not ready"

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
                print("Manual monitor restart triggered.")
                self.restart_monitor()
        elif self.status.state == STATES.NOT_READY:
            print("Manual monitor start triggered.")
            self.monitor_manager.start()

    def restart_monitor(self):
        if getattr(self, "_restarting", False):
            return
        self._restarting = True
        try:
            if self.monitor_manager.isRunning():
                self.monitor_manager.stop()
            self.monitor_manager.start()
            self.set_status(STATES.LOADING)
        finally:
            self._restarting = False

    def set_status(self, status):
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
        print(f"Message received: {message}")
        if "run complete" in message:
            print("Run complete message received. Restarting monitor.")
            self.restart_monitor()
            self.set_status(STATES.LOADING)
        elif "monitor ready" in message:
            print("Monitor ready message received.")
            self.set_status(STATES.READY)
        
    def closeEvent(self, event):
        print("Closing GUI...")
        self.udp_server.stop()
        self.server_thread.quit()
        self.server_thread.wait()
        self.monitor_manager.stop()
        event.accept()