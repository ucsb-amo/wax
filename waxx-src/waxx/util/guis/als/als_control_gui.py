"""
PyQt6 GUI for ALS Fiber Amplifier Laser Control
Accepts TCP commands, displays real-time status, and manages startup/shutdown sequences.
"""

import sys
import threading
import socket
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGridLayout, QGroupBox, QStatusBar, QFrame,
    QLineEdit, QCheckBox, QPlainTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtCore import QSize

from waxx.util.guis.als.als_gui_client import ALSGuiClient
from waxx.util.guis.als.als_fiber_amplifier import ALSLaserController, ALSLaserStartupController

LOGGER = logging.getLogger("als_laser_gui")
LOGGER.setLevel(logging.INFO)


def create_emoji_icon(emoji: str) -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        font = QFont()
        font.setPixelSize(max(12, int(size * 0.8)))
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, emoji)
        painter.end()

        icon.addPixmap(pixmap)

    return icon

class LogEmitter(QObject):
    message_logged = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord):
        self.emitter.message_logged.emit(self.format(record))


class ConnectionState(Enum):
    """Serial connection states"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


class SequenceState(Enum):
    """Sequence execution states"""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"


class StatusDot(QPushButton):
    """Clickable status indicator with red/green state."""
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.is_on = False
        self.setObjectName("StatusDotButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(34)
        self._update_color()

    def set_status(self, is_on: bool):
        """Set status: True for on (green), False for off (red)."""
        self.is_on = is_on
        self._update_color()

    def _update_color(self):
        if self.is_on:
            color = QColor(43, 163, 99)
            state_text = "ON"
        else:
            color = QColor(208, 63, 55)
            state_text = "OFF"

        self.setText(f"{self.label_text}: {state_text}")
        self.setStyleSheet(
            f"background-color: {color.name()}; color: #ffffff; border-radius: 10px; "
            f"padding: 8px 10px; font-weight: 700; text-align: left;"
        )


class PowerInputField(QLineEdit):
    """Custom line edit for power input that handles focus loss"""
    def __init__(self, parent=None, on_focus_out=None):
        super().__init__(parent)
        self.on_focus_out = on_focus_out
    
    def focusOutEvent(self, event):
        """Handle focus out event"""
        super().focusOutEvent(event)
        if self.on_focus_out:
            self.on_focus_out()


class StepIndicator(QWidget):
    """Visual indicator for a sequence step (gray/yellow/green light)"""
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.state = "not_done"  # "not_done", "doing", "done"
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # Color indicator (light)
        self.indicator = QPushButton()
        self.indicator.setFixedSize(20, 20)
        self.indicator.setEnabled(False)
        self._update_color()
        layout.addWidget(self.indicator)
        
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        # Primary label
        self.label = QLabel(label)
        self.label.setMinimumWidth(150)
        text_layout.addWidget(self.label)

        # Secondary status line (completed/already done timestamps)
        self.note_label = QLabel("")
        self.note_label.setStyleSheet("color: #5b6670; font-size: 11px;")
        self.note_label.setVisible(False)
        text_layout.addWidget(self.note_label)

        layout.addLayout(text_layout)
        layout.addStretch()
        
    def set_state(self, state: str):
        """Set state: 'not_done', 'doing', or 'done'"""
        self.state = state
        self._update_color()

    def set_note(self, note: str) -> None:
        text = note.strip()
        self.note_label.setText(text)
        self.note_label.setVisible(bool(text))
    
    def _update_color(self):
        if self.state == "not_done":
            color = QColor(200, 200, 200)  # Gray
        elif self.state == "doing":
            color = QColor(255, 255, 0)  # Yellow
        elif self.state == "done":
            color = QColor(0, 200, 0)  # Green
        else:
            color = QColor(200, 200, 200)
        
        self.indicator.setStyleSheet(f"background-color: {color.name()}; border-radius: 10px;")


class SequenceProgressWindow(QWidget):
    """Separate window for startup/shutdown progress display."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Sequence Progress")
        self.setMinimumWidth(420)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(10)

    def set_panels(self, startup_panel: QGroupBox, shutdown_panel: QGroupBox) -> None:
        self._layout.addWidget(startup_panel)
        self._layout.addWidget(shutdown_panel)


@dataclass
class LaserStatus:
    """Current laser status snapshot"""
    power_enabled: bool = False
    interlock_enabled: bool = False
    second_stage_enabled: bool = False
    power_setpoint_percent: float = 0.0
    temperature_act_p: float = 0.0
    temperature_set_p: float = 0.0
    imon_pa: float = 0.0
    lmon: float = 0.0
    pmon_w: float = 0.0
    connected: bool = False
    connection_state: ConnectionState = ConnectionState.DISCONNECTED


class SerialWorker(QObject):
    """Worker thread for serial communication with laser"""
    
    # Signals
    status_updated = pyqtSignal(LaserStatus)
    connection_state_changed = pyqtSignal(ConnectionState)
    
    def __init__(self, port: str = "COM6"):
        super().__init__()
        self.port = port
        self.laser: Optional[ALSLaserController] = None
        self.poll_interval = 1.0  # seconds
        self.poll_timer: Optional[QTimer] = None
        
    def run(self):
        """Initialize timer-driven polling in the worker thread."""
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(int(self.poll_interval * 1000))
        self.poll_timer.timeout.connect(self._poll_status)
    
    def _poll_status(self):
        """Poll laser status and emit signal"""
        if self.laser is None:
            return
        
        try:
            frame = self.laser.stop_and_read()
            if frame is None:
                return
            converted = self.laser.convert_frame(frame)
            power_raw = self.laser.cmd_ask_power_consign()
            power_enabled = bool(frame.statuses.get("STS_RELAY_PSU", 0)) or bool(
                frame.statuses.get("STS_RACK_PSU", 0)
            )
            
            status = LaserStatus(
                power_enabled=power_enabled,
                interlock_enabled=bool(self.laser.cmd_ask_interlock_sts()),
                second_stage_enabled=bool(self.laser.cmd_ask_secondstage_sts()),
                power_setpoint_percent=power_raw / 65535.0 * 100.0,
                temperature_act_p=converted.TACT_P,
                temperature_set_p=converted.TSET_P,
                imon_pa=converted.IMON_PA,
                lmon=converted.LMON,
                pmon_w=converted.PMON_W,
                connected=True,
                connection_state=ConnectionState.CONNECTED
            )
            self.status_updated.emit(status)
        except Exception as exc:
            if self.poll_timer is not None:
                self.poll_timer.stop()
            LOGGER.exception("Serial polling failed: %s", exc)
            self.connection_state_changed.emit(ConnectionState.ERROR)
    
    def connect_laser(self):
        """Establish connection"""
        try:
            if self.laser is None:
                self.laser = ALSLaserController(port=self.port)
            self.laser.connect()
            self.laser.handshake()
            LOGGER.info("Serial connection opened on %s", self.port)
            self.connection_state_changed.emit(ConnectionState.CONNECTED)
            self._poll_status()
            if self.poll_timer is not None:
                self.poll_timer.start()
        except Exception as exc:
            if self.poll_timer is not None:
                self.poll_timer.stop()
            if self.laser is not None:
                try:
                    self.laser.close()
                except Exception:
                    pass
                self.laser = None
            LOGGER.exception("Failed to connect to serial port %s: %s", self.port, exc)
            self.connection_state_changed.emit(ConnectionState.ERROR)
    
    def disconnect_laser(self):
        """Close connection"""
        if self.poll_timer is not None:
            self.poll_timer.stop()
        if self.laser is not None:
            try:
                self.laser.close()
                LOGGER.info("Serial connection closed on %s", self.port)
            except Exception:
                pass
            self.laser = None
        self.status_updated.emit(LaserStatus())
        self.connection_state_changed.emit(ConnectionState.DISCONNECTED)
    
    def set_power_percent(self, power_percent: float):
        """Set laser power and refresh state."""
        if self.laser is None:
            raise RuntimeError("Laser not connected")
        self.laser.set_power_percent(power_percent)
        LOGGER.info("Power setpoint changed to %.1f%%", power_percent)
        self._poll_status()

    def set_power_supply_on(self):
        if self.laser is None:
            raise RuntimeError("Laser not connected")
        self.laser.cmd_set_power_supply_on()
        LOGGER.info("Power supply turned on")
        self._poll_status()

    def set_power_supply_off(self):
        if self.laser is None:
            raise RuntimeError("Laser not connected")
        self.laser.cmd_set_power_supply_off()
        LOGGER.info("Power supply turned off")
        self._poll_status()

    def set_interlock_on(self):
        if self.laser is None:
            raise RuntimeError("Laser not connected")
        self.laser.cmd_set_interlock_on()
        LOGGER.info("Interlock turned on")
        self._poll_status()

    def set_interlock_off(self):
        if self.laser is None:
            raise RuntimeError("Laser not connected")
        self.laser.cmd_set_interlock_off()
        LOGGER.info("Interlock turned off")
        self._poll_status()

    def set_second_stage_on(self):
        if self.laser is None:
            raise RuntimeError("Laser not connected")
        self.laser.cmd_set_second_stage_on()
        LOGGER.info("2nd stage turned on")
        self._poll_status()

    def set_second_stage_off(self):
        if self.laser is None:
            raise RuntimeError("Laser not connected")
        self.laser.cmd_set_second_stage_off()
        LOGGER.info("2nd stage turned off")
        self._poll_status()

    def stop(self):
        """Stop worker"""
        self.disconnect_laser()


class ServerWorker(QObject):
    """Worker thread for TCP server"""
    
    # Signals
    command_received = pyqtSignal(str)
    
    def __init__(self, host: str = "192.168.1.76", port: int = 5557):
        super().__init__()
        self.host = host
        self.port = port
        self.running = True
        self.server_socket: Optional[socket.socket] = None
        self._latest_power_setpoint_percent = 0.0
        self._status_lock = threading.Lock()
    
    def run(self):
        """Main server loop"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            LOGGER.info("TCP server listening on %s:%s", self.host, self.port)
            
            while self.running:
                try:
                    self.server_socket.settimeout(1.0)
                    client_socket, client_addr = self.server_socket.accept()
                    
                    # Handle client in a thread
                    threading.Thread(
                        target=self._handle_client,
                        args=(client_socket,),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        LOGGER.exception("Server error: %s", e)
        except Exception as e:
            LOGGER.exception("Failed to start server: %s", e)
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def _handle_client(self, client_socket: socket.socket):
        """Handle client connection"""
        try:
            data = client_socket.recv(1024).decode('utf-8').strip()
            if data:
                LOGGER.info("TCP command received: %s", data)
                command = data.upper().strip()
                if command in {"GET_POWER_SETPOINT", "GET_POWER_SETPOINT_PERCENT"}:
                    value = self.get_latest_power_setpoint_percent()
                    client_socket.send(f"POWER_SETPOINT: {value:.3f}\n".encode("utf-8"))
                else:
                    self.command_received.emit(data)
                    client_socket.send(b"OK\n")
        except Exception as e:
            LOGGER.exception("Client error: %s", e)
        finally:
            client_socket.close()

    def set_latest_power_setpoint_percent(self, value: float) -> None:
        """Update cached power setpoint value used by status polling commands."""
        with self._status_lock:
            self._latest_power_setpoint_percent = float(value)

    def get_latest_power_setpoint_percent(self) -> float:
        """Read cached power setpoint value in a thread-safe way."""
        with self._status_lock:
            return self._latest_power_setpoint_percent
    
    def stop(self):
        """Stop server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        LOGGER.info("TCP server stopped")


class SequenceWorker(QObject):
    """Worker for running startup/shutdown sequences"""
    
    # Signals
    step_started = pyqtSignal(int, str)  # step_num, step_name
    step_completed = pyqtSignal(int)
    sequence_completed = pyqtSignal(str)  # 'STARTUP' or 'SHUTDOWN'
    sequence_interrupted = pyqtSignal(str)
    
    def __init__(self, startup_controller: ALSLaserStartupController):
        super().__init__()
        self.startup_controller = startup_controller
        self.running = False
        self.interrupt_requested = False
    
    def run_startup(self):
        """Run startup sequence"""
        self.running = True
        self.interrupt_requested = False
        
        try:
            steps = [
                (1, "Turn Power On", self.startup_controller.step_1_turn_laser_power_on),
                (2, "Turn Interlock On", self.startup_controller.step_2_turn_interlock_on),
                (3, "Wait for IMON-PA", self.startup_controller.step_3_wait_for_imon_pa),
                (4, "Turn On Second Stage", self.startup_controller.step_4_turn_on_second_stage),
                (5, "Ramp to 80%", self.startup_controller.step_5_ramp_to_80_percent),
                (6, "Warm Up at 80%", self.startup_controller.step_6_warm_up_at_80_percent),
                (7, "Turn to 100%", self.startup_controller.step_7_turn_to_100_percent),
            ]
            
            for step_num, step_name, step_func in steps:
                if self.interrupt_requested:
                    LOGGER.warning("Startup sequence interrupted")
                    self.sequence_interrupted.emit("STARTUP")
                    return
                
                LOGGER.info("Startup step %s: %s", step_num, step_name)
                self.step_started.emit(step_num, step_name)
                step_func()
                self.step_completed.emit(step_num)
            
            LOGGER.info("Startup sequence completed")
            self.sequence_completed.emit("STARTUP")
        except Exception as e:
            LOGGER.exception("Startup error: %s", e)
            self.sequence_interrupted.emit("STARTUP")
        finally:
            self.running = False
    
    def run_shutdown(self):
        """Run shutdown sequence"""
        self.running = True
        self.interrupt_requested = False
        
        try:
            steps = [
                (1, "Ramp Down to 0%", self.startup_controller.step_1_ramp_down_to_zero_percent),
                (2, "Turn Off Second Stage", self.startup_controller.step_2_turn_off_second_stage),
                (3, "Turn Off Interlock", self.startup_controller.step_3_turn_off_interlock),
                (4, "Turn Off Power", self.startup_controller.step_4_turn_off_laser_power),
            ]
            
            for step_num, step_name, step_func in steps:
                if self.interrupt_requested:
                    LOGGER.warning("Shutdown sequence interrupted")
                    self.sequence_interrupted.emit("SHUTDOWN")
                    return
                
                LOGGER.info("Shutdown step %s: %s", step_num, step_name)
                self.step_started.emit(step_num, step_name)
                step_func()
                self.step_completed.emit(step_num)
            
            LOGGER.info("Shutdown sequence completed")
            self.sequence_completed.emit("SHUTDOWN")
        except Exception as e:
            LOGGER.exception("Shutdown error: %s", e)
            self.sequence_interrupted.emit("SHUTDOWN")
        finally:
            self.running = False
    
    def interrupt(self):
        """Request sequence interruption"""
        self.interrupt_requested = True


class ALSControlGUI(QMainWindow):
    """Main GUI window for laser control"""

    request_serial_connect = pyqtSignal()
    request_serial_disconnect = pyqtSignal()
    request_set_power_percent = pyqtSignal(float)
    request_power_supply_on = pyqtSignal()
    request_power_supply_off = pyqtSignal()
    request_interlock_on = pyqtSignal()
    request_interlock_off = pyqtSignal()
    request_second_stage_on = pyqtSignal()
    request_second_stage_off = pyqtSignal()
    
    def __init__(self, ip: str = "192.168.1.76", port: int = 5557, serial_port: str = "COM6"):
        super().__init__()
        self.ip = ip
        self.port = port
        self.serial_port = serial_port
        self._window_icon = create_emoji_icon("🔫")
        self.setWindowIcon(self._window_icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(self._window_icon)
        
        # Current status
        self.status = LaserStatus()
        
        # Worker threads
        self.serial_worker: Optional[SerialWorker] = None
        self.serial_thread: Optional[QThread] = None
        self.server_worker: Optional[ServerWorker] = None
        self.server_thread: Optional[QThread] = None
        self.sequence_worker: Optional[SequenceWorker] = None
        self.sequence_thread: Optional[QThread] = None
        self.remote_client: Optional[ALSGuiClient] = None
        self._next_log_index = 0
        self._last_remote_error: Optional[str] = None
        self._remote_sequence_state = SequenceState.IDLE
        self._remote_sequence_type: Optional[str] = None
        self._remote_serial_port = serial_port
        self._startup_step_base_labels = [
            "1. Turn Power On",
            "2. Turn Interlock On",
            "3. Wait for IMON-PA > 3A",
            "4. Turn On Second Stage",
            "5. Ramp to 80%",
            "6. Warm Up at 80%",
            "7. Turn to 100%",
        ]
        
        # UI Setup
        self.setWindowTitle("ALS Fiber Amplifier Control")
        self.setGeometry(100, 100, 1200, 800)
        self._apply_theme()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        dashboard_layout = QHBoxLayout()
        dashboard_layout.setSpacing(16)

        left_column = QVBoxLayout()
        left_column.setSpacing(16)
        self.status_panel = self._create_status_panel()
        self.control_panel = self._create_control_panel()
        left_panel_width = max(
            self.status_panel.minimumSizeHint().width(),
            self.control_panel.minimumSizeHint().width(),
        )
        self.status_panel.setFixedWidth(left_panel_width)
        self.control_panel.setFixedWidth(left_panel_width)
        left_column.addWidget(self.status_panel)
        left_column.addWidget(self.control_panel)
        left_column.addStretch()
        dashboard_layout.addLayout(left_column, 2)

        right_column = QVBoxLayout()
        right_column.setSpacing(16)
        right_column.addWidget(self._create_measurements_panel())
        right_column.addStretch()
        dashboard_layout.addLayout(right_column, 3)

        main_layout.addLayout(dashboard_layout)
        
        # Sequence progress panels in separate window
        self.startup_sequence_panel = self._create_startup_sequence_panel()
        self.shutdown_sequence_panel = self._create_shutdown_sequence_panel()
        self.sequence_progress_window = SequenceProgressWindow(self)
        self.sequence_progress_window.set_panels(
            self.startup_sequence_panel,
            self.shutdown_sequence_panel,
        )
        self.startup_sequence_panel.hide()
        self.shutdown_sequence_panel.hide()
        self.sequence_progress_window.hide()
        
        # Status bar
        self.statusBar().showMessage("Ready")

        self.adjustSize()
        self.setMinimumSize(self.minimumSizeHint())
        self.resize(self.minimumSizeHint())
        
        # Initialize worker threads
        self._init_workers()
        
        # Timer for periodic status updates from the remote ALS server
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._sync_remote_state)
        self.status_timer.start(500)
        
        # Timer for auto-scrolling log back to bottom after user scrolls up
        self.auto_scroll_timer = QTimer()
        self.auto_scroll_timer.setSingleShot(True)
        self.auto_scroll_timer.timeout.connect(self._auto_scroll_log)

    def _apply_theme(self):
        """Apply a warm instrument-panel theme."""
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f6f1e8;
                color: #1f2a30;
                font-family: Segoe UI;
            }
            QGroupBox {
                border: 1px solid #d8cfc0;
                border-radius: 16px;
                margin-top: 14px;
                padding-top: 14px;
                background: #fffaf2;
                font-size: 13px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #5b6670;
            }
            QFrame#ConnectionCard, QFrame#PowerCard, QFrame#MetricCard, QFrame#IndicatorPanel {
                background: #fbf7f0;
                border: 1px solid #ddd3c3;
                border-radius: 14px;
            }
            QPushButton {
                background: #295c67;
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #347381;
            }
            QPushButton:disabled {
                background: #b7c1c4;
                color: #edf1f2;
            }
            QLineEdit {
                background: #fffdf9;
                border: 1px solid #d4c9ba;
                border-radius: 10px;
                padding: 8px;
            }
            QCheckBox {
                spacing: 6px;
                color: #5b6670;
            }
            QLabel#CardEyebrow {
                color: #7c847c;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            QLabel#PowerOutputLabel {
                color: #c26b2d;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#MetricIcon {
                font-size: 20px;
            }
            QLabel#MetricValue {
                font-size: 22px;
                font-weight: 700;
                color: #1f2a30;
            }
            QLabel#MetricLabel {
                color: #6f777e;
                font-size: 11px;
            }
            QPlainTextEdit {
                background: #fffdf9;
                border: 1px solid #d4c9ba;
                border-radius: 12px;
                padding: 8px;
                color: #2b3136;
                font-family: Consolas;
                font-size: 12px;
            }
            """
        )

    def _create_metric_card(self, icon: str, title: str, subtitle: str, initial_value: str) -> tuple[QFrame, QLabel]:
        """Create a compact telemetry card with icon and value."""
        card = QFrame()
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        icon_label = QLabel(icon)
        icon_label.setObjectName("MetricIcon")
        header_layout.addWidget(icon_label)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("CardEyebrow")
        title_layout.addWidget(title_label)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("MetricLabel")
        title_layout.addWidget(subtitle_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        value_label = QLabel(initial_value)
        value_label.setObjectName("MetricValue")
        layout.addWidget(value_label)

        return card, value_label
    
    def _create_status_panel(self) -> QGroupBox:
        """Create status display panel"""
        group = QGroupBox("System")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        connection_card = QFrame()
        connection_card.setObjectName("ConnectionCard")
        connection_layout = QVBoxLayout(connection_card)
        connection_layout.setContentsMargins(12, 8, 12, 8)
        connection_layout.setSpacing(6)

        connection_header_layout = QHBoxLayout()
        connection_header_layout.setSpacing(8)

        connection_title = QLabel("TCP")
        connection_title.setObjectName("CardEyebrow")
        connection_header_layout.addWidget(connection_title)
        connection_header_layout.addStretch()

        self.connection_port_label = QLabel(f"{self.ip}:{self.port}")
        self.connection_port_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #1f2a30;")
        connection_header_layout.addWidget(self.connection_port_label)

        connection_layout.addLayout(connection_header_layout)

        self.connect_button = QPushButton(f"{self._remote_serial_port} Disconnected")
        self.connect_button.setMinimumWidth(180)
        self.connect_button.setMaximumWidth(220)
        self.connect_button.setStyleSheet(
            "background-color: #d03f37; color: #ffffff; border-radius: 10px; padding: 10px 14px; font-weight: 700;"
        )
        self.connect_button.clicked.connect(self._toggle_connection)
        connection_layout.addWidget(self.connect_button)
        layout.addWidget(connection_card)

        indicators_panel = QFrame()
        indicators_panel.setObjectName("IndicatorPanel")
        indicators_layout = QVBoxLayout(indicators_panel)
        indicators_layout.setContentsMargins(10, 10, 10, 10)
        indicators_layout.setSpacing(6)

        indicators_title = QLabel("Laser Indicators")
        indicators_title.setObjectName("CardEyebrow")
        indicators_layout.addWidget(indicators_title)

        self.power_status_dot = StatusDot("Power")
        self.power_status_dot.clicked.connect(self._toggle_power_status)
        indicators_layout.addWidget(self.power_status_dot)
        self.interlock_status_dot = StatusDot("Interlock")
        self.interlock_status_dot.clicked.connect(self._toggle_interlock_status)
        indicators_layout.addWidget(self.interlock_status_dot)
        self.second_stage_status_dot = StatusDot("2nd Stage")
        self.second_stage_status_dot.clicked.connect(self._toggle_second_stage_status)
        indicators_layout.addWidget(self.second_stage_status_dot)

        layout.addWidget(indicators_panel)
        
        return group

    def _create_measurements_panel(self) -> QGroupBox:
        """Create the right-side power and telemetry panel."""
        group = QGroupBox("Readout")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        power_row = QHBoxLayout()
        power_row.setSpacing(10)

        power_box = QFrame()
        power_box.setObjectName("PowerCard")
        power_layout = QVBoxLayout(power_box)
        power_layout.setContentsMargins(14, 10, 14, 10)
        power_layout.setSpacing(8)

        header_layout = QHBoxLayout()
        power_label = QLabel("Power Setpoint")
        power_label.setObjectName("CardEyebrow")
        header_layout.addWidget(power_label)
        self.power_edit_checkbox = QCheckBox("Edit")
        self.power_edit_checkbox.setChecked(False)
        self.power_edit_checkbox.stateChanged.connect(self._on_power_edit_toggled)
        header_layout.addWidget(self.power_edit_checkbox)
        header_layout.addStretch()
        power_layout.addLayout(header_layout)

        self.power_setpoint_label = QLabel("0%")
        self.power_setpoint_label.setStyleSheet(
            "font-size: 42px; font-weight: 800; color: #295c67; text-align: center;"
        )
        self.power_setpoint_label.setMinimumHeight(58)
        self.power_setpoint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        power_layout.addWidget(self.power_setpoint_label)

        self.power_input_field = PowerInputField(on_focus_out=self._on_power_input_focus_out)
        self.power_input_field.setStyleSheet(
            "font-size: 26px; font-weight: 700; color: #295c67; text-align: center;"
        )
        self.power_input_field.setMinimumHeight(44)
        self.power_input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.power_input_field.returnPressed.connect(self._on_power_input_submit)
        power_layout.addWidget(self.power_input_field)

        self.power_submit_hint = QLabel("ENTER to submit")
        self.power_submit_hint.setStyleSheet("font-size: 10px; color: #7c847c; text-align: center;")
        self.power_submit_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        power_layout.addWidget(self.power_submit_hint)

        self._update_power_edit_mode()
        power_row.addWidget(power_box, 1)

        optical_box = QFrame()
        optical_box.setObjectName("PowerCard")
        optical_layout = QVBoxLayout(optical_box)
        optical_layout.setContentsMargins(14, 10, 14, 10)
        optical_layout.setSpacing(8)

        optical_eyebrow = QLabel("Optical Output")
        optical_eyebrow.setObjectName("CardEyebrow")
        optical_layout.addWidget(optical_eyebrow)

        self.optical_power_label = QLabel("0 W")
        self.optical_power_label.setStyleSheet(
            "font-size: 42px; font-weight: 800; color: #c26b2d; text-align: center;"
        )
        self.optical_power_label.setMinimumHeight(58)
        self.optical_power_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        optical_layout.addWidget(self.optical_power_label)
        optical_layout.addStretch()
        power_row.addWidget(optical_box, 1)

        layout.addLayout(power_row)

        telemetry_title = QLabel("Telemetry")
        telemetry_title.setObjectName("CardEyebrow")
        layout.addWidget(telemetry_title)

        telemetry_row = QHBoxLayout()
        telemetry_row.setSpacing(12)
        temp_card, self.temp_label = self._create_metric_card(
            "🌡", "Temperatures", "Actual / Setpoint", "0.0 / 0.0 °C"
        )
        telemetry_row.addWidget(temp_card)
        current_card, self.current_label = self._create_metric_card(
            "⚡", "Currents", "IMON-PA / LMON", "0.00 / 0.00 A"
        )
        telemetry_row.addWidget(current_card)
        layout.addLayout(telemetry_row)

        log_title = QLabel("Activity Log")
        log_title.setObjectName("CardEyebrow")
        layout.addWidget(log_title)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(1000)
        log_font = self.log_output.font()
        log_font.setPointSize(9)
        self.log_output.setFont(log_font)
        layout.addWidget(self.log_output, 1)

        return group
    
    def _create_control_panel(self) -> QGroupBox:
        """Create control panel with startup/shutdown buttons"""
        group = QGroupBox("Controls")
        group.setMaximumWidth(280)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.startup_button = QPushButton("Startup")
        self.startup_button.clicked.connect(self._start_startup)
        layout.addWidget(self.startup_button)
        
        self.shutdown_button = QPushButton("Shutdown")
        self.shutdown_button.clicked.connect(self._start_shutdown)
        layout.addWidget(self.shutdown_button)
        
        layout.addStretch()
        
        return group
    
    def _create_startup_sequence_panel(self) -> QGroupBox:
        """Create startup sequence progress panel"""
        group = QGroupBox("Startup Sequence Progress")
        layout = QVBoxLayout(group)
        
        self.startup_steps = [
            StepIndicator(self._startup_step_base_labels[0]),
            StepIndicator(self._startup_step_base_labels[1]),
            StepIndicator(self._startup_step_base_labels[2]),
            StepIndicator(self._startup_step_base_labels[3]),
            StepIndicator(self._startup_step_base_labels[4]),
            StepIndicator(self._startup_step_base_labels[5]),
            StepIndicator(self._startup_step_base_labels[6]),
        ]
        
        for step in self.startup_steps:
            layout.addWidget(step)

        self.startup_interrupt_button = QPushButton("Interrupt Startup")
        self.startup_interrupt_button.clicked.connect(self._interrupt_sequence)
        self.startup_interrupt_button.setEnabled(False)
        layout.addWidget(self.startup_interrupt_button)
        
        return group
    
    def _create_shutdown_sequence_panel(self) -> QGroupBox:
        """Create shutdown sequence progress panel"""
        group = QGroupBox("Shutdown Sequence Progress")
        layout = QVBoxLayout(group)
        
        self.shutdown_steps = [
            StepIndicator("1. Ramp Down to 0%"),
            StepIndicator("2. Turn Off Second Stage"),
            StepIndicator("3. Turn Off Interlock"),
            StepIndicator("4. Turn Off Power"),
        ]
        
        for step in self.shutdown_steps:
            layout.addWidget(step)

        self.shutdown_interrupt_button = QPushButton("Interrupt Shutdown")
        self.shutdown_interrupt_button.clicked.connect(self._interrupt_sequence)
        self.shutdown_interrupt_button.setEnabled(False)
        layout.addWidget(self.shutdown_interrupt_button)
        
        return group
    
    def _init_workers(self):
        """Initialize remote ALS server client and fetch initial state."""
        self.remote_client = ALSGuiClient(host=self.ip, port=self.port, timeout_s=0.75)
        QTimer.singleShot(0, self._sync_remote_state)
    
    def _toggle_connection(self):
        """Toggle connection to the remote ALS hardware server."""
        if self.remote_client is None:
            return
        if self.status.connection_state == ConnectionState.CONNECTED:
            self.remote_client.disconnect_serial()
            self.statusBar().showMessage("Disconnect requested")
        else:
            self.remote_client.connect_serial()
            self.statusBar().showMessage("Connect requested")
        self._sync_remote_state()

    def _toggle_power_status(self):
        if self.remote_client is None:
            return
        if self.status.connection_state != ConnectionState.CONNECTED:
            self.statusBar().showMessage("Laser not connected")
            return
        if self.status.power_enabled:
            self.remote_client.set_power_supply_off()
        else:
            self.remote_client.set_power_supply_on()
        self._sync_remote_state()

    def _toggle_interlock_status(self):
        if self.remote_client is None:
            return
        if self.status.connection_state != ConnectionState.CONNECTED:
            self.statusBar().showMessage("Laser not connected")
            return
        if self.status.interlock_enabled:
            self.remote_client.set_interlock_off()
        else:
            self.remote_client.set_interlock_on()
        self._sync_remote_state()

    def _toggle_second_stage_status(self):
        if self.remote_client is None:
            return
        if self.status.connection_state != ConnectionState.CONNECTED:
            self.statusBar().showMessage("Laser not connected")
            return
        if self.status.second_stage_enabled:
            self.remote_client.set_second_stage_off()
        else:
            self.remote_client.set_second_stage_on()
        self._sync_remote_state()

    def _sequence_thread_is_running(self) -> bool:
        return self._remote_sequence_state == SequenceState.RUNNING

    def _cleanup_sequence_thread(self) -> None:
        self.sequence_thread = None
        self.sequence_worker = None
    
    def _start_startup(self):
        """Request remote startup sequence."""
        if self.remote_client is None:
            return
        if self.status.connection_state != ConnectionState.CONNECTED:
            self.statusBar().showMessage("Error: Laser not connected")
            return
        if self._sequence_thread_is_running():
            self.statusBar().showMessage("A sequence is already running")
            return
        self.remote_client.run_startup_sequence()
        self.statusBar().showMessage("Startup sequence requested")
        self._sync_remote_state()
    
    def _start_shutdown(self):
        """Request remote shutdown sequence."""
        if self.remote_client is None:
            return
        if self.status.connection_state != ConnectionState.CONNECTED:
            self.statusBar().showMessage("Error: Laser not connected")
            return
        if self._sequence_thread_is_running():
            self.statusBar().showMessage("A sequence is already running")
            return
        self.remote_client.run_shutdown_sequence()
        self.statusBar().showMessage("Shutdown sequence requested")
        self._sync_remote_state()
    
    def _interrupt_sequence(self):
        """Interrupt current remote sequence."""
        if self.remote_client is None:
            return
        self.remote_client.interrupt_sequence()
        self.statusBar().showMessage("Sequence interrupt requested")
        self._sync_remote_state()

    def _sync_remote_state(self):
        """Poll the ALS server for latest state and logs."""
        if self.remote_client is None:
            return
        try:
            snapshot = self.remote_client.get_snapshot()
            self._last_remote_error = None
            self._apply_remote_snapshot(snapshot)
            self._sync_remote_logs(snapshot.get("log_count", self._next_log_index))
        except Exception as exc:
            self._handle_remote_error(exc)

    def _apply_remote_snapshot(self, snapshot: Dict[str, Any]) -> None:
        status_data = snapshot.get("status", {})
        serial_port = snapshot.get("serial_port")
        if isinstance(serial_port, str) and serial_port.strip():
            self._remote_serial_port = serial_port.strip()
        connection_state_value = status_data.get("connection_state", ConnectionState.DISCONNECTED.value)
        try:
            connection_state = ConnectionState(connection_state_value)
        except ValueError:
            connection_state = ConnectionState.ERROR

        self.status = LaserStatus(
            power_enabled=bool(status_data.get("power_enabled", False)),
            interlock_enabled=bool(status_data.get("interlock_enabled", False)),
            second_stage_enabled=bool(status_data.get("second_stage_enabled", False)),
            power_setpoint_percent=float(status_data.get("power_setpoint_percent", 0.0)),
            temperature_act_p=float(status_data.get("temperature_act_p", 0.0)),
            temperature_set_p=float(status_data.get("temperature_set_p", 0.0)),
            imon_pa=float(status_data.get("imon_pa", 0.0)),
            lmon=float(status_data.get("lmon", 0.0)),
            pmon_w=float(status_data.get("pmon_w", 0.0)),
            connected=bool(status_data.get("connected", False)),
            connection_state=connection_state,
        )
        self._on_connection_state_changed(connection_state)
        self._apply_remote_sequence_state(snapshot.get("sequence", {}))
        self._update_display()

    def _apply_remote_sequence_state(self, sequence_data: Dict[str, Any]) -> None:
        previous_state = self._remote_sequence_state
        previous_type = self._remote_sequence_type
        state_value = sequence_data.get("state", SequenceState.IDLE.value)
        try:
            sequence_state = SequenceState(state_value)
        except ValueError:
            sequence_state = SequenceState.IDLE
        sequence_type = sequence_data.get("type")
        self._remote_sequence_state = sequence_state
        self._remote_sequence_type = sequence_type
        sequence_started_epoch = sequence_data.get("started_epoch")

        startup_steps = sequence_data.get("startup_steps", [])
        shutdown_steps = sequence_data.get("shutdown_steps", [])
        startup_step_notes = sequence_data.get("startup_step_notes", [])
        shutdown_step_notes = sequence_data.get("shutdown_step_notes", [])
        for index, state in enumerate(startup_steps[:len(self.startup_steps)]):
            self.startup_steps[index].set_state(state)
            note = startup_step_notes[index] if index < len(startup_step_notes) else ""
            self.startup_steps[index].set_note(note)
        for index, state in enumerate(shutdown_steps[:len(self.shutdown_steps)]):
            self.shutdown_steps[index].set_state(state)
            note = shutdown_step_notes[index] if index < len(shutdown_step_notes) else ""
            self.shutdown_steps[index].set_note(note)

        if sequence_state == SequenceState.RUNNING:
            self.startup_button.setEnabled(False)
            self.shutdown_button.setEnabled(False)
            self.startup_interrupt_button.setEnabled(sequence_type == "STARTUP")
            self.shutdown_interrupt_button.setEnabled(sequence_type == "SHUTDOWN")
            self.connect_button.setEnabled(False)
            self.power_status_dot.setEnabled(False)
            self.interlock_status_dot.setEnabled(False)
            self.second_stage_status_dot.setEnabled(False)
            if sequence_type == "STARTUP":
                self.startup_sequence_panel.show()
                self.shutdown_sequence_panel.hide()
                self._update_startup_wait_step_labels(startup_steps, sequence_started_epoch)
            elif sequence_type == "SHUTDOWN":
                self.startup_sequence_panel.hide()
                self.shutdown_sequence_panel.show()
            self.sequence_progress_window.show()
        else:
            is_connected = self.status.connection_state == ConnectionState.CONNECTED
            self.startup_button.setEnabled(is_connected)
            self.shutdown_button.setEnabled(is_connected)
            self.startup_interrupt_button.setEnabled(False)
            self.shutdown_interrupt_button.setEnabled(False)
            self.connect_button.setEnabled(True)
            self.power_status_dot.setEnabled(is_connected)
            self.interlock_status_dot.setEnabled(is_connected)
            self.second_stage_status_dot.setEnabled(is_connected)
            self.startup_sequence_panel.hide()
            self.shutdown_sequence_panel.hide()
            self.sequence_progress_window.hide()
            self._reset_startup_step_labels()
            if previous_state == SequenceState.RUNNING and sequence_state == SequenceState.COMPLETED:
                self.statusBar().showMessage(f"{(previous_type or sequence_type or 'Sequence').title()} completed")
            elif previous_state == SequenceState.RUNNING and sequence_state == SequenceState.INTERRUPTED:
                self.statusBar().showMessage(f"{(previous_type or sequence_type or 'Sequence').title()} interrupted")

    def _reset_startup_step_labels(self) -> None:
        for index, base_label in enumerate(self._startup_step_base_labels):
            self.startup_steps[index].label.setText(base_label)

    def _update_startup_wait_step_labels(self, startup_steps: list, sequence_started_epoch: Any) -> None:
        self._reset_startup_step_labels()
        if len(startup_steps) > 2 and startup_steps[2] == "doing":
            self.startup_steps[2].label.setText(
                f"3. Wait for IMON-PA: {self.status.imon_pa:.2f} / 3.00 A"
            )

        if len(startup_steps) > 5 and startup_steps[5] == "doing":
            elapsed_s = 0
            if isinstance(sequence_started_epoch, (int, float)):
                elapsed_s = max(0, int(time.time() - sequence_started_epoch))
            minutes, seconds = divmod(elapsed_s, 60)
            self.startup_steps[5].label.setText(
                f"6. Warm Up at 80% (since turn-on: {minutes:02d}:{seconds:02d})"
            )

    def _sync_remote_logs(self, log_count: int) -> None:
        if self.remote_client is None or log_count <= self._next_log_index:
            return
        log_response = self.remote_client.get_logs_since(self._next_log_index)
        for message in log_response.get("messages", []):
            self._append_log_message(message)
        self._next_log_index = int(log_response.get("next_index", self._next_log_index))

    def _handle_remote_error(self, exc: Exception) -> None:
        error_text = str(exc)
        if error_text != self._last_remote_error:
            self.statusBar().showMessage(f"Server communication error: {error_text}")
            self._last_remote_error = error_text
        self.status = LaserStatus(connected=False, connection_state=ConnectionState.ERROR)
        self._on_connection_state_changed(ConnectionState.ERROR)
        self._update_display()
    
    def _on_startup_step_started(self, step_num: int, step_name: str):
        """Callback when startup step starts"""
        self.startup_steps[step_num - 1].set_state("doing")
    
    def _on_startup_step_completed(self, step_num: int):
        """Callback when startup step completes"""
        self.startup_steps[step_num - 1].set_state("done")
    
    def _on_startup_completed(self, _):
        """Callback when startup sequence completes"""
        self._cleanup_sequence_thread()
        self.startup_button.setEnabled(True)
        self.shutdown_button.setEnabled(True)
        self.startup_interrupt_button.setEnabled(False)
        self.shutdown_interrupt_button.setEnabled(False)
        self.connect_button.setEnabled(True)
        is_connected = self.status.connection_state == ConnectionState.CONNECTED
        self.power_status_dot.setEnabled(is_connected)
        self.interlock_status_dot.setEnabled(is_connected)
        self.second_stage_status_dot.setEnabled(is_connected)
        self.startup_sequence_panel.hide()
        self.sequence_progress_window.hide()
        self.statusBar().showMessage("Startup sequence completed")
    
    def _on_shutdown_step_started(self, step_num: int, step_name: str):
        """Callback when shutdown step starts"""
        self.shutdown_steps[step_num - 1].set_state("doing")
    
    def _on_shutdown_step_completed(self, step_num: int):
        """Callback when shutdown step completes"""
        self.shutdown_steps[step_num - 1].set_state("done")
    
    def _on_shutdown_completed(self, _):
        """Callback when shutdown sequence completes"""
        self._cleanup_sequence_thread()
        self.startup_button.setEnabled(True)
        self.shutdown_button.setEnabled(True)
        self.startup_interrupt_button.setEnabled(False)
        self.shutdown_interrupt_button.setEnabled(False)
        self.connect_button.setEnabled(True)
        is_connected = self.status.connection_state == ConnectionState.CONNECTED
        self.power_status_dot.setEnabled(is_connected)
        self.interlock_status_dot.setEnabled(is_connected)
        self.second_stage_status_dot.setEnabled(is_connected)
        self.shutdown_sequence_panel.hide()
        self.sequence_progress_window.hide()
        self.statusBar().showMessage("Shutdown sequence completed")
    
    def _on_sequence_interrupted(self, sequence_type: str):
        """Callback when sequence is interrupted"""
        self._cleanup_sequence_thread()
        self.startup_button.setEnabled(True)
        self.shutdown_button.setEnabled(True)
        self.startup_interrupt_button.setEnabled(False)
        self.shutdown_interrupt_button.setEnabled(False)
        self.connect_button.setEnabled(True)
        is_connected = self.status.connection_state == ConnectionState.CONNECTED
        self.power_status_dot.setEnabled(is_connected)
        self.interlock_status_dot.setEnabled(is_connected)
        self.second_stage_status_dot.setEnabled(is_connected)
        self.startup_sequence_panel.hide()
        self.shutdown_sequence_panel.hide()
        self.sequence_progress_window.hide()
        self.statusBar().showMessage(f"{sequence_type} sequence interrupted")
    
    def _on_status_updated(self, status: LaserStatus):
        """Callback when laser status is updated"""
        self.status = status
    
    def _on_connection_state_changed(self, state: ConnectionState):
        """Callback when connection state changes"""
        self.status.connection_state = state
        is_sequence_running = self._sequence_thread_is_running()
        if state == ConnectionState.CONNECTED:
            self.status.connected = True
            self.connect_button.setText(f"{self._remote_serial_port} Connected")
            self.connect_button.setStyleSheet(
                "background-color: #2ba363; color: #ffffff; border-radius: 10px; padding: 10px 14px; font-weight: 700;"
            )
            self.connect_button.setEnabled(not is_sequence_running)
            self.startup_button.setEnabled(not is_sequence_running)
            self.shutdown_button.setEnabled(not is_sequence_running)
            if not is_sequence_running:
                self.power_status_dot.setEnabled(True)
                self.interlock_status_dot.setEnabled(True)
                self.second_stage_status_dot.setEnabled(True)
        elif state == ConnectionState.ERROR:
            self.status.connected = False
            self.connect_button.setText(f"{self._remote_serial_port} Disconnected")
            self.connect_button.setStyleSheet(
                "background-color: #d03f37; color: #ffffff; border-radius: 10px; padding: 10px 14px; font-weight: 700;"
            )
            self.connect_button.setEnabled(True)
            self.startup_button.setEnabled(False)
            self.shutdown_button.setEnabled(False)
            self.power_status_dot.setEnabled(False)
            self.interlock_status_dot.setEnabled(False)
            self.second_stage_status_dot.setEnabled(False)
        else:
            self.status.connected = False
            self.connect_button.setText(f"{self._remote_serial_port} Disconnected")
            self.connect_button.setStyleSheet(
                "background-color: #d03f37; color: #ffffff; border-radius: 10px; padding: 10px 14px; font-weight: 700;"
            )
            self.connect_button.setEnabled(True)
            self.startup_button.setEnabled(False)
            self.shutdown_button.setEnabled(False)
            self.power_status_dot.setEnabled(False)
            self.interlock_status_dot.setEnabled(False)
            self.second_stage_status_dot.setEnabled(False)
    
    def _on_server_command(self, command: str):
        """Callback when server receives a command"""
        command = command.upper().strip()
        
        if command == "START_STARTUP":
            self._start_startup()
        elif command == "START_SHUTDOWN":
            self._start_shutdown()
        elif command == "INTERRUPT":
            self._interrupt_sequence()

        LOGGER.info("Server command dispatched: %s", command)

    def _append_log_message(self, message: str):
        """Append a log message to the GUI log pane with smart auto-scroll."""
        # Check if scrollbar is at the bottom
        scrollbar = self.log_output.verticalScrollBar()
        is_at_bottom = scrollbar.value() == scrollbar.maximum()
        
        # Append the message
        self.log_output.appendPlainText(message)
        
        # If user is at the bottom, scroll to the new bottom
        if is_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
            # Cancel any pending auto-scroll since user is already at bottom
            self.auto_scroll_timer.stop()
        else:
            # User has scrolled up; restart timer to auto-scroll back down after 10 seconds
            # (restarting gives them another 10s from this new message)
            self.auto_scroll_timer.start(10000)  # 10 seconds
    
    def _auto_scroll_log(self):
        """Auto-scroll log back to bottom after 10 seconds of user scrolling up."""
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _update_power_edit_mode(self):
        """Update power display/input visibility based on edit mode"""
        is_editing = self.power_edit_checkbox.isChecked()
        self.power_setpoint_label.setVisible(not is_editing)
        self.power_input_field.setVisible(is_editing)
        self.power_submit_hint.setVisible(is_editing)
        
        if is_editing:
            # Set input field to current value
            self.power_input_field.setText(f"{self.status.power_setpoint_percent:.1f}")
            self.power_input_field.selectAll()
            self.power_input_field.setFocus()
    
    def _on_power_edit_toggled(self, state):
        """Handle power edit checkbox toggle"""
        self._update_power_edit_mode()
    
    def _on_power_input_submit(self):
        """Handle power input submission (Enter key pressed)"""
        try:
            power_percent = float(self.power_input_field.text())
            # Clamp to valid range
            power_percent = max(0.0, min(100.0, power_percent))
            
            # Send command to laser
            if self.status.connection_state == ConnectionState.CONNECTED:
                if self.remote_client is not None:
                    self.remote_client.set_power_percent(power_percent)
                self.statusBar().showMessage(f"Power set to {power_percent:.1f}%")
                self._sync_remote_state()
            else:
                self.statusBar().showMessage("Laser not connected")
            
            # Exit edit mode
            self.power_edit_checkbox.setChecked(False)
        except ValueError:
            self.statusBar().showMessage("Invalid power value")
            self.power_input_field.setText(f"{self.status.power_setpoint_percent:.1f}")
    
    def _on_power_input_focus_out(self):
        """Handle power input focus loss (revert changes)"""
        QTimer.singleShot(0, self._maybe_exit_power_edit_mode)

    def _maybe_exit_power_edit_mode(self):
        """Exit power edit mode unless focus moved onto the checkbox or input."""
        focus_widget = QApplication.focusWidget()
        if focus_widget in (self.power_input_field, self.power_edit_checkbox):
            return
        self.power_edit_checkbox.setChecked(False)
    
    def _update_display(self):
        """Update display with current status"""
        # Connection indicator already updated via signal
        
        # Power status dot
        self.power_status_dot.set_status(self.status.power_enabled)
        
        # Interlock status dot
        self.interlock_status_dot.set_status(self.status.interlock_enabled)
        
        # Second stage status dot
        self.second_stage_status_dot.set_status(self.status.second_stage_enabled)

        all_indicators_off = (
            not self.status.power_enabled
            and not self.status.interlock_enabled
            and not self.status.second_stage_enabled
        )
        displayed_setpoint_percent = 0.0 if not self.status.power_enabled else self.status.power_setpoint_percent
        displayed_optical_power_w = 0.0 if not self.status.power_enabled else self.status.pmon_w
        
        # Power setpoint (only update label if not editing)
        if not self.power_edit_checkbox.isChecked():
            self.power_setpoint_label.setText(f"{displayed_setpoint_percent:.1f}%")
        
        # Temperature
        if all_indicators_off:
            self.temp_label.setText("-- / -- °C")
        else:
            self.temp_label.setText(
                f"{self.status.temperature_act_p:.1f} / {self.status.temperature_set_p:.1f} °C"
            )
        
        # Current
        if all_indicators_off:
            self.current_label.setText("-- / -- A")
        else:
            self.current_label.setText(f"{self.status.imon_pa:.2f} / {self.status.lmon:.2f} A")
        
        # Optical power
        self.optical_power_label.setText(f"{displayed_optical_power_w:.2f} W")
    
    def closeEvent(self, event):
        """Handle window close"""
        LOGGER.info("GUI closing")

        if self.status_timer.isActive():
            self.status_timer.stop()

        self.sequence_progress_window.close()
        
        event.accept()