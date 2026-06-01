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
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QRunnable, QThread, QThreadPool, QTimer
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
        self.setMinimumHeight(22)
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
        self.label.setMinimumWidth(80)
        text_layout.addWidget(self.label)

        # Secondary status line (completed/already done timestamps)
        self.note_label = QLabel("")
        self.note_label.setStyleSheet("color: #8a949a; font-size: 11px;")
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


class _BgCall(QRunnable):
    """Run a callable on QThreadPool; report via callback.

    Used so the GUI thread never blocks on the periodic snapshot/log
    round-trip — without this the dashboard windows stutter when moved.
    """

    def __init__(self, func, on_done) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._func = func
        self._on_done = on_done

    def run(self) -> None:  # noqa: D401
        try:
            res = self._func()
        except BaseException as exc:  # noqa: BLE001
            self._on_done(None, exc)
            return
        self._on_done(res, None)


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
    # Cross-thread plumbing for the background snapshot/log fetcher.
    _remote_state_ready = pyqtSignal(object, object)   # (snapshot, logs_payload|None)
    _remote_state_failed = pyqtSignal(str)
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
        self._remote_poll_in_flight = False
        self._remote_connect_in_flight = False
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
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        dashboard_layout = QVBoxLayout()
        dashboard_layout.setSpacing(8)

        # Vertically stacked to match the Precilaser GUI layout: the
        # panel can compress horizontally to almost any width because
        # status, controls, and measurements stack instead of competing
        # for horizontal room.
        self.status_panel = self._create_status_panel()
        self.control_panel = self._create_control_panel()
        dashboard_layout.addWidget(self.status_panel)
        dashboard_layout.addWidget(self.control_panel)
        dashboard_layout.addWidget(self._create_measurements_panel(), 1)
        dashboard_layout.addStretch()

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
        # Allow the dock panel to shrink horizontally; the layout will
        # still enforce a sensible vertical minimum.
        hint = self.minimumSizeHint()
        self.setMinimumSize(0, hint.height())
        self.resize(hint)
        
        # Initialize worker threads
        self._init_workers()
        
        # Timer for periodic status updates from the remote ALS server
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._sync_remote_state)
        self.status_timer.start(500)

        # Cross-thread completion signals from the background remote-state fetcher.
        self._remote_state_ready.connect(self._on_remote_state_ready)
        self._remote_state_failed.connect(self._on_remote_state_failed)
        
        # Timer for auto-scrolling log back to bottom after user scrolls up
        self.auto_scroll_timer = QTimer()
        self.auto_scroll_timer.setSingleShot(True)
        self.auto_scroll_timer.timeout.connect(self._auto_scroll_log)

    def _apply_theme(self):
        """Dark, dashboard-native theme.

        The panel is meant to live inside the kexp dashboard (dark
        ``#2b2b2b`` chrome).  Keep the window background transparent so
        the dock body shows through, then layer slightly-lighter cards
        and group boxes on top for depth.  Accents (teal for setpoints,
        amber for measured output) carry over from the original warm
        theme so the visual language stays consistent.
        """
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: transparent;
                color: #d8d8d8;
                font-family: 'Segoe UI';
            }
            QGroupBox {
                border: 1px solid #4a4a4a;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px 8px 8px 8px;
                background: #323232;
                font-size: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                color: #9aa3a8;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.04em;
            }
            QFrame#ConnectionCard, QFrame#PowerCard, QFrame#MetricCard, QFrame#IndicatorPanel {
                background: #3a3a3a;
                border: 1px solid #4f4f4f;
                border-radius: 8px;
            }
            QPushButton {
                background: #3d6b78;
                color: #f1f3f4;
                border: 1px solid #4f8896;
                border-radius: 6px;
                padding: 5px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #4d8294;
                border-color: #6aa3b3;
            }
            QPushButton:pressed {
                background: #355b66;
            }
            QPushButton:disabled {
                background: #2f3a3d;
                color: #6a7479;
                border-color: #3a4347;
            }
            QLineEdit {
                background: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 5px 8px;
                selection-background-color: #4d8294;
            }
            QLineEdit:focus { border: 1px solid #6aa3b3; }
            QCheckBox {
                spacing: 6px;
                color: #aab1b5;
                font-weight: 500;
            }
            QCheckBox::indicator {
                width: 13px; height: 13px;
                border: 1px solid #6a6a6a;
                border-radius: 3px;
                background: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background: #4d8294;
                border-color: #6aa3b3;
            }
            QLabel#CardEyebrow {
                color: #8a949a;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.10em;
                text-transform: uppercase;
            }
            QLabel#PowerOutputLabel {
                color: #e8a87c;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#MetricIcon {
                font-size: 18px;
            }
            QLabel#MetricValue {
                font-size: 20px;
                font-weight: 700;
                color: #e6e6e6;
            }
            QLabel#MetricLabel {
                color: #8a949a;
                font-size: 10px;
            }
            QPlainTextEdit {
                background: #262626;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                padding: 6px;
                color: #c8c8c8;
                font-family: Consolas;
                font-size: 11px;
                selection-background-color: #4d8294;
            }
            QStatusBar {
                background: transparent;
                color: #8a949a;
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Connection buttons kept as orphan widgets so existing setText /
        # setEnabled / setStyleSheet calls still work, but hidden from the
        # layout - the dashboard server panel already shows server reachability
        # and the COM-port LED, so duplicating them inside the GUI is noise.
        self.server_conn_button = QPushButton("Server: searching\u2026")
        self.server_conn_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.server_conn_button.clicked.connect(self._retry_server_connection)
        self.server_conn_button.setVisible(False)

        self.connect_button = QPushButton(f"{self._remote_serial_port} Disconnected")
        self.connect_button.setStyleSheet(
            "background-color: #d03f37; color: #ffffff; border-radius: 10px;"
            " padding: 10px 14px; font-weight: 700;"
        )
        self.connect_button.clicked.connect(self._toggle_connection)
        self.connect_button.setVisible(False)

        # Status dots live directly in the System group — no inner
        # "Laser Indicators" frame/title wrapper, to save vertical space.
        self.power_status_dot = StatusDot("Power")
        self.power_status_dot.clicked.connect(self._toggle_power_status)
        layout.addWidget(self.power_status_dot)
        self.interlock_status_dot = StatusDot("Interlock")
        self.interlock_status_dot.clicked.connect(self._toggle_interlock_status)
        layout.addWidget(self.interlock_status_dot)
        self.second_stage_status_dot = StatusDot("2nd Stage")
        self.second_stage_status_dot.clicked.connect(self._toggle_second_stage_status)
        layout.addWidget(self.second_stage_status_dot)

        return group

    def _create_measurements_panel(self) -> QWidget:
        """Create the right-side power and telemetry block.

        No outer "Readout" wrapper — each piece (Power Setpoint, Optical
        Output, Telemetry, Activity Log) is its own ``QGroupBox`` so the
        visual treatment matches the Precilaser panel.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── Power readouts side-by-side, each in its own group box ───
        power_col = QHBoxLayout()
        power_col.setSpacing(8)

        power_box = QGroupBox("Power")
        power_layout = QVBoxLayout(power_box)
        power_layout.setContentsMargins(10, 8, 10, 8)
        power_layout.setSpacing(6)

        # Edit checkbox lives on the QGroupBox title strip itself — not
        # as a separate header row inside — to save vertical space.
        self.power_edit_checkbox = QCheckBox("Edit", power_box)
        self.power_edit_checkbox.setChecked(False)
        self.power_edit_checkbox.stateChanged.connect(self._on_power_edit_toggled)
        self.power_edit_checkbox.setStyleSheet("background: transparent;")
        self._power_edit_anchor = power_box
        # Position is updated on first show + resize via an event filter.
        power_box.installEventFilter(self)

        self.power_setpoint_label = QLabel("0%")
        self.power_setpoint_label.setStyleSheet(
            "font-size: 26px; font-weight: 800; color: #5fb6c8;"
        )
        self.power_setpoint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Stretch above keeps the readout vertically centered when the
        # box grows (e.g. when neighbouring panels are hidden).
        power_layout.addStretch()
        power_layout.addWidget(self.power_setpoint_label)

        self.power_input_field = PowerInputField(on_focus_out=self._on_power_input_focus_out)
        self.power_input_field.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #5fb6c8; background: #2a2a2a;"
            " border: 1px solid #555; border-radius: 6px; padding: 4px;"
        )
        self.power_input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.power_input_field.returnPressed.connect(self._on_power_input_submit)
        power_layout.addWidget(self.power_input_field)

        self.power_submit_hint = QLabel("ENTER to submit")
        self.power_submit_hint.setStyleSheet("font-size: 10px; color: #8a949a;")
        self.power_submit_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        power_layout.addWidget(self.power_submit_hint)
        # Trailing stretch matches the leading one to centre the column.
        power_layout.addStretch()

        self._update_power_edit_mode()
        power_col.addWidget(power_box, 1)

        optical_box = QGroupBox("Optical Output")
        optical_layout = QVBoxLayout(optical_box)
        optical_layout.setContentsMargins(10, 8, 10, 8)
        optical_layout.setSpacing(6)

        self.optical_power_label = QLabel("0 W")
        self.optical_power_label.setStyleSheet(
            "font-size: 26px; font-weight: 800; color: #e8a87c;"
        )
        self.optical_power_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        optical_layout.addStretch()
        optical_layout.addWidget(self.optical_power_label)
        optical_layout.addStretch()
        power_col.addWidget(optical_box, 1)

        layout.addLayout(power_col)

        # ── Telemetry in a collapsible dropdown ───────────────────────
        try:
            from waxx.util.dashboard.widgets import CollapsibleGroupBox  # noqa: PLC0415
            telem_group = CollapsibleGroupBox(
                "Telemetry", expanded=False, scrollable=True, max_expanded_height=180,
            )
        except Exception:
            telem_group = QGroupBox("Telemetry")
            QHBoxLayout(telem_group)
        telem_inner = QHBoxLayout()
        telem_inner.setContentsMargins(8, 8, 8, 8)
        telem_inner.setSpacing(8)
        temp_card, self.temp_label = self._create_metric_card(
            "🌡", "Temperatures", "Actual / Setpoint", "0.0 / 0.0 °C"
        )
        telem_inner.addWidget(temp_card)
        current_card, self.current_label = self._create_metric_card(
            "⚡", "Currents", "IMON-PA / LMON", "0.00 / 0.00 A"
        )
        telem_inner.addWidget(current_card)
        if hasattr(telem_group, "setContentLayout"):
            telem_group.setContentLayout(telem_inner)
        else:
            telem_group.layout().addLayout(telem_inner)
        layout.addWidget(telem_group)
        self.telem_group = telem_group

        # ── Activity log: collapsible ────────────────────────────────
        try:
            from waxx.util.dashboard.widgets import CollapsibleGroupBox  # noqa: PLC0415
            log_box = CollapsibleGroupBox(
                "Activity Log", expanded=False, scrollable=True, max_expanded_height=220,
            )
        except Exception:
            log_box = QGroupBox("Activity Log")
            QVBoxLayout(log_box)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(1000)
        log_font = self.log_output.font()
        log_font.setPointSize(9)
        self.log_output.setFont(log_font)
        self.log_output.setMinimumHeight(80)
        if hasattr(log_box, "addWidget"):
            log_box.addWidget(self.log_output)
        else:
            log_box.layout().addWidget(self.log_output)
        layout.addWidget(log_box, 1)
        self.log_box = log_box

        return container
    
    def _create_control_panel(self) -> QGroupBox:
        """Create control panel with startup/shutdown buttons"""
        group = QGroupBox("Controls")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

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
        try:
            self.remote_client = ALSGuiClient(timeout_s=0.75)
            self._set_server_conn_button_state("connected")
        except RuntimeError:
            self.remote_client = None
            self._set_server_conn_button_state("searching")
            return
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
        """Poll the ALS server for state and logs without blocking the GUI.

        Both the discovery handshake and the per-tick round-trip happen on
        QThreadPool; results are marshalled back via Qt signals so window
        drags stay smooth even when the server is slow or missing.
        """
        if self._remote_poll_in_flight or self._remote_connect_in_flight:
            return

        if self.remote_client is None:
            self._remote_connect_in_flight = True
            self._set_server_conn_button_state("searching")

            def _build():
                return ALSGuiClient(discovery_timeout=0.1, timeout_s=0.75)

            def _build_done(result, exc):
                self._remote_connect_in_flight = False
                if exc is not None:
                    self._remote_state_failed.emit(str(exc))
                else:
                    self.remote_client = result
                    # Let the next tick fetch the snapshot.
                    self._remote_state_failed.emit("")
            QThreadPool.globalInstance().start(_BgCall(_build, _build_done))
            return

        self._remote_poll_in_flight = True
        client = self.remote_client
        next_idx = self._next_log_index

        def _fetch():
            snap = client.get_snapshot()
            log_count = (
                int(snap.get("log_count", next_idx)) if isinstance(snap, dict) else next_idx
            )
            logs_payload = None
            if log_count > next_idx:
                logs_payload = client.get_logs_since(next_idx)
            return (snap, logs_payload)

        def _done(result, exc):
            if exc is not None:
                self._remote_state_failed.emit(str(exc))
            else:
                snap, logs_payload = result
                self._remote_state_ready.emit(snap, logs_payload)

        QThreadPool.globalInstance().start(_BgCall(_fetch, _done))

    # ---- GUI-thread completion slots --------------------------------- #

    def _on_remote_state_ready(self, snapshot, logs_payload) -> None:
        self._remote_poll_in_flight = False
        self._set_server_conn_button_state("connected")
        self._last_remote_error = None
        if isinstance(snapshot, dict):
            self._apply_remote_snapshot(snapshot)
        if logs_payload is not None:
            for message in logs_payload.get("messages", []):
                self._append_log_message(message)
            self._next_log_index = int(
                logs_payload.get("next_index", self._next_log_index)
            )

    def _on_remote_state_failed(self, err: str) -> None:
        self._remote_poll_in_flight = False
        if not err:
            self._set_server_conn_button_state("connected")
            return
        # Surface error via the existing handler (status bar + state reset).
        self._handle_remote_error(RuntimeError(err))

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
        # Kept as a no-op shim; log syncing now piggy-backs on the
        # background snapshot fetch in ``_sync_remote_state``.
        return

    def _set_server_conn_button_state(self, state: str) -> None:
        """Update the server TCP connection button appearance.

        state: 'searching' | 'connected' | 'lost'
        """
        if state == "connected" and self.remote_client is not None:
            text = f"Server: {self.remote_client.host}:{self.remote_client.port}"
            style = "background-color: #2ba363; color: #ffffff; border-radius: 10px; padding: 10px 14px; font-weight: 700;"
        elif state == "lost":
            text = "Server: lost \u2014 click to retry"
            style = "background-color: #d03f37; color: #ffffff; border-radius: 10px; padding: 10px 14px; font-weight: 700;"
        else:
            text = "Server: searching\u2026"
            style = "background-color: #8c959e; color: #ffffff; border-radius: 10px; padding: 10px 14px; font-weight: 700;"
        self.server_conn_button.setText(text)
        self.server_conn_button.setStyleSheet(style)

    def _retry_server_connection(self) -> None:
        """Force immediate server rediscovery when the user clicks the connection button."""
        self.remote_client = None
        self._set_server_conn_button_state("searching")
        self._sync_remote_state()

    def _handle_remote_error(self, exc: Exception) -> None:
        error_text = str(exc)
        if error_text != self._last_remote_error:
            self.statusBar().showMessage(f"Server communication error: {error_text}")
            self._last_remote_error = error_text
        # Null the client so _sync_remote_state will attempt full rediscovery next tick.
        self.remote_client = None
        self._set_server_conn_button_state("lost")
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

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        # Re-anchor the Power-box Edit checkbox to the top-right of the
        # group box title strip after every resize / show, so removing the
        # in-box "Edit" header row doesn't lose the control.
        from PyQt6.QtCore import QEvent  # noqa: PLC0415
        anchor = getattr(self, "_power_edit_anchor", None)
        cb = getattr(self, "power_edit_checkbox", None)
        if anchor is not None and cb is not None and obj is anchor and event.type() in (
            QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Move,
        ):
            cb.adjustSize()
            margin = 12
            x = anchor.width() - cb.width() - margin
            y = 0  # sits on the group-box title line
            cb.move(max(0, x), y)
            cb.raise_()
        return super().eventFilter(obj, event)