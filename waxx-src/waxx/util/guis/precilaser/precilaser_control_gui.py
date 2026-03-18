"""PyQt6 GUI for Precilaser remote monitoring and control."""

from __future__ import annotations

import logging
import math
import sys
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from waxx.util.guis.precilaser.precilaser_gui_client import PrecilaserGuiClient


LOGGER = logging.getLogger("precilaser_gui")
LOGGER.setLevel(logging.INFO)


class StatusDot(QPushButton):
    def __init__(
        self,
        label: str,
        on_color: str = "#1f9d55",
        off_color: str = "#d64545",
        on_text: str = "OK",
        off_text: str = "NOT OK",
        parent=None,
    ):
        super().__init__(parent)
        self.label_text = label
        self.on_color = QColor(on_color)
        self.off_color = QColor(off_color)
        self.on_text = on_text
        self.off_text = off_text
        self.is_on = False
        self.setEnabled(False)
        self.setMinimumHeight(34)
        self._update_color()

    def set_status(self, is_on: bool):
        self.is_on = bool(is_on)
        self._update_color()

    def _update_color(self):
        if self.is_on:
            color = self.on_color
            state_text = self.on_text
        else:
            color = self.off_color
            state_text = self.off_text
        self.setText(f"{self.label_text}: {state_text}")
        self.setStyleSheet(
            f"background-color: {color.name()}; color: #ffffff; border-radius: 10px; "
            f"padding: 8px 10px; font-weight: 700; text-align: left;"
        )


class PrecilaserControlGUI(QMainWindow):
    def __init__(self, ip: str = "192.168.1.76", port: int = 5560):
        super().__init__()
        self.ip = ip
        self.port = int(port)
        self.client = PrecilaserGuiClient(host=ip, port=port)
        self._next_log_index = 0
        self._last_remote_error: str | None = None
        self._laser_enabled = False
        self._stability_enabled = False

        self.setWindowTitle("Precilaser Control")
        self.resize(1080, 760)
        self._apply_theme()

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        dashboard_layout = QHBoxLayout()
        dashboard_layout.setSpacing(12)

        left_column = QVBoxLayout()
        left_column.setSpacing(12)
        self.status_panel = self._create_status_panel()
        self.telemetry_panel = self._create_telemetry_panel()
        left_column.addWidget(self.status_panel)
        left_column.addWidget(self.telemetry_panel)
        left_column.addStretch()
        left_panel_width = max(
            self.status_panel.minimumSizeHint().width(),
            self.telemetry_panel.minimumSizeHint().width(),
        )
        self.status_panel.setFixedWidth(left_panel_width)
        self.telemetry_panel.setFixedWidth(left_panel_width)
        dashboard_layout.addLayout(left_column)

        right_column = QVBoxLayout()
        right_column.setSpacing(12)
        right_column.addWidget(self._create_control_panel())
        right_column.addWidget(self._create_log_panel(), 1)
        dashboard_layout.addLayout(right_column, 1)

        root_layout.addLayout(dashboard_layout)

        self.statusBar().showMessage("Ready")

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._sync_remote_state)
        self.status_timer.start(500)

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f7f3ed;
                color: #1f2a30;
                font-family: Segoe UI;
            }
            QGroupBox {
                border: 1px solid #d8cfc0;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 12px;
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
            QPushButton {
                background: #295c67;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
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
                border-radius: 8px;
                padding: 7px;
                font-size: 14px;
            }
            QPlainTextEdit {
                background: #fffdf9;
                border: 1px solid #d4c9ba;
                border-radius: 10px;
                padding: 8px;
                color: #2b3136;
                font-family: Consolas;
                font-size: 12px;
            }
            """
        )

    def _create_control_panel(self) -> QGroupBox:
        box = QGroupBox("Controls")
        layout = QVBoxLayout(box)

        self.laser_toggle_button = QPushButton("Enable Laser")
        self.laser_toggle_button.clicked.connect(self._toggle_laser_enable)
        layout.addWidget(self.laser_toggle_button)
        self._update_laser_button()

        current_row = QHBoxLayout()
        self.current_input = QLineEdit()
        self.current_input.setPlaceholderText("Working current (A)")
        self.current_input.returnPressed.connect(self._submit_working_current)
        current_row.addWidget(self.current_input)

        self.set_current_button = QPushButton("Set Current")
        self.set_current_button.clicked.connect(self._submit_working_current)
        current_row.addWidget(self.set_current_button)
        layout.addLayout(current_row)

        seq_row = QHBoxLayout()
        self.startup_button = QPushButton("Start Turn On")
        self.startup_button.clicked.connect(self._start_startup)
        seq_row.addWidget(self.startup_button)

        self.shutdown_button = QPushButton("Start Turn Off")
        self.shutdown_button.clicked.connect(self._start_shutdown)
        seq_row.addWidget(self.shutdown_button)

        self.interrupt_button = QPushButton("Interrupt")
        self.interrupt_button.clicked.connect(self._interrupt_sequence)
        seq_row.addWidget(self.interrupt_button)
        self.interrupt_button.hide()
        layout.addLayout(seq_row)

        self.sequence_state_value = QLabel("Sequence: IDLE")
        self.sequence_state_value.setStyleSheet("font-size: 13px; font-weight: 700;")
        layout.addWidget(self.sequence_state_value)

        return box

    def _create_status_panel(self) -> QGroupBox:
        box = QGroupBox("Status Indicators")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        self.connection_label = QLabel(f"Server: {self.ip}:{self.port}")
        self.connection_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(self.connection_label)

        self.connection_state_value = QLabel("DISCONNECTED")
        self.connection_state_value.setStyleSheet("font-size: 14px; font-weight: 700;")
        layout.addWidget(self.connection_state_value)

        self.serial_connect_button = QPushButton("Connect Serial")
        self.serial_connect_button.clicked.connect(self._toggle_serial_connection)
        layout.addWidget(self.serial_connect_button)
        self._update_connection_button("DISCONNECTED")

        self.pd_ok_dot = StatusDot("PD OK")
        self.temp_ok_dot = StatusDot("Temperature OK")
        self.laser_enable_dot = StatusDot("Laser Enable")
        self.stability_dot = StatusDot(
            "Power Stability",
            on_color="#2b6de0",
            off_color="#8b949e",
            on_text="ON",
            off_text="OFF",
        )
        self.stability_dot.setEnabled(True)
        self.stability_dot.clicked.connect(self._toggle_stability_mode)

        layout.addWidget(self.pd_ok_dot)
        layout.addWidget(self.temp_ok_dot)
        layout.addWidget(self.laser_enable_dot)
        layout.addWidget(self.stability_dot)

        return box

    def _create_telemetry_panel(self) -> QGroupBox:
        box = QGroupBox("Telemetry")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        pd_group = QGroupBox("Laser Readings")
        pd_layout = QGridLayout(pd_group)
        pd_icon = QLabel("🔆")
        pd_icon.setStyleSheet("font-size: 20px;")
        pd_layout.addWidget(pd_icon, 0, 0)
        pd_layout.addWidget(QLabel("PD1-PD5"), 0, 1)

        temp_group = QGroupBox("Temperature Readings")
        temp_layout = QGridLayout(temp_group)
        temp_icon = QLabel("🌡")
        temp_icon.setStyleSheet("font-size: 20px;")
        temp_layout.addWidget(temp_icon, 0, 0)
        temp_layout.addWidget(QLabel("T1-T4"), 0, 1)

        self.pd_labels: list[QLabel] = []
        self.temp_labels: list[QLabel] = []
        self.current_labels: list[QLabel] = []

        for i in range(5):
            title = QLabel(f"PD{i + 1}")
            value = QLabel("--")
            value.setStyleSheet("font-size: 16px; font-weight: 700;")
            pd_layout.addWidget(title, i + 1, 0)
            pd_layout.addWidget(value, i + 1, 1)
            self.pd_labels.append(value)

        for i in range(4):
            title = QLabel(f"T{i + 1} (C)")
            value = QLabel("--")
            value.setStyleSheet("font-size: 16px; font-weight: 700;")
            temp_layout.addWidget(title, i + 1, 0)
            temp_layout.addWidget(value, i + 1, 1)
            self.temp_labels.append(value)

        top_row.addWidget(pd_group, 1)
        top_row.addWidget(temp_group, 1)
        layout.addLayout(top_row)

        currents_group = QGroupBox("Currents (A)")
        currents_layout = QGridLayout(currents_group)
        for i in range(3):
            title = QLabel(f"ISET_RT[{i}]")
            value = QLabel("--")
            value.setStyleSheet("font-size: 17px; font-weight: 800;")
            currents_layout.addWidget(title, i, 0)
            currents_layout.addWidget(value, i, 1)
            self.current_labels.append(value)
        layout.addWidget(currents_group)

        return box

    def _create_log_panel(self) -> QGroupBox:
        box = QGroupBox("Logs")
        layout = QVBoxLayout(box)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return box

    def _append_log(self, message: str) -> None:
        self.log_text.appendPlainText(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_status_message(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _toggle_serial_connection(self):
        is_connected = self.connection_state_value.text().strip().upper() == "CONNECTED"
        try:
            if is_connected:
                if self.client.disconnect_serial():
                    self._set_status_message("Serial disconnect requested")
            else:
                if self.client.connect_serial():
                    self._set_status_message("Serial connection requested")
        except Exception as exc:
            self._append_log(f"ERROR toggle serial: {exc}")

    def _update_connection_button(self, connection_state: str) -> None:
        is_connected = connection_state.upper() == "CONNECTED"
        if is_connected:
            self.serial_connect_button.setText("Disconnect Serial")
            self.serial_connect_button.setStyleSheet(
                "background-color: #b54747; color: #ffffff; border-radius: 8px; padding: 8px 12px; font-weight: 700;"
            )
        else:
            self.serial_connect_button.setText("Connect Serial")
            self.serial_connect_button.setStyleSheet(
                "background-color: #2f7d50; color: #ffffff; border-radius: 8px; padding: 8px 12px; font-weight: 700;"
            )

    def _update_laser_button(self) -> None:
        if self._laser_enabled:
            self.laser_toggle_button.setText("Disable Laser")
            self.laser_toggle_button.setStyleSheet(
                "background-color: #2f7d50; color: #ffffff; border-radius: 8px; padding: 8px 12px; font-weight: 700;"
            )
        else:
            self.laser_toggle_button.setText("Enable Laser")
            self.laser_toggle_button.setStyleSheet(
                "background-color: #b54747; color: #ffffff; border-radius: 8px; padding: 8px 12px; font-weight: 700;"
            )

    def _toggle_laser_enable(self) -> None:
        self._set_laser_enable(not self._laser_enabled)

    def _toggle_stability_mode(self) -> None:
        self._set_stability_mode(not self._stability_enabled)

    def _set_laser_enable(self, enabled: bool):
        try:
            if self.client.set_laser_enable(enabled):
                self._set_status_message(f"Laser enable set to {enabled}")
                self._laser_enabled = enabled
                self._update_laser_button()
        except Exception as exc:
            self._append_log(f"ERROR set enable: {exc}")

    def _set_stability_mode(self, enabled: bool):
        try:
            if self.client.set_stability_mode(enabled):
                self._set_status_message(f"Stability mode set to {enabled}")
                self._stability_enabled = enabled
                self.stability_dot.set_status(enabled)
        except Exception as exc:
            self._append_log(f"ERROR set stability: {exc}")

    def _submit_working_current(self):
        text = self.current_input.text().strip()
        if not text:
            return
        try:
            value = float(text)
        except ValueError:
            self._set_status_message("Current must be numeric")
            return

        try:
            if self.client.set_working_current(value):
                self._set_status_message(f"Working current set to {value:.2f} A")
        except Exception as exc:
            self._append_log(f"ERROR set current: {exc}")

    def _start_startup(self):
        try:
            if self.client.run_startup_sequence():
                self._set_status_message("Turn-on sequence started")
        except Exception as exc:
            self._append_log(f"ERROR startup sequence: {exc}")

    def _start_shutdown(self):
        try:
            if self.client.run_shutdown_sequence():
                self._set_status_message("Turn-off sequence started")
        except Exception as exc:
            self._append_log(f"ERROR shutdown sequence: {exc}")

    def _interrupt_sequence(self):
        try:
            if self.client.interrupt_sequence():
                self._set_status_message("Interrupt requested")
        except Exception as exc:
            self._append_log(f"ERROR interrupt sequence: {exc}")

    def _sync_remote_state(self) -> None:
        try:
            snapshot = self.client.get_snapshot()
            self._apply_snapshot(snapshot)
            log_count = int(snapshot.get("log_count", self._next_log_index))
            self._sync_logs(log_count)
            self._last_remote_error = None
        except Exception as exc:
            err = str(exc)
            if err != self._last_remote_error:
                self._append_log(f"ERROR remote sync: {err}")
                self._last_remote_error = err

    def _sync_logs(self, log_count: int) -> None:
        if log_count <= self._next_log_index:
            return
        payload = self.client.get_logs_since(self._next_log_index)
        messages = payload.get("messages", [])
        for msg in messages:
            self._append_log(str(msg))
        self._next_log_index = int(payload.get("next_index", self._next_log_index))

    @staticmethod
    def _fmt_value(value: Any, precision: int = 2) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            return f"{value:.{precision}f}"
        return str(value)

    def _apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        status = snapshot.get("status", {})
        sequence = snapshot.get("sequence", {})

        connection_state = str(status.get("connection_state", "DISCONNECTED"))
        self.connection_state_value.setText(connection_state)
        self._update_connection_button(connection_state)

        self.pd_ok_dot.set_status(bool(status.get("pd_ok", False)))
        self.temp_ok_dot.set_status(bool(status.get("temperature_ok", False)))
        self._laser_enabled = bool(status.get("laser_enabled", False))
        self._stability_enabled = bool(status.get("power_stability_enabled", False))
        self.laser_enable_dot.set_status(self._laser_enabled)
        self.stability_dot.set_status(self._stability_enabled)
        self._update_laser_button()

        pd_values = status.get("pd_values", [])
        for i, label in enumerate(self.pd_labels):
            value = pd_values[i] if i < len(pd_values) else float("nan")
            label.setText(self._fmt_value(value, precision=1))

        temperatures = status.get("temperatures_c", [])
        for i, label in enumerate(self.temp_labels):
            value = temperatures[i] if i < len(temperatures) else float("nan")
            label.setText(self._fmt_value(value, precision=2))

        stage_currents = status.get("stage_currents_a", [])
        for i, label in enumerate(self.current_labels):
            if i < len(stage_currents):
                label.setText(f"{float(stage_currents[i]):.2f}")
            elif i == 0:
                label.setText(f"{float(status.get('working_current_a', 0.0)):.2f}")
            else:
                label.setText("--")

        seq_state = str(sequence.get("state", "IDLE"))
        seq_type = str(sequence.get("type") or "-")
        self.sequence_state_value.setText(f"Sequence: {seq_state} ({seq_type})")
        self.interrupt_button.setVisible(seq_state == "RUNNING")

    def closeEvent(self, event):
        self.status_timer.stop()
        return super().closeEvent(event)



def main(ip: str = "192.168.1.76", port: int = 5560):
    app = QApplication(sys.argv)
    window = PrecilaserControlGUI(ip=ip, port=port)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
