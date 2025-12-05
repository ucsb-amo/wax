import sys
import socket
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont
import os
from pathlib import Path

from waxx.util.device_state.monitor_manager import MonitorManager
from waxx.util.comms_server.comm_server import UdpServer, STATES, ReadyBit

from waxx.util.import_module_from_file import load_module_from_file

# class MonitorUDPServer(UdpServer):
#     def __init__(self, )

class MonitorServerGUI(QWidget):
    def __init__(self, monitor_server_ip, monitor_expt_path):
        super().__init__()

        self.server_ip = monitor_server_ip

        self.setWindowTitle("Monitor Server")
        self.setGeometry(100, 100, 250, 80)

        self.monitor_manager = MonitorManager(monitor_expt_path)
        self.monitor_manager.msg.connect(print) # For debugging

        self.is_ready = False
        self.setup_ui()
        self.setup_udp_server()

        self.set_status(False) # Initial status is "not ready"

        self.monitor_check_timer = QTimer(self)
        self.monitor_check_timer.setInterval(125)
        self.monitor_check_timer.timeout.connect(self.check_monitor_status)
        self.monitor_check_timer.start()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.status_indicator = QPushButton("NOT READY")
        self.status_indicator.clicked.connect(self.on_status_clicked)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.status_indicator.setFont(font)
        layout.addWidget(self.status_indicator)
        self.setLayout(layout)

    def setup_udp_server(self):
        self.server_thread = QThread()
        
        self.udp_server = UdpServer(self.server_ip, 6789)
        self.udp_server.moveToThread(self.server_thread)

        self.server_thread.started.connect(self.udp_server.run)
        self.udp_server.message_received.connect(self.handle_message)
        
        self.server_thread.start()

    def on_status_clicked(self):
        if self.status_indicator.text() == "READY":
            reply = QMessageBox.question(self, 'Restart Monitor',
                                         "Are you sure you'd like to restart the monitor experiment?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                print("Manual monitor restart triggered.")
                self.monitor_manager.start()
                self.set_status(STATES.LOADING)
        elif self.status_indicator.text() == "NOT READY":
            print("Manual monitor start triggered.")
            self.monitor_manager.start()

    def handle_message(self, message):
        print(f"Message received: {message}")
        if "run complete" in message:
            print("Run complete message received. Restarting monitor.")
            self.monitor_manager.start()
            self.set_status(STATES.LOADING)
        elif "monitor ready" in message:
            print("Monitor ready message received.")
            self.set_status(STATES.READY)

    def set_status(self, status: ReadyBit):
        if status == STATES.READY:
            self.status_indicator.setText("READY")
            self.status_indicator.setStyleSheet("background-color: green; color: white;")
            # self.status_indicator.setEnabled(False)
        elif status == STATES.NOT_READY:
            self.status_indicator.setText("NOT READY")
            self.status_indicator.setStyleSheet("background-color: red; color: white;")
            # self.status_indicator.setEnabled(True)
        else:
            self.status_indicator.setText("Loading...")
            self.status_indicator.setStyleSheet("background-color: orange; color: white;")

    def check_monitor_status(self):
        if self.monitor_manager.isRunning() and self.status_indicator.text() != "READY":
            self.set_status(STATES.LOADING)
        elif not self.monitor_manager.isRunning():
            self.set_status(STATES.NOT_READY)
        else:
            self.set_status(STATES.READY)
        
    def closeEvent(self, event):
        print("Closing GUI...")
        self.udp_server.stop()
        self.server_thread.quit()
        self.server_thread.wait()
        self.monitor_manager.terminate()
        self.monitor_manager.wait()
        event.accept()