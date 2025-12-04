import sys
import socket
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont
import os
from pathlib import Path

class ReadyBit:
    READY = 0
    LOADING = 1
    NOT_READY = 2
STATES = ReadyBit()

class UdpServer(QObject):
    """
    A UDP server that listens for messages in a QThread.
    """
    message_received = pyqtSignal(str)

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.running = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def run(self):
        self.sock.bind((self.host, self.port))
        self.running = True
        print(f"Listening on {self.host}:{self.port}")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                message = data.decode('utf-8')
                self.message_received.emit(message)
            except socket.error as e:
                if self.running:
                    print(f"Socket error: {e}")
                break
        print("UDP Server stopped.")

    def stop(self):
        self.running = False
        # Unblock the socket by sending a dummy message to it
        try:
            # Create a temporary socket to send a message to the listening socket
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(b'stop', (self.host, self.port))
        except Exception as e:
            print(f"Error sending stop signal to UDP server: {e}")
        self.sock.close()