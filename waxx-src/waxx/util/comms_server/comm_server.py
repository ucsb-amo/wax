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
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def run(self):

        self.sock.bind((self.host, self.port))
        self.running = True

        self.sock.listen(5)
        print(f"Server listening on {self.host}:{self.port}")
        while self.running:
            conn, addr = self.sock.accept()
            print(f"Connected by {addr}")
            try:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    message = data.decode()
                    self.on_message_received(message)
                    reply = self.generate_reply(message)
                    conn.sendall(reply.encode())
            except socket.error as e:
                if self.running:
                    print(f"Socket error: {e}")
                break
            finally:
                conn.close()
        print("UDP Server stopped.")

    def stop(self):
        self.running = False
        # Unblock the socket by sending a dummy message to it
        try:
            # Create a temporary socket to send a message to the listening socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.sendall(b'stop')
        except Exception as e:
            print(f"Error sending stop signal to UDP server: {e}")
        self.sock.close()

    def on_message_received(self, message):
        pass

    def generate_reply(self, message):
        return f'Server received {message}'