import sys
import socket
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont
import os
from pathlib import Path

from waxx.util.comms_server.waxx_server import WaxxServer

class ReadyBit:
    READY = 0
    LOADING = 1
    NOT_READY = 2
STATES = ReadyBit()

class UdpServer(QObject, WaxxServer):
    """
    A TCP server (QObject-based) that listens for connections in a QThread.
    Optionally broadcasts a UDP service-discovery beacon when server_id is given.
    """
    message_received = pyqtSignal(str)

    def __init__(self, host: str = "0.0.0.0", port: int = 0, server_id: str = None):
        super().__init__()  # QObject.__init__ (first in MRO)
        # WaxxServer.__init__ called explicitly to avoid cooperative-super MRO conflict
        if server_id is not None:
            WaxxServer.__init__(self, server_id, port)
        self.server_id = server_id
        self.host = host
        self.port = port
        self.running = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._print_connections_bool = True

    def run(self):

        self.sock.bind((self.host, self.port))
        self.port = self.sock.getsockname()[1]   # read back OS-assigned port
        self._waxx_port = self.port              # sync beacon before _start_beacon()
        self.running = True
        if self.server_id is not None:
            self._start_beacon()

        self.sock.listen(5)
        print(f"Server listening on {self.host}:{self.port}")
        while self.running:
            try:
                conn, addr = self.sock.accept()
            except socket.error:
                # accept() raised — server socket was closed (stop()) or similar.
                break
            if self._print_connections_bool:
                print(f"Connected by {addr}")
            # Per-connection timeout so a hung/slow client can never stall the
            # single-threaded accept loop (and thus every other client).
            try:
                conn.settimeout(5.0)
            except socket.error:
                pass
            try:
                # Messages are newline-framed: read until "\n", reply with a
                # newline-terminated response.  A single connection may carry
                # one or more framed messages; EOF or timeout ends it.
                buf = b""
                while True:
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        message = line.decode()
                        self.on_message_received(message)
                        reply = self.generate_reply(message)
                        conn.sendall((reply + "\n").encode())
            except socket.error as e:
                if self.running:
                    print(f"Socket error: {e}")
                # Do not break here — continue accepting new connections after transient errors.
            finally:
                conn.close()
        print("UDP Server stopped.")

    def stop(self):
        if self.server_id is not None:
            self._stop_beacon()
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