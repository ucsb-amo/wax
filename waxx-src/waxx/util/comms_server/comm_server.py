import sys
import socket
import threading
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont
import os
from pathlib import Path

from beacon.discovery.server import NetServer
# the constants live in a module that imports nothing networked; the same
# objects are re-exported here (comm_server.STATES, comm_server.ReadyBit)
from waxx.util.device_state.ready_bit import ReadyBit, STATES  # noqa: F401

class UdpServer(QObject, NetServer):
    """
    A TCP server (QObject-based) that listens for connections in a QThread.
    Optionally broadcasts a UDP service-discovery beacon when server_id is given.

    Each accepted connection is served on a thread of its own (at most
    :attr:`MAX_CONNECTIONS` at once), so a client that connects and then sends
    nothing holds up only itself, for up to :attr:`CONN_TIMEOUT_S`.  (Until
    2026-11-16 the accept loop served one connection at a time with a backlog
    of 5: one such client stalled every other for 5 s and the queued clients
    were refused.)  ``on_message_received`` / ``generate_reply`` are therefore
    called from several threads at once; a subclass serializes what must not
    interleave.
    """
    message_received = pyqtSignal(str)

    #: Connections served at once; past this the accept loop waits for a free
    #: slot (new clients wait in the listen backlog).
    MAX_CONNECTIONS = 32
    #: Seconds a connection may go without sending anything before it is closed.
    CONN_TIMEOUT_S = 5.0

    def __init__(self, host: str = "0.0.0.0", port: int = 0, server_id: str = None):
        super().__init__()  # QObject.__init__ (first in MRO)
        # NetServer.__init__ called explicitly to avoid cooperative-super MRO conflict
        if server_id is not None:
            NetServer.__init__(self, server_id, port)
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

        self.sock.listen(socket.SOMAXCONN)
        print(f"Server listening on {self.host}:{self.port}")
        slots = threading.BoundedSemaphore(self.MAX_CONNECTIONS)
        while self.running:
            try:
                conn, addr = self.sock.accept()
            except socket.error:
                # accept() raised — server socket was closed (stop()) or similar.
                break
            if self._print_connections_bool:
                print(f"Connected by {addr}")
            while not slots.acquire(timeout=0.5):
                if not self.running:
                    break
            else:
                try:
                    threading.Thread(target=self._serve_connection, args=(conn, slots),
                                     name=f"{type(self).__name__}-conn", daemon=True).start()
                except RuntimeError:
                    # no thread to be had: serve it here, as before
                    self._serve_connection(conn, slots)
                continue
            conn.close()                          # stopping while every slot was busy
        print("UDP Server stopped.")

    def _serve_connection(self, conn, slots) -> None:
        """One connection, on its own thread; releases its slot when done."""
        try:
            # Per-connection timeout: a hung/slow client ends after
            # CONN_TIMEOUT_S without data and holds up nobody else meanwhile.
            try:
                conn.settimeout(self.CONN_TIMEOUT_S)
            except socket.error:
                pass
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
        except Exception as e:                    # noqa: BLE001
            # a handler that raised: this connection ends, the server goes on
            print(f"Request handling failed: {e!r}")
        finally:
            try:
                conn.close()
            finally:
                slots.release()

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