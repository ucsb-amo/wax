"""
Viewer-side client for the camera server.

Connects to the camera server's **viewer port** (command port + 1) and
receives image broadcasts, xvar updates, and run lifecycle events.
Designed to be used inside a Qt application — runs as a ``QThread`` and
emits signals when data arrives.

Typical usage::

    from waxx.util.live_od.viewer_client import ViewerClient

    client = ViewerClient(CAMERA_SERVER_IP, CAMERA_SERVER_PORT + 1)
    client.run_started.connect(on_run_started)
    client.image_received.connect(on_image)
    client.xvars_received.connect(on_xvars)
    client.run_completed.connect(on_run_complete)
    client.start()           # starts the QThread
    ...
    client.stop()
"""

import socket
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from waxx.util.live_od.protocol import send_msg, recv_msg


class ViewerClient(QThread):
    """
    QThread that connects to a camera server's viewer port and
    re-emits incoming data as Qt signals.

    Parameters
    ----------
    server_ip : str
        Camera server IP address.
    viewer_port : int
        Viewer broadcast port (typically ``command_port + 1``).
    reconnect_interval : float
        Seconds to wait before retrying after a connection failure.
    """

    # ---------- signals ----------
    run_started = pyqtSignal(dict)
    """Emitted at the start of a run with a dict containing
    ``N_img``, ``N_shots``, ``N_pwa_per_shot``, ``camera_key``,
    ``imaging_type``, ``run_id``, ``save_data``."""

    image_received = pyqtSignal(object, int)
    """Emitted for each grabbed image: ``(np.ndarray, index)``."""

    xvars_received = pyqtSignal(dict)
    """Emitted when xvar dict is forwarded from the experiment."""

    run_completed = pyqtSignal()
    """Emitted when the server reports all images captured."""

    reset_received = pyqtSignal()
    """Emitted when the server confirms a reset (grab stopped, file deleted)."""

    connection_status = pyqtSignal(bool)
    """Emitted when the connection state changes (True = connected)."""

    def __init__(self, server_ip, viewer_port, reconnect_interval=2.0):
        super().__init__()
        self.server_address = (server_ip, viewer_port)
        self.reconnect_interval = reconnect_interval
        self._running = False
        self._sock: socket.socket | None = None

    # ------------------------------------------------------------------
    #  Thread entry point
    # ------------------------------------------------------------------

    def run(self):
        """Connect to the server and listen for broadcasts.

        Automatically reconnects on failure.
        """
        self._running = True
        while self._running:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.connect(self.server_address)
                self.connection_status.emit(True)
                print(f"[ViewerClient] Connected to {self.server_address}")

                while self._running:
                    msg = recv_msg(self._sock)
                    if msg is None:
                        # Server closed the connection
                        break
                    self._handle_message(msg)

            except ConnectionRefusedError:
                self.connection_status.emit(False)
            except OSError:
                # Socket was closed externally (e.g. stop() called)
                if not self._running:
                    break
                self.connection_status.emit(False)
            except Exception as e:
                print(f"[ViewerClient] Error: {e}")
                self.connection_status.emit(False)
            finally:
                self._close_socket()

            if self._running:
                time.sleep(self.reconnect_interval)

        print("[ViewerClient] stopped.")

    # ------------------------------------------------------------------
    #  Message dispatch
    # ------------------------------------------------------------------

    def _handle_message(self, msg):
        cmd = msg.get("cmd", "")
        if cmd == "run_start":
            self.run_started.emit(msg)
        elif cmd == "image":
            self.image_received.emit(np.asarray(msg["image"]), msg["index"])
        elif cmd == "xvars":
            self.xvars_received.emit(msg.get("xvars", {}))
        elif cmd in ("run_complete", "run_incomplete"):
            self.run_completed.emit()
        elif cmd == "reset":
            self.reset_received.emit()

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def stop(self):
        """Request the client to disconnect and stop its thread."""
        self._running = False
        self._close_socket()

    def _close_socket(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    # ------------------------------------------------------------------
    #  One-shot commands (sent via command port)
    # ------------------------------------------------------------------

    @staticmethod
    def send_reset(server_ip: str, command_port: int):
        """Open a short-lived connection to the server's **command** port
        and send a ``reset`` command.

        This tells the server to stop the grab loop, close the data
        file, and delete it from disk.  The experiment's scan loop will
        detect the missing file and terminate.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((server_ip, command_port))
            send_msg(sock, {"cmd": "reset"})
            reply = recv_msg(sock)
            print(f"[ViewerClient] reset reply: {reply}")
        except Exception as e:
            print(f"[ViewerClient] Error sending reset: {e}")
        finally:
            sock.close()
