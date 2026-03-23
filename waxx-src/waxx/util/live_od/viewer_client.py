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

    available_data_fields_received = pyqtSignal(list)
    """Emitted with the latest list of available live data-field names."""

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
        self._connected = None

    def _set_connection_status(self, connected: bool):
        connected = bool(connected)
        if self._connected is connected:
            return
        self._connected = connected
        self.connection_status.emit(connected)

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
                self._sock.settimeout(1.0)
                self._sock.connect(self.server_address)
                self._set_connection_status(True)
                print(f"[ViewerClient] Connected to {self.server_address}")

                while self._running:
                    try:
                        msg = recv_msg(self._sock)
                    except socket.timeout:
                        continue
                    if msg is None:
                        # Server closed the connection
                        break
                    self._handle_message(msg)

            except ConnectionRefusedError:
                self._set_connection_status(False)
            except socket.timeout:
                self._set_connection_status(False)
            except OSError:
                # Socket was closed externally (e.g. stop() called)
                if not self._running:
                    break
                self._set_connection_status(False)
            except Exception as e:
                print(f"[ViewerClient] Error: {e}")
                self._set_connection_status(False)
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
            self.available_data_fields_received.emit(
                list(msg.get("available_data_fields", []))
            )
            self.run_started.emit(msg)
        elif cmd == "image":
            self.image_received.emit(np.asarray(msg["image"]), msg["index"])
        elif cmd == "xvars":
            self.available_data_fields_received.emit(
                list(msg.get("available_data_fields", []))
            )
            payload = dict(msg.get("xvars", {}))
            payload.update(msg.get("data_fields", {}))
            self.xvars_received.emit(payload)
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
        if self._running:
            self._set_connection_status(False)

    # ------------------------------------------------------------------
    #  One-shot commands (sent via command port)
    # ------------------------------------------------------------------

    @staticmethod
    def send_reset(server_ip: str, command_port: int):
        """Open a short-lived connection to the server's **command** port
        and send a ``reset`` command.

        This tells the server to stop the grab loop, close the data
        file, and (for in-progress runs) delete it from disk.  If the
        run has already completed, the file is preserved.
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

    @staticmethod
    def get_status(server_ip: str, command_port: int) -> dict | None:
        """Query the camera server command port for current status."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((server_ip, command_port))
            send_msg(sock, {"cmd": "status"})
            reply = recv_msg(sock)
            return reply if isinstance(reply, dict) else None
        except Exception as e:
            print(f"[ViewerClient] Error getting status: {e}")
            return None
        finally:
            sock.close()

    @staticmethod
    def get_logs(server_ip: str, command_port: int, since: int = 0, limit: int = 10000) -> dict | None:
        """Fetch timestamped server logs from the command port.

        Returns a dict with keys: ``entries`` (list), ``next_index`` and
        ``total_count``. Returns ``None`` on communication failure.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((server_ip, command_port))
            send_msg(sock, {"cmd": "get_logs", "since": int(since), "limit": int(limit)})
            reply = recv_msg(sock)
            if isinstance(reply, dict) and reply.get("cmd") == "logs":
                return reply
            return None
        except Exception as e:
            print(f"[ViewerClient] Error getting logs: {e}")
            return None
        finally:
            sock.close()
