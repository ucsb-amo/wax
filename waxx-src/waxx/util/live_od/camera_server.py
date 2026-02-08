"""
Camera acquisition server.

Manages camera connections, image grab loops, data saving, and broadcasts
images/metadata to viewer clients. Subclasses UdpServer for the command
channel and adds a separate viewer broadcast channel.

Command port  : ``port``     – accepts experiment client connections (one at a time)
Viewer port   : ``port + 1`` – accepts viewer client connections (multiple, persistent)

Protocol
--------
All messages are length-prefixed pickle dicts (see ``protocol.py``).

Experiment → Server commands:
    new_run          – send camera_params & run info; server connects camera,
                       starts grab loop, and replies camera_ready
    xvars            – send xvar dict; server forwards to viewers
    run_complete     – experiment is done
    status           – query server state

Server → Viewer broadcasts:
    run_start     – run metadata (N_img, camera_key, imaging_type, etc.)
    image         – grabbed image + index
    xvars         – forwarded xvar dict
    run_complete  – all images captured
"""

import socket
import threading
import time
import numpy as np
from queue import Queue, Empty

from PyQt6.QtCore import QObject, pyqtSignal

from waxx.util.comms_server.comm_server import UdpServer
from waxx.util.live_od.protocol import send_msg, recv_msg
from waxa.data.increment_run_id import update_run_id


class CameraServer(UdpServer):
    """
    Camera acquisition server.

    Parameters
    ----------
    host : str
        IP address to bind.
    port : int
        Command port.  The viewer port is ``port + 1``.
    camera_nanny : object, optional
        A ``CameraNanny`` instance (or compatible) that provides
        ``persistent_get_camera`` and ``update_params``.  Pass ``None``
        to create a default instance at runtime (requires kexp on the path).
    """

    # Qt signals for optional GUI integration
    camera_status_signal = pyqtSignal(int)
    run_started_signal = pyqtSignal(dict)
    run_completed_signal = pyqtSignal()
    image_grabbed_signal = pyqtSignal(object, int)  # (np.ndarray, index)
    log_signal = pyqtSignal(str)  # human-readable status messages

    def __init__(self, host, port, camera_nanny=None):
        super().__init__(host, port)
        self.viewer_port = port + 1

        self.camera_nanny = camera_nanny

        # ---- state ----
        self._run_info = {}
        self._camera = None
        self._camera_ready = False
        self._grab_active = False
        self._interrupted = False
        self._data_file = None
        self._data_filepath = ""
        self._data_file_lock = threading.Lock()

        # ---- queues ----
        self._image_queue = Queue()

        # ---- viewer connections (protected by lock) ----
        self._viewer_lock = threading.Lock()
        self._viewer_connections: list[socket.socket] = []

        # ---- viewer listener socket ----
        self._viewer_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._viewer_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """Start listening on command and viewer ports (blocking)."""
        # Viewer listener runs in its own thread
        self._viewer_thread = threading.Thread(
            target=self._viewer_listener, daemon=True
        )
        self._viewer_thread.start()

        # Command listener (blocks in accept loop)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        self.running = True
        self._log(f"Listening — command port: {self.host}:{self.port}")
        self._log(f"Listening — viewer  port: {self.host}:{self.viewer_port}")

        while self.running:
            try:
                conn, addr = self.sock.accept()
                self._log(f"Client connected: {addr}")
                # Handle each connection in its own thread so that a
                # reset from a viewer can be processed even while an
                # experiment connection is active.
                t = threading.Thread(
                    target=self._handle_experiment_connection,
                    args=(conn,),
                    daemon=True,
                )
                t.start()
            except socket.error as e:
                if self.running:
                    self._log(f"Socket error: {e}")
                break

        self._log("Server stopped.")

    # ------------------------------------------------------------------
    #  Viewer listener
    # ------------------------------------------------------------------

    def _viewer_listener(self):
        """Accept viewer connections on ``self.viewer_port``."""
        self._viewer_sock.bind((self.host, self.viewer_port))
        self._viewer_sock.listen(10)
        while self.running:
            try:
                conn, addr = self._viewer_sock.accept()
                self._log(f"Viewer client connected: {addr}")
                with self._viewer_lock:
                    self._viewer_connections.append(conn)
            except socket.error:
                if self.running:
                    continue
                break

    # ------------------------------------------------------------------
    #  Experiment command handler  (one persistent connection per run)
    # ------------------------------------------------------------------

    def _handle_experiment_connection(self, conn):
        """Handle a full experiment run on *conn*."""
        try:
            while self.running:
                msg = recv_msg(conn)
                if msg is None:
                    break

                cmd = msg.get("cmd", "")
                self._log(f"<< command: {cmd}")

                if cmd == "new_run":
                    reply = self._handle_new_run(msg)
                elif cmd == "xvars":
                    reply = self._handle_xvars(msg)
                elif cmd == "run_complete":
                    reply = self._handle_run_complete()
                    send_msg(conn, reply)
                    break  # end of run → close connection
                elif cmd == "reset":
                    reply = self._handle_reset()
                    send_msg(conn, reply)
                    break  # connection ends after reset
                elif cmd == "status":
                    reply = {
                        "cmd": "status",
                        "camera_ready": self._camera_ready,
                        "grab_active": self._grab_active,
                    }
                else:
                    reply = {"cmd": "error", "message": f"Unknown command: {cmd}"}

                send_msg(conn, reply)

        except Exception as e:
            self._log(f"Error in experiment handler: {e}")
        finally:
            conn.close()
            self._log("Experiment client disconnected.")

    # ---- command handlers --------------------------------------------------

    def _handle_new_run(self, msg):
        """Connect camera, start grab loop, reply ``camera_ready``.

        The experiment blocks on ``send_new_run()`` until this reply is
        sent, so by the time the experiment proceeds the camera is
        connected and actively waiting for triggers.
        """
        self._run_info = msg
        camera_params = msg["camera_params"]

        self.camera_status_signal.emit(0)  # baby born
        self._log("New run requested — connecting camera...")

        self._camera = self.camera_nanny.persistent_get_camera(camera_params)
        self.camera_nanny.update_params(self._camera, camera_params)

        self.camera_status_signal.emit(1)  # cam found

        if self._camera is not None and self._camera.is_opened():
            self._camera_ready = True

            camera_key = camera_params.key
            if isinstance(camera_key, bytes):
                camera_key = camera_key.decode()

            run_id = msg.get("run_id", 0)
            N_img = msg["N_img"]
            save_data = msg.get("save_data", True)
            self._log(
                f"Camera connected: {camera_key}  |  "
                f"Run ID: {run_id}  |  "
                f"N_img: {N_img}  |  "
                f"save_data: {save_data}"
            )

            # Notify viewer clients about the new run
            run_start_info = {
                "cmd": "run_start",
                "N_img": N_img,
                "N_shots": msg["N_shots"],
                "N_pwa_per_shot": msg["N_pwa_per_shot"],
                "camera_key": camera_key,
                "imaging_type": msg.get("imaging_type", False),
                "run_id": run_id,
                "save_data": save_data,
            }
            self._broadcast_to_viewers(run_start_info)
            self.run_started_signal.emit(run_start_info)

            # Start grab loop so the camera is ready for triggers
            self._start_grab()
            self.camera_status_signal.emit(3)  # grab running
            self._log("Grab loop started — camera is ready for triggers.")

            return {"cmd": "camera_ready"}
        else:
            self._log("ERROR: Failed to connect camera.")
            return {"cmd": "error", "message": "Failed to connect camera"}

    def _start_grab(self):
        """Start grab loop and image-processing threads."""
        self._image_queue = Queue()
        self._grab_active = True
        self._interrupted = False

        threading.Thread(target=self._run_grab, daemon=True).start()
        threading.Thread(target=self._process_images, daemon=True).start()

    def _handle_xvars(self, msg):
        """Forward xvars to all connected viewers."""
        xvars = msg.get("xvars", {})
        parts = ", ".join(f"{k}={v}" for k, v in xvars.items())
        self._log(f"xvars received: {parts}")
        self._broadcast_to_viewers({"cmd": "xvars", "xvars": xvars})
        return {"cmd": "ack"}

    def _handle_run_complete(self):
        """Experiment signalled run complete."""
        self._log("Experiment signalled run complete.")
        self.camera_status_signal.emit(-1)
        try:
            update_run_id()
            self._log("Run ID advanced.")
        except Exception as e:
            self._log(f"Error advancing run ID: {e}")
        return {"cmd": "ack"}

    def _handle_reset(self):
        """Stop grab loop, close and delete the data file.

        The experiment's scan loop checks for the data file each
        iteration and will terminate when it finds it missing.
        """
        import os

        self._log("RESET requested — stopping grab loop.")
        self._interrupted = True

        # Close the HDF5 data file
        with self._data_file_lock:
            if self._data_file is not None:
                try:
                    self._data_file.close()
                    self._log("Data file closed.")
                except Exception as e:
                    self._log(f"Error closing data file: {e}")
                self._data_file = None

        # Delete the data file from disk
        if self._data_filepath:
            try:
                if os.path.exists(self._data_filepath):
                    os.remove(self._data_filepath)
                    self._log(f"Data file deleted: {self._data_filepath}")
                else:
                    self._log(f"Data file already absent: {self._data_filepath}")
            except Exception as e:
                self._log(f"Error deleting data file: {e}")

        self._camera_ready = False
        self._grab_active = False
        self.camera_status_signal.emit(-1)
        self._broadcast_to_viewers({"cmd": "reset"})
        try:
            update_run_id()
            self._log("Run ID advanced.")
        except Exception as e:
            self._log(f"Error advancing run ID: {e}")
        self._log("Reset complete.")
        return {"cmd": "ack"}

    # ------------------------------------------------------------------
    #  Grab loop & image processing
    # ------------------------------------------------------------------

    def _run_grab(self):
        """Block while camera acquires images, putting them in the queue."""
        N_img = int(self._run_info["N_img"])
        try:
            self._camera.start_grab(
                N_img,
                output_queue=self._image_queue,
                check_interrupt_method=lambda: self._interrupted,
            )
        except Exception as e:
            self._log(f"Grab error: {e}")
        finally:
            self._grab_active = False
            self._log("Grab loop finished.")

    def _process_images(self):
        """Read grabbed images from the queue, save to disk, broadcast."""
        N_img = int(self._run_info["N_img"])
        data_filepath = self._run_info.get("data_filepath", "")
        save_data = self._run_info.get("save_data", True)

        self._data_filepath = data_filepath
        if save_data and data_filepath:
            try:
                with self._data_file_lock:
                    self._data_file = self._open_data_file(data_filepath)
                self._log(f"Data file opened: {data_filepath}")
            except Exception as e:
                self._log(f"Error opening data file: {e}")

        count = 0
        while count < N_img and not self._interrupted:
            try:
                img, _, idx = self._image_queue.get(timeout=1.0)
                img_t = time.time()

                # Save to HDF5
                with self._data_file_lock:
                    if save_data and self._data_file is not None:
                        try:
                            self._data_file["data"]["images"][idx] = img
                            self._data_file["data"]["image_timestamps"][idx] = img_t
                            self._log(f"Image saved {idx + 1}/{N_img}")
                        except Exception as e:
                            self._log(f"Error saving image {idx + 1}: {e}")

                # Broadcast to viewers
                self._broadcast_to_viewers(
                    {"cmd": "image", "image": np.asarray(img), "index": idx}
                )
                self.image_grabbed_signal.emit(np.asarray(img), idx)
                count += 1

            except Empty:
                if not self._grab_active:
                    break
                continue

        # Close data file
        with self._data_file_lock:
            if self._data_file is not None:
                try:
                    self._data_file.close()
                except Exception:
                    pass
                self._data_file = None

        # Notify viewers
        if count >= N_img:
            self._log(f"All {N_img} images captured — run complete.")
            self._broadcast_to_viewers({"cmd": "run_complete"})
            self.run_completed_signal.emit()
            self.camera_status_signal.emit(-1)
        else:
            self._log(f"Grab incomplete: {count}/{N_img} images.")
            self._broadcast_to_viewers(
                {"cmd": "run_incomplete", "count": count, "total": N_img}
            )

    def _open_data_file(self, filepath, timeout=30.0, check_period=0.5):
        """Open an HDF5 data file, retrying until available."""
        import h5py

        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                return h5py.File(filepath, "r+")
            except Exception:
                time.sleep(check_period)
        raise TimeoutError(f"Could not open data file: {filepath}")

    # ------------------------------------------------------------------
    #  Viewer broadcast
    # ------------------------------------------------------------------

    def _broadcast_to_viewers(self, msg):
        """Send *msg* to every connected viewer; prune dead connections."""
        with self._viewer_lock:
            dead = []
            for conn in self._viewer_connections:
                try:
                    send_msg(conn, msg)
                except Exception:
                    dead.append(conn)
            for conn in dead:
                try:
                    conn.close()
                except Exception:
                    pass
                self._viewer_connections.remove(conn)

    # ------------------------------------------------------------------
    #  Logging helper
    # ------------------------------------------------------------------

    def _log(self, msg):
        """Print and emit a human-readable log message."""
        print(f"[CameraServer] {msg}")
        try:
            self.log_signal.emit(msg)
        except RuntimeError:
            pass  # signal not connected or object destroyed

    # ------------------------------------------------------------------
    #  Shutdown
    # ------------------------------------------------------------------

    def stop(self):
        """Shut down the server and release all resources."""
        self._interrupted = True
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self._viewer_sock.close()
        except Exception:
            pass
        with self._viewer_lock:
            for conn in self._viewer_connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._viewer_connections.clear()
        if self.camera_nanny is not None:
            try:
                self.camera_nanny.close_all()
            except Exception:
                pass
