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
import random
import numpy as np
from queue import Queue, Empty

from PyQt6.QtCore import QObject, pyqtSignal

from waxa.data.server_talk import server_talk as st

from waxx.util.comms_server.comm_server import UdpServer
from waxx.util.live_od.protocol import send_msg, recv_msg


_CAMERA_BABY_NAMES = [
    "Mochi", "Pip", "Noodle", "Sprout", "Pickle", "Biscuit", "Pebble", "Miso",
    "Comet", "Pixel", "Nova", "Jellybean", "Clover", "Poppy", "Tofu", "Bean",
    "Luna", "Cosmo", "Dumpling", "Sunny", "Kiwi", "Maple", "Muffin", "Blue",
]

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
    xvars_signal = pyqtSignal(dict)  # forwarded xvar values

    def __init__(self, host, port,
                camera_nanny=None,
                server_talk = None):
        super().__init__(host, port)
        self.viewer_port = port + 1

        if server_talk == None:
            server_talk = st()
        else:
            server_talk = server_talk
        self.server_talk = server_talk


        self.camera_nanny = camera_nanny

        # ---- state ----
        self._run_info = {}
        self._run_start_info = None  # Store run_start info for late-connecting viewers
        self._run_complete_notified = False
        self._available_data_fields: list[str] = []
        self._last_data_fields: dict = {}
        self._run_shot_count = 0  # Track shots received from experiment (reset each run)
        self._run_shot_history: list[dict] = []
        self._run_image_history: list[dict] = []
        self._camera_baby_name: str = ""
        self._camera = None
        self._camera_ready = False
        self._grab_active = False
        self._image_processing_complete = False  # Tracks when all images are processed
        self._interrupted = False
        self._data_file = None
        self._data_filepath = ""
        self._data_file_lock = threading.Lock()

        # ---- queues ----
        self._image_queue = Queue()

        # ---- viewer connections (protected by lock) ----
        self._viewer_lock = threading.Lock()
        self._viewer_connections: list[socket.socket] = []
        self._experiment_lock = threading.Lock()
        self._experiment_connections: list[socket.socket] = []
        self._experiment_threads: list[threading.Thread] = []

        # ---- log history ----
        self._log_lock = threading.Lock()
        self._log_history: list[dict] = []

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
                # Handle each connection in its own thread so that a
                # reset from a viewer can be processed even while an
                # experiment connection is active.
                t = threading.Thread(
                    target=self._handle_experiment_connection,
                    args=(conn,),
                    daemon=True,
                )
                with self._experiment_lock:
                    self._experiment_threads.append(t)
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
        try:
            self._viewer_sock.bind((self.host, self.viewer_port))
            self._viewer_sock.listen(10)
            self._log(f"Viewer listener started on {self.host}:{self.viewer_port}")
            while self.running:
                try:
                    conn, addr = self._viewer_sock.accept()
                    with self._viewer_lock:
                        self._viewer_connections.append(conn)
                    
                    # If a run is currently active, send the run_start info to the new viewer
                    if self._run_start_info is not None:
                        try:
                            send_msg(conn, self._run_start_info)
                            self._log(f"Sent run_start info to {addr}")
                        except Exception as e:
                            self._log(f"Error sending run_start info to {addr}: {e}")
                except socket.error:
                    if self.running:
                        continue
                    break
        except Exception as e:
            self._log(f"Viewer listener error: {e}")

    # ------------------------------------------------------------------
    #  Experiment command handler  (one persistent connection per run)
    # ------------------------------------------------------------------

    def _handle_experiment_connection(self, conn):
        """Handle a full experiment run on *conn*."""
        try:
            peer = conn.getpeername()
        except Exception:
            peer = ("?", "?")
        with self._experiment_lock:
            self._experiment_connections.append(conn)
        try:
            while self.running:
                msg = recv_msg(conn)
                if msg is None:
                    break

                cmd = msg.get("cmd", "")
                if cmd != "status":
                    self._log(f"<< command from {peer}: {cmd}")

                if cmd == "new_run":
                    reply = self._handle_new_run(msg, peer=peer)
                elif cmd == "xvars":
                    reply = self._handle_xvars(msg)
                elif cmd == "shot_done":
                    reply = self._handle_shot_done()
                elif cmd == "run_complete":
                    reply = self._handle_run_complete()
                    self._send_reply_and_disconnect(conn, reply)
                    return
                elif cmd == "reset":
                    reply = self._handle_reset()
                    self._send_reply_and_disconnect(conn, reply)
                    return
                elif cmd == "status":
                    try:
                        next_run_id = self.server_talk.get_run_id()
                    except Exception:
                        next_run_id = self._run_info.get("run_id", 0)
                    active_run_id = self._run_info.get("run_id", next_run_id)
                    reply = {
                        "cmd": "status",
                        "camera_ready": self._camera_ready,
                        "grab_active": self._grab_active,
                        "run_id": active_run_id if self._run_start_info is not None else next_run_id,
                        "active_run_id": active_run_id,
                        "next_run_id": next_run_id,
                    }
                elif cmd in ("get_logs", "logs"):
                    since = int(msg.get("since", 0))
                    limit = int(msg.get("limit", 10000))
                    with self._log_lock:
                        total = len(self._log_history)
                        start = max(0, min(since, total))
                        end = min(total, start + max(1, limit))
                        entries = [dict(entry) for entry in self._log_history[start:end]]
                    reply = {
                        "cmd": "logs",
                        "entries": entries,
                        "next_index": end,
                        "total_count": total,
                    }
                else:
                    reply = {"cmd": "error", "message": f"Unknown command: {cmd}"}

                send_msg(conn, reply)

        except Exception as e:
            self._log(f"Error in experiment handler: {e}")
        finally:
            with self._experiment_lock:
                if conn in self._experiment_connections:
                    self._experiment_connections.remove(conn)
                current = threading.current_thread()
                if current in self._experiment_threads:
                    self._experiment_threads.remove(current)
            try:
                conn.close()
            except Exception:
                pass

    # ---- command handlers --------------------------------------------------

    def _handle_new_run(self, msg, peer=None):
        """Connect camera, start grab loop, reply ``camera_ready``.

        The experiment blocks on ``send_new_run()`` until this reply is
        sent, so by the time the experiment proceeds the camera is
        connected and actively waiting for triggers.
        """
        run_id = msg.get("run_id", 0)

        current_run_id = self._run_info.get("run_id", None)
        if current_run_id == run_id:
            self._log(
                f"Ignoring duplicate new_run for run_id={run_id} from {peer}; "
                "current run data is still retained in memory."
            )
            return {"cmd": "camera_ready", "duplicate": True}

        self._run_info = msg
        self._run_complete_notified = False
        self._image_processing_complete = False  # Reset for new run
        self._run_shot_count = 0  # Reset shot counter for new run
        self._available_data_fields = []
        self._last_data_fields = {}
        self._run_shot_history = []
        self._run_image_history = []
        self._camera_baby_name = random.choice(_CAMERA_BABY_NAMES)
        setup_camera = bool(msg.get("setup_camera", True))
        camera_params = msg["camera_params"]

        self.camera_status_signal.emit(0)  # baby born
        self._log(f"Camera baby {self._camera_baby_name} is born for run {run_id}.")

        save_data = msg.get("save_data", True)
        N_img = int(msg["N_img"])
        N_shots = int(msg["N_shots"])

        if not setup_camera:
            self._camera = None
            self._camera_ready = True
            self._grab_active = False
            self._image_processing_complete = True  # No grab loop, so mark as complete
            self._log(
                f"Run ID: {run_id} | setup_camera=False | save_data: {save_data}"
            )

            run_start_info = {
                "cmd": "run_start",
                "N_img": 0,
                "N_shots": N_shots,
                "N_pwa_per_shot": msg["N_pwa_per_shot"],
                "camera_key": "",
                "setup_camera": False,
                "imaging_type": msg.get("imaging_type", False),
                "run_id": run_id,
                "save_data": save_data,
                "available_data_fields": [],
                "pixel_size_m": getattr(camera_params, "pixel_size_m", 0.0),
                "magnification": getattr(camera_params, "magnification", 1.0),
            }
            self._run_start_info = run_start_info
            self._broadcast_to_viewers(run_start_info)
            self.run_started_signal.emit(run_start_info)

            return {"cmd": "camera_ready"}

        self._log("New run requested — connecting camera...")

        self._camera = self.camera_nanny.persistent_get_camera(camera_params)
        self.camera_nanny.update_params(self._camera, camera_params)

        self.camera_status_signal.emit(1)  # cam found

        if self._camera is not None and self._camera.is_opened():
            self._camera_ready = True

            camera_key = camera_params.key
            if isinstance(camera_key, bytes):
                camera_key = camera_key.decode()

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
                "setup_camera": True,
                "imaging_type": msg.get("imaging_type", False),
                "run_id": run_id,
                "save_data": save_data,
                "available_data_fields": [],
                "pixel_size_m": getattr(camera_params, "pixel_size_m", 0.0),
                "magnification": getattr(camera_params, "magnification", 1.0),
            }
            # Store for late-connecting viewers
            self._run_start_info = run_start_info
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
        data_fields = msg.get("data_fields", {})
        self._last_data_fields = dict(data_fields)
        if data_fields:
            field_names = list(data_fields.keys())
            if field_names != self._available_data_fields:
                self._available_data_fields = field_names
                if self._run_start_info is not None:
                    self._run_start_info["available_data_fields"] = list(self._available_data_fields)
        parts = ", ".join(f"{k}={v}" for k, v in xvars.items())
        self._log(f"xvars received: {parts}")
        shot = {
            "shot_index": len(self._run_shot_history),
            "timestamp": time.time(),
            "xvars": dict(xvars),
            "data_fields": dict(data_fields),
        }
        for key, value in xvars.items():
            shot[f"xvar.{key}"] = value
        for key, value in data_fields.items():
            shot[f"xvar.{key}"] = value
        self._run_shot_history.append(shot)
        self._broadcast_to_viewers(
            {
                "cmd": "xvars",
                "xvars": xvars,
                "data_fields": data_fields,
                "available_data_fields": list(self._available_data_fields),
            }
        )
        try:
            self.xvars_signal.emit(dict(xvars))
        except RuntimeError:
            pass
        return {"cmd": "ack"}

    def _handle_shot_done(self):
        """Experiment signalled a shot is complete (independent shot counter)."""
        self._run_shot_count += 1
        N_shots = int(self._run_info.get("N_shots", 0))
        if N_shots > 0:
            self._log(f"Shot {self._run_shot_count}/{N_shots} complete.")
        return {"cmd": "ack"}

    def _handle_run_complete(self):
        """Experiment signalled run complete -- wait for grab and image processing."""
        self._log("Experiment signalled run complete — waiting for grab loop and image processing...")
        self._wait_for_grab_complete()
        self._log("Grab loop and image processing complete.")
        if self._camera_baby_name:
            self._log(f"Run complete. {self._camera_baby_name} has retired honorably.")
        
        self._run_start_info = None  # Clear run info since run is complete
        self.camera_status_signal.emit(-1)
        if not self._run_complete_notified:
            self._broadcast_to_viewers({"cmd": "run_complete"})
            self.run_completed_signal.emit()
            self._run_complete_notified = True
        try:
            self.server_talk.update_run_id()
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

        run_was_active = self._run_start_info is not None
        self._log("RESET requested — stopping grab loop.")
        if self._camera_baby_name:
            self._log(f"Reset received. {self._camera_baby_name} had a rough day.")
        self._interrupted = True
        self._run_start_info = None  # Clear run info since run is being reset

        # Close the HDF5 data file
        with self._data_file_lock:
            if self._data_file is not None:
                try:
                    self._data_file.close()
                    self._log("Data file closed.")
                except Exception as e:
                    self._log(f"Error closing data file: {e}")
                self._data_file = None

        # Delete the data file only for in-progress runs.
        # If the run already completed, keep the finished dataset on reset.
        if run_was_active:
            if self._data_filepath:
                try:
                    if os.path.exists(self._data_filepath):
                        os.remove(self._data_filepath)
                        self._log(f"Data file deleted: {self._data_filepath}")
                    else:
                        self._log(f"Data file already absent: {self._data_filepath}")
                except Exception as e:
                    self._log(f"Error deleting data file: {e}")
        else:
            self._log("Run already complete; data file preserved on reset.")

        self._camera_ready = False
        self._grab_active = False
        self.camera_status_signal.emit(-1)
        self._broadcast_to_viewers({"cmd": "reset"})
        try:
            self.server_talk.update_run_id()
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
        self._save_data = save_data

        if save_data and data_filepath:
            try:
                with self._data_file_lock:
                    self._data_file = self._open_data_file(data_filepath)
                self._log(f"Data file opened: {data_filepath}")
            except Exception as e:
                self._log(f"Error opening data file: {e}")
        elif not save_data:
            self._log("save_data=False — images will be broadcast but NOT saved.")

        count = 0
        while count < N_img and not self._interrupted:
            try:
                img, _, idx = self._image_queue.get(timeout=1.0)
                img_t = time.time()

                # Save to HDF5 (only when save_data is True)
                if save_data:
                    with self._data_file_lock:
                        if self._data_file is not None:
                            try:
                                self._data_file["data"]["images"][idx] = img
                                self._data_file["data"]["image_timestamps"][idx] = img_t
                                self._log(f"Image saved {idx + 1}/{N_img}")
                            except Exception as e:
                                self._log(f"Error saving image {idx + 1}: {e}")

                # Broadcast to viewers (always, regardless of save_data)
                self._broadcast_to_viewers(
                    {"cmd": "image", "image": np.asarray(img), "index": idx}
                )
                self._run_image_history.append(
                    {"image": np.asarray(img), "index": idx, "timestamp": img_t}
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

        # Mark image processing as complete
        self._image_processing_complete = True

        # Notify viewers
        if count >= N_img:
            self._log(f"All {N_img} images captured — run complete.")
            self._broadcast_to_viewers({"cmd": "run_complete"})
            self.run_completed_signal.emit()
            self._run_complete_notified = True
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

    def _wait_for_grab_complete(self, timeout=300):
        """Wait for grab loop and image processing to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._image_processing_complete:
                return
            time.sleep(0.1)
        self._log(f"WARNING: Image processing did not complete within {timeout}s timeout")

    def _send_reply_and_disconnect(self, conn, reply):
        """Send a final reply, then half-close the socket so the client sees EOF."""
        try:
            send_msg(conn, reply)
        except Exception as e:
            self._log(f"Error sending final reply before disconnect: {e}")
        self._terminate_experiment_connection(conn)

    def _terminate_experiment_connection(self, conn):
        """Gracefully terminate the per-run command connection so its handler thread exits."""
        try:
            conn.shutdown(socket.SHUT_WR)
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Logging helper
    # ------------------------------------------------------------------

    def _log(self, msg):
        """Print and emit a human-readable log message."""
        now = time.time()
        with self._log_lock:
            entry = {
                "index": len(self._log_history),
                "timestamp": now,
                "message": str(msg),
            }
            self._log_history.append(entry)
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
        with self._experiment_lock:
            for conn in self._experiment_connections:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
            self._experiment_connections.clear()
        if self.camera_nanny is not None:
            try:
                self.camera_nanny.close_all()
            except Exception:
                pass

    def get_run_shot_history(self):
        """Return a shallow copy of the accumulated shot history for this run."""
        return list(self._run_shot_history)

    def get_run_image_history(self):
        """Return a shallow copy of the accumulated image history for this run."""
        return list(self._run_image_history)
