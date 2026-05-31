"""TCP server that polls a Bristol wavemeter and serves readings to GUI clients.

Commands (line-terminated, case-insensitive):
  GET_READING  →  JSON: {"wavelength_nm", "frequency_thz", "timestamp", "connected"}
  STATUS       →  JSON: {"connected", "host", "error"}
"""
from __future__ import annotations

import atexit
import json
import logging
import signal
import socket
import threading
import time
from typing import Optional

from waxx.control.misc.bristol_wavemeter import BristolWavemeter
from waxx.util.comms_server.waxx_server import WaxxServer

LOGGER = logging.getLogger("bristol_wavemeter_server")
LOGGER.setLevel(logging.INFO)

SERVER_ID = "bristol_wavemeter"


class BristolWavemeterServer(WaxxServer):
    """Polls a Bristol wavemeter in a background thread and serves readings over TCP."""

    def __init__(
        self,
        wavemeter_host: str = "192.168.1.105",
        host: str = "0.0.0.0",
        port: int = 0,
        poll_interval_s: float = 0.1,
    ):
        WaxxServer.__init__(self, SERVER_ID, port)
        self.wavemeter_host = wavemeter_host
        self.host = host
        self.poll_interval_s = float(poll_interval_s)

        self._wavemeter: Optional[BristolWavemeter] = None
        self._reading: dict = {
            "wavelength_nm": None,
            "frequency_thz": None,
            "timestamp": None,
            "connected": False,
        }
        self._error: Optional[str] = None
        self._lock = threading.Lock()

        self.running = False
        self._server_socket: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stopped = False

    # ------------------------------------------------------------------
    # Public state accessors (safe to call from any thread)
    # ------------------------------------------------------------------

    def get_reading(self) -> dict:
        with self._lock:
            return dict(self._reading)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "connected": self._reading["connected"],
                "host": self.wavemeter_host,
                "error": self._error,
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, 0))
        sock.listen(16)
        self._server_socket = sock
        self._waxx_port = sock.getsockname()[1]
        self._start_beacon()
        self.running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name="BristolAccept")
        self._accept_thread.start()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="BristolPoll")
        self._poll_thread.start()
        LOGGER.info("Server started on port %d, polling %s", self._waxx_port, self.wavemeter_host)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        self._stop_beacon()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        self._disconnect_wavemeter()
        LOGGER.info("Server stopped")

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    def _connect_wavemeter(self) -> None:
        try:
            wm = BristolWavemeter(self.wavemeter_host)
            with self._lock:
                self._wavemeter = wm
                self._reading["connected"] = True
                self._error = None
            LOGGER.info("Connected to wavemeter at %s", self.wavemeter_host)
        except Exception as exc:
            LOGGER.warning("Failed to connect to wavemeter: %s", exc)
            with self._lock:
                self._wavemeter = None
                self._reading["connected"] = False
                self._error = str(exc)

    def _disconnect_wavemeter(self) -> None:
        with self._lock:
            wm = self._wavemeter
            self._wavemeter = None
            self._reading["connected"] = False
        if wm is not None:
            try:
                wm._dev.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        self._connect_wavemeter()
        while self.running:
            with self._lock:
                wm = self._wavemeter
            if wm is not None:
                try:
                    wl_m = wm.get_wavelength()
                    freq_thz = wm.get_frequency() / 1e12
                    with self._lock:
                        self._reading["wavelength_nm"] = wl_m * 1e9
                        self._reading["frequency_thz"] = freq_thz
                        self._reading["timestamp"] = time.time()
                        self._reading["connected"] = True
                        self._error = None
                except Exception as exc:
                    LOGGER.warning("Poll error: %s — reconnecting", exc)
                    with self._lock:
                        self._reading["connected"] = False
                        self._error = str(exc)
                    self._disconnect_wavemeter()
                    time.sleep(2.0)
                    self._connect_wavemeter()
            else:
                time.sleep(2.0)
                self._connect_wavemeter()
            time.sleep(self.poll_interval_s)

    def _accept_loop(self) -> None:
        while self.running:
            try:
                conn, addr = self._server_socket.accept()
            except OSError:
                break
            threading.Thread(
                target=self._handle_client,
                args=(conn, addr),
                daemon=True,
            ).start()

    def _handle_client(self, conn: socket.socket, addr) -> None:
        try:
            conn.settimeout(5.0)
            with conn.makefile("rb") as f:
                line = f.readline().decode("utf-8", errors="replace").strip().upper()
            if line == "GET_READING":
                response = json.dumps(self.get_reading())
            elif line == "STATUS":
                response = json.dumps(self.get_status())
            else:
                response = json.dumps({"error": f"unknown command: {line!r}"})
            conn.sendall((response + "\n").encode("utf-8"))
        except Exception as exc:
            LOGGER.debug("Client handler error (%s): %s", addr, exc)
        finally:
            try:
                conn.close()
            except OSError:
                pass


def main(wavemeter_host: str = "192.168.1.105") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    server = BristolWavemeterServer(wavemeter_host=wavemeter_host)
    atexit.register(server.stop)

    def _sigterm(signum, frame):
        server.stop()

    signal.signal(signal.SIGTERM, _sigterm)
    server.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
