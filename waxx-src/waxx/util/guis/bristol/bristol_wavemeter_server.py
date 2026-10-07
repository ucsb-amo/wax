"""TCP server that polls a Bristol wavemeter and serves readings to GUI clients.

Commands (line-terminated, case-insensitive):
  GET_READING  →  JSON: {"wavelength_nm", "frequency_thz", "timestamp", "connected"}
  STATUS       →  JSON: {"connected", "host", "error"}
  GET_AVERAGE <n> <max_age_s>
               →  JSON: {"ok", "n_requested", "n_used", "mean_hz", "std_hz",
                         "t_first", "t_last", "age_s", "max_age_s", "connected"}
                  Mean and sample std (ddof=1; 0. for a single reading) of the
                  last <n> readings taken within <max_age_s> seconds of the
                  request, on this server's clock. Fewer than <n> fresh
                  readings are averaged as they are and counted in n_used;
                  none gives ok=False, n_used=0.
"""
from __future__ import annotations

import atexit
import collections
import json
import logging
import signal
import socket
import statistics
import threading
import time
from typing import Optional

from waxx.control.misc.bristol_wavemeter import BristolWavemeter, _C_LIGHT
from beacon.discovery.server import NetServer

LOGGER = logging.getLogger("bristol_wavemeter_server")
LOGGER.setLevel(logging.INFO)

SERVER_ID = "bristol_wavemeter"

# Readings kept for GET_AVERAGE (one per poll, ~0.1-0.2 s apart).
HISTORY_LEN = 1000


class BristolWavemeterServer(NetServer):
    """Polls a Bristol wavemeter in a background thread and serves readings over TCP."""

    def __init__(
        self,
        wavemeter_host: str = "192.168.1.105",
        host: str = "0.0.0.0",
        port: int = 0,
        poll_interval_s: float = 0.1,
    ):
        NetServer.__init__(self, SERVER_ID, port)
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
        # (timestamp, frequency_hz) of successful readings, oldest first.
        # Cleared on every disconnect, so an average never spans an outage.
        self._history: collections.deque = collections.deque(maxlen=HISTORY_LEN)

        self.running = False
        self._server_socket: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stopped = False

        # Log dedup / backoff state (touched only by the poll thread)
        self._consecutive_connect_failures = 0
        self._consecutive_poll_failures = 0
        self._connected_logged_once = False

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

    def get_average(self, n: int, max_age_s: float) -> dict:
        """Mean and sample std of the last ``n`` readings no older than
        ``max_age_s`` (server clock). See the module docstring."""
        n = int(n)
        max_age_s = float(max_age_s)
        if n < 1 or not max_age_s > 0.:
            return {"ok": False, "n_used": 0,
                    "error": f"bad GET_AVERAGE arguments n={n} max_age_s={max_age_s}"}
        now = time.time()
        with self._lock:
            fresh = [(t, f) for (t, f) in self._history if now - t <= max_age_s]
            connected = self._reading["connected"]
        used = fresh[-n:]
        out = {
            "ok": bool(used),
            "n_requested": n,
            "n_used": len(used),
            "mean_hz": None,
            "std_hz": None,
            "t_first": None,
            "t_last": None,
            "age_s": None,
            "max_age_s": max_age_s,
            "connected": connected,
        }
        if not used:
            out["error"] = f"no readings within {max_age_s:g} s"
            return out
        freqs = [f for (_, f) in used]
        # Statistics of the offsets from the first reading: the absolute
        # values are ~4e14 Hz, the spread MHz, so this keeps full precision.
        f_ref = freqs[0]
        offsets = [f - f_ref for f in freqs]
        out["mean_hz"] = f_ref + statistics.fmean(offsets)
        out["std_hz"] = statistics.stdev(offsets) if len(offsets) > 1 else 0.
        out["t_first"] = used[0][0]
        out["t_last"] = used[-1][0]
        out["age_s"] = now - used[-1][0]
        return out

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
            # Only log "Connected" once per successful reconnect cycle to
            # avoid spamming the log when the device is flapping.
            if self._consecutive_connect_failures > 0 or not self._connected_logged_once:
                LOGGER.info("Connected to wavemeter at %s", self.wavemeter_host)
                self._connected_logged_once = True
            self._consecutive_connect_failures = 0
        except Exception as exc:
            self._consecutive_connect_failures += 1
            # First failure -> WARNING; subsequent -> DEBUG.
            log_fn = LOGGER.warning if self._consecutive_connect_failures == 1 else LOGGER.debug
            log_fn(
                "Failed to connect to wavemeter (#%d): %s",
                self._consecutive_connect_failures, exc,
            )
            with self._lock:
                self._wavemeter = None
                self._reading["connected"] = False
                self._error = str(exc)

    def _disconnect_wavemeter(self) -> None:
        with self._lock:
            wm = self._wavemeter
            self._wavemeter = None
            self._reading["connected"] = False
            self._history.clear()
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
                    # One measurement per poll: the frequency comes from this
                    # wavelength (get_frequency() would take a second one).
                    wl_m = wm.get_wavelength()
                    if not wl_m > 0.:
                        # no valid peak (0. was a ZeroDivisionError before);
                        # same handling as a comms error, never averaged
                        raise ValueError(f"no valid wavelength reading ({wl_m!r})")
                    freq_hz = _C_LIGHT / wl_m
                    t_now = time.time()
                    with self._lock:
                        self._reading["wavelength_nm"] = wl_m * 1e9
                        self._reading["frequency_thz"] = freq_hz / 1e12
                        self._reading["timestamp"] = t_now
                        self._reading["connected"] = True
                        self._error = None
                        self._history.append((t_now, freq_hz))
                    self._consecutive_poll_failures = 0
                except Exception as exc:
                    self._consecutive_poll_failures += 1
                    # First failure WARN, subsequent DEBUG -- avoids
                    # log spam when the device is flapping.
                    log_fn = (
                        LOGGER.warning
                        if self._consecutive_poll_failures == 1
                        else LOGGER.debug
                    )
                    log_fn(
                        "Poll error (#%d): %s -- reconnecting",
                        self._consecutive_poll_failures, exc,
                    )
                    with self._lock:
                        self._reading["connected"] = False
                        self._error = str(exc)
                    self._disconnect_wavemeter()
                    # Exponential backoff capped at 30 s.
                    backoff = min(
                        30.0,
                        2.0 * (2 ** min(self._consecutive_poll_failures - 1, 4)),
                    )
                    time.sleep(backoff)
                    self._connect_wavemeter()
            else:
                # No live connection -- back off based on connect failures.
                backoff = min(
                    30.0,
                    2.0 * (2 ** min(self._consecutive_connect_failures, 4)),
                )
                time.sleep(backoff)
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
            parts = line.split()
            if line == "GET_READING":
                response = json.dumps(self.get_reading())
            elif parts and parts[0] == "GET_AVERAGE":
                try:
                    n, max_age_s = int(parts[1]), float(parts[2])
                except (IndexError, ValueError):
                    response = json.dumps({
                        "ok": False, "n_used": 0,
                        "error": f"usage: GET_AVERAGE <n> <max_age_s>, got {line!r}"})
                else:
                    response = json.dumps(self.get_average(n, max_age_s))
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
