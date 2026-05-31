#!/usr/bin/env python3
"""HMR2300 Magnetometer TCP server.

Reads from the sensor over RS-232 and serves field data to network clients.

Protocol (one command per TCP connection, newline-terminated JSON response):
    PING                     â†’ {"ok": true, "message": "pong"}
    GET_FIELD                â†’ {"ok": true, "t": float, "Bx": float, "By": float,
                                            "Bz": float, "Btot": float}
    GET_SINCE <timestamp_s>  â†’ {"ok": true, "readings": [{...}, ...]}
                               Returns all buffered readings with t > timestamp_s,
                               ordered oldest-first.

Usage:
    python hmr_magnetometer_server.py [--serial-port COM33] [--server-port 50000]
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import socket
import threading
import time
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler

import serial
import serial.tools.list_ports

from waxx.util.comms_server.waxx_server import WaxxServer

DEFAULT_SERIAL_PORT = "COM33"
DEFAULT_BAUD = 9600
DEFAULT_DEVICE_ID = "00"
DEFAULT_POLL_INTERVAL = 0.12
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 0
MAX_HISTORY = 10000
SENSOR_COUNTS_PER_GAUSS = 15000.0
MAX_STUCK_SAME_VALUES = 20

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_logging(log_path: str | None = None) -> None:
    """Attach a StreamHandler and (optionally) a RotatingFileHandler to the
    root logger.  Safe to call multiple times — handlers are only added once."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. called by a parent process)
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            fh = RotatingFileHandler(
                log_path,
                maxBytes=5 * 1024 * 1024,  # 5 MB per file
                backupCount=5,
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            root.addHandler(fh)
            logger.info("Logging to file: %s", log_path)
        except Exception as exc:
            logger.warning("Could not open log file %r: %s", log_path, exc)

class HMR2300Reader:
    def __init__(self, port, baud=DEFAULT_BAUD, device_id=DEFAULT_DEVICE_ID, timeout=1.0):
        self.port = port
        self.baud = baud
        self.device_id = device_id
        self.timeout = timeout
        self.ser = None

    def open(self):
        self.ser = serial.Serial(
            self.port,
            baudrate=self.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )
        time.sleep(0.5)
        self.resync()

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception as exc:
                logger.debug("HMR2300Reader.close: error closing serial port: %s", exc)
        self.ser = None

    def resync(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass
            time.sleep(0.08)

    def send_cmd(self, cmd, wait=0.03):
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port is not open")
        self.ser.write((cmd + "\r").encode("ascii"))
        self.ser.flush()
        time.sleep(wait)
        return self.ser.read_until(b"\r").decode("ascii", errors="replace").strip()

    def setup(self):
        for cmd in ("A", "R=20", "P"):
            self.send_cmd(f"*{self.device_id}{cmd}")
        self.resync()

    def read_one(self):
        last_reply = ""
        for _ in range(5):
            reply = self.send_cmd(f"*{self.device_id}P")
            last_reply = reply
            if not reply:
                time.sleep(0.02)
                continue

            s = re.sub(r"([+-])\s+(\d)", r"\1\2", reply.replace(",", ""))
            nums = re.findall(r"[+-]?\d+", s)
            if len(nums) >= 3:
                return tuple(int(v) for v in nums[:3])

            self.resync()
            time.sleep(0.02)

        raise ValueError(f"Could not parse sensor reply: {last_reply!r}")


class MagnetometerServer(WaxxServer):
    """Headless server: reads HMR2300 and serves field data over TCP."""

    def __init__(
        self,
        serial_port=DEFAULT_SERIAL_PORT,
        baud=DEFAULT_BAUD,
        device_id=DEFAULT_DEVICE_ID,
        poll_interval=DEFAULT_POLL_INTERVAL,
        server_host=DEFAULT_SERVER_HOST,
        server_port=DEFAULT_SERVER_PORT,
        reference_csv_path=None,
    ):
        WaxxServer.__init__(self, "magnetometer", server_port)
        self.serial_port = serial_port
        self.baud = baud
        self.device_id = device_id
        self.poll_interval = poll_interval
        self.server_host = server_host
        self.server_port = server_port  # may be 0 until run() binds
        self.reference_csv_path = reference_csv_path

        self.reader = None
        self.stop_event = threading.Event()
        self.history_lock = threading.Lock()
        self.reference_lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.serial_should_be_connected = True
        self.history = deque(maxlen=MAX_HISTORY)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, log_path: str | None = None):
        # Default log file sits next to this script.
        if log_path is None:
            log_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "hmr_magnetometer_server.log",
            )
        _configure_logging(log_path)

        if self.reference_csv_path:
            self._ensure_reference_csv()

        # Bind and start listening before attempting the COM connection so that
        # the GUI can connect and display server status even while the serial
        # port is being opened (or has failed).
        _srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _srv.bind((self.server_host, self.server_port))
        _srv.listen(16)
        self.server_port = _srv.getsockname()[1]
        self._waxx_port = self.server_port
        self._start_beacon()
        logger.info("Starting TCP server on %s:%d", self.server_host, self.server_port)

        read_thread = threading.Thread(target=self._read_loop, daemon=True)
        read_thread.start()

        logger.info("Opening %s at %d baud (device ID: %r)", self.serial_port, self.baud, self.device_id)
        self.reader = HMR2300Reader(
            port=self.serial_port,
            baud=self.baud,
            device_id=self.device_id,
        )
        try:
            self.reader.open()
            self.reader.setup()
            logger.info("Sensor ready. Polling every %.3f s.", self.poll_interval)
        except Exception as exc:
            self.reader = None
            logger.error(
                "COM port connection failed (%s: %s) — "
                "TCP server starting without serial. "
                "Use SERIAL_RECONNECT or RESTART_SERIAL to retry.",
                type(exc).__name__, exc,
            )
        try:
            self._server_loop(_srv)
        finally:
            self.stop_event.set()
            read_thread.join(timeout=2.0)

    def shutdown(self):
        self._stop_beacon()
        self.stop_event.set()
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception as exc:
                logger.warning("shutdown: error closing reader: %s", exc)
            self.reader = None

    # ------------------------------------------------------------------
    # Sensor read loop (background thread)
    # ------------------------------------------------------------------

    def _read_loop(self):
        last_values = None
        same_count = 0

        while not self.stop_event.is_set():
            if not self.serial_should_be_connected:
                if self.stop_event.wait(self.poll_interval):
                    break
                continue

            # --- ensure serial is open (retry forever, never crash) ---
            if not self._is_serial_connected():
                logger.info("Serial not connected, attempting reconnect on %s.", self.serial_port)
                try:
                    self._reconnect()
                    last_values = None
                    same_count = 0
                    logger.info("Sensor reconnected.")
                except Exception as exc:
                    if self.stop_event.is_set():
                        break
                    logger.warning("Reconnect failed (%s: %s) — retrying in 5 s.", type(exc).__name__, exc)
                    self.stop_event.wait(5.0)
                    continue

            # --- read one sample ---
            try:
                with self.serial_lock:
                    x_counts, y_counts, z_counts = self.reader.read_one()
                values = (x_counts, y_counts, z_counts)

                if values == last_values:
                    same_count += 1
                else:
                    same_count = 0
                    last_values = values

                if same_count == MAX_STUCK_SAME_VALUES:
                    raise RuntimeError(
                        f"Sensor readings stuck for {same_count} consecutive polls"
                    )

                x_G = x_counts / SENSOR_COUNTS_PER_GAUSS
                y_G = y_counts / SENSOR_COUNTS_PER_GAUSS
                z_G = z_counts / SENSOR_COUNTS_PER_GAUSS
                btot = math.sqrt(x_G**2 + y_G**2 + z_G**2)

                reading = {"t": time.time(), "Bx": x_G, "By": y_G, "Bz": z_G, "Btot": btot}
                with self.history_lock:
                    self.history.append(reading)

            except Exception as exc:
                if self.stop_event.is_set():
                    break
                if not self.serial_should_be_connected:
                    continue
                logger.warning("Read error (%s: %s) — resetting serial for reconnect.", type(exc).__name__, exc)
                with self.serial_lock:
                    if self.reader is not None:
                        try:
                            self.reader.close()
                        except Exception as close_exc:
                            logger.debug("_read_loop: error closing reader after read failure: %s", close_exc)
                        self.reader = None
                last_values = None
                same_count = 0
                # skip to top of loop which will reconnect
                continue

            if self.stop_event.wait(self.poll_interval):
                break

    def _reconnect(self):
        if self.stop_event.is_set():
            raise RuntimeError("Server stopping")
        with self.serial_lock:
            if self.reader is not None:
                try:
                    logger.info("Disconnecting serial device on %s.", self.serial_port)
                    self.reader.close()
                    logger.info("Serial device disconnected.")
                except Exception as exc:
                    logger.debug("_reconnect: close before reconnect failed: %s", exc)
                self.reader = None

            logger.info("Reconnecting serial device on %s.", self.serial_port)
            time.sleep(0.2)
            if self.stop_event.is_set():
                raise RuntimeError("Server stopping")
            self.reader = HMR2300Reader(
                port=self.serial_port,
                baud=self.baud,
                device_id=self.device_id,
            )
            self.reader.open()
            self.reader.setup()
            logger.info("Serial device reconnected and configured.")

    def _disconnect_serial(self):
        with self.serial_lock:
            if self.reader is not None:
                try:
                    logger.info("Manually disconnecting serial device on %s.", self.serial_port)
                    self.reader.close()
                    logger.info("Serial device disconnected by operator.")
                except Exception as exc:
                    logger.warning("_disconnect_serial: close failed: %s", exc)
                self.reader = None

    def _is_serial_connected(self):
        with self.serial_lock:
            return (
                self.reader is not None
                and self.reader.ser is not None
                and self.reader.ser.is_open
            )

    # ------------------------------------------------------------------
    # TCP server loop (main thread)
    # ------------------------------------------------------------------

    def _server_loop(self, srv: socket.socket = None):
        if srv is None:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.server_host, self.server_port))
        with srv:
            srv.listen(16)
            srv.settimeout(1.0)
            print(f"[INFO] Listening on {self.server_host}:{self.server_port}")

            while not self.stop_event.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    logger.warning("_server_loop: accept() raised OSError — shutting down TCP loop: %s", exc)
                    break
                threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True,
                ).start()

    def _handle_client(self, conn, addr):
        with conn:
            try:
                raw = conn.recv(4096).decode("utf-8", errors="replace").strip()
                reply = self._dispatch(raw)
            except Exception as exc:
                logger.warning("_handle_client: error handling request from %s: %s", addr, exc)
                reply = {"ok": False, "error": str(exc)}
            try:
                conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
            except Exception as exc:
                logger.warning("_handle_client: failed to send reply to %s: %s", addr, exc)

    def _dispatch(self, command: str) -> dict:
        if command == "PING":
            return {"ok": True, "message": "pong"}

        if command == "GET_SERIAL_STATUS":
            return {
                "ok": True,
                "connected": self._is_serial_connected(),
                "target_connected": bool(self.serial_should_be_connected),
            }

        if command == "SERIAL_DISCONNECT":
            self.serial_should_be_connected = False
            self._disconnect_serial()
            return {"ok": True, "connected": False}

        if command == "SERIAL_RECONNECT":
            self.serial_should_be_connected = True
            try:
                self._reconnect()
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "connected": self._is_serial_connected()}

        if command == "RESTART_SERIAL":
            self.serial_should_be_connected = True
            try:
                self._reconnect()
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"Restart failed ({type(exc).__name__}: {exc})",
                    "connected": False,
                }
            return {
                "ok": True,
                "connected": self._is_serial_connected(),
                "message": "Serial restarted successfully",
            }

        if command == "GET_FIELD":
            with self.history_lock:
                latest = self.history[-1] if self.history else None
            if latest is None:
                return {"ok": False, "error": "No data available yet"}
            return {"ok": True, **latest}

        if command.startswith("GET_SINCE "):
            try:
                since_t = float(command[len("GET_SINCE "):])
            except ValueError:
                return {"ok": False, "error": "Invalid timestamp in GET_SINCE"}
            with self.history_lock:
                readings = [r for r in self.history if r["t"] > since_t]
            return {"ok": True, "readings": readings}

        if command == "SET_REFERENCE":
            if not self.reference_csv_path:
                return {"ok": False, "error": "Reference CSV path not configured"}
            with self.history_lock:
                latest = self.history[-1] if self.history else None
            if latest is None:
                return {"ok": False, "error": "No data available yet"}
            ref = self._append_reference(latest)
            return {"ok": True, "reference": ref}

        if command.startswith("SET_REFERENCE_VALUES "):
            if not self.reference_csv_path:
                return {"ok": False, "error": "Reference CSV path not configured"}
            try:
                parts = command.split()
                if len(parts) != 5:
                    raise ValueError("Expected 4 numeric values")
                bx, by, bz, btot = [float(v) for v in parts[1:5]]
            except ValueError:
                return {"ok": False, "error": "Invalid values in SET_REFERENCE_VALUES"}
            ref = self._append_reference_values(bx, by, bz, btot)
            return {"ok": True, "reference": ref}

        if command.startswith("GET_REFERENCE_BEFORE "):
            if not self.reference_csv_path:
                return {"ok": False, "error": "Reference CSV path not configured"}
            try:
                t_query = float(command[len("GET_REFERENCE_BEFORE "):])
            except ValueError:
                return {"ok": False, "error": "Invalid timestamp in GET_REFERENCE_BEFORE"}
            ref = self._get_reference_at_or_before(t_query)
            if ref is None:
                return {"ok": False, "error": "No reference available for requested date"}
            return {"ok": True, "reference": ref}

        return {"ok": False, "error": f"Unknown command: {command!r}"}

    def _ensure_reference_csv(self):
        parent = os.path.dirname(os.path.abspath(self.reference_csv_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self.reference_csv_path):
            with open(self.reference_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["datetime_iso", "timestamp_s", "Bx", "By", "Bz", "Btot"],
                )
                writer.writeheader()

    def _append_reference(self, reading):
        return self._append_reference_values(
            bx=float(reading["Bx"]),
            by=float(reading["By"]),
            bz=float(reading["Bz"]),
            btot=float(reading["Btot"]),
            timestamp_s=float(reading["t"]),
        )

    def _append_reference_values(self, bx, by, bz, btot, timestamp_s=None):
        if timestamp_s is None:
            timestamp_s = time.time()
        ref_row = {
            "datetime_iso": datetime.now().isoformat(timespec="seconds"),
            "timestamp_s": float(timestamp_s),
            "Bx": float(bx),
            "By": float(by),
            "Bz": float(bz),
            "Btot": float(btot),
        }
        with self.reference_lock:
            self._ensure_reference_csv()
            with open(self.reference_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["datetime_iso", "timestamp_s", "Bx", "By", "Bz", "Btot"],
                )
                writer.writerow(ref_row)
        return ref_row

    def _get_reference_at_or_before(self, t_query):
        with self.reference_lock:
            self._ensure_reference_csv()
            best = None
            with open(self.reference_csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        t_row = float(row.get("timestamp_s", "nan"))
                        bx = float(row["Bx"])
                        by = float(row["By"])
                        bz = float(row["Bz"])
                        btot = float(row["Btot"])
                    except (KeyError, TypeError, ValueError) as exc:
                        logger.debug("Skipping malformed reference CSV row (%s: %s): %s", type(exc).__name__, exc, row)
                        continue
                    if t_row <= t_query and (best is None or t_row > best["timestamp_s"]):
                        best = {
                            "datetime_iso": row.get("datetime_iso", ""),
                            "timestamp_s": t_row,
                            "Bx": bx,
                            "By": by,
                            "Bz": bz,
                            "Btot": btot,
                        }
            return best

