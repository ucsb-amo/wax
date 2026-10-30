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
    GET_SNAPSHOT             â†’ {"ok": true, "com": {...SerialSnapshot...},
                                            "connected": bool, "target_connected": bool,
                                            "latest": {...}|null, "history_len": int}
    SHUTDOWN                 â†’ {"ok": true}; the server then closes the serial
                               port, stops its beacon and exits the process (code 0).
    COM_CONNECT              â†’ {"ok": true, "message": str, "connected": bool}; releases
                               a Disconnect hold, the port opens in the background.
    COM_DISCONNECT           â†’ {"ok": true, "connected": false}; closes the port and
                               holds it closed until COM_CONNECT ({"ok": false,
                               "error": reason} if refused).
    SERIAL_DISCONNECT / SERIAL_RECONNECT / RESTART_SERIAL / GET_SERIAL_STATUS:
                               the older names, kept (disconnect = the same hold).

The serial port is owned by a ComLink (waxx.util.comms_server.com_link): it is
opened on the link's thread after the server is up, reopened with backoff after
a failure, and a person's Disconnect holds it closed.  The read loop only reads
from the open device.

Usage:
    python hmr_magnetometer_server.py [--serial-port COM33] [--server-port 50000]
"""

import argparse
import csv
import json
import logging
import math
import os
import queue
import re
import socket
import threading
import time
from collections import deque
from datetime import datetime

import serial
import serial.tools.list_ports

from beacon.discovery.server import NetServer
from waxx.util.comms_server.com_link import ComLink, ComLinkRefused
from waxx.util.dashboard import logging_setup

DEFAULT_SERIAL_PORT = "COM33"
DEFAULT_BAUD = 9600
DEFAULT_DEVICE_ID = "00"
DEFAULT_POLL_INTERVAL = 0.12
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 0
MAX_HISTORY = 10000
SENSOR_COUNTS_PER_GAUSS = 15000.0
MAX_STUCK_SAME_VALUES = 20
# A stuck sensor is reset every ~MAX_STUCK_SAME_VALUES polls (~4.5 s), and each
# reset used to put ~6 lines in the log -- most of the dashboard log while the
# sensor stayed stuck for days.  Within one stuck episode the first reset and
# the first reading after it are logged in full; later resets go to DEBUG
# (this server's own log file) and a summary is logged this often.  Detection
# and the reset itself are unchanged.
STUCK_SUMMARY_INTERVAL_S = 600.0
# an episode ends only after this long with readings and no stuck reset
STUCK_RECOVERY_QUIET_S = 60.0

# Field log: one reading per wall-clock second, one CSV per local day
# (so a file holds at most 86400 rows).  Values are in gauss, as served.
FIELD_LOG_FIELDS = ["datetime_iso", "timestamp_s", "Bx", "By", "Bz", "Btot"]
FIELD_LOG_FILENAME = "hmr2300_{date}.csv"
# Readings waiting for the writer thread (1 h at 1 Hz); beyond this they are
# dropped, with a warning, rather than growing without bound.
FIELD_LOG_QUEUE_MAX = 3600

# Grace period between answering SHUTDOWN and starting to tear down, so the
# reply is on the wire before the listening socket closes.
_SHUTDOWN_REPLY_GRACE_S = 0.2
# If run() has not returned this long after shutdown(), force the exit.
_SHUTDOWN_FORCE_EXIT_S = 1.5

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_logging(log_path: str | None = None) -> None:
    """Attach a StreamHandler and (optionally) a rotating log file to the root
    logger.  Safe to call multiple times — handlers are only added once.

    The file usually lives on the data share, so it goes through logging_setup:
    written on a background thread (the read loop must not wait on the share),
    and a handle killed by an SMB reconnect moves to a local file and the share
    is retried."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. called by a parent process)
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    sh = logging.StreamHandler()
    # stderr is what the dashboard's supervisor copies into the dashboard log:
    # INFO and up there, the DEBUG detail only in the file below.
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            logging_setup.install_file_logging(log_path, level=logging.DEBUG)
            logger.info("Logging to file: %s", log_path)
        except Exception as exc:
            logger.warning("Could not open log file %r: %s", log_path, exc)

class SensorStuckError(RuntimeError):
    """The sensor returned the same counts for MAX_STUCK_SAME_VALUES polls in
    a row; the read loop resets the serial link, as for any read error."""


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


class _LinkDebugLog:
    """ComLink's own log lines, at DEBUG (this server's log file only).

    The server logs each link event itself, with its existing dedup: a stuck
    sensor is reset every ~4.5 s, sometimes for days, and the link's per-reset
    WARNING/INFO lines would otherwise fill the dashboard log.
    """

    def debug(self, msg, *args, **kwargs):
        logger.debug(msg, *args, **kwargs)

    info = warning = debug


class HMRLink(ComLink):
    """The HMR2300's serial port; opened on the link's thread, never by the read loop."""

    # 0.2 s before reopening after a read error or stuck reset (the old
    # reconnect's pause), then the link's usual steps up to every 30 s.
    BACKOFF_S = (0.2, 2.0, 5.0, 10.0, 30.0)
    # Below the stuck detection (~4.5 s of identical samples), so a sensor that
    # stays stuck keeps being reset 0.2 s after each detection, as before,
    # instead of the link's growing backoff for devices that fail at once.
    STABLE_S = 3.0

    def __init__(self, owner, **kwargs):
        self._owner = owner
        baud = int(owner.baud) if owner.baud is not None else None
        super().__init__(owner.serial_port, baud=baud, server=owner,
                         name=f"HMR2300 {owner.serial_port}", io_lock=owner.serial_lock,
                         logger=_LinkDebugLog(), **kwargs)

    def _open(self):
        return self._owner._open_reader()

    def _close(self, device):
        device.close()

    def _on_connected(self, device):
        self._owner._link_up.set()


class MagnetometerServer(NetServer):
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
        field_log_dir=None,
    ):
        NetServer.__init__(self, "magnetometer", server_port)
        self.serial_port = serial_port
        self.baud = baud
        self.device_id = device_id
        self.poll_interval = poll_interval
        self.server_host = server_host
        self.server_port = server_port  # may be 0 until run() binds
        self.reference_csv_path = reference_csv_path
        # Folder for the daily field-log CSVs; None = no field log.
        self.field_log_dir = field_log_dir
        self._field_log_queue: queue.Queue = queue.Queue(maxsize=FIELD_LOG_QUEUE_MAX)
        # Wall-clock second of the last reading handed to the field log.
        self._field_log_last_sec: int | None = None
        self._field_log_dropped = 0
        # Last write error (type name, str), to log a repeating failure once.
        self._field_log_failure_sig: tuple[str, str] | None = None

        self.stop_event = threading.Event()
        self.history_lock = threading.Lock()
        self.reference_lock = threading.Lock()
        # Held around each read and by the link while it opens/closes the
        # port (re-entrant: ComLink's io_lock).
        self.serial_lock = threading.RLock()
        # Set when the link has (re)opened the port; the read loop waits on it.
        self._link_up = threading.Event()
        # True when the next open follows a repeat stuck reset: its log lines
        # go to DEBUG (see _note_stuck).
        self._quiet_reconnect = False
        self.history = deque(maxlen=MAX_HISTORY)
        # Signature (exc type name, exc str) of the most recent reconnect
        # failure.  Used to suppress identical repeating warnings — only the
        # first occurrence (and any change) is logged at WARNING; the rest
        # drop to DEBUG so we don't fill the dashboard log.
        self._last_reconnect_failure_sig: tuple[str, str] | None = None
        self._reconnect_failure_streak = 0
        # Set True once we've successfully opened the port at least once;
        # used to keep the very first connection attempt visible.
        self._serial_was_ever_connected = False
        # The serial port: opened, reopened and held by the link (started in run()).
        self.link = HMRLink(self)
        # The current stuck-sensor episode, for the log only (see
        # STUCK_SUMMARY_INTERVAL_S); None while the readings change.
        self._stuck_episode: dict | None = None
        # Listening socket, kept so shutdown() can close it and unblock accept().
        self._srv: socket.socket | None = None
        self._shutdown_requested = False

    @property
    def reader(self):
        """The open HMR2300Reader (the link's device), or None."""
        return self.link.device

    @property
    def serial_should_be_connected(self) -> bool:
        """False while a person's Disconnect holds the port closed."""
        return not self.link.held

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
        self._srv = _srv
        self._start_beacon()
        logger.info("Starting TCP server on %s:%d", self.server_host, self.server_port)

        if self.field_log_dir:
            logger.info("Field log: one reading per second to %s", self.field_log_dir)
            threading.Thread(
                target=self._field_log_loop, name="hmr-field-log", daemon=True
            ).start()

        # The link opens the port on its own thread (and keeps retrying); the
        # read loop only reads from it, so only one thread ever opens it.
        logger.info("Opening %s at %d baud (device ID: %r); polling every %.3f s",
                    self.serial_port, self.baud, self.device_id, self.poll_interval)
        self.link.start()
        read_thread = threading.Thread(target=self._read_loop, daemon=True)
        read_thread.start()

        try:
            self._server_loop(_srv)
        finally:
            self.stop_event.set()
            read_thread.join(timeout=2.0)

    def shutdown(self):
        """Stop the beacon, stop the loops, close the serial port.  Idempotent."""
        self._stop_beacon()
        self.stop_event.set()
        self._link_up.set()             # wake the read loop
        try:
            self.link.stop()            # closes the port (bounded)
        except Exception as exc:
            logger.warning("shutdown: error closing reader: %s", exc)
        # Unblock _server_loop's accept() immediately instead of waiting for
        # its 1 s timeout.
        srv = self._srv
        self._srv = None
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass

    def request_shutdown(self):
        """Graceful process shutdown requested over TCP (dashboard Stop).

        Returns immediately so the reply can be sent; a daemon thread then
        waits a short grace period and calls :meth:`shutdown`, which closes
        the serial port, stops the beacon and unblocks ``_server_loop`` so
        :meth:`run` returns and the launcher exits with code 0.  If the
        process is still alive shortly after that, the thread forces
        ``os._exit(0)``.
        """
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        logger.warning("Process shutdown requested over TCP; stopping magnetometer server")

        def _worker():
            time.sleep(_SHUTDOWN_REPLY_GRACE_S)
            try:
                self.shutdown()
            except Exception:
                logger.exception("shutdown() raised during requested shutdown")
            time.sleep(_SHUTDOWN_FORCE_EXIT_S)
            logger.warning("run() did not return after shutdown(); forcing process exit")
            logging.shutdown()
            os._exit(0)

        threading.Thread(target=_worker, name="hmr-shutdown", daemon=True).start()

    def _com_snapshot(self) -> dict:
        """Serial-link summary in ``SerialSnapshot.as_dict()`` shape (+ ``held``):
        the link's snapshot (connecting / connected / error / disconnected /
        held; a person's Disconnect shows as held)."""
        return self.link.snapshot()

    def get_snapshot(self) -> dict:
        with self.history_lock:
            latest = self.history[-1] if self.history else None
            history_len = len(self.history)
        return {
            "ok": True,
            "com": self._com_snapshot(),
            "connected": self._is_serial_connected_nolock(),
            "target_connected": bool(self.serial_should_be_connected),
            "latest": latest,
            "history_len": history_len,
            "poll_interval_s": self.poll_interval,
            "server_port": self.server_port,
        }

    # ------------------------------------------------------------------
    # Sensor read loop (background thread)
    # ------------------------------------------------------------------

    def _read_loop(self):
        last_values = None
        same_count = 0
        stuck_values = None
        # The reader the counts above came from: a reopened port starts over.
        last_dev = None

        while not self.stop_event.is_set():
            # --- read one sample from the port the link holds open ---
            # This loop never opens the port: the link does (retrying forever
            # on its backoff, never crashing), and a person's Disconnect holds
            # it closed -- nothing here can reopen it behind that hold.
            dev = None
            try:
                with self.serial_lock:
                    dev = self.link.device
                    if dev is not None:
                        x_counts, y_counts, z_counts = dev.read_one()
                if dev is None:
                    # not open (opening, retrying, held): wait for the link
                    self._link_up.wait(0.5)
                    self._link_up.clear()
                    continue
                if dev is not last_dev:
                    last_dev = dev
                    last_values = None
                    same_count = 0
                values = (x_counts, y_counts, z_counts)

                if values == last_values:
                    same_count += 1
                else:
                    same_count = 0
                    last_values = values

                if same_count == MAX_STUCK_SAME_VALUES:
                    # The counts go in the message so the reading taken after
                    # the reconnect (logged below) shows whether the sensor
                    # had frozen or the field was just that quiet.
                    stuck_values = values
                    raise SensorStuckError(
                        f"Sensor readings stuck for {same_count} consecutive polls "
                        f"at counts {values}"
                    )

                if stuck_values is not None:
                    episode = self._stuck_episode
                    first_of_episode = episode is None or episode["resets"] <= 1
                    (logger.info if first_of_episode else logger.debug)(
                        "First reading after a stuck reset: counts %s (stuck at %s)",
                        values, stuck_values)
                    stuck_values = None

                x_G = x_counts / SENSOR_COUNTS_PER_GAUSS
                y_G = y_counts / SENSOR_COUNTS_PER_GAUSS
                z_G = z_counts / SENSOR_COUNTS_PER_GAUSS
                btot = math.sqrt(x_G**2 + y_G**2 + z_G**2)

                reading = {"t": time.time(), "Bx": x_G, "By": y_G, "Bz": z_G, "Btot": btot}
                with self.history_lock:
                    self.history.append(reading)
                self.link.note_rx()
                self._queue_field_log(reading)
                self._note_reading_ok()

            except Exception as exc:
                if self.stop_event.is_set():
                    break
                if dev is None or self.link.device is not dev:
                    continue    # closed meanwhile (Disconnect, stop): not a device failure
                if isinstance(exc, SensorStuckError):
                    self._quiet_reconnect = self._note_stuck(exc, stuck_values)
                else:
                    logger.warning("Read error (%s: %s) — resetting serial for reconnect.",
                                   type(exc).__name__, exc)
                with self.serial_lock:
                    if self.link.device is dev:     # not closed / reopened meanwhile
                        # closes the port; the link reopens it (unless held)
                        self.link.report_error(exc, where="read")
                last_values = None
                same_count = 0
                # skip to top of loop, which waits for the reconnect
                continue

            if self.stop_event.wait(self.poll_interval):
                break

    # ------------------------------------------------------------------
    # Stuck-sensor log rate limiting (log only; detection is in _read_loop)
    # ------------------------------------------------------------------

    def _note_stuck(self, exc: Exception, counts) -> bool:
        """Log a stuck detection; True when the reset that follows is a repeat
        within the current episode (its reconnect chatter goes to DEBUG).

        The first detection of an episode is logged in full, as before.  Later
        ones go to DEBUG, with a WARNING summary every STUCK_SUMMARY_INTERVAL_S.
        """
        now = time.monotonic()
        episode = self._stuck_episode
        if episode is None:
            self._stuck_episode = {
                "since_wall": time.time(),
                "since_mono": now,
                "first_counts": counts,
                "last_counts": counts,
                "resets": 1,
                "last_summary": now,
                "last_reset_mono": now,
                "readings_since_reset": 0,
            }
            logger.warning("Read error (%s: %s) — resetting serial for reconnect.",
                           type(exc).__name__, exc)
            return False
        episode["resets"] += 1
        episode["last_counts"] = counts
        episode["last_reset_mono"] = now
        episode["readings_since_reset"] = 0
        logger.debug("Read error (%s: %s) — resetting serial for reconnect (stuck reset #%d).",
                     type(exc).__name__, exc, episode["resets"])
        if now - episode["last_summary"] >= STUCK_SUMMARY_INTERVAL_S:
            episode["last_summary"] = now
            logger.warning(
                "Sensor still stuck: %d stuck resets in %.1f min since %s "
                "(first stuck at counts %s, latest at counts %s). Each reset is "
                "logged at DEBUG in this server's log file; next summary in %.0f min.",
                episode["resets"], (now - episode["since_mono"]) / 60.0,
                datetime.fromtimestamp(episode["since_wall"]).isoformat(timespec="seconds"),
                episode["first_counts"], counts, STUCK_SUMMARY_INTERVAL_S / 60.0,
            )
        return True

    def _note_reading_ok(self) -> None:
        """Count good readings after a stuck reset; end the episode once the
        readings have kept changing for STUCK_RECOVERY_QUIET_S.

        A stuck sensor trips the detector on its (MAX_STUCK_SAME_VALUES + 1)th
        identical reading, so more good readings than that since the last
        reset mean the counts changed in between. A sensor that moves a count
        or two after each reset and then sticks again (seen 2026-10-05) would
        end and start an episode every few seconds, so the episode also needs
        STUCK_RECOVERY_QUIET_S without a stuck reset.
        """
        episode = self._stuck_episode
        if episode is None:
            return
        episode["readings_since_reset"] += 1
        if episode["readings_since_reset"] <= MAX_STUCK_SAME_VALUES:
            return
        if time.monotonic() - episode["last_reset_mono"] < STUCK_RECOVERY_QUIET_S:
            return
        self._stuck_episode = None
        logger.info(
            "Sensor readings changing again: %d readings since the last reset with no "
            "stuck detection. The episode had %d stuck reset(s) over %.1f min since %s "
            "(first stuck at counts %s, last at counts %s).",
            episode["readings_since_reset"], episode["resets"],
            (time.monotonic() - episode["since_mono"]) / 60.0,
            datetime.fromtimestamp(episode["since_wall"]).isoformat(timespec="seconds"),
            episode["first_counts"], episode["last_counts"],
        )

    # ------------------------------------------------------------------
    # Field log (daily CSV, one reading per second)
    # ------------------------------------------------------------------

    def _queue_field_log(self, reading):
        """Hand the first reading of each wall-clock second to the writer thread.

        The reading is logged as measured (no averaging).  Never blocks and
        never raises: the read loop must not depend on the data drive.
        """
        if not self.field_log_dir:
            return
        sec = int(reading["t"])
        if sec == self._field_log_last_sec:
            return
        self._field_log_last_sec = sec
        try:
            self._field_log_queue.put_nowait(reading)
        except queue.Full:
            self._field_log_dropped += 1
            if self._field_log_dropped == 1 or self._field_log_dropped % 600 == 0:
                logger.warning(
                    "Field log queue full — %d reading(s) not logged so far.",
                    self._field_log_dropped,
                )

    def _field_log_loop(self):
        while not self.stop_event.is_set():
            try:
                reading = self._field_log_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write_field_log_row(reading)
                if self._field_log_failure_sig is not None:
                    logger.info("Field log writing again.")
                    self._field_log_failure_sig = None
            except Exception as exc:
                sig = (type(exc).__name__, str(exc))
                if sig != self._field_log_failure_sig:
                    logger.warning(
                        "Field log write failed (%s: %s) — this reading is not logged; "
                        "will keep trying with later ones.", sig[0], sig[1],
                    )
                    self._field_log_failure_sig = sig

    def _write_field_log_row(self, reading):
        t = float(reading["t"])
        stamp = datetime.fromtimestamp(t)
        path = os.path.join(
            self.field_log_dir,
            FIELD_LOG_FILENAME.format(date=stamp.strftime("%Y-%m-%d")),
        )
        os.makedirs(self.field_log_dir, exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_LOG_FIELDS)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({
                "datetime_iso": stamp.isoformat(timespec="milliseconds"),
                "timestamp_s": t,
                "Bx": reading["Bx"],
                "By": reading["By"],
                "Bz": reading["Bz"],
                "Btot": reading["Btot"],
            })

    # ------------------------------------------------------------------
    # Serial port (opened by the link: HMRLink._open -> _open_reader)
    # ------------------------------------------------------------------

    def _open_reader(self):
        """Open and configure the sensor; return the reader, raise on failure.

        Runs on the link's thread.  The routine open chatter is at INFO only
        for the first attempt (not after a failure has been announced, nor
        after a repeat stuck reset); failures are logged once per distinct
        error (_note_open_failure), so retries don't spam the dashboard log.
        """
        first_try = self._last_reconnect_failure_sig is None
        loud = first_try and not self._quiet_reconnect
        self._quiet_reconnect = False
        log_open = logger.info if loud else logger.debug
        log_open("Serial not connected, attempting reconnect on %s.", self.serial_port)
        log_open("Reconnecting serial device on %s.", self.serial_port)
        reader = HMR2300Reader(
            port=self.serial_port,
            baud=self.baud,
            device_id=self.device_id,
        )
        try:
            reader.open()
            reader.setup()
        except Exception as exc:
            reader.close()      # a port that opened but failed setup must not stay open
            self._note_open_failure(exc)
            raise
        log_open("Serial device reconnected and configured.")
        log_open("Sensor reconnected.")
        self._serial_was_ever_connected = True
        self._last_reconnect_failure_sig = None
        self._reconnect_failure_streak = 0
        return reader

    def _note_open_failure(self, exc: Exception) -> None:
        """Log a failed open: WARNING once per distinct error, then DEBUG with
        an INFO reminder every 60 attempts."""
        sig = (type(exc).__name__, str(exc))
        if sig != self._last_reconnect_failure_sig:
            logger.warning(
                "Reconnect failed (%s: %s) — will keep retrying quietly.",
                sig[0], sig[1],
            )
            self._last_reconnect_failure_sig = sig
            self._reconnect_failure_streak = 1
        else:
            self._reconnect_failure_streak += 1
            # Periodically remind at INFO so the log shows the
            # condition is still active without flooding it.
            if self._reconnect_failure_streak % 60 == 0:
                logger.info(
                    "Reconnect still failing (%s) after %d attempts.",
                    sig[0], self._reconnect_failure_streak,
                )
            else:
                logger.debug(
                    "Reconnect failed (%s: %s).", sig[0], sig[1],
                )

    def _connect_serial(self):
        """A person's Connect: release the hold; the link opens the port now
        (in the background, skipping any backoff wait)."""
        if self.link.held:
            logger.info("Serial connect requested by operator on %s.", self.serial_port)
        self.link.connect()

    def _reconnect(self):
        """Close the port and reopen it (in the background), releasing any hold."""
        logger.info("Restarting serial device on %s.", self.serial_port)
        self.link.disconnect(hold=True)
        self.link.connect()

    def _disconnect_serial(self):
        """A person's Disconnect: close the port and hold it closed until a
        connect.  Raises ComLinkRefused if the link refuses."""
        logger.info("Manually disconnecting serial device on %s.", self.serial_port)
        self.link.disconnect(hold=True)
        logger.info("Serial device disconnected by operator (held closed until Connect).")

    def _await_link(self, timeout_s: float = 2.0) -> str:
        """Wait, at most ``timeout_s``, for an open the link has started; return
        the link's state.  Only the older reconnect/restart commands wait (on
        their request's own thread), so their replies still say how it went."""
        deadline = time.monotonic() + timeout_s
        while self.link.state == "connecting" and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.link.state

    def _is_serial_connected(self):
        return self.link.connected

    def _is_serial_connected_nolock(self):
        """Lock-free (snapshots must not queue behind a slow read_one())."""
        return self.link.connected

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
                    if self.stop_event.is_set():
                        break   # shutdown() closed the listening socket on purpose
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

        if command == "GET_SNAPSHOT":
            return self.get_snapshot()

        if command == "SHUTDOWN":
            self.request_shutdown()
            return {"ok": True, "message": "shutting down"}

        if command == "GET_SERIAL_STATUS":
            return {
                "ok": True,
                "connected": self._is_serial_connected(),
                "target_connected": bool(self.serial_should_be_connected),
            }

        if command == "COM_CONNECT":
            was_connected = self._is_serial_connected()
            self._connect_serial()
            return {
                "ok": True,
                "connected": self._is_serial_connected(),
                "message": ("Serial already connected" if was_connected
                            else "Serial connect requested (the port opens in the background)"),
            }

        if command in ("COM_DISCONNECT", "SERIAL_DISCONNECT"):
            try:
                self._disconnect_serial()
            except ComLinkRefused as exc:
                return {"ok": False, "error": f"Disconnect refused: {exc}",
                        "connected": self._is_serial_connected()}
            return {"ok": True, "connected": False}

        if command == "SERIAL_RECONNECT":
            self._connect_serial()
            state = self._await_link()
            if state == "error":
                return {"ok": False, "error": self.link.snapshot()["last_error"] or "connect failed"}
            reply = {"ok": True, "connected": self._is_serial_connected()}
            if not reply["connected"]:
                reply["message"] = f"Serial connect requested (link: {state})"
            return reply

        if command == "RESTART_SERIAL":
            try:
                self._reconnect()
            except ComLinkRefused as exc:
                return {"ok": False, "error": f"Restart refused: {exc}",
                        "connected": self._is_serial_connected()}
            state = self._await_link()
            if state == "error":
                return {
                    "ok": False,
                    "error": f"Restart failed ({self.link.snapshot()['last_error']})",
                    "connected": False,
                }
            if not self._is_serial_connected():
                return {"ok": True, "connected": False,
                        "message": f"Serial restart requested (link: {state})"}
            return {
                "ok": True,
                "connected": True,
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

