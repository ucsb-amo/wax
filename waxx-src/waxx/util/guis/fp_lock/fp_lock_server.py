"""FPLockServer: owns the lock's serial port and serves it over TCP.

The lock is a qNimble Quarto running the fp_lock_core firmware (protocol in
:mod:`fp_lock_protocol`).  The server

* opens the port through a :class:`ComLink` (after binding and beaconing; a
  missing board never stops the server) and puts the firmware in CSV mode
  (``fmt 0``) on every connect; a clean stop puts it back to the table
  (``fmt 1``) for a person's terminal;
* reads every line on its own thread: CSV rows go to the history (and the
  daily log), ``name is value`` replies update the settings cache, anything
  else goes to the console text;
* answers one JSON request per TCP connection (newline-terminated JSON object
  with a ``cmd``; reply is one JSON line):

  ========================= ==================================================
  ``get_snapshot``          latest row, settings, warnings, ``com`` (COM pill)
  ``get_history``           rows with ``seq`` > ``since`` (at most ``max``)
  ``get_text``              console lines with ``seq`` > ``since``
  ``set`` name value        one setting from ``SETTINGS``; read back
  ``lock`` on               ``lock 1`` / ``lock 0``
  ``reset_setpoint``        ``setp`` = the firmware's current ``sep``
  ``clear``                 zero the integral and the output
  ``raw`` line              send one line, return what came back
  ``refresh_settings``      re-read every setting
  ``com_connect``           release a hold and open the port
  ``com_disconnect``        close the port and hold it closed
  ``shutdown``              stop the process (main's stop path)
  ========================= ==================================================

Nothing here changes the lock on its own except ``fmt`` (printout format);
every setting, the lock switch and clear are a request's.

Machine-agnostic: the port, id, folders and edges come from the launcher.
"""

from __future__ import annotations

import collections
import datetime
import json
import logging
import math
import os
import socket
import threading
import time
from typing import Any, Callable, Optional

from beacon.discovery.server import NetServer
from waxx.util.comms_server.com_link import ComLink, ComLinkNotConnected, ComLinkRefused
from waxx.util.guis.fp_lock import fp_lock_protocol as proto

__all__ = ["FPLockServer", "FPLockLink", "DEFAULT_BAUD"]

log = logging.getLogger("waxx.fp_lock.server")

DEFAULT_BAUD = 115200
HISTORY_MAX = 7200          # rows kept (~2.4 h at one row per ~1.2 s)
HISTORY_REPLY_MAX = 2000    # rows per get_history reply
TEXT_MAX = 400              # console lines kept
QUERY_TIMEOUT_S = 1.0
RAW_COLLECT_S = 0.4
HOUSEKEEPING_S = 2.0        # log flush and CSV-mode check period
LOG_FLUSH_S = 10.0
FMT_REASSERT_S = 10.0
LOG_BUFFER_MAX = 20000
_SHUTDOWN_REPLY_GRACE_S = 0.2


def _open_serial(port: str, baud: Optional[int]):
    import serial  # noqa: PLC0415
    return serial.Serial(port, baud or DEFAULT_BAUD, timeout=0.2, write_timeout=1.0)


class FPLockLink(ComLink):
    """The lock board's serial port, opened on the link's thread."""

    def __init__(self, owner: "FPLockServer", port: str, baud: int,
                 serial_factory: Callable[[str, Optional[int]], Any]):
        super().__init__(port, baud=baud, server=owner, name=f"FP lock {port}", logger=log)
        self._owner = owner
        self._factory = serial_factory

    def _open(self):
        return self._factory(self.port, self.baud)

    def _on_connected(self, device):
        self._owner._on_link_up()


class _CommandError(RuntimeError):
    pass


class FPLockServer(NetServer):
    """Serial owner + TCP request server + beacon (module docstring)."""

    def __init__(
        self,
        server_id: str,
        serial_port: str,
        *,
        baud: int = DEFAULT_BAUD,
        host: str = "0.0.0.0",
        port: int = 0,
        log_dir: Optional[str] = None,
        edge: proto.EdgeConfig = proto.EdgeConfig(),
        stale_after_s: float = 5.0,
        serial_factory: Callable[[str, Optional[int]], Any] = _open_serial,
        on_shutdown_request: Optional[Callable[[], None]] = None,
    ):
        NetServer.__init__(self, server_id, port)
        self.server_id = server_id
        self.serial_port = serial_port
        self.host = host
        self.port = int(port)
        self.log_dir = log_dir
        self.edge = edge
        self.stale_after_s = float(stale_after_s)
        self._on_shutdown_request = on_shutdown_request

        self._stop = threading.Event()
        self._link_up = threading.Event()
        self._lock = threading.Lock()                 # rows, text, settings
        self._cmd_lock = threading.Lock()             # one command at a time
        self._write_lock = threading.Lock()
        self._cond = threading.Condition(self._lock)  # replies arrive

        self._seq = 0
        self._latest: Optional[dict] = None
        self._last_row_mono: Optional[float] = None
        self._history: collections.deque = collections.deque(maxlen=HISTORY_MAX)
        self._text_seq = 0
        self._text: collections.deque = collections.deque(maxlen=TEXT_MAX)
        self._reply_n = 0
        self._replies: dict[str, tuple[int, str]] = {}
        self._unknown: dict[str, int] = {}
        self._settings: dict[str, Optional[float]] = {s.name: None for s in proto.SETTINGS}
        self._missing: set[str] = set()
        self._label: Optional[str] = None
        self._fmt_asserted_at: Optional[float] = None

        self._log_rows: list[dict] = []
        self._log_path: Optional[str] = None
        self._log_error: Optional[str] = None
        self._log_dropped = 0
        self._last_flush = time.monotonic()

        self._sock: Optional[socket.socket] = None
        self._threads: list[threading.Thread] = []
        self._shutdown_requested = False
        self.link = FPLockLink(self, serial_port, baud, serial_factory)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Bind, beacon, then open the port in the background.  Returns at once."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(8)
        sock.settimeout(0.5)
        self._sock = sock
        self.port = sock.getsockname()[1]
        self._waxx_port = self.port
        self._start_beacon()
        log.info("FP lock server %s listening on %s:%d (port %s)", self.server_id,
                 self.host, self.port, self.serial_port)
        self.link.start()
        for target, name in ((self._read_loop, "fplock-read"),
                             (self._accept_loop, "fplock-accept"),
                             (self._housekeeping_loop, "fplock-house")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        """Put the firmware back to its table printout, flush the log, close
        the port and the socket.  Idempotent."""
        if self._stop.is_set():
            return
        if self.link.connected:
            try:
                self._write("fmt 1")
            except Exception as exc:  # best effort
                log.info("could not restore fmt 1 on stop: %s", exc)
        self._stop.set()
        self._link_up.set()
        self._stop_beacon()
        self._flush_log(force=True)
        self.link.stop()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        for t in self._threads:
            t.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    # Serial side
    # ------------------------------------------------------------------ #

    def _on_link_up(self) -> None:
        self._link_up.set()
        threading.Thread(target=self._configure_device, name="fplock-config", daemon=True).start()

    def _configure_device(self) -> None:
        """After every connect: CSV mode, read the settings and the board label."""
        time.sleep(0.2)
        try:
            self._query("fmt", "0")
            self._fmt_asserted_at = time.monotonic()
            self.refresh_settings()
            self._label = self._ping()
            log.info("FP lock connected on %s (board %r)", self.serial_port, self._label)
        except Exception as exc:
            (log.debug if self._stop.is_set() or not self.link.connected else log.warning)(
                "FP lock: setup after connect failed: %s", exc)

    def _read_loop(self) -> None:
        buf = b""
        while not self._stop.is_set():
            dev = self.link.device
            if dev is None:
                buf = b""
                self._link_up.wait(0.5)
                self._link_up.clear()
                continue
            try:
                n = dev.in_waiting
                chunk = dev.read(n if n else 1)
            except Exception as exc:
                if self._stop.is_set():
                    break
                if self.link.device is dev:
                    self.link.report_error(exc, where="read")
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                self._handle_line(raw.decode("utf-8", errors="replace").strip("\r\n\t "))
            if len(buf) > 65536:
                buf = b""

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        self.link.note_rx()
        row = proto.parse_row(line)
        with self._cond:
            if row is not None:
                self._seq += 1
                row["seq"] = self._seq
                row["t"] = time.time()
                self._latest = row
                self._last_row_mono = time.monotonic()
                self._history.append(row)
                if len(self._log_rows) < LOG_BUFFER_MAX:
                    self._log_rows.append(row)
                else:
                    self._log_dropped += 1
                return
            self._text_seq += 1
            self._text.append({"seq": self._text_seq, "t": time.time(), "line": line})
            reply = proto.parse_reply(line)
            if reply is not None:
                name, text = reply
                self._reply_n += 1
                self._replies[name] = (self._reply_n, text)
                if name in self._settings:
                    self._settings[name] = _to_float(text)
                    self._missing.discard(name)
            unknown = proto.parse_unknown(line)
            if unknown is not None:
                self._reply_n += 1
                self._unknown[unknown] = self._reply_n
            self._cond.notify_all()

    def _write(self, line: str) -> None:
        with self._write_lock:
            dev = self.link.require()
            try:
                dev.write((line + "\n").encode("ascii"))
            except Exception as exc:
                if self.link.device is dev:
                    self.link.report_error(exc, where="write")
                raise

    def _query(self, name: str, value: Optional[str] = None,
               timeout: float = QUERY_TIMEOUT_S) -> str:
        """Send ``name`` or ``name value``; return the ``name is ...`` text."""
        with self._cmd_lock:
            with self._cond:
                start = self._reply_n
            self._write(name if value is None else f"{name} {value}")
            deadline = time.monotonic() + timeout
            with self._cond:
                while True:
                    got = self._replies.get(name)
                    if got is not None and got[0] > start:
                        return got[1]
                    if self._unknown.get(name, 0) > start:
                        if name in self._settings:
                            self._missing.add(name)
                        raise _CommandError(f"the firmware does not know {name!r}")
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise _CommandError(f"no reply to {name!r} within {timeout:g} s")
                    self._cond.wait(left)

    def _ping(self) -> Optional[str]:
        """The board's stored label (``ping`` prints one bare line)."""
        with self._cmd_lock:
            with self._cond:
                start = self._text_seq
            self._write("ping")
            deadline = time.monotonic() + 0.6
            with self._cond:
                while time.monotonic() < deadline:
                    for item in self._text:
                        if item["seq"] > start and proto.parse_reply(item["line"]) is None:
                            return item["line"][:64]
                    self._cond.wait(0.05)
        return None

    def refresh_settings(self) -> dict:
        for s in proto.SETTINGS:
            try:
                self._query(s.name)
            except _CommandError as exc:
                log.debug("refresh %s: %s", s.name, exc)
        with self._lock:
            return dict(self._settings)

    # ------------------------------------------------------------------ #
    # Requests
    # ------------------------------------------------------------------ #

    def handle(self, req: dict) -> dict:
        """One request dict -> reply dict (also the TCP entry point)."""
        cmd = str(req.get("cmd", "")).lower()
        try:
            if cmd == "get_snapshot":
                return self.snapshot()
            if cmd == "get_history":
                return self._get_history(int(req.get("since", 0)),
                                         int(req.get("max", HISTORY_REPLY_MAX)))
            if cmd == "get_text":
                return self._get_text(int(req.get("since", 0)))
            if cmd == "set":
                return self._set(str(req.get("name", "")), req.get("value"))
            if cmd == "lock":
                return self._set("lock", 1 if req.get("on") else 0)
            if cmd == "reset_setpoint":
                return self._reset_setpoint()
            if cmd == "clear":
                return self._clear()
            if cmd == "raw":
                return self._raw(str(req.get("line", "")))
            if cmd == "refresh_settings":
                return {"status": "ok", "settings": self.refresh_settings()}
            if cmd == "com_connect":
                self.link.connect()
                return {"status": "ok", "message": "connecting"}
            if cmd == "com_disconnect":
                try:
                    self.link.disconnect(hold=True)
                except ComLinkRefused as exc:
                    return {"status": "refused", "message": str(exc)}
                return {"status": "ok", "message": "disconnected and held"}
            if cmd == "shutdown":
                return self._request_shutdown()
            return {"status": "error", "message": f"unknown cmd: {cmd!r}"}
        except (ComLinkNotConnected, _CommandError, ValueError) as exc:
            return {"status": "error", "message": str(exc)}
        except Exception as exc:
            log.exception("request %s failed", cmd)
            return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}

    def snapshot(self) -> dict:
        with self._lock:
            latest = dict(self._latest) if self._latest else None
            age = (None if self._last_row_mono is None
                   else time.monotonic() - self._last_row_mono)
            settings = dict(self._settings)
            missing = sorted(self._missing)
            seq, text_seq = self._seq, self._text_seq
        com = self.link.snapshot()
        warnings = proto.compute_warnings(latest, edge=self.edge, com_status=com["status"],
                                          data_age_s=age, stale_after_s=self.stale_after_s)
        return {
            "status": "ok",
            "ok": True,
            "server_id": self.server_id,
            "com": com,
            "connected": self.link.connected,
            "label": self._label,
            "latest": latest,
            "data_age_s": None if age is None else round(age, 2),
            "settings": settings,
            "missing_settings": missing,
            "warnings": warnings,
            "seq": seq,
            "text_seq": text_seq,
            "edge": {"right_edge_ms": self.edge.right_edge_ms, "margin_ms": self.edge.margin_ms},
            "log_path": self._log_path,
            "log_error": self._log_error,
            "log_dropped": self._log_dropped,
        }

    def _get_history(self, since: int, max_rows: int) -> dict:
        max_rows = max(1, min(int(max_rows), HISTORY_REPLY_MAX))
        with self._lock:
            rows = [r for r in self._history if r["seq"] > since]
            seq = self._seq
        return {"status": "ok", "rows": rows[-max_rows:], "seq": seq}

    def _get_text(self, since: int) -> dict:
        with self._lock:
            lines = [t for t in self._text if t["seq"] > since]
            seq = self._text_seq
        return {"status": "ok", "lines": lines, "seq": seq}

    def _set(self, name: str, value: Any) -> dict:
        setting = proto.SETTABLE.get(name)
        if setting is None:
            raise ValueError(f"not a setting: {name!r}")
        if value is None:
            raise ValueError(f"no value for {name}")
        text = proto.format_value(setting, value)
        reply = self._query(name, text)
        log.info("set %s = %s (firmware reads back %s)", name, text, reply)
        return {"status": "ok", "name": name, "value": _to_float(reply)}

    def _reset_setpoint(self) -> dict:
        sep = _to_float(self._query("sep"))
        if sep is None or not math.isfinite(sep) or sep <= 0:
            raise ValueError(f"no measured separation to use (sep = {sep})")
        with self._lock:
            old = self._settings.get("setp")
        new = _to_float(self._query("setp", repr(sep)))
        log.info("setpoint reset %s -> %s ms (measured sep %s ms)", old, new, sep)
        return {"status": "ok", "old": old, "new": new, "sep": sep}

    def _clear(self) -> dict:
        with self._cmd_lock:
            with self._cond:
                start = self._text_seq
            self._write("clear")
            deadline = time.monotonic() + QUERY_TIMEOUT_S
            with self._cond:
                while time.monotonic() < deadline:
                    for item in self._text:
                        if item["seq"] > start:
                            line = item["line"]
                            if "cleared" in line:
                                log.info("integral and output cleared")
                                return {"status": "ok", "message": line}
                            if proto.parse_unknown(line) == "clear":
                                return {"status": "error",
                                        "message": "the firmware has no clear command"}
                    self._cond.wait(0.05)
        return {"status": "error", "message": "no reply to clear"}

    def _raw(self, line: str) -> dict:
        line = line.strip()
        if not line or "\n" in line or "\r" in line or len(line) > 200:
            raise ValueError("one line of at most 200 characters")
        with self._cmd_lock:
            with self._cond:
                start = self._text_seq
            self._write(line)
            time.sleep(RAW_COLLECT_S)
            with self._lock:
                lines = [t["line"] for t in self._text if t["seq"] > start]
        log.info("raw %r -> %d line(s)", line, len(lines))
        return {"status": "ok", "lines": lines}

    def _request_shutdown(self) -> dict:
        if self._on_shutdown_request is None:
            return {"status": "error", "message": "shutdown not wired in this process"}
        if not self._shutdown_requested:
            self._shutdown_requested = True
            log.warning("process shutdown requested over TCP")

            def _later():
                time.sleep(_SHUTDOWN_REPLY_GRACE_S)
                self._on_shutdown_request()

            threading.Thread(target=_later, name="fplock-shutdown", daemon=True).start()
        return {"status": "ok"}

    # ------------------------------------------------------------------ #
    # TCP
    # ------------------------------------------------------------------ #

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5.0)
            buf = b""
            while b"\n" not in buf and len(buf) < 65536:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
            text = buf.decode("utf-8", errors="replace").strip()
            if not text:
                return
            try:
                req = json.loads(text)
            except json.JSONDecodeError:
                req = {"cmd": text}            # bare word, e.g. GET_SNAPSHOT
            if not isinstance(req, dict):
                req = {"cmd": ""}
            reply = self.handle(req)
            # a NaN from the firmware goes out as null (strict JSON)
            payload = json.dumps(_nan_to_none(reply), allow_nan=False, default=_json_default)
            conn.sendall((payload + "\n").encode("utf-8"))
        except Exception:
            log.exception("connection failed")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # Housekeeping: daily log, CSV mode
    # ------------------------------------------------------------------ #

    def _housekeeping_loop(self) -> None:
        while not self._stop.wait(HOUSEKEEPING_S):
            self._flush_log()
            self._reassert_fmt()

    def _reassert_fmt(self) -> None:
        """No rows for a while on an open port (a board reset, someone sent
        ``fmt 1`` from the console): ask for CSV again, at most every 10 s."""
        if not self.link.connected:
            return
        with self._lock:
            age = None if self._last_row_mono is None else time.monotonic() - self._last_row_mono
        now = time.monotonic()
        if age is not None and age <= self.stale_after_s:
            return
        if self._fmt_asserted_at is not None and now - self._fmt_asserted_at < FMT_REASSERT_S:
            return
        self._fmt_asserted_at = now
        try:
            self._query("fmt", "0")
            log.info("no rows for %s s: asked the firmware for CSV again",
                     "?" if age is None else f"{age:.0f}")
        except Exception as exc:
            log.debug("fmt 0 re-assert: %s", exc)

    def _flush_log(self, force: bool = False) -> None:
        if not self.log_dir:
            with self._lock:
                self._log_rows.clear()
            return
        now = time.monotonic()
        if not force and now - self._last_flush < LOG_FLUSH_S:
            return
        self._last_flush = now
        with self._lock:
            rows, self._log_rows = self._log_rows, []
        if not rows:
            return
        by_day: dict[str, list[dict]] = {}
        for r in rows:
            day = datetime.datetime.fromtimestamp(r["t"]).strftime("%Y-%m-%d")
            by_day.setdefault(day, []).append(r)
        fields = list(proto.CSV_FIELDS) + ["t2"]
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            for day, day_rows in by_day.items():
                path = os.path.join(self.log_dir, f"{day}.csv")
                new = not os.path.exists(path)
                # opened and closed each flush: a network share's stale handle
                # never outlives one write
                with open(path, "a", encoding="utf-8", newline="") as fh:
                    if new:
                        fh.write("time_iso,unix_s," + ",".join(fields) + "\n")
                    for r in day_rows:
                        iso = datetime.datetime.fromtimestamp(r["t"]).isoformat(timespec="milliseconds")
                        vals = ["" if r.get(f) is None else repr(r[f]) for f in fields]
                        fh.write(f"{iso},{r['t']:.3f}," + ",".join(vals) + "\n")
                self._log_path = path
            if self._log_error is not None:
                log.info("lock log writing again (%s)", self._log_path)
            self._log_error = None
        except OSError as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if msg != self._log_error:
                log.warning("lock log write failed (%s); rows kept for the next try", msg)
            self._log_error = msg
            with self._lock:
                keep = LOG_BUFFER_MAX - len(self._log_rows)
                if keep < len(rows):
                    self._log_dropped += len(rows) - max(keep, 0)
                    rows = rows[-max(keep, 0):] if keep > 0 else []
                self._log_rows = rows + self._log_rows


def _to_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return None


def _json_default(obj):
    return str(obj)


def _nan_to_none(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return obj
