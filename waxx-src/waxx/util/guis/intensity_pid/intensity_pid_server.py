"""IntensityPIDServer: owns the two-loop intensity PID Quarto's serial port and
serves it over TCP (protocol in :mod:`intensity_pid_protocol`).

* The port opens through a :class:`ComLink` after the server binds and
  beacons; a missing board never stops the server.  On every connect the
  server reads the gains and the board label, and **re-applies the gains last
  set through it** if the board's
  differ (a power cycle or a reflash resets them to the sketch's defaults).  A
  re-apply is logged and shown as a warning for ``REAPPLY_WARN_S``.
* Every status line (~100 per second) updates the latest row; 100 ms means go
  to the history the panel plots, 1 s means to the daily CSV log.
* One JSON request per TCP connection (newline-terminated JSON object with a
  ``cmd``; the reply is one JSON line):

  ========================= ==================================================
  ``get_snapshot``          latest row, latest 100 ms mean, gains, warnings, com
  ``get_history``           100 ms means with ``seq`` > ``since``
  ``hold`` on               true -> ``hold 1``, false -> ``hold 0``, null toggles
  ``clear``                 ``c``: zero both integrators (the board does not reply)
  ``set`` name value        one setting; read back; a gain is remembered
  ``override`` loop on volt loop 1|2: ovolt = volt (if given), override on/off
                            (never re-applied after a board reset)
  ``refresh_settings``      re-read the gains
  ``raw`` line              send one line, return what came back
  ``com_connect``           release a COM hold and open the port
  ``com_disconnect``        close the port and hold it closed
  ``shutdown``              stop the process (main's stop path)
  ========================= ==================================================

Nothing here changes the loops on its own except the gain re-apply on
connect; hold, clear and gains are a request's.

Machine-agnostic: the port, id, folders and loop names come from the launcher.
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
from waxx.util.guis.intensity_pid import intensity_pid_protocol as proto

__all__ = ["IntensityPIDServer", "IntensityPIDLink", "GainStore", "DEFAULT_BAUD"]

log = logging.getLogger("waxx.intensity_pid.server")

DEFAULT_BAUD = 115200
PLOT_BLOCK_S = 0.1          # history: 100 ms means
LOG_BLOCK_S = 1.0           # log: 1 s means
HISTORY_MAX = 3000          # 100 ms means kept (5 min)
HISTORY_REPLY_MAX = 3000
TEXT_MAX = 400
QUERY_TIMEOUT_S = 1.0
RAW_COLLECT_S = 0.4
HOUSEKEEPING_S = 1.0
LOG_FLUSH_S = 10.0
LOG_BUFFER_MAX = 20000
REAPPLY_WARN_S = 600.0
_SHUTDOWN_REPLY_GRACE_S = 0.2


def _open_serial(port: str, baud: Optional[int]):
    import serial  # noqa: PLC0415
    return serial.Serial(port, baud or DEFAULT_BAUD, timeout=0.2, write_timeout=1.0)


class IntensityPIDLink(ComLink):
    """The board's serial port, opened on the link's thread."""

    def __init__(self, owner: "IntensityPIDServer", port: str, baud: int,
                 serial_factory: Callable[[str, Optional[int]], Any]):
        super().__init__(port, baud=baud, server=owner, name=f"intensity PID {port}", logger=log)
        self._owner = owner
        self._factory = serial_factory

    def _open(self):
        return self._factory(self.port, self.baud)

    def _on_connected(self, device):
        self._owner._on_link_up()


class GainStore:
    """The gains last set through the server, in a small JSON file (local
    disk), so they can be re-applied after the board resets."""

    def __init__(self, path: Optional[str]):
        self.path = path

    def load(self) -> Optional[dict]:
        """{name: float} for every gain, or None when there is no usable file."""
        if not self.path or not os.path.exists(self.path):
            return None
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        gains = data.get("gains") if isinstance(data, dict) else None
        if not isinstance(gains, dict):
            raise ValueError(f"{self.path}: no 'gains' object")
        out = {}
        for name in proto.GAIN_NAMES:
            v = gains.get(name)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{self.path}: {name} missing or not a number")
            out[name] = float(v)
        return out

    def save(self, gains: dict, *, source: str) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"gains": gains, "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
                       "source": source}, fh, indent=1)
        os.replace(tmp, self.path)


class _CommandError(RuntimeError):
    pass


class IntensityPIDServer(NetServer):
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
        gains_path: Optional[str] = None,
        loop_names: tuple[str, str] = ("loop 1", "loop 2"),
        limits: proto.OutputLimits = proto.OutputLimits(),
        stale_after_s: float = 2.0,
        serial_factory: Callable[[str, Optional[int]], Any] = _open_serial,
        on_shutdown_request: Optional[Callable[[], None]] = None,
    ):
        NetServer.__init__(self, server_id, port)
        self.server_id = server_id
        self.serial_port = serial_port
        self.host = host
        self.port = int(port)
        self.log_dir = log_dir
        self.loop_names = tuple(loop_names)
        self.limits = limits
        self.stale_after_s = float(stale_after_s)
        self.gain_store = GainStore(gains_path)
        self._on_shutdown_request = on_shutdown_request

        self._stop = threading.Event()
        self._link_up = threading.Event()
        self._lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        self._latest: Optional[dict] = None
        self._last_row_mono: Optional[float] = None
        self._rows_total = 0
        self._plot_bin = proto.Binner(PLOT_BLOCK_S)
        self._log_bin = proto.Binner(LOG_BLOCK_S)
        self._latest_mean: Optional[dict] = None
        self._latest_log_mean: Optional[dict] = None
        self._seq = 0
        self._history: collections.deque = collections.deque(maxlen=HISTORY_MAX)
        self._text_seq = 0
        self._text: collections.deque = collections.deque(maxlen=TEXT_MAX)
        self._reply_n = 0
        self._replies: dict[str, tuple[int, str]] = {}
        self._unknown: dict[str, int] = {}
        self._settings: dict[str, Optional[float]] = {s.name: None for s in proto.SETTINGS}
        self._label: Optional[str] = None
        self._gain_warning: Optional[dict] = None     # {"level", "code", "text", "until"}

        self._log_rows: list[dict] = []
        self._log_path: Optional[str] = None
        self._log_error: Optional[str] = None
        self._log_dropped = 0
        self._last_flush = time.monotonic()

        self._sock: Optional[socket.socket] = None
        self._threads: list[threading.Thread] = []
        self._shutdown_requested = False
        self.link = IntensityPIDLink(self, serial_port, baud, serial_factory)

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
        log.info("intensity PID server %s listening on %s:%d (port %s)", self.server_id,
                 self.host, self.port, self.serial_port)
        self.link.start()
        for target, name in ((self._read_loop, "ipid-read"),
                             (self._accept_loop, "ipid-accept"),
                             (self._housekeeping_loop, "ipid-house")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        """Flush the log, close the port and the socket.  Idempotent."""
        if self._stop.is_set():
            return
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
        threading.Thread(target=self._configure_device, name="ipid-config", daemon=True).start()

    def _configure_device(self) -> None:
        """After every connect: gains (re-applied if the board lost them),
        board label."""
        time.sleep(0.2)
        try:
            board = self.refresh_settings()
            self._reapply_gains(board)
            self._label = self._ping()
            log.info("intensity PID connected on %s (board %r)", self.serial_port, self._label)
        except Exception as exc:
            (log.debug if self._stop.is_set() or not self.link.connected else log.warning)(
                "intensity PID: setup after connect failed: %s", exc)

    def _reapply_gains(self, board: dict) -> None:
        """Gains only: an override is never re-applied (a reset leaves the PID driving)."""
        board = {k: board.get(k) for k in proto.GAIN_NAMES}
        try:
            saved = self.gain_store.load()
        except Exception as exc:
            self._set_gain_warning("error", "gains_file",
                                   f"could not read the saved gains ({exc}); nothing re-applied")
            log.warning("gain store unreadable: %s", exc)
            return
        if any(v is None for v in board.values()):
            self._set_gain_warning("error", "gains_read",
                                   "could not read the board's gains; nothing re-applied")
            return
        if saved is None:
            # first run: what the board has now is the reference
            try:
                self.gain_store.save({k: float(v) for k, v in board.items()}, source="board at connect")
            except OSError as exc:
                log.warning("could not save gains: %s", exc)
            return
        tol = 1.5 * proto.READBACK_RESOLUTION
        differ = {k: (board[k], v) for k, v in saved.items() if abs(board[k] - v) > tol}
        if not differ:
            return
        failed = []
        for name, (_old, new) in differ.items():
            try:
                self._query(name, proto.format_value(proto.SETTABLE[name], new))
            except Exception as exc:
                failed.append(f"{name}: {exc}")
        listing = ", ".join(f"{k} {old:g} -> {new:g}" for k, (old, new) in differ.items())
        if failed:
            self._set_gain_warning("error", "gains_reapply",
                                   f"gain re-apply failed ({'; '.join(failed)}); tried {listing}")
            log.error("gain re-apply failed: %s (%s)", failed, listing)
        else:
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            self._set_gain_warning("warn", "gains_reapplied",
                                   f"{stamp}: the board came up with other gains (reset or "
                                   f"reflash?); re-applied the saved ones: {listing}")
            log.warning("board gains differed from the saved ones; re-applied: %s", listing)

    def _set_gain_warning(self, level: str, code: str, text: str) -> None:
        with self._lock:
            self._gain_warning = {"level": level, "code": code, "text": text,
                                  "until": time.monotonic() + REAPPLY_WARN_S}

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
                row["t"] = time.time()
                self._latest = row
                self._last_row_mono = time.monotonic()
                self._rows_total += 1
                mean = self._plot_bin.add(row)
                if mean is not None:
                    self._seq += 1
                    mean["seq"] = self._seq
                    self._latest_mean = mean
                    self._history.append(mean)
                lmean = self._log_bin.add(row)
                if lmean is not None:
                    for k in ("override1", "override2"):
                        lmean[k] = self._settings.get(k)
                    self._latest_log_mean = lmean
                    if len(self._log_rows) < LOG_BUFFER_MAX:
                        self._log_rows.append(lmean)
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
                    self._settings[name] = proto.parse_setting(proto.SETTABLE[name], text)
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
                        raise _CommandError(f"the firmware does not know {name!r} "
                                            "(old sketch on the board?)")
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise _CommandError(f"no reply to {name!r} within {timeout:g} s")
                    self._cond.wait(left)

    def _wait_text(self, sent: str, match: Callable[[str], bool], unknown_name: str,
                   timeout: float = QUERY_TIMEOUT_S) -> str:
        """Send ``sent``; return the first new console line ``match`` accepts."""
        with self._cmd_lock:
            with self._cond:
                start = self._text_seq
            self._write(sent)
            deadline = time.monotonic() + timeout
            with self._cond:
                while time.monotonic() < deadline:
                    for item in self._text:
                        if item["seq"] > start:
                            if match(item["line"]):
                                return item["line"]
                            if proto.parse_unknown(item["line"]) == unknown_name:
                                raise _CommandError(f"the firmware has no {unknown_name!r} command")
                    self._cond.wait(0.05)
        raise _CommandError(f"no reply to {sent!r}")

    def _ping(self) -> Optional[str]:
        """The board's stored label (``ping`` prints one bare line)."""
        try:
            return self._wait_text("ping", lambda l: proto.parse_reply(l) is None
                                   and proto.parse_unknown(l) is None, "ping",
                                   timeout=0.6)[:64]
        except _CommandError:
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
            if cmd == "hold":
                return self._hold(req.get("on"))
            if cmd == "clear":
                with self._cmd_lock:
                    self._write("c")
                log.info("integrators cleared (c sent)")
                return {"status": "ok", "message": "integrators cleared"}
            if cmd == "set":
                return self._set(str(req.get("name", "")), req.get("value"))
            if cmd == "override":
                return self._override(req.get("loop"), req.get("on"), req.get("volt"))
            if cmd == "refresh_settings":
                return {"status": "ok", "settings": self.refresh_settings()}
            if cmd == "raw":
                return self._raw(str(req.get("line", "")))
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
        now = time.monotonic()
        with self._lock:
            latest = dict(self._latest) if self._latest else None
            mean = dict(self._latest_mean) if self._latest_mean else None
            lmean = self._latest_log_mean
            age = None if self._last_row_mono is None else now - self._last_row_mono
            settings = dict(self._settings)
            seq = self._seq
            gw = self._gain_warning
            if gw is not None and now > gw["until"]:
                self._gain_warning = gw = None
        com = self.link.snapshot()
        extra = [] if gw is None else [{k: gw[k] for k in ("level", "code", "text")}]
        warnings = proto.compute_warnings(latest, limits=self.limits, com_status=com["status"],
                                          data_age_s=age, stale_after_s=self.stale_after_s,
                                          loop_names=self.loop_names, settings=settings,
                                          extra=extra)
        return {
            "status": "ok",
            "ok": True,
            "server_id": self.server_id,
            "com": com,
            "connected": self.link.connected,
            "label": self._label,
            "latest": latest,
            "mean": mean,
            "rate_hz": (lmean["n"] / self._log_bin.width_s
                        if lmean is not None and self._log_bin.width_s > 0 else None),
            "held": None if latest is None else bool(latest.get("hold")),
            "data_age_s": None if age is None else round(age, 3),
            "settings": settings,
            "loop_names": list(self.loop_names),
            "limits": {"low": self.limits.low, "high": self.limits.high},
            "warnings": warnings,
            "seq": seq,
            "log_path": self._log_path,
            "log_error": self._log_error,
            "log_dropped": self._log_dropped,
            "gains_path": self.gain_store.path,
        }

    def _get_history(self, since: int, max_rows: int) -> dict:
        max_rows = max(1, min(int(max_rows), HISTORY_REPLY_MAX))
        with self._lock:
            rows = [r for r in self._history if r["seq"] > since]
            seq = self._seq
        return {"status": "ok", "rows": rows[-max_rows:], "seq": seq}

    def _hold(self, on: Any) -> dict:
        if on is None:                         # toggle: read, then set the other
            now = proto.parse_switch_reply(self._query("hold"))
            if now is None:
                raise _CommandError("hold: unexpected reply")
            on = not now
        elif not (isinstance(on, bool) or on in (0, 1)):
            raise ValueError("hold: on must be true, false or null (toggle)")
        state = proto.parse_switch_reply(self._query("hold", "1" if on else "0"))
        if state is None:
            raise _CommandError("hold: unexpected reply")
        log.info("hold -> %s", int(state))
        # the next status line shows it; until then the latest row says so too
        with self._lock:
            if self._latest is not None:
                self._latest["hold"] = int(state)
        return {"status": "ok", "name": "hold", "on": state}

    def _set(self, name: str, value: Any) -> dict:
        setting = proto.SETTABLE.get(name)
        if setting is None:
            raise ValueError(f"not a setting: {name!r} (one of {', '.join(proto.SETTABLE)})")
        if value is None:
            raise ValueError(f"no value for {name}")
        text = proto.format_value(setting, value, self.limits)
        reply = proto.parse_setting(setting, self._query(name, text))
        log.info("set %s = %s (firmware reads back %s)", name, text, reply)
        out = {"status": "ok", "name": name, "value": reply}
        if setting.kind != "gain":
            return out
        self._remember_gains(f"set {name}", {name: float(text)})
        v = float(text)
        if reply is not None and v != 0 and abs(reply - v) > 0.01 * abs(v):
            out["note"] = (f"the board reads back {reply:g}: its replies have 6 decimals; the "
                           f"board itself has {v:g}, and the server remembers {v:g}")
        return out

    def _override(self, loop: Any, on: Any, volt: Any = None) -> dict:
        """Loop 1 or 2: ``volt`` (if given) becomes ovolt, then override on / off."""
        if loop not in (1, 2):
            raise ValueError("override: loop must be 1 or 2")
        if on is None:
            raise ValueError("override: on must be true or false")
        out: dict = {"status": "ok", "loop": loop}
        if volt is not None:
            out["volt"] = self._set(f"ovolt{loop}", volt)["value"]
        out["on"] = bool(self._set(f"override{loop}", bool(on))["value"])
        with self._lock:
            out["volt"] = self._settings.get(f"ovolt{loop}")
        log.info("override%d -> %s at %s V", loop, int(out["on"]), out["volt"])
        return out

    def _remember_gains(self, source: str, sent: Optional[dict] = None) -> None:
        """Save the gains: the values just sent as sent (the board's replies
        round to 6 decimals), the others as the board reads them back."""
        with self._lock:
            gains = {k: self._settings.get(k) for k in proto.GAIN_NAMES}
        if any(v is None for v in gains.values()):
            board = self.refresh_settings()
            gains = {k: board.get(k) for k in proto.GAIN_NAMES}
        gains.update(sent or {})
        if any(v is None for v in gains.values()):
            log.warning("gains not saved: could not read them all (%s)", gains)
            return
        try:
            self.gain_store.save({k: float(v) for k, v in gains.items()}, source=source)
        except OSError as exc:
            log.warning("could not save gains: %s", exc)
        with self._lock:
            if self._gain_warning is not None and self._gain_warning["code"] == "gains_reapplied":
                self._gain_warning = None      # a person has set the gains since

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
        parts = line.split()
        if len(parts) >= 2 and parts[0] in proto.GAIN_NAMES:
            try:
                sent = {parts[0]: float(proto.format_value(proto.SETTABLE[parts[0]], parts[1]))}
            except ValueError:
                sent = None
            self._remember_gains(f"raw {parts[0]}", sent)
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

            threading.Thread(target=_later, name="ipid-shutdown", daemon=True).start()
        return {"status": "ok"}

    # ------------------------------------------------------------------ #
    # TCP
    # ------------------------------------------------------------------ #

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
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
                req = {"cmd": text}
            if not isinstance(req, dict):
                req = {"cmd": ""}
            reply = self.handle(req)
            payload = json.dumps(_nan_to_none(reply), allow_nan=False, default=str)
            conn.sendall((payload + "\n").encode("utf-8"))
        except Exception:
            log.exception("connection failed")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # Housekeeping: daily log
    # ------------------------------------------------------------------ #

    def _housekeeping_loop(self) -> None:
        while not self._stop.wait(HOUSEKEEPING_S):
            self._flush_log()

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
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            for day, day_rows in by_day.items():
                path = os.path.join(self.log_dir, f"{day}.csv")
                new = not os.path.exists(path)
                # opened and closed each flush: a network share's stale handle
                # never outlives one write
                with open(path, "a", encoding="utf-8", newline="") as fh:
                    if new:
                        fh.write("# 1 s means of the board's status lines; n = lines in the "
                                 "second, *_sd = their sample SD, hold_frac = fraction held; "
                                 "override1/2 = the server's last reading of the switch when "
                                 "the second closed; t = the last line's time\n")
                        fh.write("time_iso,unix_s," + ",".join(proto.LOG_FIELDS) + "\n")
                    for r in day_rows:
                        iso = datetime.datetime.fromtimestamp(r["t"]).isoformat(timespec="milliseconds")
                        vals = [_csv(r.get(f)) for f in proto.LOG_FIELDS]
                        fh.write(f"{iso},{r['t']:.3f}," + ",".join(vals) + "\n")
                self._log_path = path
            if self._log_error is not None:
                log.info("log writing again (%s)", self._log_path)
            self._log_error = None
        except OSError as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if msg != self._log_error:
                log.warning("log write failed (%s); rows kept for the next try", msg)
            self._log_error = msg
            with self._lock:
                keep = LOG_BUFFER_MAX - len(self._log_rows)
                if keep < len(rows):
                    self._log_dropped += len(rows) - max(keep, 0)
                    rows = rows[-max(keep, 0):] if keep > 0 else []
                self._log_rows = rows + self._log_rows


def _csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return str(v)


def _to_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return None


def _nan_to_none(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return obj
