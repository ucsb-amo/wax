"""K-machine HMR2300 magnetometer field -> Zabbix.

The magnetometer server (``waxx.util.guis.HMR_magnetometer``) appends one
reading per second to a daily field log,
``<data>/magnetometer_data/hmr2300_YYYY-MM-DD.csv`` (gauss).
:class:`MagnetometerPusher` tails those files and pushes the ``Bx``, ``By``,
``Bz`` and ``Btot`` columns to Zabbix *trapper* items, keeping each reading's
own timestamp and value (decimated, never averaged).  It only reads the
files, so it never touches the magnetometer and can run on any PC that sees
the data drive.

>>> from waxa.climate.magnetometer import MagnetometerPusher
>>> p = MagnetometerPusher()             # host "K", keys k.magnetometer.*
>>> p.push_new()                         # send everything new since the last call

Command line::

    python -m waxa.climate.magnetometer latest        # newest reading, nothing sent
    python -m waxa.climate.magnetometer push --once   # send new readings once

Zabbix side (one-time, needs a Zabbix admin): on host ``K`` (technical name
``Vertiv Geist 100-P 4``) add four items of type *Zabbix trapper*, type of
information *Numeric (float)*, units ``G``, with keys
``k.magnetometer.bx``, ``k.magnetometer.by``, ``k.magnetometer.bz`` and
``k.magnetometer.btot``.  A value for a key that has no item comes back as
``failed``.
"""
from __future__ import annotations

import csv
import datetime as _dt
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from waxa.climate.coil import DEFAULT_ZABBIX_HOST, last_pushed_clock
from waxa.climate.sender import SenderResult, TrapperValue, ZabbixSender

_LOG = logging.getLogger("waxa.climate.magnetometer")

# Field-log column -> trapper item key.
DEFAULT_KEYS = {
    "Bx": "k.magnetometer.bx",
    "By": "k.magnetometer.by",
    "Bz": "k.magnetometer.bz",
    "Btot": "k.magnetometer.btot",
}
TIME_COLUMN = "timestamp_s"
# Must match FIELD_LOG_FILENAME in waxx's hmr_magnetometer_server (waxa cannot import waxx).
FIELD_LOG_FILENAME = "hmr2300_{date}.csv"
MAX_PENDING = 20_000   # readings kept for retry while the server is unreachable (~2.3 days at 10 s)


def default_log_dir() -> Path:
    """``<data>/magnetometer_data``, where the magnetometer server writes its field log."""
    data = os.environ.get("data") or os.environ.get("DATA_DIR")
    if not data:
        raise RuntimeError("Neither %data% nor DATA_DIR is set; pass log_dir explicitly")
    return Path(data) / "magnetometer_data"


def day_file(log_dir, day: _dt.date) -> Path:
    return Path(log_dir) / FIELD_LOG_FILENAME.format(date=day.isoformat())


@dataclass(frozen=True)
class FieldReading:
    """One field-log row: unix time and ``{column: gauss}``."""
    epoch: float
    fields: dict


def _parse_rows(lines: list[str], header: list[str], columns) -> list[FieldReading]:
    try:
        i_t = header.index(TIME_COLUMN)
        idx = {c: header.index(c) for c in columns}
    except ValueError:
        raise ValueError(f"field log header lacks {TIME_COLUMN}/{list(columns)}: {header}")
    out = []
    for row in csv.reader(lines):
        if len(row) <= max(i_t, *idx.values()):
            continue
        try:
            out.append(FieldReading(float(row[i_t]), {c: float(row[i]) for c, i in idx.items()}))
        except ValueError:
            continue
    return out


def read_day(log_dir, day: _dt.date, columns=tuple(DEFAULT_KEYS)) -> list[FieldReading]:
    """Every reading in one daily field log (empty if it does not exist)."""
    path = day_file(log_dir, day)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        return []
    return _parse_rows(lines[1:], next(csv.reader([lines[0]])), columns)


def latest_reading(log_dir=None) -> Optional[FieldReading]:
    """Newest reading from today's (or else yesterday's) field log."""
    log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
    today = _dt.date.today()
    for day in (today, today - _dt.timedelta(days=1)):
        rows = read_day(log_dir, day)
        if rows:
            return rows[-1]
    return None


def last_pushed_clocks(host: str = DEFAULT_ZABBIX_HOST, keys=None, api=None) -> dict:
    """``{key: unix time of the newest value Zabbix holds, or None}`` (read-only, guest)."""
    keys = list(DEFAULT_KEYS.values()) if keys is None else list(keys)
    if api is None:
        from waxa.climate.zabbix import ZabbixAPI   # noqa: PLC0415
        api = ZabbixAPI()
    return {k: last_pushed_clock(host, k, api=api) for k in keys}


class _Tail:
    """Incremental reader of one growing CSV: returns only complete new lines."""

    def __init__(self, path: Path, columns):
        self.path = path
        self.columns = columns
        self.offset = 0
        self.header: Optional[list[str]] = None

    def read_new(self) -> list[FieldReading]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return []   # nothing complete yet
        self.offset += end + 1
        lines = chunk[:end + 1].decode("utf-8", errors="replace").splitlines()
        if self.header is None:
            if not lines:
                return []
            self.header = next(csv.reader([lines[0]]))
            lines = lines[1:]
        return _parse_rows(lines, self.header, self.columns)


class MagnetometerPusher:
    """Tail the magnetometer field logs and push the field to Zabbix.

    Parameters
    ----------
    sender : ZabbixSender, optional
        Default: the lab server's trapper port.
    log_dir : path, optional
        Default: :func:`default_log_dir`.
    host : str
        Zabbix host (technical name).
    keys : dict, optional
        Field-log column -> trapper item key.  Default: :data:`DEFAULT_KEYS`.
    min_interval_s : float
        Send at most one reading per this many seconds (the log has one per
        second).  A sent reading is one log row as written, not an average.
    backfill_s : float
        On the first call, also send readings up to this old (0 = start
        from the newest reading only).
    resume_after : float, optional
        Unix time of the last reading Zabbix already holds.  On the first
        call, send every reading in today's file newer than this, so a
        restart leaves no gap and sends nothing twice.  Overrides
        ``backfill_s``.
    """

    def __init__(self, sender: Optional[ZabbixSender] = None, log_dir=None,
                 host: str = DEFAULT_ZABBIX_HOST, keys: Optional[dict] = None,
                 min_interval_s: float = 10.0, backfill_s: float = 0.0,
                 resume_after: Optional[float] = None):
        self.sender = sender if sender is not None else ZabbixSender()
        self.log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
        self.host = host
        self.keys = dict(DEFAULT_KEYS if keys is None else keys)
        self.min_interval_s = float(min_interval_s)
        self.backfill_s = float(backfill_s)
        self.resume_after = resume_after
        self._tail: Optional[_Tail] = None
        self._day: Optional[_dt.date] = None
        self._pending: list[FieldReading] = []
        self._last_queued_epoch = float("-inf")
        self._started = False
        self._warned_failed = False

    @property
    def key(self) -> str:
        """All item keys, for status displays."""
        return ", ".join(self.keys.values())

    @property
    def pending(self) -> int:
        """Readings waiting for a reachable server."""
        return len(self._pending)

    def _collect(self, now: float) -> list[FieldReading]:
        today = _dt.date.fromtimestamp(now)
        new: list[FieldReading] = []
        if self._day != today:
            if self._tail is not None:
                new += self._tail.read_new()   # rows written to yesterday's file before midnight
            self._day = today
            self._tail = _Tail(day_file(self.log_dir, today), tuple(self.keys))
        new += self._tail.read_new()
        if not self._started:
            self._started = True
            if self.resume_after is not None:
                new = [r for r in new if r.epoch > self.resume_after]
                self._last_queued_epoch = self.resume_after
            elif new:
                cutoff = new[-1].epoch - self.backfill_s
                new = [r for r in new if r.epoch >= cutoff] if self.backfill_s > 0 else new[-1:]
        return new

    def _decimate(self, rows: list[FieldReading]) -> list[FieldReading]:
        out = []
        for r in rows:
            if r.epoch - self._last_queued_epoch >= self.min_interval_s:
                out.append(r)
                self._last_queued_epoch = r.epoch
        return out

    def push_new(self, now: Optional[float] = None) -> Optional[SenderResult]:
        """Send readings that appeared since the last call.  ``None`` if there was nothing to send.

        If the server is unreachable the readings are kept (up to
        :data:`MAX_PENDING`) and sent on a later call.
        """
        now = time.time() if now is None else now
        self._pending += self._decimate(self._collect(now))
        if len(self._pending) > MAX_PENDING:
            self._pending = self._pending[-MAX_PENDING:]
        if not self._pending:
            return None
        values = [TrapperValue(self.host, key, r.fields[col], r.epoch)
                  for r in self._pending for col, key in self.keys.items()]
        result = self.sender.send(values)   # ConnectionError -> readings stay pending
        self._pending = []
        if result.failed and not self._warned_failed:
            self._warned_failed = True
            _LOG.warning(
                "Zabbix refused %d/%d magnetometer values (%s). Does host %r have "
                "Zabbix trapper items with keys %s?",
                result.failed, result.total, "; ".join(result.info), self.host, self.key)
        elif result.ok and self._warned_failed:
            self._warned_failed = False
            _LOG.info("Zabbix is accepting magnetometer values again")
        return result


def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m waxa.climate.magnetometer",
                                description="Push the K magnetometer field (field log) to Zabbix.")
    p.add_argument("--log-dir", default=None,
                   help="field-log folder (default: %%data%%/magnetometer_data)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("latest", help="print the newest reading; sends nothing")
    ps = sub.add_parser("push", help="send readings to Zabbix")
    ps.add_argument("--once", action="store_true", help="send what is new and exit")
    ps.add_argument("--interval", type=float, default=30.0, help="seconds between sends (default 30)")
    ps.add_argument("--min-spacing", type=float, default=10.0,
                    help="min seconds between sent readings (default 10)")
    ps.add_argument("--host", default=DEFAULT_ZABBIX_HOST,
                    help=f"Zabbix host (default {DEFAULT_ZABBIX_HOST!r})")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    if args.cmd == "latest":
        r = latest_reading(args.log_dir)
        if r is None:
            print("no magnetometer readings today or yesterday")
            return 1
        fields = "  ".join(f"{c}={v:+.5f} G" for c, v in r.fields.items())
        print(f"{_dt.datetime.fromtimestamp(r.epoch):%Y-%m-%d %H:%M:%S}  {fields}")
        return 0

    pusher = MagnetometerPusher(log_dir=args.log_dir, host=args.host,
                                min_interval_s=args.min_spacing)
    while True:
        try:
            r = pusher.push_new()
            print("nothing to send" if r is None else "; ".join(r.info))
        except ConnectionError as exc:
            _LOG.warning("%s (%d readings pending)", exc, pusher.pending)
        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["DEFAULT_KEYS", "DEFAULT_ZABBIX_HOST", "FIELD_LOG_FILENAME", "FieldReading",
           "MagnetometerPusher", "default_log_dir", "last_pushed_clocks", "latest_reading",
           "read_day"]
