"""K-machine magnet coil temperature -> Zabbix.

The interlock server (``kexp.util.guis.interlock``) reads the coil
temperature from the interlock PLC and appends every good frame to a daily
history file, ``<data>/interlock_logs/YYYY-MM-DD.csv``, flushed about every
10 s.  :class:`CoilTemperaturePusher` tails those files and pushes the
``temperature_c`` column to a Zabbix *trapper* item, keeping each reading's
own timestamp.  It only reads the files, so it never touches the interlock
itself and can run on any PC that sees the data drive.

>>> from waxa.climate.coil import CoilTemperaturePusher
>>> p = CoilTemperaturePusher()          # host "K", key "k.coil.temperature"
>>> p.push_new()                         # send everything new since the last call
>>> p.run_forever(interval_s=30)

Command line::

    python -m waxa.climate.coil latest            # newest reading, nothing sent
    python -m waxa.climate.coil push --once       # send new readings once
    python -m waxa.climate.coil push              # keep sending (Ctrl-C stops)

Zabbix side (one-time, needs a Zabbix admin): on host ``K`` (technical name
``Vertiv Geist 100-P 4``) add an item of type *Zabbix trapper*, key
``k.coil.temperature``, type of information *Numeric (float)*, units ``C``.
Until it exists every value comes back as ``failed``.
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

from waxa.climate.sender import SenderResult, TrapperValue, ZabbixSender

_LOG = logging.getLogger("waxa.climate.coil")

DEFAULT_ZABBIX_HOST = "Vertiv Geist 100-P 4"   # technical name of the "K" host
DEFAULT_KEY = "k.coil.temperature"
TEMPERATURE_COLUMN = "temperature_c"
MAX_PENDING = 100_000   # readings kept for retry while the server is unreachable (~1.5 days)


def default_log_dir() -> Path:
    """``<data>/interlock_logs``, the same env vars the interlock server uses."""
    data = os.environ.get("data") or os.environ.get("DATA_DIR")
    if not data:
        raise RuntimeError("Neither %data% nor DATA_DIR is set; pass log_dir explicitly")
    return Path(data) / "interlock_logs"


def day_file(log_dir, day: _dt.date) -> Path:
    return Path(log_dir) / f"{day.isoformat()}.csv"


@dataclass(frozen=True)
class CoilReading:
    epoch: float
    temperature_c: float


def _parse_rows(lines: list[str], header: list[str]) -> list[CoilReading]:
    out = []
    try:
        i_t = header.index("epoch")
        i_v = header.index(TEMPERATURE_COLUMN)
    except ValueError:
        raise ValueError(f"interlock history header lacks epoch/{TEMPERATURE_COLUMN}: {header}")
    for row in csv.reader(lines):
        if len(row) <= max(i_t, i_v) or not row[i_v].strip():
            continue   # blank cell = temperature absent from that PLC frame
        try:
            out.append(CoilReading(float(row[i_t]), float(row[i_v])))
        except ValueError:
            continue
    return out


def read_day(log_dir, day: _dt.date) -> list[CoilReading]:
    """Every coil reading in one daily history file (empty if it does not exist)."""
    path = day_file(log_dir, day)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        return []
    return _parse_rows(lines[1:], next(csv.reader([lines[0]])))


def latest_reading(log_dir=None) -> Optional[CoilReading]:
    """Newest coil reading from today's (or else yesterday's) history file."""
    log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
    today = _dt.date.today()
    for day in (today, today - _dt.timedelta(days=1)):
        rows = read_day(log_dir, day)
        if rows:
            return rows[-1]
    return None


def last_pushed_clock(host: str = DEFAULT_ZABBIX_HOST, key: str = DEFAULT_KEY,
                      api=None) -> Optional[float]:
    """Unix time of the newest value Zabbix holds for the trapper item.

    Read-only (guest login).  ``None`` if the item does not exist or has
    never received a value.
    """
    from waxa.climate.zabbix import ZabbixAPI   # noqa: PLC0415

    api = api if api is not None else ZabbixAPI()
    rows = api.call("item.get", {"output": ["itemid", "lastclock"], "host": host,
                                 "filter": {"key_": key}})
    if not rows:
        return None
    clock = float(rows[0].get("lastclock") or 0)
    return clock if clock > 0 else None


class _Tail:
    """Incremental reader of one growing CSV: returns only complete new lines."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.header: Optional[list[str]] = None

    def read_new(self) -> list[CoilReading]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return []   # nothing complete yet (the writer flushes whole rows)
        self.offset += end + 1
        lines = chunk[:end + 1].decode("utf-8", errors="replace").splitlines()
        if self.header is None:
            if not lines:
                return []
            self.header = next(csv.reader([lines[0]]))
            lines = lines[1:]
        return _parse_rows(lines, self.header)


class CoilTemperaturePusher:
    """Tail the interlock history files and push coil temperature to Zabbix.

    Parameters
    ----------
    sender : ZabbixSender, optional
        Default: the lab server's trapper port.
    log_dir : path, optional
        Default: :func:`default_log_dir`.
    host, key : str
        Zabbix host (technical name) and trapper item key.
    min_interval_s : float
        Send at most one reading per this many seconds.  The PLC reports
        about once a second; the room sensors are once a minute.
    backfill_s : float
        On the first call, also send readings up to this old (0 = start
        from the newest reading only).
    resume_after : float, optional
        Unix time of the last reading Zabbix already holds
        (:func:`last_pushed_clock`).  On the first call, send every reading
        in today's file newer than this, so a restart leaves no gap and
        sends nothing twice.  Overrides ``backfill_s``.
    """

    def __init__(self, sender: Optional[ZabbixSender] = None, log_dir=None,
                 host: str = DEFAULT_ZABBIX_HOST, key: str = DEFAULT_KEY,
                 min_interval_s: float = 10.0, backfill_s: float = 0.0,
                 resume_after: Optional[float] = None):
        self.sender = sender if sender is not None else ZabbixSender()
        self.log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
        self.host = host
        self.key = key
        self.min_interval_s = float(min_interval_s)
        self.backfill_s = float(backfill_s)
        self.resume_after = resume_after
        self._tail: Optional[_Tail] = None
        self._day: Optional[_dt.date] = None
        self._pending: list[CoilReading] = []
        self._last_queued_epoch = float("-inf")
        self._started = False
        self._warned_failed = False

    def _collect(self, now: float) -> list[CoilReading]:
        today = _dt.date.fromtimestamp(now)
        new: list[CoilReading] = []
        if self._day != today:
            if self._tail is not None:
                new += self._tail.read_new()   # rows written to yesterday's file before midnight
            self._day = today
            self._tail = _Tail(day_file(self.log_dir, today))
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

    @property
    def pending(self) -> int:
        """Readings waiting for a reachable server."""
        return len(self._pending)

    def _decimate(self, rows: list[CoilReading]) -> list[CoilReading]:
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
        values = [TrapperValue(self.host, self.key, round(r.temperature_c, 4), r.epoch)
                  for r in self._pending]
        result = self.sender.send(values)   # ConnectionError -> readings stay pending
        self._pending = []
        if result.failed and not self._warned_failed:
            self._warned_failed = True
            _LOG.warning(
                "Zabbix refused %d/%d coil readings (%s). Does host %r have a "
                "Zabbix trapper item with key %r?",
                result.failed, result.total, "; ".join(result.info), self.host, self.key)
        elif result.ok and self._warned_failed:
            self._warned_failed = False
            _LOG.info("Zabbix is accepting coil readings again")
        return result

    def run_forever(self, interval_s: float = 30.0) -> None:
        """Call :meth:`push_new` every ``interval_s`` until interrupted."""
        _LOG.info("pushing %s -> %s:%s host=%r key=%r every %.0f s",
                  self.log_dir, self.sender.server, self.sender.port,
                  self.host, self.key, interval_s)
        while True:
            try:
                r = self.push_new()
                if r is not None:
                    _LOG.debug("sent %d, failed %d", r.processed, r.failed)
            except ConnectionError as exc:
                _LOG.warning("%s (%d readings pending)", exc, len(self._pending))
            except Exception:  # noqa: BLE001 - keep the loop alive
                _LOG.exception("coil push failed")
            time.sleep(interval_s)


def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m waxa.climate.coil",
                                description="Push K coil temperature (interlock log) to Zabbix.")
    p.add_argument("--log-dir", default=None, help="interlock_logs folder (default: %%data%%/interlock_logs)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("latest", help="print the newest reading; sends nothing")
    ps = sub.add_parser("push", help="send readings to Zabbix")
    ps.add_argument("--once", action="store_true", help="send what is new and exit")
    ps.add_argument("--interval", type=float, default=30.0, help="seconds between sends (default 30)")
    ps.add_argument("--min-spacing", type=float, default=10.0, help="min seconds between sent readings (default 10)")
    ps.add_argument("--backfill", default="0", help="on start also send this much history, e.g. 90m, 6h (default 0)")
    ps.add_argument("--host", default=DEFAULT_ZABBIX_HOST, help=f"Zabbix host (default {DEFAULT_ZABBIX_HOST!r})")
    ps.add_argument("--key", default=DEFAULT_KEY, help=f"trapper item key (default {DEFAULT_KEY!r})")
    ps.add_argument("--server", default=None, help="Zabbix server (default: lab server)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    if args.cmd == "latest":
        r = latest_reading(args.log_dir)
        if r is None:
            print("no coil readings today or yesterday")
            return 1
        print(f"{_dt.datetime.fromtimestamp(r.epoch):%Y-%m-%d %H:%M:%S}  {r.temperature_c:.3f} C")
        return 0

    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    b = args.backfill.strip()
    backfill_s = float(b[:-1]) * mult[b[-1]] if b and b[-1] in mult else float(b)
    sender = ZabbixSender(args.server) if args.server else ZabbixSender()
    pusher = CoilTemperaturePusher(sender, args.log_dir, host=args.host, key=args.key,
                                   min_interval_s=args.min_spacing, backfill_s=backfill_s)
    if args.once:
        r = pusher.push_new()
        if r is None:
            print("nothing to send")
            return 0
        print("; ".join(r.info))
        return 0 if r.ok else 2
    try:
        pusher.run_forever(args.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["CoilReading", "CoilTemperaturePusher", "DEFAULT_KEY", "DEFAULT_ZABBIX_HOST",
           "default_log_dir", "last_pushed_clock", "latest_reading", "read_day"]
