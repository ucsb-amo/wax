"""DailyCsvLog: append-only CSV history, one file per local day.

Built for servers that must never block on file I/O (the interlock poll loop):
``add()`` only queues a row in memory; a background thread appends queued rows
to ``<directory>/YYYY-MM-DD.csv`` every ``interval_s``. Files are only ever
opened in append mode -- nothing already written is rewritten or truncated.

Failure handling
----------------
* A failed write (share down, disk full) keeps the unwritten rows queued and
  retries them on the next flush. The failure is logged once per outage, and
  recovery is logged with the number of rows it caught up.
* The queue is bounded (``max_pending`` rows). When a long outage fills it the
  oldest rows are dropped, and every drop is logged with its count.
* If a write fails part-way through a day's rows, those rows are retried, so
  the file can hold a torn last line or rows written twice (same ``epoch``).
  Duplicates can be removed on read; lost rows cannot be recovered, so the
  retry errs toward duplicates.

The day of a row is the local date of its ``epoch`` (``time.localtime``), so a
flush that spans midnight splits across two files.
"""

from __future__ import annotations

import collections
import csv
import io
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

_LOG = logging.getLogger("waxx.util.daily_csv")


def local_day(epoch: float) -> str:
    """``YYYY-MM-DD`` of *epoch* in local time -- the file a row belongs to."""
    return time.strftime("%Y-%m-%d", time.localtime(epoch))


class DailyCsvLog:
    """Queue rows from any thread; a background thread appends them to disk.

    Args:
        directory: folder holding the ``YYYY-MM-DD.csv`` files.
        header: column names, written once at the top of each new file.
        max_pending: rows kept in memory while the disk is unreachable.
        name: thread name and log label.
    """

    def __init__(self, directory: str | Path, header: Sequence[str], *,
                 max_pending: int = 50_000, name: str = "daily-csv",
                 logger: Optional[logging.Logger] = None):
        self.directory = Path(directory)
        self.header = list(header)
        self.name = name
        self._log = logger or _LOG
        self._max_pending = int(max_pending)
        self._pending: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._dropped = 0
        self._failing = False
        self._flush_lock = threading.Lock()   # one flush at a time
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- producer side (cheap, no I/O) --------------------------------

    def add(self, epoch: float, row: Sequence) -> None:
        """Queue one row stamped *epoch* (seconds since the Unix epoch)."""
        with self._lock:
            if len(self._pending) >= self._max_pending:
                self._pending.popleft()
                self._dropped += 1
            self._pending.append((epoch, row))

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- writer side --------------------------------------------------

    def flush(self) -> int:
        """Append every queued row to its day's file.  Never raises.

        Returns the number of rows written.
        """
        with self._flush_lock:
            with self._lock:
                batch = list(self._pending)
                self._pending.clear()
                dropped, self._dropped = self._dropped, 0
            if dropped:
                self._log.warning("%s: queue full while the disk was unreachable; "
                                  "dropped the %d oldest rows", self.name, dropped)
            if not batch:
                return 0

            by_day: dict[str, list] = {}
            for item in batch:
                by_day.setdefault(local_day(item[0]), []).append(item)

            written = 0
            unwritten: list = []
            error: Optional[BaseException] = None
            for day, items in by_day.items():
                if error is not None:
                    unwritten.extend(items)
                    continue
                try:
                    self._append(day, [row for _, row in items])
                    written += len(items)
                except Exception as exc:   # keep the rows for the next flush
                    error = exc
                    unwritten.extend(items)

            if unwritten:
                self._requeue(unwritten)
            if error is not None:
                if not self._failing:
                    self._failing = True
                    self._log.error("%s: append to %s failed; %d rows kept for retry: %r",
                                    self.name, self.directory, len(unwritten), error)
                else:
                    self._log.debug("%s: append still failing: %r", self.name, error)
            elif self._failing:
                self._failing = False
                self._log.warning("%s: appends to %s recovered (%d rows written)",
                                  self.name, self.directory, written)
            return written

    def _append(self, day: str, rows: list) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{day}.csv"
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        if not path.exists() or path.stat().st_size == 0:
            writer.writerow(self.header)
        writer.writerows(rows)
        # One write per file keeps a failure from leaving half a batch.
        with open(path, "a", newline="", encoding="utf-8") as f:
            f.write(buf.getvalue())

    def _requeue(self, items: list) -> None:
        """Put unwritten rows back in front of rows queued since the flush."""
        with self._lock:
            merged = list(items) + list(self._pending)
            over = len(merged) - self._max_pending
            if over > 0:
                merged = merged[over:]
                self._dropped += over
            self._pending = collections.deque(merged)

    # -- background thread --------------------------------------------

    def start(self, interval_s: float = 10.0) -> None:
        """Flush every *interval_s* on a daemon thread until ``stop()``."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(float(interval_s),),
                                        name=self.name, daemon=True)
        self._thread.start()

    def _loop(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            self.flush()
        self.flush()   # rows queued since the last tick

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the thread after a final flush (waits at most *timeout* s)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self._log.warning("%s: final flush still running after %.1f s; "
                                  "%d rows may not be written", self.name, timeout,
                                  self.pending)


__all__ = ["DailyCsvLog", "local_day"]
