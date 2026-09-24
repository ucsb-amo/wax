"""Logging for the liveOD process.

Everything the liveOD server, camera threads and GUI have to say goes through the
``waxx.live_od`` logger rather than ``print``. ``setup_logging`` (called once by
the acquisition window) attaches four handlers:

* a stream handler, so the terminal shows what it always showed,
* a rotating file under ``~/.waxx/logs``, for working out afterwards what happened,
* a :class:`QtLogHandler`, which re-emits every record as a Qt signal. Records are
  logged from the server, CameraBaby, DataHandler and SaveWorker threads; the
  signal is how they reach the GUI thread safely,
* a :class:`LogBuffer`, an in-memory ring of recent records tagged with the run
  they belong to. The server answers ``GET_LOG`` from it, so a process that was
  not watching while a run happened can still ask what became of it.

Until ``setup_logging`` is called (tests, or the server used without its window)
records of WARNING and above still reach stderr through logging's last-resort
handler, and nothing is written to disk. The buffer is attached on first use of
``get_log_buffer()`` either way, so the server can tag runs without the window.
"""

import logging
import os
import sys
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler

from PyQt6.QtCore import QObject, pyqtSignal

LOGGER_NAME = "waxx.live_od"
DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), ".waxx", "logs")

# The buffer's size. At the per-shot DEBUG rate (a few lines a shot) this is
# many hundreds of runs; the file under ~/.waxx/logs is the long-term record.
MAX_BUFFERED_RECORDS = 20_000
MAX_BUFFERED_RUNS = 500


def get_logger(name: str = "") -> logging.Logger:
    """The liveOD logger, or a child of it (``get_logger("server")``)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


class _Emitter(QObject):
    record = pyqtSignal(int, str, float)    # levelno, formatted message, record.created


class QtLogHandler(logging.Handler):
    """Re-emits log records as ``record_signal(levelno, message, created)``.

    The signal belongs to a QObject created on the GUI thread, so a slot connected
    to it runs on the GUI thread whichever thread logged the record.
    """

    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self._emitter = _Emitter()
        self.record_signal = self._emitter.record
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            self._emitter.record.emit(record.levelno, self.format(record), record.created)
        except RuntimeError:
            pass    # the QObject is gone: the application is shutting down
        except Exception:
            self.handleError(record)


class LogBuffer(logging.Handler):
    """A ring of recent records, each tagged with the run that was current when
    it was logged, plus an index of those runs and how each one ended.

    The server calls :meth:`begin_run` at INIT_RUN and :meth:`end_run` when the
    run is saved, discarded or fails to save. Every record logged in between
    carries that run's ``seq`` (a counter that is unique within this process, so
    unsaved runs, which all have ``run_id=0``, stay apart) and ``run_id``.
    Records logged before the first run, or after a run ended and before the
    next began, carry the last run's tags: the lines that explain what became
    of a run often arrive after its END_RUN.

    Thread-safe: records come from the server, camera and writer threads.
    """

    def __init__(self, maxlen: int = MAX_BUFFERED_RECORDS, max_runs: int = MAX_BUFFERED_RUNS):
        super().__init__(logging.DEBUG)
        self.setFormatter(logging.Formatter("%(message)s"))
        self._lock_ = threading.Lock()
        self._records = deque(maxlen=maxlen)
        self._runs = deque(maxlen=max_runs)
        self._seq = 0
        self._run_id = 0
        self._current = None

    def clear(self):
        """Forget every record and run (tests; the buffer is process-wide)."""
        with self._lock_:
            self._records.clear()
            self._runs.clear()
            self._seq = 0
            self._run_id = 0
            self._current = None

    # -- run index -------------------------------------------------------

    def begin_run(self, run_id: int, name: str = "", **info) -> int:
        """A new run is current from now on. Returns its ``seq``."""
        with self._lock_:
            self._seq += 1
            self._run_id = int(run_id)
            self._current = {
                "seq": self._seq, "run_id": int(run_id), "name": str(name),
                "t_start": time.time(), "t_end": None,
                "outcome": "in_progress", "detail": "",
            }
            self._current.update(info)
            self._runs.append(self._current)
            return self._seq

    def end_run(self, outcome: str, detail: str = "", **info):
        """How the current run ended: ``saved``, ``saved_incomplete``,
        ``discarded``, ``save_failed`` or ``nothing_written``."""
        with self._lock_:
            if self._current is None:
                return
            self._current["outcome"] = str(outcome)
            self._current["detail"] = str(detail)
            self._current["t_end"] = time.time()
            self._current.update(info)

    def update_run(self, **info):
        """Attach live facts (shot counts, frames received) to the current run."""
        with self._lock_:
            if self._current is not None:
                self._current.update(info)

    def runs(self, limit: int = 50) -> list:
        """The most recent runs, oldest first."""
        with self._lock_:
            runs = list(self._runs)
        return [dict(r) for r in runs[-int(limit):]] if limit else [dict(r) for r in runs]

    def find_run(self, run_id=None, seq=None):
        """The run index entry for ``seq``, else the latest run with ``run_id``,
        else the current run. None if nothing matches."""
        with self._lock_:
            runs = list(self._runs)
            current = self._current
        if seq is not None:
            for r in reversed(runs):
                if r["seq"] == int(seq):
                    return dict(r)
            return None
        if run_id:
            for r in reversed(runs):
                if r["run_id"] == int(run_id):
                    return dict(r)
            return None
        return dict(current) if current is not None else None

    # -- records ---------------------------------------------------------

    def emit(self, record):
        try:
            entry = {
                "t": record.created,
                "level": record.levelno,
                "levelname": record.levelname,
                "msg": self.format(record),
                "thread": record.threadName,
                "logger": record.name,
            }
            with self._lock_:
                entry["seq"] = self._seq
                entry["run_id"] = self._run_id
                self._records.append(entry)
        except Exception:
            self.handleError(record)

    def records(self, run_id=None, seq=None, since=None,
                min_level: int = logging.DEBUG, limit: int = 2000) -> list:
        """Records for one run (by ``seq``, else the latest run with ``run_id``,
        else the current run), or every record when ``run_id`` is ``"all"``.
        ``since`` (epoch seconds) keeps only later records. The last ``limit``
        matches, oldest first."""
        run = None
        if run_id != "all":
            run = self.find_run(run_id=run_id, seq=seq)
            if run is None:
                return []
        with self._lock_:
            recs = list(self._records)
        out = []
        for r in recs:
            if run is not None and r["seq"] != run["seq"]:
                continue
            if r["level"] < min_level:
                continue
            if since is not None and r["t"] <= float(since):
                continue
            out.append(dict(r))
        if limit and len(out) > int(limit):
            out = out[-int(limit):]
        return out


_qt_handler = None
_buffer = None


def get_log_buffer() -> LogBuffer:
    """The process's :class:`LogBuffer`, attached to the liveOD logger on first
    use. Attaching it does not change what reaches the terminal: without
    ``setup_logging`` the logger still propagates to the root, whose last-resort
    handler prints WARNING and above as before."""
    global _buffer
    if _buffer is None:
        logger = get_logger()
        _buffer = LogBuffer()
        logger.addHandler(_buffer)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.DEBUG)
    return _buffer


def setup_logging(log_dir=DEFAULT_LOG_DIR, level=logging.INFO) -> QtLogHandler:
    """Attach the terminal, file, Qt and buffer handlers (once) and return the Qt handler.

    ``level`` is the GUI's: the terminal, the file and the buffer take everything,
    DEBUG included (per-shot lines are DEBUG except for about one in twenty).
    ``log_dir=None`` skips the file. A log directory that cannot be written is
    reported and otherwise ignored: liveOD must start without it.
    """
    global _qt_handler
    logger = get_logger()
    if _qt_handler is not None:
        return _qt_handler

    logger.setLevel(logging.DEBUG)
    logger.propagate = False    # the root logger may have handlers of its own

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(stream)

    _qt_handler = QtLogHandler(level)
    logger.addHandler(_qt_handler)
    get_log_buffer()

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, "live_od.log"),
                maxBytes=2_000_000, backupCount=5, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"))
            logger.addHandler(file_handler)
        except Exception as exc:
            logger.warning(f"Could not open the log file in {log_dir}: {exc}")

    return _qt_handler


def install_excepthooks():
    """Send uncaught exceptions to the logger, so a traceback from a Qt slot or a
    worker thread shows up in the GUI's log and not only in the terminal.

    With the default ``sys.excepthook`` PyQt aborts the process on an exception in
    a slot; with this one installed liveOD logs it and keeps acquiring.
    """
    logger = get_logger()

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.error(f"Uncaught {exc_type.__name__}: {exc}", exc_info=(exc_type, exc, tb))

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "?"
        logger.error(f"Uncaught {args.exc_type.__name__} in thread {name}: {args.exc_value}",
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _hook
    threading.excepthook = _thread_hook
