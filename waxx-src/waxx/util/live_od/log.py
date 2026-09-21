"""Logging for the liveOD process.

Everything the liveOD server, camera threads and GUI have to say goes through the
``waxx.live_od`` logger rather than ``print``. ``setup_logging`` (called once by
the acquisition window) attaches three handlers:

* a stream handler, so the terminal shows what it always showed,
* a rotating file under ``~/.waxx/logs``, for working out afterwards what happened,
* a :class:`QtLogHandler`, which re-emits every record as a Qt signal. Records are
  logged from the server, CameraBaby, DataHandler and SaveWorker threads; the
  signal is how they reach the GUI thread safely.

Until ``setup_logging`` is called (tests, or the server used without its window)
records of WARNING and above still reach stderr through logging's last-resort
handler, and nothing is written to disk.
"""

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler

from PyQt6.QtCore import QObject, pyqtSignal

LOGGER_NAME = "waxx.live_od"
DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), ".waxx", "logs")


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


_qt_handler = None


def setup_logging(log_dir=DEFAULT_LOG_DIR, level=logging.INFO) -> QtLogHandler:
    """Attach the terminal, file and Qt handlers (once) and return the Qt handler.

    ``level`` is the GUI's: the terminal and the file take everything, DEBUG
    included (per-shot lines are DEBUG except for about one in twenty).
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
