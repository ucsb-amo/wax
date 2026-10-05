"""Centralized logging setup for dashboards, servers and embedded panels.

A single helper module that every server ``main()`` and every dashboard ``main()``
calls once at startup.  Goals:

* Uniform log file locations: ``<log_root>/_logs/{server,client}/<id>__<host>.log``
* Rotating files (5 MB x 5 backups) so the network share never fills.
* The log file is written from a background thread, so a hung share never
  blocks the thread that logs; if the share fails mid-run the file moves to
  the local-appdata mirror and the share is retried every 5 minutes.
* ``faulthandler`` installed before any third-party import so native crashes
  (pylonsdk SEGVs, pyserial driver faults, pyzmq aborts) write Python tracebacks
  to the same log file.
* Unhandled exceptions (main thread, other threads, PyQt slots) are logged
  with their traceback; in a process that has loaded Qt, Qt's own
  warning / critical / fatal messages are logged too (a ``pythonw.exe``
  dashboard has no stderr for them).
* Fallback to ``%LOCALAPPDATA%/<app_name>/dashboard/_logs/`` if the primary
  log dir is unmapped or read-only.
* Visible banner emitted via a "boot warning" list that the dashboard reads at
  startup and surfaces in the status bar.

Generic library.  Lab-specific apps configure it once::

    from waxx.util.dashboard import logging_setup
    logging_setup.configure(app_name='kexp', log_root='Z:/lab_share/_logs')

Then each process calls :func:`configure_server_logging` or
:func:`configure_client_logging` as before.

Public API:

* :func:`configure`              - one-time app config (log root + app name).
* :func:`configure_server_logging` - call at top of every ``*_server.py`` ``main()``.
* :func:`configure_client_logging` - call at top of every dashboard / GUI ``main()``.
* :func:`attach_panel_logger`    - returns a child logger tagged with the panel id.
* :func:`active_log_dir`         - returns the directory the helper is currently writing to.
* :func:`pop_boot_warnings`      - returns and clears any startup warnings.
* :func:`install_excepthooks` / :func:`install_qt_message_handler` - run by
  the ``configure_*_logging`` calls; callable on their own.
"""

from __future__ import annotations

import atexit
import faulthandler
import logging
import logging.handlers
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional


_BOOT_WARNINGS: list[str] = []
_ACTIVE_LOG_DIR: Optional[Path] = None
_FAULTHANDLER_FILE = None  # kept alive so faulthandler can write to it
_FILE_WRITERS: dict[str, logging.handlers.QueueListener] = {}  # resolved log path -> writer thread
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# App-level config (set via configure()).
_APP_NAME = "dashboard"
_LOG_ROOT: Optional[Path] = None
_LOGGER_NS = "dashboard"   # logger-name prefix for attach_panel_logger / server_child_logger


def configure(
    *,
    app_name: str = "dashboard",
    log_root: Optional[str] = None,
    logger_namespace: Optional[str] = None,
) -> None:
    """One-time application configuration.

    Call this once near program startup (before the first
    ``configure_*_logging`` call) to tell the helper what the app is
    called and where logs should land.

    Parameters
    ----------
    app_name:
        Used in the local fallback path (``%LOCALAPPDATA%/<app_name>/...``).
    log_root:
        Primary log directory.  Server logs go to ``<log_root>/server`` and
        client logs to ``<log_root>/client``.  If ``None`` or unwritable,
        the local fallback is used and a boot warning is recorded.
    logger_namespace:
        Logger-name prefix; defaults to *app_name*.  Panel loggers become
        ``<namespace>.client.<panel_id>``.
    """
    global _APP_NAME, _LOG_ROOT, _LOGGER_NS
    _APP_NAME = app_name or "dashboard"
    _LOG_ROOT = Path(log_root) if log_root else None
    _LOGGER_NS = logger_namespace or _APP_NAME


def _hostname() -> str:
    try:
        return socket.gethostname().lower()
    except Exception:
        return "unknown-host"


def _local_fallback_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / _APP_NAME / "dashboard" / "_logs"


def _resolve_log_dir(kind: str) -> Path:
    """Return the directory log files should live in for *kind* in {"server","client"}.

    Prefers the configured ``log_root``/<kind>; if unconfigured or unwritable,
    falls back to the local-appdata mirror and records a boot warning.
    """
    primary = (_LOG_ROOT / kind) if _LOG_ROOT else None

    if primary is not None:
        try:
            primary.mkdir(parents=True, exist_ok=True)
            probe = primary / ".write_probe"
            try:
                probe.touch()
                probe.unlink()
            except Exception:
                raise
            return primary
        except Exception as exc:
            _BOOT_WARNINGS.append(
                f"logging_setup: cannot write to {primary} ({exc!r}); "
                f"falling back to local-appdata logs."
            )
    else:
        _BOOT_WARNINGS.append(
            "logging_setup: log_root not configured; "
            "using local-appdata logs (call logging_setup.configure(log_root=...) early in main)."
        )

    fallback = _local_fallback_root() / kind
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


class _SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """``RotatingFileHandler`` that survives Windows file-locking races and a
    failing log share.

    On Windows, if any other process or another open handle (e.g. a stale
    faulthandler file) still holds the active log file when rollover
    triggers, ``os.rename`` raises ``PermissionError [WinError 32]`` and
    the entire ``emit()`` call crashes the calling thread.  We catch that
    here, write one warning to stderr, and continue logging into the
    current file — losing rotation for this cycle is far better than
    crashing the dashboard.

    When opening, writing or flushing the file raises ``OSError`` — an SMB
    reconnect leaves the open handle dead (``Errno 22`` on every ``tell``),
    and the NAS can then refuse new opens of that one file (``Errno 13``
    after ~70 s) — the handler switches to *fallback*, says so on stderr and
    in the fallback file, and tries the primary again every
    ``RETRY_PRIMARY_S``.  Without this every record re-tried the dead file
    and printed a "--- Logging error ---" traceback.
    """

    RETRY_PRIMARY_S = 300.0
    _rollover_warned: bool = False

    def __init__(self, filename, *, fallback=None, **kwargs):
        super().__init__(filename, **kwargs)
        self._primary = self.baseFilename
        self._fallback = os.path.abspath(fallback) if fallback else None
        self._primary_failed_at: Optional[float] = None  # monotonic; None while on the primary

    def emit(self, record):
        self._maybe_return_to_primary()
        try:
            self._write(record)
        except RecursionError:
            raise
        except OSError as exc:
            if not self._switch_to_fallback(exc):
                self.handleError(record)
                return
            try:
                self._write(record)
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)

    def _write(self, record) -> None:
        """Rotate if due and write *record*.  Unlike the stdlib ``emit`` this
        lets ``OSError`` through, so :meth:`emit` can fall back."""
        if self.shouldRollover(record):
            self.doRollover()
        if self.stream is None:
            self.stream = self._open()
        self.stream.write(self.format(record) + self.terminator)
        self.stream.flush()

    def _switch_to_fallback(self, exc: BaseException) -> bool:
        if self._fallback is None or self.baseFilename == self._fallback:
            return False
        self._drop_stream()
        self.baseFilename = self._fallback
        self._primary_failed_at = time.monotonic()
        try:
            os.makedirs(os.path.dirname(self._fallback), exist_ok=True)
        except OSError:
            pass
        self._note(f"cannot write {self._primary} ({exc!r}); logging to {self._fallback} "
                   f"and retrying {self._primary} every {self.RETRY_PRIMARY_S:.0f} s.")
        return True

    def _maybe_return_to_primary(self) -> None:
        if self._primary_failed_at is None:
            return
        if time.monotonic() - self._primary_failed_at < self.RETRY_PRIMARY_S:
            return
        self._primary_failed_at = time.monotonic()
        self.baseFilename = self._primary
        try:
            stream = self._open()
        except OSError:
            self.baseFilename = self._fallback
            return
        self._drop_stream()
        self.stream = stream
        self._primary_failed_at = None
        self._note(f"{self._primary} is writable again; logging there "
                   f"(the gap is in {self._fallback}).")

    def _drop_stream(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def _note(self, msg: str) -> None:
        """Say *msg* on stderr and, best effort, in the file now in use."""
        line = f"logging_setup: {msg}"
        if sys.stderr is not None:
            try:
                sys.stderr.write(line + "\n")
            except Exception:
                pass
        try:
            self._write(logging.LogRecord(
                "logging_setup", logging.WARNING, __file__, 0, line, None, None))
        except Exception:
            pass

    def doRollover(self):  # noqa: N802 - stdlib API
        try:
            super().doRollover()
        except (PermissionError, OSError) as exc:
            if not _SafeRotatingFileHandler._rollover_warned:
                _SafeRotatingFileHandler._rollover_warned = True
                try:
                    sys.stderr.write(
                        f"logging_setup: log rotation skipped for {self.baseFilename!r} "
                        f"({exc!r}); will keep writing to the current file.\n"
                    )
                except Exception:
                    pass
            # Re-open the stream if super() closed it before failing,
            # otherwise subsequent emits will raise ValueError.
            try:
                if self.stream is None or getattr(self.stream, "closed", False):
                    self.stream = self._open()
            except Exception:
                self.stream = None


def _install_handlers(log_path: Path, fallback: Path, level: int = logging.INFO) -> None:
    """Attach a rotating file handler (behind a queue) + console handler to
    the root logger.

    Idempotent for the same path: if a handler already writes this file we
    leave it alone.
    """
    global _FAULTHANDLER_FILE

    root = logging.getLogger()
    root.setLevel(level)

    target = str(log_path.resolve())
    if target in _FILE_WRITERS:
        return  # already configured

    file_handler = _SafeRotatingFileHandler(
        filename=str(log_path),
        fallback=fallback,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setFormatter(_FORMATTER)
    file_handler.setLevel(level)
    # The file usually lives on a network share, where one open can block for
    # ~70 s: write it from a background thread so the thread that logs (a
    # server's request loop) never waits on the share.
    records: queue.SimpleQueue = queue.SimpleQueue()
    queue_handler = logging.handlers.QueueHandler(records)
    queue_handler.setLevel(level)
    root.addHandler(queue_handler)
    writer = logging.handlers.QueueListener(records, file_handler, respect_handler_level=True)
    writer.start()
    atexit.register(writer.stop)  # drains the queue before logging.shutdown
    _FILE_WRITERS[target] = writer

    # Console handler - only add one if there isn't a StreamHandler already,
    # and only when there is a console at all (under pythonw.exe sys.stderr
    # is None; a StreamHandler on it would raise on every record).
    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        for h in root.handlers
    )
    if not has_console and sys.stderr is not None:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(_FORMATTER)
        console.setLevel(level)
        root.addHandler(console)
    install_excepthooks()
    # Qt's own warnings / fatals go to stderr, which a pythonw-launched
    # dashboard does not have.  Only when Qt is already loaded: a server that
    # never imports Qt must not start doing so here.
    if "PyQt6.QtCore" in sys.modules:
        install_qt_message_handler()

    # faulthandler: write native tracebacks to a sibling ``.fault`` file,
    # NOT the rotating log itself.  If we share the file handle with the
    # ``RotatingFileHandler``, Windows refuses ``os.rename`` at rollover
    # time (PermissionError WinError 32) because faulthandler still has
    # the file open.  Keep the file handle alive in a module-level global
    # — faulthandler does NOT keep its own reference.
    try:
        fault_path = log_path.parent / (log_path.name + ".fault")
        _FAULTHANDLER_FILE = open(fault_path, "a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=_FAULTHANDLER_FILE, all_threads=True)
    except Exception as exc:
        _BOOT_WARNINGS.append(f"logging_setup: faulthandler.enable failed ({exc!r})")


def _is_ours(hook) -> bool:
    return bool(getattr(hook, "_waxx_logging_hook", False))


def install_excepthooks() -> None:
    """Log every unhandled exception, on the main thread and on other threads.

    Under PyQt6 an exception escaping a slot goes to ``sys.excepthook``; with
    Python's default hook PyQt then aborts the whole process (under
    ``python.exe``), and under ``pythonw.exe`` the traceback has nowhere to
    go.  This hook writes the traceback to the log and returns, so the event
    loop carries on.  ``KeyboardInterrupt`` keeps Python's default behaviour.
    A hook installed earlier by someone else still runs after the log line.
    Idempotent.
    """
    previous = sys.excepthook
    if not _is_ours(previous):
        chain = None if previous is sys.__excepthook__ else previous

        def _log_uncaught(exc_type, exc, tb):  # noqa: ANN001
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc, tb)
                return
            logging.getLogger("uncaught").critical(
                "uncaught exception", exc_info=(exc_type, exc, tb))
            if chain is not None:
                try:
                    chain(exc_type, exc, tb)
                except Exception:
                    pass

        _log_uncaught._waxx_logging_hook = True
        sys.excepthook = _log_uncaught

    previous_thread = threading.excepthook
    if not _is_ours(previous_thread):
        thread_chain = None if previous_thread is threading.__excepthook__ else previous_thread

        def _log_thread_uncaught(args):  # noqa: ANN001
            if args.exc_type is SystemExit:
                return
            name = args.thread.name if args.thread is not None else "?"
            logging.getLogger("uncaught").critical(
                "uncaught exception in thread %s", name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
            if thread_chain is not None:
                try:
                    thread_chain(args)
                except Exception:
                    pass

        _log_thread_uncaught._waxx_logging_hook = True
        threading.excepthook = _log_thread_uncaught


#: Identical Qt messages (same type and text) are logged at most once per
#: this many seconds; the next one that gets through says how many were held
#: back.  Qt can repeat a warning on every paint or timer tick.
QT_REPEAT_WINDOW_S = 60.0
_QT_HANDLER_INSTALLED = False
_QT_PREVIOUS_HANDLER = None  # what qInstallMessageHandler returned (for tests)


def _flush_before_abort(line: str) -> None:
    """Get *line* and everything already logged onto disk before Qt aborts.

    The rotating file is written by a background thread, so the record just
    logged is still in its queue: stop the writers (which drains them), and
    also put the line in the ``.fault`` file directly, which is unbuffered.
    """
    try:
        if _FAULTHANDLER_FILE is not None:
            _FAULTHANDLER_FILE.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
            _FAULTHANDLER_FILE.flush()
    except Exception:
        pass
    for writer in list(_FILE_WRITERS.values()):
        try:
            writer.stop()
        except Exception:
            pass
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def install_qt_message_handler() -> bool:
    """Route Qt's own messages (qWarning / qCritical / qFatal) into logging.

    Qt prints them to stderr, which a ``pythonw.exe`` dashboard does not have:
    the warnings that come before a freeze, and the fatal message that
    explains a crash, were simply lost.  Debug -> DEBUG, info -> INFO,
    warning -> WARNING, critical -> ERROR, fatal -> CRITICAL (logged, then
    flushed to disk, before Qt aborts the process).  Logger name ``qt``.
    Returns False when PyQt6 is not importable.  Idempotent.
    """
    global _QT_HANDLER_INSTALLED, _QT_PREVIOUS_HANDLER
    if _QT_HANDLER_INSTALLED:
        return True
    try:
        from PyQt6.QtCore import QtMsgType, qInstallMessageHandler  # noqa: PLC0415
    except Exception:
        return False

    log = logging.getLogger("qt")
    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }
    lock = threading.Lock()
    # (type, text) -> [monotonic time last logged, repeats held back since]
    recent: dict[tuple, list] = {}
    # A log handler that itself makes Qt warn (e.g. a widget-backed handler
    # used from the wrong thread) must not recurse back in here.
    busy = threading.local()

    def _handler(mode, context, message):  # noqa: ANN001 - Qt callback
        if getattr(busy, "active", False):
            return
        busy.active = True
        try:
            _handle(mode, context, message)
        finally:
            busy.active = False

    def _handle(mode, context, message):  # noqa: ANN001
        try:
            where = ""
            source = getattr(context, "file", None)
            if source:
                where = (f" ({source}:{getattr(context, 'line', '?')}"
                         f" {getattr(context, 'function', '') or ''})")
            if mode == QtMsgType.QtFatalMsg:
                log.critical("Qt fatal: %s%s", message, where)
                _flush_before_abort(f"Qt fatal: {message}{where}")
                return
            level = levels.get(mode, logging.WARNING)
            if not log.isEnabledFor(level):
                return
            now = time.monotonic()
            key = (int(getattr(mode, "value", 0)), message)
            with lock:
                entry = recent.get(key)
                if entry is not None and now - entry[0] < QT_REPEAT_WINDOW_S:
                    entry[1] += 1
                    return
                held_back = entry[1] if entry is not None else 0
                if len(recent) > 512:
                    recent.clear()
                recent[key] = [now, 0]
            note = ""
            if held_back:
                note = (f" [repeated {held_back} more time(s) in the "
                        f"{QT_REPEAT_WINDOW_S:.0f} s before]")
            log.log(level, "Qt: %s%s%s", message, where, note)
        except Exception:
            pass  # never raise into Qt

    _QT_PREVIOUS_HANDLER = qInstallMessageHandler(_handler)
    _QT_HANDLER_INSTALLED = True
    return True


def configure_server_logging(server_id: str, level: int = logging.INFO) -> Path:
    """Configure logging for a server process.

    Call exactly once near the top of every ``*_server.py`` ``main()`` (and
    before any third-party hardware import).

    Parameters
    ----------
    server_id:
        Stable identifier used in the log filename, e.g. ``"als"``, ``"interlock"``.
    level:
        Root log level (default INFO).

    Returns
    -------
    Path
        The absolute path to the log file that was configured.
    """
    global _ACTIVE_LOG_DIR
    log_dir = _resolve_log_dir("server")
    _ACTIVE_LOG_DIR = log_dir
    log_path = log_dir / f"{server_id}__{_hostname()}.log"
    _install_handlers(log_path, _local_fallback_root() / "server" / log_path.name, level=level)
    logging.getLogger().info(
        "configure_server_logging: id=%s host=%s pid=%d log=%s",
        server_id, _hostname(), os.getpid(), log_path,
    )
    return log_path


def configure_client_logging(level: int = logging.INFO) -> Path:
    """Configure logging for a dashboard / GUI process.

    All dashboards (server dashboard, client dashboard) and standalone client
    GUIs use a single shared ``dashboard__<host>.log`` file in the client log
    directory.  Panel-specific events are tagged via :func:`attach_panel_logger`.
    """
    global _ACTIVE_LOG_DIR
    log_dir = _resolve_log_dir("client")
    _ACTIVE_LOG_DIR = log_dir
    log_path = log_dir / f"dashboard__{_hostname()}.log"
    _install_handlers(log_path, _local_fallback_root() / "client" / log_path.name, level=level)
    logging.getLogger().info(
        "configure_client_logging: host=%s pid=%d log=%s",
        _hostname(), os.getpid(), log_path,
    )
    return log_path


def attach_panel_logger(panel_id: str) -> logging.Logger:
    """Return a child logger named after a dashboard panel.

    Records emitted through this logger flow into the same dashboard log file
    configured by :func:`configure_client_logging`, but are prefixed with the
    panel id so per-panel issues can be grepped easily.
    """
    return logging.getLogger(f"{_LOGGER_NS}.dashboard.client.{panel_id}")


def server_child_logger(server_id: str, suffix: Optional[str] = None) -> logging.Logger:
    """Return a structured child logger for a server.

    With ``configure(logger_namespace='kexp')``:
    ``server_child_logger("interlock", "com")`` -> ``waxx.dashboard.server.interlock.com``.
    """
    name = f"{_LOGGER_NS}.dashboard.server.{server_id}"
    if suffix:
        name = f"{name}.{suffix}"
    return logging.getLogger(name)


def active_log_dir() -> Optional[Path]:
    """Return the directory the most recent ``configure_*`` call is writing to."""
    return _ACTIVE_LOG_DIR


def pop_boot_warnings() -> list[str]:
    """Return and clear any boot warnings accumulated during configuration.

    Dashboard ``main()`` calls this after configuring logging so it can surface
    each warning as a status-bar banner.
    """
    out, _BOOT_WARNINGS[:] = list(_BOOT_WARNINGS), []
    return out


__all__ = [
    "configure",
    "configure_server_logging",
    "configure_client_logging",
    "attach_panel_logger",
    "server_child_logger",
    "active_log_dir",
    "pop_boot_warnings",
    "install_excepthooks",
    "install_qt_message_handler",
]
