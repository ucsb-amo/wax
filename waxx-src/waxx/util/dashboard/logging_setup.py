"""Centralized logging setup for dashboards, servers and embedded panels.

A single helper module that every server ``main()`` and every dashboard ``main()``
calls once at startup.  Goals:

* Uniform log file locations: ``<log_root>/_logs/{server,client}/<id>__<host>.log``
* Rotating files (5 MB x 5 backups) so the network share never fills.
* ``faulthandler`` installed before any third-party import so native crashes
  (pylonsdk SEGVs, pyserial driver faults, pyzmq aborts) write Python tracebacks
  to the same log file.
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
"""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import socket
import sys
from pathlib import Path
from typing import Optional


_BOOT_WARNINGS: list[str] = []
_ACTIVE_LOG_DIR: Optional[Path] = None
_FAULTHANDLER_FILE = None  # kept alive so faulthandler can write to it
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
    """``RotatingFileHandler`` that survives Windows file-locking races.

    On Windows, if any other process or another open handle (e.g. a stale
    faulthandler file) still holds the active log file when rollover
    triggers, ``os.rename`` raises ``PermissionError [WinError 32]`` and
    the entire ``emit()`` call crashes the calling thread.  We catch that
    here, write one warning to stderr, and continue logging into the
    current file — losing rotation for this cycle is far better than
    crashing the dashboard.
    """

    _rollover_warned: bool = False

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


def _install_handlers(log_path: Path, level: int = logging.INFO) -> None:
    """Attach a rotating file handler + console handler to the root logger.

    Idempotent for the same path: if a handler already exists pointing at this
    file we leave it alone.
    """
    global _FAULTHANDLER_FILE

    root = logging.getLogger()
    root.setLevel(level)

    abs_target = str(log_path.resolve())
    for h in root.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            try:
                if Path(h.baseFilename).resolve() == log_path.resolve():
                    return  # already configured
            except Exception:
                continue

    file_handler = _SafeRotatingFileHandler(
        filename=str(log_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setFormatter(_FORMATTER)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    # Console handler - only add one if there isn't a StreamHandler already.
    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        for h in root.handlers
    )
    if not has_console:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(_FORMATTER)
        console.setLevel(level)
        root.addHandler(console)

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
    _install_handlers(log_path, level=level)
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
    _install_handlers(log_path, level=level)
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
]
