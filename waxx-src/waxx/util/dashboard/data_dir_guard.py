"""DATA_DIR / network-share resilience helper.

A single entry point used everywhere the dashboard or a supervised server
needs to make sure the lab's shared data directory is reachable before
touching files inside it.

Behavior
--------
* If the configured ``data_dir`` (or a specific sub-path) already exists,
  return ``ok=True`` immediately.
* If the data dir is missing **and** the configured ``.bat`` re-map script
  exists, run the script once (synchronously, ``CREATE_NO_WINDOW``,
  30 s timeout) and re-check.
* If the bat path itself does not exist, return ``ok=False`` with
  ``reason='bat_missing'`` and never spawn a subprocess.
* Never raises.  All exceptions are folded into ``ok=False``.

The lab application registers the data dir + bat path once at startup via
:func:`configure`.  This module is otherwise lab-agnostic so it can live
in ``waxx``.

Usage::

    from waxx.util.dashboard import data_dir_guard
    data_dir_guard.configure(DATA_DIR, MAP_BAT_PATH)
    status = data_dir_guard.ensure_data_dir()
    if not status.ok:
        log.error("DATA_DIR unreachable: %s", status.reason)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional


_LOG = logging.getLogger("waxx.dashboard.data_dir_guard")

# Configured-at-startup state.
_DATA_DIR: Optional[str] = None
_BAT_PATH: Optional[str] = None
# Per-(path, reason) one-shot cache so callers can spam ``ensure_data_dir``
# without log spam.  Used only by :func:`safe_read_json`.
_LOGGED_REASONS: set[tuple[str, str]] = set()


@dataclass
class DataDirStatus:
    """Result of an :func:`ensure_data_dir` call.

    Attributes
    ----------
    ok:
        True if the target path exists (either was already present or the
        bat-remap restored it).
    reason:
        Short machine-readable tag:

        * ``present``       — target already existed; no remap attempted.
        * ``remapped``      — target appeared after running the bat.
        * ``data_dir_unset`` — :func:`configure` was never called / DATA_DIR is None.
        * ``bat_missing``   — DATA_DIR was missing and the bat path doesn't exist
          on disk; subprocess was NOT spawned.
        * ``remap_failed``  — bat ran but the target is still missing.
        * ``exception:...`` — an unexpected exception was caught; included
          for diagnostics.
    attempted_remap:
        True if the bat script was actually run.
    bat_exists:
        True if the configured bat path is a real file on disk.
    bat_path:
        The resolved bat path (with surrounding quotes stripped), or None.
    """

    ok: bool
    reason: str
    attempted_remap: bool = False
    bat_exists: bool = False
    bat_path: Optional[str] = None


def configure(data_dir: Optional[str], bat_path: Optional[str]) -> None:
    """One-time configuration.  Idempotent; safe to call multiple times.

    Parameters
    ----------
    data_dir:
        Root of the shared data directory (e.g. ``B:\\_K\\PotassiumData``).
        May be None if the env var is unset; ``ensure_data_dir`` will then
        return ``ok=False, reason='data_dir_unset'``.
    bat_path:
        Path to a ``.bat`` script that maps the network share.  May contain
        surrounding double-quotes (Windows convention for paths with
        spaces); they are stripped automatically.  May be None.
    """
    global _DATA_DIR, _BAT_PATH
    _DATA_DIR = data_dir if data_dir else None
    _BAT_PATH = _strip_quotes(bat_path) if bat_path else None


def is_configured() -> bool:
    return _DATA_DIR is not None or _BAT_PATH is not None


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def ensure_data_dir(
    path: Optional[str] = None,
    *,
    log: Optional[logging.Logger] = None,
) -> DataDirStatus:
    """Ensure *path* (or the configured DATA_DIR root) is reachable.

    Never raises.  See :class:`DataDirStatus` for the result shape.
    """
    log = log or _LOG
    bat = _BAT_PATH
    bat_exists = bool(bat) and os.path.exists(bat)

    target = path or _DATA_DIR
    if not target:
        return DataDirStatus(
            ok=False, reason="data_dir_unset",
            attempted_remap=False, bat_exists=bat_exists, bat_path=bat,
        )

    try:
        if os.path.exists(target):
            return DataDirStatus(
                ok=True, reason="present",
                attempted_remap=False, bat_exists=bat_exists, bat_path=bat,
            )
    except Exception as exc:
        return DataDirStatus(
            ok=False, reason=f"exception:{exc!r}",
            attempted_remap=False, bat_exists=bat_exists, bat_path=bat,
        )

    # Path missing — try the bat remap if we can.
    if not bat_exists:
        return DataDirStatus(
            ok=False, reason="bat_missing",
            attempted_remap=False, bat_exists=False, bat_path=bat,
        )

    if sys.platform != "win32":
        # No-op everywhere except Windows; bat files don't apply.
        return DataDirStatus(
            ok=False, reason="remap_failed",
            attempted_remap=False, bat_exists=True, bat_path=bat,
        )

    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log.info("data_dir_guard: target %s missing, running %s", target, bat)
        result = subprocess.run(
            bat,
            shell=True,
            creationflags=creationflags,
            timeout=30,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            log.debug("data_dir_guard: bat stdout: %s", result.stdout.strip())
        if result.stderr:
            log.debug("data_dir_guard: bat stderr: %s", result.stderr.strip())
    except Exception as exc:
        return DataDirStatus(
            ok=False, reason=f"exception:{exc!r}",
            attempted_remap=True, bat_exists=True, bat_path=bat,
        )

    try:
        if os.path.exists(target):
            return DataDirStatus(
                ok=True, reason="remapped",
                attempted_remap=True, bat_exists=True, bat_path=bat,
            )
    except Exception as exc:
        return DataDirStatus(
            ok=False, reason=f"exception:{exc!r}",
            attempted_remap=True, bat_exists=True, bat_path=bat,
        )

    return DataDirStatus(
        ok=False, reason="remap_failed",
        attempted_remap=True, bat_exists=True, bat_path=bat,
    )


def safe_read_json(
    path: str,
    *,
    default: Any = None,
    log: Optional[logging.Logger] = None,
) -> Any:
    """Read *path* as JSON after ensuring its parent data dir is reachable.

    Returns *default* on any failure.  Logs one warning per (path, reason)
    pair so noisy callers don't flood the log.
    """
    log = log or _LOG
    status = ensure_data_dir(path, log=log)
    if not status.ok:
        key = (path, status.reason)
        if key not in _LOGGED_REASONS:
            _LOGGED_REASONS.add(key)
            log.warning(
                "safe_read_json: cannot read %s (%s; bat=%s)",
                path, status.reason, status.bat_path or "<unset>",
            )
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        key = (path, f"read:{type(exc).__name__}")
        if key not in _LOGGED_REASONS:
            _LOGGED_REASONS.add(key)
            log.warning("safe_read_json: %s raised %r", path, exc)
        return default


__all__ = [
    "DataDirStatus",
    "configure",
    "is_configured",
    "ensure_data_dir",
    "safe_read_json",
]
