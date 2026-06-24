"""Atomic device-state JSON I/O — used exclusively by the monitor server.

The monitor server is the only process that writes the shared device-state
JSON.  Writes go through :func:`apply_delta`, which performs an atomic
read-modify-write (write to a temp file in the same directory, then
``os.replace``).  ``os.replace`` is atomic on both Windows and POSIX, so
readers (the ARTIQ monitor experiment, other tooling) never observe a
partially written file.

A module-level lock serialises concurrent calls within the server process.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading

_write_lock = threading.Lock()


def read_state(path) -> dict:
    """Read and parse the full device-state JSON."""
    with open(path, "r") as f:
        return json.load(f)


def atomic_write(path, data: dict) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace``)."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def apply_delta(path, device_type: str, device_name: str, changes: dict) -> dict:
    """Merge ``changes`` into one device and atomically persist the result.

    Read-modify-write under a process-wide lock so simultaneous deltas to
    *different* devices merge piece-by-piece instead of clobbering each other.
    Returns the full updated config dict.
    """
    with _write_lock:
        try:
            data = read_state(path)
        except FileNotFoundError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        section = data.setdefault(device_type, {})
        device = section.setdefault(device_name, {})
        device.update(changes)
        atomic_write(path, data)
        return data
