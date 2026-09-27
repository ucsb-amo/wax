"""One process at a time per device: an OS file lock taken when a driver opens.

A camera SDK or USB device opened from two processes misbehaves in ways that
are hard to read (the Andor SDK2 errors out in Initialize, or worse, succeeds
and fights over the camera).  ``DeviceLock(key)`` makes the second opener fail
at once with ``DeviceBusy``, whose message says who holds the device:

    Andor SDK is held by pid 1234 (python.exe, main_window) since 14:02

The lock is an OS byte-range lock (``msvcrt.locking``; ``fcntl.flock`` on
POSIX) on ``<dir>/<key>.lock``, so the OS drops it when the holding process
dies, however it dies.  The holder's details sit in a JSON sidecar next to it,
written after the lock is taken and removed by the holder on release (never by
anyone else).  A sidecar left by a killed process is simply overwritten by the
next holder.

Directory: ``%LOCALAPPDATA%\\waxx\\device_locks`` (``~/.waxx/device_locks``
where LOCALAPPDATA is unset), or ``$WAXX_DEVICE_LOCK_DIR`` (tests).

Within one process a key is held by at most one ``DeviceLock`` object; a
second one gets ``DeviceBusy`` naming this process, which is the message for
"this process already opened that camera and did not close it".
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
import threading
import time

try:
    # A subclass of the camera-server error, so a camera worker that catches
    # CameraUnavailable at open catches this too.
    from beacon.camera.backend import CameraUnavailable as _BusyBase
except Exception:  # beacon not importable: the lock still works on its own
    class _BusyBase(RuntimeError):
        def __init__(self, message: str, holder: dict | None = None):
            super().__init__(message)
            self.holder = holder or {}

ENV_LOCK_DIR = "WAXX_DEVICE_LOCK_DIR"
ENV_PROCESS_LABEL = "WAXX_PROCESS_LABEL"
POLL_INTERVAL = 0.1     # s between attempts when acquire() is given a timeout


class DeviceBusy(_BusyBase):
    """The device is held by another process (or by another handle in this one).

    ``holder`` is the holder's sidecar record: ``{pid, exe, label, since, ...}``,
    or ``{}`` if it could not be read."""
    def __init__(self, message: str, holder: dict | None = None):
        super().__init__(message, holder)


# key -> DeviceLock, for locks held by this process
_HELD: dict = {}
_HELD_GUARD = threading.RLock()

_process_label: str | None = None


def set_process_label(label: str | None) -> None:
    """Name this process in holder records (e.g. "liveOD"); shown in DeviceBusy
    messages other processes get."""
    global _process_label
    _process_label = label


def _default_label() -> str:
    if _process_label:
        return _process_label
    env = os.environ.get(ENV_PROCESS_LABEL)
    if env:
        return env
    try:
        argv0 = sys.argv[0] if sys.argv else ""
    except Exception:
        argv0 = ""
    stem = os.path.splitext(os.path.basename(argv0))[0]
    if stem == "__main__":              # python -m pkg: name the package
        stem = os.path.basename(os.path.dirname(argv0))
    return stem or "python"


def lock_dir() -> str:
    """The directory holding the lock files (created if missing)."""
    path = os.environ.get(ENV_LOCK_DIR)
    if not path:
        base = os.environ.get("LOCALAPPDATA")
        if base:
            path = os.path.join(base, "waxx", "device_locks")
        else:
            path = os.path.join(os.path.expanduser("~"), ".waxx", "device_locks")
    os.makedirs(path, exist_ok=True)
    return path


def describe_key(key: str) -> str:
    """A human name for a lock key, for messages."""
    if key.startswith("andor_sdk2:"):
        return "Andor SDK"
    if key.startswith("basler:"):
        return f"Basler camera {key.split(':', 1)[1]}"
    return f"Device {key!r}"


def _safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", key)


if os.name == "nt":
    import msvcrt

    def _os_lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)   # OSError if held elsewhere

    def _os_unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _os_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _os_unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _read_holder(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        return rec if isinstance(rec, dict) else {}
    except (OSError, ValueError):
        return {}


def _format_since(rec: dict) -> str:
    epoch = rec.get("since_epoch")
    if isinstance(epoch, (int, float)):
        t = _dt.datetime.fromtimestamp(epoch)
        if t.date() == _dt.date.today():
            return t.strftime("%H:%M")
        return t.strftime("%Y-%m-%d %H:%M")
    return str(rec.get("since", "?"))


class DeviceLock:
    """An exclusive, process-wide lock on one device.

    ``acquire(timeout_s=0.0)`` raises ``DeviceBusy`` if another process (or
    another DeviceLock in this process) holds the key; ``release()`` is
    idempotent.  Also a context manager.  Never released by garbage
    collection: a handle leaked while the device is still open really does
    still hold the device, until close or process exit.
    """

    def __init__(self, key: str, *, description: str | None = None, label: str | None = None):
        self.key = str(key)
        self.description = description or describe_key(self.key)
        self.label = label
        self._fd: int | None = None
        self._since: float | None = None
        base = os.path.join(lock_dir(), _safe_name(self.key))
        self._lock_path = base + ".lock"
        self._info_path = base + ".json"

    # -- state ---------------------------------------------------------------
    @property
    def held(self) -> bool:
        return self._fd is not None

    def holder(self) -> dict:
        """This lock's holder record if held here, else the sidecar's (which
        may be stale when nobody holds the lock)."""
        if self._fd is not None:
            return self._record()
        return _read_holder(self._info_path)

    def _record(self) -> dict:
        since = self._since if self._since is not None else time.time()
        return {
            "key": self.key,
            "pid": os.getpid(),
            "exe": os.path.basename(sys.executable or "") or "python",
            "label": self.label or _default_label(),
            "since": _dt.datetime.fromtimestamp(since).isoformat(timespec="seconds"),
            "since_epoch": since,
        }

    def busy_message(self, holder: dict) -> str:
        if not holder:
            return (f"{self.description} is held by another process "
                    f"(no holder record in {self._info_path})")
        exe = holder.get("exe") or "?"
        label = holder.get("label")
        who = f"{exe}, {label}" if label else exe
        own = " (this process)" if holder.get("pid") == os.getpid() else ""
        return (f"{self.description} is held by pid {holder.get('pid', '?')} ({who}) "
                f"since {_format_since(holder)}{own}")

    # -- acquire / release ---------------------------------------------------
    def acquire(self, timeout_s: float = 0.0) -> "DeviceLock":
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            try:
                self._try_acquire()
                return self
            except DeviceBusy:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise
                time.sleep(min(POLL_INTERVAL, left))

    def _try_acquire(self) -> None:
        with _HELD_GUARD:
            if self._fd is not None:
                return
            other = _HELD.get(self.key)
            if other is not None:
                rec = other._record()
                raise DeviceBusy(self.busy_message(rec), rec)
            fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                _os_lock(fd)
            except OSError:
                os.close(fd)
                rec = _read_holder(self._info_path)
                raise DeviceBusy(self.busy_message(rec), rec) from None
            self._fd = fd
            self._since = time.time()
            _HELD[self.key] = self
            self._write_record()

    def _write_record(self) -> None:
        # Only the message depends on the sidecar; the lock is what counts, so
        # a failed write is not an error.
        tmp = f"{self._info_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._record(), f)
            for _ in range(10):
                try:
                    os.replace(tmp, self._info_path)
                    return
                except PermissionError:     # a reader has it open (Windows)
                    time.sleep(0.01)
            with open(self._info_path, "w", encoding="utf-8") as f:
                json.dump(self._record(), f)
        except OSError:
            pass
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def release(self) -> None:
        with _HELD_GUARD:
            fd = self._fd
            if fd is None:
                return
            # Remove the sidecar while still holding the lock, and only if it
            # is ours: once unlocked, a new holder may already have written its own.
            rec = _read_holder(self._info_path)
            if rec.get("pid") == os.getpid() and rec.get("since_epoch") == self._since:
                try:
                    os.remove(self._info_path)
                except OSError:
                    pass
            try:
                _os_unlock(fd)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            self._fd = None
            self._since = None
            if _HELD.get(self.key) is self:
                del _HELD[self.key]

    def __enter__(self) -> "DeviceLock":
        return self.acquire()

    def __exit__(self, *exc) -> bool:
        self.release()
        return False

    def __repr__(self) -> str:
        return f"DeviceLock({self.key!r}, held={self.held})"
