"""A lock file next to the file it guards (``<file>.lock``), made with O_EXCL.

Only kcal / the emit take it; it orders their writes, it does not stop an
editor. The holder's pid and start time are in the lock file, so a refusal can
say who holds it. A lock is never broken here: a stale one is reported and left
for a person to remove.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path


class LockTimeout(RuntimeError):
    pass


@contextlib.contextmanager
def file_lock(target, timeout: float = 10.0, poll: float = 0.05):
    lock = Path(str(target) + ".lock")
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() - t0 > timeout:
                try:
                    holder = lock.read_text(errors="replace").strip()
                except OSError:
                    holder = "?"
                raise LockTimeout(f"{lock} is held ({holder or 'no holder recorded'}) for more "
                                  f"than {timeout:g} s; if no kcal / emit is running, a person "
                                  f"may delete it")
            time.sleep(poll)
    try:
        os.write(fd, f"pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}".encode())
    finally:
        os.close(fd)
    try:
        yield lock
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def replace_bytes(path, data: bytes, retries: int = 20, wait: float = 0.05):
    """Write ``data`` to a temp file beside ``path`` and os.replace it in.
    Windows refuses the replace while another process has the file open
    without FILE_SHARE_DELETE; that is retried briefly."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.kcal-{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == retries - 1:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
            time.sleep(wait)
