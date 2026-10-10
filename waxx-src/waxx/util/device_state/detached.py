"""Start a shell command outside this process's tree, and follow it by pid.

The run queue's jobs must outlive the monitor server: the Server Dashboard
stops (and restarts) its servers with ``taskkill /T``, which kills every
process whose parent chain leads to the server.  As for the connection agents
(:mod:`~waxx.util.device_state.connection_agent`), a short-lived *launcher*
process starts the command and exits, so the command's parent is gone and it
is outside the server's tree.  The server then follows the command by pid:

* :func:`launch` runs ``python -m waxx.util.device_state.detached`` with the
  command, working folder and log file on its stdin; the launcher starts
  ``cmd /c <command>`` (``shell=True``) with stdout and stderr appended to the
  log file, in a new process group with no console window, prints its pid,
  and waits for the server's "ok" before it exits -- by then the server holds
  a handle to the process (:class:`ProcessWatch`), so the pid cannot be given
  to another process, and its exit code can be read after it ends.
* :meth:`ProcessWatch.adopt` takes a process back after a server restart, by
  its pid and its creation time (recorded at launch), refusing a pid now used
  by another process.

Nothing here ever terminates a process.  Windows only for the detached part;
elsewhere :func:`launch` starts the command in a new session and follows it
through its Popen object.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from subprocess import DEVNULL, PIPE, STDOUT, Popen

_log = logging.getLogger(__name__)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_STILL_ACTIVE = 259
_ERROR_INVALID_PARAMETER = 87
#: FILETIME (100 ns since 1601-01-01) -> epoch seconds
_EPOCH_AS_FILETIME = 116444736000000000

#: How long the launcher may take to report its child's pid.
LAUNCH_TIMEOUT_S = 30.0
#: Creation times closer than this are the same process (rounding).
SAME_PROCESS_S = 0.01


def _k32():
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    return ctypes, wintypes, k32


class ProcessWatch:
    """A process followed by pid.  ``poll()``: its exit code once it has
    ended, else None (as ``Popen.poll``); ``started``: its creation time
    (epoch seconds; None when it cannot be read).  On Windows it holds a
    process handle until :meth:`close`; elsewhere it wraps a Popen (or, for an
    adopted process, only knows whether the pid is alive: exit code None)."""

    def __init__(self, pid: int, handle=None, popen=None, started: float | None = None):
        self.pid = int(pid)
        self._handle = handle
        self._popen = popen
        self.started = started
        self._code = None
        if started is None and handle is not None:
            self.started = _creation_time(handle)

    @classmethod
    def open(cls, pid: int) -> "ProcessWatch | None":
        """A watch on process ``pid`` (Windows).  None when Windows says there
        is no such process; a watch with no handle -- following the pid's
        liveness only, exit code unknown -- when the process exists but cannot
        be opened (access denied): never taken for gone."""
        if sys.platform != "win32":
            return None
        ctypes, wintypes, k32 = _k32()
        handle = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, False,
                                 int(pid))
        if not handle:
            if ctypes.get_last_error() == _ERROR_INVALID_PARAMETER:
                return None                           # no such process
            return cls(pid)                           # it exists; not ours to open
        return cls(pid, handle=handle)

    @classmethod
    def adopt(cls, pid, started: float | None) -> "ProcessWatch | None":
        """Take a process back after a restart: a watch on ``pid`` when it is
        alive and was created at ``started`` (None: not recorded -- then any
        live process with that pid is taken, which a reused pid could fool);
        None when it is gone or is provably another process (its creation time
        differs).  A process that cannot be opened, or whose creation time
        cannot be read, is followed by liveness (never called gone)."""
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return None
        if pid <= 0:
            return None
        if sys.platform != "win32":
            from waxx.util.device_state.run_gate import pid_alive  # noqa: PLC0415
            return cls(pid, started=started) if pid_alive(pid) else None
        watch = cls.open(pid)
        if watch is None:
            return None
        if watch.poll() is not None:
            watch.close()
            return None
        if (started is not None and watch.started is not None
                and abs(watch.started - float(started)) > SAME_PROCESS_S):
            watch.close()
            return None
        return watch

    def poll(self):
        if self._code is not None:
            return self._code
        if self._popen is not None:
            self._code = self._popen.poll()
            return self._code
        if self._handle is not None:
            ctypes, wintypes, k32 = _k32()
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(self._handle, ctypes.byref(code)):
                return None
            if code.value == _STILL_ACTIVE:
                return None
            self._code = int(code.value)
            return self._code
        from waxx.util.device_state.run_gate import pid_alive  # noqa: PLC0415
        return None if pid_alive(self.pid) else -1

    @property
    def exit_code_known(self) -> bool:
        """False for an adopted process off Windows (only its liveness is known)."""
        return self._handle is not None or self._popen is not None

    def close(self) -> None:
        if self._handle is not None:
            _, _, k32 = _k32()
            k32.CloseHandle(self._handle)
            self._handle = None


def process_started(pid: int | None = None) -> float | None:
    """Process ``pid``'s creation time (this process when None), in epoch
    seconds, read as :meth:`ProcessWatch.adopt` reads it (``GetProcessTimes``);
    None off Windows, or when the process cannot be opened or read.  Two
    readings of one process agree within :data:`SAME_PROCESS_S`; a live pid
    whose creation time differs from a recorded one is another process (the
    pid was reused).  Only opens a handle to query it: never touches the
    process."""
    if sys.platform != "win32":
        return None
    try:
        pid = os.getpid() if pid is None else int(pid)
        if pid <= 0:
            return None
        _, _, k32 = _k32()
        handle = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            return _creation_time(handle)
        finally:
            k32.CloseHandle(handle)
    except Exception:                                 # noqa: BLE001
        return None


def _creation_time(handle) -> float | None:
    try:
        ctypes, wintypes, k32 = _k32()
        times = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        ft = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        return (ft - _EPOCH_AS_FILETIME) / 1e7
    except Exception:                                 # noqa: BLE001
        return None


class LaunchUnknown(OSError):
    """The launcher did not say within its time whether it started the
    command: the command may be running (the caller must look for it, not
    start it again)."""


def _read_line(stream, timeout: float) -> str | None:
    """One line from ``stream``, read on a thread; None when none came within
    ``timeout`` seconds (the thread is left to finish on its own)."""
    out: list = []
    reader = threading.Thread(target=lambda: out.append(stream.readline()), daemon=True,
                              name="run-queue-launcher-reply")
    reader.start()
    reader.join(timeout)
    return out[0] if out else None


def launch(command: str, *, cwd: str, env: dict, log_path: str,
           timeout: float = LAUNCH_TIMEOUT_S) -> ProcessWatch:
    """Start ``command`` (a shell command line) detached, its output appended
    to ``log_path``; return a :class:`ProcessWatch` on it.

    Raises OSError when the command was certainly not started (the launcher
    reported an error, or exited without a pid), :class:`LaunchUnknown` when
    the launcher gave no answer within ``timeout`` (it may have started it).
    Once the launcher has reported a pid, a watch is always returned: an error
    after that (the "ok" write, the launcher's exit) is logged, never raised.
    The launcher's pipes are closed on every path; the launcher itself is
    never killed."""
    if sys.platform != "win32":
        with open(log_path, "ab") as out:
            popen = Popen(command, shell=True, cwd=cwd, env=env, stdin=DEVNULL, stdout=out,
                          stderr=STDOUT, start_new_session=True)
        return ProcessWatch(popen.pid, popen=popen)
    launcher = Popen([sys.executable, "-m", "waxx.util.device_state.detached"],
                     stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env,
                     creationflags=_NO_WINDOW, close_fds=True, text=True,
                     encoding="utf-8", errors="replace")
    watch = None
    reader_owns_stdout = False
    try:
        try:
            # not_after: a launcher that only gets to the command after the
            # server has stopped waiting for it refuses to start it
            launcher.stdin.write(json.dumps({"command": command, "cwd": cwd,
                                             "log_path": log_path,
                                             "not_after": time.time() + timeout}) + "\n")
            launcher.stdin.flush()
        except OSError as exc:
            raise OSError(f"could not hand the command to the launcher: {exc}") from exc
        line = _read_line(launcher.stdout, timeout)
        if line is None:
            # the reader thread is still blocked in readline: closing stdout
            # here would wait on the reader's lock as long as the launcher
            # lives -- the daemon reader keeps it, and the pipe goes with it
            reader_owns_stdout = True
            raise LaunchUnknown(f"the launcher (pid {launcher.pid}) reported nothing within "
                                f"{timeout:.0f} s: the command may or may not be running")
        try:
            reply = json.loads(line) if line.strip() else {}
        except ValueError:
            reply = {"error": f"unreadable launcher reply {line!r}"}
        if "pid" not in reply:
            err = ""
            try:
                launcher.stdin.close()
                launcher.wait(5.0)
                err = (launcher.stderr.read() or "").strip()
            except Exception:                         # noqa: BLE001
                pass
            raise OSError(f"the launcher did not start the command: "
                          f"{reply.get('error') or err or 'no reply'}")
        # a pid: from here on a watch is returned whatever happens
        watch = ProcessWatch.open(reply["pid"])
        if watch is None:
            # gone before a handle could be taken (the launcher held one until
            # our "ok", so this is not a reused pid): it ran and ended; its
            # output is in the log, its exit code is lost
            watch = ProcessWatch(reply["pid"])
            watch._code = -1
        try:
            launcher.stdin.write("ok\n")              # handle held: the launcher may go
            launcher.stdin.flush()
            launcher.stdin.close()
            launcher.wait(timeout)
        except Exception as exc:                      # noqa: BLE001
            _log.warning("launcher (pid %s) for pid %s did not finish cleanly: %r -- the "
                         "command is followed anyway", launcher.pid, watch.pid, exc)
        return watch
    finally:
        for stream in (launcher.stdin, launcher.stdout, launcher.stderr):
            if stream is launcher.stdout and reader_owns_stdout:
                continue
            try:
                stream.close()
            except Exception:                         # noqa: BLE001
                pass


def _start_job(job: dict, now: float | None = None):
    """The launcher's work: start ``job["command"]`` -- unless its
    ``not_after`` (epoch seconds) has passed, when the server has stopped
    waiting for this launcher and may already be looking for the job
    elsewhere: then nothing is started.  Returns the child's Popen, or raises
    (the reason goes back to the server)."""
    now = time.time() if now is None else now
    not_after = job.get("not_after")
    if not_after is not None and now > float(not_after):
        raise TimeoutError(f"too late: the server stopped waiting {now - float(not_after):.1f} s "
                           "ago; the command was not started")
    with open(job["log_path"], "ab") as out:
        return Popen(job["command"], shell=True, cwd=job.get("cwd") or None,
                     stdin=DEVNULL, stdout=out, stderr=STDOUT,
                     creationflags=_NO_WINDOW | _NEW_GROUP, close_fds=True)


def _launcher_main() -> None:
    """The launcher process (see the module docstring)."""
    try:
        child = _start_job(json.loads(sys.stdin.readline()))
    except Exception as exc:                          # noqa: BLE001
        sys.stdout.write(json.dumps({"error": repr(exc)}) + "\n")
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.write(json.dumps({"pid": child.pid}) + "\n")
    sys.stdout.flush()
    sys.stdin.readline()                              # the server holds a handle now
    os._exit(0)


if __name__ == "__main__":
    _launcher_main()
