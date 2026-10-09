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
import os
import subprocess
import sys
from subprocess import DEVNULL, PIPE, STDOUT, Popen

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_STILL_ACTIVE = 259
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
        """A handle to process ``pid`` (Windows); None when there is none."""
        if sys.platform != "win32":
            return None
        ctypes, wintypes, k32 = _k32()
        handle = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, False,
                                 int(pid))
        if not handle:
            return None
        return cls(pid, handle=handle)

    @classmethod
    def adopt(cls, pid, started: float | None) -> "ProcessWatch | None":
        """Take a process back after a restart: a watch on ``pid`` when it is
        alive and was created at ``started`` (None: not recorded -- then any
        live process with that pid is taken, which a reused pid could fool);
        None when it is gone or is another process."""
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
        if started is not None and (watch.started is None
                                    or abs(watch.started - float(started)) > SAME_PROCESS_S):
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


def launch(command: str, *, cwd: str, env: dict, log_path: str,
           timeout: float = LAUNCH_TIMEOUT_S) -> ProcessWatch:
    """Start ``command`` (a shell command line) detached, its output appended
    to ``log_path``; return a :class:`ProcessWatch` on it.  Raises OSError
    when it could not be started."""
    if sys.platform != "win32":
        with open(log_path, "ab") as out:
            popen = Popen(command, shell=True, cwd=cwd, env=env, stdin=DEVNULL, stdout=out,
                          stderr=STDOUT, start_new_session=True)
        return ProcessWatch(popen.pid, popen=popen)
    launcher = Popen([sys.executable, "-m", "waxx.util.device_state.detached"],
                     stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env,
                     creationflags=_NO_WINDOW, close_fds=True, text=True,
                     encoding="utf-8", errors="replace")
    try:
        launcher.stdin.write(json.dumps({"command": command, "cwd": cwd,
                                         "log_path": log_path}) + "\n")
        launcher.stdin.flush()
        line = launcher.stdout.readline()
        try:
            reply = json.loads(line) if line.strip() else {}
        except ValueError:
            reply = {"error": f"unreadable launcher reply {line!r}"}
        if "pid" not in reply:
            launcher.stdin.close()
            launcher.wait(timeout)
            err = (launcher.stderr.read() or "").strip()
            raise OSError(f"the launcher did not start the command: "
                          f"{reply.get('error') or err or 'no reply'}")
        watch = ProcessWatch.open(reply["pid"])
        launcher.stdin.write("ok\n")                  # handle held: the launcher may go
        launcher.stdin.flush()
        launcher.stdin.close()
        launcher.wait(timeout)
        if watch is None:
            # it ended before a handle could be taken: its exit code is lost,
            # but nothing else is (the log file has its output)
            watch = ProcessWatch(reply["pid"], started=reply.get("started"))
            watch._code = -1
        return watch
    except subprocess.TimeoutExpired as exc:
        raise OSError(f"the launcher did not finish within {timeout:.0f} s") from exc
    finally:
        for stream in (launcher.stdout, launcher.stderr):
            try:
                stream.close()
            except Exception:                         # noqa: BLE001
                pass


def _launcher_main() -> None:
    """The launcher process (see the module docstring)."""
    try:
        job = json.loads(sys.stdin.readline())
        with open(job["log_path"], "ab") as out:
            child = Popen(job["command"], shell=True, cwd=job.get("cwd") or None,
                          stdin=DEVNULL, stdout=out, stderr=STDOUT,
                          creationflags=_NO_WINDOW | _NEW_GROUP, close_fds=True)
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
