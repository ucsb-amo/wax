"""Keep one server process running: the Qt-free supervision core.

The dashboard's :class:`~waxx.util.dashboard.server_supervisor.ServerSupervisor`
runs the lab's servers from a Qt GUI (``QProcess``). A server on a PC with no
dashboard -- the SLM server on the SLM PC -- needs the same care without Qt:
start it, collect its output, restart it when it asks to be restarted or dies,
back off when it keeps dying, kill it when it hangs, stop it gracefully. That
is :class:`ProcessSupervisor`. The dashboard's process-table helpers live here
too (``kill_pid_trees``, ``install_console_signal_guard``), so both use one
copy.

Exit codes are the child's protocol with its supervisor:

* a code in ``stop_codes`` (default ``{0}``): the child stopped on purpose --
  the supervisor stops too;
* a code in ``restart_codes`` (default ``{75}``, ``EX_TEMPFAIL``): the child
  asks to be started again -- at once, and not counted as a crash;
* anything else is a crash: started again after a bounded exponential delay
  (:class:`RestartPolicy`, the dashboard's policy), or -- after too many crashes
  in a short window -- at a slow retry period instead of giving up, since the
  point of an unattended supervisor is that it comes back on its own.

On Windows the child is put in a job object that kills it when the supervisor
exits however it exits (closed console, killed process), so a server can never
outlive its supervisor and hold its port against the next one. It is also
started in its own process group, so Ctrl+C in the shared console reaches only
the supervisor, which then stops the child gracefully.

Standard library only: this module is imported on PCs that have no Qt.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

_LOG = logging.getLogger("waxx.supervise")

_IS_WINDOWS = sys.platform.startswith("win")
# Windows CreateProcess flags (see ServerSupervisor for why the dashboard uses them).
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

#: The child asks to be started again (``EX_TEMPFAIL``).
EXIT_RESTART = 75


# --------------------------------------------------------------------------------------
# Process-table helpers (moved here from waxx.util.dashboard.server_supervisor)
# --------------------------------------------------------------------------------------

def kill_pid_trees(pids: Iterable[int], timeout_s: float = 5.0) -> bool:
    """Forcibly terminate several processes *and their child trees* (Windows).

    One ``taskkill`` call with repeated ``/PID`` flags, so closing a
    dashboard with nine servers spawns one helper process instead of nine.
    Scoped strictly to the given pids and their descendants; it can never
    reach sibling servers or the dashboard.  Returns True if the command was
    issued.  No-op (returns False) off Windows or with no pids.
    """
    pids = [int(p) for p in pids if p and int(p) > 0]
    if not _IS_WINDOWS or not pids:
        return False
    try:
        args = ["taskkill"]
        for pid in pids:
            args += ["/PID", str(pid)]
        args += ["/T", "/F"]
        subprocess.run(
            args,
            creationflags=CREATE_NO_WINDOW,
            capture_output=True,
            timeout=timeout_s,
        )
        return True
    except Exception as exc:  # pragma: no cover
        _LOG.debug("kill_pid_trees(%s) raised: %r", pids, exc)
        return False


def kill_pid_tree(pid: int) -> bool:
    """Single-pid convenience wrapper around :func:`kill_pid_trees`."""
    return kill_pid_trees([pid])


# Keep strong module-level references to the installed console-control
# handler and its ctypes prototype.  A handler that gets garbage-collected
# while still registered crashes the process when the OS next invokes it.
_CONSOLE_GUARD_INSTALLED = False
_CONSOLE_GUARD_CB = None  # type: ignore[var-annotated]


def install_console_signal_guard() -> bool:
    """Make this process immune to CTRL_C / CTRL_BREAK.

    A process may run as a console app (``python.exe``) sharing its console
    with child processes.  A console signal aimed at a child can leak back to
    its own process group and kill it.  Installing a console-control handler
    that reports CTRL_C / CTRL_BREAK as *handled* suppresses the default
    terminating behaviour.

    Other control events (CLOSE / LOGOFF / SHUTDOWN) are left unhandled so
    normal window-close shutdown still works.

    Returns True if the guard is installed (or was already installed).
    No-op (returns False) off Windows.
    """
    global _CONSOLE_GUARD_INSTALLED, _CONSOLE_GUARD_CB
    if not _IS_WINDOWS:
        return False
    if _CONSOLE_GUARD_INSTALLED:
        return True
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        CTRL_C_EVENT = 0
        CTRL_BREAK_EVENT = 1

        handler_proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _handler(ctrl_type):  # noqa: ANN001 - ctypes callback
            return ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT)

        cb = handler_proto(_handler)
        kernel32 = ctypes.windll.kernel32
        ok = bool(kernel32.SetConsoleCtrlHandler(cb, True))
        if not ok:
            _LOG.debug("SetConsoleCtrlHandler failed err=%s", ctypes.get_last_error())
            return False
        _CONSOLE_GUARD_CB = cb
        _CONSOLE_GUARD_INSTALLED = True
        _LOG.info("console signal guard installed (CTRL_C/CTRL_BREAK ignored)")
        return True
    except Exception as exc:  # pragma: no cover
        _LOG.debug("install_console_signal_guard raised: %r", exc)
        return False


class KillOnCloseJob:
    """A Windows job object that kills its processes when its last handle closes.

    The supervisor holds the only handle, so its exit -- clean, killed, or a
    closed console window -- takes the child with it. ``assign`` returns False
    (and the child simply is not in a job) off Windows or when the OS refuses,
    e.g. a parent job that forbids nested jobs; the supervisor then says so.
    """

    def __init__(self):
        self._handle = None
        self.error = ""
        if not _IS_WINDOWS:
            self.error = "not Windows"
            return
        try:
            import ctypes  # noqa: PLC0415
            from ctypes import wintypes  # noqa: PLC0415

            class _IoCounters(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in
                            ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                             "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class _Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                            ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class _Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", _Basic),
                            ("IoInfo", _IoCounters),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                    ctypes.c_void_p, wintypes.DWORD]
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            job = k32.CreateJobObjectW(None, None)
            if not job:
                self.error = f"CreateJobObject failed ({ctypes.get_last_error()})"
                return
            info = _Extended()
            info.BasicLimitInformation.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                self.error = f"SetInformationJobObject failed ({ctypes.get_last_error()})"
                k32.CloseHandle(job)
                return
            self._k32 = k32
            self._handle = job
        except Exception as exc:  # pragma: no cover
            self.error = f"{type(exc).__name__}: {exc}"

    def assign(self, proc: subprocess.Popen) -> bool:
        if self._handle is None:
            return False
        import ctypes  # noqa: PLC0415
        handle = getattr(proc, "_handle", None)
        if handle is None:
            self.error = "the process has no handle"
            return False
        if not self._k32.AssignProcessToJobObject(self._handle, int(handle)):
            self.error = f"AssignProcessToJobObject failed ({ctypes.get_last_error()})"
            return False
        return True


# --------------------------------------------------------------------------------------
# Restart policy (the dashboard's: bounded exponential backoff)
# --------------------------------------------------------------------------------------

@dataclass
class RestartPolicy:
    """Delay before restarting a crashed child.

    ``initial_s`` doubles with every crash inside ``window_s``, up to ``max_s``.
    After ``max_in_window`` crashes inside ``window_s`` the child is failing:
    ``slow_retry_s`` between attempts, or -- with ``slow_retry_s`` None -- no
    more attempts (the dashboard's FAILED). A child that ran ``stable_s``
    without crashing clears the history.
    """

    initial_s: float = 2.0
    max_s: float = 60.0
    window_s: float = 600.0
    max_in_window: int = 5
    slow_retry_s: Optional[float] = 600.0
    stable_s: float = 300.0
    history: list = field(default_factory=list)

    def ran_for(self, seconds: float) -> None:
        if seconds >= self.stable_s:
            self.history.clear()

    def next_delay(self, now: float) -> Optional[float]:
        """Record a crash at ``now``; the delay before the next start, or None
        for "stop trying"."""
        self.history = [t for t in self.history if t >= now - self.window_s]
        self.history.append(now)
        n = len(self.history)
        if n > self.max_in_window:
            return self.slow_retry_s
        return min(self.initial_s * 2 ** (n - 1), self.max_s)

    @property
    def failing(self) -> bool:
        return len(self.history) > self.max_in_window


# --------------------------------------------------------------------------------------
# The supervisor
# --------------------------------------------------------------------------------------

@dataclass
class ChildExit:
    code: Optional[int]
    reason: str               # "stopped" | "restart" | "crash" | "hung" | "stop_requested"
    ran_s: float
    detail: str = ""


class ProcessSupervisor:
    """Run ``cmd`` and keep it running (see the module docstring).

    * ``health(pid) -> (ok, why)``: asked every ``health_period_s`` once the
      child has run ``health_grace_s``; ``ok`` False ``health_failures`` times
      in a row kills the child (a hang), which then restarts like a crash.
      ``ok`` None means "no information yet" and counts as a failure only
      after the grace period.
    * ``graceful_stop(pid) -> bool``: asks the child to exit by itself (its own
      protocol); after ``stop_timeout_s`` the child's tree is killed.
    * ``on_line(text)``: every line the child writes (stdout and stderr, merged).
    * ``on_event(kind, info)``: "start", "exit", "restart_in", "hung",
      "stopping", "job" -- for logs and status files.
    * The child's environment gets ``SUPERVISOR_PID``, ``SUPERVISOR_START_COUNT``
      (1, 2, ...) and ``SUPERVISOR_LAST_EXIT`` ("" or "<code> <reason>").
    """

    def __init__(self, name: str, cmd: list, *, cwd: Optional[str] = None,
                 env: Optional[dict] = None,
                 stop_codes: Iterable[int] = (0,), restart_codes: Iterable[int] = (EXIT_RESTART,),
                 policy: Optional[RestartPolicy] = None,
                 health: Optional[Callable[[int], tuple]] = None,
                 health_period_s: float = 5.0, health_grace_s: float = 60.0,
                 health_failures: int = 2,
                 graceful_stop: Optional[Callable[[int], bool]] = None,
                 stop_timeout_s: float = 10.0,
                 on_line: Optional[Callable[[str], None]] = None,
                 on_event: Optional[Callable[[str, dict], None]] = None,
                 new_process_group: bool = True, kill_on_close: bool = True,
                 poll_s: float = 0.2):
        self.name = name
        self.cmd = list(cmd)
        self.cwd = cwd
        self.env = dict(env or {})
        self.stop_codes = set(stop_codes)
        self.restart_codes = set(restart_codes)
        self.policy = policy or RestartPolicy()
        self.health = health
        self.health_period_s = float(health_period_s)
        self.health_grace_s = float(health_grace_s)
        self.health_failures = int(health_failures)
        self.graceful_stop = graceful_stop
        self.stop_timeout_s = float(stop_timeout_s)
        self.on_line = on_line or (lambda text: None)
        self.on_event = on_event or (lambda kind, info: None)
        self.new_process_group = bool(new_process_group)
        self.poll_s = float(poll_s)

        self._stop = threading.Event()
        self._stop_reason = ""
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.starts = 0
        self.exits: list[ChildExit] = []
        self.job = KillOnCloseJob() if kill_on_close else None

    # -- control ------------------------------------------------------------------------

    def stop(self, reason: str = "") -> None:
        """Stop the child (gracefully, then by force) and end :meth:`run`."""
        self._stop_reason = reason or "stop requested"
        self._stop.set()

    @property
    def pid(self) -> Optional[int]:
        p = self._proc
        return None if p is None else p.pid

    def child_alive(self) -> bool:
        p = self._proc
        return p is not None and p.poll() is None

    # -- the loop -----------------------------------------------------------------------

    def run(self) -> Optional[int]:
        """Run until the child stops by itself (a stop code), the policy gives
        up, or :meth:`stop`. Returns the child's last exit code."""
        last: Optional[ChildExit] = None
        while not self._stop.is_set():
            last = self._run_once(last)
            self.exits.append(last)
            self.on_event("exit", {"code": last.code, "reason": last.reason,
                                   "ran_s": round(last.ran_s, 1), "detail": last.detail})
            if last.reason in ("stopped", "stop_requested"):
                break
            if last.reason == "restart":
                delay = 0.0
            else:
                self.policy.ran_for(last.ran_s)
                delay = self.policy.next_delay(time.monotonic())
                if delay is None:
                    self.on_event("give_up", {"crashes": len(self.policy.history)})
                    break
            self.on_event("restart_in", {"delay_s": delay, "failing": self.policy.failing,
                                         "after": last.reason})
            try:
                if delay > 0 and self._stop.wait(delay):
                    break
            except KeyboardInterrupt:
                self.stop("Ctrl+C")
                break
        return None if last is None else last.code

    def _spawn(self, last: Optional[ChildExit]) -> subprocess.Popen:
        env = dict(os.environ)
        env.update(self.env)
        self.starts += 1
        env["SUPERVISOR_PID"] = str(os.getpid())
        env["SUPERVISOR_START_COUNT"] = str(self.starts)
        env["SUPERVISOR_LAST_EXIT"] = "" if last is None else f"{last.code} {last.reason}"
        env.setdefault("PYTHONUNBUFFERED", "1")
        flags = CREATE_NEW_PROCESS_GROUP if (_IS_WINDOWS and self.new_process_group) else 0
        proc = subprocess.Popen(self.cmd, cwd=self.cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=flags)
        if self.job is not None:
            in_job = self.job.assign(proc)
            self.on_event("job", {"ok": in_job, "error": "" if in_job else self.job.error})
        return proc

    def _pump(self, proc: subprocess.Popen) -> threading.Thread:
        def run():
            for raw in iter(proc.stdout.readline, b""):
                try:
                    self.on_line(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
                except Exception:
                    pass
        t = threading.Thread(target=run, name=f"{self.name}-output", daemon=True)
        t.start()
        return t

    def _run_once(self, last: Optional[ChildExit]) -> ChildExit:
        t0 = time.monotonic()
        try:
            proc = self._spawn(last)
        except OSError as e:
            return ChildExit(None, "crash", 0.0, f"could not start: {e}")
        with self._lock:
            self._proc = proc
        self.on_event("start", {"pid": proc.pid, "start": self.starts, "cmd": self.cmd})
        pump = self._pump(proc)
        failures, next_health = 0, t0 + self.health_grace_s
        reason, detail = "", ""
        try:
            while True:
                try:
                    code = proc.wait(timeout=self.poll_s)
                    break
                except subprocess.TimeoutExpired:
                    pass
                except KeyboardInterrupt:           # Ctrl+C: stop the child gracefully
                    self.stop("Ctrl+C")
                if self._stop.is_set():
                    code = self._stop_child(proc)
                    reason, detail = "stop_requested", self._stop_reason
                    break
                now = time.monotonic()
                if self.health is not None and now >= next_health:
                    next_health = now + self.health_period_s
                    try:
                        ok, why = self.health(proc.pid)
                    except Exception as e:
                        ok, why = None, f"health check raised {type(e).__name__}: {e}"
                    if ok:
                        failures = 0
                    else:
                        failures += 1
                        if failures >= self.health_failures:
                            self.on_event("hung", {"pid": proc.pid, "why": why})
                            kill_pid_tree(proc.pid) or proc.kill()
                            code = proc.wait()
                            reason, detail = "hung", why
                            break
        finally:
            pump.join(timeout=2.0)
            with self._lock:
                self._proc = None
        ran = time.monotonic() - t0
        if not reason:
            if code in self.stop_codes:
                reason = "stopped"
            elif code in self.restart_codes:
                reason = "restart"
            else:
                reason = "crash"
        return ChildExit(code, reason, ran, detail)

    def _stop_child(self, proc: subprocess.Popen) -> Optional[int]:
        self.on_event("stopping", {"pid": proc.pid, "reason": self._stop_reason})
        if self.graceful_stop is not None:
            try:
                self.graceful_stop(proc.pid)
            except Exception as e:
                self.on_event("stopping", {"pid": proc.pid, "graceful_error": repr(e)})
            try:
                return proc.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                pass
        kill_pid_tree(proc.pid) or proc.kill()
        try:
            return proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            return None


def stamp() -> str:
    """Local wall-clock time for log lines."""
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
