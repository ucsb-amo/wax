"""Launch and supervise the monitor experiment subprocess.

Everything reported here goes through :mod:`logging` rather than ``print``.
The dashboard runs the monitor server as a child process and drains its
stdout/stderr line by line; ``print`` writes to stdout, which Python
block-buffers when it is a pipe, so printed errors only surfaced in the
server dashboard terminal whenever the 8 kB buffer happened to flush (often
never, since the monitor process is long-lived).  Logging writes to stderr,
which is line buffered, so a failure shows up in the terminal as it happens.

Failures are reported with the exit code, the launch command, a tail of the
child's output, a guess at the cause, and the environment this process
actually sees — enough to tell a missing env var from a compile error from
another experiment stealing the core device.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import traceback
from collections import deque
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

from PyQt6.QtCore import pyqtSignal, QThread

log = logging.getLogger(__name__)

# How many lines of the monitor's own output to keep for the post-mortem.
_TAIL_LINES = 25

# Environment variables the launch command depends on, reported verbatim when
# a start fails so a wrong/unset one is obvious instead of being inferred.
_LAUNCH_ENV_VARS = ("kpy", "code", "db", "data")

# Recognised failure signatures, in priority order: (substrings, explanation).
_FAILURE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("is not recognized as an internal or external command",
      "is not recognized as the name of a cmdlet",
      "cannot find the path specified"),
     "the shell could not resolve the launch command -- the lab environment "
     "variables (%kpy%) or the shortcuts folder holding 'ar' are missing from "
     "this process's environment. See the environment block below."),
    (("CompileError", "artiq.compiler", "compilation failed"),
     "the monitor experiment failed to COMPILE -- fix the compiler error above "
     "in the monitor experiment file."),
    (("ModuleNotFoundError", "ImportError"),
     "the monitor experiment failed to IMPORT -- a module is missing or the "
     "wrong Python environment was activated (%kpy%)."),
    (("ConnectionRefusedError", "WinError 10061", "WinError 10060",
      "Cannot connect to device", "TimeoutError", "timed out"),
     "the ARTIQ core device did not answer -- check that the Kasli is powered "
     "and reachable, and that no other process is holding the core device."),
    (("device_db", "DeviceError", "KeyError: 'core'"),
     "the device database could not be read or is missing a device -- check "
     "that %db% points at a valid device_db.py."),
    (("No such file or directory", "can't open file", "FileNotFoundError"),
     "a file in the launch could not be opened -- most often the monitor "
     "experiment path itself."),
    (("RTIOUnderflow",),
     "the monitor experiment hit an RTIOUnderflow -- the monitor sequence "
     "itself needs fixing (slack ran out)."),
    (("SyntaxError", "IndentationError"),
     "the monitor experiment (or something it imports) has a syntax error."),
)

# Signatures meaning "another experiment took the core device", which is
# expected every time a run is submitted and is not a failure.
_INTERRUPTED_SIGNATURES = ("WinError 10054", "forcibly closed by the remote host")


def _matches(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _diagnose(text: str) -> list[str]:
    """Return human explanations for every failure signature found in *text*."""
    return [why for needles, why in _FAILURE_HINTS if _matches(text, needles)]


class MonitorManager(QThread):
    msg = pyqtSignal(str)
    monitor_stopped = pyqtSignal(str)

    def __init__(self, monitor_expt_path):
        super().__init__()
        self.monitor_expt_path = monitor_expt_path
        # Child process running the monitor experiment ("ar <expt>").  Owned by
        # this thread so it can be killed from the outside without ever calling
        # QThread.terminate() (which on Windows force-kills the thread mid-Python
        # execution and corrupts the interpreter -> 0xC0000005 access violation).
        self._proc = None
        self._proc_lock = threading.Lock()
        self._stop_requested = False
        # Last failure, kept so a GUI can show it (tooltip/status) without
        # having to scrape the terminal.
        self.last_error: str | None = None
        self.last_exit_code: int | None = None
        # How the monitor last ended, machine-readable, for the server's
        # structured status (``status_json`` sub_state).  ``None`` until the
        # first start.  One of: "interrupted_by_run", "exited", "failed",
        # "preflight_failed", "stopped_on_request".  ``last_stop_reason`` is
        # the human-readable string that was emitted with ``monitor_stopped``.
        self.last_stop_kind: str | None = None
        self.last_stop_reason: str | None = None

    @property
    def pid(self) -> int | None:
        """PID of the running monitor experiment process, or ``None``."""
        with self._proc_lock:
            proc = self._proc
        return proc.pid if proc is not None else None

    def _ended(self, kind: str, reason: str) -> None:
        """Record how the monitor ended and tell listeners."""
        self.last_stop_kind = kind
        self.last_stop_reason = reason
        self.monitor_stopped.emit(reason)

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    @property
    def launch_command(self) -> str:
        return r"%kpy% & ar " + str(self.monitor_expt_path)

    def preflight_problems(self) -> list[str]:
        """Return fatal reasons the monitor cannot be started, if any.

        Checked *before* spawning so a bad path is reported as a bad path
        instead of as whatever the shell makes of ``ar None``.
        """
        problems: list[str] = []
        path = self.monitor_expt_path
        if path is None or str(path).strip() in ("", "None"):
            problems.append(
                "the monitor experiment path is not set (got "
                f"{path!r}) -- whoever constructed MonitorManager could not "
                "resolve it, which normally means the env var its config is "
                "built from is unset in this process."
            )
            return problems
        try:
            resolved = Path(str(path))
            if not resolved.exists():
                problems.append(
                    f"the monitor experiment file does not exist: {resolved} "
                    "-- check the path config and that the data/code drives are "
                    "mapped for this user."
                )
            elif resolved.is_dir():
                problems.append(
                    f"the monitor experiment path is a directory, not a file: {resolved}"
                )
        except OSError as exc:
            problems.append(f"the monitor experiment path could not be checked: {exc!r}")
        return problems

    def _environment_report(self) -> list[str]:
        """Lines describing the launch environment, for failure post-mortems."""
        lines = []
        for var in _LAUNCH_ENV_VARS:
            value = os.environ.get(var)
            lines.append(f"%{var}% = {value if value else '<UNSET>'}")
        found = shutil.which("ar") or shutil.which("artiq_run")
        lines.append(f"'ar'/'artiq_run' on PATH = {found or '<NOT FOUND>'}")
        lines.append(f"working directory = {os.getcwd()}")
        return lines

    # ------------------------------------------------------------------
    # Start / run
    # ------------------------------------------------------------------

    def start(self, *args, **kwargs):
        """Pre-flight, then start the QThread that spawns the experiment.

        Refuses (loudly, instead of letting Qt warn and do nothing) when the
        monitor is already running or the experiment path is unusable.
        """
        if self.isRunning():
            log.warning(
                "Monitor start ignored: the monitor experiment is already running."
            )
            return
        problems = self.preflight_problems()
        if problems:
            log.error("Cannot start the monitor experiment:")
            for problem in problems:
                log.error("  - %s", problem)
            log.error("  attempted command: %s", self.launch_command)
            for line in self._environment_report():
                log.error("  %s", line)
            self.last_error = problems[0]
            self.last_exit_code = None
            self._ended("preflight_failed", f"not started: {problems[0]}")
            return
        super().start(*args, **kwargs)

    def run(self):
        self._stop_requested = False
        try:
            self.run_expt()
        except Exception:
            # A crash here would otherwise vanish with the thread.
            log.error("Monitor manager thread crashed:\n%s", traceback.format_exc())
            self.last_error = "monitor manager thread crashed (see traceback above)"
            self._ended("failed", self.last_error)

    def run_expt(self):
        expt_path = self.monitor_expt_path
        command = self.launch_command
        tail: deque[str] = deque(maxlen=_TAIL_LINES)
        self.last_error = None
        self.last_exit_code = None

        try:
            log.info("Starting monitor experiment: %s", expt_path)
            log.debug("Monitor launch command: %s", command)
            self.msg.emit("Starting monitor...")

            with self._proc_lock:
                if self._stop_requested:
                    log.info("Monitor start aborted: stop was requested before launch.")
                    self.last_stop_kind = "stopped_on_request"
                    self.last_stop_reason = "stopped on request before launch"
                    return
                try:
                    # stderr is merged into stdout so the child's output keeps
                    # its original ordering, and is streamed line by line (not
                    # collected by communicate()) so a compile error or a hang
                    # is visible while it happens rather than after the fact.
                    self._proc = Popen(command, stdout=PIPE, stderr=STDOUT,
                                       universal_newlines=True, bufsize=1,
                                       errors="replace", shell=True)
                except OSError as exc:
                    log.error("Could not spawn the monitor process: %r", exc)
                    log.error("  attempted command: %s", command)
                    for line in self._environment_report():
                        log.error("  %s", line)
                    self.last_error = f"could not spawn the monitor process: {exc!r}"
                    self._ended("failed", self.last_error)
                    return
            proc = self._proc
            log.info("Monitor experiment started (pid %s).", proc.pid)

            if proc.stdout is not None:
                for raw_line in proc.stdout:
                    line = raw_line.rstrip()
                    if not line:
                        continue
                    tail.append(line)
                    log.info("[monitor] %s", line)
            returncode = proc.wait()
            self.last_exit_code = returncode
            self._report_exit(returncode, command, expt_path, tail)
        except Exception:
            log.error(
                "Monitor experiment supervision failed:\n%s", traceback.format_exc()
            )
            self.last_error = "monitor supervision failed (see traceback above)"
            self._ended("failed", self.last_error)
        finally:
            with self._proc_lock:
                self._proc = None

    # ------------------------------------------------------------------
    # Exit reporting
    # ------------------------------------------------------------------

    def _report_exit(self, returncode, command, expt_path, tail) -> None:
        """Classify how the monitor ended and say so at the right log level."""
        output = "\n".join(tail)

        if self._stop_requested:
            log.info("Monitor experiment stopped on request (exit code %s).", returncode)
            self._ended("stopped_on_request", "stopped on request")
            return

        if _matches(output, _INTERRUPTED_SIGNATURES):
            log.info(
                "Monitor interrupted (exit code %s) -- the core device connection "
                "was closed. Expected if an experiment was just submitted.",
                returncode,
            )
            self._ended(
                "interrupted_by_run",
                "interrupted -- another experiment was probably submitted",
            )
            return

        if returncode == 0:
            log.warning(
                "Monitor experiment exited on its own with code 0 -- the hardware "
                "is no longer being held in the monitor idle state."
            )
            self._ended("exited", "exited with code 0")
            return

        hints = _diagnose(output)
        log.error("Monitor experiment FAILED (exit code %s).", returncode)
        log.error("  experiment file: %s", expt_path)
        log.error("  command: %s", command)
        if hints:
            for hint in hints:
                log.error("  likely cause: %s", hint)
        else:
            log.error(
                "  no known failure signature in the output -- read the last "
                "lines below for the actual error."
            )
        if tail:
            log.error("  last %d line(s) of monitor output:", len(tail))
            for line in tail:
                log.error("    | %s", line)
        else:
            log.error(
                "  the monitor produced NO output at all, which usually means the "
                "command never ran (see the environment below)."
            )
        for line in self._environment_report():
            log.error("  %s", line)

        summary = f"exit code {returncode}"
        if hints:
            summary = f"{summary}: {hints[0]}"
        elif tail:
            summary = f"{summary}: {tail[-1]}"
        self.last_error = summary
        self._ended("failed", summary)

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self, timeout_ms=1500):
        """Gracefully stop the monitor experiment.

        Kills the spawned child process tree (``ar`` + its descendants) so the
        blocking read of the child's output returns and ``run()`` exits on its
        own.  This replaces ``QThread.terminate()``, which force-kills the
        thread while it holds the GIL and crashes the whole dashboard
        subprocess.
        """
        self._stop_requested = True
        with self._proc_lock:
            proc = self._proc
        if proc is not None:
            pid = proc.pid
            log.info("Stopping monitor experiment (pid %s).", pid)
            killed = False
            try:
                from waxx.util.dashboard.server_supervisor import _kill_pid_tree  # noqa: PLC0415
                killed = _kill_pid_tree(pid)
            except Exception as exc:
                log.warning(
                    "Could not use the process-tree killer for pid %s (%r); "
                    "falling back to kill().", pid, exc,
                )
                killed = False
            if not killed:
                try:
                    proc.kill()
                except Exception as exc:
                    log.error("Failed to kill the monitor process (pid %s): %r", pid, exc)
        if not self.wait(timeout_ms):
            log.warning(
                "Monitor manager thread did not exit within %d ms -- the monitor "
                "process may still be alive.", timeout_ms,
            )
