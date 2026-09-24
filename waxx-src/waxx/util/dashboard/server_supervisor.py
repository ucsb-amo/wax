"""Subprocess supervisor for server panels.

Wraps :class:`QProcess` with:

* state machine reported via Qt signals (IDLE -> STARTING -> RUNNING -> STOPPING -> IDLE / CRASHED / FAILED / EXTERNAL)
* graceful stop: ask the server to shut itself down over its own TCP
  protocol (``client.request_shutdown()``), wait a bounded time, then kill
  the process tree
* bounded exponential restart on crash (5 attempts within 60 s, then FAILED)
* external-instance detection through the discovery beacon registry so a
  dashboard never double-starts a server somebody launched from a ``.bat``
  (panel enters EXTERNAL)
* the shared-data-dir precheck runs on a worker thread so a slow or
  unmapped network share can never freeze the GUI
* always-drained stdout/stderr (a UI that pauses must never block the child)

The supervisor is the only piece of the framework that touches the OS process
table, so all of the "don't leak processes" logic lives here.  The dashboard
window's ``closeEvent`` is the one authoritative shutdown path; there is no
second atexit pass.
"""

from __future__ import annotations

import enum
import logging
import socket
import sys
import threading
import time
from typing import Callable, Iterable, Optional

from PyQt6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QRunnable,
    QThreadPool,
    QTimer,
    pyqtSignal,
)


_LOG = logging.getLogger("waxx.dashboard.supervisor")

_IS_WINDOWS = sys.platform.startswith("win")
# Windows CreateProcess flags.
# CREATE_NEW_PROCESS_GROUP — child gets its own process group so console
# signals aimed at it can never reach the dashboard.
# CREATE_NO_WINDOW — the headless Python child neither inherits nor allocates
# a console.  Without this, QProcess.terminate() posts WM_CLOSE to the shared
# console window, which terminates every process attached to that console —
# including the dashboard itself.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000


def _kill_pid_trees(pids: Iterable[int], timeout_s: float = 5.0) -> bool:
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
        import subprocess  # noqa: PLC0415
        args = ["taskkill"]
        for pid in pids:
            args += ["/PID", str(pid)]
        args += ["/T", "/F"]
        subprocess.run(
            args,
            creationflags=_CREATE_NO_WINDOW,
            capture_output=True,
            timeout=timeout_s,
        )
        return True
    except Exception as exc:  # pragma: no cover
        _LOG.debug("_kill_pid_trees(%s) raised: %r", pids, exc)
        return False


def _kill_pid_tree(pid: int) -> bool:
    """Single-pid convenience wrapper around :func:`_kill_pid_trees`."""
    return _kill_pid_trees([pid])


# Keep strong module-level references to the installed console-control
# handler and its ctypes prototype.  A handler that gets garbage-collected
# while still registered crashes the process when the OS next invokes it.
_CONSOLE_GUARD_INSTALLED = False
_CONSOLE_GUARD_CB = None  # type: ignore[var-annotated]


def install_console_signal_guard() -> bool:
    """Make the dashboard process immune to CTRL_C / CTRL_BREAK.

    The dashboard may run as a console app (``python.exe``) sharing its
    console with child processes.  A console signal aimed at a child can
    leak back to the dashboard's own process group and kill the whole GUI.
    Installing a console-control handler that reports CTRL_C / CTRL_BREAK
    as *handled* suppresses the default terminating behaviour.

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


class SupervisorState(enum.Enum):
    IDLE = "IDLE"             # not started, no pending action
    STARTING = "STARTING"     # precheck running or QProcess.start() invoked, not yet RUNNING
    RUNNING = "RUNNING"       # subprocess alive
    STOPPING = "STOPPING"     # shutdown requested, waiting for exit
    CRASHED = "CRASHED"       # exited non-zero; eligible for restart if enabled
    FAILED = "FAILED"         # too many restart attempts in window
    EXTERNAL = "EXTERNAL"     # another instance is already advertising / bound


def _is_port_in_use(host: str, port: int, timeout_s: float = 0.2) -> bool:
    """Return True if a TCP listener is already bound on (host, port)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout_s)
            try:
                return s.connect_ex((host or "127.0.0.1", int(port))) == 0
            except OSError:
                return False
    except Exception:
        return False


def _beacon_seen(server_id: str) -> bool:
    """True if the discovery registry currently holds a beacon for *server_id*.

    Cache-only lookup (zero timeout): never blocks the GUI thread.  Callers
    that want a fresh answer give the registry time to collect beacons first
    (they arrive every 0.5 s) - the dashboard delays autostart for that.
    """
    try:
        from beacon.discovery.client import discover  # noqa: PLC0415
        return discover(server_id, timeout=0.0) is not None
    except Exception:
        return False


class _PrecheckTask(QRunnable):
    """Run the data-dir precheck on a pool thread and report back via *done*."""

    def __init__(self, fn: Callable[[], tuple[bool, str]], done: Callable[[bool, str], None]):
        super().__init__()
        self.setAutoDelete(True)
        self._fn = fn
        self._done = done

    def run(self) -> None:  # noqa: D401 - QRunnable hook
        try:
            ok, msg = self._fn()
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"precheck raised {exc!r}"
        self._done(ok, msg)


class ServerSupervisor(QObject):
    """Owns a single supervised subprocess.

    Signals
    -------
    state_changed(SupervisorState)
        Emitted on every state transition.
    log_line(str)
        Emitted once per line of stdout/stderr (including drained even when
        the dashboard's log tail widget is hidden).
    crashed(int)
        Emitted with the exit code when the subprocess crashes.
    """

    state_changed = pyqtSignal(object)  # SupervisorState
    log_line = pyqtSignal(str)
    crashed = pyqtSignal(int)
    # Internal: precheck worker -> main thread.
    _precheck_finished = pyqtSignal(bool, str)

    # Restart-storm guard: at most this many starts within RESTART_WINDOW_S.
    MAX_RESTARTS = 5
    RESTART_WINDOW_S = 60.0
    INITIAL_RESTART_DELAY_S = 0.5
    MAX_RESTART_DELAY_S = 30.0

    def __init__(
        self,
        server_id: str,
        cmd: list[str],
        *,
        cwd: Optional[str] = None,
        env_extra: Optional[dict[str, str]] = None,
        graceful_stop_timeout_s: float = 1.5,
        restart_on_crash: bool = False,
        snapshot_host: Optional[str] = None,
        snapshot_port: Optional[int] = None,
        requires_data_dir: bool = True,
        beacon_id: Optional[str] = None,
        label: Optional[str] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.server_id = server_id
        self.label = label or server_id
        self.cmd = list(cmd)
        self.cwd = cwd
        self.env_extra = dict(env_extra or {})
        self.graceful_stop_timeout_s = float(graceful_stop_timeout_s)
        self.restart_on_crash = bool(restart_on_crash)
        self.snapshot_host = snapshot_host
        self.snapshot_port = snapshot_port
        self.requires_data_dir = bool(requires_data_dir)
        # Discovery beacon id advertised by the server (for EXTERNAL detection).
        self.beacon_id = beacon_id
        # Zero-arg callable that asks the running server to shut itself down
        # over its own protocol; returns truthy on success.  Set by the
        # dashboard once the server's client exists.  ``None`` -> hard kill.
        self.shutdown_request: Optional[Callable[[], bool]] = None

        self._state = SupervisorState.IDLE
        self._proc: Optional[QProcess] = None
        self._restart_history: list[float] = []
        self._stop_requested = False
        self._restart_pending = False
        self._precheck_in_flight = False
        self._line_buffer: dict[str, str] = {"stdout": "", "stderr": ""}
        self._last_data_dir_fail: Optional[str] = None
        self._restart_suppressed = False
        self._graceful_sent_at: Optional[float] = None
        self._precheck_finished.connect(self._on_precheck_finished)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> SupervisorState:
        return self._state

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def is_running(self) -> bool:
        """Alias for :meth:`is_alive` used by the dashboard close dialog."""
        return self.is_alive()

    def pid(self) -> int:
        try:
            return int(self._proc.processId()) if self._proc is not None else 0
        except Exception:
            return 0

    def check_external(self) -> bool:
        """Probe for an instance we did not start.

        Checks the discovery beacon registry for ``beacon_id`` (cache only,
        non-blocking) and, as a legacy fallback, the snapshot port.  If
        another instance is found the supervisor enters EXTERNAL and returns
        True.
        """
        if self.is_alive():
            return False
        if self.beacon_id and _beacon_seen(self.beacon_id):
            _LOG.info(
                "%s: beacon '%s' already on the subnet, marking EXTERNAL",
                self.server_id, self.beacon_id,
            )
            self._set_state(SupervisorState.EXTERNAL)
            return True
        if self.snapshot_port and _is_port_in_use(self.snapshot_host or "127.0.0.1", self.snapshot_port):
            _LOG.info(
                "%s: port %s:%s already bound, marking EXTERNAL",
                self.server_id, self.snapshot_host, self.snapshot_port,
            )
            self._set_state(SupervisorState.EXTERNAL)
            return True
        return False

    # Backwards-compatible name.
    check_port_external = check_external

    def clear_external(self) -> None:
        """Leave EXTERNAL (e.g. the foreign instance went away) back to IDLE."""
        if self._state == SupervisorState.EXTERNAL:
            self._set_state(SupervisorState.IDLE)

    def start(self) -> None:
        """Start the supervised subprocess (no-op if already running).

        The shared-data-dir precheck runs on a worker thread; the process is
        spawned from the main thread once it passes.  State is STARTING for
        the whole span so the header LED turns yellow immediately.
        """
        if self.is_alive() or self._precheck_in_flight:
            return
        if self._state == SupervisorState.EXTERNAL:
            # Re-probe: the external instance may have gone away.
            if self.check_external():
                _LOG.info("%s: refusing to start - another instance is running", self.server_id)
                return
            self._set_state(SupervisorState.IDLE)
        if self._state == SupervisorState.FAILED:
            _LOG.warning(
                "%s: in FAILED state; manual reset required (call reset_and_start())",
                self.server_id,
            )
            return
        if self.check_external():
            return

        self._stop_requested = False
        self._set_state(SupervisorState.STARTING)
        if not self.requires_data_dir:
            self._spawn()
            return
        self._precheck_in_flight = True
        QThreadPool.globalInstance().start(
            _PrecheckTask(self._precheck_data_dir, self._precheck_finished.emit)
        )

    def _precheck_data_dir(self) -> tuple[bool, str]:
        """Worker-thread body: (ok, message).  Never raises."""
        try:
            from waxx.util.dashboard import data_dir_guard  # noqa: PLC0415
        except Exception:
            return True, ""
        if not data_dir_guard.is_configured():
            return True, ""
        status = data_dir_guard.ensure_data_dir(log=_LOG)
        if status.ok:
            return True, ""
        if status.reason == "bat_missing":
            msg = (f"DATA_DIR unreachable; map-network-drives bat not found "
                   f"at {status.bat_path!s} — cannot start")
        elif status.reason == "remap_failed":
            msg = f"DATA_DIR still missing after running {status.bat_path!s} — cannot start"
        elif status.reason == "data_dir_unset":
            msg = "DATA_DIR is not configured — cannot start"
        else:
            msg = f"DATA_DIR unreachable ({status.reason}) — cannot start"
        return False, msg

    def _on_precheck_finished(self, ok: bool, msg: str) -> None:
        self._precheck_in_flight = False
        if self._stop_requested:
            # Stop clicked while the precheck ran.
            self._set_state(SupervisorState.IDLE)
            return
        if ok:
            self._last_data_dir_fail = None
            self._spawn()
            return
        # Throttle identical failure messages so repeated Start clicks don't
        # spam the log; stay IDLE (not FAILED) so a later click retries.
        if self._last_data_dir_fail != msg:
            self._last_data_dir_fail = msg
            _LOG.error("%s: %s", self.server_id, msg)
            try:
                self.log_line.emit(f"[ERR] {msg}")
            except Exception:
                pass
        self._set_state(SupervisorState.IDLE)

    def reset_and_start(self) -> None:
        """Clear failure state and try again from scratch."""
        self._restart_history.clear()
        if self._state in (SupervisorState.FAILED, SupervisorState.CRASHED):
            self._set_state(SupervisorState.IDLE)
        self.start()

    def stop(self, *, blocking: bool = False) -> None:
        """Stop the subprocess: graceful request first, kill after the timeout.

        If *blocking* is True the call waits up to ``graceful_stop_timeout_s``
        for the child to exit before killing it and returning.  Idempotent.
        Termination is scoped strictly to this server's own PID tree; console
        signals are never used because they fan out to every process on the
        dashboard's console.
        """
        self._stop_requested = True
        self._restart_pending = False
        if self._proc is None or not self.is_alive():
            if not self._precheck_in_flight:
                self._set_state(SupervisorState.IDLE)
            return
        self._set_state(SupervisorState.STOPPING)
        if not callable(self.shutdown_request):
            # No protocol-level shutdown for this one: nothing to wait for.
            self.force_kill(wait_ms=1000 if blocking else 200)
            return
        self._send_graceful_shutdown()
        if blocking:
            grace_ms = int(self.graceful_stop_timeout_s * 1000)
            if not self._proc.waitForFinished(grace_ms):
                _LOG.info("%s: graceful shutdown not honored in %d ms, killing",
                          self.server_id, grace_ms)
                self.force_kill(wait_ms=1000)
        else:
            QTimer.singleShot(int(self.graceful_stop_timeout_s * 1000), self._force_kill_if_alive)

    # ------------------------------------------------------------------ #
    # Fast shutdown helpers used by the dashboard close path.
    # ------------------------------------------------------------------ #

    def request_terminate(self) -> None:
        """Ask the child to exit without waiting.  Idempotent.

        Sends the graceful shutdown request (if a client is wired) and marks
        STOPPING.  The caller collects survivors after a grace window and
        kills them in one :func:`_kill_pid_trees` call.
        """
        self._stop_requested = True
        self._restart_pending = False
        if self._proc is None or not self.is_alive():
            if not self._precheck_in_flight:
                self._set_state(SupervisorState.IDLE)
            return
        self._set_state(SupervisorState.STOPPING)
        self._send_graceful_shutdown()

    def _send_graceful_shutdown(self) -> None:
        """Fire ``shutdown_request`` on a daemon thread (it does network I/O)."""
        fn = self.shutdown_request
        if not callable(fn) or self._graceful_sent_at is not None:
            return
        self._graceful_sent_at = time.monotonic()

        def _run():
            try:
                ok = bool(fn())
                _LOG.info("%s: graceful shutdown request %s", self.server_id,
                          "accepted" if ok else "not accepted")
            except Exception as exc:  # noqa: BLE001
                _LOG.info("%s: graceful shutdown request failed: %r", self.server_id, exc)

        threading.Thread(target=_run, name=f"shutdown-{self.server_id}", daemon=True).start()

    def force_kill(self, *, wait_ms: int = 500) -> None:
        """Hard-kill the child tree if it's still alive, then wait briefly."""
        if self._proc is None or not self.is_alive():
            return
        if not _kill_pid_tree(self.pid()):
            try:
                self._proc.kill()
            except Exception as exc:
                _LOG.error("%s: kill() raised: %r", self.server_id, exc)
        try:
            self._proc.waitForFinished(wait_ms)
        except Exception:
            pass

    def wait_for_finished(self, timeout_ms: int) -> bool:
        """Wait for the subprocess to exit; ``True`` if it did."""
        if self._proc is None or not self.is_alive():
            return True
        try:
            return bool(self._proc.waitForFinished(int(timeout_ms)))
        except Exception:
            return not self.is_alive()

    def suppress_restart(self) -> None:
        """Disable crash-restart for the remainder of this supervisor's life."""
        self._restart_suppressed = True

    def restart(self) -> None:
        """Stop the subprocess, wait for it to fully exit, then start again.

        Deterministic and synchronous: the blocking stop waits for the
        targeted process tree to die before respawning, so a restart can
        never race a still-running instance or leave two copies contending
        for the same port / COM device.
        """
        self._restart_pending = False
        if self._proc is not None and self.is_alive():
            self.stop(blocking=True)
            try:
                self._proc.waitForFinished(2000)
            except Exception:
                pass
        self.start()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn(self) -> None:
        proc = QProcess(self)
        if self.cwd:
            proc.setWorkingDirectory(self.cwd)
        if self.env_extra:
            env = QProcessEnvironment.systemEnvironment()
            for k, v in self.env_extra.items():
                env.insert(k, v)
            proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        if _IS_WINDOWS:
            try:
                def _add_flags(args):  # noqa: ANN001 - Qt callback type
                    args.flags |= _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW
                proc.setCreateProcessArgumentsModifier(_add_flags)
            except Exception as exc:  # pragma: no cover - PyQt API guard
                _LOG.debug("%s: setCreateProcessArgumentsModifier unavailable: %r",
                           self.server_id, exc)
        proc.readyReadStandardOutput.connect(self._drain_stdout)
        proc.readyReadStandardError.connect(self._drain_stderr)
        proc.errorOccurred.connect(self._on_error)
        proc.finished.connect(self._on_finished)
        proc.started.connect(self._on_started)

        self._proc = proc
        self._graceful_sent_at = None
        self._restart_history.append(time.monotonic())

        program = self.cmd[0]
        args = self.cmd[1:]
        _LOG.info("%s: spawning program=%s args=%s cwd=%s",
                  self.server_id, program, args, self.cwd or "<inherited>")
        self._set_state(SupervisorState.STARTING)
        proc.start(program, args)

    def _force_kill_if_alive(self) -> None:
        if self._proc is not None and self.is_alive():
            _LOG.warning("%s: graceful stop timed out, killing", self.server_id)
            self.force_kill(wait_ms=200)

    def _on_started(self) -> None:
        _LOG.info("%s: subprocess started pid=%s", self.server_id, self.pid())
        self._set_state(SupervisorState.RUNNING)

    def _on_error(self, err: QProcess.ProcessError) -> None:
        _LOG.error("%s: QProcess errorOccurred: %r", self.server_id, err)
        if err == QProcess.ProcessError.FailedToStart:
            self.log_line.emit(f"[ERR] failed to start: {self.cmd}")
            self._set_state(SupervisorState.CRASHED)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        _LOG.info("%s: subprocess exited code=%d status=%s",
                  self.server_id, exit_code, exit_status.name)
        self._drain_stdout()
        self._drain_stderr()
        self._graceful_sent_at = None

        crashed = exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0
        if crashed and not self._stop_requested:
            self._set_state(SupervisorState.CRASHED)
            self.crashed.emit(exit_code)
            if self.restart_on_crash:
                self._maybe_auto_restart()
            return

        if self._restart_pending:
            self._restart_pending = False
            self._set_state(SupervisorState.IDLE)
            QTimer.singleShot(200, self.start)
            return

        self._set_state(SupervisorState.IDLE)

    def _maybe_auto_restart(self) -> None:
        if self._restart_suppressed:
            _LOG.info("%s: auto-restart suppressed (shutdown in progress)", self.server_id)
            return
        cutoff = time.monotonic() - self.RESTART_WINDOW_S
        self._restart_history = [t for t in self._restart_history if t >= cutoff]
        if len(self._restart_history) >= self.MAX_RESTARTS:
            _LOG.error("%s: %d crashes within %.0fs - giving up, state -> FAILED",
                       self.server_id, len(self._restart_history), self.RESTART_WINDOW_S)
            self._set_state(SupervisorState.FAILED)
            return
        delay = min(
            self.INITIAL_RESTART_DELAY_S * (2 ** (len(self._restart_history) - 1)),
            self.MAX_RESTART_DELAY_S,
        )
        _LOG.warning("%s: auto-restart in %.1fs (history=%d)",
                     self.server_id, delay, len(self._restart_history))
        QTimer.singleShot(int(delay * 1000), self.start)

    def _drain_stdout(self) -> None:
        if self._proc is None:
            return
        self._emit_lines("stdout", bytes(self._proc.readAllStandardOutput()))

    def _drain_stderr(self) -> None:
        if self._proc is None:
            return
        self._emit_lines("stderr", bytes(self._proc.readAllStandardError()))

    def _emit_lines(self, stream: str, data: bytes) -> None:
        if not data:
            return
        text = self._line_buffer[stream] + data.decode("utf-8", errors="replace")
        *lines, tail = text.split("\n")
        self._line_buffer[stream] = tail
        tag = "ERR" if stream == "stderr" else "OUT"
        for line in lines:
            self.log_line.emit(f"[{tag}] {line.rstrip(chr(13))}")

    def _set_state(self, new: SupervisorState) -> None:
        if new == self._state:
            return
        _LOG.debug("%s: state %s -> %s", self.server_id, self._state.name, new.name)
        self._state = new
        self.state_changed.emit(new)


def shutdown_all(supervisors: Iterable["ServerSupervisor"], *, grace_ms: int = 1500) -> list[str]:
    """Stop every supervisor: graceful requests fan out, one collective wait,
    then a single ``taskkill`` for the survivors.

    Returns the ids that had to be hard-killed.  Used by the dashboard close
    path; bounded to about ``grace_ms`` plus one taskkill regardless of how
    many servers there are.
    """
    sups = [s for s in supervisors if s is not None]
    for sup in sups:
        try:
            sup.suppress_restart()
            sup.request_terminate()
        except Exception:
            _LOG.exception("request_terminate() failed for id=%s", getattr(sup, "server_id", "?"))

    # Supervisors without a protocol-level shutdown cannot exit on request;
    # kill their trees right away instead of making everyone wait the grace
    # window for nothing.
    immediate = [s for s in sups if s.is_alive() and not callable(s.shutdown_request)]
    if immediate:
        if not _kill_pid_trees([s.pid() for s in immediate]):
            for s in immediate:
                s.force_kill(wait_ms=200)

    from PyQt6.QtCore import QDeadlineTimer  # noqa: PLC0415
    deadline = QDeadlineTimer(int(grace_ms))
    for sup in sups:
        if sup in immediate:
            continue
        remaining = deadline.remainingTime()
        if remaining <= 0:
            break
        try:
            sup.wait_for_finished(remaining)
        except Exception:
            pass

    survivors = [s for s in sups if s.is_alive()]
    killed: list[str] = []
    if survivors:
        pids = [s.pid() for s in survivors]
        _LOG.warning("hard-killing %d server(s) that did not exit in %d ms: %s",
                     len(survivors), grace_ms, [s.server_id for s in survivors])
        if not _kill_pid_trees(pids):
            for s in survivors:
                s.force_kill(wait_ms=200)
        for s in survivors:
            try:
                s.wait_for_finished(300)
            except Exception:
                pass
            killed.append(s.server_id)
    return killed


__all__ = ["ServerSupervisor", "SupervisorState", "shutdown_all",
           "install_console_signal_guard"]
