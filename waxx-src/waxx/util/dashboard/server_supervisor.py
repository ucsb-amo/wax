"""Subprocess supervisor for server panels.

Wraps :class:`QProcess` with:

* state machine reported via Qt signals (IDLE -> STARTING -> RUNNING -> STOPPING -> IDLE / CRASHED / FAILED / EXTERNAL)
* graceful stop: terminate -> wait -> kill, with configurable timeout
* bounded exponential restart on crash (5 attempts within 60 s, then FAILED)
* port-already-bound detection so a dashboard never double-starts a server
  that the lab's tray launcher already brought up (panel enters EXTERNAL)
* always-drained stdout/stderr (a UI that pauses must never block the child)
* atexit cleanup so closing the dashboard never orphans children

The supervisor is the only piece of the framework that touches the OS process
table, so all of the "don't leak processes" logic lives here.
"""

from __future__ import annotations

import atexit
import enum
import logging
import socket
import time
import weakref
from typing import Optional

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal


_LOG = logging.getLogger("waxx.dashboard.supervisor")


class SupervisorState(enum.Enum):
    IDLE = "IDLE"             # not started, no pending action
    STARTING = "STARTING"     # QProcess.start() invoked, not yet RUNNING
    RUNNING = "RUNNING"       # subprocess alive
    STOPPING = "STOPPING"     # terminate() sent, waiting for exit
    CRASHED = "CRASHED"       # exited non-zero; eligible for restart if enabled
    FAILED = "FAILED"         # too many restart attempts in window
    EXTERNAL = "EXTERNAL"     # port already bound by another process


_ALL_SUPERVISORS: "weakref.WeakSet[ServerSupervisor]" = weakref.WeakSet()


def _atexit_kill_all() -> None:
    """Make sure no supervised child outlives the dashboard."""
    for sup in list(_ALL_SUPERVISORS):
        try:
            sup.stop(blocking=True)
        except Exception as exc:  # pragma: no cover
            _LOG.warning("atexit cleanup failed for %s: %r", getattr(sup, "server_id", "?"), exc)


atexit.register(_atexit_kill_all)


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
        graceful_stop_timeout_s: float = 5.0,
        restart_on_crash: bool = False,
        snapshot_host: Optional[str] = None,
        snapshot_port: Optional[int] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.server_id = server_id
        self.cmd = list(cmd)
        self.cwd = cwd
        self.env_extra = dict(env_extra or {})
        self.graceful_stop_timeout_s = float(graceful_stop_timeout_s)
        self.restart_on_crash = bool(restart_on_crash)
        self.snapshot_host = snapshot_host
        self.snapshot_port = snapshot_port

        self._state = SupervisorState.IDLE
        self._proc: Optional[QProcess] = None
        self._restart_history: list[float] = []
        self._stop_requested = False
        self._line_buffer: dict[str, str] = {"stdout": "", "stderr": ""}

        _ALL_SUPERVISORS.add(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> SupervisorState:
        return self._state

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def check_port_external(self) -> bool:
        """Probe the snapshot port; if bound, transition to EXTERNAL.

        Returns True if the port is bound (panel should not autostart).
        """
        if not self.snapshot_port:
            return False
        if _is_port_in_use(self.snapshot_host or "127.0.0.1", self.snapshot_port):
            _LOG.info(
                "%s: port %s:%s already bound, marking EXTERNAL",
                self.server_id, self.snapshot_host, self.snapshot_port,
            )
            self._set_state(SupervisorState.EXTERNAL)
            return True
        return False

    def start(self) -> None:
        """Start the supervised subprocess (no-op if already running)."""
        if self.is_alive():
            return
        if self._state == SupervisorState.EXTERNAL:
            _LOG.info("%s: refusing to start - port externally bound", self.server_id)
            return
        if self._state == SupervisorState.FAILED:
            _LOG.warning(
                "%s: in FAILED state; manual reset required (call reset_and_start())",
                self.server_id,
            )
            return
        if self.check_port_external():
            return

        self._stop_requested = False
        self._spawn()

    def reset_and_start(self) -> None:
        """Clear failure state and try again from scratch."""
        self._restart_history.clear()
        if self._state in (SupervisorState.FAILED, SupervisorState.CRASHED):
            self._set_state(SupervisorState.IDLE)
        self.start()

    def stop(self, *, blocking: bool = False) -> None:
        """Stop the subprocess gracefully.

        If *blocking* is True the call waits up to ``graceful_stop_timeout_s``
        for the child to exit before returning.  Always idempotent.
        """
        self._stop_requested = True
        if self._proc is None or not self.is_alive():
            self._set_state(SupervisorState.IDLE)
            return
        self._set_state(SupervisorState.STOPPING)
        try:
            self._proc.terminate()
        except Exception as exc:
            _LOG.warning("%s: terminate() raised: %r", self.server_id, exc)
        if blocking:
            timeout_ms = int(self.graceful_stop_timeout_s * 1000)
            if not self._proc.waitForFinished(timeout_ms):
                _LOG.warning("%s: terminate() timed out, killing", self.server_id)
                try:
                    self._proc.kill()
                    self._proc.waitForFinished(1000)
                except Exception as exc:
                    _LOG.error("%s: kill() raised: %r", self.server_id, exc)
        else:
            # Schedule a kill if terminate didn't take effect in time.
            QTimer.singleShot(
                int(self.graceful_stop_timeout_s * 1000),
                self._force_kill_if_alive,
            )

    def restart(self) -> None:
        """Stop and then start again."""
        self.stop()
        # Schedule a start after the current stop completes.  Re-checked in
        # _on_finished so we don't double-spawn.
        self._restart_pending = True  # type: ignore[attr-defined]

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
        # We want stdout and stderr separated for log-level tagging.
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        proc.readyReadStandardOutput.connect(self._drain_stdout)
        proc.readyReadStandardError.connect(self._drain_stderr)
        proc.errorOccurred.connect(self._on_error)
        proc.finished.connect(self._on_finished)
        proc.started.connect(self._on_started)

        self._proc = proc
        self._restart_history.append(time.monotonic())

        program = self.cmd[0]
        args = self.cmd[1:]
        _LOG.info(
            "%s: spawning program=%s args=%s cwd=%s",
            self.server_id, program, args, self.cwd or "<inherited>",
        )
        self._set_state(SupervisorState.STARTING)
        proc.start(program, args)

    def _force_kill_if_alive(self) -> None:
        if self._proc is not None and self.is_alive():
            _LOG.warning("%s: graceful stop timed out, killing", self.server_id)
            try:
                self._proc.kill()
            except Exception as exc:
                _LOG.error("%s: kill() raised: %r", self.server_id, exc)

    def _on_started(self) -> None:
        _LOG.info("%s: subprocess started pid=%s", self.server_id, self._proc.processId() if self._proc else "?")
        self._set_state(SupervisorState.RUNNING)

    def _on_error(self, err: QProcess.ProcessError) -> None:
        _LOG.error("%s: QProcess errorOccurred: %r", self.server_id, err)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        _LOG.info(
            "%s: subprocess exited code=%d status=%s",
            self.server_id, exit_code, exit_status.name,
        )
        # Flush any tail data sitting in the pipe.
        self._drain_stdout()
        self._drain_stderr()

        crashed = exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0
        if crashed and not self._stop_requested:
            self._set_state(SupervisorState.CRASHED)
            self.crashed.emit(exit_code)
            if self.restart_on_crash:
                self._maybe_auto_restart()
            return

        if getattr(self, "_restart_pending", False):
            self._restart_pending = False  # type: ignore[attr-defined]
            QTimer.singleShot(200, self.start)
            self._set_state(SupervisorState.IDLE)
            return

        self._set_state(SupervisorState.IDLE)

    def _maybe_auto_restart(self) -> None:
        # Trim history to the rolling window.
        cutoff = time.monotonic() - self.RESTART_WINDOW_S
        self._restart_history = [t for t in self._restart_history if t >= cutoff]
        if len(self._restart_history) >= self.MAX_RESTARTS:
            _LOG.error(
                "%s: %d crashes within %.0fs - giving up, state -> FAILED",
                self.server_id, len(self._restart_history), self.RESTART_WINDOW_S,
            )
            self._set_state(SupervisorState.FAILED)
            return
        delay = min(
            self.INITIAL_RESTART_DELAY_S * (2 ** (len(self._restart_history) - 1)),
            self.MAX_RESTART_DELAY_S,
        )
        _LOG.warning(
            "%s: auto-restart in %.1fs (history=%d)",
            self.server_id, delay, len(self._restart_history),
        )
        QTimer.singleShot(int(delay * 1000), self.start)

    def _drain_stdout(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput())
        self._emit_lines("stdout", data)

    def _drain_stderr(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardError())
        self._emit_lines("stderr", data)

    def _emit_lines(self, stream: str, data: bytes) -> None:
        if not data:
            return
        text = self._line_buffer[stream] + data.decode("utf-8", errors="replace")
        *lines, tail = text.split("\n")
        self._line_buffer[stream] = tail
        for line in lines:
            stripped = line.rstrip("\r")
            tag = "ERR" if stream == "stderr" else "OUT"
            self.log_line.emit(f"[{tag}] {stripped}")

    def _set_state(self, new: SupervisorState) -> None:
        if new == self._state:
            return
        _LOG.debug("%s: state %s -> %s", self.server_id, self._state.name, new.name)
        self._state = new
        self.state_changed.emit(new)


__all__ = ["ServerSupervisor", "SupervisorState"]
