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
import sys
import time
import weakref
from typing import Optional

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal


_LOG = logging.getLogger("waxx.dashboard.supervisor")

_IS_WINDOWS = sys.platform.startswith("win")
# Windows CreateProcess flags.
# CREATE_NEW_PROCESS_GROUP — child gets its own process group so we can
# send it CTRL_BREAK_EVENT for cooperative shutdown without affecting the
# dashboard's own console.
# CREATE_NO_WINDOW — prevents the headless Python child from inheriting (or
# allocating) the dashboard's console.  Without this, QProcess.terminate()
# posts WM_CLOSE to the shared console window, which terminates every
# process attached to that console — including the dashboard itself.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000


def _send_ctrl_break(pid: int) -> bool:
    """Send ``CTRL_BREAK_EVENT`` to a Windows process group.

    Returns True on success.  No-op (returns False) off Windows.
    """
    if not _IS_WINDOWS or pid <= 0:
        return False
    try:
        import ctypes  # noqa: PLC0415
        CTRL_BREAK_EVENT = 1
        kernel32 = ctypes.windll.kernel32
        ok = bool(kernel32.GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, int(pid)))
        if not ok:
            _LOG.debug("GenerateConsoleCtrlEvent failed err=%s", ctypes.get_last_error())
        return ok
    except Exception as exc:  # pragma: no cover
        _LOG.debug("_send_ctrl_break(%s) raised: %r", pid, exc)
        return False


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
    """Make sure no supervised child outlives the dashboard.

    Fast path: fan out terminate, short collective wait, then kill any
    survivors.  Same shape as ``DashboardMainWindow.closeEvent`` to keep
    shutdown bounded to ~1 s even with many supervisors.
    """
    sups = list(_ALL_SUPERVISORS)
    for sup in sups:
        try:
            sup.request_terminate()
        except Exception as exc:  # pragma: no cover
            _LOG.warning("atexit terminate failed for %s: %r", getattr(sup, "server_id", "?"), exc)
    # Short collective grace window across all supervisors.
    for sup in sups:
        try:
            sup.wait_for_finished(200)
        except Exception:  # pragma: no cover
            pass
    for sup in sups:
        try:
            sup.force_kill(wait_ms=200)
        except Exception as exc:  # pragma: no cover
            _LOG.warning("atexit kill failed for %s: %r", getattr(sup, "server_id", "?"), exc)


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
        requires_data_dir: bool = True,
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
        self.requires_data_dir = bool(requires_data_dir)

        self._state = SupervisorState.IDLE
        self._proc: Optional[QProcess] = None
        self._restart_history: list[float] = []
        self._stop_requested = False
        self._line_buffer: dict[str, str] = {"stdout": "", "stderr": ""}
        # Throttle: last (reason, bat_path) tuple emitted for a data-dir
        # pre-spawn failure.  Suppresses log spam on repeated Start clicks
        # until the underlying reason changes.
        self._last_data_dir_fail: Optional[tuple[str, Optional[str]]] = None
        # Set by ``suppress_restart()`` during dashboard close so the
        # crash-restart loop doesn't fight a clean shutdown.
        self._restart_suppressed = False

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

        if not self._precheck_data_dir():
            return

        self._stop_requested = False
        self._spawn()

    def _precheck_data_dir(self) -> bool:
        """Return False (and emit one log line) if DATA_DIR is unreachable.

        Servers marked ``requires_data_dir=False`` always pass.  Failures
        are throttled per ``(reason, bat_path)`` tuple so repeated Start
        clicks don't spam the dashboard log.
        """
        if not self.requires_data_dir:
            return True
        try:
            from waxx.util.dashboard import data_dir_guard  # noqa: PLC0415
        except Exception:
            return True
        if not data_dir_guard.is_configured():
            return True
        status = data_dir_guard.ensure_data_dir(log=_LOG)
        if status.ok:
            self._last_data_dir_fail = None
            return True
        key = (status.reason, status.bat_path)
        if self._last_data_dir_fail != key:
            self._last_data_dir_fail = key
            if status.reason == "bat_missing":
                msg = (
                    f"DATA_DIR unreachable; map-network-drives bat not found "
                    f"at {status.bat_path!s} — cannot start"
                )
            elif status.reason == "remap_failed":
                msg = (
                    f"DATA_DIR still missing after running {status.bat_path!s} "
                    f"— cannot start"
                )
            elif status.reason == "data_dir_unset":
                msg = "DATA_DIR is not configured — cannot start"
            else:
                msg = f"DATA_DIR unreachable ({status.reason}) — cannot start"
            _LOG.error("%s: %s", self.server_id, msg)
            try:
                self.log_line.emit(f"[ERR] {msg}")
            except Exception:
                pass
        # Stay in IDLE rather than FAILED: FAILED is sticky and would force
        # the user to call reset_and_start().  We want subsequent Start
        # clicks to retry the precheck (cheap; throttle prevents log spam),
        # so re-mapping the share and clicking Start "just works".
        self._set_state(SupervisorState.IDLE)
        return False

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

        On Windows the preferred stop mechanism is ``CTRL_BREAK_EVENT``
        sent to the child's process group (requires ``CREATE_NEW_PROCESS_GROUP``
        to have been applied at spawn time).  If that path is unavailable
        we fall back to ``kill()`` (``TerminateProcess``) rather than
        ``terminate()`` because ``terminate()`` posts ``WM_CLOSE`` which
        can propagate to a shared console window and kill the dashboard.
        """
        self._stop_requested = True
        if self._proc is None or not self.is_alive():
            self._set_state(SupervisorState.IDLE)
            return
        self._set_state(SupervisorState.STOPPING)
        sent_break = False
        if _IS_WINDOWS:
            try:
                pid = int(self._proc.processId())
            except Exception:
                pid = 0
            sent_break = _send_ctrl_break(pid)
        if not sent_break:
            if _IS_WINDOWS:
                # On Windows, QProcess.terminate() posts WM_CLOSE to the
                # process's console window.  If CREATE_NO_WINDOW was not
                # applied (e.g. setCreateProcessArgumentsModifier silently
                # failed in this PyQt6 build), the child shares the
                # dashboard's console and WM_CLOSE would kill the whole
                # dashboard.  Use kill() (TerminateProcess) instead — it is
                # process-local and never affects the caller.
                _LOG.info("%s: CTRL_BREAK unavailable, using kill()", self.server_id)
                try:
                    self._proc.kill()
                except Exception as exc:
                    _LOG.warning("%s: kill() raised: %r", self.server_id, exc)
            else:
                try:
                    self._proc.terminate()
                except Exception as exc:
                    _LOG.warning("%s: terminate() raised: %r", self.server_id, exc)
        if blocking:
            # Short grace window for cooperative shutdown, then kill.
            # The previous 5 s timeout multiplied by N supervisors froze
            # the dashboard for ~30 s when the user closed the window.
            grace_ms = min(int(self.graceful_stop_timeout_s * 1000), 500)
            if not self._proc.waitForFinished(grace_ms):
                _LOG.info("%s: terminate() not honored, killing", self.server_id)
                try:
                    self._proc.kill()
                    self._proc.waitForFinished(1000)
                except Exception as exc:
                    _LOG.error("%s: kill() raised: %r", self.server_id, exc)
        else:
            # Schedule a kill if terminate didn't take effect in time.
            QTimer.singleShot(
                500,
                self._force_kill_if_alive,
            )

    # ------------------------------------------------------------------ #
    # Fast shutdown helpers used by the dashboard close path so the GUI
    # thread doesn't block for ``N_supervisors * graceful_stop_timeout_s``
    # seconds when the user closes the window.
    # ------------------------------------------------------------------ #

    def request_terminate(self) -> None:
        """Ask the child to exit cleanly without waiting.  Idempotent.

        On Windows we send ``CTRL_BREAK_EVENT`` to the child's process
        group (the child was spawned with ``CREATE_NEW_PROCESS_GROUP``);
        Python turns that into ``SIGBREAK`` → ``KeyboardInterrupt``, so a
        normal server loop unwinds with a 0 exit code instead of being
        hard-killed.  If CTRL_BREAK is unavailable we fall back to
        ``kill()`` rather than ``terminate()`` to avoid ``WM_CLOSE``
        reaching a shared console window.
        """
        self._stop_requested = True
        if self._proc is None or not self.is_alive():
            self._set_state(SupervisorState.IDLE)
            return
        self._set_state(SupervisorState.STOPPING)
        sent_break = False
        if _IS_WINDOWS:
            try:
                pid = int(self._proc.processId())
            except Exception:
                pid = 0
            sent_break = _send_ctrl_break(pid)
        if not sent_break:
            if _IS_WINDOWS:
                # Same rationale as in stop(): avoid terminate() on Windows
                # to prevent WM_CLOSE reaching a shared console window.
                _LOG.info("%s: CTRL_BREAK unavailable, using kill()", self.server_id)
                try:
                    self._proc.kill()
                except Exception as exc:
                    _LOG.warning("%s: kill() raised: %r", self.server_id, exc)
            else:
                try:
                    self._proc.terminate()
                except Exception as exc:
                    _LOG.warning("%s: terminate() raised: %r", self.server_id, exc)

    def force_kill(self, *, wait_ms: int = 500) -> None:
        """Hard-kill the child if it's still alive, then wait briefly."""
        if self._proc is None or not self.is_alive():
            return
        try:
            self._proc.kill()
            self._proc.waitForFinished(wait_ms)
        except Exception as exc:
            _LOG.error("%s: kill() raised: %r", self.server_id, exc)

    def wait_for_finished(self, timeout_ms: int) -> bool:
        """Wait for the subprocess to exit; ``True`` if it did."""
        if self._proc is None or not self.is_alive():
            return True
        try:
            return bool(self._proc.waitForFinished(int(timeout_ms)))
        except Exception:
            return not self.is_alive()

    def suppress_restart(self) -> None:
        """Disable crash-restart for the remainder of this supervisor's life.

        Called by the dashboard's close path so a clean ``CTRL_BREAK``
        shutdown isn't fought by the restart-on-crash loop.
        """
        self._restart_suppressed = True

    def is_running(self) -> bool:
        """Alias for :meth:`is_alive` used by the dashboard close dialog."""
        return self.is_alive()

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
        # On Windows put the child in its own process group so the
        # supervisor can send it CTRL_BREAK_EVENT for cooperative
        # shutdown without also signalling the dashboard's console.
        if _IS_WINDOWS:
            try:
                def _add_new_group(args):  # noqa: ANN001 - Qt callback type
                    args.flags |= _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW
                proc.setCreateProcessArgumentsModifier(_add_new_group)
            except Exception as exc:  # pragma: no cover - PyQt API guard
                _LOG.debug(
                    "%s: setCreateProcessArgumentsModifier unavailable: %r",
                    self.server_id, exc,
                )
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
        if self._restart_suppressed:
            _LOG.info("%s: auto-restart suppressed (shutdown in progress)", self.server_id)
            return
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
