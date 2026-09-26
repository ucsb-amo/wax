"""Reset the device state by running one experiment file (kexp: mot_observe.py).

The device state becomes *untrusted* when a run takes the core and never
reports its end state (see ``MonitorUDPServer``).  Restarting the monitor does
not fix that: the monitor writes nothing to the hardware when it starts.  A
reset experiment does.  It sets the hardware itself, and its ``end()`` then
reports its end state through ``replace_state`` like any run's -- which is
what marks the state trusted again.

:class:`StateReset` launches that file on request (the Device Control GUI's
Run <reset experiment> button, through the monitor server), the same way the monitor
experiment is launched (``%kpy% & ar <file>``), and follows it to the end.  It
never marks anything trusted itself: a reset that fails, is killed, or exits
without reporting its end state leaves the state untrusted and says why.
"""

from __future__ import annotations

import ast
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

from waxx.util.device_state.monitor_manager import (
    _diagnose, ar_command, environment_report)

log = logging.getLogger(__name__)

#: A reset still running after this long is killed.  It normally takes well
#: under a minute (imports, one compile, one short kernel).
RESET_TIMEOUT_S = 180.0

_TAIL_LINES = 25
#: Lines of a failed reset's output handed to the GUIs (the server log has
#: the full tail).
_TAIL_SHOWN = 8


def describe_expt(path) -> str:
    """The experiment file's module docstring, else its first class's -- shown
    by the GUI before a reset, so what the reset leaves on is written next to
    the code that does it.  Paragraphs are unwrapped (a dialog wraps them
    itself).  "" when there is none or the file cannot be read."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return ""
    doc = ast.get_docstring(tree)
    if not doc:
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                break
    paragraphs = (" ".join(p.split()) for p in (doc or "").split("\n\n"))
    return "\n\n".join(p for p in paragraphs if p)


def _spawn(command: str):
    # Merged, line-streamed output, exactly as MonitorManager runs the monitor.
    return Popen(command, stdout=PIPE, stderr=STDOUT, universal_newlines=True,
                 bufsize=1, errors="replace", shell=True)


def _kill_tree(proc) -> None:
    try:
        from waxx.util.dashboard.server_supervisor import _kill_pid_tree  # noqa: PLC0415
        if _kill_pid_tree(proc.pid):
            return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


class StateReset:
    """One reset experiment, run at most once at a time.

    ``info()`` is what GUIs are told: ``None`` when no reset experiment is
    configured, else ``{"expt", "about", "state", ...}`` with ``state`` one of
    ``"idle"``, ``"running"``, ``"done"``, ``"failed"`` and, once a reset has
    been started, who asked (``operator``, ``client``), ``started`` /
    ``ended`` (epoch s), ``text``, ``end_state`` (its end state arrived),
    ``exit_code`` and, on failure, ``tail`` (its last output lines).

    ``on_change(info)`` is called on every change, from whichever thread made
    it.  ``spawn`` / ``kill_tree`` / ``clock`` are for tests.
    """

    def __init__(self, expt_path=None, on_change=None, journal=None, spawn=None,
                 kill_tree=None, clock=time.time, timeout_s: float = RESET_TIMEOUT_S):
        self.expt_path = str(expt_path) if expt_path else None
        self.expt = Path(self.expt_path).stem if self.expt_path else ""
        self.timeout_s = float(timeout_s)
        self._on_change = on_change
        self._journal = journal
        self._spawn = spawn or _spawn
        self._kill_tree = kill_tree or _kill_tree
        self._clock = clock
        self._lock = threading.Lock()
        self._current: dict | None = None
        self._thread: threading.Thread | None = None
        self._about = ""
        self._about_mtime = None

    # -- state ------------------------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self.expt_path)

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._current) and self._current["state"] == "running"

    def running_info(self, expt: str | None = None) -> dict | None:
        """The running reset (a copy), or None -- also None when *expt* is
        given and is not the reset experiment's name."""
        with self._lock:
            cur = self._current
            if not cur or cur["state"] != "running":
                return None
            if expt is not None and str(expt) != self.expt:
                return None
            return dict(cur)

    def info(self) -> dict | None:
        if not self.configured:
            return None
        with self._lock:
            cur = dict(self._current) if self._current else {"state": "idle"}
        cur.update(expt=self.expt, about=self._read_about())
        return cur

    def _read_about(self) -> str:
        try:
            mtime = os.stat(self.expt_path).st_mtime_ns
        except (OSError, TypeError):
            return ""
        if mtime != self._about_mtime:
            self._about = describe_expt(self.expt_path)
            self._about_mtime = mtime
        return self._about

    def preflight_problems(self) -> list[str]:
        if not self.configured:
            return ["no reset experiment is configured on this monitor server"]
        path = Path(self.expt_path)
        if not path.exists():
            return [f"the reset experiment file does not exist: {path}"]
        if path.is_dir():
            return [f"the reset experiment path is a directory, not a file: {path}"]
        return []

    # -- start ------------------------------------------------------------------

    def start(self, operator: str = "", client: str = "") -> dict:
        """Launch the reset experiment; the reply the server sends back."""
        problems = self.preflight_problems()
        if problems:
            return self._refused(problems[0], operator, client)
        with self._lock:
            cur = self._current
            if cur and cur["state"] == "running":
                who = "@".join(p for p in (cur.get("operator"), cur.get("client")) if p) or "?"
                return self._refused(
                    f"a reset is already running ({self.expt}, started by {who} at "
                    f"{time.strftime('%H:%M:%S', time.localtime(cur['started']))})",
                    operator, client)
            self._current = {"state": "running", "operator": operator, "client": client,
                             "started": self._clock(), "ended": None, "end_state": False,
                             "exit_code": None, "text": f"{self.expt} started"}
        command = ar_command(self.expt_path)
        try:
            proc = self._spawn(command)
        except OSError as exc:
            log.error("State reset: could not spawn %r: %r", command, exc)
            for line in environment_report():
                log.error("  %s", line)
            self._finish(None, [], spawn_error=repr(exc))
            return {"status": "error", "msg": f"could not start {self.expt}: {exc!r}"}
        log.info("State reset: %s started (pid %s), requested by %s.", self.expt,
                 getattr(proc, "pid", "?"), "@".join(p for p in (operator, client) if p) or "?")
        self._record("state_reset_started", operator=operator, client=client)
        self._thread = threading.Thread(target=self._follow, args=(proc,), daemon=True,
                                        name="state-reset")
        self._thread.start()
        self._notify()
        return {"status": "ok", "reset": self.info()}

    def _refused(self, msg: str, operator: str, client: str) -> dict:
        log.warning("State reset refused: %s", msg)
        self._record("state_reset_refused", operator=operator, client=client, msg=msg)
        return {"status": "error", "msg": msg}

    # -- follow -----------------------------------------------------------------

    def _follow(self, proc) -> None:
        tail: deque[str] = deque(maxlen=_TAIL_LINES)
        timer = threading.Timer(self.timeout_s, self._on_timeout, args=(proc,))
        timer.daemon = True
        timer.start()
        code = None
        try:
            if proc.stdout is not None:
                for raw in proc.stdout:
                    line = raw.rstrip()
                    if line:
                        tail.append(line)
                        log.info("[reset] %s", line)
            code = proc.wait()
        except Exception as exc:
            tail.append(f"(following the reset failed: {exc!r})")
        finally:
            timer.cancel()
        self._finish(code, list(tail))

    def _on_timeout(self, proc) -> None:
        with self._lock:
            cur = self._current
            if not cur or cur["state"] != "running":
                return
            cur["timed_out"] = True
        log.error("State reset: %s still running after %.0f s -- killing it.",
                  self.expt, self.timeout_s)
        self._kill_tree(proc)

    def end_state_received(self, expt: str) -> bool:
        """The server accepted an end state from *expt*: if it is the running
        reset's, remember it (what makes the reset a success).  Returns
        whether it was."""
        with self._lock:
            cur = self._current
            if not cur or cur["state"] != "running" or str(expt) != self.expt:
                return False
            cur["end_state"] = True
            cur["text"] = f"{self.expt} reported its end state"
        self._notify()
        return True

    def _finish(self, code, tail: list[str], spawn_error: str = "") -> None:
        with self._lock:
            cur = self._current
            if cur is None:
                return
            end_state = bool(cur.get("end_state"))
            if end_state:
                state = "done"
                text = (f"{self.expt} ended; its end state replaced the state file and "
                        "marked it trusted")
                if code not in (0, None):
                    text += f" (it then exited with code {code})"
            else:
                state = "failed"
                if spawn_error:
                    text = f"{self.expt} could not be started: {spawn_error}"
                elif cur.get("timed_out"):
                    text = (f"{self.expt} did not finish within {self.timeout_s:.0f} s and "
                            "was killed; the hardware is wherever it stopped")
                else:
                    text = f"{self.expt} exited with code {code} without reporting its end state"
                    hints = _diagnose("\n".join(tail))
                    if hints:
                        text += f" -- likely cause: {hints[0]}"
                    elif tail:
                        text += f" -- last line: {tail[-1]}"
                text += ". The device state is still untrusted"
            cur.update(state=state, ended=self._clock(), exit_code=code, text=text)
            cur.pop("timed_out", None)
            if state == "failed":
                cur["tail"] = tail[-_TAIL_SHOWN:]
            elapsed = cur["ended"] - cur["started"]
            operator, client = cur.get("operator", ""), cur.get("client", "")
        if state == "done":
            log.info("State reset: %s (%.1f s).", text, elapsed)
        else:
            log.error("State reset FAILED: %s.", text)
            if tail:
                log.error("  last %d line(s) of its output:", len(tail))
                for line in tail:
                    log.error("    | %s", line)
            for line in environment_report():
                log.error("  %s", line)
        self._record(f"state_reset_{state}", operator=operator, client=client, text=text,
                     exit_code=code, elapsed=round(elapsed, 2))
        self._notify()

    # -- out --------------------------------------------------------------------

    def _record(self, kind: str, **fields) -> None:
        if self._journal is None:
            return
        try:
            self._journal.record(kind, expt=self.expt, **fields)
        except Exception:
            log.exception("Could not journal %s", kind)

    def _notify(self) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(self.info())
        except Exception:
            log.exception("State reset change notification failed")

    def join(self, timeout: float | None = None) -> None:
        """Wait for the reset being followed to finish (tests, shutdown)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
