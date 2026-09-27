"""Run one experiment file back to back, from the monitor server.

The lab's use: the BEC TOF loop (kexp: ``experiments/tools/auto_tof.py``),
started and stopped from a card on the Device Control GUI's Composite tab.  It
runs in the monitor server because that process outlives the GUIs and already
launches experiments the lab's way (``%kpy% & ar <file>``, as
:class:`~waxx.util.device_state.state_reset.StateReset` does).

One run at a time:

* Before every run the machine must be free: liveOD reachable, no run in
  progress there and no Abort pending, no other run announced to the server,
  nothing else of the server's (a state reset) holding the core.  At Start a
  failed check refuses the Start; later it ends the loop.
* A run that ends cleanly -- exit code 0 and liveOD outcome ``saved`` -- is
  followed by the next.  Anything else ends the loop and it stays off
  (*latched*) until someone presses Start again: an Abort in liveOD (during a
  run, or pressed while the next run was starting -- liveOD spends that on no
  run, it only skips a run id), a nonzero exit, a run saved incomplete, the
  core taken by another process, the monitor started by someone.
* Stop lets the run in progress finish and save, then ends the loop.
* When the loop ends the monitor is started again -- the loop's experiment
  chains runs without restarting it -- unless someone else's run holds the
  machine or starting the monitor is what ended the loop.  "Start" means
  start unless it is already running: an aborted run has usually restarted it
  already (its abort path sends ``run complete``).

The loop only ever sends liveOD read-only POLLs, and never kills a run.

Every run's terminal output is kept (``RunLoop.output``, an
:class:`~waxx.util.device_state.output_log.OutputLog`) for the GUIs' log view,
with a line of the loop's own before each run and at the end.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Callable, Iterable, Mapping

from waxx.util.device_state.monitor_manager import (
    _INTERRUPTED_SIGNATURES, _diagnose, _matches, ar_command, environment_report)
from waxx.util.device_state.output_log import OutputLog
from waxx.util.device_state.state_reset import describe_expt

log = logging.getLogger(__name__)

#: Pause between one run's end and the next launch.
GAP_S = 2.0
#: How often liveOD is polled for an Abort while a run is starting.
POLL_S = 0.5

RUN_ID_RE = re.compile(r"Run ID:\s*(\d+)")
ABORTED_RE = re.compile(r"Acquisition for run (\d+) aborted")

_TAIL_LINES = 25
_TAIL_SHOWN = 8

#: States: ``idle`` (never started), ``running``, ``stopping`` (Stop pressed,
#: the run in progress finishes first), ``stopped`` (ended by Stop),
#: ``latched`` (ended by anything else; off until Start).
ACTIVE = ("running", "stopping")


@dataclass(frozen=True)
class LoopSpec:
    """A loop the server offers: ``key`` (what GUIs ask for), a title, and the
    experiment file.  Only files given to the server this way can be run."""

    key: str
    title: str
    expt_path: str


def loop_specs(mapping: Mapping | None) -> list[LoopSpec]:
    """``{key: (title, path)}`` -> specs, skipping entries without a path (an
    unset env var leaves the lab's paths None)."""
    return [LoopSpec(str(k), str(title), str(path))
            for k, (title, path) in (mapping or {}).items() if path]


def _spawn(command: str):
    return Popen(command, stdout=PIPE, stderr=STDOUT, universal_newlines=True,
                 bufsize=1, errors="replace", shell=True)


class _LiveOD:
    """POLL only, through one lazily made client; a failed call drops it so the
    next one rediscovers the server."""

    def __init__(self):
        self._client = None

    def __call__(self) -> dict:
        from waxx.util.live_od.live_od_client import LiveODClient  # noqa: PLC0415
        if self._client is None:
            self._client = LiveODClient(timeout_ms=3000, discovery_timeout=3.0)
        try:
            reply = self._client.poll()
        except Exception:
            self._client = None
            raise
        if not reply.get("ok", False):
            raise RuntimeError(f"liveOD POLL failed: {reply}")
        return reply


class RunLoop:
    """One loop.  ``info()`` is what GUIs are told (see the keys it sets).

    ``poll()`` returns liveOD's POLL reply (raises when unreachable);
    ``fence()`` the run announced to the server, or None; ``busy()`` why
    something else of the server's holds the core, or ""; ``start_monitor(why)``
    starts the monitor unless it is running.  ``on_change(info)`` is called on
    every change, from whichever thread made it.
    """

    def __init__(self, spec: LoopSpec, *, poll: Callable[[], dict] | None = None,
                 fence: Callable[[], dict | None] | None = None,
                 busy: Callable[[], str] | None = None,
                 start_monitor: Callable[[str], None] | None = None,
                 on_change: Callable[[dict], None] | None = None, journal=None,
                 spawn=None, clock: Callable[[], float] = time.time,
                 gap_s: float = GAP_S, poll_s: float = POLL_S):
        self.spec = spec
        self.expt = Path(spec.expt_path).stem
        self._poll = poll or _LiveOD()
        self._fence = fence
        self._busy = busy
        self._start_monitor = start_monitor
        self._on_change = on_change
        self._journal = journal
        self._spawn = spawn or _spawn
        self._clock = clock
        self._gap_s = float(gap_s)
        self._poll_s = float(poll_s)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._own_run_ids: deque = deque(maxlen=20)
        self._stop_by = ""
        self._external = ""
        self._about, self._about_mtime = "", None
        self._s: dict = {"state": "idle", "text": "not started", "runs": 0,
                         "run_id": None, "last": None}
        #: The runs' terminal output, for the GUIs (the server log has it too).
        self.output = OutputLog()

    # -- state ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        with self._lock:
            return self._s["state"] in ACTIVE

    def info(self) -> dict:
        with self._lock:
            out = dict(self._s)
        out.update(key=self.spec.key, title=self.spec.title, expt=self.expt,
                   path=self.spec.expt_path, about=self._read_about())
        return out

    def _read_about(self) -> str:
        try:
            mtime = Path(self.spec.expt_path).stat().st_mtime_ns
        except OSError:
            return ""
        if mtime != self._about_mtime:
            self._about, self._about_mtime = describe_expt(self.spec.expt_path), mtime
        return self._about

    def _set(self, **fields) -> None:
        with self._lock:
            self._s.update(fields)
        self._notify()

    # -- requests ---------------------------------------------------------------

    def start(self, operator: str = "", client: str = "") -> dict:
        who = _who(operator, client)
        path = Path(self.spec.expt_path)
        if not path.is_file():
            return self._refused(f"the loop's experiment file does not exist: {path}", who)
        with self._lock:
            if self._s["state"] in ACTIVE:
                return self._refused(f"{self.spec.title} is already running", who)
        blocked = self._gate()
        if blocked is not None:
            return self._refused(blocked[0], who)
        with self._lock:
            self._stop_by, self._external = "", ""
            self._wake.clear()
            self._s = {"state": "running", "text": f"started by {who}", "runs": 0,
                       "run_id": None, "last": None, "started": self._clock(),
                       "ended": None, "operator": operator, "client": client}
        log.info("%s: started by %s (%s).", self.spec.title, who, self.spec.expt_path)
        self._record("run_loop_start", operator=operator, client=client)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"run-loop-{self.spec.key}")
        self._thread.start()
        self._notify()
        return {"status": "ok", "loop": self.info()}

    def stop(self, operator: str = "", client: str = "") -> dict:
        """Finish the run in progress, then end the loop."""
        who = _who(operator, client)
        with self._lock:
            if self._s["state"] not in ACTIVE:
                return {"status": "error", "msg": f"{self.spec.title} is not running"}
            self._stop_by = who
            self._s["state"] = "stopping"
            self._s["text"] = (f"Stop pressed by {who}: run {self._s['run_id']} finishes first"
                               if self._s.get("run_id") else f"Stop pressed by {who}")
        log.info("%s: stop requested by %s.", self.spec.title, who)
        self._record("run_loop_stop", operator=operator, client=client)
        self._wake.set()
        self._notify()
        return {"status": "ok", "loop": self.info()}

    def note_external(self, why: str) -> None:
        """Something else is taking the machine (the monitor being started):
        end the loop after the run in progress, latched, without starting the
        monitor."""
        with self._lock:
            if self._s["state"] not in ACTIVE:
                return
            self._external = why
        log.warning("%s: %s -- the loop ends.", self.spec.title, why)
        self._wake.set()

    def shutdown(self) -> None:
        """The server is going away: no further runs (the one in progress is
        left alone)."""
        self.note_external("the monitor server is shutting down")

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _refused(self, msg: str, who: str) -> dict:
        log.warning("%s: start refused: %s", self.spec.title, msg)
        self._record("run_loop_refused", who=who, msg=msg)
        return {"status": "error", "msg": msg}

    # -- the loop ---------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:
            log.exception("%s: the loop failed", self.spec.title)
            self._end("latched", f"the loop itself failed: {exc!r}", start_monitor=True)

    def _loop(self) -> None:
        first = True
        while True:
            if not first:
                self._wake.wait(self._gap_s)
            ending = self._between_runs()
            if ending is not None:
                self._end(*ending)
                return
            first = False
            ending = self._one_run()
            if ending is not None:
                self._end(*ending)
                return

    def _between_runs(self) -> tuple[str, str, bool] | None:
        with self._lock:
            external, stop_by, runs = self._external, self._stop_by, self._s["runs"]
        if external:
            return "latched", external, False
        if stop_by:
            return "stopped", f"stopped by {stop_by} after {_n_runs(runs)}", True
        blocked = self._gate()
        if blocked is not None:
            return "latched", blocked[0], blocked[1]
        return None

    def _gate(self) -> tuple[str, bool] | None:
        """(why the machine is not free, whether the monitor may be started
        then) -- or None when it is free."""
        busy = self._busy() if self._busy is not None else ""
        if busy:
            return busy, False
        pending = self._fence() if self._fence is not None else None
        if pending and pending.get("run_id") not in self._own_run_ids:
            return (f"run {pending.get('run_id')} ({pending.get('expt') or 'experiment'}) "
                    "announced itself to the monitor server"), False
        try:
            poll = self._poll()
        except Exception as exc:
            return f"liveOD is not reachable ({exc}) -- a run could not save", True
        if poll.get("run_in_progress"):
            return (f"run {poll.get('run_id')} ({poll.get('expt_name') or 'experiment'}) "
                    "is in progress in liveOD"), False
        if poll.get("reset_requested"):
            return "an Abort is pending in liveOD", True
        return None

    def _one_run(self) -> tuple[str, str, bool] | None:
        """Launch the experiment once and follow it; None when it saved."""
        command = ar_command(self.spec.expt_path)
        n = self.info()["runs"] + 1
        try:
            proc = self._spawn(command)
        except OSError as exc:
            log.error("%s: could not spawn %r: %r", self.spec.title, command, exc)
            for line in environment_report():
                log.error("  %s", line)
            return "latched", f"could not start {self.expt}: {exc!r}", True
        self.output.mark(f"{_nth(n)} run of the loop: {self.expt}", self._clock)
        self._set(run_id=None, run_started=self._clock(), tail=None,
                  text=f"run {n} of the loop starting ({self.expt})")
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=_pump, args=(proc, lines), daemon=True,
                         name=f"run-loop-{self.spec.key}-out").start()
        tail: deque[str] = deque(maxlen=_TAIL_LINES)
        run_id, abort_seen = None, False
        while True:
            try:
                line = lines.get(timeout=self._poll_s)
            except queue.Empty:
                if run_id is None and not abort_seen:
                    # Before INIT_RUN an Abort is spent on no run (liveOD
                    # skips an id and lets this run through): notice it here.
                    try:
                        abort_seen = bool(self._poll().get("reset_requested"))
                    except Exception:
                        pass
                continue
            if line is None:
                break
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            self.output.append(line)
            log.info("[%s] %s", self.spec.key, line)
            m = RUN_ID_RE.search(line)
            if m and run_id is None:
                run_id = int(m.group(1))
                self._own_run_ids.append(run_id)
                self._set(run_id=run_id, text=f"run {run_id} in progress ({_nth(n)} of the loop)")
                self._record("run_loop_run", run_id=run_id, n=n)
        code = proc.wait()
        return self._judge(code, list(tail), run_id, abort_seen)

    def _judge(self, code, tail: list[str], run_id, abort_seen: bool):
        name = f"run {run_id}" if run_id is not None else "the run (it never got a run id)"
        text = "\n".join(tail)
        if code == 0 and run_id is not None:
            outcome = self._outcome(run_id)
            if outcome is None:
                return self._failed(f"{name} exited but liveOD reports no outcome for it",
                                    tail, run_id, code)
            if outcome.get("outcome") != "saved":
                detail = outcome.get("detail") or ""
                return self._failed(f"{name} ended {outcome.get('outcome')}"
                                    + (f" ({detail})" if detail else ""), tail, run_id, code)
            with self._lock:
                self._s["runs"] += 1
                self._s["last"] = {"run_id": run_id, "outcome": "saved", "ended": self._clock()}
                self._s["run_id"] = None
                self._s["text"] = f"run {run_id} saved"
            self._notify()
            if abort_seen:
                return ("latched", f"an Abort was pressed in liveOD while {name} was starting; "
                        f"liveOD spent it on no run (it skipped a run id), so {name} ran to the "
                        "end and was saved", True)
            return None
        m = ABORTED_RE.search(text)
        if m:
            return self._failed(f"run {m.group(1)} was aborted in liveOD (Abort): its file "
                                "was discarded", tail, run_id, code, show_tail=False)
        if _matches(text, _INTERRUPTED_SIGNATURES):
            return self._failed(f"{name} lost the core device to another process (the monitor "
                                "or another experiment was started)", tail, run_id, code,
                                start_monitor=False)
        if code == 0:
            return self._failed(f"{self.expt} exited without a run id -- its prepare() did "
                                "not register a run with liveOD", tail, run_id, code)
        why = f"{name} failed (exit code {code})"
        hints = _diagnose(text)
        if hints:
            why += f" -- likely cause: {hints[0]}"
        elif tail:
            why += f" -- last line: {tail[-1]}"
        return self._failed(why, tail, run_id, code)

    def _failed(self, why, tail, run_id, code, show_tail=True, start_monitor=True):
        with self._lock:
            self._s["last"] = {"run_id": run_id, "outcome": "failed", "exit_code": code,
                               "ended": self._clock()}
            if show_tail:
                self._s["tail"] = tail[-_TAIL_SHOWN:]
        if show_tail and tail:
            log.error("%s: %s. Last %d line(s):", self.spec.title, why, len(tail))
            for line in tail:
                log.error("    | %s", line)
        return "latched", why, start_monitor

    def _outcome(self, run_id) -> dict | None:
        """liveOD's record of how run ``run_id`` ended (POLL's last_outcome)."""
        for _ in range(3):
            try:
                last = self._poll().get("last_outcome") or {}
            except Exception:
                last = {}
            if last.get("run_id") == run_id:
                return last
            time.sleep(self._poll_s)
        return None

    def _end(self, state: str, why: str, start_monitor: bool) -> None:
        with self._lock:
            self._s.update(state=state, text=why, ended=self._clock(), run_id=None)
            runs = self._s["runs"]
        (log.info if state == "stopped" else log.warning)(
            "%s %s after %s: %s", self.spec.title,
            "stopped" if state == "stopped" else "LATCHED OFF", _n_runs(runs), why)
        self.output.mark(f"{'stopped' if state == 'stopped' else 'LATCHED OFF'} after "
                         f"{_n_runs(runs)}: {why}", self._clock)
        self._record("run_loop_end", state=state, why=why, runs=runs,
                     start_monitor=bool(start_monitor))
        self._notify()
        if start_monitor and self._start_monitor is not None:
            try:
                self._start_monitor(f"{self.spec.title} ended: {why}")
            except Exception:
                log.exception("%s: could not ask for the monitor", self.spec.title)

    # -- out --------------------------------------------------------------------

    def _record(self, kind: str, **fields) -> None:
        if self._journal is None:
            return
        try:
            self._journal.record(kind, loop=self.spec.key, expt=self.expt, **fields)
        except Exception:
            log.exception("Could not journal %s", kind)

    def _notify(self) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(self.info())
        except Exception:
            log.exception("Run loop change notification failed")


def _pump(proc, lines: queue.Queue) -> None:
    try:
        if proc.stdout is not None:
            for raw in proc.stdout:
                lines.put(raw)
    except Exception as exc:
        lines.put(f"(reading the run's output failed: {exc!r})")
    finally:
        lines.put(None)


def _who(operator: str, client: str) -> str:
    return "@".join(p for p in (operator, client) if p) or "?"


def _n_runs(n: int) -> str:
    return f"{n} saved run{'s' if n != 1 else ''}"


def _nth(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def active_loop(loops: Iterable[RunLoop]) -> RunLoop | None:
    for loop in loops:
        if loop.active:
            return loop
    return None
