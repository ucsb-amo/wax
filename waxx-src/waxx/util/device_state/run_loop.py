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
  failed check refuses the Start; later it ends the loop.  liveOD's side is
  read by :func:`waxx.util.device_state.run_gate.classify`: a run in progress
  whose process is known to be gone (an Abort pending or not) is *waived* --
  one WARNING naming it, and the launch goes ahead (liveOD finalizes or
  supersedes it at the next INIT_RUN).  An Abort pending with no run in
  progress still ends the loop.
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

A loop is either fixed to one file (``LoopSpec.expt_path``) or a *pick* loop
(``LoopSpec.root``): its file is chosen at every Start -- the Sequences tab's
file dialog -- and must be a ``.py`` file inside ``root`` on the server's
machine, not one of ``LoopSpec.exclude`` (the monitor experiment).  The file is
read at every launch, so an edit takes effect at the loop's next run.

The loop never kills a run.  It sends liveOD read-only POLLs, and one message
more: when a run's process has exited and liveOD still has that run in progress
without having heard of the exit -- the process's own exit notice (an atexit
handler) never ran because it was killed or crashed hard, or it did not get
through -- the loop sends the RUN_EXITED notice for it, naming the run id, so
liveOD closes the run instead of showing it in progress until the next run.

Every run's terminal output is kept (``RunLoop.output``, an
:class:`~waxx.util.device_state.output_log.OutputLog`) for the GUIs' log view,
with a line of the loop's own before each run and at the end.
"""

from __future__ import annotations

import logging
import os
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
from waxx.util.device_state import loop_scan, run_gate
from waxx.util.device_state.loop_scan import ScanSettingsError, ScanSpec
from waxx.util.device_state.output_log import OutputLog
from waxx.util.device_state.state_reset import describe_expt

log = logging.getLogger(__name__)

#: Pause between one run's end and the next launch.
GAP_S = 0.5
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


#: Characters refused in a picked file's path: the run is launched through the
#: shell (``%kpy% & ar <file>``).
_SHELL_CHARS = set('&|<>^%"!')


@dataclass(frozen=True)
class LoopSpec:
    """A loop the server offers: ``key`` (what GUIs ask for), a title, and the
    experiment file.  Only files given to the server this way can be run --
    or, for a pick loop (``root`` set, ``expt_path`` empty), a ``.py`` file
    inside ``root`` chosen at Start, other than those in ``exclude``."""

    key: str
    title: str
    expt_path: str = ""
    root: str = ""
    exclude: tuple = ()
    #: Scan settings the GUI may set (:mod:`~waxx.util.device_state.loop_scan`).
    scan: ScanSpec | None = None

    @property
    def pick(self) -> bool:
        return bool(self.root)


def loop_specs(mapping: Mapping | None) -> list[LoopSpec]:
    """``{key: (title, path)}`` or ``{key: (title, path, scan)}`` -> specs,
    skipping entries without a path (an unset env var leaves the lab's paths
    None); ``scan`` is a :class:`~waxx.util.device_state.loop_scan.ScanSpec`
    or its fields as a mapping.  A pick loop's entry is ``{key: {"title": ...,
    "root": folder, "exclude": (paths,)}}`` (skipped without a root; a
    ``"scan"`` there too)."""
    specs = []
    for k, entry in (mapping or {}).items():
        if isinstance(entry, Mapping):
            if entry.get("root"):
                specs.append(LoopSpec(str(k), str(entry.get("title") or k),
                                      root=str(entry["root"]),
                                      exclude=tuple(str(p) for p in entry.get("exclude") or ()
                                                    if p),
                                      scan=_scan_spec(entry.get("scan"))))
            continue
        title, path, *rest = entry
        if path:
            specs.append(LoopSpec(str(k), str(title), str(path),
                                  scan=_scan_spec(rest[0] if rest else None)))
    return specs


def _scan_spec(scan) -> ScanSpec | None:
    if scan is None or isinstance(scan, ScanSpec):
        return scan
    return ScanSpec.from_mapping(scan)


def resolve_pick(spec: LoopSpec, path) -> tuple[Path | None, str]:
    """A pick loop's file from what a GUI sent (absolute, or relative to the
    root): ``(path, "")``, or ``(None, why it is refused)``."""
    if not spec.pick:
        return None, f"{spec.title} runs a fixed file"
    text = str(path or "").strip()
    if not text:
        return None, "no experiment file was chosen"
    bad = sorted(_SHELL_CHARS & set(text))
    if bad:
        return None, f"the file's path contains {' '.join(bad)} (not allowed): {text}"
    root = Path(spec.root).resolve()
    candidate = Path(text.replace("\\", "/"))
    full = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if full != root and root not in full.parents:
        return None, f"only experiments inside {root} can be looped, not {full}"
    if full.suffix.lower() != ".py":
        return None, f"not a Python file: {full}"
    if not full.is_file():
        return None, f"no such file on the monitor server's machine: {full}"
    for ex in spec.exclude:
        try:
            if full == Path(ex).resolve():
                return None, f"{full.name} cannot be looped (it is the monitor's own experiment)"
        except OSError:
            continue
    return full, ""


def _spawn(command: str, extra_env: Mapping | None = None):
    # unbuffered: the run's "Run ID:" line arrives when printed, and is not lost
    # in a buffer when the process is killed. WAXX_LAUNCHER: the run tells
    # liveOD at INIT_RUN that this loop launched it (run_gate).
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env[run_gate.LAUNCHER_ENV] = "run_loop"
    env.update(extra_env or {})
    return Popen(command, stdout=PIPE, stderr=STDOUT, universal_newlines=True,
                 bufsize=1, errors="replace", shell=True, env=env)


class _LiveOD:
    """POLL (and the exit notice, ``run_exited``) through one lazily made client;
    a failed call drops it so the next one rediscovers the server.

    A POLL that fails is tried once more on a fresh client: after a liveOD
    restart the kept client still points at the old port, and without the
    retry the first Start after the restart was refused ("liveOD is not
    reachable") although liveOD was up.  POLL is read-only, so repeating it is
    safe; the exit notice is not repeated."""

    def __init__(self):
        self._client = None

    def _call(self, fn, retry: bool = False):
        from waxx.util.live_od.live_od_client import LiveODClient  # noqa: PLC0415
        for attempt in range(2 if retry else 1):
            if self._client is None:
                self._client = LiveODClient(timeout_ms=3000, discovery_timeout=3.0)
            try:
                return fn(self._client)
            except Exception:
                client, self._client = self._client, None
                try:
                    client.close()
                except Exception:
                    pass
                if attempt == (1 if retry else 0):
                    raise
                log.info("liveOD POLL failed on the kept client; retrying on a fresh one "
                         "(liveOD may have restarted)")

    def __call__(self) -> dict:
        reply = self._call(lambda c: c.poll(), retry=True)
        if not reply.get("ok", False):
            raise RuntimeError(f"liveOD POLL failed: {reply}")
        return reply

    def run_exited(self, run_id: int, reason: str) -> dict:
        """RUN_EXITED for run ``run_id`` on behalf of its exited process (no run
        token: liveOD matches the run id instead)."""
        return self._call(lambda c: c._send_recv(
            {"tag": "RUN_EXITED", "run_id": int(run_id), "reason": reason}))


class RunLoop:
    """One loop.  ``info()`` is what GUIs are told (see the keys it sets).

    ``poll()`` returns liveOD's POLL reply (raises when unreachable);
    ``run_exited(run_id, reason)`` sends liveOD RUN_EXITED for a run whose
    process is gone (default: the default poll's own; none with a custom poll);
    ``fence()`` the run announced to the server, or None; ``busy()`` why
    something else of the server's holds the core, or ""; ``start_monitor(why)``
    starts the monitor unless it is running.  ``on_change(info)`` is called on
    every change, from whichever thread made it.
    """

    def __init__(self, spec: LoopSpec, *, poll: Callable[[], dict] | None = None,
                 run_exited: Callable[[int, str], dict] | None = None,
                 fence: Callable[[], dict | None] | None = None,
                 busy: Callable[[], str] | None = None,
                 start_monitor: Callable[[str], None] | None = None,
                 on_change: Callable[[dict], None] | None = None, journal=None,
                 spawn=None, clock: Callable[[], float] = time.time,
                 gap_s: float = GAP_S, poll_s: float = POLL_S):
        self.spec = spec
        #: The file run: the spec's, or a pick loop's last chosen one ("" until then).
        self.path = spec.expt_path
        self.expt = Path(self.path).stem if self.path else ""
        self._poll = poll or _LiveOD()
        self._run_exited = run_exited or getattr(self._poll, "run_exited", None)
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
        #: (run id, state) of the last dead run the gate waived (warned once)
        self._waived: tuple | None = None
        self._stop_by = ""
        self._external = ""
        self._about, self._about_key = "", None
        self._s: dict = {"state": "idle", "text": "not started", "runs": 0,
                         "run_id": None, "last": None}
        #: The scan settings the next run gets (None: the loop has none).  Kept
        #: in memory: a server restart goes back to the spec's defaults.
        self._scan = spec.scan.defaults() if spec.scan is not None else None
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
            scan = dict(self._scan) if self._scan is not None else None
        out.update(key=self.spec.key, title=self.spec.title, expt=self.expt,
                   path=self.path, about=self._read_about())
        if scan is not None:
            out["scan"] = dict(self.spec.scan.info(), settings=scan,
                               text=loop_scan.describe(self.spec.scan, scan))
        if self.spec.pick:
            out.update(pick=True, root=str(Path(self.spec.root).resolve()),
                       rel=self._rel(self.path))
        return out

    def _rel(self, path) -> str:
        if not path:
            return ""
        try:
            return Path(path).resolve().relative_to(Path(self.spec.root).resolve()).as_posix()
        except (OSError, ValueError):
            return str(path)

    def _read_about(self) -> str:
        path = self.path
        if not path:
            return ""
        try:
            key = (path, Path(path).stat().st_mtime_ns)
        except OSError:
            return ""
        if key != self._about_key:
            self._about, self._about_key = describe_expt(path), key
        return self._about

    def describe(self, path) -> dict:
        """What a pick loop would run for ``path`` (checked as at Start): the
        server's full path and the file's docstring, for the GUI's confirm."""
        full, why = resolve_pick(self.spec, path)
        if full is None:
            return {"status": "error", "msg": why}
        return {"status": "ok", "path": str(full), "rel": self._rel(full), "expt": full.stem,
                "about": describe_expt(full)}

    def _set(self, **fields) -> None:
        with self._lock:
            self._s.update(fields)
        self._notify()

    # -- requests ---------------------------------------------------------------

    def start(self, operator: str = "", client: str = "", path=None) -> dict:
        """Start the loop; a pick loop needs ``path`` (see :func:`resolve_pick`)."""
        who = _who(operator, client)
        if self.spec.pick:
            full, why = resolve_pick(self.spec, path)
            if full is None:
                return self._refused(why, who)
        else:
            full = Path(self.path)
            if not full.is_file():
                return self._refused(f"the loop's experiment file does not exist: {full}", who)
        with self._lock:
            if self._s["state"] in ACTIVE:
                return self._refused(f"{self.spec.title} is already running", who)
        blocked = self._gate()
        if blocked is not None:
            return self._refused(blocked[0], who)
        with self._lock:
            if self._s["state"] in ACTIVE:
                return self._refused(f"{self.spec.title} is already running", who)
            if self.spec.pick:
                self.path, self.expt = str(full), full.stem
            self._stop_by, self._external = "", ""
            self._wake.clear()
            self._s = {"state": "running", "text": f"started by {who}", "runs": 0,
                       "run_id": None, "last": None, "started": self._clock(),
                       "ended": None, "operator": operator, "client": client}
        log.info("%s: started by %s (%s).", self.spec.title, who, self.path)
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

    def configure(self, settings, operator: str = "", client: str = "") -> dict:
        """Set the scan settings (loop_scan's dict, SI); while the loop runs
        they apply from its next run."""
        who = _who(operator, client)
        if self.spec.scan is None:
            return {"status": "error", "msg": f"{self.spec.title} has no scan settings"}
        try:
            new = loop_scan.normalize(self.spec.scan, settings)
        except ScanSettingsError as exc:
            return {"status": "error", "msg": str(exc)}
        with self._lock:
            old, self._scan = self._scan, new
            active = self._s["state"] in ACTIVE
        text = loop_scan.describe(self.spec.scan, new)
        log.info("%s: scan set by %s: %s (was %s).", self.spec.title, who, text,
                 loop_scan.describe(self.spec.scan, old))
        self._record("run_loop_configure", operator=operator, client=client, scan=new,
                     previous=old)
        if active:
            self.output.mark(f"scan set by {who}: {text} -- from the next run", self._clock)
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
        # liveOD's side is the shared classifier's (the fence was handled above,
        # with this loop's own runs exempt)
        verdict = run_gate.classify(poll, None)
        if verdict.state == "free":
            return None
        if poll.get("run_in_progress"):
            if verdict.waivable:
                # its process is gone: the next INIT_RUN finalizes (an Abort
                # pending) or supersedes it -- no person needs to step in.
                # Said once per run (Start and the first launch both ask).
                key = (verdict.run_id, verdict.state)
                if key != self._waived:
                    self._waived = key
                    log.warning("%s: run %s is %s in liveOD and is waived: %s",
                                self.spec.title, verdict.run_id, verdict.state,
                                verdict.reason)
                    self._record("run_loop_waived", run_id=verdict.run_id,
                                 state=verdict.state, reason=verdict.reason)
                return None
            if verdict.state in ("wedged", "reset_pending"):
                return verdict.reason, False         # it names the run itself
            return (f"run {poll.get('run_id')} ({poll.get('expt_name') or 'experiment'}) "
                    "is in progress in liveOD"), False
        if poll.get("reset_requested"):
            # no run: a person's Abort between runs ends the loop, as before
            return "an Abort is pending in liveOD", True
        return verdict.reason, False

    def _one_run(self) -> tuple[str, str, bool] | None:
        """Launch the experiment once and follow it; None when it saved."""
        path = self.path
        command = ar_command(f'"{path}"' if " " in path else path)
        n = self.info()["runs"] + 1
        with self._lock:
            scan = dict(self._scan) if self._scan is not None else None
        try:
            if scan is None:
                proc = self._spawn(command)
            else:
                proc = self._spawn(command, {loop_scan.ENV_VAR: loop_scan.to_env(scan)})
        except OSError as exc:
            log.error("%s: could not spawn %r: %r", self.spec.title, command, exc)
            for line in environment_report():
                log.error("  %s", line)
            return "latched", f"could not start {self.expt}: {exc!r}", True
        mark = f"{_nth(n)} run of the loop: {self.expt}"
        if scan is not None:
            mark += f" ({loop_scan.describe(self.spec.scan, scan)})"
        self.output.mark(mark, self._clock)
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
                    # Not the Abort of a dead run the gate waived: it stays set
                    # until this run's INIT_RUN finalizes that run.
                    try:
                        poll = self._poll()
                        abort_seen = (bool(poll.get("reset_requested"))
                                      and not self._waived_reset(poll))
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
        self._tell_live_od_it_exited(run_id, code, list(tail), n)
        return self._judge(code, list(tail), run_id, abort_seen)

    def _waived_reset(self, poll: dict) -> bool:
        """POLL's pending Abort is the one of the dead run the gate waived: liveOD's
        current run is still that run (no INIT_RUN since). An Abort a person
        presses in that window cannot be told apart; liveOD spends it on the
        dead run at this run's INIT_RUN."""
        waived = self._waived
        return (waived is not None and waived[1] == "reset_pending"
                and poll.get("run_id") == waived[0])

    def _tell_live_od_it_exited(self, run_id, code, tail: list[str], n: int) -> None:
        """The run's process has exited. If liveOD still has the run in progress
        and has not heard that its process exited (``run_state`` "exited"), send
        RUN_EXITED for it: the process's own notice (an atexit handler) never ran
        -- it was killed or crashed hard -- or did not get through. Without it
        the run stays "in progress" in liveOD until the next run starts."""
        if self._run_exited is None:
            return
        try:
            poll = self._poll()
        except Exception as exc:
            log.warning("%s: could not ask liveOD whether it knows the run ended: %s",
                        self.spec.title, exc)
            return
        if not poll.get("run_in_progress") or poll.get("run_state") == "exited":
            return
        live_id = poll.get("run_id")
        if run_id is None:
            # killed before its "Run ID:" line came through: liveOD's run is this
            # one only if it is our experiment and started after this launch
            age = poll.get("init_run_age_s")
            started = self.info().get("run_started")
            if (Path(str(poll.get("expt_name") or "")).stem != self.expt or age is None
                    or started is None or age > self._clock() - started):
                return
        elif live_id != run_id:
            return
        reason = f"its process ended with exit code {code} and sent no exit notice"
        if tail:
            reason += f"; last line: {tail[-1]}"
        reason += " (sent by the monitor server's run loop)"
        try:
            # require_known_dead stays False here: the loop never kills its child,
            # and proc.wait() returned because the shell ended after `ar` did, so
            # an unknown pid (older client) still means the experiment has exited.
            # A pid liveOD knows to be alive is never told (the helper's rule).
            sent = run_gate.tell_live_od_run_exited(None, live_id, reason, poll=poll,
                                                    send=self._run_exited)
        except Exception as exc:
            log.error("%s: run %s is still in progress in liveOD after its process exited, "
                      "and telling liveOD failed: %s", self.spec.title, live_id, exc)
            self.output.mark(f"run {live_id}: liveOD still shows it in progress, and could "
                             f"not be told its process exited ({exc})", self._clock)
            return
        if not sent["sent"]:
            # liveOD is saving it, or its experiment process still lives
            log.warning("%s: run %s is still in progress in liveOD after the loop's process "
                        "for it exited (exit code %s); no exit notice sent: %s",
                        self.spec.title, live_id, code, sent["why"])
            return
        reply = sent["reply"] if isinstance(sent["reply"], dict) else {"ok": sent["ok"]}
        ok = bool(reply.get("ok"))
        log.warning("%s: run %s was still in progress in liveOD after its process exited "
                    "(exit code %s); told liveOD: %s", self.spec.title, live_id, code,
                    "done" if ok else reply)
        self.output.mark(f"run {live_id}: its process exited without telling liveOD; "
                         + ("the loop told it" if ok else f"liveOD refused the notice: {reply}"),
                         self._clock)
        self._record("run_loop_told_live_od_exited", run_id=live_id, n=n, exit_code=code,
                     ok=ok)

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
