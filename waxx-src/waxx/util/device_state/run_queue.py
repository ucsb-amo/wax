"""The K-machine run queue: experiment jobs, one at a time, run by the monitor
server (run-queue plan, phase 1).

The queue generalises the server's run loop (:mod:`~waxx.util.device_state.run_loop`)
from "one file back to back" to "these jobs, in this order": it reuses the
loop's launch command (:func:`~waxx.util.device_state.monitor_manager.ar_command`,
``artiq_run`` directly), its judging of a finished run
(:func:`~waxx.util.device_state.run_loop.judge_run`: exit code 0 and liveOD's
outcome "saved"), its RUN_EXITED-on-behalf rule
(:func:`~waxx.util.device_state.run_loop.tell_live_od_exited`) and the shared
liveOD verdict (:func:`~waxx.util.device_state.run_gate.classify`).  This module
is pure logic: the server passes liveOD, its fence, its monitor state and its
loops in, and calls :meth:`RunQueue.tick` from its watch thread.

**Jobs.**  A job is one run of one experiment file: its absolute path and the
file's SHA-256 at submit, extra ``artiq_run`` arguments, working folder, a
label, its owner (``"person"`` -- the default -- or ``"agent"``), a priority
(higher first), an optional due time, jobs it must wait for (``after``: each
must end *saved*), an optional chain name, and the submitter's write-back veto.
``repeat`` N submits N jobs in one chain.  States: ``queued`` -> ``running``
-> ``ending`` (its process has exited; the outcome is being read) ->
``saved`` | ``failed``; or ``cancelled`` / ``skipped`` (never launched, or
launched and cancelled).

**Scheduling.**  One slot.  A job is *eligible* when it is queued, its due time
has passed, every job in ``after`` is saved, it is not paused (scope "all", or
"agent" for an agent's job) and -- for an agent's job -- no person's hold is on
(:mod:`~waxx.util.device_state.person_hold`).  Of the eligible jobs the one
with the largest ``(priority, -due, -id)`` goes next (ARTIQ's scheduler key;
no due time counts as 0).  It is launched only when the previous job's process
has exited and the machine is free: nothing of the server's own holds it
(``server_busy``: a state reset, the monitor starting), no run loop is active,
and liveOD + the run fence say free (:func:`run_gate.classify`; a dead run is
waived as by the loop).  A job ``after`` one that ended failed / cancelled /
skipped is skipped; a job in a chain that fails or is skipped (with
``stop_on_failure``, the default in a chain) cancels the chain's other queued
jobs.

**Launch.**  At launch the file's SHA-256 is computed again: a file changed
since submit is *skipped* ("source changed since submit") unless the job
allows drift.  The command is ``ar_command(path) + argv``, run detached
(:mod:`~waxx.util.device_state.detached`: the job survives a server restart)
with ``WAXX_LAUNCHER=kq``, ``WAXX_QUEUE_JOB=<id>``, ``WAXX_OWNER=<owner>``,
``PYTHONUNBUFFERED=1`` and, for a write-back veto, ``WAXX_CAL_NO_WRITE_BACK=1``;
its output goes to ``<dir>/logs/<id>_<label>.out``.  The job's run is known
from liveOD: the experiment sends ``queue_job`` (WAXX_QUEUE_JOB = "<id>:<token>") at INIT_RUN
and POLL / last_outcome carry it next to ``launcher`` and ``client_pid`` (the
experiment's own pid); the "Run ID:" line in the log (printed whatever
WAX_VERBOSITY when a launcher is set) is the fallback.

**Cancel.**  A queued job is cancelled at once.  A running job is never
terminated: the queue sends liveOD's Abort (RESET) for it -- only when liveOD's
run in progress is that job's run id, on a POLL just before -- and the job
stays ``running`` until its process ends; it then ends ``cancelled`` (or
``saved``, if it saved before the Abort reached it).  An Abort is what liveOD's
own button does: the run stops at its next shot and liveOD discards its file.
An agent may not cancel a person's running job.

**Owners.**  A job's ``owner`` (submit) and the asker's ``owner`` on cancel,
release and resume ("person" | "agent"; required on those three, refused
without) decide what an agent may undo: an agent may not cancel a person's
job, release a person's hold or resume (or replace) a person's pause.  A
hold or pause records who put it on (``owner``, "person" when absent; liveOD's
Reset and the fail-closed hold/pause are a person's).  The kq client sets
``owner`` from the environment variable ``WAXX_OWNER`` ("person" when unset;
the run-experiment skill sets "agent"); the Device Control GUI sends "person".

**Pause / hold.**  ``pause`` with scope "agent" or "all" stops new launches for
those jobs (the running one is not touched); ``resume`` lifts it.  A person's
hold is pause("agent") with provenance (since, by, reason) and is never lifted
by itself.

**Loops.**  When a job is eligible and one of the server's run loops is active
(the TOF loop), the queue asks it to stop gracefully (its run in progress
finishes and saves; no monitor start), remembers it, and runs its jobs; when no
job is eligible or running any more -- with no hold and nothing paused -- it
starts that loop again (not a loop that latched off).  Otherwise, when the
queue has launched jobs and has none left to run, it asks for the monitor (the
queue's jobs do not restart it; see ``WAXX_LAUNCHER`` in ``Expt.end_wax``).

**Alarm.**  An eligible job waiting with no job running for more than
:data:`ALARM_S` raises an alarm: one WARNING (and journal record) per
:data:`ALARM_S`, ``alarm`` in the status, cleared by the next launch.

**Records.**  ``<dir>`` is on local disk (the server's default:
:func:`default_dir`, ``~/.waxx/run_queue``).  ``<dir>/queue.json`` holds every job not yet ended and the last
:data:`KEEP_ENDED` ended ones (atomic replace on every change);
``<dir>/journal.jsonl`` gets one line per transition (append only), and the
server's ops journal the same records (kinds ``run_queue_*``; copied by
:meth:`RunQueue.flush_journal` outside the queue's lock, since the ops journal
appends to the lab's share synchronously).  At a server
restart the queue is read back; a running job whose process is still alive
(same pid, same creation time) is adopted and followed as before; one whose
process is gone is judged from liveOD's record (saved, or failed "server
restarted; process gone").

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import socket
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Mapping

from waxx.util.device_state import run_gate
from waxx.util.device_state.monitor_manager import ar_command, environment_report
from waxx.util.device_state.detached import LAUNCH_TIMEOUT_S
from waxx.util.device_state.person_hold import PersonHold
from waxx.util.device_state.run_loop import (
    RUN_ID_RE, _SHELL_CHARS, judge_run, live_od_outcome, tell_live_od_exited)

log = logging.getLogger(__name__)

STATES = ("queued", "launching", "running", "ending", "saved", "failed", "cancelled",
          "skipped")
#: States a job ends in.
ENDED = ("saved", "failed", "cancelled", "skipped")
#: States of the one job in the slot ("launching": saved before its process is
#: started, so a server that dies meanwhile never starts it twice).
IN_SLOT = ("launching", "running", "ending")
OWNERS = ("person", "agent")
PAUSE_SCOPES = ("agent", "all")

#: The value of ``WAXX_LAUNCHER`` for the queue's jobs (run_gate.KNOWN_LAUNCHERS).
LAUNCHER = "kq"
JOB_ENV = "WAXX_QUEUE_JOB"
OWNER_ENV = "WAXX_OWNER"
#: A submitter's write-back veto (submit "write_back": false) reaches the job as
#: this variable = "1".  It is honoured by the calibration emit on the
#: jep/calibration-writeback branch (waxx/calibration/emit.py: "[cal] not
#: applied: vetoed by WAXX_CAL_NO_WRITE_BACK"); this branch only sets it.
NO_WRITE_BACK_ENV = "WAXX_CAL_NO_WRITE_BACK"

#: A person's job's priority when none is given (an agent's: 0).
PERSON_PRIORITY = 10
#: An eligible job waiting this long with nothing running raises the alarm.
ALARM_S = 600.0
#: liveOD is polled at most this often for the hold's watch and the status
#: (a launch, a cancel and an outcome poll afresh).
POLL_EVERY_S = 2.0
#: Pause after a job's end before the next launch.
GAP_S = 0.5
#: Ended jobs kept in queue.json (all of them stay in journal.jsonl).
KEEP_ENDED = 500
MAX_REPEAT = 1000
#: Lines of a job's output kept in memory for its judging and ``describe``.
TAIL_LINES = 25
#: A launch taking longer than this is logged (once) while it is waited for.
SPAWN_WARN_S = 60.0
#: A job left "launching" with no process to follow (the server stopped while
#: launching it, or the launcher never answered) is looked for in liveOD this
#: long after its launch (INIT_RUN comes at the end of its prepare), then
#: ended failed.
ORPHAN_WAIT_S = 300.0

_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")

#: Environment variable that moves the queue's folder (tests point it at a
#: temporary folder; a lab may point it at another LOCAL folder).
DIR_ENV = "WAXX_RUN_QUEUE_DIR"


#: Environment variable listing the folders the queue runs files from
#: (os.pathsep-separated); see :func:`default_roots`.
ROOTS_ENV = "WAXX_RUN_QUEUE_ROOTS"


def default_roots() -> list[str]:
    """The folders a job's file must be inside: ``$WAXX_RUN_QUEUE_ROOTS``
    (os.pathsep-separated), else the lab's code folder (``%code%``) and
    ``C:\\lab\\skynet_log`` (agents' work).  A server may pass its own
    (``RunQueue(roots=)``); the lab can override them there later."""
    env = os.environ.get(ROOTS_ENV)
    if env:
        return [r for r in env.split(os.pathsep) if r]
    roots = [os.environ.get("code") or ""]
    roots.append(r"C:\lab\skynet_log")
    return [r for r in roots if r]


def default_dir() -> str:
    """The queue's folder when the server is given none: ``$WAXX_RUN_QUEUE_DIR``,
    else ``~/.waxx/run_queue`` -- local disk, next to the other ~/.waxx state
    (the job logs are written by the running experiments: a network share
    hiccup there would stall their output)."""
    return os.environ.get(DIR_ENV) or str(Path.home() / ".waxx" / "run_queue")


def _clock_text(t) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(t)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _close(proc) -> None:
    """Close a process watch (its handle); never raises."""
    close = getattr(proc, "close", None)
    if close is not None:
        try:
            close()
        except Exception:                             # noqa: BLE001
            log.exception("Run queue: closing a process watch failed")


def _quote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or "\t" in arg) else arg


@dataclass
class Job:
    """One job (see the module docstring).  ``to_dict()`` is its record in
    queue.json and in every reply."""

    id: int
    token: str
    path: str
    sha256: str
    label: str
    argv: list = field(default_factory=list)
    cwd: str = ""
    owner: str = "person"
    priority: int = 0
    due: float | None = None
    after: list = field(default_factory=list)
    chain: str | None = None
    stop_on_failure: bool = False
    write_back: bool | None = None
    allow_drift: bool = False
    repeat_index: int = 1
    repeat_of: int = 1
    submitted_at: float = 0.0
    submitted_by: str = ""
    state: str = "queued"
    #: why it was skipped / cancelled / failed, or "" (saved, or not ended)
    reason: str = ""
    #: the launched shell's pid and creation time (epoch s)
    pid: int | None = None
    pid_started: float | None = None
    #: the experiment process's pid, from liveOD's POLL (client_pid)
    client_pid: int | None = None
    run_id: int | None = None
    log_path: str | None = None
    exit_code: int | None = None
    #: {"outcome": liveOD's outcome or None, "detail", "why", "exit_code",
    #:  "aborted", "lost_core", "told_live_od"}
    outcome: dict | None = None
    launched_at: float | None = None
    ended_at: float | None = None
    #: {"by", "at", "abort_sent": bool, "abort_note": str} once a cancel of a
    #: running job was asked for
    cancel: dict | None = None
    adopted: bool = False

    @property
    def name(self) -> str:
        return f"job {self.id} ({self.label})"

    @property
    def queue_job(self) -> str:
        """"<id>:<token>": the job's WAXX_QUEUE_JOB, sent by its experiment at
        INIT_RUN and reported by liveOD (POLL, last_outcome) as queue_job."""
        return f"{self.id}:{self.token}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "Job":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in known})


def requester_owner(obj: Mapping) -> tuple[str, str]:
    """``(owner, "")`` from a request's required ``owner`` ("person" |
    "agent": who is asking -- the kq client sends WAXX_OWNER, the Device
    Control GUI "person"), or ``("", why it is refused)``."""
    owner = obj.get("owner")
    if owner not in OWNERS:
        return "", (f"owner is required ({' or '.join(OWNERS)}: who is asking)"
                    if owner in (None, "") else
                    f"owner must be one of {', '.join(OWNERS)}, not {owner!r}")
    return str(owner), ""


class QueueError(ValueError):
    """A request the queue refuses (the message says why)."""


class RunQueue:
    """The queue.  Requests (:meth:`submit`, :meth:`cancel`, :meth:`list`,
    :meth:`describe`, :meth:`pause`, :meth:`resume`, :meth:`hold`,
    :meth:`release`) return reply dicts and may come from any thread;
    :meth:`tick` does all the work and is called by one thread (the server's
    watch).

    Inputs (callables, so it runs without a network in tests):
    ``poll()`` -> liveOD's POLL reply (raises when unreachable);
    ``run_exited(run_id, reason)`` / ``live_od_reset()`` -> liveOD's RUN_EXITED /
    RESET; ``fence()`` -> the server's run fence or None; ``monitor_state()`` ->
    the server's ``status_json`` state; ``server_busy()`` -> why something of the
    server's own holds the machine ("" when nothing); ``loops`` -> the server's
    run loops by key; ``start_monitor(why)``; ``spawn(command, cwd=, env=,
    log_path=)`` -> a process with ``pid``, ``started`` and ``poll()``
    (default: :func:`~waxx.util.device_state.detached.launch`);
    ``adopt(pid, started)`` -> such a process or None; ``journal`` (the ops
    journal); ``on_change(info)``; ``exclude``: files the queue refuses (the
    monitor experiment)."""

    def __init__(self, directory: str | None, *, poll: Callable[[], dict] | None = None,
                 run_exited: Callable[[int, str], dict] | None = None,
                 live_od_reset: Callable[[], dict] | None = None,
                 fence: Callable[[], dict | None] | None = None,
                 monitor_state: Callable[[], object] | None = None,
                 server_busy: Callable[[], str] | None = None,
                 loops: Mapping | None = None,
                 start_monitor: Callable[[str], None] | None = None,
                 hold: PersonHold | None = None, journal=None,
                 on_change: Callable[[dict], None] | None = None,
                 spawn=None, adopt=None, exclude=(), roots=None,
                 clock: Callable[[], float] = time.time,
                 poll_every_s: float = POLL_EVERY_S, gap_s: float = GAP_S,
                 alarm_s: float = ALARM_S, outcome_wait_s: float = 0.5,
                 client_name: str | None = None):
        self.directory = directory
        self._poll_fn = poll
        self._run_exited = run_exited
        self._live_od_reset = live_od_reset
        self._fence = fence
        self._monitor_state = monitor_state
        self._server_busy = server_busy
        self._loops = loops if loops is not None else {}
        self._start_monitor = start_monitor
        self.hold = hold if hold is not None else PersonHold(
            os.path.join(directory, "person_hold.json") if directory else None,
            journal=journal)
        self._journal = journal
        self._on_change = on_change
        if spawn is None:
            from waxx.util.device_state.detached import launch  # noqa: PLC0415
            spawn = launch
        if adopt is None:
            from waxx.util.device_state.detached import ProcessWatch  # noqa: PLC0415
            adopt = ProcessWatch.adopt
        self._spawn = spawn
        self._adopt = adopt
        self._exclude = {self._norm(p) for p in exclude if p}
        #: the folders a job's file must be inside (default_roots())
        self.roots = list(roots) if roots is not None else default_roots()
        self._clock = clock
        self.poll_every_s = float(poll_every_s)
        self._gap_s = float(gap_s)
        self.alarm_s = float(alarm_s)
        self._outcome_wait_s = float(outcome_wait_s)
        self._host = client_name if client_name is not None else socket.gethostname()

        self._lock = threading.RLock()
        self._tick_lock = threading.Lock()
        self._abort_lock = threading.Lock()
        #: queue.json snapshots: the last numbered, the last written (_save)
        self._save_lock = threading.Lock()
        self._save_seq = 0
        self._saved_seq = 0
        #: an unreadable queue.json that could not be moved aside is never overwritten
        self._no_save = False
        #: records waiting to be copied to the ops journal (flush_journal)
        self._mirror: deque = deque()
        self._mirror_lock = threading.Lock()
        self._jobs: dict[int, Job] = {}
        self._next_id = 1
        self._paused: dict[str, dict | None] = {"agent": None, "all": None}
        #: the loop the queue stopped and will start again: {"key", "path", "since"}
        self._resume_loop: dict | None = None
        #: the queue has launched a job since it last ran out: ask for the monitor then
        self._owe_monitor = False
        self._current: int | None = None
        self._procs: dict[int, object] = {}
        self._log_pos: dict[int, int] = {}
        self._log_partial: dict[int, str] = {}
        self._tails: dict[int, deque] = {}
        self._last_poll: dict | None = None
        self._last_poll_t: float | None = None
        self._last_end_t: float | None = None
        self._waiting = ""
        self._waived: tuple | None = None
        self._owed_since: float | None = None
        self._alarm: dict | None = None
        self._own_abort_ids: set = set()
        #: dependencies no longer kept in queue.json: their state from the journal
        self._dep_cache: dict[int, str] = {}
        #: the last gate's kind: "free" | "live" | "blocked" (see _gate)
        self._gate_kind = ""
        #: the run id of the job that ended last (its fence may still be up)
        self._last_ended_run_id: int | None = None
        #: the launch in progress: {"job", "thread", "result", "since", "warned"}
        self._spawning: dict | None = None
        #: how long _launch waits for its launch thread before the tick follows it
        self._spawn_join_s = 5.0
        self._load()
        self.flush_journal()

    # -- paths ----------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.directory)

    def _path(self, *parts) -> str:
        return os.path.join(self.directory, *parts)

    @staticmethod
    def _norm(path) -> str:
        try:
            return os.path.normcase(str(Path(path).resolve()))
        except OSError:
            return os.path.normcase(str(path))

    # -- requests -------------------------------------------------------------------

    def submit(self, obj: Mapping) -> dict:
        """``{"path", "argv", "cwd", "label", "owner", "priority", "due",
        "after", "repeat", "chain", "stop_on_failure", "write_back",
        "allow_drift", "by"}`` -> ``{"status": "ok", "jobs": [...], "ids": [...]}``."""
        try:
            jobs = self._new_jobs(obj)
        except QueueError as exc:
            self._record("run_queue_refused", what="submit", msg=str(exc),
                         path=str(obj.get("path") or ""), by=str(obj.get("by") or ""))
            log.warning("Run queue: submit refused: %s", exc)
            self.flush_journal()
            return {"status": "error", "msg": str(exc)}
        for job in jobs:
            log.info("Run queue: %s submitted by %s (%s, owner %s, priority %d%s).", job.name,
                     job.submitted_by or "?", job.path, job.owner, job.priority,
                     f", due {_clock_text(job.due)}" if job.due else "")
            self._record("run_queue_submit", job=job.id, label=job.label, path=job.path,
                         sha256=job.sha256, argv=job.argv, cwd=job.cwd, owner=job.owner,
                         priority=job.priority, due=job.due, after=job.after, chain=job.chain,
                         stop_on_failure=job.stop_on_failure, write_back=job.write_back,
                         allow_drift=job.allow_drift, repeat_index=job.repeat_index,
                         repeat_of=job.repeat_of, by=job.submitted_by)
        self._save()
        self._notify()
        # TODO(kq client): `ar <file>` / `kq submit` attach here, then follow the
        # job by `describe` and tail its log_path (phase 1, client part).
        return {"status": "ok", "ids": [j.id for j in jobs],
                "jobs": [j.to_dict() for j in jobs]}

    def _new_jobs(self, obj: Mapping) -> list[Job]:
        if not self.enabled:
            raise QueueError("this monitor server has no run queue folder (no journal_dir / "
                             "run_queue_dir was given): jobs cannot be logged")
        text = str(obj.get("path") or "").strip()
        if not text:
            raise QueueError("no experiment file (path) was given")
        bad = sorted(_SHELL_CHARS & set(text))
        if bad:
            raise QueueError(f"the file's path contains {' '.join(bad)} (not allowed): {text}")
        path = Path(text)
        if not path.is_absolute():
            raise QueueError(f"the path must be absolute (the server's machine): {text}")
        path = path.resolve()
        bad = sorted(_SHELL_CHARS & set(str(path)))
        if bad:
            # a link resolving to such a path: the command line would carry it
            raise QueueError(f"the file's resolved path contains {' '.join(bad)} (not "
                             f"allowed): {path}")
        roots = [Path(r).resolve() for r in self.roots]
        if not any(path == r or r in path.parents for r in roots):
            raise QueueError(f"{path} is outside the folders the queue runs files from ("
                             + ", ".join(str(r) for r in roots) + ")")
        if path.suffix.lower() != ".py":
            raise QueueError(f"not a Python file: {path}")
        if not path.is_file():
            raise QueueError(f"no such file on the monitor server's machine: {path}")
        if self._norm(path) in self._exclude:
            raise QueueError(f"{path.name} is the monitor's own experiment: the monitor server "
                             "runs it itself")
        argv = obj.get("argv") or []
        if not isinstance(argv, (list, tuple)) or not all(isinstance(a, str) for a in argv):
            raise QueueError("argv must be a list of strings")
        for a in argv:
            bad = sorted(_SHELL_CHARS & set(a))
            if bad:
                raise QueueError(f"an argument contains {' '.join(bad)} (not allowed): {a}")
            if a.endswith("\\"):
                # quoted, it would escape its own closing quote
                raise QueueError(f"an argument ends in a backslash (not allowed): {a}")
        cwd = str(obj.get("cwd") or path.parent)
        if not Path(cwd).is_dir():
            raise QueueError(f"no such working folder on the monitor server's machine: {cwd}")
        owner = str(obj.get("owner") or "person")
        if owner not in OWNERS:
            raise QueueError(f"owner must be one of {', '.join(OWNERS)}, not {owner!r}")
        try:
            # a person's `ar` goes to the front: priority 10 unless given
            priority = (int(obj.get("priority")) if obj.get("priority") is not None
                        else PERSON_PRIORITY if str(obj.get("owner") or "person") == "person"
                        else 0)
        except (TypeError, ValueError):
            raise QueueError(f"priority must be an integer, not {obj.get('priority')!r}")
        due = obj.get("due")
        if due is not None:
            try:
                due = float(due)
            except (TypeError, ValueError):
                raise QueueError(f"due must be epoch seconds or null, not {due!r}")
            if due != due or due in (float("inf"), float("-inf")):
                raise QueueError(f"due must be a finite time, not {obj.get('due')!r}")
        after = obj.get("after") or []
        if not isinstance(after, (list, tuple)):
            after = [after]
        try:
            after = [int(a) for a in after]
        except (TypeError, ValueError):
            raise QueueError(f"after must be a list of job ids, not {obj.get('after')!r}")
        try:
            repeat = 1 if obj.get("repeat") is None else int(obj.get("repeat"))
        except (TypeError, ValueError):
            raise QueueError(f"repeat must be an integer, not {obj.get('repeat')!r}")
        if not 1 <= repeat <= MAX_REPEAT:
            raise QueueError(f"repeat must be 1 to {MAX_REPEAT}, not {repeat}")
        chain = obj.get("chain")
        chain = str(chain).strip() if chain not in (None, "") else None
        write_back = obj.get("write_back")
        if write_back is True:
            raise QueueError("write_back true is not a submitter's to give: the experiment "
                             "declares its write-back; a submitter may only veto it "
                             "(write_back false)")
        if write_back not in (None, False):
            raise QueueError(f"write_back must be null or false, not {write_back!r}")
        label = _LABEL_RE.sub("_", str(obj.get("label") or path.stem)).strip("_") or path.stem
        label = label[:60]
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "")
        sha = file_sha256(path)
        with self._lock:
            unknown = [a for a in after if a not in self._jobs
                       and self._dep_state(a) == "failed" and a >= self._next_id]
            if unknown:
                raise QueueError(f"after names unknown job(s): {', '.join(map(str, unknown))}")
            first = self._next_id
            if chain is None and repeat > 1:
                chain = f"repeat-{first}"
            stop = obj.get("stop_on_failure")
            stop = bool(chain) if stop is None else bool(stop)
            now = self._clock()
            jobs = []
            for i in range(repeat):
                job = Job(id=self._next_id, token=secrets.token_hex(3), path=str(path),
                          sha256=sha, label=label, argv=list(argv), cwd=cwd, owner=owner,
                          priority=priority, due=due, after=list(after), chain=chain,
                          stop_on_failure=stop, write_back=write_back,
                          allow_drift=bool(obj.get("allow_drift")), repeat_index=i + 1,
                          repeat_of=repeat, submitted_at=now, submitted_by=by)
                self._next_id += 1
                self._jobs[job.id] = job
                jobs.append(job)
        return jobs

    def _find(self, obj: Mapping) -> Job:
        try:
            job_id = int(obj.get("id"))
        except (TypeError, ValueError):
            raise QueueError(f"no job id given (id: {obj.get('id')!r})")
        job = self._jobs.get(job_id)
        if job is None:
            raise QueueError(f"no job {job_id} in the run queue")
        token = obj.get("token")
        if token and token != job.token:
            raise QueueError(f"job {job_id} has token {job.token}, not {token} (a different "
                             "queue's job?)")
        return job

    def cancel(self, obj: Mapping) -> dict:
        """``{"id", "token"?, "by", "owner"}``: a queued job is cancelled; a
        launching or running one gets liveOD's Abort for its run (never a
        kill) and ends when its process does.  ``owner`` -- who asks, "person"
        or "agent" -- is required: an agent may not cancel a person's job."""
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        with self._lock:
            try:
                job = self._find(obj)
            except QueueError as exc:
                return {"status": "error", "msg": str(exc)}
            if job.owner == "person" and as_owner == "agent" and job.state not in ENDED:
                return {"status": "error",
                        "msg": f"{job.name} is a person's job: an agent may not cancel it"}
            if job.state == "queued":
                self._end(job, "cancelled", f"cancelled by {by}")
                reply = {"status": "ok", "job": job.to_dict()}
            elif job.state in ("launching", "running"):
                if job.cancel is None:
                    job.cancel = {"by": by, "at": self._clock(), "abort_sent": False,
                                  "abort_note": ""}
                    self._record("run_queue_cancel_requested", job=job.id, by=by,
                                 run_id=job.run_id)
                    log.warning("Run queue: cancel of running %s asked by %s: liveOD's Abort "
                                "goes to run %s (its file is discarded, as for any Abort); "
                                "the job ends when its process does.", job.name, by,
                                job.run_id if job.run_id is not None else "(no run id yet)")
                # the tick sends liveOD's Abort (and retries): a request never
                # waits on liveOD
                reply = {"status": "ok", "job": job.to_dict(), "pending": True}
            else:
                return {"status": "error", "msg": f"{job.name} is already {job.state}"}
        self._save()
        self._notify()
        return reply

    def list(self, obj: Mapping | None = None) -> dict:
        """``{"states"?: [...], "limit"?: n}`` -> the jobs, newest last."""
        obj = obj or {}
        states = obj.get("states")
        try:
            limit = int(obj.get("limit") or 200)
        except (TypeError, ValueError):
            limit = 200
        with self._lock:
            jobs = [j for j in sorted(self._jobs.values(), key=lambda j: j.id)
                    if not states or j.state in states]
            order = [j.id for j in self._order(self._clock())]
        return {"status": "ok", "jobs": [j.to_dict() for j in jobs[-limit:]],
                "next": order, "run_queue": self.info()}

    def describe(self, obj: Mapping) -> dict:
        """``{"id", "token"?}`` -> the job, why it waits, its last lines."""
        with self._lock:
            try:
                job = self._find(obj)
            except QueueError as exc:
                return {"status": "error", "msg": str(exc)}
            out = {"status": "ok", "job": job.to_dict(),
                   "waiting": self._why_waiting(job, self._clock()),
                   "tail": list(self._tails.get(job.id, ()))}
        return out

    def pause(self, obj: Mapping) -> dict:
        """``{"scope": "agent" | "all", "by", "reason", "owner"}`` (``owner``:
        who pauses, "person" when absent; stored -- an agent may not resume a
        person's pause)."""
        scope = str(obj.get("scope") or "agent")
        if scope not in PAUSE_SCOPES:
            return {"status": "error", "msg": f"scope must be one of {', '.join(PAUSE_SCOPES)}"}
        owner = str(obj.get("owner") or "person")
        if owner not in OWNERS:
            return {"status": "error", "msg": f"owner must be one of {', '.join(OWNERS)}"}
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        reason = str(obj.get("reason") or "")
        with self._lock:
            held = self._paused.get(scope)
            if held and held.get("owner", "person") == "person" and owner == "agent":
                return {"status": "error",
                        "msg": f"{scope} jobs are already paused by a person ({held.get('by')})"
                               ": an agent may not replace that pause"}
            self._paused[scope] = {"by": by, "since": self._clock(), "reason": reason,
                                   "owner": owner}
        log.warning("Run queue: %s jobs paused by %s%s.", "agent" if scope == "agent" else "all",
                    by, f": {reason}" if reason else "")
        self._record("run_queue_pause", scope=scope, by=by, reason=reason, owner=owner)
        self._save()
        self._notify()
        return {"status": "ok", "run_queue": self.info()}

    def resume(self, obj: Mapping) -> dict:
        scope = str(obj.get("scope") or "agent")
        if scope not in PAUSE_SCOPES:
            return {"status": "error", "msg": f"scope must be one of {', '.join(PAUSE_SCOPES)}"}
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        with self._lock:
            paused = self._paused.get(scope)
            if paused is None:
                return {"status": "error", "msg": f"{scope} jobs are not paused"}
            if paused.get("owner", "person") == "person" and as_owner == "agent":
                return {"status": "error",
                        "msg": f"{scope} jobs were paused by a person ({paused.get('by')}): an "
                               "agent may not resume them"}
            self._paused[scope] = None
        log.info("Run queue: %s jobs resumed by %s.", scope, by)
        self._record("run_queue_resume", scope=scope, by=by, owner=as_owner)
        self._save()
        self._notify()
        return {"status": "ok", "run_queue": self.info()}

    def hold_request(self, obj: Mapping) -> dict:
        """``{"reason", "by", "owner"}`` (``owner`` "person" when absent;
        stored -- an agent may not release a person's hold)."""
        owner = str(obj.get("owner") or "person")
        if owner not in OWNERS:
            return {"status": "error", "msg": f"owner must be one of {', '.join(OWNERS)}"}
        reply = self.hold.hold(str(obj.get("reason") or ""),
                               str(obj.get("by") or obj.get("operator") or obj.get("client")
                                   or ""), owner=owner)
        self._notify()
        return reply

    def release_request(self, obj: Mapping) -> dict:
        """``{"by", "owner"}``; ``owner`` -- who asks -- is required."""
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        reply = self.hold.release(str(obj.get("by") or obj.get("operator")
                                      or obj.get("client") or ""), owner=as_owner)
        self._notify()
        return reply

    # -- state ----------------------------------------------------------------------

    def eligible_or_running(self) -> str:
        """"" unless a job is in the slot (launching, running, ending) or one
        is eligible to launch now; else which.  Queued jobs that are due
        later, held, paused or waiting on another job do not count: they
        leave a loop's Start alone."""
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            if cur is not None:
                return f"the run queue's {cur.name} is {cur.state}"
            order = self._order(self._clock())
        if order:
            return f"the run queue's {order[0].name} is about to launch"
        return ""

    def monitor_busy(self) -> str:
        """Why a monitor (re)start must wait ("" when it need not): a job is
        in the slot, or one is eligible AND the queue's last gate found the
        machine free or legitimately in use (it launches as soon as that run
        ends).  A blocked gate (liveOD unreachable, a wedged run, the server
        busy...) never defers it: the hardware must not be left without the
        monitor while the queue cannot launch anyway."""
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            if cur is not None:
                return f"the run queue's {cur.name} is {cur.state}"
            order = self._order(self._clock())
            kind = self._gate_kind
        if order and kind in ("free", "live"):
            return f"the run queue's {order[0].name} is about to launch"
        return ""

    def defer_monitor(self, what: str, why: str) -> None:
        """A monitor (re)start asked for while the queue has work: not done
        now; the queue asks for the monitor when it runs out."""
        with self._lock:
            self._owe_monitor = True
        log.warning("Run queue: %s deferred -- %s; the monitor starts when the queue runs out.",
                    what, why)
        self._record("run_queue_monitor_deferred", what=what, why=why)
        self.flush_journal()

    def current_job(self) -> dict | None:
        """The job in the slot (launching / running / ending), or None."""
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            return cur.to_dict() if cur is not None else None

    def has_work(self) -> bool:
        """Jobs queued or in the slot (a loop's Start is refused then)."""
        with self._lock:
            return any(j.state in ("queued",) + IN_SLOT for j in self._jobs.values())

    def work_text(self) -> str:
        with self._lock:
            queued = sum(j.state == "queued" for j in self._jobs.values())
            cur = self._jobs.get(self._current) if self._current is not None else None
        parts = []
        if cur is not None:
            parts.append(f"{cur.name} is {cur.state}")
        if queued:
            parts.append(f"{queued} job{'s' if queued != 1 else ''} queued")
        return " and ".join(parts)

    def agent_run_ids(self) -> set:
        with self._lock:
            return {j.run_id for j in self._jobs.values()
                    if j.owner == "agent" and j.run_id is not None}

    def own_abort_ids(self) -> set:
        with self._lock:
            return set(self._own_abort_ids)

    def own_run_ids(self) -> set:
        with self._lock:
            return {j.run_id for j in self._jobs.values() if j.run_id is not None}

    def info(self) -> dict:
        """``status_json["run_queue"]``."""
        with self._lock:
            now = self._clock()
            counts = {s: 0 for s in STATES}
            for j in self._jobs.values():
                counts[j.state] += 1
            cur = self._jobs.get(self._current) if self._current is not None else None
            order = self._order(now)
            paused = {k: (dict(v) if v else None) for k, v in self._paused.items()}
            hold = self.hold.info()
            if cur is not None:
                state = "running"
                text = f"{cur.name} {cur.state}" + (f", run {cur.run_id}" if cur.run_id else "")
            elif order:
                state = "waiting"
                text = f"{order[0].name} next: {self._waiting or 'launching'}"
            elif counts["queued"]:
                state = "held" if hold.get("active") else (
                    "paused" if any(paused.values()) else "waiting")
                text = f"{counts['queued']} queued, none can start now"
            else:
                state, text = "idle", "no jobs"
            return {"enabled": self.enabled, "state": state, "text": text,
                    "current": cur.to_dict() if cur is not None else None,
                    "next": [j.id for j in order[:20]], "counts": counts,
                    "waiting": self._waiting, "alarm": dict(self._alarm) if self._alarm else None,
                    "paused": paused, "person_hold": hold,
                    "resume_loop": dict(self._resume_loop) if self._resume_loop else None,
                    "directory": self.directory}

    # -- scheduling -------------------------------------------------------------------

    def _blocked_by(self, job: Job, now: float) -> str:
        """Why a queued job may not go now ("" when it is eligible) -- the
        job's own conditions, not the machine's."""
        if job.state != "queued":
            return f"it is {job.state}"
        if job.due is not None and job.due > now:
            return f"due at {_clock_text(job.due)}"
        waits = [a for a in job.after if self._dep_state(a) != "saved"]
        if waits:
            return "waiting for job " + ", ".join(map(str, waits))
        if self._paused.get("all"):
            p = self._paused["all"]
            return f"all jobs paused by {p.get('by')}"
        if job.owner == "agent" and self._paused.get("agent"):
            p = self._paused["agent"]
            return f"agent jobs paused by {p.get('by')}"
        if job.owner == "agent" and self.hold.active:
            return self.hold.text()
        return ""

    def _dep_state(self, job_id: int) -> str:
        """A dependency's state: its record, or -- for a job no longer kept in
        queue.json (beyond KEEP_ENDED, or lost with an unreadable file) -- its
        last end in the journal; a job found nowhere counts as "failed" (its
        dependents are skipped, never left waiting)."""
        dep = self._jobs.get(job_id)
        if dep is not None:
            return dep.state
        cached = self._dep_cache.get(job_id)
        if cached is None:
            cached = "failed"
            try:
                with open(self._path("journal.jsonl"), "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if rec.get("kind") == "run_queue_end" and rec.get("job") == job_id:
                            cached = str(rec.get("state") or "failed")
            except (OSError, TypeError):
                pass
            self._dep_cache[job_id] = cached
        return cached

    def _order(self, now: float) -> list[Job]:
        """The eligible jobs, the next one first."""
        eligible = [j for j in self._jobs.values()
                    if j.state == "queued" and not self._blocked_by(j, now)]
        return sorted(eligible, key=lambda j: (j.priority, -(j.due or 0.0), -j.id),
                      reverse=True)

    def _why_waiting(self, job: Job, now: float) -> str:
        if job.state != "queued":
            return ""
        own = self._blocked_by(job, now)
        if own:
            return own
        if self._current is not None:
            return f"job {self._current} is in the slot"
        order = self._order(now)
        if order and order[0].id != job.id:
            return f"job {order[0].id} goes first"
        return self._waiting or "launching"

    def tick(self) -> None:
        """One step: follow the job in the slot, settle dependencies, stop or
        restart a loop, launch the next job when the machine is free, and the
        alarm.  Called by one thread: the monitor server's watch, every
        ``WATCH_S`` (0.5 s).  liveOD is polled at most every
        ``poll_every_s`` (:data:`POLL_EVERY_S`, 2 s) for the hold's watch and
        the slot's job, and afresh before a launch, an Abort and an outcome."""
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            now = self._clock()
            if self._last_poll_t is None or now - self._last_poll_t >= self.poll_every_s:
                self._poll_now()
            self.hold.tick()
            if self._current is None:
                self._take_leftover()
            cur = self._jobs.get(self._current) if self._current is not None else None
            if cur is not None:
                self._follow(cur)
            self._settle(now)
            if self._current is None:
                self._schedule(self._clock())
            self._alarm_tick(self._clock())
        finally:
            self._tick_lock.release()
            self.flush_journal()

    def _poll_now(self) -> dict | None:
        """A fresh POLL (fed to the person hold's watch); None when liveOD does
        not answer."""
        self._last_poll_t = self._clock()
        if self._poll_fn is None:
            return None
        try:
            poll = self._poll_fn()
        except Exception:                             # noqa: BLE001
            self._last_poll = None
            return None
        self._last_poll = poll
        self.hold.observe_poll(poll, self.agent_run_ids(), self.own_abort_ids(),
                               on_person_reset=self._claim_person_reset)
        return poll

    def _claim_person_reset(self, last: dict) -> bool:
        """A person pressed liveOD's Reset while the job in the slot has no
        run id yet (it is launching or preparing; liveOD had no run, so the
        Reset names none).  The person's own job: taken as that job's cancel
        -- liveOD would spend that Abort on no run and let the run through, so
        the queue sends its own Abort once the run has an id -- and no hold.
        Anyone else's job (or a Reset naming a run): not claimed, the hold
        goes on as for any person's Reset."""
        if (last or {}).get("run_id") is not None:
            return False
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            if (cur is None or cur.run_id is not None or cur.owner != "person"
                    or cur.state not in ("launching", "running")):
                return False
            if cur.cancel is None:
                cur.cancel = {"by": "a person's Reset in liveOD", "at": self._clock(),
                              "abort_sent": False,
                              "abort_note": "Reset pressed before the run had an id: the queue "
                                            "sends liveOD's Abort once it has one"}
        log.warning("Run queue: a person pressed Reset in liveOD while %s (a person's job) had "
                    "no run yet: taken as its cancel.", cur.name)
        self._record("run_queue_cancel_requested", job=cur.id, by="a person's Reset in liveOD",
                     run_id=None, via="live_od_reset")
        self._save()
        return True

    def _settle(self, now: float) -> None:
        """Skip jobs waiting for a job that will never be saved."""
        changed = True
        while changed:
            changed = False
            with self._lock:
                for job in sorted(self._jobs.values(), key=lambda j: j.id):
                    if job.state != "queued":
                        continue
                    for a in job.after:
                        state = self._dep_state(a)
                        if state in ENDED and state != "saved":
                            dep = self._jobs.get(a)
                            why = (dep.reason if dep is not None else
                                   "no longer in queue.json; from the journal")
                            self._end(job, "skipped", f"job {a} ended {state}"
                                      + (f" ({why})" if why else ""))
                            changed = True
                            break
        if changed:
            self._save()

    def _schedule(self, now: float) -> None:
        with self._lock:
            order = self._order(now)
        loop = self._active_loop()
        if order:
            if loop is not None:
                stopper = next((j for j in order if self._may_stop(j, loop)), None)
                if stopper is not None:
                    self._stop_loop_for_jobs(loop, stopper)
            if self._last_end_t is not None and now - self._last_end_t < self._gap_s:
                return
            why = self._gate()
            if why:
                if why != self._waiting:
                    log.info("Run queue: %s waits: %s", order[0].name, why)
                self._waiting = why
                return
            self._waiting = ""
            for job in order:
                if self._launch(job):          # else skipped at launch: try the next
                    return
            return
        self._waiting = ""
        self._drained(loop)

    def _active_loop(self):
        for loop in self._loops.values():
            if getattr(loop, "active", False):
                return loop
        return None

    def _own_ended_fence(self, fence: dict) -> bool:
        """The run fence is the one the queue's last ended job left up (it
        exited without lifting it): only that job's run id, and only a real
        one (a positive int -- every save_data=False run is run id 0)."""
        fid = fence.get("run_id")
        last = self._last_ended_run_id
        return (isinstance(fid, int) and not isinstance(fid, bool) and fid > 0
                and fid == last)

    def _gate(self) -> str:
        """Why the next job may not launch now ("" when it may).  Sets
        ``_gate_kind``: "free", "live" (someone's run is legitimately using
        the machine: a run in progress or announced, a loop finishing its run
        after a stop) or "blocked" (anything else -- wedged, an Abort pending,
        liveOD unreachable or unknown, the server busy, a loop that does not
        stop): only "blocked" runs the alarm's clock."""
        self._gate_kind = "blocked"
        busy = self._server_busy() if self._server_busy is not None else ""
        if busy:
            return busy
        loop = self._active_loop()
        if loop is not None:
            info = loop.info()
            if info.get("state") == "stopping":
                self._gate_kind = "live"
                return (f"{loop.spec.title} is finishing its run in progress (asked to stop "
                        "for the queue)")
            if (info.get("owner") or "person") == "person":
                self._gate_kind = "live"          # a person's loop: legitimate use
                return (f"{loop.spec.title} was started by a person "
                        f"({info.get('operator') or info.get('client') or '?'}): an agent's "
                        "job does not stop it")
            return f"{loop.spec.title} is running and has not been asked to stop"
        poll = self._poll_now()
        if poll is None:
            return "liveOD is not reachable -- a run could not save"
        fence = self._fence() if self._fence is not None else None
        if fence and self._own_ended_fence(fence):
            fence = None                  # the fence of the queue's own job that just ended
        monitor = self._monitor_state() if self._monitor_state is not None else None
        verdict = run_gate.classify(poll, fence, monitor_state=monitor)
        if verdict.state == "free":
            self._gate_kind = "free"
            return ""
        if verdict.state == "live":
            self._gate_kind = "live"
        if verdict.waivable:
            self._gate_kind = "free"
            key = (verdict.run_id, verdict.state)
            if key != self._waived:
                self._waived = key
                log.warning("Run queue: run %s is %s in liveOD and is waived: %s",
                            verdict.run_id, verdict.state, verdict.reason)
                self._record("run_queue_waived", run_id=verdict.run_id, state=verdict.state,
                             reason=verdict.reason)
            return ""
        return verdict.reason

    # -- launch -----------------------------------------------------------------------

    def _launch(self, job: Job) -> bool:
        """Launch ``job``; False when it was skipped at launch instead (the
        next eligible job may go), True otherwise.

        The job is saved as ``launching`` (with its launch time) BEFORE its
        process is started, and the process is started on a thread of its own
        outside the queue's lock: the tick follows the launch (:meth:`_follow`),
        a hung launcher stalls nothing else, and a server that dies meanwhile
        finds a ``launching`` job at restart -- never launched again, looked
        for in liveOD instead (:meth:`_follow_orphan`)."""
        # TODO(phase 2): warm-ahead / GO handshake and liveOD RESERVE/START
        # attach here (run-queue plan); nothing of it in phase 1.
        try:
            sha = file_sha256(job.path)
        except OSError as exc:
            sha, unreadable = None, exc
        else:
            unreadable = None
        with self._lock:
            if job.state != "queued":                  # cancelled meanwhile
                return False
            if unreadable is not None:
                self._end(job, "skipped", f"the file cannot be read at launch ({unreadable})")
                skipped = True
            elif sha != job.sha256 and not job.allow_drift:
                self._end(job, "skipped", "source changed since submit")
                skipped = True
            else:
                skipped = False
                log_path = self._path("logs", f"{job.id}_{job.label}.out")
                command = ar_command(_quote(job.path))
                if job.argv:
                    command += " " + " ".join(_quote(a) for a in job.argv)
                job.state = "launching"
                job.launched_at = self._clock()
                job.log_path = log_path
                self._current = job.id
                self._procs[job.id] = None
                self._log_pos[job.id] = 0
                self._log_partial[job.id] = ""
                self._tails[job.id] = deque(maxlen=TAIL_LINES)
                self._owe_monitor = True
                self._owed_since = None
        if skipped:
            self._save()
            self._notify()
            return False
        self._save()                                   # "launching" is on disk first
        self._record("run_queue_launching", job=job.id, command=command, log_path=log_path,
                     owner=job.owner, drift=sha != job.sha256)
        # unbuffered: output reaches the log as printed; UTF-8: the log is read
        # as UTF-8, and a console code page cannot fail a print of "µs" or "─"
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env[run_gate.LAUNCHER_ENV] = LAUNCHER
        env[JOB_ENV] = job.queue_job
        env[OWNER_ENV] = job.owner
        if job.write_back is False:
            env[NO_WRITE_BACK_ENV] = "1"
        else:
            env.pop(NO_WRITE_BACK_ENV, None)
        try:
            os.makedirs(self._path("logs"), exist_ok=True)
            with open(log_path, "ab") as f:
                f.write((f"── run queue {time.strftime('%Y-%m-%d %H:%M:%S')}: {job.name}, "
                         f"owner {job.owner}, sha256 {sha[:12]}"
                         + (" (drift allowed: file changed since submit)"
                            if sha != job.sha256 else "")
                         + f" ──\n$ {command}\n").encode("utf-8"))
        except OSError as exc:
            self._launch_failed(job, OSError(f"cannot write its log {log_path}: {exc}"))
            return True
        result: dict = {}
        record: dict = {}

        def spawn():
            try:
                proc = self._spawn(command, cwd=job.cwd, env=env, log_path=log_path)
            except BaseException as exc:              # noqa: BLE001
                result["error"] = exc
                return
            with self._lock:
                result["proc"] = proc
                abandoned = self._spawning is not record
            if abandoned:
                # the queue stopped waiting for this launch (it adopted the run
                # from liveOD, or gave the job up): its watch is not followed
                _close(proc)
        thread = threading.Thread(target=spawn, daemon=True, name=f"run-queue-launch-{job.id}")
        record.update({"job": job.id, "thread": thread, "result": result,
                       "since": self._clock(), "warned": False})
        self._spawning = record
        thread.start()
        thread.join(self._spawn_join_s)                # usually done at once
        self._follow_launch(job)
        if self._alarm is not None:
            self._clear_alarm("a job was launched")
        return True

    def _follow_launch(self, job: Job) -> None:
        """A ``launching`` job whose launch thread this server started: take
        its process when the thread is done."""
        sp = self._spawning
        if sp is None or sp["job"] != job.id:
            self._follow_orphan(job)
            return
        if sp["thread"].is_alive():
            waited = self._clock() - sp["since"]
            if waited > SPAWN_WARN_S and not sp["warned"]:
                sp["warned"] = True
                log.warning("Run queue: %s: its launch has not finished after %.0f s; the slot "
                            "stays taken (the job is looked for in liveOD).", job.name, waited)
            if waited <= LAUNCH_TIMEOUT_S + ORPHAN_WAIT_S:
                self._follow_orphan(job, keep_waiting=True)
                return
            # the launch thread never came back: stop waiting for it (a late
            # answer is closed by the thread itself) and treat the job as an
            # orphan -- found in liveOD, or ended failed now
            self._abandon_spawn()
            self._record("run_queue_launch_abandoned", job=job.id, waited_s=round(waited, 1))
            self._follow_orphan(job)
            return
        self._spawning = None
        result = sp["result"]
        if "proc" in result:
            self._launched(job, result["proc"])
            return
        exc = result.get("error")
        from waxx.util.device_state.detached import LaunchUnknown  # noqa: PLC0415
        if isinstance(exc, LaunchUnknown):
            # it may be running: not failed, not launched again -- looked for
            log.error("Run queue: %s: %s", job.name, exc)
            self._record("run_queue_launch_unknown", job=job.id, error=str(exc))
            self._follow_orphan(job)
            return
        self._launch_failed(job, exc)

    def _abandon_spawn(self) -> None:
        """Stop waiting for the launch thread: a watch it already returned is
        closed here, one it returns later is closed by the thread."""
        with self._lock:
            sp, self._spawning = self._spawning, None
            proc = (sp or {}).get("result", {}).get("proc")
        if proc is not None:
            _close(proc)

    def _launched(self, job: Job, proc) -> None:
        with self._lock:
            if job.state != "launching":
                # ended meanwhile (an orphan check found its run's outcome):
                # the process is not followed twice
                close = True
            else:
                close = False
                job.state = "running"
                job.pid = int(proc.pid)
                job.pid_started = getattr(proc, "started", None)
                self._procs[job.id] = proc
        if close:
            _close(proc)
            return
        log.info("Run queue: %s launched (pid %s).", job.name, job.pid)
        self._record("run_queue_launch", job=job.id, pid=job.pid, log_path=job.log_path,
                     owner=job.owner)
        self._save()
        self._notify()

    def _launch_failed(self, job: Job, exc) -> None:
        log.error("Run queue: could not start %s: %r", job.name, exc)
        for line in environment_report():
            log.error("  %s", line)
        with self._lock:
            if job.state == "launching":
                self._end(job, "failed", f"could not start: {exc!r}")
            self._current = None
            self._procs.pop(job.id, None)
            self._last_end_t = self._clock()
        self._save()
        self._notify()

    def _follow_orphan(self, job: Job, keep_waiting: bool = False) -> None:
        """A ``launching`` job with no process to follow: the server stopped
        while launching it, or its launcher never answered.  It is never
        launched again.  Its run is looked for in liveOD (launcher "kq",
        queue_job = "<id>:<token>"): a run in progress is adopted (its experiment's
        pid), a recorded outcome judges it; after :data:`ORPHAN_WAIT_S` with
        neither it ends failed ("server stopped while launching -- check")."""
        poll = self._last_poll
        if self._is_job_run(job, poll) and poll.get("run_in_progress"):
            proc = None
            try:
                proc = self._adopt(poll.get("client_pid"), None)
            except Exception:                         # noqa: BLE001
                proc = None
            if proc is not None:
                with self._lock:
                    job.run_id = int(poll["run_id"])
                    job.client_pid = poll.get("client_pid")
                    job.adopted = True
                if keep_waiting:
                    self._abandon_spawn()          # the launch thread's answer is not needed
                log.warning("Run queue: %s was launching; its run %s is in liveOD (pid %s): "
                            "followed.", job.name, job.run_id, job.client_pid)
                self._record("run_queue_adopted", job=job.id, pid=job.client_pid,
                             run_id=job.run_id, via="liveOD")
                self._launched(job, proc)
                return
        last = poll.get("last_outcome") if isinstance(poll, dict) else None
        if self._is_job_run(job, last):
            with self._lock:
                job.run_id = int(last["run_id"])
                job.state = "running"
                self._procs[job.id] = None
            self._finish(job, None, -1)
            return
        if keep_waiting:
            return
        started = job.launched_at or job.submitted_at
        if self._clock() - float(started or 0) < ORPHAN_WAIT_S:
            return
        with self._lock:
            if job.state != "launching":
                return
            self._end(job, "failed", "server stopped while launching -- check (no run of this "
                                     f"job appeared in liveOD within {ORPHAN_WAIT_S:.0f} s)")
            self._current = None
            self._last_end_t = self._clock()
        log.error("Run queue: %s ended failed: it was launching when the server stopped (or "
                  "its launcher never answered) and no run of it appeared in liveOD. Check "
                  "that nothing of it is running.", job.name)
        self._save()
        self._notify()

    def _take_leftover(self) -> None:
        """A job left in the slot's states (after a restart several may be):
        the one with a live process first, then the lowest id."""
        with self._lock:
            left = [j for j in self._jobs.values() if j.state in IN_SLOT]
            if not left:
                return
            left.sort(key=lambda j: (self._procs.get(j.id) is None, j.id))
            self._current = left[0].id

    # -- following the job in the slot ---------------------------------------------------

    def _read_log(self, job: Job) -> list[str]:
        if not job.log_path:
            return []
        try:
            with open(job.log_path, "rb") as f:
                f.seek(self._log_pos.get(job.id, 0))
                data = f.read()
        except OSError:
            return []
        self._log_pos[job.id] = self._log_pos.get(job.id, 0) + len(data)
        text = self._log_partial.get(job.id, "") + data.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._log_partial[job.id] = lines.pop()
        return [ln.rstrip("\r") for ln in lines]

    def _take_lines(self, job: Job, lines, final: bool = False) -> None:
        if final and self._log_partial.get(job.id):
            lines = list(lines) + [self._log_partial.pop(job.id)]
        tail = self._tails.setdefault(job.id, deque(maxlen=TAIL_LINES))
        for line in lines:
            if not line.strip():
                continue
            tail.append(line)
            m = RUN_ID_RE.search(line)
            if m and job.run_id is None:
                with self._lock:
                    job.run_id = int(m.group(1))
                log.info("Run queue: %s is run %s.", job.name, job.run_id)
                self._record("run_queue_run_id", job=job.id, run_id=job.run_id, via="output")
                self._save()
                self._notify()

    @staticmethod
    def _is_job_run(job: Job, record) -> bool:
        """A liveOD POLL reply or last_outcome record is about ``job``'s run:
        launched by the queue, with this job's id AND token (``queue_job``
        "<id>:<token>": ids start again at 1 in another queue folder, so an id
        alone could match an old run's record; sent by the
        experiment at INIT_RUN from WAXX_QUEUE_JOB)."""
        return (isinstance(record, dict) and record.get("launcher") == LAUNCHER
                and str(record.get("queue_job") or "") == job.queue_job
                and bool(record.get("run_id")))

    def _identify(self, job: Job, poll) -> None:
        """The job's run id and experiment pid from liveOD (``queue_job``) --
        the primary way; the "Run ID:" line in its output is the fallback (an
        older liveOD or experiment that does not send ``queue_job``)."""
        if not self._is_job_run(job, poll):
            return
        changed = False
        with self._lock:
            if job.run_id is None:
                job.run_id = int(poll["run_id"])
                changed = True
            if job.client_pid is None and poll.get("client_pid"):
                job.client_pid = poll.get("client_pid")
                changed = True
        if changed:
            if job.run_id == poll.get("run_id"):
                self._record("run_queue_run_id", job=job.id, run_id=job.run_id, via="liveOD",
                             client_pid=job.client_pid)
            self._save()
            self._notify()

    def _follow(self, job: Job) -> None:
        if job.state == "launching":
            self._take_lines(job, self._read_log(job))
            self._identify(job, self._last_poll)
            self._follow_launch(job)
            return
        self._take_lines(job, self._read_log(job))
        poll = self._last_poll
        self._identify(job, poll)
        if (job.run_id is not None and job.client_pid is None and isinstance(poll, dict)
                and poll.get("run_id") == job.run_id and poll.get("client_pid")):
            with self._lock:
                job.client_pid = poll.get("client_pid")
            self._save()
        if job.cancel is not None and not job.cancel.get("abort_sent"):
            self._send_abort(job)
        proc = self._procs.get(job.id)
        code = proc.poll() if proc is not None else -1
        if code is None:
            return
        self._finish(job, proc, code)

    def _finish(self, job: Job, proc, code) -> None:
        with self._lock:
            job.state = "ending"
        self._save()
        self._notify()
        self._take_lines(job, self._read_log(job), final=True)
        if job.run_id is None:
            fresh = self._poll_now()
            if isinstance(fresh, dict):
                self._identify(job, fresh)
                self._identify(job, fresh.get("last_outcome"))
        known = proc is not None and getattr(proc, "exit_code_known", True) and code != -1
        exit_code = int(code) if known else None
        tail = list(self._tails.get(job.id, ()))
        stem = Path(job.path).stem
        told = None
        if self._run_exited is not None and self._poll_fn is not None:
            told = tell_live_od_exited(self._poll_fn, self._run_exited, job.run_id, exit_code,
                                       tail, expt=stem, started=job.launched_at,
                                       now=self._clock(), who=f"Run queue {job.name}",
                                       sender="the monitor server's run queue")
            if told is not None and told["status"] != "not_sent":
                self._record("run_queue_told_live_od_exited", job=job.id,
                             run_id=told["run_id"], ok=told["ok"], status=told["status"])
        if self._poll_fn is not None:
            v = judge_run(exit_code, tail, job.run_id, stem,
                          lambda rid: live_od_outcome(self._poll_fn, rid,
                                                      wait_s=self._outcome_wait_s))
        else:
            v = judge_run(exit_code, tail, job.run_id, stem, lambda rid: None)
        outcome = {"outcome": (v.outcome or {}).get("outcome"),
                   "detail": (v.outcome or {}).get("detail") or "", "why": v.why,
                   "exit_code": exit_code, "aborted": v.aborted,
                   "lost_core": not v.start_monitor,
                   "told_live_od": None if told is None else told["status"]}
        with self._lock:
            job.exit_code = exit_code
            job.outcome = outcome
            if v.saved:
                state, reason = "saved", ""
                if job.cancel is not None:
                    reason = (f"saved before the cancel by {job.cancel.get('by')} reached it"
                              + (f" ({job.cancel['abort_note']})"
                                 if job.cancel.get("abort_note") else ""))
            elif job.cancel is not None:
                state = "cancelled"
                reason = f"cancelled by {job.cancel.get('by')} while running: {v.why}"
            else:
                state, reason = "failed", v.why
            if proc is None and job.adopted is False and state == "failed":
                reason = f"server restarted; process gone ({v.why})"
            elif proc is None and state == "saved":
                reason = (f"its process ended while the monitor server was down; liveOD "
                          f"recorded run {job.run_id} saved")
            self._end(job, state, reason, stop_chain=state in ("failed", "cancelled"))
            self._current = None
            self._last_end_t = self._clock()
            self._last_ended_run_id = job.run_id
            self._procs.pop(job.id, None)
        if proc is not None and hasattr(proc, "close"):
            try:
                proc.close()
            except Exception:                         # noqa: BLE001
                pass
        (log.info if state == "saved" else log.warning)(
            "Run queue: %s %s%s%s", job.name, state if state == "saved" else state.upper(),
            f" (run {job.run_id})" if job.run_id is not None else "",
            f": {reason}" if reason else "")
        if state != "saved" and tail:
            for line in tail[-8:]:
                log.warning("    | %s", line)
        self._save()
        self._notify()

    def _send_abort(self, job: Job) -> None:
        """liveOD's Abort for a running job being cancelled -- only when
        liveOD's run in progress is that job's run, and it is not being saved.
        Sent once (the request thread and the tick may both try)."""
        with self._abort_lock:
            self._send_abort_once(job)

    def _cancel_note(self, job: Job, **fields) -> None:
        with self._lock:
            if job.cancel is not None:
                job.cancel.update(fields)

    def _send_abort_once(self, job: Job) -> None:
        with self._lock:
            pending = job.cancel is not None and not job.cancel.get("abort_sent")
        if not pending:
            return
        if job.run_id is None:
            self._cancel_note(job, abort_note="no run id yet: the Abort goes once the run has "
                                              "one")
            return
        poll = self._poll_now()
        if poll is None:
            self._cancel_note(job, abort_note="liveOD is not reachable: the Abort is not sent "
                                              "yet")
            return
        if not poll.get("run_in_progress") or poll.get("run_id") != job.run_id:
            current = poll.get("run_id") if poll.get("run_in_progress") else "none"
            self._cancel_note(job, abort_note=f"liveOD's run in progress is {current}, not "
                                              f"{job.run_id}: no Abort sent")
            return
        if poll.get("save_in_progress") or poll.get("run_state") == "saving":
            self._cancel_note(job, abort_note=f"liveOD is saving run {job.run_id}: no Abort "
                                              "sent")
            return
        if poll.get("reset_requested"):
            self._cancel_note(job, abort_sent=True, abort_note="an Abort was already pending")
            return
        if self._live_od_reset is None:
            self._cancel_note(job, abort_note="this queue has no way to send liveOD's Abort")
            return
        with self._lock:
            self._own_abort_ids.add(job.run_id)
        try:
            reply = self._live_od_reset(run_id=job.run_id, source="queue")
        except Exception as exc:                      # noqa: BLE001
            self._cancel_note(job, abort_note=f"sending the Abort failed: {exc}")
            log.error("Run queue: liveOD's Abort for run %s (%s) failed: %s", job.run_id,
                      job.name, exc)
            return
        ok = bool(isinstance(reply, dict) and reply.get("ok"))
        if isinstance(reply, dict) and reply.get("refused"):
            # liveOD's run is no longer this job's (it ended, or another
            # started): nothing was aborted; tried again while the job runs
            self._cancel_note(job, abort_note=f"liveOD refused the Abort: {reply.get('error')}")
            log.warning("Run queue: liveOD refused the Abort for run %s (%s): %s", job.run_id,
                        job.name, reply.get("error"))
            return
        self._cancel_note(job, abort_sent=True,
                          abort_note="Abort sent" if ok else f"liveOD refused: {reply}")
        log.warning("Run queue: liveOD's Abort sent for run %s (%s): %s", job.run_id, job.name,
                    "acknowledged" if ok else reply)
        self._record("run_queue_abort_sent", job=job.id, run_id=job.run_id, ok=ok,
                     by=job.cancel.get("by"))
        self._save()

    # -- ending jobs and chains ---------------------------------------------------------------

    def _end(self, job: Job, state: str, reason: str, stop_chain: bool | None = None) -> None:
        """Put a job in an ended state (under the lock), journal it, and stop
        its chain (``stop_chain`` None: when it failed or was skipped; a
        running job that was cancelled stops it too)."""
        job.state = state
        job.reason = reason
        job.ended_at = self._clock()
        self._record("run_queue_end", job=job.id, state=state, reason=reason,
                     run_id=job.run_id, exit_code=job.exit_code, outcome=job.outcome)
        if stop_chain is None:
            stop_chain = state in ("failed", "skipped")
        if stop_chain and job.chain and job.stop_on_failure:
            for other in sorted(self._jobs.values(), key=lambda j: j.id):
                if other.chain == job.chain and other.state == "queued" and other.id != job.id:
                    other.state = "cancelled"
                    other.reason = (f"chain {job.chain} stopped: {job.name} {state}"
                                    + (f" ({reason})" if reason else ""))
                    other.ended_at = job.ended_at
                    self._record("run_queue_end", job=other.id, state="cancelled",
                                 reason=other.reason, run_id=None, exit_code=None,
                                 outcome=None)

    # -- loops and the monitor ----------------------------------------------------------------

    @staticmethod
    def _may_stop(job: Job, loop) -> bool:
        """A person's job stops any loop; an agent's job only a loop the queue,
        or an agent (the TOF-idle restart), started -- never a person's."""
        if job.owner == "person":
            return True
        return (loop.info().get("owner") or "person") in ("queue", "agent")

    def loop_stopped_by_someone(self, key: str, who: str) -> None:
        """Someone (not the queue) asked a loop to stop: if the queue meant
        to start that loop again once it runs out, it no longer does."""
        with self._lock:
            resume = self._resume_loop
            if resume is None or resume.get("key") != key:
                return
            self._resume_loop = None
        log.info("Run queue: %s stopped the loop %s: the queue will not start it again.", who,
                 key)
        self._record("run_queue_loop_resume_cancelled", loop=key, by=who)
        self._save()
        self._notify()

    def _stop_loop_for_jobs(self, loop, job: Job) -> None:
        if loop.info().get("state") != "running":
            # already stopping: by the queue (asked before), or by a person --
            # whose stop it is, and who did not ask for it to start again
            return
        reply = loop.stop(operator="run queue", client=self._host, start_monitor=False)
        if reply.get("status") != "ok":
            return
        with self._lock:
            self._resume_loop = {"key": loop.spec.key,
                                 "path": loop.path if loop.spec.pick else None,
                                 "since": self._clock()}
        log.info("Run queue: %s stops after its run in progress for %s; it starts again "
                 "when the queue has run out.", loop.spec.title, job.name)
        self._record("run_queue_loop_stop", loop=loop.spec.key, for_job=job.id)
        self._save()
        self._notify()

    def _drained(self, active_loop) -> None:
        """No job eligible, none in the slot: start the loop the queue stopped
        (no hold, nothing paused), else ask for the monitor if the queue has
        run jobs since it last ran out."""
        if active_loop is not None:
            return
        resume = self._resume_loop
        blocked = ""
        if resume is not None:
            blocked = (self.hold.text() if self.hold.active else
                       ("jobs are paused" if any(self._paused.values()) else ""))
            if blocked:
                pass                                  # kept until the hold / pause lifts
            else:
                with self._lock:
                    self._resume_loop = None
                loop = self._loops.get(resume.get("key"))
                info = loop.info() if loop is not None else {}
                text = str(info.get("text") or "")
                if loop is None:
                    self._record("run_queue_loop_not_resumed", loop=resume.get("key"),
                                 why="the server has no such loop now")
                elif info.get("state") != "stopped" or "stopped by run queue" not in text:
                    # it latched off (or someone else ended it): it stays off
                    log.warning("Run queue: %s is not started again: it ended %s (%s).",
                                loop.spec.title, info.get("state"), text)
                    self._record("run_queue_loop_not_resumed", loop=loop.spec.key,
                                 state=info.get("state"), why=text)
                else:
                    reply = loop.start(operator="run queue", client=self._host,
                                       path=resume.get("path"), owner="queue")
                    ok = reply.get("status") == "ok"
                    self._record("run_queue_loop_resume", loop=loop.spec.key, ok=ok,
                                 msg=reply.get("msg", ""))
                    if ok:
                        log.info("Run queue: out of jobs -- %s started again.", loop.spec.title)
                        self._owe_monitor = False
                    else:
                        log.warning("Run queue: out of jobs; %s could not start again: %s",
                                    loop.spec.title, reply.get("msg"))
                self._save()
                self._notify()
        if self._owe_monitor and (self._resume_loop is None or blocked):
            # (a loop waiting on a hold or a pause starts later, taking the core
            # from the monitor as any run does)
            self._owe_monitor = False
            self._record("run_queue_monitor_start")
            if self._start_monitor is not None:
                try:
                    self._start_monitor("the run queue has no job to run")
                except Exception:                     # noqa: BLE001
                    log.exception("Run queue: could not ask for the monitor")

    # -- the alarm ------------------------------------------------------------------------

    def _alarm_tick(self, now: float) -> None:
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            stuck = (cur is not None and cur.state == "launching"
                     and now - float(cur.launched_at or now) > SPAWN_WARN_S)
        if stuck:
            self._raise_alarm(now, cur, now - float(cur.launched_at),
                              f"{cur.name} has been launching for "
                              f"{(now - float(cur.launched_at)) / 60.0:.1f} min (its launcher "
                              "has not answered and no run of it is in liveOD)")
            return
        with self._lock:
            owed = self._current is None and bool(self._order(now))
            nxt = self._order(now)[0] if owed else None
        if owed and self._gate_kind != "blocked":
            # someone's run is legitimately using the machine (or it is free
            # and the job launches): no alarm, and the clock starts again
            self._owed_since = None
            if self._alarm is not None:
                self._clear_alarm("the machine is in legitimate use")
            return
        if not owed:
            self._owed_since = None
            if self._alarm is not None:
                self._clear_alarm("no job is waiting to start")
            return
        if self._owed_since is None:
            self._owed_since = now
            return
        waited = now - self._owed_since
        if waited <= self.alarm_s:
            return
        self._raise_alarm(now, nxt, waited,
                          f"{nxt.name} has been ready to start for {waited / 60.0:.0f} min and "
                          f"nothing has launched -- {self._waiting or 'the reason is not known'}")

    def _raise_alarm(self, now: float, job: Job, waited: float, text: str) -> None:
        """One WARNING and journal record per alarm_s while the alarm holds."""
        if self._alarm is not None and now - self._alarm["warned"] < self.alarm_s:
            return
        first = self._alarm is None
        why = self._waiting if job.state == "queued" else text
        self._alarm = {"since": now - waited, "warned": now, "job": job.id,
                       "waited_s": waited, "why": why}
        log.warning("RUN QUEUE ALARM: %s", text)
        self._record("run_queue_alarm", job=job.id, waited_s=round(waited, 1), why=why,
                     first=first)
        self._notify()

    def _clear_alarm(self, why: str) -> None:
        alarm, self._alarm = self._alarm, None
        log.info("Run queue alarm cleared: %s.", why)
        self._record("run_queue_alarm_cleared", why=why,
                     waited_s=round(self._clock() - alarm["since"], 1) if alarm else None)
        self._notify()

    # -- persistence -----------------------------------------------------------------------

    def _save(self) -> None:
        """Write queue.json.  Two threads may save at once (a request and the
        tick): each snapshot gets a sequence number under the queue's lock, and
        a snapshot older than the one last written is not written -- the file
        never goes back to an earlier state.  Deadlock-free whatever lock the
        caller holds (the save lock is never held while taking the queue's)."""
        if not self.enabled or self._no_save:
            return
        with self._lock:
            self._save_seq += 1
            seq = self._save_seq
            jobs = sorted(self._jobs.values(), key=lambda j: j.id)
            ended = [j for j in jobs if j.state in ENDED]
            keep = {j.id for j in ended[-KEEP_ENDED:]} | {j.id for j in jobs
                                                          if j.state not in ENDED}
            data = {"version": 1, "saved_at": self._clock(), "next_id": self._next_id,
                    "jobs": [j.to_dict() for j in jobs if j.id in keep],
                    "paused": {k: (dict(v) if v else None) for k, v in self._paused.items()},
                    "resume_loop": dict(self._resume_loop) if self._resume_loop else None,
                    "own_abort_ids": sorted(self._own_abort_ids)[-200:], "seq": seq}
        with self._save_lock:
            if seq <= self._saved_seq:
                return                    # a newer snapshot is already on disk
            try:
                from waxx.util.device_state.state_file_io import atomic_write  # noqa: PLC0415
                os.makedirs(self.directory, exist_ok=True)
                atomic_write(self._path("queue.json"), data)
                self._saved_seq = seq
            except Exception as exc:                  # noqa: BLE001
                log.error("Run queue: could not store the queue in %s (%s).",
                          self._path("queue.json"), exc)

    def _load_failed(self, exc) -> None:
        """queue.json could not be read: fail closed.  The file is moved aside
        (timestamped, never overwritten -- if it cannot be moved, the queue
        never writes queue.json this session), the queue starts empty with
        every scope paused (a person looks, then resumes), and job ids go on
        from the journal's highest."""
        from waxx.util.device_state.person_hold import move_aside  # noqa: PLC0415
        original = self._path("queue.json")
        aside = move_aside(original)
        if not aside:
            self._no_save = True
        why = (f"queue.json could not be read ({exc}); "
               + (f"moved to {aside}" if aside else "left in place, not overwritten")
               + " -- check it, then resume")
        now = self._clock()
        for scope in PAUSE_SCOPES:
            self._paused[scope] = {"by": "monitor server", "since": now, "reason": why,
                                   "owner": "person"}
        self._next_id = self._next_id_from_journal()
        log.error("Run queue: %s. The queue starts EMPTY and PAUSED (all jobs); the journal "
                  "has the jobs' history.", why)
        self._record("run_queue_load_failed", error=str(exc), moved_to=aside or None,
                     next_id=self._next_id)
        self._record("run_queue_pause", scope="all", by="monitor server", reason=why)
        self._save()

    def _next_id_from_journal(self) -> int:
        highest = 0
        try:
            with open(self._path("journal.jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        highest = max(highest, int(json.loads(line).get("job") or 0))
                    except (ValueError, TypeError, AttributeError):
                        continue
        except OSError:
            pass
        return highest + 1

    def _load(self) -> None:
        if not self.enabled:
            return
        try:
            with open(self._path("queue.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            jobs = [Job.from_dict(d) for d in data.get("jobs") or []]
            next_id = max([int(data.get("next_id") or 1)] + [j.id + 1 for j in jobs])
        except FileNotFoundError:
            return
        except Exception as exc:                      # noqa: BLE001
            self._load_failed(exc)
            return
        self._jobs = {j.id: j for j in jobs}
        self._next_id = next_id
        for scope in PAUSE_SCOPES:
            self._paused[scope] = (data.get("paused") or {}).get(scope) or None
        self._resume_loop = data.get("resume_loop") or None
        self._own_abort_ids = set(data.get("own_abort_ids") or ())
        for job in sorted(jobs, key=lambda j: j.id):
            if job.state not in IN_SLOT:
                continue
            self._log_pos[job.id] = 0
            self._log_partial[job.id] = ""
            self._tails[job.id] = deque(maxlen=TAIL_LINES)
            self._owe_monitor = True
            if job.state == "launching":
                # never launched again: looked for in liveOD (_follow_orphan)
                self._procs[job.id] = None
                log.warning("Run queue: %s was launching when the server stopped: not launched "
                            "again; its run is looked for in liveOD.", job.name)
                self._record("run_queue_launching_at_restart", job=job.id,
                             launched_at=job.launched_at)
                continue
            proc = None
            try:
                proc = self._adopt(job.pid, job.pid_started)
            except Exception:                         # noqa: BLE001
                proc = None
            job.state = "running"
            self._procs[job.id] = proc
            if proc is not None:
                job.adopted = True
                log.warning("Run queue: %s (pid %s, run %s) was running when the server "
                            "stopped and still is: followed again.", job.name, job.pid,
                            job.run_id)
                self._record("run_queue_adopted", job=job.id, pid=job.pid, run_id=job.run_id)
            else:
                # judged when it comes to the slot, from liveOD's record
                log.warning("Run queue: %s (pid %s, run %s) was running when the server "
                            "stopped; its process is gone -- judged from liveOD's record.",
                            job.name, job.pid, job.run_id)
                self._record("run_queue_process_gone", job=job.id, pid=job.pid,
                             run_id=job.run_id)
        self._take_leftover()
        self._save()

    # -- out ------------------------------------------------------------------------------

    def _record(self, kind: str, **fields) -> None:
        ts = self._clock()
        entry = {"ts": ts, "t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                 "kind": kind}
        entry.update(fields)
        if self.enabled:
            try:
                os.makedirs(self.directory, exist_ok=True)
                with open(self._path("journal.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=repr) + "\n")
            except OSError as exc:
                log.error("Run queue: could not append to its journal (%s).", exc)
        if self._journal is not None:
            # the ops journal appends to the lab's share synchronously: the copy
            # is written by flush_journal(), never while the queue's lock is held
            self._mirror.append((kind, fields))

    def flush_journal(self) -> None:
        """Copy the records made since the last flush to the ops journal.
        Called at the end of every request and tick, outside the lock."""
        if self._journal is None:
            return
        is_owned = getattr(self._lock, "_is_owned", None)
        if is_owned is not None and is_owned():
            return                        # under the lock: the next flush writes them
        with self._mirror_lock:
            while self._mirror:
                kind, fields = self._mirror.popleft()
                try:
                    self._journal.record(kind, **fields)
                except Exception:                     # noqa: BLE001
                    log.exception("Could not journal %s", kind)

    def _notify(self) -> None:
        self.flush_journal()
        if self._on_change is None:
            return
        try:
            self._on_change(self.info())
        except Exception:                             # noqa: BLE001
            log.exception("Run queue change notification failed")
