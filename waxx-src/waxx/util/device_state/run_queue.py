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
its output goes to ``<dir>/logs/<id>_<label>.out``.  "Run ID:" is read from
that log; the experiment's own pid comes from liveOD's POLL (``client_pid``).

**Cancel.**  A queued job is cancelled at once.  A running job is never
terminated: the queue sends liveOD's Abort (RESET) for it -- only when liveOD's
run in progress is that job's run id, on a POLL just before -- and the job
stays ``running`` until its process ends; it then ends ``cancelled`` (or
``saved``, if it saved before the Abort reached it).  An Abort is what liveOD's
own button does: the run stops at its next shot and liveOD discards its file.
An agent may not cancel a person's running job.

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

**Records.**  ``<dir>/queue.json`` holds every job not yet ended and the last
:data:`KEEP_ENDED` ended ones (atomic replace on every change);
``<dir>/journal.jsonl`` gets one line per transition (append only), and the
server's ops journal the same records (kinds ``run_queue_*``).  At a server
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
from waxx.util.device_state.person_hold import PersonHold
from waxx.util.device_state.run_loop import (
    RUN_ID_RE, _SHELL_CHARS, judge_run, live_od_outcome, tell_live_od_exited)

log = logging.getLogger(__name__)

STATES = ("queued", "running", "ending", "saved", "failed", "cancelled", "skipped")
#: States a job ends in.
ENDED = ("saved", "failed", "cancelled", "skipped")
#: States of the one job in the slot.
IN_SLOT = ("running", "ending")
OWNERS = ("person", "agent")
PAUSE_SCOPES = ("agent", "all")

#: The value of ``WAXX_LAUNCHER`` for the queue's jobs (run_gate.KNOWN_LAUNCHERS).
LAUNCHER = "kq"
JOB_ENV = "WAXX_QUEUE_JOB"
OWNER_ENV = "WAXX_OWNER"
NO_WRITE_BACK_ENV = "WAXX_CAL_NO_WRITE_BACK"

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

_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")


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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "Job":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in known})


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
                 spawn=None, adopt=None, exclude=(),
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
        self._clock = clock
        self.poll_every_s = float(poll_every_s)
        self._gap_s = float(gap_s)
        self.alarm_s = float(alarm_s)
        self._outcome_wait_s = float(outcome_wait_s)
        self._host = client_name if client_name is not None else socket.gethostname()

        self._lock = threading.RLock()
        self._tick_lock = threading.Lock()
        self._abort_lock = threading.Lock()
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
        self._load()

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
        cwd = str(obj.get("cwd") or path.parent)
        if not Path(cwd).is_dir():
            raise QueueError(f"no such working folder on the monitor server's machine: {cwd}")
        owner = str(obj.get("owner") or "person")
        if owner not in OWNERS:
            raise QueueError(f"owner must be one of {', '.join(OWNERS)}, not {owner!r}")
        try:
            priority = int(obj.get("priority") or 0)
        except (TypeError, ValueError):
            raise QueueError(f"priority must be an integer, not {obj.get('priority')!r}")
        due = obj.get("due")
        if due is not None:
            try:
                due = float(due)
            except (TypeError, ValueError):
                raise QueueError(f"due must be epoch seconds or null, not {due!r}")
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
            unknown = [a for a in after if a not in self._jobs]
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
        """``{"id", "token"?, "by", "owner"?}``: a queued job is cancelled; a
        running one gets liveOD's Abort for its run (never a kill) and ends
        when its process does."""
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        as_owner = str(obj.get("owner") or "person")
        with self._lock:
            try:
                job = self._find(obj)
            except QueueError as exc:
                return {"status": "error", "msg": str(exc)}
            if job.state == "queued":
                self._end(job, "cancelled", f"cancelled by {by}")
                reply = {"status": "ok", "job": job.to_dict()}
            elif job.state == "running":
                if job.owner == "person" and as_owner == "agent":
                    return {"status": "error",
                            "msg": f"{job.name} is a person's run in progress: an agent may "
                                   "not abort it"}
                if job.cancel is None:
                    job.cancel = {"by": by, "at": self._clock(), "abort_sent": False,
                                  "abort_note": ""}
                    self._record("run_queue_cancel_requested", job=job.id, by=by,
                                 run_id=job.run_id)
                    log.warning("Run queue: cancel of running %s asked by %s: liveOD's Abort "
                                "goes to run %s (its file is discarded, as for any Abort); "
                                "the job ends when its process does.", job.name, by,
                                job.run_id if job.run_id is not None else "(no run id yet)")
                reply = {"status": "ok", "job": job.to_dict(), "pending": True}
            else:
                return {"status": "error", "msg": f"{job.name} is already {job.state}"}
        if job.state == "running":
            self._send_abort(job)
            reply["job"] = job.to_dict()
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
        """``{"scope": "agent" | "all", "by", "reason"}``."""
        scope = str(obj.get("scope") or "agent")
        if scope not in PAUSE_SCOPES:
            return {"status": "error", "msg": f"scope must be one of {', '.join(PAUSE_SCOPES)}"}
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        reason = str(obj.get("reason") or "")
        with self._lock:
            self._paused[scope] = {"by": by, "since": self._clock(), "reason": reason}
        log.warning("Run queue: %s jobs paused by %s%s.", "agent" if scope == "agent" else "all",
                    by, f": {reason}" if reason else "")
        self._record("run_queue_pause", scope=scope, by=by, reason=reason)
        self._save()
        self._notify()
        return {"status": "ok", "run_queue": self.info()}

    def resume(self, obj: Mapping) -> dict:
        scope = str(obj.get("scope") or "agent")
        if scope not in PAUSE_SCOPES:
            return {"status": "error", "msg": f"scope must be one of {', '.join(PAUSE_SCOPES)}"}
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        with self._lock:
            if self._paused.get(scope) is None:
                return {"status": "error", "msg": f"{scope} jobs are not paused"}
            self._paused[scope] = None
        log.info("Run queue: %s jobs resumed by %s.", scope, by)
        self._record("run_queue_resume", scope=scope, by=by)
        self._save()
        self._notify()
        return {"status": "ok", "run_queue": self.info()}

    def hold_request(self, obj: Mapping) -> dict:
        reply = self.hold.hold(str(obj.get("reason") or ""),
                               str(obj.get("by") or obj.get("operator") or obj.get("client")
                                   or ""))
        self._notify()
        return reply

    def release_request(self, obj: Mapping) -> dict:
        reply = self.hold.release(str(obj.get("by") or obj.get("operator")
                                      or obj.get("client") or ""))
        self._notify()
        return reply

    # -- state ----------------------------------------------------------------------

    def current_job(self) -> dict | None:
        """The job in the slot (running / ending), or None."""
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
        waits = [a for a in job.after if (self._jobs.get(a) is None
                                           or self._jobs[a].state != "saved")]
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
        alarm.  Called by one thread."""
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            now = self._clock()
            if self._last_poll_t is None or now - self._last_poll_t >= self.poll_every_s:
                self._poll_now()
            self.hold.tick()
            cur = self._jobs.get(self._current) if self._current is not None else None
            if cur is not None:
                self._follow(cur)
            self._settle(now)
            if self._current is None:
                self._schedule(self._clock())
            self._alarm_tick(self._clock())
        finally:
            self._tick_lock.release()

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
        self.hold.observe_poll(poll, self.agent_run_ids(), self.own_abort_ids())
        return poll

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
                        dep = self._jobs.get(a)
                        if dep is not None and dep.state in ENDED and dep.state != "saved":
                            self._end(job, "skipped", f"job {a} ended {dep.state}"
                                      + (f" ({dep.reason})" if dep.reason else ""))
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
                self._stop_loop_for_jobs(loop, order[0])
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

    def _gate(self) -> str:
        """Why the next job may not launch now ("" when it may)."""
        busy = self._server_busy() if self._server_busy is not None else ""
        if busy:
            return busy
        loop = self._active_loop()
        if loop is not None:
            return (f"{loop.spec.title} is finishing its run in progress (asked to stop for "
                    "the queue)")
        poll = self._poll_now()
        if poll is None:
            return "liveOD is not reachable -- a run could not save"
        fence = self._fence() if self._fence is not None else None
        if fence and fence.get("run_id") in self.own_run_ids():
            fence = None                  # a fence of the queue's own (ended) job
        monitor = self._monitor_state() if self._monitor_state is not None else None
        verdict = run_gate.classify(poll, fence, monitor_state=monitor)
        if verdict.state == "free":
            return ""
        if verdict.waivable:
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
        next eligible job may go), True otherwise (launched, or failed to
        start)."""
        # TODO(phase 2): warm-ahead / GO handshake and liveOD RESERVE/START
        # attach here (run-queue plan); nothing of it in phase 1.
        try:
            sha = file_sha256(job.path)
        except OSError as exc:
            with self._lock:
                if job.state == "queued":
                    self._end(job, "skipped", f"the file cannot be read at launch ({exc})")
            self._save()
            self._notify()
            return False
        with self._lock:
            if job.state != "queued":                  # cancelled meanwhile
                return False
            if sha != job.sha256 and not job.allow_drift:
                self._end(job, "skipped", "source changed since submit")
                self._save()
                self._notify()
                return False
            os.makedirs(self._path("logs"), exist_ok=True)
            log_path = self._path("logs", f"{job.id}_{job.label}.out")
            command = ar_command(_quote(job.path))
            if job.argv:
                command += " " + " ".join(_quote(a) for a in job.argv)
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            env[run_gate.LAUNCHER_ENV] = LAUNCHER
            env[JOB_ENV] = str(job.id)
            env[OWNER_ENV] = job.owner
            if job.write_back is False:
                env[NO_WRITE_BACK_ENV] = "1"
            else:
                env.pop(NO_WRITE_BACK_ENV, None)
            try:
                with open(log_path, "ab") as f:
                    f.write((f"── run queue {time.strftime('%Y-%m-%d %H:%M:%S')}: {job.name}, "
                             f"owner {job.owner}, sha256 {sha[:12]}"
                             + (" (drift allowed: file changed since submit)"
                                if sha != job.sha256 else "")
                             + f" ──\n$ {command}\n").encode("utf-8"))
                proc = self._spawn(command, cwd=job.cwd, env=env, log_path=log_path)
            except OSError as exc:
                log.error("Run queue: could not start %s: %r", job.name, exc)
                for line in environment_report():
                    log.error("  %s", line)
                job.log_path = log_path
                self._end(job, "failed", f"could not start: {exc!r}")
                self._last_end_t = self._clock()
                self._save()
                self._notify()
                return True
            job.state = "running"
            job.pid = int(proc.pid)
            job.pid_started = getattr(proc, "started", None)
            job.log_path = log_path
            job.launched_at = self._clock()
            self._current = job.id
            self._procs[job.id] = proc
            self._log_pos[job.id] = 0
            self._log_partial[job.id] = ""
            self._tails[job.id] = deque(maxlen=TAIL_LINES)
            self._owe_monitor = True
            self._owed_since = None
        log.info("Run queue: %s launched (pid %s): %s", job.name, job.pid, command)
        self._record("run_queue_launch", job=job.id, pid=job.pid, command=command,
                     log_path=log_path, owner=job.owner, drift=sha != job.sha256)
        if self._alarm is not None:
            self._clear_alarm("a job was launched")
        self._save()
        self._notify()
        return True

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
                self._record("run_queue_run_id", job=job.id, run_id=job.run_id)
                self._save()
                self._notify()

    def _follow(self, job: Job) -> None:
        self._take_lines(job, self._read_log(job))
        poll = self._last_poll
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
            self._end(job, state, reason)
            self._current = None
            self._last_end_t = self._clock()
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

    def _send_abort_once(self, job: Job) -> None:
        if job.cancel is None or job.cancel.get("abort_sent"):
            return
        if job.run_id is None:
            note = "no run id yet: the Abort goes once the run has one"
            if job.cancel.get("abort_note") != note:
                job.cancel["abort_note"] = note
            return
        poll = self._poll_now()
        if poll is None:
            job.cancel["abort_note"] = "liveOD is not reachable: the Abort is not sent yet"
            return
        if not poll.get("run_in_progress") or poll.get("run_id") != job.run_id:
            job.cancel["abort_note"] = (f"liveOD's run in progress is "
                                        f"{poll.get('run_id') if poll.get('run_in_progress') else 'none'}"
                                        f", not {job.run_id}: no Abort sent")
            return
        if poll.get("save_in_progress") or poll.get("run_state") == "saving":
            job.cancel["abort_note"] = f"liveOD is saving run {job.run_id}: no Abort sent"
            return
        if poll.get("reset_requested"):
            job.cancel.update(abort_sent=True, abort_note="an Abort was already pending")
            return
        if self._live_od_reset is None:
            job.cancel["abort_note"] = "this queue has no way to send liveOD's Abort"
            return
        with self._lock:
            self._own_abort_ids.add(job.run_id)
        try:
            reply = self._live_od_reset()
        except Exception as exc:                      # noqa: BLE001
            job.cancel["abort_note"] = f"sending the Abort failed: {exc}"
            log.error("Run queue: liveOD's Abort for run %s (%s) failed: %s", job.run_id,
                      job.name, exc)
            return
        ok = bool(isinstance(reply, dict) and reply.get("ok"))
        job.cancel.update(abort_sent=True,
                          abort_note="Abort sent" if ok else f"liveOD refused: {reply}")
        log.warning("Run queue: liveOD's Abort sent for run %s (%s): %s", job.run_id, job.name,
                    "acknowledged" if ok else reply)
        self._record("run_queue_abort_sent", job=job.id, run_id=job.run_id, ok=ok,
                     by=job.cancel.get("by"))
        self._save()

    # -- ending jobs and chains ---------------------------------------------------------------

    def _end(self, job: Job, state: str, reason: str) -> None:
        """Put a job in an ended state (under the lock), journal it, and stop
        its chain when it failed or was skipped."""
        job.state = state
        job.reason = reason
        job.ended_at = self._clock()
        self._record("run_queue_end", job=job.id, state=state, reason=reason,
                     run_id=job.run_id, exit_code=job.exit_code, outcome=job.outcome)
        if state in ("failed", "skipped") and job.chain and job.stop_on_failure:
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
                                       path=resume.get("path"))
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
            owed = self._current is None and bool(self._order(now))
            nxt = self._order(now)[0] if owed else None
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
        if self._alarm is not None and now - self._alarm["warned"] < self.alarm_s:
            return
        first = self._alarm is None
        self._alarm = {"since": self._owed_since, "warned": now, "job": nxt.id,
                       "waited_s": waited, "why": self._waiting}
        log.warning("RUN QUEUE ALARM: %s has been ready to start for %.0f min and nothing has "
                    "launched -- %s", nxt.name, waited / 60.0,
                    self._waiting or "the reason is not known")
        self._record("run_queue_alarm", job=nxt.id, waited_s=round(waited, 1),
                     why=self._waiting, first=first)
        self._notify()

    def _clear_alarm(self, why: str) -> None:
        alarm, self._alarm = self._alarm, None
        log.info("Run queue alarm cleared: %s.", why)
        self._record("run_queue_alarm_cleared", why=why,
                     waited_s=round(self._clock() - alarm["since"], 1) if alarm else None)
        self._notify()

    # -- persistence -----------------------------------------------------------------------

    def _save(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.id)
            ended = [j for j in jobs if j.state in ENDED]
            keep = {j.id for j in ended[-KEEP_ENDED:]} | {j.id for j in jobs
                                                          if j.state not in ENDED}
            data = {"version": 1, "saved_at": self._clock(), "next_id": self._next_id,
                    "jobs": [j.to_dict() for j in jobs if j.id in keep],
                    "paused": {k: (dict(v) if v else None) for k, v in self._paused.items()},
                    "resume_loop": dict(self._resume_loop) if self._resume_loop else None,
                    "own_abort_ids": sorted(self._own_abort_ids)[-200:]}
        try:
            from waxx.util.device_state.state_file_io import atomic_write  # noqa: PLC0415
            os.makedirs(self.directory, exist_ok=True)
            atomic_write(self._path("queue.json"), data)
        except Exception as exc:                      # noqa: BLE001
            log.error("Run queue: could not store the queue in %s (%s).",
                      self._path("queue.json"), exc)

    def _load(self) -> None:
        if not self.enabled:
            return
        try:
            with open(self._path("queue.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except Exception as exc:                      # noqa: BLE001
            log.error("Run queue: could not read %s (%s): starting EMPTY -- the jobs in it are "
                      "not run; the journal has their history.", self._path("queue.json"), exc)
            self._record("run_queue_load_failed", error=str(exc))
            return
        jobs = [Job.from_dict(d) for d in data.get("jobs") or []]
        self._jobs = {j.id: j for j in jobs}
        self._next_id = max([int(data.get("next_id") or 1)] + [j.id + 1 for j in jobs])
        for scope in PAUSE_SCOPES:
            self._paused[scope] = (data.get("paused") or {}).get(scope) or None
        self._resume_loop = data.get("resume_loop") or None
        self._own_abort_ids = set(data.get("own_abort_ids") or ())
        for job in sorted(jobs, key=lambda j: j.id):
            if job.state not in IN_SLOT:
                continue
            proc = None
            try:
                proc = self._adopt(job.pid, job.pid_started)
            except Exception:                         # noqa: BLE001
                proc = None
            self._log_pos[job.id] = 0
            self._log_partial[job.id] = ""
            self._tails[job.id] = deque(maxlen=TAIL_LINES)
            if proc is not None and self._current is None:
                job.adopted = True
                job.state = "running"
                self._current = job.id
                self._procs[job.id] = proc
                self._owe_monitor = True
                log.warning("Run queue: %s (pid %s, run %s) was running when the server "
                            "stopped and still is: followed again.", job.name, job.pid,
                            job.run_id)
                self._record("run_queue_adopted", job=job.id, pid=job.pid, run_id=job.run_id)
            else:
                # its process is gone: judged at the first tick from liveOD's record
                job.state = "running"
                if self._current is None:
                    self._current = job.id
                    self._procs[job.id] = None
                    log.warning("Run queue: %s (pid %s, run %s) was running when the server "
                                "stopped; its process is gone -- judged from liveOD's record.",
                                job.name, job.pid, job.run_id)
                    self._record("run_queue_process_gone", job=job.id, pid=job.pid,
                                 run_id=job.run_id)
                else:
                    self._end(job, "failed", "server restarted; process gone")
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
            try:
                self._journal.record(kind, **fields)
            except Exception:                         # noqa: BLE001
                log.exception("Could not journal %s", kind)

    def _notify(self) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(self.info())
        except Exception:                             # noqa: BLE001
            log.exception("Run queue change notification failed")
