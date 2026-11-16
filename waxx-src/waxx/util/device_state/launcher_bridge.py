"""The monitor server's side of ``kq launch``: the run queue's jobs run on
the PC that submitted them (user, 2026-11-15).

The queue (:mod:`~waxx.util.device_state.run_queue`) decides the order, the
gate and GO; it never starts, follows or kills a job's process itself.  Each
job's **launcher** (:mod:`~waxx.util.device_state.launcher`, a hidden ``kq
launch`` process on the job's ``run_host``) does, and reports in a heartbeat
-- the ``{"type": "run_queue", "action": "launcher"}`` request, answered here
from memory and the queue's records only.

:class:`LauncherBridge` gives the queue the four callables it used to run
processes with -- ``spawn``, the adopt step (:meth:`LauncherBridge.proc`),
``kill`` (an order) and ``set_priority`` (an order) -- and a
:class:`RemoteProc` in place of a local process handle, so the queue's
following, judging and warm-start logic is unchanged.

Rules (see the plan, ``C:\\lab\\skynet_log\\plans\\2026-11-15_kq_submitter_launch_rev2.md``):

* **Orders come from the queue's saved state**, never from memory alone: a
  job ``launching`` (or ``running`` with no process yet) whose launcher has no
  child for it gets a *launch* order; a queued job with a live warm record
  whose token its launcher has not started gets a *warm* order.  The queue
  saves those states before it calls ``spawn``, and the launcher writes
  "starting" to its own state file before it starts a child, so neither side
  restarting can start a job twice.  A job the queue knows ran (a pid, a run
  id or liveOD's client pid on record) never gets a launch order again.
* **A missing or silent launcher never means "gone".**  :meth:`RemoteProc.poll`
  is None until the launcher reports an exit -- however long it is silent.
  The slot stays taken (an alarm names the host) until it reports, or a
  person declares the job gone (``kq kill <id> --gone``,
  :meth:`LauncherBridge.declare_gone`).  The ARTIQ core device discards the
  running run when a second session connects (user, 2026-11-15): GO while
  the previous process may be alive would end that run without its cleanup.
* **A process alive after its job ended** (one declared gone that comes back)
  blocks every launch (:meth:`LauncherBridge.gate_block`) until it exits.
* **One launcher per job**: the first to attach keeps it while it beats; a
  second is refused until the first has been silent :data:`LAUNCHER_SILENT_S`.
* A queued job with no launcher attached is not eligible ("waiting for
  <host>'s launcher"); for :data:`REATTACH_GRACE_S` after the server starts,
  nothing launches, so launchers can attach again first.
* The server starts ``kq launch`` itself only for jobs of its own host
  (``start_launcher``; a run-again of one of its jobs, a launcher that died);
  for other hosts the Run Queue panel running there does.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

#: A launcher not heard from for this long is *silent*: its queued jobs are
#: not eligible, its job in the slot raises the alarm, another launcher may
#: take its jobs.
LAUNCHER_SILENT_S = 30.0
#: After the server starts, nothing launches for this long (launchers attach
#: again first; their jobs would otherwise look launcher-less and be skipped).
REATTACH_GRACE_S = 15.0
#: How long ``spawn`` waits for the launcher to report the child it was
#: ordered to start (then LaunchUnknown: the queue follows the job anyway).
SPAWN_WAIT_S = 30.0
#: Heartbeat intervals the server asks for.
FAST_BEAT_S = 1.0
SLOW_BEAT_S = 5.0
#: The server starts a launcher for one of its own host's jobs at most this often,
SELF_START_EVERY_S = 60.0
#: and only for a job that has had no launcher this long since its submit (the
#: submitting kq starts one itself; this is for one that never came or died).
SELF_START_AFTER_S = 10.0
#: Child states a launcher reports.  "starting": recorded before the child
#: was started (no pid yet); "unknown": a launcher restarted and found a
#: "starting" with no pid (the child may or may not run: held for a person).
RUNNING_STATES = ("starting", "running", "unknown")
TERMINAL_STATES = ("exited", "refused", "skipped", "cancelled")
#: The key of a job's cold (non-warm) child; a warm child's key is its warm token.
COLD = "cold"


class SkippedAtLaunch(OSError):
    """The launcher did not start the job: its file changed since submit
    (and drift is not allowed).  The job ends ``skipped``."""


class NotStarted(OSError):
    """The launcher refused to start the job (its file, folder or
    environment), or it was killed before it started.  The job ends failed
    (cancelled after a kill)."""


def _now_text(t) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(t)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


class RemoteProc:
    """A job's process on its launcher's PC, as the queue follows a local
    one: ``pid`` / ``started`` (0 / None until the launcher reports them),
    ``poll()`` (None while it lives OR the launcher is silent; its exit code
    once the launcher reports it; -1 once a person declared the job gone),
    ``close()`` (the queue is done with it: its exit is acknowledged to the
    launcher).  ``not_started``: ``(state, why)`` when the launcher refused,
    skipped or cancelled it instead of starting it."""

    exit_code_known = True
    _handle = None                  # (RunQueue.kill: never "vouched" as a local handle)

    def __init__(self, bridge: "LauncherBridge", job_id: int, key: str):
        self.bridge = bridge
        self.job_id = int(job_id)
        self.key = str(key)

    def _child(self) -> dict:
        s = self.bridge.sessions.get(self.job_id)
        return dict((s.children.get(self.key) or {})) if s is not None else {}

    @property
    def pid(self) -> int:
        try:
            return int(self._child().get("pid") or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def started(self):
        return self._child().get("started")

    @property
    def host(self) -> str:
        s = self.bridge.sessions.get(self.job_id)
        return s.host if s is not None else ""

    @property
    def not_started(self) -> tuple | None:
        c = self._child()
        if c.get("state") in ("refused", "skipped", "cancelled"):
            return c.get("state"), str(c.get("why") or c.get("state"))
        return None

    def poll(self):
        if self.job_id in self.bridge.gone:
            return -1
        c = self._child()
        state = c.get("state")
        if state == "exited":
            try:
                return int(c.get("code"))
            except (TypeError, ValueError):
                return -1
        if state in ("refused", "skipped", "cancelled"):
            return -1
        return None

    def close(self) -> None:
        self.bridge.consumed.add((self.job_id, self.key))


class Session:
    """One job's launcher as last heard from."""

    def __init__(self, lid: str, host: str, pid, started, now: float):
        self.lid = lid
        self.host = host
        self.pid = pid
        self.started = started
        self.first_at = now
        self.last_beat = now
        #: key -> the launcher's report of that child
        self.children: dict = {}
        #: the file as the launcher reads it now
        self.info: dict = {}
        self.source_changed: bool | None = None
        #: the next byte of the launcher's local log the server wants
        self.log_ack: int | None = None

    def public(self, now: float, silent_s: float) -> dict:
        silent = max(0.0, now - self.last_beat)
        return {"host": self.host, "pid": self.pid, "attached": silent <= silent_s,
                "silent_s": round(silent, 1), "since": self.first_at}


class LauncherBridge:
    """The queue's link to its jobs' launchers (see the module docstring).
    ``queue``: the :class:`~waxx.util.device_state.run_queue.RunQueue`;
    ``start_launcher(pairs)``: starts ``kq launch`` on THIS machine for
    ``[(id, token), ...]`` (default :func:`launcher.start_detached`);
    ``host``: this machine's name."""

    def __init__(self, queue, *, clock: Callable[[], float] | None = None,
                 start_launcher=None, host: str | None = None,
                 spawn_wait_s: float = SPAWN_WAIT_S, silent_s: float = LAUNCHER_SILENT_S,
                 grace_s: float = REATTACH_GRACE_S, wait_clock=time.monotonic):
        self.q = queue
        self._clock = clock or queue._clock
        self._wait_clock = wait_clock
        if start_launcher is None:
            from waxx.util.device_state.launcher import start_detached  # noqa: PLC0415
            start_launcher = start_detached
        self._start_launcher = start_launcher
        self.host = host if host is not None else socket.gethostname()
        self.spawn_wait_s = float(spawn_wait_s)
        self.silent_s = float(silent_s)
        self.grace_s = float(grace_s)
        self.started_at = self._clock()
        self.sessions: dict[int, Session] = {}
        #: (job id, key) the queue is done with (exit acknowledged)
        self.consumed: set = set()
        #: job ids a person declared gone (their RemoteProc polls -1)
        self.gone: set = set()
        #: (job id, key) -> the kill order's sequence number, until answered
        self.kills: dict = {}
        self._kill_seq = 0
        #: job id -> {"pid", "started"}: raise the experiment's priority (GO)
        self.priority: dict = {}
        #: job id -> text: a process alive after its job ended
        self.zombies: dict = {}
        self._self_started: dict = {}
        self._cond = threading.Condition()

    # -- the queue's callables -----------------------------------------------------------

    def spawn(self, command: str = "", *, cwd: str = "", env: dict | None = None,
              log_path: str = "", below_normal: bool = False, **_ignored) -> RemoteProc:
        """In place of starting ``command``: the queue has saved the job as
        launching (or its warm record), so its launcher gets the order on its
        next heartbeat (:meth:`orders`); this waits for the launcher's report
        of the child.  -> a :class:`RemoteProc`; :class:`SkippedAtLaunch` /
        :class:`NotStarted` when the launcher did not start it;
        ``LaunchUnknown`` after :data:`SPAWN_WAIT_S` with no report (the queue
        then follows the job anyway -- it is never started a second time)."""
        from waxx.util.device_state.detached import LaunchUnknown  # noqa: PLC0415
        from waxx.util.device_state.run_queue import JOB_ENV, WARM_ENV, _parse_queue_job  # noqa: PLC0415
        env = env or {}
        jid, _token = _parse_queue_job(env.get(JOB_ENV))
        if jid is None:
            raise OSError("no job named in the launch's environment")
        key = env.get(WARM_ENV) or COLD
        deadline = self._wait_clock() + self.spawn_wait_s
        with self._cond:
            while True:
                if jid in self.gone:
                    raise NotStarted("declared gone by a person while it was launching")
                s = self.sessions.get(jid)
                c = (s.children.get(key) if s is not None else None) or {}
                state = c.get("state")
                if state in ("running", "exited"):
                    return RemoteProc(self, jid, key)
                if state == "skipped":
                    raise SkippedAtLaunch(str(c.get("why") or "source changed since submit"))
                if state in ("refused", "cancelled"):
                    raise NotStarted(str(c.get("why") or state))
                left = deadline - self._wait_clock()
                if left <= 0:
                    host = s.host if s is not None else "its PC"
                    raise LaunchUnknown(f"the launcher on {host} reported no process for job "
                                        f"{jid} within {self.spawn_wait_s:.0f} s (state: "
                                        f"{state or 'not started yet'}); it is followed, never "
                                        "started again")
                self._cond.wait(min(0.5, left))

    def proc(self, job_id: int, key: str) -> RemoteProc:
        """The job's child ``key`` as the queue follows it -- at a restart, on
        the orphan path, for a warm process: always a handle, never None (a
        process this server cannot see is not a process that is gone)."""
        return RemoteProc(self, job_id, key)

    def set_priority(self, pid, started) -> bool:
        """GO's backstop: the job whose warm process is ``pid`` gets an order
        to raise it to normal priority (its launcher does it, on its PC)."""
        with self.q._lock:
            for job in self.q._jobs.values():
                w = job.warm or {}
                if w.get("py_pid") == pid:
                    self.priority[job.id] = {"pid": pid, "started": started}
                    return True
        return False

    def kill_order(self, job_id: int, key: str) -> int:
        """Order the job's launcher to kill child ``key`` with its tree (or,
        not started yet, never to start it).  -> the order's number; the
        result comes in a heartbeat (:meth:`_take_kills`)."""
        with self._cond:
            self._kill_seq += 1
            self.kills[(int(job_id), str(key))] = self._kill_seq
            return self._kill_seq

    def declare_gone(self, job_id: int) -> None:
        """A person says nothing of the job runs any more (its launcher is
        silent): its RemoteProc polls -1 and the queue ends it."""
        with self._cond:
            self.gone.add(int(job_id))
            self._cond.notify_all()

    # -- what the queue asks -----------------------------------------------------------

    def in_grace(self) -> bool:
        return self._clock() - self.started_at < self.grace_s

    def silent_for(self, job_id: int) -> float | None:
        """Seconds since the job's launcher was last heard from; None when no
        launcher has attached for it (since this server started)."""
        s = self.sessions.get(int(job_id))
        return None if s is None else max(0.0, self._clock() - s.last_beat)

    def attached(self, job) -> bool:
        silent = self.silent_for(job.id)
        return silent is not None and silent <= self.silent_s

    def source_changed(self, job_id: int) -> bool | None:
        s = self.sessions.get(int(job_id))
        return None if s is None else s.source_changed

    def gate_block(self) -> str:
        """Why nothing may launch because of a launcher's report ("" when
        nothing): a job's process alive after the job ended."""
        if self.zombies:
            jid = sorted(self.zombies)[0]
            return self.zombies[jid] + ": nothing launches until it exits"
        return ""

    def public(self, job) -> dict | None:
        s = self.sessions.get(job.id)
        out = s.public(self._clock(), self.silent_s) if s is not None else None
        if out is not None:
            out["kill_pending"] = any(k[0] == job.id for k in self.kills)
        return out

    def silent_text(self, job) -> str:
        s = self.sessions.get(job.id)
        host = (s.host if s is not None else "") or job.run_host or "its PC"
        silent = self.silent_for(job.id)
        return (f"{job.name}'s launcher on {host} "
                + ("never attached" if silent is None else
                   f"has been silent {silent:.0f} s")
                + ": the slot is held (nothing else gets GO) until it reports. A person: "
                f"once nothing of it runs on {host}, kq kill {job.id} --gone")

    # -- the heartbeat ---------------------------------------------------------------------

    def request(self, obj) -> dict:
        """A launcher's heartbeat -> orders.  ``{"launcher": {"id", "host",
        "pid", "started"}, "jobs": [{"id", "token", "children": {key: {...}},
        "kills": {key: {"seq", "result"}}, "log": {"offset", "end", "text"},
        "info": {...}}]}`` -> ``{"status": "ok", "next_beat_s", "jobs":
        {"<id>": {...}}}``."""
        me = obj.get("launcher") if isinstance(obj.get("launcher"), dict) else {}
        lid = str(me.get("id") or "")
        host = str(me.get("host") or "").strip()
        if not lid or not host:
            return {"status": "error", "msg": "a launcher must name its id and host"}
        reports = obj.get("jobs") if isinstance(obj.get("jobs"), list) else []
        out, fast, changed = {}, False, False
        now = self._clock()
        q = self.q
        with q._lock:
            head = [j.id for j in q._queued_in_order()[:2]]
            for rep in reports:
                if not isinstance(rep, dict):
                    continue
                try:
                    jid = int(rep.get("id"))
                except (TypeError, ValueError):
                    continue
                reply, c, f = self._one(jid, rep, lid, host, me, now, head)
                out[str(jid)] = reply
                changed |= c
                fast |= f
        with self._cond:
            self._cond.notify_all()
        if changed:
            q._save()
            q._notify()
        else:
            q.flush_journal()
        return {"status": "ok", "jobs": out, "next_beat_s": FAST_BEAT_S if fast else SLOW_BEAT_S,
                "server_time": now}

    def _one(self, jid: int, rep: dict, lid: str, host: str, me: dict, now: float,
             head: list) -> tuple[dict, bool, bool]:
        """One job of a heartbeat (under the queue's lock) -> (reply, changed, fast)."""
        q = self.q
        job = q._jobs.get(jid)
        token = str(rep.get("token") or "")
        if job is None or not token or job.token != token:
            return {"drop": f"job {jid} is not a job of this run queue"}, False, False
        if job.run_host and job.run_host.lower() != host.lower():
            return {"drop": f"{job.name} runs on {job.run_host}, not {host}"}, False, False
        s = self.sessions.get(jid)
        if s is not None and s.lid != lid:
            if now - s.last_beat <= self.silent_s:
                return {"refused": f"{job.name} has a launcher already (pid {s.pid} on "
                                   f"{s.host}, heard from {now - s.last_beat:.0f} s ago)"}, \
                    False, False
            log.warning("Run queue: %s: a new launcher (pid %s on %s) takes over from pid %s, "
                        "silent %.0f s.", job.name, me.get("pid"), host, s.pid,
                        now - s.last_beat)
            q._record("run_queue_launcher_replaced", job=jid, host=host, pid=me.get("pid"),
                      old_pid=s.pid, silent_s=round(now - s.last_beat, 1))
            s = None
        changed = False
        if s is None:
            s = Session(lid, host, me.get("pid"), me.get("started"), now)
            old = self.sessions.get(jid)
            if old is not None:
                s.children, s.log_ack = old.children, old.log_ack
            self.sessions[jid] = s
            q._record("run_queue_launcher_attached", job=jid, host=host, pid=me.get("pid"))
            log.info("Run queue: %s: launcher attached (pid %s on %s).", job.name,
                     me.get("pid"), host)
        s.last_beat = now
        changed |= self._take_info(job, s, rep.get("info"))
        changed |= self._take_children(job, s, rep.get("children"))
        changed |= self._take_kills(job, s, rep.get("kills"))
        log_ack = self._take_log(job, s, rep.get("log"))
        reply = {"path": job.path, "sha256": job.sha256, "allow_drift": job.allow_drift,
                 "log_ack": log_ack, "ack": self._acks(job, s)}
        order = self._order(job, s)
        if order is not None:
            reply["order"] = order
        kills = {k: seq for (j, k), seq in self.kills.items() if j == jid}
        if kills:
            reply["kill"] = kills
        if jid in self.priority:
            reply["priority"] = self.priority.pop(jid)
        if self._done(job, s):
            reply["drop"] = f"{job.name} is {job.state}"
        fast = (job.state not in ("queued",) + tuple(_ended()) or job.warm is not None
                or jid in head or bool(kills) or order is not None)
        return reply, changed, fast

    def _take_info(self, job, s: Session, info) -> bool:
        """The file as the launcher reads it: fills a job submitted without it
        (a run-again from another PC), and says whether it changed since submit."""
        if not isinstance(info, dict):
            return False
        s.info = dict(info)
        sha = info.get("sha256")
        changed = False
        if job.state == "queued" and not job.sha256 and sha:
            job.sha256 = str(sha)
            job.expt_class = str(info.get("expt_class") or "")
            job.calibrates_declared = [str(k) for k in (info.get("calibrates_declared") or [])]
            job.cold_why = str(info.get("cold_why") or "")
            self.q._record("run_queue_source_from_launcher", job=job.id, sha256=job.sha256,
                           expt_class=job.expt_class, calibrates=job.calibrates_declared,
                           cold_why=job.cold_why)
            changed = True
        s.source_changed = (None if not (sha and job.sha256) else str(sha) != job.sha256)
        return changed

    def _current_keys(self, job) -> set:
        """The children the queue may still be following for ``job``."""
        w = job.warm or {}
        keys = set()
        if w.get("token") and job.state not in _ended():
            keys.add(str(w["token"]))
        if job.state in ("launching", "running", "ending") and not w.get("go"):
            keys.add(COLD)
        return keys

    def _take_children(self, job, s: Session, children) -> bool:
        if not isinstance(children, dict):
            return False
        changed = False
        alive = []
        for key, c in children.items():
            if not isinstance(c, dict):
                continue
            key = str(key)
            old = s.children.get(key) or {}
            s.children[key] = dict(c)
            state = c.get("state")
            if state in RUNNING_STATES:
                alive.append((key, c))
            if state != old.get("state"):
                self.q._record("run_queue_launcher_child", job=job.id, key=_key_text(key),
                               state=state, pid=c.get("pid"), code=c.get("code"),
                               why=c.get("why"), host=s.host)
            if state == "running" and c.get("pid"):
                w = job.warm or {}
                if key == COLD and not job.pid and job.state in ("launching", "running"):
                    job.pid, job.pid_started = int(c["pid"]), c.get("started")
                    changed = True
                elif w.get("token") == key and not w.get("cmd_pid"):
                    w["cmd_pid"], w["cmd_started"] = int(c["pid"]), c.get("started")
                    changed = True
        if job.state in _ended() and alive:
            key, c = alive[0]
            text = (f"{job.name}'s process ({_key_text(key)}, pid {c.get('pid') or '?'}) is "
                    f"{c.get('state')} on {s.host} although the job ended {job.state}")
            if self.zombies.get(job.id) != text:
                log.error("RUN QUEUE: %s", text)
                self.q._record("run_queue_process_after_end", job=job.id, key=_key_text(key),
                               pid=c.get("pid"), host=s.host, state=job.state)
            self.zombies[job.id] = text
        else:
            self.zombies.pop(job.id, None)
        return changed

    def _take_kills(self, job, s: Session, kills) -> bool:
        if not isinstance(kills, dict):
            return False
        changed = False
        for key, k in kills.items():
            if not isinstance(k, dict):
                continue
            want = self.kills.get((job.id, str(key)))
            if want is None or k.get("seq") != want or not isinstance(k.get("result"), dict):
                continue
            del self.kills[(job.id, str(key))]
            r = dict(k["result"])
            r.setdefault("role", "shell")
            r["host"] = s.host
            ok = bool(r.get("ok") or r.get("gone"))
            note = ((f"killed on {s.host}" if r.get("ok") else
                     f"nothing to kill on {s.host}: {r.get('note')}") if ok else
                    f"NOT killed on {s.host}: {r.get('note')}")
            if job.kill is not None and job.state not in _ended():
                job.kill.update(results=list(job.kill.get("results") or []) + [r], note=note,
                                pending=False)
            w = job.warm or {}
            if w.get("token") == str(key) and w.get("state") == "aborting":
                w["kill_note"] = note
            self.q._record("run_queue_kill_result", job=job.id, key=_key_text(str(key)),
                           result=r, host=s.host)
            (log.warning if ok else log.error)("Run queue: %s: kill on %s: %s", job.name,
                                               s.host, r.get("note"))
            changed = True
        return changed

    def _take_log(self, job, s: Session, entry) -> int | None:
        """Append the launcher's new log text to the job's log on this
        machine (the queue reads it there).  -> the next byte of the
        launcher's log wanted.  The launcher's offset is kept beside the log
        (``<log>.rpos``), so a server restart neither loses nor repeats text."""
        if not job.log_path:
            if not isinstance(entry, dict) or not entry.get("text"):
                return s.log_ack
            job.log_path = self.q._path("logs", f"{job.id}_{job.label}.out")
        if s.log_ack is None:
            s.log_ack = _read_rpos(job.log_path)
        if not isinstance(entry, dict):
            return s.log_ack
        try:
            offset, end = int(entry.get("offset")), int(entry.get("end"))
        except (TypeError, ValueError):
            return s.log_ack
        text = str(entry.get("text") or "")
        if offset != s.log_ack or end < offset:
            return s.log_ack                          # the launcher re-reads from here
        if end > offset:
            try:
                os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
                with open(job.log_path, "ab") as f:
                    f.write(text.encode("utf-8"))
                _write_rpos(job.log_path, end)
                s.log_ack = end
            except OSError as exc:
                log.error("Run queue: %s: could not append to its log %s (%s); the launcher "
                          "sends it again.", job.name, job.log_path, exc)
        return s.log_ack

    def _acks(self, job, s: Session) -> list:
        """Children whose end the queue has taken (or never will need): the
        launcher may forget them."""
        current = self._current_keys(job)
        out = []
        for key, c in s.children.items():
            if c.get("state") not in TERMINAL_STATES:
                continue
            if (job.id, key) in self.consumed or key not in current:
                out.append(key)
        return out

    def _ran(self, job) -> bool:
        """The queue knows a process of this job ran (it is never started
        again by an order)."""
        return bool(job.pid or job.run_id is not None or job.client_pid)

    def _order(self, job, s: Session) -> dict | None:
        """The launch or warm order the job's saved state gives (None when
        none): derived on every heartbeat, so it survives either side's
        restart without ever starting a child twice."""
        if job.id in self.gone:
            return None
        w = job.warm or {}
        key = None
        if (job.state in ("launching", "running") and not w.get("go") and COLD not in s.children
                and (job.id, COLD) not in self.kills and not self._ran(job)):
            key = COLD
        elif (job.state == "queued" and w.get("token") and not w.get("go")
              and w.get("state") in ("warming", "ready") and str(w["token"]) not in s.children):
            key = str(w["token"])
        if key is None:
            return None
        return {"key": key, "path": job.path, "argv": list(job.argv), "cwd": job.cwd,
                "env": self.q._job_env_vars(job, None if key == COLD else key),
                "below_normal": key != COLD, "allow_drift": bool(job.allow_drift),
                "sha256": job.sha256}

    def _done(self, job, s: Session) -> bool:
        """The launcher may forget the job: it has ended and nothing of it is
        alive or unacknowledged."""
        if job.state not in _ended():
            return False
        for key, c in s.children.items():
            if c.get("state") in RUNNING_STATES:
                return False
        return not any(j == job.id for j, _ in self.kills)

    # -- the server's own host ---------------------------------------------------------------

    def tick(self) -> None:
        """Start ``kq launch`` on this machine for this host's jobs that have
        no launcher (a run-again of one of its jobs, a launcher that died) --
        at most every :data:`SELF_START_EVERY_S` per job, never during the
        restart grace."""
        if self.in_grace():
            return
        now = self._clock()
        pairs = []
        with self.q._lock:
            for job in self.q._jobs.values():
                if job.state in _ended() or not job.run_host:
                    continue
                if job.run_host.lower() != self.host.lower() or self.attached(job):
                    continue
                if (self.silent_for(job.id) is None
                        and now - float(job.submitted_at or 0.0) < SELF_START_AFTER_S):
                    continue                          # its submitter's launcher is coming
                if now - self._self_started.get(job.id, float("-inf")) < SELF_START_EVERY_S:
                    continue
                self._self_started[job.id] = now
                pairs.append((job.id, job.token))
        if not pairs:
            return
        try:
            self._start_launcher(pairs)
            self.q._record("run_queue_launcher_started_here", jobs=[p[0] for p in pairs],
                           host=self.host)
            log.info("Run queue: started a launcher here for job(s) %s (no launcher attached).",
                     ", ".join(str(p[0]) for p in pairs))
        except Exception as exc:                      # noqa: BLE001
            log.error("Run queue: could not start a launcher here for job(s) %s: %r",
                      ", ".join(str(p[0]) for p in pairs), exc)
            self.q._record("run_queue_launcher_start_failed", jobs=[p[0] for p in pairs],
                           error=repr(exc))


def _ended():
    from waxx.util.device_state.run_queue import ENDED  # noqa: PLC0415
    return ENDED


def _key_text(key: str) -> str:
    """A child key for records: "cold", or "warm" (the warm token is the warm
    process's credential and never goes in a record)."""
    return COLD if key == COLD else "warm"


def _read_rpos(log_path: str) -> int:
    try:
        with open(log_path + ".rpos", encoding="utf-8") as f:
            return int(json.load(f).get("offset") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def _write_rpos(log_path: str, offset: int) -> None:
    tmp = log_path + ".rpos.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"offset": int(offset)}, f)
    os.replace(tmp, log_path + ".rpos")
