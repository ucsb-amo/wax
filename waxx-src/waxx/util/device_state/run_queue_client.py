"""Client side of the run queue (:mod:`waxx.util.device_state.run_queue`).

The run queue lives in the monitor server.  :class:`RunQueueClient` finds that
server by UDP discovery (through
:class:`~waxx.util.comms_server.comm_client.MonitorClient`, the hardware-scoped
``monitor`` beacon) and sends it ``{"type": "run_queue", "action": ...}``
requests as JSON over TCP.  One method per action; each returns the server's
reply dict and raises :class:`RunQueueError` when the server refuses
(``{"status": "error", "msg": ...}``) or does not answer.

* No monitor server beaconing, or one without a run queue (an older version):
  :class:`NoRunQueue` (a ``RuntimeError``).  The client never falls back to
  running anything itself.
* Requests that change something (submit, cancel, pause, resume, hold,
  release) are sent once: a lost reply is reported, never retried, because
  the server may have acted on the first copy.  Reads (list, describe, tail,
  status) may be retried once after a rediscovery.
* :meth:`RunQueueClient.follow` follows one job: it waits while the job is
  queued, then copies the job's log (on the server's disk) to ``out`` line by
  line through the ``tail`` action until the job has ended and the log is
  read to its end, and returns the final job record.  The server launches the
  job; the client only watches.  Nothing here starts, stops or kills a
  process.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import getpass
import os
import socket
import sys
import time
from typing import Callable, Iterable

#: States a job ends in (as run_queue.ENDED; repeated so a client does not
#: import the server's module).
ENDED = ("saved", "failed", "cancelled", "skipped")
#: States of the job in the queue's one slot ("launching": between the
#: queue's record of the launch and the process starting).
IN_SLOT = ("launching", "running", "ending")
STATES = ("queued",) + IN_SLOT + ENDED

NO_QUEUE_MSG = "no run queue (monitor server) is beaconing"

#: The environment variable naming who a client acts for ("agent" for an
#: agent; anything else, or unset, is a person).  The queue sets it for its
#: own jobs; the agents' skill sets it for agents.
OWNER_ENV = "WAXX_OWNER"
#: An agent's name for its jobs (the queue shows it as the submitter).
AGENT_LABEL_ENV = "WAXX_AGENT_LABEL"


def owner_from_env() -> str:
    """"agent" when ``WAXX_OWNER`` is ``agent``, else "person"."""
    return "agent" if os.environ.get(OWNER_ENV, "").strip().lower() == "agent" else "person"


class RunQueueError(RuntimeError):
    """The server refused a request, or did not answer.  ``reply`` is the
    server's reply dict (None when there was none)."""

    def __init__(self, msg: str, reply: dict | None = None):
        super().__init__(msg)
        self.msg = msg
        self.reply = reply


class NoRunQueue(RuntimeError):
    """No monitor server is beaconing, or the one found has no run queue."""


def default_by() -> str:
    """``user@host`` for the ``by`` field."""
    try:
        user = getpass.getuser()
    except Exception:                                 # noqa: BLE001
        user = "?"
    return f"{user}@{socket.gethostname()}"


def safe_write(out, text: str) -> None:
    """Write ``text`` to ``out``; characters its encoding cannot carry become
    replacement characters instead of raising (a console or pipe in cp1252 or
    ASCII never ends a follow)."""
    try:
        out.write(text)
    except UnicodeEncodeError:
        enc = getattr(out, "encoding", None) or "ascii"
        out.write(text.encode(enc, "replace").decode(enc, "replace"))


def local_sha256(path) -> str | None:
    """SHA-256 of this machine's copy of ``path``; None when it cannot be read.
    Sent with a submit: the server runs ITS copy at that path and refuses one
    that differs (or, from another PC, one this client could not hash)."""
    import hashlib  # noqa: PLC0415
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 16), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def connect_monitor(discovery_timeout: float = 3.0):
    """The monitor server's client, found by discovery (imported here: the
    discovery module starts its listener when it is imported).  RuntimeError
    when no monitor server answers."""
    from waxx.util.comms_server.comm_client import MonitorClient  # noqa: PLC0415
    return MonitorClient(discovery_timeout=discovery_timeout)


class RunQueueClient:
    """The run queue's requests.

    ``transport``: an object with ``request(obj, timeout=, attempts=)`` ->
    reply dict or None, and ``get_status()`` -> the ``status_json`` dict or
    None (a :class:`~waxx.util.comms_server.comm_client.MonitorClient`).  When
    not given, a MonitorClient is made by discovery (up to
    ``discovery_timeout`` s); :class:`NoRunQueue` when none answers.
    ``by``: who is asking (default ``user@host``).  ``connect``: makes the
    transport from ``discovery_timeout`` (default :func:`connect_monitor`).
    ``sleep``: the wait between polls in :meth:`follow`; ``wait_poll_s``: its
    default interval between ``describe`` calls (why a job waits, a cancel).  ``owner``: who the
    client acts for, sent with every request that changes something
    ("person" | "agent"; default from ``WAXX_OWNER``, see
    :func:`owner_from_env`)."""

    def __init__(self, transport=None, *, discovery_timeout: float = 3.0,
                 by: str | None = None, timeout: float = 8.0,
                 connect: Callable[[float], object] | None = None,
                 sleep: Callable[[float], None] = time.sleep, owner: str | None = None,
                 wait_poll_s: float = 5.0):
        self.sleep = sleep
        self.wait_poll_s = float(wait_poll_s)
        self.owner = owner or owner_from_env()
        if transport is None:
            try:
                transport = (connect or connect_monitor)(discovery_timeout)
            except RuntimeError as exc:
                raise NoRunQueue(f"{NO_QUEUE_MSG} ({exc})") from exc
        self.transport = transport
        self.by = by or default_by()
        self.timeout = float(timeout)

    # -- transport ------------------------------------------------------------------

    def _ask(self, action: str, fields: dict, *, retry: bool) -> dict:
        obj = {"type": "run_queue", "action": action}
        obj.update({k: v for k, v in fields.items() if v is not None})
        reply = self.transport.request(obj, timeout=self.timeout, attempts=2 if retry else 1)
        if reply is None:
            msg = f"the monitor server did not answer the run queue's {action} request"
            if not retry:
                msg += " (it may have acted on it: check with kq list)"
            raise RunQueueError(msg)
        if reply.get("status") != "ok":
            msg = str(reply.get("msg") or reply)
            if "unknown type" in msg and "run_queue" in msg:
                raise NoRunQueue("the monitor server found has no run queue (an older "
                                 "version)")
            raise RunQueueError(msg, reply)
        return reply

    # -- requests ---------------------------------------------------------------------

    def _owner(self, owner: str | None) -> str:
        return owner or self.owner

    def submit(self, path: str, *, argv: Iterable[str] = (), cwd: str | None = None,
               label: str | None = None, owner: str | None = None,
               priority: int | None = None, due: float | None = None,
               after: Iterable[int] = (), repeat: int = 1, chain: str | None = None,
               stop_on_failure: bool | None = None, write_back: bool | None = None,
               allow_drift: bool = False, at_end: bool = False,
               at_index: int | None = None, before_id: int | None = None,
               after_id: int | None = None) -> dict:
        """Queue ``path`` (absolute on the server's machine; a relative path is
        made absolute here).  Placement: by default the server's owner rule (a
        person's job ahead of every queued agent job); ``at_end`` opts out;
        ``at_index`` (0-based among the queued jobs) / ``before_id`` /
        ``after_id`` place it there (the ``insert`` action; give at most one).
        ``priority`` (None: not sent) is only a placement hint within the
        owner's block.  ``write_back`` may only be None or False (a veto).
        ``agent_label`` comes from ``WAXX_AGENT_LABEL`` when set; ``host`` is
        this machine's name.  ``client_sha256`` (this machine's hash of the
        file, when readable) and ``client_host`` let the server refuse a path
        whose copy on its machine -- the one that runs -- differs.  ``cwd`` is
        made absolute here.  -> ``{"status": "ok", "ids": [...], "jobs": [...]}``."""
        position = {"at_index": None if at_index is None else int(at_index),
                    "before_id": None if before_id is None else int(before_id),
                    "after_id": None if after_id is None else int(after_id)}
        action = "insert" if any(v is not None for v in position.values()) else "submit"
        path = os.path.abspath(str(path))
        return self._ask(action, dict({
            "path": path, "argv": [str(a) for a in argv],
            # the file that runs is the server's copy: it must match this one
            "client_sha256": local_sha256(path), "client_host": socket.gethostname(),
            "cwd": None if cwd is None else os.path.abspath(str(cwd)),
            "label": label, "owner": self._owner(owner),
            "priority": None if priority is None else int(priority),
            "due": None if due is None else float(due), "after": [int(a) for a in after],
            "repeat": int(repeat), "chain": chain, "stop_on_failure": stop_on_failure,
            "write_back": write_back, "allow_drift": bool(allow_drift),
            "at_end": True if at_end else None,
            "agent_label": os.environ.get(AGENT_LABEL_ENV, "").strip() or None,
            "host": socket.gethostname(), "by": self.by}, **position), retry=False)

    def move(self, job_id: int, *, to_index: int | None = None, before_id: int | None = None,
             after_id: int | None = None, token: str | None = None,
             owner: str | None = None) -> dict:
        """Put queued job ``job_id`` elsewhere in the order: ``to_index``
        (0-based among the queued jobs) or ``before_id`` / ``after_id`` (one).
        An agent may move only agent jobs.  -> ``{"job", "position"}``."""
        return self._ask("move", {"id": int(job_id), "token": token,
                                  "to_index": None if to_index is None else int(to_index),
                                  "before_id": None if before_id is None else int(before_id),
                                  "after_id": None if after_id is None else int(after_id),
                                  "owner": self._owner(owner), "by": self.by}, retry=False)

    def edit(self, job_id: int, fields: dict, *, token: str | None = None,
             owner: str | None = None) -> dict:
        """Change a queued job: ``fields`` from argv, label, after, chain,
        stop_on_failure, write_back (None | False), due (epoch | None),
        allow_drift, paused.  An agent may edit only agent jobs.  -> ``{"job",
        "changed": [names]}``."""
        return self._ask("edit", {"id": int(job_id), "token": token, "fields": dict(fields),
                                  "owner": self._owner(owner), "by": self.by}, retry=False)

    def cancel(self, job_id: int, *, token: str | None = None, owner: str | None = None,
               queued_only: bool = False) -> dict:
        """Cancel a job.  A queued job ends at once; a running one gets
        liveOD's Abort (its data file is discarded) and the reply has
        ``pending``.  ``queued_only``: refuse (RunQueueError) when the job is
        no longer queued, so a job that has just launched is not aborted by a
        cancel meant for a queued one.  ``owner``: who is cancelling (default
        this client's owner; an agent may not cancel a person's job)."""
        return self._ask("cancel", {"id": int(job_id), "token": token,
                                    "owner": self._owner(owner),
                                    "queued_only": True if queued_only else None,
                                    "by": self.by}, retry=False)

    def list(self, states: Iterable[str] | None = None, limit: int | None = None) -> dict:
        """-> ``{"jobs": [...] in the server's order (ended by id, the slot,
        then the queued in rank order), "next": [ids], "run_queue": info}``."""
        return self._ask("list", {"states": list(states) if states else None,
                                  "limit": limit}, retry=True)

    def describe(self, job_id: int, token: str | None = None) -> dict:
        """-> ``{"job": {...}, "waiting": str, "tail": [lines]}``."""
        return self._ask("describe", {"id": int(job_id), "token": token}, retry=True)

    def job_token(self, job_id: int) -> str | None:
        """Job ``job_id``'s token, from ``describe`` (for a follower that was
        given only the id: from then on every ``tail`` names this job, not
        another queue's job that reused the id)."""
        return self.describe(job_id)["job"].get("token")

    def tail(self, job_id: int, token: str | None = None, offset: int = 0) -> dict:
        """The job's log from byte ``offset`` -> ``{"lines", "offset", "done",
        "state", "run_id"}`` (read on the server's machine)."""
        return self._ask("tail", {"id": int(job_id), "token": token, "offset": int(offset)},
                         retry=True)

    def pause(self, scope: str = "agent", reason: str = "", *,
              owner: str | None = None) -> dict:
        return self._ask("pause", {"scope": scope, "reason": reason,
                                   "owner": self._owner(owner), "by": self.by}, retry=False)

    def resume(self, scope: str = "agent", *, owner: str | None = None) -> dict:
        return self._ask("resume", {"scope": scope, "owner": self._owner(owner),
                                    "by": self.by}, retry=False)

    def hold(self, reason: str = "", *, owner: str | None = None) -> dict:
        return self._ask("hold", {"reason": reason, "owner": self._owner(owner),
                                  "by": self.by}, retry=False)

    def release(self, *, owner: str | None = None) -> dict:
        return self._ask("release", {"owner": self._owner(owner), "by": self.by},
                         retry=False)

    def status(self) -> dict:
        """The server's ``status_json`` (``run_queue``, ``person_hold``, the
        monitor's state); NoRunQueue when it has no run queue."""
        status = self.transport.get_status()
        if status is None:
            raise RunQueueError("the monitor server did not answer status_json")
        if "run_queue" not in status:
            raise NoRunQueue("the monitor server found has no run queue (an older version)")
        return status

    # -- following a job ----------------------------------------------------------------

    def follow(self, job_id: int, token: str | None = None, out=None, poll_s: float = 0.5,
               *, cursor: dict | None = None, on_wait: Callable[[dict], None] | None = None,
               wait_poll_s: float | None = None, lost_s: float = 600.0,
               on_lost: Callable[[str], None] | None = None,
               on_abort: Callable[[dict], None] | None = None,
               sleep: Callable[[float], None] | None = None,
               clock: Callable[[], float] = time.monotonic) -> dict:
        """Copy job ``job_id``'s log to ``out`` (default stdout) until the job
        has ended and its log is read to the end; return the final job record.

        ``cursor``: a dict holding ``offset`` (the next byte to read), updated
        as lines are written -- pass the same dict to a second follow to go on
        where an interrupted one stopped.  ``on_wait(describe_reply)`` is called
        while the job is queued, at most every ``wait_poll_s`` s (the caller
        prints why it waits).  A server that does not answer is retried; after
        ``lost_s`` s without an answer RunQueueError is raised.  ``on_lost(msg)``
        is called once when answers stop and once when they come back.
        ``on_abort(job)`` is called once (``cursor["abort_reported"]``) when the
        job in the slot has a cancel asked -- the queue's tick sends liveOD's
        Abort; the follow goes on until the run ends -- looked for with
        ``describe`` at most every ``wait_poll_s`` s."""
        out = sys.stdout if out is None else out
        sleep = sleep or self.sleep
        wait_poll_s = self.wait_poll_s if wait_poll_s is None else float(wait_poll_s)
        cursor = cursor if cursor is not None else {}
        cursor.setdefault("offset", 0)
        last_wait = None
        lost_since = None
        while True:
            try:
                if not token:                         # given only the id: look it up once
                    token = self.job_token(job_id)
                reply = self.tail(job_id, token, cursor["offset"])
            except RunQueueError as exc:
                if exc.reply is not None:             # a refusal, not silence
                    raise
                now = clock()
                if lost_since is None:
                    lost_since = now
                    if on_lost is not None:
                        on_lost(f"the monitor server is not answering ({exc.msg}); still "
                                "trying -- the job itself is not affected")
                elif now - lost_since > lost_s:
                    raise RunQueueError(f"the monitor server has not answered for "
                                        f"{lost_s:.0f} s; the job goes on without this "
                                        f"terminal") from exc
                sleep(max(poll_s, 1.0))
                continue
            if lost_since is not None:
                lost_since = None
                if on_lost is not None:
                    on_lost("the monitor server answers again")
            lines = reply.get("lines") or []
            # the cursor moves before the lines are written: a Ctrl-C while
            # writing loses the rest of this batch on a resumed follow (kq
            # tail shows them) rather than printing any line twice
            cursor["offset"] = int(reply.get("offset") or cursor["offset"])
            cursor["state"] = reply.get("state")
            cursor["run_id"] = reply.get("run_id")
            for line in lines:
                safe_write(out, line + "\n")
            if lines:
                try:
                    out.flush()
                except Exception:                     # noqa: BLE001
                    pass
            if reply.get("done"):
                break
            state = reply.get("state")
            watch = ((state == "queued" and on_wait is not None)
                     or (state in IN_SLOT and on_abort is not None
                         and not cursor.get("abort_reported")))
            if watch:
                now = clock()
                if last_wait is None or now - last_wait >= wait_poll_s:
                    last_wait = now
                    try:
                        described = self.describe(job_id, token)
                    except RunQueueError as exc:
                        if exc.reply is not None:
                            raise
                        described = None              # silence: the tail retry handles it
                    if described is not None and state == "queued":
                        on_wait(described)
                    elif described is not None and (described.get("job") or {}).get("cancel"):
                        cursor["abort_reported"] = True
                        on_abort(described["job"])
            if not lines:
                sleep(poll_s)
        return self._final_job(job_id, token, cursor, sleep, poll_s)

    def _final_job(self, job_id, token, cursor, sleep, poll_s, tries: int = 5) -> dict:
        """The ended job's record from ``describe``, asked a few times; when
        the server stays silent, a record built from the last ``tail`` reply
        (its state and run id; ``final_record_missing`` set) -- the job has
        ended either way, so a lost reply must not read as a refusal."""
        why = ""
        for i in range(tries):
            try:
                return self.describe(job_id, token)["job"]
            except RunQueueError as exc:
                if exc.reply is not None:
                    raise
                why = exc.msg
                if i < tries - 1:
                    sleep(max(poll_s, 1.0))
        return {"id": job_id, "token": token, "label": "?", "state": cursor.get("state"),
                "run_id": cursor.get("run_id"), "exit_code": None,
                "reason": f"its final record could not be read ({why}); kq show {job_id}",
                "final_record_missing": True}
