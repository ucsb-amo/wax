"""May a run start now?  One verdict, shared by every launcher.

The machine is busy while liveOD has a run, or while the monitor server holds a
run's announcement (its *run fence*: the run finished ``prepare()`` and has not
taken the core yet).  Until 2026-10-09 each launcher read those signals its own
way, and a run whose process had been killed (85528: killed with its client,
liveOD left with ``run_in_progress`` and a pending Abort) blocked every gate
until a person stepped in.  :func:`classify` is the one reading now, used by
the monitor server's run loop (:mod:`~waxx.util.device_state.run_loop`) and by
the agents' ``occupancy.py``.

Inputs (both plain dicts, so the rules are testable without a network):

* liveOD's POLL reply -- ``run_in_progress``, ``run_id``, ``reset_requested``,
  ``init_run_age_s``, ``last_shot_age_s``, ``n_shots``, ``n_shots_expected``,
  ``run_state``, ``last_outcome``, and since 2026-10-09 the run's client:
  ``client_pid``, ``client_host``, ``launcher`` (sent by the experiment at
  INIT_RUN; ``None``/"" from an older client or server);
* the monitor server's ``status_json`` ``run_pending`` (the fence): ``run_id``,
  ``expt``, ``since`` (epoch seconds), ``client``, ``token``; ``None`` when no
  run is announced.

States (:class:`GateState`):

``free``           no run in progress, no Abort pending, no fence younger than
                   :data:`FENCE_TTL_S`.
``live``           a run is in progress and alive by its timing: a shot within
                   max(:data:`LIVE_MIN_S`, :data:`LIVE_SHOT_FACTOR` x its shot
                   period), or no shot yet within :data:`YOUNG_RUN_S` of its
                   INIT_RUN, or it is saving -- or a run has announced itself
                   to the monitor server (fence) and not started yet.  Never
                   waivable.
``reset_pending``  an Abort is pending in liveOD.  Waivable only when the
                   client's process is known to be dead (pid recorded, host is
                   this machine, process gone): liveOD finalizes that run at
                   the next INIT_RUN (its file is discarded, as for any abort).
``dead_client``    a run is in progress, no Abort, and its client's process is
                   known to be dead.  Waivable: the next INIT_RUN supersedes
                   the run (its file is closed with what it has, not deleted).
``wedged``         a run is in progress, nothing has happened for longer than
                   the limits above, and its process is alive or cannot be
                   checked.  NOT waivable: a person must look.
``unknown``        POLL failed, or its reply lacks ``run_in_progress``, or the
                   fence could not be read.  Not waivable: anything unknown
                   counts as busy.

A client's pid is only checked on this machine: ``client_host`` must equal
:func:`socket.gethostname` (case-insensitive).  On another host, or without a
recorded pid, the process counts as *unknown* -- never as dead.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

STATES = ("free", "live", "reset_pending", "dead_client", "wedged", "unknown")

#: A run with no shot yet counts as live for this long after its INIT_RUN
#: (compile, MOT load, warm-ups and the first shot).
YOUNG_RUN_S = 600.0
#: A run is live while its last shot is younger than this many shot periods
#: (period: INIT_RUN to the last shot, over the shots taken)...
LIVE_SHOT_FACTOR = 5.0
#: ...and never less than this.
LIVE_MIN_S = 120.0
#: The limit when POLL gives no shot period.
LIVE_DEFAULT_S = 600.0
#: The monitor server lets an announced run's fence lapse after this
#: (``MonitorServer.RUN_PENDING_TTL_S``); an older fence is not counted.
FENCE_TTL_S = 120.0

#: Launchers that set ``WAXX_LAUNCHER`` for the experiments they start.
KNOWN_LAUNCHERS = ("run_lock", "run_loop", "kq")
#: The environment variable an experiment's launcher sets; the experiment sends
#: it at INIT_RUN as ``launcher`` (``Expt._serialize_init_payload``).
LAUNCHER_ENV = "WAXX_LAUNCHER"


@dataclass
class GateState:
    """The verdict: ``state`` (one of :data:`STATES`), the run it is about,
    one line of why, whether a launcher may proceed anyway (``waivable``),
    and the inputs that decided it (``detail``)."""

    state: str
    run_id: int | None = None
    reason: str = ""
    waivable: bool = False
    detail: dict = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        """True when a launcher must not start a run."""
        return self.state != "free" and not self.waivable

    def to_dict(self) -> dict:
        return asdict(self)


# -- the client's process ----------------------------------------------------------

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87


def pid_alive(pid) -> bool:
    """Whether process ``pid`` exists on this machine.  Conservative: True
    unless the process is known to be gone.

    Windows: ``OpenProcess`` -- refused with "invalid parameter" means no such
    process; access denied means it exists (not ours); a handle whose exit code
    is not STILL_ACTIVE is a process that has exited.  Never ``os.kill(pid, 0)``
    there: on Windows that *terminates* the process.  Elsewhere:
    ``os.kill(pid, 0)``."""
    pid = int(pid)
    if pid <= 0:
        return True
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() != _ERROR_INVALID_PARAMETER
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    import os  # noqa: PLC0415
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


_default_pid_alive = pid_alive


def _client(poll: dict, alive: Callable[[int], bool]) -> dict:
    """The run's client as POLL reports it, and whether its process is alive:
    ``alive`` True / False / None (cannot tell, ``why`` says why)."""
    pid, host = poll.get("client_pid"), str(poll.get("client_host") or "")
    out = {"pid": pid, "host": host, "launcher": str(poll.get("launcher") or ""),
           "alive": None, "why": ""}
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        pid = None
    if pid is None or pid <= 0:
        out["why"] = "liveOD has no client pid for the run (older client or liveOD)"
        return out
    here = socket.gethostname()
    if not host or host.lower() != here.lower():
        out["why"] = f"the client ran on {host or 'an unknown host'}, not on {here}"
        return out
    try:
        out["alive"] = bool(alive(pid))
    except Exception as exc:                      # noqa: BLE001
        out["why"] = f"checking pid {pid} failed: {exc!r}"
        return out
    out["why"] = f"pid {pid} on {host} is {'alive' if out['alive'] else 'gone'}"
    return out


def _who(client: dict) -> str:
    text = f"pid {client['pid']} on {client['host']}"
    if client.get("launcher"):
        text += f", launched by {client['launcher']}"
    return text


#: run_states during which an older liveOD (no ``save_in_progress`` in POLL)
#: may be running an asynchronous END_RUN save.
_MAYBE_SAVING_STATES = ("saving", "aborting", "no_reply")


def _saving(poll: dict) -> bool:
    """liveOD is saving the run: ``save_in_progress`` (liveOD from 2026-10-09
    on), or ``run_state`` "saving" from an older one."""
    return bool(poll.get("save_in_progress")) or poll.get("run_state") == "saving"


def _num(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# -- the verdict ----------------------------------------------------------------------

def classify(poll: dict | None, fence: dict | None, *, now: float | None = None,
             pid_alive: Callable[[int], bool] | None = None) -> GateState:
    """The gate's verdict from liveOD's POLL reply and the monitor server's run
    fence (see the module docstring for the states and their rules).

    ``poll`` None, or a reply with ``ok`` False or without ``run_in_progress``,
    is ``unknown``.  ``fence`` None means no run is announced.  ``now`` (epoch
    seconds) dates the fence; ``pid_alive(pid)`` checks a client process on
    this machine (default: :func:`pid_alive`)."""
    now = time.time() if now is None else float(now)
    check = pid_alive if pid_alive is not None else _default_pid_alive
    if not isinstance(poll, dict) or poll.get("ok") is False:
        error = poll.get("error") if isinstance(poll, dict) else None
        return GateState("unknown", None,
                         "liveOD did not answer POLL" + (f" ({error})" if error else "")
                         + " -- cannot confirm the machine is free")
    if "run_in_progress" not in poll:
        return GateState("unknown", poll.get("run_id"),
                         "liveOD's POLL has no run_in_progress (liveOD predates it) -- "
                         "cannot confirm the machine is free")

    in_progress = bool(poll.get("run_in_progress"))
    reset = bool(poll.get("reset_requested"))
    rid = poll.get("run_id")
    if _saving(poll):
        # First, before any client check: a run being saved is live whatever its
        # process does (it has exited by design) and whatever run_state says (an
        # Abort pressed during the save shows "aborting"). Closing it as aborted
        # would delete the file being written.
        return GateState("live", rid, f"run {rid} is being saved by liveOD",
                         detail={k: poll.get(k) for k in (
                             "run_in_progress", "run_id", "reset_requested", "run_state",
                             "save_in_progress", "save_status")})
    client = _client(poll, check) if (in_progress or reset) else {
        "pid": poll.get("client_pid"), "host": poll.get("client_host"),
        "launcher": str(poll.get("launcher") or ""), "alive": None, "why": "not checked"}
    detail = {k: poll.get(k) for k in (
        "run_in_progress", "run_id", "reset_requested", "run_state", "expt_name", "n_shots",
        "n_shots_expected", "last_shot_age_s", "init_run_age_s")}
    detail["client"] = client

    fence_info = None
    if fence:
        since = _num(fence.get("since"))
        age = None if since is None else now - since
        fence_info = {"run_id": fence.get("run_id"), "expt": fence.get("expt") or "",
                      "age_s": age, "active": age is None or age < FENCE_TTL_S}
    detail["fence"] = fence_info

    if in_progress:
        if reset:
            if client["alive"] is False:
                st = GateState("reset_pending", rid,
                               f"run {rid}: an Abort is pending in liveOD and the run's process "
                               f"({_who(client)}) is gone -- liveOD finalizes the run at the next "
                               "INIT_RUN (its file is discarded, as for any abort)", True)
            else:
                st = GateState("reset_pending", rid,
                               f"run {rid}: an Abort is pending in liveOD, waiting for the run's "
                               f"answer ({client['why']})")
        elif client["alive"] is False:
            st = GateState("dead_client", rid,
                           f"run {rid} is in progress in liveOD but its process ({_who(client)}) "
                           "is gone -- the next INIT_RUN supersedes it (its file is closed with "
                           "what it has, not deleted)", True)
        else:
            st = _by_timing(poll, rid, client, detail)
    elif reset:
        last = poll.get("last_outcome") or {}
        if last.get("outcome") == "save_failed":
            # an older liveOD finalizes this Abort at the next INIT_RUN, which
            # deletes the failed save's file (kept for a retry)
            st = GateState("reset_pending", rid,
                           "an Abort is pending in liveOD with no run in progress, and run "
                           f"{last.get('run_id')}'s save failed (its file is kept for a "
                           "retry) -- not waived: a person must look")
        elif client["alive"] is False:
            st = GateState("reset_pending", rid,
                           "an Abort is pending in liveOD with no run in progress -- the next "
                           "INIT_RUN clears it", True)
        else:
            st = GateState("reset_pending", rid,
                           "an Abort is pending in liveOD with no run in progress -- the next "
                           "INIT_RUN clears it (not waived: the last run's process is not "
                           "known to be gone)")
    else:
        st = GateState("free", rid, "no run in progress in liveOD")

    if fence_info is not None:
        own = in_progress and fence_info["run_id"] is not None and fence_info["run_id"] == rid
        if fence_info["active"] and not own and (st.state == "free" or st.waivable):
            age = fence_info["age_s"]
            st = GateState("live", fence_info["run_id"],
                           f"run {fence_info['run_id']} ({fence_info['expt'] or 'experiment'}) "
                           "announced itself to the monitor server"
                           + (f" {age:.0f} s ago" if age is not None else "")
                           + " and has not started yet"
                           + (f" (liveOD: {st.reason})" if st.state != "free" else ""))
        elif not fence_info["active"] and st.state == "free":
            st.reason += (f"; run {fence_info['run_id']}'s fence on the monitor server is "
                          f"{fence_info['age_s']:.0f} s old (over {FENCE_TTL_S:.0f} s) -- "
                          "not counted")
    st.detail = detail
    return st


def _by_timing(poll: dict, rid, client: dict, detail: dict) -> GateState:
    """A run in progress whose process is alive or cannot be checked: live or
    wedged, by how long ago it last did something."""
    last = _num(poll.get("last_shot_age_s"))
    init = _num(poll.get("init_run_age_s"))
    n = _num(poll.get("n_shots"))
    name = f"run {rid}" + (f" ({poll.get('expt_name')})" if poll.get("expt_name") else "")
    process = f"its process: {client['why']}"
    if last is None:
        if init is None:
            return GateState("live", rid, f"{name} is in progress in liveOD (POLL gives no "
                             "timing)")
        detail["live_limit_s"] = YOUNG_RUN_S
        if init < YOUNG_RUN_S:
            return GateState("live", rid, f"{name} started {init:.0f} s ago, no shot yet")
        return GateState("wedged", rid,
                         f"WEDGED: {name} has taken no shot in the {init:.0f} s since its "
                         f"INIT_RUN (limit {YOUNG_RUN_S:.0f} s); {process}. Not waived -- a "
                         "person must look (Abort it in liveOD, or end its process)")
    period = None
    if init is not None and n and n >= 1 and init >= last:
        period = (init - last) / n
    limit = max(LIVE_MIN_S, LIVE_SHOT_FACTOR * period) if period else LIVE_DEFAULT_S
    detail["shot_period_s"] = period
    detail["live_limit_s"] = limit
    if last < limit:
        return GateState("live", rid, f"{name} is in progress in liveOD (last shot "
                         f"{last:.0f} s ago)")
    return GateState("wedged", rid,
                     f"WEDGED: {name} has taken no shot for {last:.0f} s (limit {limit:.0f} s"
                     + (f", {LIVE_SHOT_FACTOR:.0f} shot periods" if period else "")
                     + f"); {process}. Not waived -- a person must look (Abort it in liveOD, "
                     "or end its process)")


# -- fetching the inputs ------------------------------------------------------------------

def fetch_poll(live_od_client=None, timeout: float = 5.0) -> dict:
    """liveOD's POLL reply (raises when liveOD does not answer).  Without a
    client one is made (discovery) and closed again."""
    own = live_od_client is None
    if own:
        from waxx.util.live_od.live_od_client import LiveODClient  # noqa: PLC0415
        live_od_client = LiveODClient(timeout_ms=int(timeout * 1000),
                                      discovery_timeout=timeout)
    try:
        reply = live_od_client.poll()
    finally:
        if own:
            try:
                live_od_client.close()
            except Exception:                     # noqa: BLE001
                pass
    if not isinstance(reply, dict) or not reply.get("ok"):
        raise RuntimeError(f"liveOD POLL failed: {reply!r}")
    return reply


def fetch_monitor_status(monitor_client=None, timeout: float = 5.0) -> dict:
    """The monitor server's ``status_json`` (its run fence ``run_pending``, its
    ``state``, its ``run_loops``...).  Raises when the server does not answer."""
    if monitor_client is None:
        from waxx.util.comms_server.comm_client import MonitorClient  # noqa: PLC0415
        monitor_client = MonitorClient(discovery_timeout=timeout)
    status = monitor_client.get_status()
    if not isinstance(status, dict):
        raise RuntimeError("the monitor server did not answer status_json")
    return status


def fetch_fence(monitor_client=None, timeout: float = 5.0) -> dict | None:
    """The monitor server's run fence (``run_pending`` of ``status_json``), or
    None when no run is announced.  Raises when the server does not answer."""
    return fetch_monitor_status(monitor_client, timeout).get("run_pending") or None


#: A run loop in these states (``RunLoop`` "running" / "stopping": run_loop.ACTIVE)
#: launches runs of its own.
LOOP_ACTIVE_STATES = ("running", "stopping")


def active_loops(status: dict | None) -> list[str]:
    """The monitor server's run loops (e.g. the TOF loop) that are active, by
    title, from its ``status_json``: another launcher must not start a run
    between their runs."""
    loops = (status or {}).get("run_loops") or {}
    return [str(info.get("title") or key) for key, info in loops.items()
            if isinstance(info, dict) and info.get("state") in LOOP_ACTIVE_STATES]


def loops_verdict(status: dict | None) -> GateState | None:
    """``live`` (not waivable) while a run loop of the monitor server is active;
    None otherwise."""
    active = active_loops(status)
    if not active:
        return None
    return GateState("live", None,
                     f"{', '.join(active)} active on the monitor server -- stop it first",
                     detail={"run_loops": active})


def assess(live_od_client=None, monitor_client=None, timeout: float = 5.0, *,
           now: float | None = None, pid_alive: Callable[[int], bool] | None = None
           ) -> GateState:
    """Fetch POLL and the monitor's status and :func:`classify` them; an active
    run loop of the monitor server is busy too.  Either fetch failing gives
    ``unknown`` (counted as busy)."""
    try:
        poll = fetch_poll(live_od_client, timeout)
    except Exception as exc:                      # noqa: BLE001
        return GateState("unknown", None, f"liveOD did not answer POLL ({exc}) -- cannot "
                         "confirm the machine is free", detail={"error": str(exc)})
    try:
        status = fetch_monitor_status(monitor_client, timeout)
    except Exception as exc:                      # noqa: BLE001
        return GateState("unknown", poll.get("run_id"),
                         f"the monitor server did not answer ({exc}) -- cannot read its run "
                         "fence", detail={"error": str(exc), "poll_run_id": poll.get("run_id")})
    verdict = classify(poll, status.get("run_pending") or None, now=now, pid_alive=pid_alive)
    if verdict.state == "free" or verdict.waivable:
        loops = loops_verdict(status)
        if loops is not None:
            return loops
    return verdict


# -- telling liveOD a run's process has exited ------------------------------------------

def tell_live_od_run_exited(client, run_id, reason: str, *, poll: dict | None = None,
                            send: Callable[[int, str], dict] | None = None,
                            pid_alive: Callable[[int], bool] | None = None,
                            require_known_dead: bool = False) -> dict:
    """Send liveOD RUN_EXITED for run ``run_id`` on behalf of its process, which
    has exited -- only if that run is still liveOD's current run, in progress,
    and liveOD has not already heard of the exit (``run_state`` "exited").
    The process's own notice (an atexit handler) never ran when it was killed
    or crashed hard.  No run token is sent: liveOD matches the run id instead.

    Never sent while liveOD is saving the run (``run_state`` "saving": closing
    it then would cut the save short), nor when liveOD names the run's client
    process on this machine and that process is still alive (a launcher that
    ran the experiment through a shell has only seen the shell exit).

    ``client`` has ``poll()`` and ``_send_recv(msg)`` (a LiveODClient);
    ``poll`` is a POLL reply already in hand; ``send(run_id, reason)``, if
    given, sends the notice instead of ``client`` (the run loop's own sender);
    ``pid_alive`` as for :func:`classify`.  ``require_known_dead``: send only
    when liveOD's recorded client process is known to be gone (pid on this
    machine, process dead) -- for a caller that cannot be sure the experiment
    process itself has exited (it killed only the shell it started).
    Returns ``{"sent": bool, "ok": bool, "why": str, "reply": dict | None}``;
    raises only if polling or sending the notice itself fails."""
    if run_id is None:
        return {"sent": False, "ok": False, "why": "no run id", "reply": None}
    if poll is None:
        poll = client.poll()
    if not poll.get("run_in_progress"):
        return {"sent": False, "ok": False, "why": "liveOD has no run in progress",
                "reply": None}
    if poll.get("run_state") == "exited":
        return {"sent": False, "ok": False, "why": "liveOD already knows the run exited",
                "reply": None}
    if poll.get("run_id") != run_id:
        return {"sent": False, "ok": False,
                "why": f"liveOD's current run is {poll.get('run_id')}, not {run_id}",
                "reply": None}
    if poll.get("save_in_progress"):
        return {"sent": False, "ok": False, "why": "liveOD is saving the run", "reply": None}
    if "save_in_progress" not in poll and poll.get("run_state") in _MAYBE_SAVING_STATES:
        # an older liveOD does not say whether a save runs; during one an Abort
        # shows "aborting"/"no_reply", and RUN_EXITED then deletes the file
        return {"sent": False, "ok": False,
                "why": f"liveOD may be saving the run (run_state {poll.get('run_state')!r}, "
                       "no save_in_progress in POLL)", "reply": None}
    proc = _client(poll, pid_alive if pid_alive is not None else _default_pid_alive)
    if proc["alive"] is True:
        return {"sent": False, "ok": False,
                "why": f"the run's process is still alive ({proc['why']})", "reply": None}
    if require_known_dead and proc["alive"] is not False:
        return {"sent": False, "ok": False,
                "why": f"the run's process is not known to be gone ({proc['why']})",
                "reply": None}
    if send is not None:
        reply = send(int(run_id), str(reason))
    else:
        reply = client._send_recv({"tag": "RUN_EXITED", "run_id": int(run_id),
                                   "reason": str(reason)})
    ok = bool(isinstance(reply, dict) and reply.get("ok"))
    return {"sent": True, "ok": ok, "why": "sent" if ok else f"liveOD refused: {reply}",
            "reply": reply}
