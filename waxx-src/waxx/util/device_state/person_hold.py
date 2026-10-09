"""A person's hold on the machine, kept by the monitor server.

While the hold is on, nothing an *agent* asked for starts: the run queue's
agent jobs wait (:mod:`~waxx.util.device_state.run_queue`), the server's run
loops do not start or continue (their gate reports it), and
:func:`waxx.util.device_state.run_gate.assess` -- the agents' occupancy check --
reports the machine busy.  A person's own queued runs still go.

It is set

* by a request -- ``{"type": "run_queue", "action": "hold", "reason", "by"}``,
  e.g. the Device Control GUI's "Hold -- a person has the machine" button on
  the Sequences tab;
* by the server itself when liveOD's ``reset_requested`` turns on (the Reset /
  Abort button) for a run that the run queue did NOT launch for an agent -- a
  person's run, a run loop's run, or no run at all.  An agent resetting its own
  run, and the queue's own Abort of a job it was asked to cancel, do not set it.

and released only by a request (``"action": "release"``).  It never expires on
its own: while it is on, the server logs a reminder every
:data:`REMINDER_S`.  It is kept in a small JSON file next to the run queue's,
so a server restart does not drop it.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable, Iterable

log = logging.getLogger(__name__)

#: A reminder line in the server log this often while the hold is on.
REMINDER_S = 1800.0
#: Who sets the hold when liveOD's Reset is pressed.
LIVE_OD_RESET_BY = "liveOD"


def _clock_text(t) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(t)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


class PersonHold:
    """The hold's state and its rules.

    ``info()`` is what ``status_json`` serves as ``person_hold``:
    ``{"active": bool, "since": epoch | None, "by": str, "reason": str,
    "source": "request" | "live_od_reset" | "", "run_id": int | None}``
    (``run_id``: the run liveOD had when its Reset set the hold).

    ``path``: where the hold is kept across restarts (None: memory only).
    ``journal``: an :class:`~waxx.util.device_state.op_journal.OpJournal`
    (records ``run_queue_hold`` / ``run_queue_release``); ``on_change(info)``
    is called on every change, from whichever thread made it."""

    def __init__(self, path: str | None = None, *, journal=None,
                 on_change: Callable[[dict], None] | None = None,
                 clock: Callable[[], float] = time.time, reminder_s: float = REMINDER_S):
        self.path = path
        self._journal = journal
        self._on_change = on_change
        self._clock = clock
        self._reminder_s = float(reminder_s)
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._save_seq = self._saved_seq = 0
        self._s = {"active": False, "since": None, "by": "", "reason": "", "source": "",
                   "run_id": None}
        self._last_reminder: float | None = None
        #: liveOD's reset_requested at the last POLL seen (None: none seen yet)
        self._last_reset: bool | None = None
        self._load()

    # -- state ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._s["active"])

    def info(self) -> dict:
        with self._lock:
            return dict(self._s)

    def text(self) -> str:
        """"person hold since 14:03:11 by jp@kong: reason" ("" when off)."""
        s = self.info()
        if not s["active"]:
            return ""
        return describe(s)

    # -- requests -----------------------------------------------------------------

    def hold(self, reason: str = "", by: str = "", *, source: str = "request",
             run_id=None) -> dict:
        """Put the hold on.  Already on: nothing changes (the first hold's
        since/by/reason stay) and the reply says ``already``."""
        reason = str(reason or "").strip() or "a person has the machine"
        by = str(by or "").strip() or "?"
        with self._lock:
            if self._s["active"]:
                return {"status": "ok", "already": True, "person_hold": dict(self._s)}
            self._s = {"active": True, "since": self._clock(), "by": by, "reason": reason,
                       "source": source, "run_id": run_id}
            self._last_reminder = self._s["since"]
            info = dict(self._s)
        log.warning("PERSON HOLD on: %s -- agents' runs wait until it is released.",
                    describe(info))
        self._record("run_queue_hold", by=by, reason=reason, source=source, run_id=run_id)
        self._save()
        self._notify()
        return {"status": "ok", "person_hold": info}

    def release(self, by: str = "") -> dict:
        by = str(by or "").strip() or "?"
        with self._lock:
            if not self._s["active"]:
                return {"status": "error", "msg": "no person hold is on"}
            held = dict(self._s)
            self._s = {"active": False, "since": None, "by": "", "reason": "", "source": "",
                       "run_id": None}
            self._last_reminder = None
            info = dict(self._s)
        log.warning("Person hold released by %s (it was %s).", by, describe(held))
        self._record("run_queue_release", by=by, held_since=held["since"],
                     held_by=held["by"], reason=held["reason"])
        self._save()
        self._notify()
        return {"status": "ok", "person_hold": info, "released": held}

    # -- the server's ticks -------------------------------------------------------

    def tick(self) -> None:
        """A reminder line every ``reminder_s`` while the hold is on."""
        with self._lock:
            if not self._s["active"]:
                return
            now = self._clock()
            last = self._last_reminder if self._last_reminder is not None else self._s["since"]
            if now - float(last or now) < self._reminder_s:
                return
            self._last_reminder = now
            info = dict(self._s)
        log.warning("Reminder: the PERSON HOLD is still on (%s, %.0f min) -- agents' runs "
                    "wait until someone releases it.", describe(info),
                    (now - float(info["since"] or now)) / 60.0)

    def observe_poll(self, poll: dict | None, agent_run_ids: Iterable = (),
                     own_abort_ids: Iterable = ()) -> bool:
        """Read one liveOD POLL reply: when ``reset_requested`` has just turned
        on and the run it is for is not one the queue launched for an agent
        (``agent_run_ids``) nor one the queue itself aborted (``own_abort_ids``),
        put the hold on.  The first POLL seen only sets the baseline (a Reset
        already pending when the server starts is not taken as a new one).
        Returns True when it set the hold."""
        if not isinstance(poll, dict) or poll.get("ok") is False:
            return False
        reset = bool(poll.get("reset_requested"))
        previous, self._last_reset = self._last_reset, reset
        if previous is None or not reset or previous:
            return False
        run_id = poll.get("run_id") if poll.get("run_in_progress") else None
        if run_id is not None and run_id in set(own_abort_ids or ()):
            log.info("liveOD Reset for run %s: the run queue's own Abort (a cancel) -- no "
                     "person hold.", run_id)
            return False
        if run_id is not None and run_id in set(agent_run_ids or ()):
            log.info("liveOD Reset for run %s, an agent's queued run -- no person hold.",
                     run_id)
            return False
        if self.active:
            return False
        self.hold(f"Reset in liveOD at {_clock_text(self._clock())}", LIVE_OD_RESET_BY,
                  source="live_od_reset", run_id=run_id)
        return True

    # -- persistence -------------------------------------------------------------

    def _load(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except Exception as exc:                      # noqa: BLE001
            log.error("Could not read the person hold from %s (%s): starting with NO hold -- "
                      "check whether one was on.", self.path, exc)
            return
        if isinstance(data, dict) and data.get("active"):
            self._s = {"active": True, "since": data.get("since"), "by": str(data.get("by") or ""),
                       "reason": str(data.get("reason") or ""),
                       "source": str(data.get("source") or ""), "run_id": data.get("run_id")}
            self._last_reminder = self._clock()
            log.warning("PERSON HOLD still on from before the server restarted: %s",
                        describe(self._s))

    def _save(self) -> None:
        """Write the hold's file.  A hold and a release racing each other:
        each snapshot is numbered under the hold's lock, and an older one is
        never written after a newer one."""
        if not self.path:
            return
        with self._lock:
            self._save_seq += 1
            seq, data = self._save_seq, dict(self._s)
        with self._save_lock:
            if seq <= self._saved_seq:
                return
            try:
                from waxx.util.device_state.state_file_io import atomic_write  # noqa: PLC0415
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                atomic_write(self.path, data)
                self._saved_seq = seq
            except Exception as exc:                  # noqa: BLE001
                log.error("Could not store the person hold in %s (%s); a server restart "
                          "would drop it.", self.path, exc)

    # -- out --------------------------------------------------------------------

    def _record(self, kind: str, **fields) -> None:
        if self._journal is None:
            return
        try:
            self._journal.record(kind, **fields)
        except Exception:                             # noqa: BLE001
            log.exception("Could not journal %s", kind)

    def _notify(self) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(self.info())
        except Exception:                             # noqa: BLE001
            log.exception("Person hold change notification failed")


def describe(info: dict | None) -> str:
    """"person hold since 14:03:11 by jp@kong: reason" for a ``person_hold``
    dict (from ``status_json`` or :meth:`PersonHold.info`)."""
    info = info or {}
    return (f"person hold since {_clock_text(info.get('since'))} by "
            f"{info.get('by') or '?'}: {info.get('reason') or 'a person has the machine'}")
