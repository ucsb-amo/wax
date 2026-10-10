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
* by the server itself when a person presses liveOD's Reset / Abort: liveOD
  counts every Abort by source (POLL ``reset_counts`` / ``reset_count`` +
  ``last_reset``) and a person's count going up sets the hold -- whatever run
  it was for, and even if liveOD cleared the Abort again between two polls.
  The queue's own Abort (source "queue") and an agent's (source "agent",
  ``reset_liveod.py --own-run``) never set it.  A liveOD without the counts is
  watched by the level of ``reset_requested`` (see :meth:`observe_poll`).

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
_OFF = {"active": False, "since": None, "by": "", "reason": "", "source": "", "run_id": None,
        "owner": ""}

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
    "source": "request" | "live_od_reset" | "unreadable_file" | "", "run_id": int |
    None, "owner": "person" | "agent" | ""}`` (``run_id``: the run liveOD had
    when its Reset set the hold; ``owner``: who put it on -- an agent may not
    release a person's hold).

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
        self._s = dict(_OFF)
        self._last_reminder: float | None = None
        #: liveOD's reset_requested at the last POLL seen (None: none seen yet)
        #: -- the fallback for a liveOD without reset counts
        self._last_reset: bool | None = None
        #: liveOD's person Reset count at the last POLL seen (None: none yet)
        self._last_count: int | None = None
        self._warned_level = False
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
             run_id=None, owner: str = "person") -> dict:
        """Put the hold on (``owner``: who puts it on).  Already on: nothing
        changes (the first hold's since/by/reason/owner stay) and the reply
        says ``already``."""
        reason = str(reason or "").strip() or "a person has the machine"
        by = str(by or "").strip() or "?"
        with self._lock:
            if self._s["active"]:
                return {"status": "ok", "already": True, "person_hold": dict(self._s)}
            self._s = {"active": True, "since": self._clock(), "by": by, "reason": reason,
                       "source": source, "run_id": run_id, "owner": owner}
            self._last_reminder = self._s["since"]
            info = dict(self._s)
        log.warning("PERSON HOLD on: %s -- agents' runs wait until it is released.",
                    describe(info))
        self._record("run_queue_hold", by=by, reason=reason, source=source, run_id=run_id,
                     owner=owner)
        self._save()
        self._notify()
        return {"status": "ok", "person_hold": info}

    def release(self, by: str = "", owner: str = "person") -> dict:
        """Lift the hold (``owner``: who asks).  An agent may not lift a
        person's hold (one a person put on, liveOD's Reset set, or the server
        set on an unreadable file)."""
        by = str(by or "").strip() or "?"
        with self._lock:
            if not self._s["active"]:
                return {"status": "error", "msg": "no person hold is on"}
            if owner == "agent" and self._s.get("owner", "person") != "agent":
                return {"status": "error",
                        "msg": f"the hold is a person's ({describe(self._s)}): an agent may not "
                               "release it"}
            held = dict(self._s)
            self._s = dict(_OFF)
            self._last_reminder = None
            info = dict(self._s)
        log.warning("Person hold released by %s (it was %s).", by, describe(held))
        self._record("run_queue_release", by=by, owner=owner, held_since=held["since"],
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
                     own_abort_ids: Iterable = (),
                     on_person_reset: Callable[[dict], bool] | None = None) -> bool:
        """Read one liveOD POLL reply and put the hold on for a person's Reset.

        liveOD from 2026-10-09 counts every Abort set, by source: a person's
        Reset is ``reset_counts["person"]`` going up (or, with only
        ``reset_count``, the count going up with ``last_reset.source``
        "person").  A count sees a quick Reset that liveOD cleared again
        between two POLLs, and the queue's own Abort ("queue") or an agent's
        ("agent") never counts as a person's.  ``on_person_reset(last_reset)``
        may claim the Reset first (True: no hold -- the run queue's cancel of
        a person's own starting job).

        An older liveOD (no count) is watched by the level of
        ``reset_requested``, with one WARNING: a Reset just turned on whose run
        is not one the queue launched for an agent (``agent_run_ids``) nor one
        the queue aborted itself (``own_abort_ids``).  The first POLL seen only
        sets the baseline.  Returns True when it set the hold."""
        if not isinstance(poll, dict) or poll.get("ok") is False:
            return False
        if "reset_counts" in poll or "reset_count" in poll:
            return self._observe_count(poll, on_person_reset)
        if not self._warned_level:
            self._warned_level = True
            log.warning("liveOD gives no reset_count (it predates 2026-10-09): the person hold "
                        "watches reset_requested's level, which misses a Reset liveOD clears "
                        "between two polls. Restart liveOD to get the count.")
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

    def _observe_count(self, poll: dict, on_person_reset) -> bool:
        counts = poll.get("reset_counts")
        last = poll.get("last_reset") if isinstance(poll.get("last_reset"), dict) else {}
        if isinstance(counts, dict):
            count = int(counts.get("person") or 0)
            person = True
        else:
            count = int(poll.get("reset_count") or 0)
            person = str(last.get("source") or "person") == "person"
        previous, self._last_count = self._last_count, count
        if previous is None or count == previous:
            return False                 # the baseline, or no new Reset
        if count < previous:
            # liveOD restarted (its counts start again): a person's Reset since
            # then shows as a count above zero with a person's last_reset
            if count == 0 or str(last.get("source") or "person") != "person":
                return False
        if not person:
            log.info("liveOD Reset (source %s) -- not a person's: no person hold.",
                     last.get("source"))
            return False
        if on_person_reset is not None:
            try:
                if on_person_reset(dict(last)):
                    return False
            except Exception:                         # noqa: BLE001
                log.exception("Person reset handler failed; the hold goes on")
        if self.active:
            return False
        at = last.get("at") or self._clock()
        self.hold(f"Reset in liveOD at {_clock_text(at)}", LIVE_OD_RESET_BY,
                  source="live_od_reset", run_id=last.get("run_id"))
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
            data = exc
        if not isinstance(data, dict):
            # fail closed: a hold that may have been on is taken to be on, and
            # the file is kept (moved aside, never overwritten) for a person
            aside = move_aside(self.path)
            why = data if isinstance(data, Exception) else f"not a JSON object: {data!r:.80}"
            self._s = {"active": True, "since": self._clock(), "by": "monitor server",
                       "reason": f"could not read the hold file ({why})"
                                 + (f"; kept as {aside}" if aside else ""),
                       "source": "unreadable_file", "run_id": None, "owner": "person"}
            self._last_reminder = self._clock()
            original = self.path
            if not aside:
                self.path = None              # never overwrite the unreadable original
            log.error("Could not read the person hold from %s (%s): starting HELD -- a person "
                      "must release it. %s", original, why,
                      f"The file was moved to {aside}." if aside
                      else "The file could not be moved aside: the hold is kept in memory only.")
            self._record("run_queue_hold", by="monitor server", reason=self._s["reason"],
                         source="unreadable_file", run_id=None)
            self._save()
            return
        if data.get("active"):
            self._s = {"active": True, "since": data.get("since"), "by": str(data.get("by") or ""),
                       "reason": str(data.get("reason") or ""),
                       "source": str(data.get("source") or ""), "run_id": data.get("run_id"),
                       "owner": str(data.get("owner") or "person")}
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


def move_aside(path: str) -> str:
    """Rename an unreadable state file to ``<name>.unreadable-<YYYYmmdd-HHMMSS>``
    (never deleted, never overwritten); the new name, or "" when it could not
    be moved."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = f"{path}.unreadable-{stamp}"
    n = 1
    while os.path.exists(target):
        target, n = f"{path}.unreadable-{stamp}-{n}", n + 1
    try:
        os.rename(path, target)
    except OSError as exc:
        log.error("Could not move the unreadable %s aside: %s", path, exc)
        return ""
    return target


def describe(info: dict | None) -> str:
    """"person hold since 14:03:11 by jp@kong: reason" for a ``person_hold``
    dict (from ``status_json`` or :meth:`PersonHold.info`)."""
    info = info or {}
    return (f"person hold since {_clock_text(info.get('since'))} by "
            f"{info.get('by') or '?'}: {info.get('reason') or 'a person has the machine'}")
