"""A person's hold on the machine, kept by the monitor server.

While the hold is on, nothing an *agent* asked for starts: the run queue's
agent jobs wait (:mod:`~waxx.util.device_state.run_queue`), the server's run
loops do not start or continue (their gate reports it), and
:func:`waxx.util.device_state.run_gate.assess` -- the agents' occupancy check --
reports the machine busy.  A person's own queued runs still go.

It is set

* by a request -- ``{"type": "run_queue", "action": "hold", "reason", "by",
  "expiry"}``, e.g. the Monitor panel's Hold button;
* by the server itself when a person presses liveOD's Reset / Abort: liveOD
  counts every Abort by source (POLL ``reset_counts`` / ``reset_count`` +
  ``last_reset``) and a person's count going up sets the hold -- whatever run
  it was for, and even if liveOD cleared the Abort again between two polls.
  The queue's own Abort (source "queue") and an agent's (source "agent",
  ``reset_liveod.py --own-run``) never set it.  A liveOD without the counts is
  watched by the level of ``reset_requested`` (see :meth:`observe_poll`).

It ends by a request (``"action": "release"``), or -- since 2026-11-16 --
**lapses by itself** under a rule set in advance, so a hold nobody remembers
does not block the machine all night.  No agent decides it: the server
applies the rule, journals it (``run_queue_hold_lapsed``) and logs it.  How a
hold lapses is its ``expiry``:

* ``{"mode": "policy"}`` (the default) -- by the server's
  :class:`ExpiryPolicy`, from the last *person* activity the server saw: a
  hold request while it is on, a person's Reset in liveOD, a person's queue
  request, a Device Control change, a run announcing itself, and a person's
  queued job that is running or about to (while one is, the hold never lapses).
  Without a policy (the waxx default) such a hold never lapses, as before.
* ``{"mode": "never"}`` -- only a release ends it (hands-on work).
* ``{"mode": "at", "at": epoch}`` -- at that time (``kq hold --for/--until``),
  unless a person's job is running or about to.

A hold the server set because it could not read its file never lapses.  While
the hold is on, the server logs a reminder every :data:`REMINDER_S`.  It is kept
in a small JSON file next to the run queue's, so a server restart does not drop
it (nor its last activity: a hold whose time passed while the server was down
lapses at the first tick, journaled).

Machine-agnostic: nothing here knows the K machine.  The lab's numbers are an
:class:`ExpiryPolicy` the lab passes to the server.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

log = logging.getLogger(__name__)

#: A reminder line in the server log this often while the hold is on.
REMINDER_S = 1800.0
#: Person activity moves the lapse time at once; the hold's file (and the
#: broadcast) follow at most this often -- a restart loses at most this much.
TOUCH_STORE_S = 60.0
#: ``{"mode": "at"}`` may be at most this far ahead (longer: ``never``).
MAX_AT_S = 7 * 86400.0

EXPIRY_MODES = ("policy", "never", "at")
_OFF = {"active": False, "since": None, "by": "", "reason": "", "source": "", "run_id": None,
        "owner": "", "expiry": None, "last_activity": None, "activity": "", "lapses_at": None}

#: Who sets the hold when liveOD's Reset is pressed.
LIVE_OD_RESET_BY = "liveOD"
#: Holds of these sources never lapse.
NEVER_LAPSE_SOURCES = ("unreadable_file",)

#: The GUIs' choice of how a hold lapses: (label, expiry).  Labels say what
#: happens.
LAPSE_CHOICES = (
    ("Lapses by itself once nobody has been active for a while (the lab's rule)",
     {"mode": "policy"}),
    ("Never lapses: I am working on the hardware (release it when done)", {"mode": "never"}),
)


def _clock_text(t) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(t)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def _when_text(t, now: float | None = None) -> str:
    """"21:30" today, "Tue 07:00" another day."""
    try:
        t = float(t)
        now = time.time() if now is None else float(now)
        lt, ln = time.localtime(t), time.localtime(now)
        if (lt.tm_year, lt.tm_yday) == (ln.tm_year, ln.tm_yday):
            return time.strftime("%H:%M", lt)
        return time.strftime("%a %H:%M", lt)
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def parse_expiry(value, now: float) -> tuple[dict | None, str]:
    """A request's ``expiry`` -> ``(expiry, "")``, or ``(None, why it is
    refused)``.  Accepted: None / "" / "policy" (the server's rule), "never",
    ``{"mode": ...}`` with ``"at"`` for mode "at", ``{"at": epoch}``,
    ``{"for_s": seconds}``."""
    if value is None or value == "" or value == "policy":
        return {"mode": "policy"}, ""
    if value == "never":
        return {"mode": "never"}, ""
    if not isinstance(value, Mapping):
        return None, f"expiry must be 'policy', 'never' or a mapping, not {value!r:.60}"
    mode = value.get("mode")
    at = value.get("at")
    if "for_s" in value:
        try:
            for_s = float(value["for_s"])
        except (TypeError, ValueError):
            return None, f"expiry for_s must be a number of seconds, not {value['for_s']!r:.40}"
        if not for_s > 0:
            return None, "expiry for_s must be above 0"
        at, mode = now + for_s, "at"
    if mode is None and at is not None:
        mode = "at"
    if mode not in EXPIRY_MODES:
        return None, f"expiry mode must be one of {', '.join(EXPIRY_MODES)}, not {mode!r}"
    if mode != "at":
        return {"mode": mode}, ""
    try:
        at = float(at)
    except (TypeError, ValueError):
        return None, f"expiry at must be a time (epoch seconds), not {at!r:.40}"
    if at <= now:
        return None, f"expiry at {_when_text(at, now)} has already passed"
    if at - now > MAX_AT_S:
        return None, (f"expiry at {_when_text(at, now)} is more than {MAX_AT_S / 86400:.0f} days "
                      "ahead: use 'never' and release it by hand")
    return {"mode": "at", "at": at}, ""


@dataclass(frozen=True)
class ExpiryPolicy:
    """When a hold whose expiry is ``policy`` lapses, from the last person
    activity ``last`` (epoch seconds; hours are local time):

    * ``idle_s[source]`` -- at any hour, ``last + idle_s[source]`` (a source
      not listed: no such limit);
    * ``quiet_idle_s`` -- in quiet hours, once ``quiet_idle_s`` has passed with
      no activity: ``last + quiet_idle_s`` if that falls in quiet hours, else
      the next quiet-hours start after it.

    Quiet hours: from ``quiet_start_h`` to ``quiet_end_h`` (across midnight
    when start > end; None: none by the clock), and all of Saturday and Sunday
    when ``quiet_weekends``.  The earlier of the two rules wins.  Sources in
    ``never_sources`` never lapse."""
    idle_s: Mapping[str, float] = field(default_factory=dict)
    quiet_idle_s: float | None = None
    quiet_start_h: float | None = None
    quiet_end_h: float | None = None
    quiet_weekends: bool = False
    never_sources: tuple = NEVER_LAPSE_SOURCES

    @staticmethod
    def _hours(dt: _dt.datetime) -> float:
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    def is_quiet(self, t: float) -> bool:
        dt = _dt.datetime.fromtimestamp(float(t))
        if self.quiet_weekends and dt.weekday() >= 5:
            return True
        start, end = self.quiet_start_h, self.quiet_end_h
        if start is None or end is None or start == end:
            return False
        h = self._hours(dt)
        if start > end:
            return h >= start or h < end
        return start <= h < end

    def next_quiet(self, t: float) -> float | None:
        """``t`` if it is in quiet hours, else when quiet hours next begin
        (None: never)."""
        if self.is_quiet(t):
            return float(t)
        dt = _dt.datetime.fromtimestamp(float(t))
        midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        candidates = []
        if (self.quiet_start_h is not None and self.quiet_end_h is not None
                and self.quiet_start_h != self.quiet_end_h):
            c = midnight + _dt.timedelta(hours=float(self.quiet_start_h))
            if c <= dt:
                c = midnight + _dt.timedelta(days=1, hours=float(self.quiet_start_h))
            candidates.append(c)
        if self.quiet_weekends:
            c = midnight + _dt.timedelta(days=(5 - dt.weekday()) % 7)
            if c <= dt:
                c += _dt.timedelta(days=7)
            candidates.append(c)
        if not candidates:
            return None
        return min(candidates).timestamp()

    def lapses_at(self, source: str, last: float) -> float | None:
        if source in self.never_sources:
            return None
        candidates = []
        idle = self.idle_s.get(source) if self.idle_s else None
        if idle:
            candidates.append(float(last) + float(idle))
        if self.quiet_idle_s is not None:
            q = self.next_quiet(float(last) + float(self.quiet_idle_s))
            if q is not None:
                candidates.append(q)
        return min(candidates) if candidates else None

    def text(self) -> str:
        """One line for the server's start log."""
        parts = [f"{src}: after {s / 60:.0f} min with no person activity"
                 for src, s in (self.idle_s or {}).items() if s]
        if self.quiet_idle_s is not None:
            when = []
            if self.quiet_start_h is not None and self.quiet_end_h is not None:
                when.append(f"{self.quiet_start_h:g}:00-{self.quiet_end_h:g}:00")
            if self.quiet_weekends:
                when.append("weekends")
            parts.append(f"any hold: in quiet hours ({', '.join(when) or 'none'}) after "
                         f"{self.quiet_idle_s / 60:.0f} min with no person activity")
        return "; ".join(parts) or "never"


class PersonHold:
    """The hold's state and its rules.

    ``info()`` is what ``status_json`` serves as ``person_hold``:
    ``{"active": bool, "since": epoch | None, "by": str, "reason": str,
    "source": "request" | "live_od_reset" | "unreadable_file" | "", "run_id": int |
    None, "owner": "person" | "agent" | "", "expiry": {"mode", ["at"]} | None,
    "last_activity": epoch | None, "activity": str, "lapses_at": epoch | None}``
    (``run_id``: the run liveOD had when its Reset set the hold; ``owner``: who
    put it on -- an agent may not release a person's hold; ``activity``: the
    last person activity seen; ``lapses_at``: when it lapses if nothing
    happens -- None: it does not lapse by itself).

    ``path``: where the hold is kept across restarts (None: memory only).
    ``journal``: an :class:`~waxx.util.device_state.op_journal.OpJournal`
    (records ``run_queue_hold`` / ``run_queue_hold_changed`` /
    ``run_queue_release`` / ``run_queue_hold_lapsed``); ``on_change(info)`` is
    called on every change, from whichever thread made it.  ``policy``: an
    :class:`ExpiryPolicy` (None: a ``policy`` hold never lapses).  ``busy()``
    (or :meth:`set_busy_check`): why a person is busy on the machine right now
    ("" when not) -- while it says so, the hold counts it as activity and does
    not lapse."""

    def __init__(self, path: str | None = None, *, journal=None,
                 on_change: Callable[[dict], None] | None = None,
                 clock: Callable[[], float] = time.time, reminder_s: float = REMINDER_S,
                 policy: ExpiryPolicy | None = None,
                 busy: Callable[[], str] | None = None):
        self.path = path
        self._journal = journal
        self._on_change = on_change
        self._clock = clock
        self._reminder_s = float(reminder_s)
        self.policy = policy
        self._busy = busy
        self._busy_failed = False
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._save_seq = self._saved_seq = 0
        self._s = dict(_OFF)
        self._last_reminder: float | None = None
        #: when person activity was last stored (TOUCH_STORE_S)
        self._last_store: float | None = None
        #: liveOD's reset_requested at the last POLL seen (None: none seen yet)
        #: -- the fallback for a liveOD without reset counts
        self._last_reset: bool | None = None
        #: liveOD's person Reset count at the last POLL seen (None: none yet)
        self._last_count: int | None = None
        self._warned_level = False
        self._load()

    def set_busy_check(self, busy: Callable[[], str] | None) -> None:
        self._busy = busy

    # -- state ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._s["active"])

    def info(self) -> dict:
        with self._lock:
            return self._copy()

    def _copy(self) -> dict:
        out = dict(self._s)
        if isinstance(out.get("expiry"), dict):
            out["expiry"] = dict(out["expiry"])
        return out

    def text(self) -> str:
        """"person hold since 14:03:11 by jp@kong: reason" ("" when off),
        with when it lapses (see :func:`describe`)."""
        s = self.info()
        if not s["active"]:
            return ""
        return describe(s)

    def _lapse_time(self, s: dict) -> float | None:
        """When the hold ``s`` lapses if nothing happens (None: not by itself)."""
        if not s.get("active") or s.get("source") in NEVER_LAPSE_SOURCES:
            return None
        expiry = s.get("expiry") or {"mode": "policy"}
        mode = expiry.get("mode")
        if mode == "never":
            return None
        if mode == "at":
            try:
                return float(expiry.get("at"))
            except (TypeError, ValueError):
                return None
        if self.policy is None:
            return None
        last = s.get("last_activity")
        if last is None:
            last = s.get("since")
        try:
            return self.policy.lapses_at(str(s.get("source") or ""), float(last))
        except (TypeError, ValueError, OverflowError, OSError):
            log.exception("Could not work out when the person hold lapses; it does not lapse")
            return None

    # -- requests -----------------------------------------------------------------

    def hold(self, reason: str = "", by: str = "", *, source: str = "request",
             run_id=None, owner: str = "person", expiry=None) -> dict:
        """Put the hold on (``owner``: who puts it on; ``expiry``: how it
        lapses, see :func:`parse_expiry` -- None: by the policy).

        Already on: the first hold's since/by/reason/owner stay and the reply
        says ``already``.  A person's request then counts as activity (the
        lapse moves) and, with an ``expiry``, changes how the hold lapses --
        that is how a person extends it or makes it never lapse."""
        reason = str(reason or "").strip() or "a person has the machine"
        by = str(by or "").strip() or "?"
        now = self._clock()
        parsed = None
        if expiry is not None:
            parsed, refusal = parse_expiry(expiry, now)
            if refusal:
                return {"status": "error", "msg": refusal}
        changed = None
        with self._lock:
            if self._s["active"]:
                if owner != "person":
                    return {"status": "ok", "already": True, "person_hold": self._copy()}
                if parsed is not None and parsed != self._s.get("expiry"):
                    changed = (self._s.get("expiry"), parsed)
                    self._s["expiry"] = parsed
                was_lapse = self._s.get("lapses_at")
                self._s["last_activity"] = now
                self._s["activity"] = f"a hold request by {by}"
                self._s["lapses_at"] = self._lapse_time(self._s)
                moved = changed is not None or self._s["lapses_at"] != was_lapse
                self._last_store = now
                info = self._copy()
                already = True
            else:
                self._s = {"active": True, "since": now, "by": by, "reason": reason,
                           "source": source, "run_id": run_id, "owner": owner,
                           "expiry": parsed or {"mode": "policy"}, "last_activity": now,
                           "activity": "the hold was put on", "lapses_at": None}
                self._s["lapses_at"] = self._lapse_time(self._s)
                self._last_reminder = self._s["since"]
                self._last_store = now
                info = self._copy()
                already = False
        if already:
            if changed is not None:
                log.warning("Person hold: how it lapses changed by %s (%s -> %s): %s", by,
                            changed[0], changed[1], describe(info))
                self._record("run_queue_hold_changed", by=by, owner=owner, was=changed[0],
                             expiry=changed[1], lapses_at=info["lapses_at"])
            self._save()
            if moved:
                self._notify()
            return {"status": "ok", "already": True, "person_hold": info}
        log.warning("PERSON HOLD on: %s -- agents' runs wait until it is released%s.",
                    describe(info), " or lapses" if info["lapses_at"] is not None else "")
        self._record("run_queue_hold", by=by, reason=reason, source=source, run_id=run_id,
                     owner=owner, expiry=info["expiry"], lapses_at=info["lapses_at"])
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
            held = self._copy()
            self._s = dict(_OFF)
            self._last_reminder = None
            info = self._copy()
        log.warning("Person hold released by %s (it was %s).", by, describe(held))
        self._record("run_queue_release", by=by, owner=owner, held_since=held["since"],
                     held_by=held["by"], reason=held["reason"])
        self._save()
        self._notify()
        return {"status": "ok", "person_hold": info, "released": held}

    def touch(self, what: str, owner: str = "person") -> bool:
        """A person did something on the machine (``what``): while the hold is
        on, its lapse moves.  An agent's activity (``owner`` "agent") never
        counts.  True when it counted."""
        if owner == "agent":
            return False
        with self._lock:
            if not self._s["active"]:
                return False
            now = self._clock()
            was = self._s.get("lapses_at")
            self._s["last_activity"] = now
            self._s["activity"] = str(what or "")
            self._s["lapses_at"] = self._lapse_time(self._s)
            moved = self._s["lapses_at"] != was
            store = self._last_store is None or now - self._last_store >= TOUCH_STORE_S
            if store:
                self._last_store = now
        if store:
            self._save()
            if moved:
                self._notify()
        return True

    # -- the server's ticks -------------------------------------------------------

    def tick(self) -> bool:
        """A reminder line every ``reminder_s`` while the hold is on, and the
        lapse.  While ``busy()`` says a person is busy (or the check fails) the
        hold counts it as activity and does not lapse.  True when the hold
        lapsed in this tick."""
        with self._lock:
            if not self._s["active"]:
                return False
            now = self._clock()
            last = self._last_reminder if self._last_reminder is not None else self._s["since"]
            remind = now - float(last or now) >= self._reminder_s
            if remind:
                self._last_reminder = now
            info = self._copy()
        if remind:
            log.warning("Reminder: the PERSON HOLD is still on (%s, %.0f min) -- agents' runs "
                        "wait until someone releases it%s.", describe(info),
                        (now - float(info["since"] or now)) / 60.0,
                        " or it lapses" if info.get("lapses_at") is not None else "")
        busy = self._busy_text()
        if busy:
            self.touch(busy)
            return False
        lapse = info.get("lapses_at")
        if lapse is None or now < float(lapse):
            return False
        return self._lapse(now)

    def _busy_text(self) -> str:
        if self._busy is None:
            return ""
        try:
            text = str(self._busy() or "")
        except Exception:                             # noqa: BLE001
            if not self._busy_failed:
                log.exception("The person hold's busy check failed: the hold does not lapse "
                              "while it fails")
            self._busy_failed = True
            return "the busy check failed"
        self._busy_failed = False
        return text

    def _lapse(self, now: float) -> bool:
        with self._lock:
            lapse = self._s.get("lapses_at")
            if not self._s["active"] or lapse is None or now < float(lapse):
                return False                          # released or touched meanwhile
            held = self._copy()
            self._s = dict(_OFF)
            self._last_reminder = None
        rule = lapse_rule_text(held)
        log.warning("PERSON HOLD LAPSED at %s: %s -- %s. Agents' runs may start now.",
                    _clock_text(now), describe(held), rule)
        self._record("run_queue_hold_lapsed", by="lapse", rule=rule, expiry=held.get("expiry"),
                     lapses_at=held.get("lapses_at"), last_activity=held.get("last_activity"),
                     activity=held.get("activity"), held_since=held["since"],
                     held_by=held["by"], reason=held["reason"], source=held.get("source"),
                     owner=held.get("owner"))
        self._save()
        self._notify()
        return True

    def observe_poll(self, poll: dict | None, agent_run_ids: Iterable = (),
                     own_abort_ids: Iterable = (),
                     on_person_reset: Callable[[dict], bool] | None = None) -> bool:
        """Read one liveOD POLL reply and put the hold on for a person's Reset
        (a person's Reset while the hold is on counts as activity).

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
            self.touch("a Reset in liveOD")
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
            self.touch("a Reset in liveOD")
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
            now = self._clock()
            self._s = {"active": True, "since": now, "by": "monitor server",
                       "reason": f"could not read the hold file ({why})"
                                 + (f"; kept as {aside}" if aside else ""),
                       "source": "unreadable_file", "run_id": None, "owner": "person",
                       "expiry": {"mode": "never"}, "last_activity": now,
                       "activity": "", "lapses_at": None}
            self._last_reminder = now
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
            now = self._clock()
            expiry = data.get("expiry")
            if not (isinstance(expiry, dict) and expiry.get("mode") in EXPIRY_MODES):
                expiry = {"mode": "policy"}           # a file from before 2026-11-16
            last = data.get("last_activity")
            activity = str(data.get("activity") or "")
            if not isinstance(last, (int, float)):
                # no activity time stored (an older file): the idle time
                # counts from now, never from before
                last, activity = now, "the server started (no activity time stored)"
            self._s = {"active": True, "since": data.get("since"), "by": str(data.get("by") or ""),
                       "reason": str(data.get("reason") or ""),
                       "source": str(data.get("source") or ""), "run_id": data.get("run_id"),
                       "owner": str(data.get("owner") or "person"), "expiry": dict(expiry),
                       "last_activity": float(last), "activity": activity, "lapses_at": None}
            self._s["lapses_at"] = self._lapse_time(self._s)
            self._last_reminder = now
            self._last_store = now
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
            seq, data = self._save_seq, self._copy()
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


def lapse_text(info: dict | None, now: float | None = None) -> str:
    """How an active hold ends, for the end of its line: "lapses at 21:30
    unless a person is active", "lapses at 21:30", "never lapses by itself",
    or "" (it does not lapse by itself and nobody chose that: the old
    behaviour, no text)."""
    info = info or {}
    if not info.get("active"):
        return ""
    mode = (info.get("expiry") or {}).get("mode")
    at = info.get("lapses_at")
    if mode == "never":
        return "never lapses by itself"
    if at is None:
        return ""
    if mode == "at":
        return f"lapses at {_when_text(at, now)}"
    return f"lapses at {_when_text(at, now)} unless a person is active"


def lapse_rule_text(held: dict) -> str:
    """Why a hold lapsed, for the log and the journal."""
    mode = (held.get("expiry") or {}).get("mode")
    if mode == "at":
        return f"the time it was set to end ({_when_text(held.get('lapses_at'))}) came"
    last = held.get("last_activity")
    what = held.get("activity") or "?"
    return (f"no person activity since {_when_text(last)} ({what}); the lab's rule lapses it "
            f"at {_when_text(held.get('lapses_at'))}")


def describe(info: dict | None) -> str:
    """"person hold since 14:03:11 by jp@kong: reason" for a ``person_hold``
    dict (from ``status_json`` or :meth:`PersonHold.info`), followed by
    " -- lapses at 21:30 unless a person is active" (see :func:`lapse_text`)
    when it lapses by itself or never may."""
    info = info or {}
    text = (f"person hold since {_clock_text(info.get('since'))} by "
            f"{info.get('by') or '?'}: {info.get('reason') or 'a person has the machine'}")
    tail = lapse_text(info)
    return f"{text} -- {tail}" if tail else text
