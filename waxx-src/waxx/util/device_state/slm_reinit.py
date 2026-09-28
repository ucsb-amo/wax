"""The SLM's hourly re-initialisation, done only while no run could be using it.

The SLM server (``waxx/control/slm/server/run_server.py``) used to re-initialise
the SLM once an hour by itself, whenever it had been idle for 20 s, and then put
up a blank mask. A run that writes its mask only at the start (the feedback
experiments, the sigma_z / B0 calibrations) kept running on a blank SLM from
then on -- the APD readout flipped mid-run with no error (run 83344,
2026-09-28). Now the SLM server only marks a reinit *due* every hour, and puts
back the pattern it showed after one; this service, in the monitor server,
asks for it when nothing is using the SLM:

* the monitor experiment is running (proof that no run holds the core),
* no run has announced itself (``run_pending``),
* no state reset and no run loop is running,

all for at least ``settle_s``. The check and the request are one step under
a lock that ``run_starting`` also takes, so a run that announces itself either
stops the request or waits (at most ``SEND_WAIT_S``) until the request is on
its way. From then on the server's queue does the rest: it takes commands in
order, so the run's first mask write lands after the reinit and its pattern.

The request is an ``SLMCTL`` control line (see ``slm_protocol``), which SLM
servers from before this change drop without applying anything: against such
a server this service only says so (state ``no_control``, checked again every
``retry_after_failure_s``) -- that server still re-initialises by itself.

What the service reports (``snapshot()``, served in ``status_json`` as
``slm_reinit`` and broadcast as ``{"type": "slm_reinit"}``)::

    {"state", "detail", "since", "reinit_due", "due_for_s", "pattern_epoch",
     "pattern", "last_reinit", "last_poll", "blocked_by"}

``state`` is one of :data:`STATES`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

UNKNOWN = "unknown"            # not polled yet
IDLE = "idle"                  # reachable, no reinit due
DUE = "due"                    # due, waiting for the machine to be idle
REINITIALISING = "reinitialising"
FAILED = "failed"              # the last reinit failed; retried after a pause
UNREACHABLE = "unreachable"
NO_CONTROL = "no_control"      # a server from before control commands
STATES = (UNKNOWN, IDLE, DUE, REINITIALISING, FAILED, UNREACHABLE, NO_CONTROL)

#: The longest a run's announcement waits for a reinit request being sent.
#: The announcement's reply also waits for the connections' release
#: (connections.RELEASE_TIMEOUT_S), and clients give up after 5 s.
SEND_WAIT_S = 1.0


@dataclass(frozen=True)
class SlmReinitConfig:
    """Where the SLM server is and how the service paces itself.

    * ``poll_s``: how often it asks the server for its status.
    * ``settle_s``: how long the machine must have been idle before a reinit.
    * ``done_timeout_s``: how long a reinit may take before it is called failed.
    * ``retry_after_failure_s``: the pause before asking again after a failure.
    """

    host: str
    port: int = 5000
    label: str = "SLM"
    poll_s: float = 20.0
    settle_s: float = 5.0
    done_timeout_s: float = 60.0
    retry_after_failure_s: float = 600.0


def _default_link(host, port, payload, **kw):
    from waxx.control.slm.slm_link import exchange  # noqa: PLC0415
    return exchange(host, port, payload, **kw)


def _link_errors():
    from waxx.control.slm.slm_link import NoReply, ReplyTimeout  # noqa: PLC0415
    return NoReply, ReplyTimeout


class SlmReinitService:
    """See the module docstring.

    * ``blocker()`` -> why the machine is not idle ("" when it is); called
      under this service's lock, so it must not call back into the service.
    * ``on_change(snapshot)`` -- on any state change, from the service thread.
    * ``journal`` -- an ``OpJournal`` (records ``slm_reinit_*``), or None.
    * ``link`` / ``clock`` -- for tests: ``slm_link.exchange`` and
      ``time.monotonic`` stand-ins.
    """

    TICK_S = 0.5

    def __init__(self, config: SlmReinitConfig, blocker: Callable[[], str],
                 on_change: Callable[[dict], None] | None = None, journal=None,
                 log: Callable[[str], None] | None = None, link=None, clock=None):
        self.config = config
        self._blocker = blocker
        self._on_change = on_change
        self._journal = journal
        self._log = log or (lambda text: None)
        self._link = link or _default_link
        self._clock = clock or time.monotonic
        self._cond = threading.Condition()
        self._sending = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._state = UNKNOWN
        self._detail = ""
        self._since = time.time()
        self._status: dict = {}
        self._last_poll: float | None = None       # wall time
        self._next_poll = 0.0                       # clock time
        self._idle_since: float | None = None       # clock time
        self._blocked_by = ""
        self._reinit_t0: float | None = None        # clock time
        self._reinit_epoch0: int | None = None
        self._retry_at = 0.0                        # clock time
        self._last_reinit: dict | None = None
        self._ever_answered = False

    # --- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="slm-reinit")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.TICK_S):
            try:
                self.tick()
            except Exception as e:  # never let the thread die
                self._log(f"SLM reinit service: tick failed: {e!r}")

    # --- the run fence -----------------------------------------------------------

    def run_starting(self, timeout: float = SEND_WAIT_S) -> None:
        """A run has announced itself (the caller has already recorded it, so
        ``blocker()`` says so): wait until a reinit request being sent is on
        its way -- the server then takes it before the run's mask."""
        with self._cond:
            self._cond.wait_for(lambda: not self._sending, timeout)

    # --- one pass ----------------------------------------------------------------

    def tick(self) -> None:
        now = self._clock()
        with self._cond:
            why = self._blocker()
        if why:
            self._idle_since = None
        elif self._idle_since is None:
            self._idle_since = now
        self._blocked_by = why

        if self._state == REINITIALISING:
            self._follow_reinit(now)
            return
        if now >= self._next_poll:
            self._poll(now)
        if (self._state == DUE and self._idle_since is not None
                and now - self._idle_since >= self.config.settle_s
                and now >= self._retry_at):
            self._request_reinit(now)

    def _poll(self, now: float) -> dict | None:
        """Ask for the server's status. A failure while re-initialising
        changes nothing here (``_follow_reinit`` times it out): the server
        serves one connection at a time, and a run's mask write waiting
        behind the reinit holds it."""
        self._next_poll = now + self.config.poll_s
        no_reply, reply_timeout = _link_errors()
        following = self._state == REINITIALISING
        try:
            status = self._link(self.config.host, self.config.port, {"cmd": "status"},
                                control=True, until=("ok", "error"), connect_s=2.0,
                                first_reply_s=3.0, total_s=3.0)
        except no_reply as e:
            if following:
                return None
            if self._ever_answered:
                # It has answered before: busy with another client, not old.
                self._set(UNREACHABLE, f"no answer ({e})")
            else:
                self._set(NO_CONTROL, "the SLM server does not take control commands (a "
                                      "server from before 2026-09-28, which re-initialises "
                                      "by itself)")
                self._next_poll = now + self.config.retry_after_failure_s
            return None
        except (reply_timeout, OSError) as e:
            if not following:
                self._set(UNREACHABLE, str(e) or type(e).__name__)
            return None
        self._ever_answered = True
        self._last_poll = time.time()
        if status.get("status") != "ok":
            if not following:
                self._set(UNREACHABLE, f"status refused: {status.get('error')}")
            return None
        self._status = status
        if not following:
            if status.get("reinit_in_progress"):
                self._set(REINITIALISING, "the SLM server is re-initialising")
                self._next_poll = now + 1.0
            elif status.get("reinit_due"):
                if self._state != FAILED or now >= self._retry_at:
                    self._set(DUE, "")
            else:
                self._set(IDLE, "")
        return status

    def _request_reinit(self, now: float) -> None:
        with self._cond:
            why = self._blocker()
            if why:
                self._idle_since = None
                self._blocked_by = why
                return
            self._sending = True
        no_reply, reply_timeout = _link_errors()

        def sent():
            with self._cond:
                self._sending = False
                self._cond.notify_all()

        epoch0 = self._status.get("pattern_epoch")
        try:
            reply = self._link(self.config.host, self.config.port,
                               {"cmd": "reinit", "by": "the monitor server"}, control=True,
                               until=("queued", "error"), connect_s=SEND_WAIT_S,
                               first_reply_s=3.0, total_s=3.0, on_sent=sent)
        except no_reply:
            if self._ever_answered:
                # Sent, so a busy server still takes it -- ahead of any
                # command sent later; the next poll shows what happened.
                self._set(UNREACHABLE, "the reinit request was not answered in time "
                                       "(the SLM server may still do it)")
            else:
                self._set(NO_CONTROL, "the SLM server did not answer a reinit request")
                self._next_poll = now + self.config.retry_after_failure_s
            return
        except (reply_timeout, OSError) as e:
            self._set(UNREACHABLE, f"reinit request failed: {e or type(e).__name__}")
            return
        finally:
            sent()          # idempotent: covers a link that never called on_sent
        if reply.get("status") != "queued":
            self._fail(now, f"reinit refused: {reply.get('error')}")
            return
        self._reinit_t0 = now
        self._reinit_epoch0 = epoch0
        self._log(f"{self.config.label}: re-initialising (asked for by the monitor server; "
                  f"the machine was idle).")
        self._record("slm_reinit_requested", epoch=epoch0,
                     due_for_s=self._status.get("due_for_s"))
        self._set(REINITIALISING, "asked for by the monitor server")
        self._next_poll = now + 1.0

    def _follow_reinit(self, now: float) -> None:
        """While re-initialising: poll every second until the epoch moves."""
        if self._reinit_t0 is None:
            # the server was already re-initialising when first polled
            self._reinit_t0, self._reinit_epoch0 = now, self._status.get("pattern_epoch")
        if now < self._next_poll:
            return
        status = self._poll(now)
        self._next_poll = now + 1.0
        if status is None:
            if now - self._reinit_t0 > self.config.done_timeout_s:
                self._fail(now, f"no answer for {self.config.done_timeout_s:.0f} s")
            return
        epoch = status.get("pattern_epoch")
        if isinstance(epoch, int) and (self._reinit_epoch0 is None or epoch != self._reinit_epoch0) \
                and not status.get("reinit_in_progress"):
            t = now - self._reinit_t0
            self._last_reinit = {"at": time.time(), "t_s": round(t, 1), "epoch": epoch,
                                 "pattern": status.get("pattern")}
            self._log(f"{self.config.label}: re-initialised in {t:.1f} s; pattern put back "
                      f"({status.get('pattern')}).")
            self._record("slm_reinit_done", epoch=epoch, t_s=round(t, 1),
                         pattern=status.get("pattern"))
            self._reinit_t0 = None
            self._next_poll = now + self.config.poll_s
            self._set(DUE if status.get("reinit_due") else IDLE, "")
            return
        if not status.get("reinit_in_progress") and status.get("last_reinit_error"):
            self._fail(now, f"reinit failed on the SLM server: {status['last_reinit_error']}")
            return
        if now - self._reinit_t0 > self.config.done_timeout_s:
            self._fail(now, f"not done after {self.config.done_timeout_s:.0f} s")

    def _fail(self, now: float, text: str) -> None:
        self._reinit_t0 = None
        self._retry_at = now + self.config.retry_after_failure_s
        self._next_poll = now + self.config.poll_s
        self._log(f"{self.config.label}: {text}; asking again in "
                  f"{self.config.retry_after_failure_s / 60:.0f} min.")
        self._record("slm_reinit_failed", text=text)
        self._set(FAILED, text)

    # --- reporting ---------------------------------------------------------------

    def snapshot(self) -> dict:
        st = self._status
        return {"label": self.config.label, "state": self._state, "detail": self._detail,
                "since": self._since, "reinit_due": bool(st.get("reinit_due")),
                "due_for_s": st.get("due_for_s"), "pattern_epoch": st.get("pattern_epoch"),
                "pattern": st.get("pattern"), "last_reinit": self._last_reinit,
                "last_poll": self._last_poll, "blocked_by": self._blocked_by}

    def _set(self, state: str, detail: str) -> None:
        if state == self._state and detail == self._detail:
            return
        if state != self._state:
            self._since = time.time()
            if state in (UNREACHABLE, NO_CONTROL, FAILED):
                self._log(f"{self.config.label}: {state} -- {detail}")
            self._record("slm_reinit_state", state=state, detail=detail)
        self._state, self._detail = state, detail
        if self._on_change is not None:
            try:
                self._on_change(self.snapshot())
            except Exception:
                pass

    def _record(self, kind: str, **fields) -> None:
        if self._journal is not None:
            try:
                self._journal.record(kind, **fields)
            except Exception:
                pass
