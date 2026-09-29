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

An operator can ask for one now (``request_now``, the Device Control GUI's SLM
pill): the same check, no settle time. The SLM server restarts its hour after
every reinit, however it was asked for.

An operator can also restart the SLM server process (``request_restart``, the
pill's "Restart SLM server"): the same check, then the ``restart`` control
command. The server exits and its supervisor (``supervisor.py`` on the SLM PC)
starts a new one, which puts the saved pattern back; the service follows it
until a server with a new ``instance`` answers. Refused unless the server says
it is supervised -- a server nobody restarts would just be gone.

What the service reports (``snapshot()``, served in ``status_json`` as
``slm_reinit`` and broadcast as ``{"type": "slm_reinit"}``)::

    {"label", "state", "detail", "since", "reinit_due", "due_for_s",
     "next_due_at", "interval_s", "pattern_epoch", "pattern", "last_reinit",
     "last_poll", "blocked_by", "manual_pending", "last_manual",
     "supervised", "can_restart", "server_instance", "start_count", "last_exit",
     "slm_ready", "restart_pending", "last_restart"}

``next_due_at`` is the wall time the SLM server next marks a reinit due (as of
the last poll); the reinit itself follows once the machine is idle.

``state`` is one of :data:`STATES`.

The SLM server's log (``log_since``, the Device Control GUI's "View SLM server
log"): the supervisor on the SLM PC writes every line of the server, and its
own events, to a daily file; the ``log`` control command reads it back from a
cursor. While a GUI asks for the log (every second while its window is open)
this service fetches the new lines every ``LOG_POLL_S`` into :attr:`log`, an
:class:`~waxx.util.device_state.output_log.OutputLog` served through the
monitor server's ``output`` request (``kind`` ``"slm"``). Nobody looking:
nothing is fetched, and the next look carries on from the cursor (a long gap
is jumped, and the view says how much). The fetch runs on this service's
thread, after the reinit's work, so it never holds up a reinit request; the
SLM server answers it at once, never queued behind the SLM.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from waxx.util.device_state.output_log import OutputLog

UNKNOWN = "unknown"            # not polled yet
IDLE = "idle"                  # reachable, no reinit due
DUE = "due"                    # due, waiting for the machine to be idle
REINITIALISING = "reinitialising"
FAILED = "failed"              # the last reinit failed; retried after a pause
UNREACHABLE = "unreachable"
NO_CONTROL = "no_control"      # a server from before control commands
RESTARTING = "restarting"      # the server process is being restarted by its supervisor
STATES = (UNKNOWN, IDLE, DUE, REINITIALISING, FAILED, UNREACHABLE, NO_CONTROL, RESTARTING)
#: states in which the service is following a request (poll failures are expected)
_FOLLOWING = (REINITIALISING, RESTARTING)

#: The longest a run's announcement waits for a reinit request being sent.
#: The announcement's reply also waits for the connections' release
#: (connections.RELEASE_TIMEOUT_S), and clients give up after 5 s.
SEND_WAIT_S = 1.0

#: How often the SLM server's log is fetched while a GUI shows it.
LOG_POLL_S = 2.0
#: One GUI request for the log keeps it being fetched this long.
LOG_WANTED_S = 10.0
#: The pause after a fetch of the log failed.
LOG_RETRY_S = 10.0
#: The pause after the SLM server refused to send its log.
LOG_REFUSED_RETRY_S = 60.0
#: How often a reason not to ask (an old or unsupervised server) is looked at again.
LOG_RECHECK_S = 2.0
#: Lines of the SLM server's log kept here for GUIs.
LOG_KEEP_LINES = 3000
#: Lines the first fetch starts with.
LOG_TAIL_LINES = 300


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


def _amount(n_bytes: int) -> str:
    if n_bytes < 10_000:
        return f"{n_bytes} bytes"
    if n_bytes < 10_000_000:
        return f"{n_bytes / 1000:.0f} kB"
    return f"{n_bytes / 1e6:.0f} MB"


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
        self._next_due_at: float | None = None      # wall time the server marks it due
        self._manual: str | None = None             # who asked for a reinit now
        self._last_manual: dict | None = None       # {"by", "at", "result"}
        self._restart_req: str | None = None        # who asked for a server restart
        self._restart_t0: float | None = None       # clock time the restart was sent
        self._restart_instance0: str | None = None  # the server instance it replaces
        self._restart_by: str = ""
        self._last_restart: dict | None = None      # {"at", "t_s", "by", "instance", ...}

        #: The SLM server's log, as fetched while a GUI shows it (log_since).
        self.log = OutputLog(LOG_KEEP_LINES)
        self._log_cursor: dict | None = None        # the SLM server's cursor in its files
        self._log_wanted_until: float | None = None  # clock time
        self._log_next = 0.0                        # clock time
        self._log_problem = ""                      # the problem last said in the log
        self._log_follow = {"state": "idle", "detail": "", "path": "", "last_fetch": None}

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

    # --- an operator's reinit ------------------------------------------------------

    def request_now(self, who: str = "") -> str:
        """Re-initialise now rather than when due -- still only while nothing
        could be using the SLM: the same check as a due reinit here, and again
        under the lock when it is sent (the next tick, within ``TICK_S``); no
        settle time and no pause after a failure.  Returns "" when it will be
        sent, else why not."""
        with self._cond:
            why = self._blocker()
            if why:
                return why
            if self._state == REINITIALISING:
                return "the SLM is re-initialising already"
            if self._state == NO_CONTROL:
                return ("the SLM server takes no control commands (a server from before "
                        "2026-09-28, which re-initialises by itself)")
            if self._state == UNKNOWN:
                return "the SLM server has not answered yet"
            if self._manual is not None:
                return f"a reinit asked for by {self._manual} is about to be sent"
            self._manual = who or "an operator"
        self._notify()
        return ""

    # --- an operator's server restart ------------------------------------------------

    def restart_refusal(self) -> str:
        """Why a restart cannot be asked for now ("" when it can); the machine
        check (``blocker``) aside."""
        if self._state in _FOLLOWING:
            return f"the SLM server is {self._state.replace('_', ' ')} already"
        if self._state == NO_CONTROL:
            return ("the SLM server takes no control commands (a server from before "
                    "2026-09-28)")
        if self._state == UNKNOWN:
            return "the SLM server has not answered yet"
        if self._state == UNREACHABLE:
            return "the SLM server is not answering (restart it at the SLM PC)"
        st = self._status
        if "restart" not in (st.get("capabilities") or ()):
            return "this SLM server has no restart command (a server from before its supervisor)"
        if not st.get("supervised"):
            return ("the SLM server is not running under its supervisor (supervisor.py): "
                    "nothing would start it again")
        return ""

    def request_restart(self, who: str = "") -> str:
        """Restart the SLM server process: the same check as a reinit, sent on
        the next tick. Returns "" when it will be sent, else why not."""
        with self._cond:
            why = self._blocker() or self.restart_refusal()
            if why:
                return why
            if self._restart_req is not None:
                return f"a restart asked for by {self._restart_req} is about to be sent"
            if self._manual is not None:
                return f"a reinit asked for by {self._manual} is about to be sent"
            self._restart_req = who or "an operator"
        self._notify()
        return ""

    # --- one pass ----------------------------------------------------------------

    def tick(self) -> None:
        self._tick_reinit(self._clock())
        try:
            self._tick_log(self._clock())
        except Exception as e:  # the reinit's work must go on whatever the log does
            self._log_trouble(self._clock(), "bug", f"fetching the log failed here: {e!r}",
                              LOG_RETRY_S)

    def _tick_reinit(self, now: float) -> None:
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
        if self._state == RESTARTING:
            self._follow_restart(now)
            return
        if now >= self._next_poll:
            self._poll(now)
        with self._cond:
            restart, self._restart_req = self._restart_req, None
        if restart is not None:
            why = self.restart_refusal()
            if why:
                self._refused_restart(restart, why)
            else:
                self._request_restart(now, by=restart)
            return
        with self._cond:
            manual, self._manual = self._manual, None
        if manual is not None:
            if self._state in (REINITIALISING, NO_CONTROL):
                self._refused_manual(manual, f"the SLM server is {self._state.replace('_', ' ')}")
            else:
                self._request_reinit(now, by=manual)
            return
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
        following = self._state in _FOLLOWING
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
        next_in = status.get("next_due_in_s")
        self._next_due_at = (self._last_poll + next_in
                             if isinstance(next_in, (int, float)) else None)
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

    def _request_reinit(self, now: float, by: str | None = None) -> None:
        """Send the reinit: the due one (``by`` None) or an operator's."""
        with self._cond:
            why = self._blocker()
            if why:
                self._idle_since = None
                self._blocked_by = why
                if by is not None:
                    self._refused_manual(by, why)
                return
            self._sending = True
        no_reply, reply_timeout = _link_errors()

        def sent():
            with self._cond:
                self._sending = False
                self._cond.notify_all()

        epoch0 = self._status.get("pattern_epoch")
        asked = "the monitor server" if by is None else f"{by} (through the monitor server)"
        try:
            reply = self._link(self.config.host, self.config.port,
                               {"cmd": "reinit", "by": asked}, control=True,
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
        self._retry_at = 0.0
        if by is not None:
            self._last_manual = {"by": by, "at": time.time(), "result": "sent"}
        self._log(f"{self.config.label}: re-initialising (asked for by {asked}; "
                  f"the machine was idle).")
        self._record("slm_reinit_requested", epoch=epoch0, by=by or "",
                     due_for_s=self._status.get("due_for_s"))
        self._set(REINITIALISING, f"asked for by {by}" if by else "asked for by the monitor server")
        self._next_poll = now + 1.0

    def _refused_manual(self, who: str, why: str) -> None:
        self._log(f"{self.config.label}: the reinit {who} asked for was not sent: {why}.")
        self._record("slm_reinit_refused", by=who, msg=why)
        self._last_manual = {"by": who, "at": time.time(), "result": f"not sent: {why}"}
        self._notify()

    # --- a server restart --------------------------------------------------------------

    def _request_restart(self, now: float, by: str) -> None:
        """Send ``restart``; the run fence as for a reinit (a run announcing
        itself waits until the request is on its way; its own mask write then
        fails loudly if it lands while the server is down, never silently)."""
        with self._cond:
            why = self._blocker()
            if why:
                self._idle_since = None
                self._blocked_by = why
                self._refused_restart(by, why)
                return
            self._sending = True
        no_reply, reply_timeout = _link_errors()

        def sent():
            with self._cond:
                self._sending = False
                self._cond.notify_all()

        instance0 = self._status.get("instance")
        asked = f"{by} (through the monitor server)"
        try:
            reply = self._link(self.config.host, self.config.port,
                               {"cmd": "restart", "by": asked}, control=True,
                               until=("queued", "error"), connect_s=SEND_WAIT_S,
                               first_reply_s=3.0, total_s=3.0, on_sent=sent)
        except (no_reply, reply_timeout, OSError) as e:
            self._refused_restart(by, f"the restart request failed: {e or type(e).__name__}")
            return
        finally:
            sent()
        if reply.get("status") != "queued":
            self._refused_restart(by, f"the SLM server refused it: {reply.get('error')}")
            return
        self._restart_t0, self._restart_instance0, self._restart_by = now, instance0, by
        self._log(f"{self.config.label}: restarting the SLM server (asked for by {by}; the "
                  f"machine was idle).")
        self._record("slm_restart_requested", by=by, instance=instance0)
        self._set(RESTARTING, f"asked for by {by}")
        self._next_poll = now + 1.0

    def _refused_restart(self, who: str, why: str) -> None:
        self._log(f"{self.config.label}: the SLM server restart {who} asked for was not "
                  f"sent: {why}.")
        self._record("slm_restart_refused", by=who, msg=why)
        self._last_restart = {"by": who, "at": time.time(), "result": f"not sent: {why}"}
        self._notify()

    def _follow_restart(self, now: float) -> None:
        """While restarting: poll every second until a server with a new
        ``instance`` answers (the old one is gone for a few seconds)."""
        if now < self._next_poll:
            return
        status = self._poll(now)
        self._next_poll = now + 1.0
        t = now - (self._restart_t0 if self._restart_t0 is not None else now)
        if status is not None and status.get("instance") and \
                status.get("instance") != self._restart_instance0:
            self._last_restart = {"by": self._restart_by, "at": time.time(), "t_s": round(t, 1),
                                  "result": "done", "instance": status.get("instance"),
                                  "start_count": status.get("start_count"),
                                  "pattern_source": status.get("pattern_source"),
                                  "pattern": status.get("pattern"),
                                  "slm_ready": status.get("slm_ready")}
            ready = "" if status.get("slm_ready") else (
                f" -- but the SLM is not ready: {status.get('slm_not_ready')}")
            self._log(f"{self.config.label}: SLM server restarted in {t:.1f} s (instance "
                      f"{status.get('instance')}, {status.get('pattern_source')}){ready}.")
            self._record("slm_restart_done", t_s=round(t, 1), instance=status.get("instance"),
                         pattern_source=status.get("pattern_source"),
                         slm_ready=status.get("slm_ready"))
            self._restart_t0 = None
            self._next_poll = now + self.config.poll_s
            self._set(DUE if status.get("reinit_due") else IDLE, "")
            return
        if t > self.config.done_timeout_s:
            text = (f"no new SLM server answered within {self.config.done_timeout_s:.0f} s of "
                    f"the restart (check the supervisor window on the SLM PC)")
            self._restart_t0 = None
            self._last_restart = {"by": self._restart_by, "at": time.time(), "result": text}
            self._log(f"{self.config.label}: {text}.")
            self._record("slm_restart_failed", text=text)
            self._next_poll = now + self.config.poll_s
            self._set(UNREACHABLE, text)

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

    # --- the SLM server's log ---------------------------------------------------------

    def log_since(self, after=0) -> dict:
        """The log lines numbered above `after` (``OutputLog.since``) and
        ``follow`` (how the fetching goes: ``state`` idle / waiting / following
        / trouble, ``detail``, ``path`` of the files on the SLM PC,
        ``last_fetch``); the log is fetched for the next ``LOG_WANTED_S``."""
        self._log_wanted_until = self._clock() + LOG_WANTED_S
        return dict(self.log.since(after), follow=dict(self._log_follow))

    def _tick_log(self, now: float) -> None:
        until = self._log_wanted_until
        if until is None or now > until:
            if self._log_follow["state"] != "idle":
                self._set_log_follow("idle", "nobody is looking")
            return
        if now < self._log_next:
            return
        st = self._status
        if self._state == RESTARTING:
            # the supervisor's lines about it are fetched once the new server answers
            self._set_log_follow("waiting", "the SLM server is restarting")
            self._log_next = now + LOG_RECHECK_S
            return
        if not st:
            self._set_log_follow("waiting", "the SLM server has not answered the monitor "
                                            "server yet")
            self._log_next = now + LOG_RECHECK_S
            return
        if self._state == NO_CONTROL or "log" not in (st.get("capabilities") or ()):
            self._log_trouble(now, "old", "this SLM server has no log command (it is from "
                                          "before 2026-09-28): pull wax on the SLM PC, then "
                                          "restart the SLM server", LOG_RECHECK_S)
            return
        if not st.get("supervised"):
            self._log_trouble(now, "unsupervised", "the SLM server is not running under its "
                                                   "supervisor (supervisor.py), which writes its "
                                                   "log: its output is only in its window on "
                                                   "the SLM PC", LOG_RECHECK_S)
            return
        self._fetch_log(now)

    def _fetch_log(self, now: float) -> None:
        no_reply, reply_timeout = _link_errors()
        try:
            reply = self._link(self.config.host, self.config.port,
                               {"cmd": "log", "cursor": self._log_cursor, "tail": LOG_TAIL_LINES},
                               control=True, until=("ok", "error"), connect_s=1.0,
                               first_reply_s=2.0, total_s=4.0)
        except (no_reply, reply_timeout, OSError) as e:
            self._log_trouble(now, "no_answer", f"the SLM server did not send its log "
                                                f"({e or type(e).__name__}); trying again",
                              LOG_RETRY_S)
            return
        if reply.get("status") != "ok":
            self._log_trouble(now, "refused", f"the SLM server did not send its log: "
                                              f"{reply.get('error')}", LOG_REFUSED_RETRY_S)
            return
        if self._log_problem:
            self.log.mark("the SLM server's log comes through again")
            self._log_problem = ""
        if reply.get("restarted"):
            self.log.mark("the SLM server's log files changed (one was removed or cut short): "
                          "its last lines again")
        skipped = reply.get("skipped_bytes")
        if isinstance(skipped, int) and not isinstance(skipped, bool) and skipped > 0:
            self.log.mark(f"{_amount(skipped)} of the SLM server's log not fetched (nobody was "
                          f"looking); all of it is in {reply.get('path')} on the SLM PC")
        for line in reply.get("lines") or ():
            self.log.append(str(line))
        cursor = reply.get("cursor")
        self._log_cursor = cursor if isinstance(cursor, dict) else None
        self._log_next = now if reply.get("more") else now + LOG_POLL_S
        self._set_log_follow("following", "", path=str(reply.get("path") or ""),
                             last_fetch=time.time())

    def _log_trouble(self, now: float, key: str, text: str, retry_s: float) -> None:
        """A reason the log is not coming: said in the log once (until it
        changes or the log comes through again), and in ``follow``."""
        self._log_next = now + retry_s
        if key != self._log_problem:
            self._log_problem = key
            self.log.mark(text)
        self._set_log_follow("trouble", text)

    def _set_log_follow(self, state: str, detail: str, **kw) -> None:
        self._log_follow = dict(self._log_follow, state=state, detail=detail, **kw)

    # --- reporting ---------------------------------------------------------------

    def snapshot(self) -> dict:
        st = self._status
        return {"label": self.config.label, "state": self._state, "detail": self._detail,
                "since": self._since, "reinit_due": bool(st.get("reinit_due")),
                "due_for_s": st.get("due_for_s"), "next_due_at": self._next_due_at,
                "interval_s": st.get("interval_s"), "pattern_epoch": st.get("pattern_epoch"),
                "pattern": st.get("pattern"), "last_reinit": self._last_reinit,
                "last_poll": self._last_poll, "blocked_by": self._blocked_by,
                "manual_pending": self._manual, "last_manual": self._last_manual,
                "supervised": bool(st.get("supervised")),
                "can_restart": not self.restart_refusal(),
                "server_instance": st.get("instance"), "start_count": st.get("start_count"),
                "last_exit": st.get("last_exit"), "slm_ready": st.get("slm_ready"),
                "restart_pending": self._restart_req, "last_restart": self._last_restart}

    def _set(self, state: str, detail: str) -> None:
        if state == self._state and detail == self._detail:
            return
        if state != self._state:
            self._since = time.time()
            if state in (UNREACHABLE, NO_CONTROL, FAILED):
                self._log(f"{self.config.label}: {state} -- {detail}")
            self._record("slm_reinit_state", state=state, detail=detail)
        self._state, self._detail = state, detail
        self._notify()

    def _notify(self) -> None:
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
