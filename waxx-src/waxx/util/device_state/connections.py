"""Host-side connections the monitor server holds between runs.

Some devices the Composite tab drives are not ARTIQ channels but host-side
connections: the tweezer AWG is a Spectrum netbox that takes one connection at
a time.  The monitor *server* holds each :class:`Connection` while the machine
is idle and hands it to a run:

* **Open** when the monitor experiment is running -- the proof that no run has
  the core -- or when a GUI asks for it, or when a run that announced itself
  exits without taking the core.  Not when the server starts: a run may be
  in progress.
* **Release** when a run announces itself (``run_pending``, from its
  ``finish_prepare``): the server closes the device *before it replies*, so by
  the time the run's ``finish_prepare`` returns the card is free.  A run opens
  the AWG in ``init_kernel``, right after it takes the core.  A run that does
  not announce itself is noticed when it takes the core (the monitor is
  interrupted) and the device is closed then; the run's ``awg_init`` retries
  while the card is in use.
* **Stay released** until the monitor runs again (so a run loop, which runs
  experiments back to back without the monitor in between, is not
  interrupted by reconnects).  A GUI may connect once the run has ended.
* The monitor restarting on its own (the restart button, a schema change)
  changes nothing: the device stays open, with whatever it was set to.

The driver runs in an agent process per connection session
(:mod:`waxx.util.device_state.connection_agent`), which closes the device if
the server goes away and can be killed if a close hangs.

A :class:`Connection` is the definition -- the driver's ``module:factory``
and its keyword arguments -- and imports nothing, so the GUI loads it for the
labels.  :class:`ConnectionService` is the server's runtime.

What the server reports per connection (broadcast as ``connections``, served
in ``status_json`` and ``get_state``): ``{"label", "state", "detail", "since",
"want", "tooltip"}``, ``state`` one of :data:`STATES`.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

CONNECTED = "connected"
CONNECTING = "connecting"
DISCONNECTED = "disconnected"
FAILED = "failed"
STATES = (CONNECTED, CONNECTING, DISCONNECTED, FAILED)

#: Actions a GUI may request (``{"type": "connection", "key", "action"}``).
ACTIONS = ("connect", "disconnect")

#: The whole synchronous release when a run announces itself.  The
#: announcement's reply waits for it, and clients give up after 5 s.
RELEASE_TIMEOUT_S = 3.5
#: How long a release waits for an operation in flight (an open, a write)
#: before it kills the agent instead.
RELEASE_WAIT_BUSY_S = 0.5

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Connection:
    """One host-side connection the monitor server holds.

    * ``driver``: ``"module:factory"``, built in the agent process with
      ``driver_kwargs`` (see :mod:`waxx.util.device_state.connection_agent`
      for what the object must have).
    * ``tooltip`` describes it on the GUI's pill; ``confirm`` is what the GUI
      says when asking whether to disconnect it (what stops).
    * ``auto``: open it whenever the machine is idle (see the module
      docstring); False: only when a GUI asks.
    * Timeouts: ``start_timeout_s`` for the agent to import and build the
      driver, ``open_timeout_s`` for its open (including any wait for a
      device someone else holds), ``close_timeout_s`` before a close is
      given up and the agent killed, ``call_timeout_s`` for other commands.
    """

    key: str
    label: str
    driver: str
    driver_kwargs: Mapping = field(default_factory=dict)
    tooltip: str = ""
    confirm: str = ""
    auto: bool = True
    start_timeout_s: float = 30.0
    open_timeout_s: float = 30.0
    close_timeout_s: float = 3.0
    call_timeout_s: float = 5.0

    def validate(self) -> None:
        where = f"connection {self.key!r}"
        if not _KEY.match(self.key or ""):
            raise ValueError(f"{where}: key must be an identifier")
        module, _, name = str(self.driver).partition(":")
        if not module or not name:
            raise ValueError(f"{where}: driver must be 'module:factory', got {self.driver!r}")
        if not isinstance(self.driver_kwargs, Mapping):
            raise ValueError(f"{where}: driver_kwargs must be a mapping")


def validate_connections(connections: Iterable[Connection]) -> tuple:
    connections = tuple(connections)
    for c in connections:
        if not isinstance(c, Connection):
            raise ValueError(f"not a Connection: {c!r}")
        c.validate()
    keys = [c.key for c in connections]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate connection keys {keys}")
    return connections


def error_text(e: BaseException) -> str:
    text = str(e).strip()
    return text if text else type(e).__name__


def describe_pending(pending) -> str:
    """ "run 81234 (hf_bec)" for a run_pending record; "a run" otherwise."""
    if isinstance(pending, Mapping):
        run_id, expt = pending.get("run_id"), str(pending.get("expt") or "")
        if run_id:
            return f"run {run_id}" + (f" ({expt})" if expt else "")
        if expt:
            return expt
    return "a run"


def _default_agent_factory(conn: Connection, log, on_exit):
    from waxx.util.device_state.connection_agent import AgentClient  # noqa: PLC0415
    return AgentClient(conn.driver, dict(conn.driver_kwargs), name=f"{conn.key} agent",
                       log=log, on_exit=on_exit)


class _Item:
    def __init__(self, conn: Connection, now: float):
        self.conn = conn
        self.want = bool(conn.auto)
        self.state = DISCONNECTED
        self.detail = ("opens when the monitor is running" if conn.auto else "not connected")
        self.since = now
        self.agent = None
        # Serializes everything that talks to the agent: the worker's opens
        # and closes, the monitor's commands, a release.
        self.op_lock = threading.Lock()
        self.pending_open = False
        self.close_why = ""


class ConnectionService:
    """The monitor server's connections (see the module docstring).

    Driven by the server: :meth:`run_starting` (synchronous release, bounded),
    :meth:`run_running`, :meth:`run_over`, :meth:`request` (a GUI's connect /
    disconnect), :meth:`call` (a driver command, e.g. the monitor's "Apply
    traps"), :meth:`stop`.  Opens and ordinary closes happen on its own worker
    thread (:meth:`start`), never on the server's request thread.
    ``on_change(snapshot, changed_keys)`` is called whenever a state changes,
    from whichever thread changed it."""

    def __init__(self, connections: Iterable[Connection] = (),
                 on_change: Callable[[dict, list], None] | None = None,
                 log: Callable[[str], None] | None = None,
                 agent_factory=None, clock: Callable[[], float] = time.time):
        self.connections = validate_connections(connections)
        self._clock = clock
        self._log = log or (lambda text: None)
        self._on_change = on_change
        self._agent_factory = agent_factory or _default_agent_factory
        self._cv = threading.Condition(threading.RLock())
        self._items = {c.key: _Item(c, clock()) for c in self.connections}
        self._run = None            # None, or {"phase": "starting"|"running", "name": str}
        self._stopping = False
        self._thread = None
        self._notify_lock = threading.Lock()
        self._reported: dict = {}

    def __bool__(self) -> bool:
        return bool(self._items)

    def keys(self) -> list[str]:
        return list(self._items)

    # --- thread -------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None and self._items:
            self._thread = threading.Thread(target=self._loop, name="connections", daemon=True)
            self._thread.start()

    def stop(self, why: str = "the monitor server is stopping") -> None:
        """Close everything (each close bounded) and stop the worker."""
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        for item in self._items.values():
            self._release(item, why, time.monotonic() + item.conn.close_timeout_s + 2.)
        if self._thread is not None:
            self._thread.join(timeout=2.)

    # --- events from the server ---------------------------------------------------------

    def run_starting(self, name: str, timeout: float = RELEASE_TIMEOUT_S) -> None:
        """A run announced itself: close everything before returning (bounded)."""
        with self._cv:
            self._run = {"phase": "starting", "name": str(name)}
            self._cv.notify_all()
        deadline = time.monotonic() + timeout
        for item in self._items.values():
            self._release(item, self._why_not_held(item), deadline)

    def run_running(self, name: str = "") -> None:
        """A run took the core (the monitor was interrupted).  Anything still
        open -- a run that did not announce itself -- is closed now, by the
        worker; the run's own open retries while the device is in use."""
        with self._cv:
            if self._run is None:
                self._run = {"phase": "running", "name": str(name or "a run")}
            else:
                self._run = dict(self._run, phase="running")
            self._cv.notify_all()

    def run_over(self, why: str, reopen: bool) -> None:
        """No run holds the core any more.  ``reopen``: open what should be
        open now (the monitor is running, or the run never took the core);
        otherwise leave it to the next of those, or a GUI."""
        with self._cv:
            self._run = None
            changed = []
            for item in self._items.values():
                if reopen:
                    if item.want and item.state not in (CONNECTED, CONNECTING):
                        item.pending_open = True
                    continue
                # Not now: only the monitor running (or a GUI) opens it again.
                item.pending_open = False
                if item.state == DISCONNECTED and item.want:
                    item.detail = "the run has ended; opens when the monitor is running"
                    changed.append(item)
            self._cv.notify_all()
        if changed:
            self._notify()

    def request(self, key: str, action: str, who: str = "") -> str:
        """A GUI's connect / disconnect.  Returns "" or why it was refused."""
        item = self._items.get(str(key))
        if item is None:
            return f"the monitor server has no connection {key!r}"
        who = f" by {who}" if who else ""
        with self._cv:
            if action == "connect":
                if self._run is not None:
                    return (f"{self._run['name']} is {self._run['phase']} and uses it -- it is "
                            f"opened again when the monitor is running")
                item.want = True
                item.pending_open = item.state not in (CONNECTED, CONNECTING)
                self._cv.notify_all()
                return ""
            if action == "disconnect":
                item.want = False
                item.pending_open = False
                item.close_why = f"disconnected{who}"
                if item.state == FAILED:
                    self._set(item, DISCONNECTED, f"not connected (cleared{who})")
                self._cv.notify_all()
            else:
                return f"unknown action {action!r}"
        self._notify()
        return ""

    def call(self, key: str, cmd: str, kwargs: Mapping | None = None) -> Any:
        """Run a driver command on an open connection (e.g. ``write_traps``).
        Raises :class:`ConnectionRefusedError` when it is not open (or a run
        has it), :class:`RuntimeError` when the command failed."""
        item = self._items.get(str(key))
        if item is None:
            raise ConnectionRefusedError(f"the monitor server has no connection {key!r}")
        with self._cv:
            if self._run is not None:
                raise ConnectionRefusedError(f"{self._run['name']} is {self._run['phase']} "
                                             f"and has the {item.conn.label}")
            if item.state != CONNECTED:
                raise ConnectionRefusedError(
                    f"the {item.conn.label} is not connected ({item.state}"
                    + (f": {item.detail}" if item.detail else "") + ")")
        if not item.op_lock.acquire(timeout=1.0):
            raise ConnectionRefusedError(f"the {item.conn.label} is busy (opening or closing)")
        try:
            from waxx.util.device_state.connection_agent import (  # noqa: PLC0415
                AgentCommandError, AgentError)
            agent = item.agent
            if agent is None or item.state != CONNECTED:
                raise ConnectionRefusedError(f"the {item.conn.label} is not connected")
            try:
                result = agent.call(cmd, timeout=item.conn.call_timeout_s, **dict(kwargs or {}))
            except AgentCommandError as e:
                self._update_detail(item, agent)
                raise RuntimeError(f"{item.conn.label} {cmd}: {e}") from None
            except AgentError as e:
                self._drop_agent(item)
                with self._cv:
                    self._set(item, FAILED, f"its agent stopped during {cmd} ({e}); the "
                                            f"connection was dropped, not closed")
                self._notify()
                raise RuntimeError(f"{item.conn.label} {cmd}: {e}") from None
            self._update_detail(item, agent)
            return result
        finally:
            item.op_lock.release()

    def snapshot(self) -> dict:
        with self._cv:
            return {key: {"label": item.conn.label, "state": item.state,
                          "detail": item.detail, "since": item.since, "want": item.want,
                          "tooltip": item.conn.tooltip}
                    for key, item in self._items.items()}

    # --- worker -------------------------------------------------------------------------

    def _hold(self, item: _Item) -> bool:
        return item.want and self._run is None and not self._stopping

    def _next_job(self):
        for item in self._items.values():
            if self._hold(item) and item.pending_open and item.state != CONNECTED:
                return item, "open"
            if not self._hold(item) and item.state == CONNECTED:
                return item, "close"
        return None, None

    def _loop(self) -> None:
        while True:
            with self._cv:
                item, job = self._next_job()
                while job is None and not self._stopping:
                    self._cv.wait(1.0)
                    item, job = self._next_job()
                if self._stopping:
                    return
            try:
                if job == "open":
                    self._open(item)
                else:
                    with item.op_lock:
                        self._close_locked(item, self._why_not_held(item),
                                           item.conn.close_timeout_s + 1.)
            except Exception as e:
                self._log(f"connection {item.conn.key}: {job} failed unexpectedly: "
                          f"{error_text(e)}")
                with self._cv:
                    item.pending_open = False
                    self._set(item, FAILED, f"{job} failed: {error_text(e)}")
                self._notify()

    def _open(self, item: _Item) -> None:
        from waxx.util.device_state.connection_agent import AgentError  # noqa: PLC0415
        conn = item.conn
        with item.op_lock:
            with self._cv:
                item.pending_open = False
                if not self._hold(item) or item.state == CONNECTED:
                    return
                self._set(item, CONNECTING, "starting its agent process")
                agent = self._agent_factory(conn, self._log,
                                            lambda a, i=item: self._agent_exited(i, a))
                item.agent = agent
            self._notify()
            try:
                agent.start(timeout=conn.start_timeout_s)
                with self._cv:
                    if self._hold(item):
                        self._set(item, CONNECTING, "opening")
                self._notify()
                agent.call("open", timeout=conn.open_timeout_s)
            except AgentError as e:
                self._drop_agent(item)
                with self._cv:
                    if self._hold(item):
                        self._set(item, FAILED, error_text(e))
                        self._log(f"{conn.label}: could not connect: {item.detail}")
                    else:
                        self._set(item, DISCONNECTED, self._why_not_held(item))
                self._notify()
                return
            with self._cv:
                # Connected even if a release came in just now: that release
                # (waiting for op_lock) or the worker's next pass closes it.
                self._set(item, CONNECTED, agent.state.get("detail", ""))
            self._log(f"{conn.label}: connected"
                      + (f" ({item.detail})." if item.detail else "."))
            self._notify()

    def _release(self, item: _Item, why: str, deadline: float) -> None:
        """Close one connection from a thread other than the worker, within
        ``deadline``: wait briefly for an operation in flight, else kill the
        agent (which ends that operation at once)."""
        with self._cv:
            if item.agent is None and item.state not in (CONNECTED, CONNECTING):
                if item.state == DISCONNECTED and item.want and self._run is not None:
                    item.detail = why
                    changed = True
                else:
                    changed = False
            else:
                changed = None
        if changed is not None:
            if changed:
                self._notify()
            return
        got = item.op_lock.acquire(timeout=max(min(RELEASE_WAIT_BUSY_S,
                                                   deadline - time.monotonic()), 0.))
        if not got:
            agent = item.agent
            if agent is not None:
                self._log(f"{item.conn.label}: busy ({item.state}) when it had to be "
                          f"released -- stopping its agent.")
                agent.kill()
            got = item.op_lock.acquire(timeout=max(deadline - time.monotonic(), 0.))
            if not got:
                self._log(f"WARNING: {item.conn.label}: could not confirm the release in "
                          f"time; its agent was stopped.")
                return
        try:
            self._close_locked(item, why, max(deadline - time.monotonic(), 0.2))
        finally:
            item.op_lock.release()

    def _close_locked(self, item: _Item, why: str, timeout: float) -> None:
        """Close through the agent and let it exit; kill it if that does not
        finish in time.  The caller holds ``item.op_lock``."""
        from waxx.util.device_state.connection_agent import AgentError  # noqa: PLC0415
        agent = item.agent
        detail = why
        if agent is not None and agent.alive:
            t0 = time.monotonic()
            closed = False
            try:
                closed = agent.call("close", timeout=min(item.conn.close_timeout_s,
                                                         timeout)) is not False
            except AgentError as e:
                self._log(f"{item.conn.label}: close failed: {error_text(e)}")
            if closed:
                agent.exit(timeout=max(min(1.0, timeout - (time.monotonic() - t0)), 0.2))
                self._log(f"{item.conn.label}: closed ({why}).")
            else:
                detail = (f"{why} -- the close did not finish, so its agent process was "
                          f"stopped: the connection was dropped, the device not stopped")
                self._log(f"WARNING: {item.conn.label}: {detail}.")
        self._drop_agent(item)
        with self._cv:
            self._set(item, DISCONNECTED, detail)
            item.close_why = ""
            item.pending_open = False
        self._notify()

    def _drop_agent(self, item: _Item) -> None:
        agent, item.agent = item.agent, None
        if agent is not None and agent.alive:
            agent.kill()

    def _agent_exited(self, item: _Item, agent) -> None:
        """An agent's process ended.  Expected after exit()/kill(); otherwise
        the connection is gone with it."""
        if getattr(agent, "expected_exit", True):
            return
        with self._cv:
            if item.agent is not agent:
                return
            item.agent = None
            if item.state in (CONNECTED, CONNECTING):
                self._set(item, FAILED, "its agent process exited unexpectedly (see the "
                                        "monitor server log)")
        self._log(f"WARNING: {item.conn.label}: its agent process exited unexpectedly.")
        self._notify()

    def _update_detail(self, item: _Item, agent) -> None:
        detail = str(agent.state.get("detail", ""))
        with self._cv:
            if item.state == CONNECTED and detail != item.detail:
                item.detail = detail
                changed = True
            else:
                changed = False
        if changed:
            self._notify()

    def _why_not_held(self, item: _Item) -> str:
        if self._stopping:
            return "the monitor server is stopping"
        if self._run is not None:
            return (f"released for {self._run['name']}, which "
                    + ("opens it itself" if self._run["phase"] == "starting"
                       else "has the core"))
        return item.close_why or "disconnected"

    def _set(self, item: _Item, state: str, detail: str) -> None:
        """Caller holds the condition's lock; call _notify() after releasing it."""
        if state != item.state:
            item.since = self._clock()
        item.state = state
        item.detail = detail

    def _notify(self) -> None:
        if self._on_change is None:
            return
        with self._notify_lock:
            snap = self.snapshot()
            changed = [k for k, v in snap.items() if self._reported.get(k) != v]
            if not changed:
                return
            self._reported = {k: dict(v) for k, v in snap.items()}
            try:
                self._on_change(snap, changed)
            except Exception as e:
                self._log(f"connection change report failed: {error_text(e)}")
