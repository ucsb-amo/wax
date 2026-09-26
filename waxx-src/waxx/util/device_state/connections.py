"""Connections the monitor experiment holds open while it runs.

Some devices the monitor drives are not ARTIQ channels but host-side
connections: the tweezer AWG is a Spectrum netbox that takes one connection at
a time.  The monitor opens each :class:`Connection` once its loop is running,
holds it so the Composite tab's ops can use it, and lets go of it whenever
something else needs it:

* A run announces itself (``run_pending``, from its ``finish_prepare``).  A
  run opens the AWG in ``init_kernel``, right after it takes the core, which
  is before the interrupted monitor would have exited.  So the monitor closes
  its connections as soon as the run is announced, while the run is still
  compiling.  If the run exits without taking the core (its fence is lifted),
  the monitor opens them again.
* The monitor exits: interrupted by a run, stopped, or restarted.  The monitor
  experiment's ``run()`` closes everything in a ``finally``.  A hard kill
  runs no Python at all: the OS drops the socket and the netbox frees the
  card, but the card is never stopped.  So the monitor server first asks a
  monitor that holds a connection to exit on its own, and kills it only if it
  does not (``MonitorManager.stop``).

A :class:`Connection` is the definition (the lab supplies its callables);
:class:`ConnectionManager` is the monitor-side runtime.  Neither imports Qt
or ARTIQ, so the GUI can load the definitions for their labels.

What the monitor reports per connection (the server keeps the last report,
broadcasts it as ``connections`` and serves it in ``status_json``):
``{"label", "state", "detail", "since", "want", "tooltip"}``, ``state`` one of
:data:`STATES`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

CONNECTED = "connected"
CONNECTING = "connecting"
DISCONNECTED = "disconnected"
FAILED = "failed"
STATES = (CONNECTED, CONNECTING, DISCONNECTED, FAILED)

#: Actions a GUI may request (``{"type": "connection", "key", "action"}``).
ACTIONS = ("connect", "disconnect")

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Connection:
    """One host-side connection the monitor holds.

    * ``connect(expt)`` opens it; raises on failure (its message is what the
      GUI shows).  It runs inside the monitor loop, which is stalled while it
      waits, so it must give up after a few seconds.
    * ``close(expt)`` closes it and must never hang.  Return False when the
      close could not be confirmed (a driver call that did not return): the
      monitor then will not reopen it in this process.
    * ``is_connected(expt)`` says whether it is open now.
    * ``detail(expt)`` is a short text shown while it is connected.
    * ``tooltip`` describes it on the GUI's pill; ``confirm`` is what the GUI
      says when asking whether to disconnect it (what stops).
    * ``at_start``: connect when the monitor starts.
    * ``busy_s``: the longest a connect or a close may stall the monitor loop
      (the server is told, so ops queued meanwhile do not expire).
    """

    key: str
    label: str
    connect: Callable[[Any], None]
    close: Callable[[Any], Any]
    is_connected: Callable[[Any], bool]
    detail: Callable[[Any], str] | None = None
    tooltip: str = ""
    confirm: str = ""
    at_start: bool = True
    busy_s: float = 10.0

    def validate(self) -> None:
        where = f"connection {self.key!r}"
        if not _KEY.match(self.key or ""):
            raise ValueError(f"{where}: key must be an identifier")
        for name in ("connect", "close", "is_connected"):
            if not callable(getattr(self, name)):
                raise ValueError(f"{where}: {name} must be callable")
        if self.detail is not None and not callable(self.detail):
            raise ValueError(f"{where}: detail must be callable")


def validate_connections(connections: Iterable[Connection]) -> tuple:
    connections = tuple(connections)
    keys = [c.key for c in connections]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate connection keys {keys}")
    for c in connections:
        if not isinstance(c, Connection):
            raise ValueError(f"not a Connection: {c!r}")
        c.validate()
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


class _Item:
    __slots__ = ("conn", "want", "state", "detail", "since", "blocked")

    def __init__(self, conn: Connection, now: float):
        self.conn = conn
        self.want = bool(conn.at_start)
        self.state = DISCONNECTED
        self.detail = ("connects once the monitor loop is running" if conn.at_start
                       else "not connected")
        self.since = now
        # Set when a close could not be confirmed: this process will not
        # reopen the connection (the old one may still be half open).
        self.blocked = ""


class ConnectionManager:
    """The monitor's side.  Drive it from the monitor loop:

    * :meth:`set_run_pending` with each poll's answer (None until the first
      poll: nothing is opened before the server has said no run is starting);
    * :meth:`request` for each GUI request the poll carried;
    * :meth:`service` to open/close what that implies;
    * :meth:`refresh` after composite ops (they may change a detail);
    * :meth:`close_all` when the monitor exits;
    * :meth:`flush_report` to send the states when they changed.

    ``report(snapshot, timeout)`` returns True when the server accepted it;
    ``busy(seconds)`` tells the server the loop is about to stall.  All calls
    come from the monitor's host thread (its RPCs), one at a time.
    """

    def __init__(self, expt, connections: Iterable[Connection],
                 report: Callable[[dict, float | None], bool] | None = None,
                 busy: Callable[[float], None] | None = None,
                 log: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.time):
        self.expt = expt
        self.connections = validate_connections(connections)
        self._clock = clock
        self._items = {c.key: _Item(c, clock()) for c in self.connections}
        self._report = report
        self._busy = busy
        self._log = log
        self._run_pending = None       # unknown until the first poll
        self._exiting = False
        self._dirty = True
        # key -> what the next close is reported as (a GUI's disconnect)
        self._close_why: dict[str, str] = {}

    # --- inputs -------------------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self._items)

    def keys(self) -> list[str]:
        return list(self._items)

    def set_run_pending(self, pending) -> None:
        """The server's answer: a run is starting (its record, or True), or
        not (None / False)."""
        self._run_pending = pending if pending else False

    def request(self, key: str, action: str, who: str = "") -> str:
        """A GUI's connect / disconnect.  Returns "" or why it was refused
        (the server checked the rest before handing it over)."""
        item = self._items.get(str(key))
        if item is None:
            return f"this monitor has no connection {key!r}"
        who = f" by {who}" if who else ""
        if action == "connect":
            if item.blocked:
                return item.blocked
            item.want = True
            if item.state != CONNECTED:
                self._log(f"[Monitor] {item.conn.label}: connect requested{who}.")
            return ""
        if action == "disconnect":
            item.want = False
            if item.state in (CONNECTED, CONNECTING):
                self._log(f"[Monitor] {item.conn.label}: disconnect requested{who}.")
                self._close_why[item.conn.key] = f"disconnected{who}"
            elif item.state == FAILED and not item.blocked:
                self._set(item, DISCONNECTED, f"not connected (cleared{who})")
            return ""
        return f"unknown action {action!r}"

    # --- work ------------------------------------------------------------------------

    def service(self) -> None:
        """Open what should be open, close what should not."""
        for item in self._items.values():
            connected = self._is_connected(item)
            if item.state == CONNECTED and not connected:
                # Closed by something other than this manager.
                item.want = False
                self._set(item, DISCONNECTED, "closed outside the connection bar")
            hold = item.want and not self._exiting and self._run_pending is False
            if hold and not connected and not item.blocked:
                self._open(item)
            elif not hold and connected:
                self._close(item, self._why_not_held(item))

    def refresh(self) -> None:
        """Re-read each connection (after ops): its detail, and whether it is
        still open."""
        for item in self._items.values():
            connected = self._is_connected(item)
            if item.state == CONNECTED:
                if not connected:
                    item.want = False
                    self._set(item, DISCONNECTED, "closed outside the connection bar")
                else:
                    detail = self._detail(item)
                    if detail != item.detail:
                        item.detail = detail
                        self._dirty = True

    def close_all(self, why: str = "the monitor is exiting") -> None:
        """Close everything and open nothing more.  Never raises."""
        self._exiting = True
        for item in self._items.values():
            if self._is_connected(item):
                self._close(item, why)
            elif item.state in (CONNECTING, CONNECTED):
                self._set(item, DISCONNECTED, why)

    def snapshot(self) -> dict:
        return {key: {"label": item.conn.label, "state": item.state,
                      "detail": item.detail, "since": item.since, "want": item.want,
                      "tooltip": item.conn.tooltip}
                for key, item in self._items.items()}

    def flush_report(self, timeout: float | None = None) -> bool:
        """Send the states if they changed since the last accepted report."""
        if not self._dirty or self._report is None:
            return not self._dirty
        try:
            accepted = bool(self._report(self.snapshot(), timeout))
        except Exception as e:
            self._log(f"[Monitor] WARNING: could not report the connection states ({e!r}).")
            accepted = False
        if accepted:
            self._dirty = False
        return accepted

    # --- internal --------------------------------------------------------------------

    def _set(self, item: _Item, state: str, detail: str) -> None:
        if state != item.state:
            item.since = self._clock()
        item.state = state
        item.detail = detail
        self._dirty = True

    def _is_connected(self, item: _Item) -> bool:
        try:
            return bool(item.conn.is_connected(self.expt))
        except Exception as e:
            self._log(f"[Monitor] {item.conn.label}: is_connected failed ({e!r}).")
            return False

    def _detail(self, item: _Item) -> str:
        if item.conn.detail is None:
            return ""
        try:
            return str(item.conn.detail(self.expt) or "")
        except Exception as e:
            return f"(detail failed: {e!r})"

    def _why_not_held(self, item: _Item) -> str:
        if self._exiting:
            return "the monitor is exiting"
        if self._run_pending:
            return (f"released for {describe_pending(self._run_pending)}, which is "
                    f"starting and opens it itself")
        return self._close_why.pop(item.conn.key, "disconnected")

    def _notify_busy(self, item: _Item) -> None:
        if self._busy is None:
            return
        try:
            self._busy(float(item.conn.busy_s))
        except Exception:
            pass

    def _open(self, item: _Item) -> None:
        label = item.conn.label
        self._set(item, CONNECTING, "")
        # The GUI shows "connecting" while this waits (it can take seconds).
        self.flush_report()
        self._notify_busy(item)
        try:
            item.conn.connect(self.expt)
        except Exception as e:
            item.want = False
            self._set(item, FAILED, error_text(e))
            self._log(f"[Monitor] {label}: could not connect: {item.detail}")
            return
        if not self._is_connected(item):
            item.want = False
            self._set(item, FAILED, "the connect returned, but the device reports no "
                                    "connection")
            self._log(f"[Monitor] {label}: {item.detail}")
            return
        self._set(item, CONNECTED, self._detail(item))
        self._log(f"[Monitor] {label}: connected" +
                  (f" ({item.detail})." if item.detail else "."))

    def _close(self, item: _Item, why: str) -> None:
        label = item.conn.label
        self._notify_busy(item)
        problem = ""
        try:
            result = item.conn.close(self.expt)
            if result is False:
                problem = "closing did not finish (see the monitor log)"
        except Exception as e:
            problem = f"closing raised {error_text(e)}"
        if not problem and self._is_connected(item):
            problem = "the close returned, but the device still reports a connection"
        if problem:
            item.blocked = (f"{problem}; this monitor will not reopen it -- restart the "
                            f"monitor to connect again")
            item.want = False
            self._set(item, FAILED, item.blocked)
            self._log(f"[Monitor] WARNING: {label}: {item.blocked}.")
            return
        self._set(item, DISCONNECTED, why)
        self._log(f"[Monitor] {label}: closed ({why}).")
