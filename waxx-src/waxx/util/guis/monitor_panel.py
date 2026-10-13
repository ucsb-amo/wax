"""The monitor server's panel for the Server Dashboard: one panel, over the
network -- the monitor server's state as a compact bar on top, its run queue
below.

The dashboard runs the monitor server headless (a subprocess with no window),
so the server's own window
(:class:`~waxx.util.guis.monitor_server_gui.MonitorServerGUI`) cannot be
embedded from it.  :class:`MonitorServerPanel` shows the same from any PC:

* the bar -- :class:`~waxx.util.guis.monitor_state_panel.MonitorStatePanel`:
  the monitor experiment's READY / LOADING / NOT READY pill (click: start it,
  or restart it after a confirm -- the server's ``reset``, through
  :class:`MonitorControl`), trust, the run fence, connections, the SLM lease
  and the run loops as chips; its arrow opens the details, the loop cards and
  the ops journal;
* the queue -- :class:`~waxx.util.guis.run_queue_panel.RunQueuePanel`: its
  toolbar (filter, hold, pause), the jobs, each job's right-click menu.

How it talks to the server:

* requests -- :class:`MonitorLink`, a requester backed by a
  :class:`~waxx.util.comms_server.comm_client.MonitorClient`, run on one
  worker thread (:class:`~waxx.util.guis.request_runner.RequestRunner`).
  The client is made by discovery lazily, on the worker thread, at the first
  request -- never at import or construction -- and dropped after a request
  that got no answer, so the next one discovers the server again (a
  restarted server on a new port is found).  A request that changes
  something is sent once (attempts=1: the server may have acted on a copy
  whose reply was lost); reads may be retried once.
* ``status_json`` -- every :data:`MonitorServerPanel.STATUS_POLL_MS` while
  the panel is visible;
* broadcasts -- a :class:`~waxx.util.comms_server.state_broadcast.StateListener`
  (UDP, receive only), started when the panel is first shown; ``run_queue``,
  ``person_hold``, trust, the run fence, connections, the SLM reinit and the
  loops are fed to the bar and the queue as they come.

The dashboard calls :meth:`MonitorServerPanel.cleanup` when the panel goes:
the poll stops, the listener and the worker thread end.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox, QVBoxLayout

from waxx.util.dashboard.embed_helpers import WidgetPanelBase
from waxx.util.guis.monitor_state_panel import MonitorStatePanel
from waxx.util.guis.qt_upkeep import delete_later
from waxx.util.guis.request_runner import RequestRunner
from waxx.util.guis.run_queue_panel import RunQueuePanel

#: run_queue actions that only read (retried once after a rediscovery)
_QUEUE_READS = ("list", "describe", "tail")
#: request types that only read
_READ_TYPES = ("get_journal", "output", "get_state", "get_version", "op_status")


def _is_read(obj: dict) -> bool:
    kind = obj.get("type")
    if kind in _READ_TYPES:
        return True
    if kind == "run_queue":
        return obj.get("action") in _QUEUE_READS
    if kind == "run_loop":
        return obj.get("action") == "describe"
    return False


def _monitor_client(discovery_timeout: float):
    """A MonitorClient found by discovery (imported here, never at module
    import).  RuntimeError when no monitor server answers."""
    from waxx.util.comms_server.comm_client import MonitorClient  # noqa: PLC0415
    return MonitorClient(discovery_timeout=discovery_timeout)


class MonitorLink:
    """The panel's link to the monitor server (see the module docstring).

    ``client_factory(discovery_timeout)`` makes the client (default: a
    MonitorClient by discovery; tests pass a fake).  Thread-safe; meant to be
    called from the panel's one worker thread."""

    #: Seconds a request waits for its reply.
    TIMEOUT_S = 8.0

    def __init__(self, client_factory: Callable[[float], Any] | None = None,
                 discovery_timeout: float = 1.0):
        self._factory = client_factory or _monitor_client
        self._discovery_timeout = float(discovery_timeout)
        self._client = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._client is None:
                self._client = self._factory(self._discovery_timeout)
            return self._client

    def _drop(self, client) -> None:
        with self._lock:
            if self._client is client:
                self._client = None

    def request(self, obj: dict) -> dict | None:
        """``obj`` to the server -> its reply dict; an error dict when no
        server is found, None when it did not answer (the client is dropped:
        the next request discovers the server again)."""
        try:
            client = self._get()
        except Exception as exc:                      # noqa: BLE001
            return {"status": "error", "msg": f"no monitor server found ({exc})",
                    "no_reply": True}
        reply = client.request(obj, timeout=self.TIMEOUT_S, attempts=2 if _is_read(obj) else 1)
        if reply is None:
            self._drop(client)
        return reply

    def status(self) -> dict | None:
        """The server's ``status_json``, or None (no server, no answer)."""
        try:
            client = self._get()
        except Exception:                             # noqa: BLE001
            return None
        status = client.get_status()
        if status is None:
            self._drop(client)
        return status

    def send_text(self, text: str) -> str | None:
        """A plain-text command (``reset``) sent once; the raw reply or None."""
        try:
            client = self._get()
        except Exception:                             # noqa: BLE001
            return None
        reply = client.send_message(text, timeout=self.TIMEOUT_S, attempts=1)
        if reply is None:
            self._drop(client)
        return reply


class MonitorControl:
    """The monitor experiment's start / restart, behind the state bar's
    monitor pill (:class:`~waxx.util.guis.monitor_state_panel.MonitorStatePanel`):
    READY (click: restart it, after a confirm), NOT READY (click: start it; a
    confirm first, as in Device Control, whenever the status shows the core
    may be held -- :meth:`core_signs`), LOADING (no click).  The click sends
    the server's ``reset`` command, which (re)starts the monitor -- unless the
    run queue has a job in its slot or about to launch, in which case the
    server defers it until the queue runs out.  What happened goes to the
    queue panel's message line."""

    def __init__(self, panel: "MonitorServerPanel"):
        self.panel = panel
        self.state: dict = {}
        #: a reset sent and not answered yet: no click (one click, one
        #: command -- as Device Control's _command_in_flight)
        self.in_flight = False
        panel.state_panel.set_monitor_clickable(True)
        panel.state_panel.monitor_clicked.connect(self.clicked)

    @property
    def button(self):
        """The pill that shows the monitor's state (the state bar's)."""
        return self.panel.state_panel.monitor_pill

    def set_state(self, state: dict | None) -> None:
        self.state = dict(state) if isinstance(state, dict) else {}

    def core_signs(self) -> list[str]:
        """Every sign in ``status_json`` that an experiment may hold the core:
        the monitor was interrupted by a run; a run has announced itself (the
        run fence, ``run_pending``); a run queue job is in its slot
        (``run_queue.current``); liveOD says a run is in progress (``live_od``
        -- only if the server reports it; status_json does not today)."""
        s = self.state
        signs = []
        if s.get("sub_state") == "interrupted_by_run":
            signs.append("a run interrupted the monitor and has not reported its end")
        pending = s.get("run_pending")
        if isinstance(pending, dict):
            signs.append(f"run {pending.get('run_id')} ({pending.get('expt') or 'experiment'}) "
                         "has announced itself and holds the run fence")
        current = (s.get("run_queue") or {}).get("current")
        if isinstance(current, dict):
            run = f", run {current['run_id']}" if current.get("run_id") else ""
            signs.append(f"the run queue's job {current.get('id')} "
                         f"({current.get('label') or '?'}) is {current.get('state')}{run}")
        live_od = s.get("live_od")
        if isinstance(live_od, dict) and live_od.get("run_in_progress"):
            signs.append(f"liveOD says run {live_od.get('run_id')} "
                         f"({live_od.get('expt_name') or 'experiment'}) is in progress")
        return signs

    def clicked(self) -> bool:
        if self.in_flight or not self.state:
            return False
        name = self.state.get("state_name")
        signs = self.core_signs()
        why = ("\n\n" + "\n".join(f"- {s}" for s in signs)) if signs else ""
        if name == "READY":
            # Device Control's restart question, plus what the status shows
            if not self.panel.confirm(
                    "Restart monitor",
                    "Restart the monitor experiment? If another experiment is currently "
                    "using the core device this will interrupt it." + why):
                return False
        elif name == "NOT_READY":
            # Device Control's start question, whenever the core may be held
            if signs and not self.panel.confirm(
                    "Start monitor",
                    "An experiment probably holds the core device." + why + "\n\n"
                    "Starting the monitor takes the core and cuts that experiment off.\n\n"
                    "Start the monitor anyway?"):
                return False
        else:
            return False
        self.panel.queue_panel.say("asked the monitor server to (re)start the monitor...")
        self.in_flight = True
        self.panel.state_panel.set_monitor_busy(True)

        def done(reply) -> None:
            self.in_flight = False
            self.panel.state_panel.set_monitor_busy(False)
            self.panel.queue_panel.say(
                "monitor: the monitor server did not answer" if reply is None else
                "monitor: asked; the monitor server starts it unless the run queue has a job "
                "to run first")
            self.panel.poll_status(force=True)
        self.panel.runner.call(lambda: self.panel.link.send_text("reset"), done)
        return True


class MonitorServerPanel(WidgetPanelBase):
    """The monitor server's dashboard panel (see the module docstring).

    ``link``: a :class:`MonitorLink` (tests pass one over a fake client);
    ``listener_factory``: makes the broadcast listener (a QThread with
    ``state_received(dict)``, ``start``, ``stop``, ``wait``), None for none;
    ``synchronous``: requests on the GUI thread (tests); ``by``: who is
    clicking (default ``user@host``); ``settings``: a QSettings for the queue
    table's columns and filter (``run_queue_panel.panel_settings()``), None
    remembers nothing."""

    #: How often status_json is asked for while the panel is visible.
    STATUS_POLL_MS = 1000

    def __init__(self, parent=None, *, link: MonitorLink | None = None,
                 listener_factory: Callable[[], Any] | None | str = "default",
                 synchronous: bool = False, by: str | None = None, settings=None):
        super().__init__(parent)
        self.link = link or MonitorLink()
        # "default": a StateListener, imported at first show (importing the
        # waxx.util.comms_server package starts the discovery listener;
        # constructing this panel binds nothing)
        self._listener_factory = listener_factory
        self.listener = None
        self._polling = False
        self._cleaned = False
        self.runner = RequestRunner(self.link.request, synchronous=synchronous, parent=self)
        self.state_panel = MonitorStatePanel(runner=self.runner, by=by)
        self.queue_panel = RunQueuePanel(runner=self.runner, by=by, settings=settings)
        self.monitor_control = MonitorControl(self)
        box = QVBoxLayout(self)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(2)
        box.addWidget(self.state_panel)
        box.addWidget(self.queue_panel, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(self.STATUS_POLL_MS)
        self.timer.timeout.connect(self.poll_status)
        self.timer.start()

    # -- feeding the parts ------------------------------------------------------------------

    def showEvent(self, event):                                 # noqa: N802
        super().showEvent(event)
        if self.listener is None and self._listener_factory is not None and not self._cleaned:
            factory = self._listener_factory
            if factory == "default":
                from waxx.util.comms_server.state_broadcast import StateListener  # noqa: PLC0415
                factory = StateListener
            self.listener = factory()
            self.listener.state_received.connect(self.on_broadcast)
            self.listener.start()
        self.poll_status()

    def poll_status(self, force: bool = False) -> None:
        if self._polling or self._cleaned or not (force or self.isVisible()):
            return
        self._polling = True
        self.runner.call(self.link.status, self._on_status)

    def _on_status(self, status) -> None:
        self._polling = False
        if self._cleaned:
            return
        status = status if isinstance(status, dict) else None
        self.queue_panel.set_state(status)
        self.state_panel.set_state(status)
        self.monitor_control.set_state(status)

    def on_broadcast(self, payload) -> None:
        """A broadcast to the bar and the queue; a no-op once cleaned up (a listener's
        queued datagram may still arrive)."""
        if self._cleaned or not isinstance(payload, dict):
            return
        self.queue_panel.on_broadcast(payload)
        self.state_panel.on_broadcast(payload)

    # -- dialogs (tests replace them) ----------------------------------------------------------

    def confirm(self, title: str, text: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(text)
        yes = box.addButton("Yes", QMessageBox.ButtonRole.AcceptRole)
        no = box.addButton("No", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(no)
        box.exec()
        clicked = box.clickedButton()
        delete_later(box)
        return clicked is yes

    # -- teardown ------------------------------------------------------------------------------

    def cleanup(self) -> None:
        """The dashboard's teardown: the poll, the listener, the worker
        (safe to call twice)."""
        if self._cleaned:
            return
        self._cleaned = True
        self.timer.stop()
        if self.listener is not None:
            try:
                self.listener.stop()
                self.listener.wait(2000)
            except Exception:                         # noqa: BLE001
                pass
            self.listener = None
        self.queue_panel.shutdown()
        self.state_panel.shutdown()
        self.runner.shutdown()

    def closeEvent(self, event):                                # noqa: N802
        self.cleanup()
        super().closeEvent(event)


__all__ = ["MonitorControl", "MonitorLink", "MonitorServerPanel"]
