"""The monitor server's panel for the Server Dashboard: its run queue, its
state, and the monitor experiment's status button -- over the network.

The dashboard runs the monitor server headless (a subprocess with no window),
so the Queue and State tabs of the server's own window
(:class:`~waxx.util.guis.monitor_server_gui.MonitorServerGUI`) cannot be
embedded from it.  :class:`MonitorServerPanel` shows the same three tabs from
any PC:

* **Queue** -- :class:`~waxx.util.guis.run_queue_panel.RunQueuePanel`;
* **State** -- :class:`~waxx.util.guis.monitor_state_panel.MonitorStatePanel`;
* **Monitor** -- :class:`MonitorExperimentTab`: the monitor experiment's big
  READY / LOADING / NOT READY button as in the server's window (click:
  start it, or restart it after a confirm -- the server's ``reset``).

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
  loops are fed to the tabs as they come.

The dashboard calls :meth:`MonitorServerPanel.cleanup` when the panel goes:
the poll stops, the listener and the worker thread end.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QMessageBox, QPushButton, QTabWidget, QVBoxLayout, QWidget

from waxx.util.dashboard import theme
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


#: monitor state name -> (button text, background)
_LOOK = {"READY": ("READY", "green"), "LOADING": ("Loading...", "orange"),
         "NOT_READY": ("NOT READY", "#c46666")}


class MonitorExperimentTab(QWidget):
    """The monitor experiment's status button, as in the server's own window:
    READY (click: restart it, after a confirm), NOT READY (click: start it; a
    confirm first when a run has taken the core from it), Loading... (no
    click).  The click sends the server's ``reset`` command, which (re)starts
    the monitor -- unless the run queue has a job in its slot or about to
    launch, in which case the server defers it until the queue runs out."""

    def __init__(self, panel: "MonitorServerPanel"):
        super().__init__()
        self.panel = panel
        self.state: dict = {}
        box = QVBoxLayout(self)
        self.button = QPushButton("?")
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.button.setFont(font)
        self.button.clicked.connect(lambda _=False: self.clicked())
        box.addWidget(self.button)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(f"color: {theme.FG_MUTED}; font-size: 11px;")
        box.addWidget(self.detail)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet(f"color: {theme.FG_MUTED}; font-size: 11px;")
        box.addWidget(self.message)
        box.addStretch(1)
        self.set_state(None)

    def set_state(self, state: dict | None) -> None:
        self.state = dict(state) if isinstance(state, dict) else {}
        if not self.state:
            self.button.setText("monitor server not answering")
            self.button.setStyleSheet("background-color: #454545; color: white;")
            self.button.setEnabled(False)
            self.detail.setText("")
            return
        name = str(self.state.get("state_name") or "")
        text, bg = _LOOK.get(name, (name or "?", "#454545"))
        self.button.setText(text)
        self.button.setStyleSheet(f"background-color: {bg}; color: white;")
        self.button.setEnabled(name in ("READY", "NOT_READY"))
        sub = str(self.state.get("sub_state") or "").replace("_", " ")
        reason = str(self.state.get("reason") or "")
        self.detail.setText(" -- ".join(p for p in (sub, reason) if p))

    def clicked(self) -> bool:
        name = self.state.get("state_name")
        if name == "READY":
            if not self.panel.confirm("Restart Monitor", "Are you sure you'd like to restart "
                                      "the monitor experiment?"):
                return False
        elif name == "NOT_READY":
            if self.state.get("sub_state") == "interrupted_by_run" and not self.panel.confirm(
                    "Start Monitor", "A run has taken the core from the monitor and has not "
                    "ended. Starting the monitor takes the core back from that run. Start it?"):
                return False
        else:
            return False
        self.message.setText("asked the monitor server to (re)start the monitor...")

        def done(reply) -> None:
            self.message.setText("the monitor server did not answer" if reply is None else
                                 "asked; the monitor server starts it unless the run queue "
                                 "has a job to run first")
            self.panel.poll_status(force=True)
        self.panel.runner.call(lambda: self.panel.link.send_text("reset"), done)
        return True


class MonitorServerPanel(WidgetPanelBase):
    """The monitor server's dashboard panel (see the module docstring).

    ``link``: a :class:`MonitorLink` (tests pass one over a fake client);
    ``listener_factory``: makes the broadcast listener (a QThread with
    ``state_received(dict)``, ``start``, ``stop``, ``wait``), None for none;
    ``synchronous``: requests on the GUI thread (tests); ``by``: who is
    clicking (default ``user@host``)."""

    #: How often status_json is asked for while the panel is visible.
    STATUS_POLL_MS = 1000

    def __init__(self, parent=None, *, link: MonitorLink | None = None,
                 listener_factory: Callable[[], Any] | None | str = "default",
                 synchronous: bool = False, by: str | None = None):
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
        self.queue_panel = RunQueuePanel(runner=self.runner, by=by)
        self.state_panel = MonitorStatePanel(runner=self.runner, by=by)
        self.monitor_tab = MonitorExperimentTab(self)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.queue_panel, "Queue")
        self.tabs.addTab(self.state_panel, "State")
        self.tabs.addTab(self.monitor_tab, "Monitor")
        box.addWidget(self.tabs)
        self.timer = QTimer(self)
        self.timer.setInterval(self.STATUS_POLL_MS)
        self.timer.timeout.connect(self.poll_status)
        self.timer.start()

    # -- feeding the tabs ------------------------------------------------------------------

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
        status = status if isinstance(status, dict) else None
        self.queue_panel.set_state(status)
        self.state_panel.set_state(status)
        self.monitor_tab.set_state(status)

    def on_broadcast(self, payload) -> None:
        if not isinstance(payload, dict):
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


__all__ = ["MonitorExperimentTab", "MonitorLink", "MonitorServerPanel"]
