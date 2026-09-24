"""Per-server client link: builds the TCP client off-thread, polls snapshots,
drives the panel header's conn badge + COM pill, and offers the graceful
shutdown request to the supervisor.

One :class:`ServerLink` per ``ServerSpec`` with a ``client_factory``.  The
link follows the supervisor: it connects once the server is RUNNING (or
EXTERNAL), polls while it is up, and goes quiet when it stops, so the header
never shows a stale "OK" for a dead server.

Client contract (duck-typed):

* ``get_snapshot() -> dict``  - raises on network failure
* ``request_shutdown() -> bool``  - optional; enables graceful stop
* ``timeout`` attribute <= 2 s  - checked by :class:`SnapshotPoller`
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from waxx.util.dashboard.server_supervisor import SupervisorState
from waxx.util.dashboard.snapshot_poller import SnapshotPoller


_LOG = logging.getLogger("waxx.dashboard.link")


class _ClientBuildWorker(QThread):
    """Construct the client on a background thread (discovery can take seconds)."""

    done = pyqtSignal(object, object)  # (client_or_None, exception_or_None)

    def __init__(self, factory: Callable[[], Any], parent: Optional[QObject] = None):
        super().__init__(parent)
        self._factory = factory

    def run(self) -> None:
        try:
            self.done.emit(self._factory(), None)
        except BaseException as exc:  # noqa: BLE001
            self.done.emit(None, exc)


class ServerLink(QObject):
    """Wire a server's client to its panel header and supervisor."""

    #: Emitted with the snapshot dict on every successful poll.
    snapshot_received = pyqtSignal(dict)
    #: Emitted once the client exists.
    client_ready = pyqtSignal(object)

    # How long after RUNNING to wait before the first connection attempt so
    # the server has bound its socket and sent a beacon.
    CONNECT_DELAY_MS = 1500
    RETRY_DELAY_MS = 4000

    def __init__(
        self,
        panel_id: str,
        client_factory: Callable[[], Any],
        *,
        header=None,
        supervisor=None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.panel_id = panel_id
        self._factory = client_factory
        self._header = header
        self._sup = supervisor
        self._client: Any = None
        self._poller: Optional[SnapshotPoller] = None
        self._worker: Optional[_ClientBuildWorker] = None
        self._want_connected = False
        self._retry = QTimer(self)
        self._retry.setSingleShot(True)
        self._retry.timeout.connect(self._build_client)

        if supervisor is not None:
            supervisor.state_changed.connect(self._on_supervisor_state)
            supervisor.shutdown_request = self.request_shutdown
            self._on_supervisor_state(supervisor.state)
        else:
            # No supervisor (client dashboard): connect right away.
            self.connect_now()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        return self._client

    def connect_now(self, delay_ms: int = 0) -> None:
        self._want_connected = True
        if self._client is not None:
            self._start_poller()
            return
        self._retry.start(max(0, int(delay_ms)))

    def disconnect_now(self) -> None:
        self._want_connected = False
        self._retry.stop()
        if self._poller is not None:
            self._poller.stop()
        if self._header is not None:
            self._header.set_conn("disconnected", "server not running")
            com_btn = self._header.com_button()
            if com_btn is not None:
                com_btn.set_status("disconnected", detail="server not running")

    def request_shutdown(self) -> bool:
        """Ask the server to exit cleanly.  Runs on the caller's thread."""
        client = self._client
        if client is None:
            # Server may be running from a previous dashboard session; try a
            # one-off client with a short discovery budget.
            try:
                client = self._factory()
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("%s: no client for shutdown request: %r", self.panel_id, exc)
                return False
        fn = getattr(client, "request_shutdown", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("%s: request_shutdown raised: %r", self.panel_id, exc)
            return False

    def stop(self) -> None:
        self.disconnect_now()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_supervisor_state(self, state) -> None:
        if state in (SupervisorState.RUNNING, SupervisorState.EXTERNAL):
            if self._header is not None and self._client is None:
                self._header.set_conn("connecting", "waiting for the server to answer")
            self.connect_now(delay_ms=self.CONNECT_DELAY_MS)
        elif state in (SupervisorState.IDLE, SupervisorState.CRASHED,
                       SupervisorState.FAILED, SupervisorState.STOPPING):
            self.disconnect_now()

    def _build_client(self) -> None:
        if not self._want_connected or self._worker is not None:
            return
        self._worker = _ClientBuildWorker(self._factory, parent=self)
        self._worker.done.connect(self._on_client_built)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_client_built(self, client, exc) -> None:
        self._worker = None
        if not self._want_connected:
            return
        if client is None:
            _LOG.info("%s: client not available yet (%s); retrying in %d ms",
                      self.panel_id, type(exc).__name__ if exc else "?", self.RETRY_DELAY_MS)
            if self._header is not None:
                self._header.set_conn("error", f"discovery: {exc!r}"[:160])
            self._retry.start(self.RETRY_DELAY_MS)
            return
        self._client = client
        self.client_ready.emit(client)
        self._start_poller()

    def _start_poller(self) -> None:
        if self._client is None:
            return
        if self._poller is None:
            self._poller = SnapshotPoller(self._client, panel_id=self.panel_id, parent=self)
            self._poller.snapshot_received.connect(self._on_snapshot)
            if self._header is not None:
                self._poller.conn_changed.connect(self._header.set_conn)
        self._poller.start()

    def _on_snapshot(self, snap: dict) -> None:
        if self._header is not None:
            com_btn = self._header.com_button()
            if com_btn is not None:
                self._header.set_com(snap.get("com") if isinstance(snap, dict) else None)
        self.snapshot_received.emit(snap)


__all__ = ["ServerLink"]
