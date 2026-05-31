"""Snapshot poller: drives the conn-badge + panel body from a remote server.

Per Threat 9 in the plan, snapshot polls must:

* never block the Qt main thread
* enforce a hard socket timeout (1.5 s for snapshots; 5 s ceiling)
* throttle log spam on consecutive failures
* auto-throttle poll rate after 5 consecutive failures
* deliver results to the main thread via Qt signals

The poller owns no socket itself; it accepts a ``client`` object that exposes
``get_snapshot()``.  Existing TCP wrappers fit this contract without changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QTimer, QThreadPool, QRunnable, pyqtSignal


_LOG = logging.getLogger("waxx.dashboard.snapshot")


class _PollTask(QRunnable):
    """Run ``client.get_snapshot()`` on a thread-pool thread."""

    def __init__(self, client: Any, on_done: Callable[[Optional[dict], Optional[BaseException]], None]):
        super().__init__()
        self.setAutoDelete(True)
        self._client = client
        self._on_done = on_done

    def run(self) -> None:  # noqa: D401 - QRunnable hook
        try:
            snap = self._client.get_snapshot()
        except BaseException as exc:  # noqa: BLE001 - we re-raise the type to caller
            self._on_done(None, exc)
            return
        self._on_done(snap if isinstance(snap, dict) else {"raw": snap}, None)


class SnapshotPoller(QObject):
    """Periodically calls ``client.get_snapshot()`` and emits the result.

    Signals
    -------
    snapshot_received(dict)
        Emitted on every successful poll.  The dict comes verbatim from the
        client (already JSON-decoded by the client wrapper).
    conn_changed(str, str)
        Emitted whenever the connection-status label changes.  ``status`` is
        one of ``"connected"``, ``"disconnected"``, ``"error"``; ``detail`` is
        a short tooltip-suitable string (e.g. last error class).
    """

    snapshot_received = pyqtSignal(dict)
    conn_changed = pyqtSignal(str, str)

    NORMAL_INTERVAL_MS = 1000
    THROTTLED_INTERVAL_MS = 5000
    THROTTLE_AFTER_FAILURES = 5
    HARD_TIMEOUT_CEILING_S = 5.0

    def __init__(
        self,
        client: Any,
        *,
        panel_id: str = "",
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._client = client
        self._panel_id = panel_id
        self._consecutive_failures = 0
        self._in_flight = False
        self._last_status: Optional[str] = None

        # Sanity-check the client's socket timeout (Threat 9).
        client_timeout = getattr(client, "timeout", None)
        if isinstance(client_timeout, (int, float)) and client_timeout > 2.0:
            _LOG.warning(
                "panel=%s: client.timeout=%.1fs exceeds 2s recommendation",
                panel_id, client_timeout,
            )

        self._timer = QTimer(self)
        self._timer.setInterval(self.NORMAL_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
            # First poll runs immediately for snappier startup.
            QTimer.singleShot(0, self._tick)

    def stop(self) -> None:
        self._timer.stop()

    def force_unknown(self) -> None:
        """Reset conn badge to disconnected (used on app suspend / resume)."""
        self._last_status = None
        self.conn_changed.emit("disconnected", "")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if self._in_flight:
            return  # Skip overlapping polls; the previous one is still working.
        self._in_flight = True
        task = _PollTask(self._client, self._on_poll_done)
        QThreadPool.globalInstance().start(task)

    def _on_poll_done(self, snap: Optional[dict], exc: Optional[BaseException]) -> None:
        # NOTE: this runs on a worker thread.  Emit signals so the slots run
        # on the main thread (Qt::AutoConnection -> Qt::QueuedConnection across
        # threads).
        self._in_flight = False
        if exc is not None:
            self._record_failure(exc)
            return
        self._record_success(snap or {})

    def _record_success(self, snap: dict) -> None:
        if self._consecutive_failures >= self.THROTTLE_AFTER_FAILURES:
            _LOG.info(
                "panel=%s: snapshot recovered after %d consecutive failures",
                self._panel_id, self._consecutive_failures,
            )
            self._timer.setInterval(self.NORMAL_INTERVAL_MS)
        self._consecutive_failures = 0
        if self._last_status != "connected":
            self._last_status = "connected"
            self.conn_changed.emit("connected", "")
        self.snapshot_received.emit(snap)

    def _record_failure(self, exc: BaseException) -> None:
        self._consecutive_failures += 1
        detail = f"{type(exc).__name__}: {exc}"[:160]
        if self._consecutive_failures == 1:
            _LOG.warning("panel=%s: snapshot poll failed: %s", self._panel_id, detail)
        else:
            _LOG.debug("panel=%s: snapshot poll failed (#%d): %s",
                       self._panel_id, self._consecutive_failures, detail)
        if (self._consecutive_failures == self.THROTTLE_AFTER_FAILURES
                and self._timer.interval() == self.NORMAL_INTERVAL_MS):
            self._timer.setInterval(self.THROTTLED_INTERVAL_MS)
            _LOG.warning(
                "panel=%s: throttling snapshot poll to %d ms after %d failures",
                self._panel_id, self.THROTTLED_INTERVAL_MS, self._consecutive_failures,
            )
        if self._last_status != "error":
            self._last_status = "error"
            self.conn_changed.emit("error", detail)


__all__ = ["SnapshotPoller"]
