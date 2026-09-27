"""The camera host's snapshots, as a Qt signal on the GUI thread.

The host calls its snapshot callbacks on its own thread (after every change and
at least once a second).  ``HostQtBridge.snapshot_changed`` re-emits each one;
a slot on a QObject of the GUI thread receives it there through Qt's queued
delivery.  Nothing blocks the host's thread (no BlockingQueuedConnection).
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class HostQtBridge(QObject):
    """``snapshot_changed(snapshot: dict)``; ``latest()`` is the host's snapshot now.
    ``close()`` (before the bridge is deleted) stops the host calling it."""

    snapshot_changed = pyqtSignal(object)

    def __init__(self, host, parent=None) -> None:
        super().__init__(parent)
        self._host = host
        self._closed = False
        self._unsubscribe = host.on_snapshot(self._on_snapshot)

    def _on_snapshot(self, snapshot) -> None:
        if self._closed:
            return
        try:
            self.snapshot_changed.emit(snapshot)
        except RuntimeError:            # the QObject is already gone
            self._closed = True

    def latest(self) -> dict:
        return self._host.snapshot()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._unsubscribe()
        except Exception:
            pass


__all__ = ["HostQtBridge"]
