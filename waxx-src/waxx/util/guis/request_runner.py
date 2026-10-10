"""Requests to the monitor server through an injected callable, off the GUI
thread.

The run queue panel, the monitor state panel and (optionally) the Sequences
panel never discover the monitor server or open a socket themselves: the host
gives them a ``requester(request_dict) -> reply_dict``.  In the monitor
server's own window that is a direct call into the server
(``generate_reply``); elsewhere it can be a
:class:`~waxx.util.comms_server.comm_client.MonitorClient`-backed callable.

:class:`RequestRunner` runs the requester on one worker thread, in order, and
hands each reply to its callback on the GUI thread.  Every reply is a dict:
``None`` (no answer), a non-dict, or an exception become
``{"status": "error", "msg": ..., "no_reply": True}`` -- a panel never sees
anything else.  ``synchronous=True`` calls the requester at once, on the
caller's thread (tests; a host that is itself off the GUI thread).

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Callable

from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger(__name__)

#: Words in a refusal that mean "this server does not know that request"
#: (an older server): the control that sent it is disabled, not retried.
_UNKNOWN_WORDS = ("unknown run_queue action", "unknown action", "unknown type",
                  "unknown run loop action")


def normalize_reply(reply: Any) -> dict:
    """The reply as a dict (see the module docstring)."""
    if reply is None:
        return {"status": "error", "msg": "no reply from the monitor server", "no_reply": True}
    if not isinstance(reply, dict):
        return {"status": "error", "msg": f"unexpected reply from the monitor server: "
                                          f"{str(reply)[:120]}", "no_reply": True}
    return reply


def is_unknown_request(reply: dict | None) -> bool:
    """True when ``reply`` is a server's refusal of a request (or action) it
    does not know -- an older server -- rather than a refusal on the merits."""
    if not isinstance(reply, dict) or reply.get("status") == "ok":
        return False
    msg = str(reply.get("msg") or "").lower()
    return any(w in msg for w in _UNKNOWN_WORDS)


class RequestRunner(QObject):
    """Runs requests through ``requester`` and calls back on the GUI thread.

    ``send(obj, callback)`` queues ``requester(obj)``; ``call(fn, callback)``
    queues any ``fn()`` (the host's ``status_json`` poll).  Both return a
    request number.  Callbacks run on the thread this object lives on (the
    GUI thread), in the order the requests were sent.  ``shutdown()`` stops the
    worker (requests still queued are dropped, their callbacks never run)."""

    _finished = pyqtSignal(int, object)

    def __init__(self, requester: Callable[[dict], Any] | None, *,
                 synchronous: bool = False, parent=None):
        super().__init__(parent)
        self.requester = requester
        self.synchronous = bool(synchronous)
        self._n = 0
        self._callbacks: dict[int, Callable[[Any], None] | None] = {}
        self._jobs: deque = deque()
        self._cond = threading.Condition()
        self._running = True
        self._thread: threading.Thread | None = None
        self._finished.connect(self._deliver)

    # -- public -------------------------------------------------------------------

    def send(self, obj: dict, callback: Callable[[dict], None] | None = None) -> int:
        """Send one request dict; ``callback(reply_dict)``."""
        requester = self.requester
        if requester is None:
            def fn():
                return {"status": "error", "msg": "no monitor server link", "no_reply": True}
        else:
            def fn(obj=dict(obj)):
                return requester(obj)
        return self._queue(fn, callback, normalize=True)

    def call(self, fn: Callable[[], Any], callback: Callable[[Any], None] | None = None) -> int:
        """Run ``fn()`` on the worker; ``callback(result)`` (None when it raised)."""
        return self._queue(fn, callback, normalize=False)

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._running = False
            self._jobs.clear()
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                # a request still in the requester (a server not answering):
                # the daemon thread ends with it, its reply is dropped
                log.warning("The monitor panel's request worker is still waiting on a "
                            "request after %.1f s; it is left to finish (its reply is "
                            "dropped).", timeout)

    # -- plumbing -------------------------------------------------------------------

    def _queue(self, fn, callback, normalize: bool) -> int:
        self._n += 1
        n = self._n
        self._callbacks[n] = callback
        if self.synchronous:
            self._deliver(n, self._run_one(fn, normalize))
            return n
        with self._cond:
            if not self._running:
                self._callbacks.pop(n, None)
                return n
            self._jobs.append((n, fn, normalize))
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, daemon=True,
                                                name="monitor-panel-requests")
                self._thread.start()
            self._cond.notify()
        return n

    @staticmethod
    def _run_one(fn, normalize: bool):
        try:
            result = fn()
        except Exception as exc:                      # noqa: BLE001
            if not normalize:
                return None
            return {"status": "error", "msg": f"the request failed: {exc}", "no_reply": True}
        return normalize_reply(result) if normalize else result

    def _work(self) -> None:
        while True:
            with self._cond:
                while self._running and not self._jobs:
                    self._cond.wait(0.5)
                if not self._running:
                    return
                n, fn, normalize = self._jobs.popleft()
            result = self._run_one(fn, normalize)
            with self._cond:
                if not self._running:
                    return
            try:
                self._finished.emit(n, result)
            except RuntimeError:                      # the object is gone
                return

    def _deliver(self, n: int, result) -> None:
        callback = self._callbacks.pop(n, None)
        if callback is not None:
            callback(result)


__all__ = ["RequestRunner", "is_unknown_request", "normalize_reply"]
