"""Abort, for panels away from liveOD's window (the Server Dashboard's monitor
panel): what liveOD's own Abort button and the remote viewer's do.

A click asks liveOD (POLL) which run is in progress; with none, nothing is
sent ("there is no run to abort") -- a RESET with no run would count as a
person's Reset and put the person hold on.  Otherwise it sends liveOD's
``RESET`` (``source: "person"``) naming that run id, so a late click never
aborts the next run.  liveOD then aborts the run at its next shot and discards
its file (liveOD's rule, unchanged), ignores it while the run is being saved,
and the monitor server puts the person hold on, as for a press in liveOD.

No confirmation (it has to be fast) and never disabled: if the panel's idea of
whether a run is active were wrong, a disabled Abort would be the worst way to
find out.  It is bold red while :meth:`LiveODAbortButton.set_state` sees a run
holding the machine.

The liveOD client is made at the first click, on a worker thread, never at
construction; a failed request drops it so the next click rediscovers liveOD.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QPushButton

ACTIVE_CSS = "background-color: #e53935; color: white; font-weight: bold;"

TOOLTIP = ("Abort the run in progress (sent to liveOD, as its own Abort button): it stops at "
           "its next shot and its data file is deleted; ignored while a run is being saved (it "
           "is complete then). No confirmation: it has to be fast.")


def reply_notice_text(label: str, reply) -> str:
    """The text for a liveOD reply that says the request was ignored or
    refused (a Reset while the run is being saved, or naming another run; any
    other request answered ``ok: False`` with an error) or that it closed a
    run whose process is gone, keeping its file; "" for any other reply."""
    if not isinstance(reply, dict):
        return ""
    if not (reply.get('ignored') or reply.get('saving') or reply.get('refused')
            or reply.get('kept')):
        if reply.get('ok', True) is False and reply.get('error'):
            return f"{label} refused: {reply['error']}"
        return ""
    if reply.get('kept') and reply.get('message'):
        return str(reply['message'])
    text = str(reply.get('message') or reply.get('error') or '')
    if not text:
        what = "ignored" if reply.get('ignored') else "refused"
        run = f"run {reply['run_id']} " if reply.get('run_id') else "the run "
        text = f"{label} {what}: {run}is being saved (it is complete)"
    return text


def run_is_active(status: dict | None) -> bool:
    """The monitor server's ``status_json`` shows a run holding the machine:
    a run has announced itself (``run_pending``), the run queue has a job in
    its slot, the monitor was interrupted by a run, or liveOD says a run is in
    progress (when the server reports it)."""
    if not isinstance(status, dict):
        return False
    if isinstance(status.get("run_pending"), dict):
        return True
    if isinstance((status.get("run_queue") or {}).get("current"), dict):
        return True
    if status.get("sub_state") == "interrupted_by_run":
        return True
    live_od = status.get("live_od")
    return bool(isinstance(live_od, dict) and live_od.get("run_in_progress"))


class _Unanswered(Exception):
    """A liveOD request got no answer; ``str()`` says what was sent."""


def _live_od_client():
    from waxx.util.live_od.live_od_client import LiveODClient  # noqa: PLC0415
    return LiveODClient(timeout_ms=3000, discovery_timeout=3.0)


def abort_run(client) -> tuple[str, bool]:
    """POLL, then RESET the run in progress through ``client`` (a
    LiveODClient: ``poll()``, ``_send_recv(msg)``) -> (what happened, ok).
    A network error is reported, with whether the Reset had been sent; the
    exception is re-raised for the caller to drop the client."""
    try:
        poll = client.poll()
    except Exception as exc:
        raise _Unanswered(f"Abort: liveOD did not answer ({exc}); nothing was sent.") from exc
    if not isinstance(poll, dict) or not poll.get("ok", True):
        return f"Abort: liveOD's POLL failed ({poll}); nothing was sent.", False
    if not poll.get("run_in_progress"):
        return "Abort: there is no run to abort.", True
    run_id = int(poll.get("run_id") or 0)
    payload: dict[str, Any] = {"tag": "RESET", "source": "person"}
    if run_id:
        payload["run_id"] = run_id          # refused if another run is in progress by then
    name = f"run {run_id}" if run_id else "the unsaved run"
    try:
        reply = client._send_recv(payload)
    except Exception as exc:
        raise _Unanswered(f"Abort: the Reset for {name} was sent but liveOD did not answer "
                          f"({exc}); look at liveOD to see whether the run stopped.") from exc
    notice = reply_notice_text("Abort", reply)
    if notice:
        return notice, False
    if not isinstance(reply, dict) or reply.get("ok") is False:
        return f"Abort: liveOD did not take the Reset for {name} ({reply}).", False
    return f"Abort: sent to liveOD for {name}; it stops at its next shot.", True


class LiveODAbortButton(QPushButton):
    """See the module docstring.  ``say(text)`` gets what happened (on the
    GUI thread); ``client_factory()`` makes the liveOD client (tests pass a
    fake); ``synchronous``: the click's request on the GUI thread (tests)."""

    #: (text, ok) from the worker thread
    result_ready = pyqtSignal(str, bool)

    def __init__(self, parent=None, *, say: Callable[[str], None] | None = None,
                 client_factory: Callable[[], Any] | None = None, synchronous: bool = False):
        super().__init__("Abort", parent)
        self.setToolTip(TOOLTIP)
        self.setFixedWidth(56)          # turning red must not change the bar's width
        self._say = say
        self._factory = client_factory or _live_od_client
        self._synchronous = synchronous
        self._client = None
        self._lock = threading.Lock()
        self.active = False
        self.result_ready.connect(self._on_result)
        self.clicked.connect(lambda _=False: self.abort())

    def set_state(self, status: dict | None) -> None:
        """The monitor server's status: red while a run holds the machine."""
        active = run_is_active(status)
        if active != self.active:
            self.active = active
            self.setStyleSheet(ACTIVE_CSS if active else "")

    def abort(self) -> None:
        if self._synchronous:
            self._work()
        else:
            threading.Thread(target=self._work, name="live_od_abort", daemon=True).start()

    def _work(self) -> None:
        # one click at a time: a second click waits for the first, then asks liveOD
        # again (liveOD itself answers a Reset pressed again on an unanswered abort)
        with self._lock:
            try:
                if self._client is None:
                    self._client = self._factory()
                text, ok = abort_run(self._client)
            except Exception as exc:                  # noqa: BLE001
                client, self._client = self._client, None
                if client is not None:
                    try:
                        client.close()
                    except Exception:                 # noqa: BLE001
                        pass
                text = (str(exc) if isinstance(exc, _Unanswered) else
                        f"Abort: liveOD not found ({exc}); nothing was sent.")
                ok = False
        self.result_ready.emit(text, ok)

    def _on_result(self, text: str, ok: bool) -> None:
        if self._say is not None:
            self._say(text)

    def shutdown(self) -> None:
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:                         # noqa: BLE001
                pass


__all__ = ["LiveODAbortButton", "abort_run", "reply_notice_text", "run_is_active"]
