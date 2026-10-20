"""The person hold in liveOD: an amber button, shown only while a person hold
is on (:mod:`waxx.util.device_state.person_hold`), that releases it.

The hold lives in the monitor server.  liveOD's Abort is one of the things
that puts it on (the monitor server sees the Reset counted in liveOD's POLL),
so the place a person presses Abort is also where they see the hold and end
it.  The button wears the colour of the HELD banner in the Monitor panel's
queue (``theme.HELD``).

:class:`HoldBadge` asks the monitor server for its ``status_json`` every
:attr:`HoldBadge.POLL_MS` on a worker thread (a :class:`~waxx.util.guis.
monitor_panel.MonitorLink`: discovery of this machine's ``monitor:<hw>``,
the client dropped and found again on no answer) and reads ``person_hold``.
The monitor server's UDP state broadcast (``person_hold``) only makes it ask
again at once: that broadcast is not scoped to a machine, the poll is.

Nothing is discovered or bound until :meth:`HoldBadge.start` (the windows'
``main()`` calls it; tests that build a window never do).  Release sends the
run queue's ``release`` (owner ``person``, by ``liveOD on <host>``) on one
click, no question (as the queue panel's button).  With no monitor server, or one without a hold, the badge is hidden.
"""

from __future__ import annotations

import socket
from typing import Any, Callable

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QPushButton

from waxx.util.dashboard import theme as dash_theme
from waxx.util.device_state.person_hold import describe as describe_hold

#: shown on the button while the hold is on
BADGE_TEXT = "HELD · Release"


def release_request(by: str) -> dict:
    return {"type": "run_queue", "action": "release", "owner": "person", "by": by}


class HoldBadge(QPushButton):
    """See the module docstring.  ``link``: a MonitorLink-like object with
    ``status() -> dict | None`` and ``request(dict) -> dict | None`` (default:
    a MonitorLink, made at :meth:`start`); ``synchronous``: requests on the
    GUI thread (tests); ``log(text, level)``: where the badge reports what it
    did (the window's log)."""

    #: How often the monitor server's status is asked for.
    POLL_MS = 3000

    hold_changed = pyqtSignal(object)       # the person_hold dict ({} when none)

    def __init__(self, parent=None, *, link=None, synchronous: bool = False,
                 log: Callable[[str, int], Any] | None = None, by: str | None = None):
        super().__init__(BADGE_TEXT, parent)
        self.link = link
        self._synchronous = bool(synchronous)
        self._log = log
        self.by = by or f"liveOD on {socket.gethostname()}"
        self.info: dict = {}
        self.runner = None
        self.listener = None
        self._polling = False
        self._closed = False
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedWidth(118)       # appearing must not push the window wider than this
        self.setStyleSheet(dash_theme.held_button_css())
        self.clicked.connect(lambda _=False: self.release())
        self.timer = QTimer(self)
        self.timer.setInterval(self.POLL_MS)
        self.timer.timeout.connect(self.poll)
        self.hide()

    # -- the link ------------------------------------------------------------------------

    def start(self, listen: bool = True) -> None:
        """Start asking (and, with ``listen``, follow the monitor server's
        broadcast).  Imports the comms package only here: importing it starts
        the discovery listener."""
        if self._closed:
            return
        from waxx.util.guis.request_runner import RequestRunner  # noqa: PLC0415
        if self.link is None:
            from waxx.util.guis.monitor_panel import MonitorLink  # noqa: PLC0415
            self.link = MonitorLink(discovery_timeout=2.0)
        if self.runner is None:
            self.runner = RequestRunner(self.link.request, synchronous=self._synchronous,
                                        parent=self)
        if listen and self.listener is None:
            try:
                from waxx.util.comms_server.state_broadcast import StateListener  # noqa: PLC0415
                self.listener = StateListener(parent=self)
                self.listener.state_received.connect(self.on_broadcast)
                self.listener.start()
            except Exception as exc:                  # noqa: BLE001
                self.listener = None
                self._say(f"Hold badge: no monitor broadcast ({exc}); asking every "
                          f"{self.POLL_MS / 1000:.0f} s", 30)
        self.timer.start()
        self.poll()

    def poll(self) -> None:
        if self._polling or self._closed or self.runner is None:
            return
        self._polling = True
        self.runner.call(self.link.status, self._on_status)

    def _on_status(self, status) -> None:
        self._polling = False
        if self._closed:
            return
        if not isinstance(status, dict):
            return          # no answer: keep what was last known (the button says when)
        hold = status.get("person_hold")
        self.set_hold(hold if isinstance(hold, dict) else {})

    def on_broadcast(self, payload) -> None:
        """The monitor server's UDP broadcast: a ``person_hold`` one makes the
        badge ask now (the broadcast names no machine; the poll does)."""
        if isinstance(payload, dict) and payload.get("type") == "person_hold":
            self.poll()

    # -- showing it -------------------------------------------------------------------------

    def held(self) -> bool:
        return bool(self.info.get("active"))

    def set_hold(self, info: dict | None) -> None:
        before = self.held()
        self.info = dict(info) if isinstance(info, dict) else {}
        on = self.held()
        self.setVisible(on)
        if on:
            self.setToolTip(describe_hold(self.info)[0].upper() + describe_hold(self.info)[1:]
                            + "\n\nAgents' queued runs and the run loops wait until a person "
                            "releases it. liveOD's Abort puts it on.\n\nClick to release it (no question asked).")
        if on != before:
            self._say(("Person hold ON: " + describe_hold(self.info)) if on else
                      "Person hold released.", 30 if on else 20)
            self.hold_changed.emit(dict(self.info))

    # -- releasing it -----------------------------------------------------------------------

    def release(self) -> bool:
        """One click releases it, no question (as the queue panel's button)."""
        if not self.held() or self.runner is None:
            return False
        self.setEnabled(False)

        def done(reply) -> None:
            self.setEnabled(True)
            if self._closed:
                return
            if isinstance(reply, dict) and reply.get("status") == "ok":
                hold = reply.get("person_hold")
                self.set_hold(hold if isinstance(hold, dict) else {"active": False})
            else:
                why = (reply or {}).get("msg") if isinstance(reply, dict) else None
                self._say(f"Release hold failed: {why or 'the monitor server did not answer'}",
                          40)
                self.poll()
        self.runner.send(release_request(self.by), done)
        return True

    # -- the rest ---------------------------------------------------------------------------

    def _say(self, text: str, level: int = 20) -> None:
        if self._log is not None:
            try:
                self._log(text, level)
                return
            except Exception:                         # noqa: BLE001
                pass
        print(f"[liveOD] {text}")

    def shutdown(self) -> None:
        """Stop polling and listening (safe to call twice)."""
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        if self.listener is not None:
            try:
                self.listener.stop()
                self.listener.wait(1500)
            except Exception:                         # noqa: BLE001
                pass
            self.listener = None
        if self.runner is not None:
            try:
                self.runner.shutdown()
            except Exception:                         # noqa: BLE001
                pass


__all__ = ["BADGE_TEXT", "HoldBadge", "release_request"]
