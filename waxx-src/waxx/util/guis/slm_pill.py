"""The SLM pill in the Device Control GUI's status row.

It shows the SLM's hourly reinit as the monitor server reports it
(``status_json`` and the ``slm_reinit`` broadcast; see
:mod:`waxx.util.device_state.slm_reinit`), and its menu (right or left click)
has:

* when the next reinit falls due, and how the last one went;
* **Re-initialise SLM now** -- the monitor server sends it only while the
  machine is idle (monitor running, no run starting, no reset or run loop),
  so the item is off, with the reason, while it is not;
* **Launch spot finder** -- when the host GUI was given a launcher (the lab's
  SLM spot finder, started on this PC).

The pill never talks to the network: it emits ``reinit_requested`` /
``spot_finder_requested`` and the host GUI does the rest.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QMenu, QPushButton

from waxx.util.dashboard import theme

#: service state -> (pill colour, word); colours as the connection bar's pills
LOOK = {
    "idle": ("#43a047", "ready"),
    "due": ("#ef6c00", "reinit due"),
    "reinitialising": ("#ba68c8", "re-initialising"),
    "failed": ("#c62828", "last reinit failed"),
    "unreachable": ("#c62828", "SLM server not answering"),
    "no_control": ("#9e9e9e", "SLM server from before 2026-09-28 (reinits by itself)"),
    "unknown": ("#9e9e9e", "not asked yet"),
}
_NOT_REPORTED = ("#9e9e9e", "not reported by the monitor server")

#: states in which "Re-initialise SLM now" may be sent
REINIT_STATES = ("idle", "due", "failed", "unreachable")

_SUFFIX = {"due": " · due", "reinitialising": " · reinit…", "failed": " · failed",
           "unreachable": " · ?"}


def _css(color: str) -> str:
    return (f"QPushButton {{ background-color: {color}; color: white; font-weight: bold; "
            f"border: none; border-radius: 8px; padding: 1px 8px; }} "
            f"QPushButton:disabled {{ color: rgba(255, 255, 255, 140); }}")


def _hhmm(epoch, seconds: bool = False) -> str:
    try:
        return time.strftime("%H:%M:%S" if seconds else "%H:%M", time.localtime(float(epoch)))
    except (TypeError, ValueError, OverflowError):
        return "?"


def _span(seconds: float) -> str:
    seconds = max(float(seconds), 0.)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def describe_next(snap: dict | None, now: float | None = None) -> str:
    """When the next reinit happens, for people ("" when nobody knows)."""
    if not isinstance(snap, dict):
        return ""
    now = time.time() if now is None else now
    state = snap.get("state")
    if state == "reinitialising":
        return "Re-initialising now"
    if snap.get("reinit_due"):
        due_for = snap.get("due_for_s")
        since = (f" since {_hhmm(now - due_for)}"
                 if isinstance(due_for, (int, float)) else "")
        why = snap.get("blocked_by") or ""
        return (f"Reinit due{since}: sent once the machine is idle"
                + (f" (now: {why})" if why else ""))
    at = snap.get("next_due_at")
    if isinstance(at, (int, float)):
        return f"Next reinit due {_hhmm(at)} (in {_span(at - now)})"
    return ""


def describe_last(snap: dict | None) -> str:
    last = (snap or {}).get("last_reinit")
    if not isinstance(last, dict):
        return ""
    text = f"Last reinit {_hhmm(last.get('at'), seconds=True)}"
    if isinstance(last.get("t_s"), (int, float)):
        text += f" ({last['t_s']:.1f} s)"
    return text + ", pattern put back"


class SlmPill(QPushButton):
    """See the module docstring."""

    reinit_requested = pyqtSignal()
    spot_finder_requested = pyqtSignal()

    def __init__(self, can_launch: bool = False, parent=None):
        super().__init__("SLM", parent)
        self.can_launch = bool(can_launch)
        self.snapshot: dict | None = None
        self.reported = False
        self.reachable = False
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: self._show_menu(pos))
        self.clicked.connect(lambda _=False: self._show_menu(self.rect().bottomLeft()))
        self.refresh()

    # -- inputs ---------------------------------------------------------------------

    def set_snapshot(self, snap: dict | None, reported: bool = True) -> None:
        """The service's snapshot; ``reported`` False: the monitor server's
        status carries no ``slm_reinit`` (older code) or says None (not
        configured)."""
        self.snapshot = dict(snap) if isinstance(snap, dict) else None
        self.reported = bool(reported) and self.snapshot is not None
        self.reachable = True
        self.refresh()

    def set_reachable(self, reachable: bool) -> None:
        if bool(reachable) != self.reachable:
            self.reachable = bool(reachable)
            self.refresh()

    # -- what it shows --------------------------------------------------------------

    @property
    def state(self) -> str:
        return str((self.snapshot or {}).get("state") or "unknown")

    def reinit_allowed(self) -> tuple[bool, str]:
        """Whether "Re-initialise SLM now" may be sent (the server checks again)."""
        if not self.reachable:
            return False, "the monitor server is unreachable"
        if not self.reported:
            return False, ("the monitor server does not report the SLM reinit (older code, "
                           "or no SLM configured)")
        snap = self.snapshot or {}
        if snap.get("manual_pending"):
            return False, f"a reinit asked for by {snap['manual_pending']} is being sent"
        if self.state not in REINIT_STATES:
            return False, LOOK.get(self.state, _NOT_REPORTED)[1]
        if snap.get("blocked_by"):
            return False, f"the machine is not idle: {snap['blocked_by']}"
        return True, ""

    def refresh(self) -> None:
        if not self.reachable:
            color, word = LOOK["unknown"][0], "monitor server unreachable"
        elif not self.reported:
            color, word = _NOT_REPORTED
        else:
            color, word = LOOK.get(self.state, LOOK["unknown"])
        suffix = _SUFFIX.get(self.state, "") if (self.reachable and self.reported) else ""
        self.setText("SLM" + suffix)
        self.setStyleSheet(_css(color))
        self.setToolTip("\n".join(self.tooltip_lines(word)))
        want = self.reported or self.can_launch
        # Never show() a parentless pill: it would flash up as its own window
        # (the host adds it to its status row after building it).
        if self.parentWidget() is not None:
            self.setVisible(want)
        elif not want:
            self.hide()

    def tooltip_lines(self, word: str | None = None) -> list[str]:
        snap = self.snapshot or {}
        if word is None:
            word = LOOK.get(self.state, _NOT_REPORTED)[1]
        detail = str(snap.get("detail") or "") if self.reported else ""
        lines = [f"SLM hourly reinit: {word}." + (f" {detail}" if detail else "")]
        for text in (describe_next(snap) if self.reported else "",
                     describe_last(snap) if self.reported else ""):
            if text:
                lines.append(text)
        manual = snap.get("last_manual") if self.reported else None
        if isinstance(manual, dict):
            lines.append(f"Asked for by {manual.get('by')} at {_hhmm(manual.get('at'))}: "
                         f"{manual.get('result')}")
        lines.append("Right-click: reinit now" + (", launch the spot finder"
                                                   if self.can_launch else "") + ".")
        return lines

    # -- the menu -------------------------------------------------------------------

    def build_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.setStyleSheet(f"QMenu {{ background: {theme.BG_RAISED}; color: {theme.FG}; }}"
                           f"QMenu::item:disabled {{ color: {theme.FG_MUTED}; }}")
        info = [describe_next(self.snapshot) if self.reported else "",
                describe_last(self.snapshot) if self.reported else ""]
        if not self.reported:
            info = ["SLM reinit: " + (LOOK["unknown"][1] if not self.reachable
                                       else _NOT_REPORTED[1])]
        for text in info:
            if text:
                menu.addAction(text).setEnabled(False)
        menu.addSeparator()
        allowed, why = self.reinit_allowed()
        reinit = menu.addAction("Re-initialise SLM now…")
        reinit.setObjectName("slm_reinit_now")
        reinit.setEnabled(allowed)
        reinit.triggered.connect(self.reinit_requested.emit)
        if not allowed:
            menu.addAction(f"    not now: {why}").setEnabled(False)
        if self.can_launch:
            menu.addSeparator()
            launch = menu.addAction("Launch spot finder")
            launch.setObjectName("slm_launch_spot_finder")
            launch.triggered.connect(self.spot_finder_requested.emit)
        return menu

    def _show_menu(self, pos) -> None:
        self.build_menu().exec(self.mapToGlobal(pos))
