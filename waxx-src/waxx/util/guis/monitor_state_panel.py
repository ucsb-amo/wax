"""The monitor server's state as one compact, collapsible bar: the top of the
monitor panel (:mod:`waxx.util.guis.monitor_panel`, over the network) and of
the monitor server's own window, above the run queue
(:mod:`waxx.util.guis.run_queue_panel`).

:class:`MonitorStatePanel` is fed like the queue panel -- :meth:`set_state`
with the server's ``status_json`` dict (None when it does not answer),
:meth:`on_broadcast` with its broadcasts -- and acts through an injected
``requester(request_dict) -> reply_dict`` (run off the GUI thread by a
:class:`~waxx.util.guis.request_runner.RequestRunner`).  It never discovers
the server or opens a socket.

The bar (always shown, one line; plain since the user's 2026-10-31 review --
nothing is shown that needs no attention):

* one sentence: a dot and :func:`~waxx.util.guis.device_summary.live_control_text`
  -- is live control (the monitor experiment) ready, and if not, which run has
  the machine (a queue job: the queue's line below names it).  A normal run is
  blue, never red;
* "Start live control" -- :attr:`MonitorStatePanel.monitor_button`, only while
  live control is off and no run is known to have the machine.  A click emits
  :attr:`MonitorStatePanel.monitor_clicked` when the host says it may
  (:meth:`set_monitor_clickable`), and the host (re)starts the monitor.  A host
  that drives its own button passes it as ``monitor_button``; the panel then
  places it and never writes to it;
* "the device tabs may not match the hardware" -- only while the state is
  untrusted and no run has the machine (a run's end state settles it);
* "Loops" -- a dropdown (collapsed at start; :meth:`set_sequences_expanded`)
  with a loop's name beside it while one is active or waiting for the queue
  (kexp: the BEC TOF loop and the Experiment loop).  Open, it shows the loops'
  cards under the bar: Start / Stop / category / scan settings and output
  (:class:`~waxx.util.guis.sequences_panel.SequencesPanel`, sending through the
  same requester; a person's Start carries ``owner`` "person").  This is the
  only GUI with them;
* "Start liveOD" (only when the host gives the launcher, ``live_od_bat``, and
  liveOD can be started) --
  :class:`~waxx.util.guis.live_od_launch.LiveODButton`: gone while a liveOD
  beacons, for 30 s after a launch, or when the host refuses this PC; each
  press is sent to the server's ops journal (``live_od_launch``);
* ⋯ -- Start / Restart live control, and the run queue's warm start (queue
  phase 2), from ``status_json``'s ``run_queue.warm_ahead``; a click sends
  ``run_queue`` / ``warm_ahead`` as a person, and the server stores it (it holds
  across its restarts).

The SLM reinit lease and the server's host-side connections (kexp: the
tweezer AWG) are not shown here (user, 2026-10-21): the Device Control GUI
shows both (its "Connected devices" group), and the journal below has their
records.

The details (the arrow at the left, or :meth:`set_expanded`):

* live control: sub-state, reason, since when, pid, its experiment file;
* the device tabs: why they may not match, since when;
* a run that announced itself and has not ended: which, from where, since
  when, and whether it is dropped by itself (only while the monitor is READY);
* the last ops-journal records, from the server's ``get_journal`` request
  (the records the server keeps in memory since it started; at most
  :data:`JOURNAL_LINES`; the client never names a file).

Requests: ``get_journal`` when the details open, every
:data:`JOURNAL_POLL_MS` while they are shown, and shortly after a broadcast;
the loops' own requests (start, stop, output); ``run_queue`` / ``warm_ahead`` on
a click of the warm start box;
``live_od_launch`` after a press of Start liveOD.  Nothing else -- the host
feeds ``status_json``.  Messages for the person go out on
:attr:`MonitorStatePanel.message` (the monitor panel shows them on the queue's
message line).

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QMenu, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget, QWidgetAction,
)

from waxx.util.dashboard import theme
from waxx.util.device_state.op_journal import describe_entry
from waxx.util.device_state.run_queue_client import default_by
from waxx.util.guis.device_summary import live_control_text, run_in_use
from waxx.util.guis.qt_upkeep import set_style_if_changed
from waxx.util.guis.request_runner import RequestRunner, is_unknown_request
from waxx.util.guis.run_queue_panel import (
    _PILL_BG, ERR_TEXT, OK_TEXT, WARN_TEXT, _clock, _dur,
)

#: Journal records asked for (the newest).
JOURNAL_LINES = 60
#: How often the journal is asked for while the details are shown.
JOURNAL_POLL_MS = 5000
#: Delay between a broadcast and the journal request it causes (coalesced).
JOURNAL_DEBOUNCE_MS = 1000
#: The "held N s" refresh of an announced run.
TICK_MS = 1000
#: The details' height cap (they scroll beyond it).
DETAILS_MAX_HEIGHT = 380

#: The Sequences area's height cap (it scrolls beyond it).
SEQUENCES_MAX_HEIGHT = 420

#: :func:`live_control_text`'s level -> the dot's colour
_DOT_COLOR = {"on": OK_TEXT, "run": "#8cc4ff", "partial": WARN_TEXT, "hazard": ERR_TEXT}
_LOOP_LEVEL = {"running": "on", "starting": "partial", "stopping": "partial",
               "latched": "hazard", "failed": "hazard"}
#: the summary chip shows the loop whose state ranks highest here
_LOOP_RANK = {"failed": 5, "latched": 5, "running": 4, "stopping": 3, "starting": 3}


def monitor_button_css(level: str) -> str:
    """The monitor pill as a (clickable) push button."""
    bg = _PILL_BG.get(level, "#454545")
    return (f"QPushButton {{ background: {bg}; color: white; border: 1px solid {bg};"
            f" border-radius: 9px; min-height: 18px; padding: 0px 12px; font-size: 11px;"
            f" font-weight: 700; }}"
            f" QPushButton:hover:enabled {{ border: 1px solid white; }}"
            f" QPushButton:disabled {{ color: rgba(255,255,255,0.80); }}")


def _line(color: str = theme.FG) -> QLabel:
    label = QLabel("")
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setStyleSheet(f"color: {color}; font-size: 11px;")
    return label


def _head(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {theme.FG_MUTED}; font-size: 11px; font-weight: 600;")
    label.setFixedWidth(78)
    label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    return label


def monitor_text(state: dict) -> str:
    """"running since 14:03:11 (for 2.0 min), pid 1234"."""
    sub = str(state.get("sub_state") or "").replace("_", " ")
    parts = [sub] if sub else []
    if state.get("reason"):
        parts.append(f"-- {state['reason']}")
    since = state.get("since")
    if since:
        parts.append(f"since {_clock(since, True)} (for {_dur(time.time() - float(since))})")
    if state.get("pid"):
        parts.append(f"pid {state['pid']}")
    return " ".join(parts)


def fence_text(pending: dict | None, monitor_state: str, now: float | None = None) -> str:
    """A run that announced itself and has not ended, in one line ("" for
    none).  The server cannot tell whether it has the machine yet: with the
    monitor already stopped, nothing shows when a run takes the core."""
    if not isinstance(pending, dict):
        return ""
    now = time.time() if now is None else now
    since = pending.get("since")
    held = f", {_dur(now - float(since))} ago" if since else ""
    who = f" from {pending['client']}" if pending.get("client") else ""
    ready = ("dropped by itself if the run never starts" if monitor_state == "READY" else
             "this stays until the run ends or someone unlocks the controls")
    return (f"Run {pending.get('run_id')} ({pending.get('expt') or 'experiment'}){who} started "
            f"at {_clock(since, True)}{held} and has not ended; controls are blocked -- {ready}.")


def slm_text(slm: dict) -> str:
    parts = [f"{slm.get('label') or 'SLM reinit'}: {slm.get('state') or '?'}"]
    if slm.get("detail"):
        parts.append(f"-- {slm['detail']}")
    if slm.get("blocked_by"):
        parts.append(f"; blocked by: {slm['blocked_by']}")
    if slm.get("reinit_due"):
        parts.append("; reinit due")
    elif slm.get("next_due_at"):
        parts.append(f"; next due {_clock(slm['next_due_at'])}")
    if slm.get("last_reinit"):
        last = slm["last_reinit"]
        at = last.get("at") if isinstance(last, dict) else last
        if isinstance(at, (int, float)):
            parts.append(f"; last reinit {_clock(at)}")
    return " ".join(parts)


def loop_text(key: str, info: dict) -> str:
    """"BEC TOF loop: running · holds" (the category only while it runs)."""
    state = str(info.get("state") or "?")
    category = str(info.get("category") or "") if state in ("running", "stopping") else ""
    return f"{info.get('title') or key}: {state}" + (f" · {category}" if category else "")


def loop_detail(info: dict) -> str:
    """A loop's tooltip lines: what it is doing, its category, its owner."""
    state = str(info.get("state") or "?")
    active = state in ("running", "stopping")
    category = str(info.get("category") or "") if active else ""
    tip = str(info.get("text") or "")
    if category == "holds":
        tip += "\nHolds: agents' queued jobs wait behind it; agents may not stop it."
    elif category == "yields":
        tip += ("\nYields: an agent's queued job stops it after its run in progress; "
                "the queue starts it again when it has run out.")
    if active and info.get("owner"):
        owner = str(info["owner"])
        if owner == "queue" and info.get("on_behalf_of"):
            owner += f" (for {info['on_behalf_of']})"
        tip += f"\nOwner: {owner}"
    return tip.strip()


def sequences_summary(loops: dict, resume: dict | None = None) -> tuple[str, str, str]:
    """``(text, level, tooltip)`` for the run loops: the one loop that is
    active (or failed) by name, else how many are, else the loop the queue
    stopped and starts again (``resume``, its ``resume_loop``), else ""."""
    if not loops:
        return "", "off", "The server runs no loops."
    infos = [(key, info or {}) for key, info in loops.items()]
    ranked = sorted(infos, key=lambda ki: -_LOOP_RANK.get(str(ki[1].get("state")), 0))
    active = [ki for ki in ranked if _LOOP_RANK.get(str(ki[1].get("state")), 0)]
    tip = "\n\n".join(loop_text(k, i) + (f"\n{loop_detail(i)}" if loop_detail(i) else "")
                      for k, i in infos)
    text, level = "", "off"
    if len(active) == 1:
        key, info = active[0]
        text = f"{info.get('title') or key} {info.get('state')}"
        level = _LOOP_LEVEL.get(str(info.get("state")), "off")
    elif active:
        text = f"{len(active)} active"
        level = _LOOP_LEVEL.get(str(active[0][1].get("state")), "off")
    elif isinstance(resume, dict):
        title = (loops.get(resume.get("key")) or {}).get("title") or resume.get("key")
        text = f"{title} waiting for the queue"
        tip += (f"\n\nThe queue stopped it for its jobs (since {_clock(resume.get('since'))}) "
                "and starts it again when it has run out.")
    return text, level, tip + "\n\nStart / Stop: open Loops (the arrow)."


class MonitorStatePanel(QWidget):
    """The monitor server's state bar (see the module docstring).

    ``requester``: ``request_dict -> reply_dict``; ``synchronous``: call it on
    the GUI thread (tests); ``runner``: an existing RequestRunner to share;
    ``by``: who is clicking (default ``user@host``); ``monitor_button``: a
    button the host drives itself, shown in place of the panel's own Start;
    ``expanded``: open the details at start.

    Start liveOD (shown only with ``live_od_bat``, the launcher's path):
    ``live_od_start(bat)``, ``live_od_probe()`` and ``live_od_refusal()`` --
    see :class:`~waxx.util.guis.live_od_launch.LiveODButton` (defaults: the
    real start and beacon probe; tests pass fakes)."""

    #: Start / Restart live control was clicked (the host (re)starts the monitor).
    monitor_clicked = pyqtSignal()
    #: A message for the person (warm start, liveOD start).
    message = pyqtSignal(str)

    def __init__(self, requester: Callable[[dict], Any] | None = None, *,
                 synchronous: bool = False, runner: RequestRunner | None = None,
                 by: str | None = None, parent=None,
                 monitor_button: QPushButton | None = None, expanded: bool = False,
                 live_od_bat: str | None = None, live_od_start=None, live_od_probe=None,
                 live_od_refusal=None):
        super().__init__(parent)
        from waxx.util.guis.sequences_panel import SequencesPanel  # noqa: PLC0415
        self.by = by or default_by()
        self._own_runner = runner is None
        self.runner = runner or RequestRunner(requester, synchronous=synchronous, parent=self)
        self.state: dict = {}
        self.reachable = False
        self.journal_supported = True
        self._journal_fetching = False
        self._monitor_clickable = False
        self._monitor_busy = False
        #: a warm start switch sent and not answered yet
        self._warm_busy = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # -- the bar ------------------------------------------------------------------
        self.bar = QFrame()
        self.bar.setObjectName("state_bar")
        self.bar.setStyleSheet(f"QFrame#state_bar {{ background: {theme.BG_CARD};"
                               f" border: 1px solid {theme.BORDER}; border-radius: 6px; }}")
        row = QHBoxLayout(self.bar)
        row.setContentsMargins(4, 3, 8, 3)
        row.setSpacing(6)
        self.toggle = QToolButton()
        self.toggle.setAutoRaise(True)
        self.toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggle.clicked.connect(lambda _=False: self.set_expanded(not self.expanded))
        row.addWidget(self.toggle)
        self.dot = QLabel("●")
        row.addWidget(self.dot)
        self.headline = QLabel("")
        self.headline.setStyleSheet(f"color: {theme.FG_STRONG}; font-size: 12px;"
                                    " font-weight: 600;")
        self.headline.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.headline)
        self._external_monitor = monitor_button is not None
        self.monitor_button = monitor_button or QPushButton("?")
        self.monitor_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.monitor_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        if not self._external_monitor:
            self.monitor_button.setStyleSheet(monitor_button_css("unknown"))
            self.monitor_button.clicked.connect(lambda _=False: self.monitor_clicked.emit())
        row.addWidget(self.monitor_button)
        self.trust_chip = QLabel("The device tabs may not match the hardware")
        self.trust_chip.setStyleSheet(f"color: {ERR_TEXT}; font-size: 11px;")
        self.trust_chip.hide()
        row.addWidget(self.trust_chip)
        row.addStretch(1)
        # the run loops behind one dropdown, their summary beside it
        self.sequences_chip = QLabel("")
        row.addWidget(self.sequences_chip)
        self.sequences_toggle = QToolButton()
        self.sequences_toggle.setText("Loops")
        self.sequences_toggle.setAutoRaise(True)
        self.sequences_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sequences_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.sequences_toggle.setStyleSheet(f"QToolButton {{ color: {theme.FG}; font-size: 11px;"
                                            " font-weight: 600; }")
        self.sequences_toggle.clicked.connect(
            lambda _=False: self.set_sequences_expanded(not self.sequences_expanded))
        row.addWidget(self.sequences_toggle)
        self.live_od_button = None
        if live_od_bat:
            from waxx.util.guis.live_od_launch import LiveODButton  # noqa: PLC0415
            self.live_od_button = LiveODButton(
                live_od_bat, start=live_od_start, probe=live_od_probe, refusal=live_od_refusal,
                record=self._journal_live_od_launch, say=self.message.emit, by=self.by)
            row.addWidget(self.live_od_button)
            self.live_od_button.refresh()             # shown only when it can start one
        row.addWidget(self._build_more_button())
        outer.addWidget(self.bar)

        # -- Loops (the run loops' cards; collapsed at start) ---------------------------
        self.sequences_area = QScrollArea()
        self.sequences_area.setWidgetResizable(True)
        self.sequences_area.setFrameShape(QFrame.Shape.NoFrame)
        self.sequences_area.setMaximumHeight(SEQUENCES_MAX_HEIGHT)
        # the loop cards; their requests go through this panel's runner (no
        # discovery, no socket)
        self.sequences = SequencesPanel(start_sender=False, show_hold=False, show_queue=False,
                                        runner=self.runner, by=self.by)
        self.sequences_area.setWidget(self.sequences)
        outer.addWidget(self.sequences_area)
        self.sequences_expanded = True
        self.set_sequences_expanded(False)

        # -- the details --------------------------------------------------------------
        self.details = QScrollArea()
        self.details.setWidgetResizable(True)
        self.details.setFrameShape(QFrame.Shape.NoFrame)
        self.details.setMaximumHeight(DETAILS_MAX_HEIGHT)
        page = QWidget()
        self.details.setWidget(page)
        box = QVBoxLayout(page)
        box.setContentsMargins(8, 6, 8, 6)
        box.setSpacing(4)

        heads: dict[str, QLabel] = {}

        def section(head: str) -> QVBoxLayout:
            line = QHBoxLayout()
            line.setSpacing(6)
            heads[head] = _head(head)
            line.addWidget(heads[head])
            col = QVBoxLayout()
            col.setSpacing(2)
            line.addLayout(col, 1)
            box.addLayout(line)
            return col

        sec = section("Live control")
        self.monitor_line = _line()
        sec.addWidget(self.monitor_line)
        self.expt_line = _line(theme.FG_MUTED)
        sec.addWidget(self.expt_line)

        sec = section("Device tabs")
        self.trust_banner = QLabel("")
        self.trust_banner.setWordWrap(True)
        sec.addWidget(self.trust_banner)
        self.trust_line = _line(theme.FG_MUTED)
        sec.addWidget(self.trust_line)

        sec = section("Run")
        self.fence_head = heads["Run"]
        self.fence_line = _line()
        sec.addWidget(self.fence_line)

        sec = section("Activity")
        head = QHBoxLayout()
        self.journal_note = _line(theme.FG_MUTED)
        head.addWidget(self.journal_note, 1)
        self.journal_button = QPushButton("Refresh")
        self.journal_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.journal_button.clicked.connect(lambda _=False: self.refresh_journal())
        head.addWidget(self.journal_button)
        sec.addLayout(head)
        self.journal = QPlainTextEdit()
        self.journal.setReadOnly(True)
        self.journal.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.journal.setFont(mono)
        self.journal.setMinimumHeight(140)
        self.journal.setPlaceholderText("No journal records yet (newest last).")
        sec.addWidget(self.journal)
        outer.addWidget(self.details)

        self._journal_timer = QTimer(self)
        self._journal_timer.timeout.connect(self._poll_journal)
        self._journal_timer.start(JOURNAL_POLL_MS)
        self._journal_soon = QTimer(self)
        self._journal_soon.setSingleShot(True)
        self._journal_soon.timeout.connect(self.refresh_journal)
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(TICK_MS)
        self.expanded = not expanded
        self.set_expanded(expanded)
        self._render()

    # -- the bar's controls -------------------------------------------------------------

    def _build_more_button(self) -> QToolButton:
        """⋯: Start / Restart live control, and the warm start box."""
        tool = QToolButton()
        tool.setText("⋯")
        tool.setAutoRaise(True)
        tool.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        tool.setToolTip("More")
        tool.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tool.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        menu = QMenu(tool)
        menu.setToolTipsVisible(True)
        self.monitor_action = menu.addAction("Start live control")
        self.monitor_action.triggered.connect(lambda _=False: self.monitor_clicked.emit())
        self.monitor_action.setVisible(not self._external_monitor)
        # warm start (the run queue's), a person's switch
        self.warm_box = QCheckBox("Prepare the next run early (warm start)")
        self.warm_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.warm_box.setStyleSheet(f"QCheckBox {{ color: {theme.FG}; padding: 4px 12px; }}")
        self.warm_box.clicked.connect(self.set_warm_ahead)
        action = QWidgetAction(menu)
        action.setDefaultWidget(self.warm_box)
        menu.addAction(action)
        tool.setMenu(menu)
        self.more_button = tool
        return tool

    def set_expanded(self, expanded: bool) -> None:
        """Show or hide the details (the journal is asked for on opening)."""
        expanded = bool(expanded)
        if expanded == self.expanded:
            return
        self.expanded = expanded
        self.details.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setToolTip("Hide the details" if expanded else
                               "Show the details and the recent activity")
        if expanded and not self._shut:
            self.refresh_journal()

    def set_sequences_expanded(self, expanded: bool) -> None:
        """Show or hide the run loops' cards (Loops)."""
        expanded = bool(expanded)
        if expanded == self.sequences_expanded:
            return
        self.sequences_expanded = expanded
        self.sequences_area.setVisible(expanded)
        self.sequences_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else
                                           Qt.ArrowType.RightArrow)
        self.sequences_toggle.setToolTip("Hide the loops" if expanded else
                                         "Show the loops: Start / Stop, their output")

    def set_monitor_clickable(self, clickable: bool) -> None:
        """Whether the panel's own Start / Restart does something (the host
        listens to :attr:`monitor_clicked`)."""
        self._monitor_clickable = bool(clickable)
        self._render_monitor()

    def set_monitor_busy(self, busy: bool) -> None:
        """A (re)start the host sent and has no answer to yet: no click."""
        self._monitor_busy = bool(busy)
        self._render_monitor()

    # -- inputs ---------------------------------------------------------------------------

    #: set by shutdown(): every input is then a no-op
    _shut = False

    def set_state(self, state: dict | None) -> None:
        """The server's ``status_json`` (None: it did not answer)."""
        if self._shut:
            return
        if not isinstance(state, dict):
            self.reachable = False
            self.sequences.set_reachable(False)
            self._render()
            return
        self.reachable = True
        self.state = dict(state)
        self.sequences.set_reachable(True)
        if isinstance(state.get("run_loops"), dict):
            self.sequences.set_loops(state["run_loops"])
        self._render()

    def on_broadcast(self, payload: dict) -> None:
        """A server broadcast: trust, the run fence, the loops and the queue's
        warm start update at once; any of them asks for the journal again
        shortly.  Ignored after shutdown."""
        if self._shut or not isinstance(payload, dict):
            return
        kind = payload.get("type")
        if kind == "trust":
            self.state["trust"] = payload.get("trust")
        elif kind == "run_pending":
            self.state["run_pending"] = payload.get("run_pending")
        elif kind == "run_queue" and isinstance(payload.get("run_queue"), dict):
            self.state["run_queue"] = payload["run_queue"]
        elif kind == "run_loop":
            loop = payload.get("loop")
            self.sequences.on_run_loop(loop)
            if isinstance(loop, dict) and loop.get("key"):
                loops = dict(self.state.get("run_loops") or {})
                loops[loop["key"]] = dict(loops.get(loop["key"]) or {}, **loop)
                self.state["run_loops"] = loops
        elif kind in ("state_update", "op_result"):
            pass                                      # frequent; the journal poll covers them
        self._render()
        if kind not in ("state_update",):
            self.journal_soon()

    # -- rendering -----------------------------------------------------------------------------

    def _render(self) -> None:
        self._render_monitor()
        self._render_trust()
        self._render_fence()
        self._render_loops()
        self._render_warm()

    def run_text(self) -> str:
        """The run the status shows using the machine, in words ("" for none)."""
        s = self.state
        return run_in_use(s.get("run_queue"), s.get("live_od"), s.get("run_pending"))

    def _render_monitor(self) -> None:
        s = self.state
        name = str(s.get("state_name") or "?") if self.reachable else ""
        run = self.run_text() if self.reachable else ""
        text, level = live_control_text(name, str(s.get("sub_state") or ""), run, self.reachable)
        if isinstance((s.get("run_queue") or {}).get("current"), dict):
            text = text.split(":", 1)[0]              # the queue's line below names the job
        self.headline.setText(text)
        set_style_if_changed(self.dot, f"color: {_DOT_COLOR[level]}; font-size: 12px;")
        if not self.reachable:
            self.monitor_line.setText("The monitor server is not answering.")
            self.expt_line.setText("")
        else:
            self.monitor_line.setText(monitor_text(s))
            self.expt_line.setText(f"monitor experiment: {s.get('expt_path') or '?'}")
        tip = ["Live control is the monitor experiment."]
        if self.reachable and monitor_text(s):
            tip.append(monitor_text(s))
        self.headline.setToolTip("\n".join(tip))
        if self._external_monitor:
            return                                    # the host drives its own button
        button, action = self.monitor_button, self.monitor_action
        if not self.reachable:
            button.setText("?")
            button.setEnabled(False)
            button.hide()
            action.setEnabled(False)
            button.setToolTip("The monitor server is not answering.")
            return
        can = self._monitor_clickable and name in ("READY", "NOT_READY") and not self._monitor_busy
        word = "Restart" if name == "READY" else "Start"
        button.setText(f"{word} live control")
        action.setText(f"{word} live control")
        button.setEnabled(can)
        action.setEnabled(can)
        button.setVisible(name == "NOT_READY" and not run)
        tip = []
        if self._monitor_busy:
            tip.append("A (re)start was sent; waiting for the server's answer.")
        elif can and name == "READY":
            tip.append("Asks first.")
        button.setToolTip("\n".join(tip))
        action.setToolTip("\n".join(tip))

    def _render_trust(self) -> None:
        trust = self.state.get("trust") if self.reachable else None
        chip = self.trust_chip
        if isinstance(trust, dict) and not trust.get("trusted", True):
            self.trust_banner.setText("The device tabs may not match the hardware.")
            self.trust_banner.setStyleSheet("QLabel { background: #c62828; color: white;"
                                            " border-radius: 6px; padding: 3px 8px;"
                                            " font-size: 11px; }")
            self.trust_banner.setToolTip(str(trust.get("reason") or ""))
            self.trust_line.setText(f"why: {trust.get('reason') or '?'} "
                                    f"(since {_clock(trust.get('since'), True)})")
            chip.setToolTip(f"why: {trust.get('reason') or '?'}")
            # a run's end state settles it, so not while a run has the machine
            chip.setVisible(not self.run_text())
        elif isinstance(trust, dict):
            self.trust_banner.setText("No known mismatch with the hardware.")
            self.trust_banner.setStyleSheet(f"QLabel {{ color: {OK_TEXT}; font-size: 11px; }}")
            self.trust_banner.setToolTip("")
            self.trust_line.setText(f"{trust.get('reason') or ''}"
                                    + (f" (since {_clock(trust.get('since'), True)})"
                                       if trust.get("since") else ""))
            chip.hide()
        else:
            self.trust_banner.setText("")
            self.trust_banner.setStyleSheet("")
            self.trust_line.setText("" if not self.reachable else "The server reports no trust "
                                                                  "flag.")
            chip.hide()

    def _render_fence(self) -> None:
        pending = self.state.get("run_pending") if self.reachable else None
        text = fence_text(pending, str(self.state.get("state_name") or "?"))
        self.fence_line.setText(text)
        set_style_if_changed(self.fence_line, f"color: {WARN_TEXT}; font-size: 11px;")
        self.fence_line.setVisible(bool(text))
        self.fence_head.setVisible(bool(text))

    def _render_loops(self) -> None:
        loops = self.state.get("run_loops") if self.reachable else None
        loops = loops if isinstance(loops, dict) else {}
        rq = self.state.get("run_queue") if self.reachable else None
        text, level, tip = sequences_summary(loops, rq.get("resume_loop")
                                             if isinstance(rq, dict) else None)
        chip = self.sequences_chip
        chip.setText(text)
        set_style_if_changed(chip, f"color: {ERR_TEXT if level == 'hazard' else theme.FG_MUTED};"
                                   " font-size: 11px;")
        chip.setToolTip(tip)
        chip.setVisible(self.reachable and bool(text))

    def _render_warm(self) -> None:
        rq = self.state.get("run_queue") if self.reachable else None
        on = rq.get("warm_ahead") if isinstance(rq, dict) else None
        box = self.warm_box
        box.setChecked(bool(on))
        if not isinstance(on, bool):
            box.setEnabled(False)
            box.setToolTip("The monitor server is not answering." if not self.reachable else
                           "This monitor server reports no warm start (older code).")
            return
        box.setEnabled(not self._warm_busy)
        source = str(rq.get("warm_ahead_source") or "")
        env = rq.get("warm_ahead_env")
        lines = [f"Warm start is {'ON' if on else 'OFF'}"
                 + {"stored": " (stored)", "default": " (the default: nothing stored)",
                    "env": f" (set by {env} where the server started)",
                    "given": " (given at the server's start)"}.get(source, ""),
                 "On: the queue starts the next job's process while the job before it runs "
                 "(it imports, prepares and compiles, then waits for its turn).",
                 "A click switches it for the queue and stores it: it holds across server "
                 "restarts" + (f", except that {env} overrides it at the server's start"
                               if env else "") + "."]
        if self._warm_busy:
            lines.append("A switch was sent; waiting for the server's answer.")
        box.setToolTip("\n".join(lines))

    # -- the bar's actions ------------------------------------------------------------------------

    def set_warm_ahead(self, on: bool) -> bool:
        """Warm start on / off at the server, as a person (the checkbox's
        click).  The box shows the server's answer; False when nothing was sent."""
        rq = self.state.get("run_queue") if self.reachable else None
        if (self._shut or self._warm_busy or not isinstance(rq, dict)
                or not isinstance(rq.get("warm_ahead"), bool)):
            self._render_warm()
            return False
        on = bool(on)
        self._warm_busy = True
        self._render_warm()
        self.warm_box.setChecked(on)

        def done(reply: dict) -> None:
            self._warm_busy = False
            if self._shut:
                return
            if reply.get("status") == "ok" and isinstance(reply.get("warm_ahead"), bool):
                rq_now = dict(self.state.get("run_queue") or {})
                rq_now["warm_ahead"] = reply["warm_ahead"]
                self.state["run_queue"] = rq_now
                now = "on" if reply["warm_ahead"] else "off"
                if reply.get("persisted") is False:
                    note = (f"; NOT stored ({reply.get('persist_error') or '?'}): until the "
                            "monitor server restarts")
                elif "persisted" not in reply:
                    note = " (this server does not store it: until it restarts)"
                elif reply.get("env_override"):
                    note = (f"; stored, but {reply['env_override']} overrides it at the "
                            "server's next start")
                else:
                    note = "; stored"
                self.message.emit(f"warm start {now}{note}")
            else:
                self.message.emit(f"warm start: {reply.get('msg') or 'no answer'}")
            self._render_warm()

        self.runner.send({"type": "run_queue", "action": "warm_ahead", "on": on,
                          "owner": "person", "by": self.by}, done)
        return True

    def _journal_live_od_launch(self, record: dict) -> None:
        """A press of Start liveOD, to the server's ops journal."""
        if self._shut:
            return

        def done(reply: dict) -> None:
            if reply.get("status") != "ok":
                self.message.emit("liveOD start NOT journalled: "
                                  f"{reply.get('msg') or 'the monitor server did not answer'}")

        self.runner.send(dict(record, type="live_od_launch"), done)

    def _on_tick(self) -> None:
        if self.isVisible() and self.reachable:
            self._render_fence()
            self.monitor_line.setText(monitor_text(self.state))

    # -- the journal ------------------------------------------------------------------------------

    def _poll_journal(self) -> None:
        if self.expanded and self.isVisible():
            self.refresh_journal()

    def journal_soon(self) -> None:
        if not self._shut and not self._journal_soon.isActive():
            self._journal_soon.start(JOURNAL_DEBOUNCE_MS)

    def journal_request(self) -> dict:
        return {"type": "get_journal", "n": JOURNAL_LINES}

    def refresh_journal(self) -> None:
        if self._shut or self._journal_fetching or not self.journal_supported:
            return
        self._journal_fetching = True
        self.runner.send(self.journal_request(), self._on_journal)

    def _on_journal(self, reply: dict) -> None:
        self._journal_fetching = False
        if reply.get("status") != "ok":
            if is_unknown_request(reply):
                self.journal_supported = False
                self.journal_note.setText("This monitor server serves no journal (older code).")
                self.journal_button.setEnabled(False)
            else:
                self.journal_note.setText(f"journal: {reply.get('msg')}")
            return
        entries = [e for e in (reply.get("entries") or []) if isinstance(e, dict)]
        lines = []
        for e in entries[-JOURNAL_LINES:]:
            try:
                lines.append(describe_entry(e))
            except Exception:                         # noqa: BLE001
                lines.append(str(e))
        text = "\n".join(lines)
        if text != self.journal.toPlainText():
            bar = self.journal.verticalScrollBar()
            at_end = bar.value() >= bar.maximum() - 2
            self.journal.setPlainText(text)
            if at_end:
                bar.setValue(bar.maximum())
        self.journal_note.setText(f"last {len(lines)} records the server keeps in memory, "
                                  "newest last"
                                  + (f"; on disk: {reply['path']}" if reply.get("path") else ""))

    # -- shutdown -------------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the timers; from now on state, broadcasts and the journal are
        ignored (nothing is sent)."""
        self._shut = True
        self._journal_timer.stop()
        self._journal_soon.stop()
        self._tick.stop()
        self.sequences.shutdown()                     # leaves the shared runner alone
        if self.live_od_button is not None:
            self.live_od_button.shutdown()
        if self._own_runner:
            self.runner.shutdown()


__all__ = ["MonitorStatePanel", "fence_text", "loop_detail", "loop_text", "monitor_button_css",
           "monitor_text", "sequences_summary", "slm_text"]
