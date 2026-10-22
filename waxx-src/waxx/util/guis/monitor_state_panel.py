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

The bar (always shown, one line):

* the monitor experiment's pill -- READY / LOADING / NOT READY.  Clickable
  when the host says so (:meth:`set_monitor_clickable`): a click emits
  :attr:`MonitorStatePanel.monitor_clicked`, and the host (re)starts the
  monitor.  A host that drives its own button passes it as
  ``monitor_button``; the panel then places it and never writes to it;
* chips: device-state trust (red while UNTRUSTED), the run fence (only while
  a run holds it).  Every chip's tooltip has the whole story;
* "Sequences" -- a dropdown (collapsed at start; :meth:`set_sequences_expanded`)
  with a one-chip summary of the run loops beside it (kexp: the BEC TOF loop
  and the Experiment loop).  Open, it shows the loops' cards under the bar,
  Start / Stop / category / scan settings and output as on the Device Control
  GUI's Sequences tab (:class:`~waxx.util.guis.sequences_panel.SequencesPanel`,
  sending through the same requester; a person's Start carries ``owner``
  "person");
* "Warm start" -- the run queue's warm start (queue phase 2), from
  ``status_json``'s ``run_queue.warm_ahead``; a click sends ``run_queue`` /
  ``warm_ahead`` as a person, and the server stores it (it holds across its
  restarts);
* "Start liveOD" (only when the host gives the launcher, ``live_od_bat``) --
  :class:`~waxx.util.guis.live_od_launch.LiveODButton`: off while a liveOD
  beacons, for 30 s after a launch, or when the host refuses this PC; each
  press is sent to the server's ops journal (``live_od_launch``).

The SLM reinit lease and the server's host-side connections (kexp: the
tweezer AWG) are not shown here (user, 2026-10-21): the Device Control GUI
shows both (its SLM pill and the Composite tab's connection bar), and the
journal below has their records.

The details (the arrow at the left, or :meth:`set_expanded`):

* the monitor: sub-state, reason, since when, pid, its experiment file;
* trust: why, since when;
* the run fence: which run, from where, since when, for how long, and
  whether it lapses (only while the monitor is READY);
* the last ops-journal records, from the server's ``get_journal`` request
  (the records the server keeps in memory since it started; at most
  :data:`JOURNAL_LINES`; the client never names a file).

Requests: ``get_journal`` when the details open, every
:data:`JOURNAL_POLL_MS` while they are shown, and shortly after a broadcast;
the loops' own requests (start, stop, output) as the Sequences tab sends
them; ``run_queue`` / ``warm_ahead`` on a click of Warm start;
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
    QCheckBox, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.device_state.op_journal import describe_entry
from waxx.util.device_state.run_queue_client import default_by
from waxx.util.guis.qt_upkeep import set_style_if_changed
from waxx.util.guis.request_runner import RequestRunner, is_unknown_request
from waxx.util.guis.run_queue_panel import (
    _PILL_BG, ERR_TEXT, OK_TEXT, WARN_TEXT, _clock, _dur, chip_css,
)

#: Journal records asked for (the newest).
JOURNAL_LINES = 60
#: How often the journal is asked for while the details are shown.
JOURNAL_POLL_MS = 5000
#: Delay between a broadcast and the journal request it causes (coalesced).
JOURNAL_DEBOUNCE_MS = 1000
#: The fence's "held N s" refresh.
TICK_MS = 1000
#: The details' height cap (they scroll beyond it).
DETAILS_MAX_HEIGHT = 380

#: The Sequences area's height cap (it scrolls beyond it).
SEQUENCES_MAX_HEIGHT = 420

_MONITOR_LEVEL = {"READY": "on", "LOADING": "partial", "NOT_READY": "hazard"}
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


def _chip(text: str = "") -> QLabel:
    chip = QLabel(text)
    chip.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
    return chip


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
    """The run fence in one line ("No run fence." when there is none)."""
    if not isinstance(pending, dict):
        return "No run fence: composite ops and the run queue are not held back by a run."
    now = time.time() if now is None else now
    since = pending.get("since")
    held = f", held {_dur(now - float(since))}" if since else ""
    who = f" from {pending['client']}" if pending.get("client") else ""
    ready = ("the monitor is READY, so it lapses if the run never takes the core"
             if monitor_state == "READY" else
             f"the monitor is {monitor_state}, so it does not lapse by itself")
    return (f"Run fence: run {pending.get('run_id')} ({pending.get('expt') or 'experiment'}){who}"
            f" since {_clock(since, True)}{held} -- {ready}.")


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


def sequences_summary(loops: dict) -> tuple[str, str, str]:
    """``(chip text, level, tooltip)`` for the run loops: the one loop that is
    active (or failed) by name, else how many are, else "idle"."""
    if not loops:
        return "", "off", "The server runs no loops."
    infos = [(key, info or {}) for key, info in loops.items()]
    ranked = sorted(infos, key=lambda ki: -_LOOP_RANK.get(str(ki[1].get("state")), 0))
    active = [ki for ki in ranked if _LOOP_RANK.get(str(ki[1].get("state")), 0)]
    if not active:
        text, level = "idle", "off"
    elif len(active) == 1:
        key, info = active[0]
        text, level = loop_text(key, info), _LOOP_LEVEL.get(str(info.get("state")), "off")
    else:
        text = f"{len(active)} active"
        level = _LOOP_LEVEL.get(str(active[0][1].get("state")), "off")
    tip = "\n\n".join(loop_text(k, i) + (f"\n{loop_detail(i)}" if loop_detail(i) else "")
                      for k, i in infos)
    return text, level, tip + "\n\nStart / Stop: open Sequences (the arrow)."


class MonitorStatePanel(QWidget):
    """The monitor server's state bar (see the module docstring).

    ``requester``: ``request_dict -> reply_dict``; ``synchronous``: call it on
    the GUI thread (tests); ``runner``: an existing RequestRunner to share;
    ``by``: who is clicking (default ``user@host``); ``monitor_button``: a
    button the host drives itself, shown in place of the panel's own monitor
    pill; ``expanded``: open the details at start.

    Start liveOD (shown only with ``live_od_bat``, the launcher's path):
    ``live_od_start(bat)``, ``live_od_probe()`` and ``live_od_refusal()`` --
    see :class:`~waxx.util.guis.live_od_launch.LiveODButton` (defaults: the
    real start and beacon probe; tests pass fakes)."""

    #: The monitor pill was clicked (the host starts / restarts the monitor).
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
        title = QLabel("Monitor")
        title.setStyleSheet(f"color: {theme.FG_MUTED}; font-size: 11px; font-weight: 600;")
        row.addWidget(title)
        self._external_monitor = monitor_button is not None
        self.monitor_pill = monitor_button or QPushButton("?")
        self.monitor_pill.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.monitor_pill.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        if not self._external_monitor:
            self.monitor_pill.clicked.connect(lambda _=False: self.monitor_clicked.emit())
        row.addWidget(self.monitor_pill)
        self.trust_chip = _chip()
        row.addWidget(self.trust_chip)
        self.fence_chip = _chip()
        row.addWidget(self.fence_chip)
        # Sequences: the run loops behind one dropdown, a summary chip beside it
        self.sequences_toggle = QToolButton()
        self.sequences_toggle.setText("Sequences")
        self.sequences_toggle.setAutoRaise(True)
        self.sequences_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sequences_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.sequences_toggle.setStyleSheet(f"QToolButton {{ color: {theme.FG}; font-size: 11px;"
                                            " font-weight: 600; }")
        self.sequences_toggle.clicked.connect(
            lambda _=False: self.set_sequences_expanded(not self.sequences_expanded))
        row.addWidget(self.sequences_toggle)
        self.sequences_chip = _chip()
        row.addWidget(self.sequences_chip)
        # warm start (the run queue's), a person's switch
        self.warm_box = QCheckBox("Warm start")
        self.warm_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.warm_box.setStyleSheet(f"QCheckBox {{ color: {theme.FG}; font-size: 11px; }}")
        self.warm_box.clicked.connect(self.set_warm_ahead)
        row.addWidget(self.warm_box)
        row.addStretch(1)
        self.live_od_button = None
        if live_od_bat:
            from waxx.util.guis.live_od_launch import LiveODButton  # noqa: PLC0415
            self.live_od_button = LiveODButton(
                live_od_bat, start=live_od_start, probe=live_od_probe, refusal=live_od_refusal,
                record=self._journal_live_od_launch, say=self.message.emit, by=self.by)
            row.addWidget(self.live_od_button)
        outer.addWidget(self.bar)

        # -- Sequences (the run loops' cards; collapsed at start) ----------------------
        self.sequences_area = QScrollArea()
        self.sequences_area.setWidgetResizable(True)
        self.sequences_area.setFrameShape(QFrame.Shape.NoFrame)
        self.sequences_area.setMaximumHeight(SEQUENCES_MAX_HEIGHT)
        # the Sequences tab's loop cards; their requests go through this
        # panel's runner (no discovery, no socket)
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

        def section(head: str) -> QVBoxLayout:
            line = QHBoxLayout()
            line.setSpacing(6)
            line.addWidget(_head(head))
            col = QVBoxLayout()
            col.setSpacing(2)
            line.addLayout(col, 1)
            box.addLayout(line)
            return col

        sec = section("Monitor")
        self.monitor_line = _line()
        sec.addWidget(self.monitor_line)
        self.expt_line = _line(theme.FG_MUTED)
        sec.addWidget(self.expt_line)

        sec = section("Trust")
        self.trust_banner = QLabel("")
        self.trust_banner.setWordWrap(True)
        sec.addWidget(self.trust_banner)
        self.trust_line = _line(theme.FG_MUTED)
        sec.addWidget(self.trust_line)

        sec = section("Run fence")
        self.fence_line = _line()
        sec.addWidget(self.fence_line)

        sec = section("Journal")
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

    def set_expanded(self, expanded: bool) -> None:
        """Show or hide the details (the journal is asked for on opening)."""
        expanded = bool(expanded)
        if expanded == self.expanded:
            return
        self.expanded = expanded
        self.details.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setToolTip("Hide the details" if expanded else
                               "Show the details: monitor, trust, fence, ops journal")
        if expanded and not self._shut:
            self.refresh_journal()

    def set_sequences_expanded(self, expanded: bool) -> None:
        """Show or hide the run loops' cards (Sequences)."""
        expanded = bool(expanded)
        if expanded == self.sequences_expanded:
            return
        self.sequences_expanded = expanded
        self.sequences_area.setVisible(expanded)
        self.sequences_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else
                                           Qt.ArrowType.RightArrow)
        self.sequences_toggle.setToolTip("Hide the sequences" if expanded else
                                         "Show the sequences: the run loops, Start / Stop, "
                                         "their output")

    def set_monitor_clickable(self, clickable: bool) -> None:
        """Whether a click on the panel's own monitor pill does something (the
        host listens to :attr:`monitor_clicked`)."""
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

    def _render_monitor(self) -> None:
        s = self.state
        name = str(s.get("state_name") or "?") if self.reachable else ""
        if not self.reachable:
            self.monitor_line.setText("The monitor server is not answering.")
            self.expt_line.setText("")
        else:
            self.monitor_line.setText(monitor_text(s))
            self.expt_line.setText(f"monitor experiment: {s.get('expt_path') or '?'}")
        if self._external_monitor:
            return                                    # the host drives its own button
        if not self.reachable:
            self.monitor_pill.setText("?")
            set_style_if_changed(self.monitor_pill, monitor_button_css("unknown"))
            self.monitor_pill.setEnabled(False)
            self.monitor_pill.setToolTip("The monitor server is not answering.")
            return
        self.monitor_pill.setText(name.replace("_", " "))
        set_style_if_changed(self.monitor_pill,
                             monitor_button_css(_MONITOR_LEVEL.get(name, "unknown")))
        can = self._monitor_clickable and name in ("READY", "NOT_READY") and not self._monitor_busy
        self.monitor_pill.setEnabled(can)
        tip = [f"Monitor experiment: {name.replace('_', ' ')}"]
        if monitor_text(s):
            tip.append(monitor_text(s))
        if self._monitor_busy:
            tip.append("A (re)start was sent; waiting for the server's answer.")
        elif can:
            tip.append("Click: restart it (asks first)" if name == "READY" else
                       "Click: start it")
        self.monitor_pill.setToolTip("\n".join(tip))

    def _render_trust(self) -> None:
        trust = self.state.get("trust") if self.reachable else None
        chip = self.trust_chip
        if isinstance(trust, dict) and not trust.get("trusted", True):
            self.trust_banner.setText("Device state UNTRUSTED: the device tabs may not match "
                                      "the hardware.")
            self.trust_banner.setStyleSheet("QLabel { background: #c62828; color: white;"
                                            " border-radius: 6px; padding: 3px 8px;"
                                            " font-size: 11px; }")
            self.trust_banner.setToolTip(str(trust.get("reason") or ""))
            self.trust_line.setText(f"why: {trust.get('reason') or '?'} "
                                    f"(since {_clock(trust.get('since'), True)})")
            chip.setText("state UNTRUSTED")
            set_style_if_changed(chip, chip_css("hazard"))
            chip.setToolTip("Device state UNTRUSTED: the device tabs may not match the "
                            f"hardware.\nwhy: {trust.get('reason') or '?'}")
            chip.show()
        elif isinstance(trust, dict):
            self.trust_banner.setText("Device state trusted.")
            self.trust_banner.setStyleSheet(f"QLabel {{ color: {OK_TEXT}; font-size: 11px; }}")
            self.trust_banner.setToolTip("")
            self.trust_line.setText(f"{trust.get('reason') or ''}"
                                    + (f" (since {_clock(trust.get('since'), True)})"
                                       if trust.get("since") else ""))
            chip.setText("state trusted")
            set_style_if_changed(chip, chip_css("off"))
            chip.setToolTip("Device state trusted." + (f"\n{trust['reason']}"
                                                       if trust.get("reason") else ""))
            chip.show()
        else:
            self.trust_banner.setText("")
            self.trust_banner.setStyleSheet("")
            self.trust_line.setText("" if not self.reachable else "The server reports no trust "
                                                                  "flag.")
            chip.hide()

    def _render_fence(self) -> None:
        if not self.reachable:
            self.fence_line.setText("")
            self.fence_chip.hide()
            return
        pending = self.state.get("run_pending")
        text = fence_text(pending, str(self.state.get("state_name") or "?"))
        self.fence_line.setText(text)
        set_style_if_changed(self.fence_line, f"color: {WARN_TEXT if pending else theme.FG_MUTED};"
                                              " font-size: 11px;")
        if isinstance(pending, dict):
            self.fence_chip.setText(f"fence: run {pending.get('run_id')}")
            set_style_if_changed(self.fence_chip, chip_css("partial"))
            self.fence_chip.setToolTip(text)
            self.fence_chip.show()
        else:
            self.fence_chip.hide()

    def _render_loops(self) -> None:
        loops = self.state.get("run_loops") if self.reachable else None
        loops = loops if isinstance(loops, dict) else {}
        text, level, tip = sequences_summary(loops)
        chip = self.sequences_chip
        if not self.reachable or not loops:
            chip.hide()
        else:
            chip.setText(text)
            set_style_if_changed(chip, chip_css(level))
            chip.setToolTip(tip)
            chip.show()

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
