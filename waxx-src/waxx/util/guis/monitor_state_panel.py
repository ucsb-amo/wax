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
  a run holds it), each connection the server holds, the SLM's reinit lease,
  each run loop.  Every chip's tooltip has the whole story.

The details (the arrow at the left, or :meth:`set_expanded`):

* the monitor: sub-state, reason, since when, pid, its experiment file;
* trust: why, since when;
* the run fence: which run, from where, since when, for how long, and
  whether it lapses (only while the monitor is READY);
* the connections and the SLM lease, one line each;
* the run loops with Start / Stop and their output -- the Sequences tab's own
  cards (:class:`~waxx.util.guis.sequences_panel.SequencesPanel`, sending
  through the same requester; a person's Start carries ``owner`` "person");
* the last ops-journal records, from the server's ``get_journal`` request
  (the records the server keeps in memory since it started; at most
  :data:`JOURNAL_LINES`; the client never names a file).

Requests: ``get_journal`` when the details open, every
:data:`JOURNAL_POLL_MS` while they are shown, and shortly after a broadcast;
the loops' own requests (start, stop, output) as the Sequences tab sends
them.  Nothing else -- the host feeds ``status_json``.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.device_state import connections as conns
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

_MONITOR_LEVEL = {"READY": "on", "LOADING": "partial", "NOT_READY": "hazard"}
_CONN_LEVEL = {conns.CONNECTED: "on", conns.CONNECTING: "partial",
               conns.DISCONNECTED: "off", conns.FAILED: "hazard"}
_LOOP_LEVEL = {"running": "on", "starting": "partial", "stopping": "partial",
               "latched": "hazard", "failed": "hazard"}


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


class MonitorStatePanel(QWidget):
    """The monitor server's state bar (see the module docstring).

    ``requester``: ``request_dict -> reply_dict``; ``synchronous``: call it on
    the GUI thread (tests); ``runner``: an existing RequestRunner to share;
    ``by``: who is clicking (default ``user@host``); ``monitor_button``: a
    button the host drives itself, shown in place of the panel's own monitor
    pill; ``expanded``: open the details at start."""

    #: The monitor pill was clicked (the host starts / restarts the monitor).
    monitor_clicked = pyqtSignal()

    def __init__(self, requester: Callable[[dict], Any] | None = None, *,
                 synchronous: bool = False, runner: RequestRunner | None = None,
                 by: str | None = None, parent=None,
                 monitor_button: QPushButton | None = None, expanded: bool = False):
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
        self.conn_chips = QHBoxLayout()
        self.conn_chips.setSpacing(6)
        row.addLayout(self.conn_chips)
        self.slm_chip = _chip()
        row.addWidget(self.slm_chip)
        self.loop_chips_box = QHBoxLayout()
        self.loop_chips_box.setSpacing(6)
        row.addLayout(self.loop_chips_box)
        self.loop_chips: dict[str, QLabel] = {}
        row.addStretch(1)
        outer.addWidget(self.bar)

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

        sec = section("Connections")
        self.conn_box = QVBoxLayout()
        self.conn_box.setSpacing(2)
        sec.addLayout(self.conn_box)
        self.conn_rows: dict[str, tuple[QLabel, QLabel]] = {}
        self.no_conn = _line(theme.FG_MUTED)
        sec.addWidget(self.no_conn)
        self.slm_line = _line()
        sec.addWidget(self.slm_line)

        sec = section("Run loops")
        # the Sequences tab's loop cards; their requests go through this
        # panel's runner (no discovery, no socket)
        self.sequences = SequencesPanel(start_sender=False, show_hold=False, show_queue=False,
                                        runner=self.runner, by=self.by)
        sec.addWidget(self.sequences)

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
                               "Show the details: monitor, trust, fence, connections, run "
                               "loops, ops journal")
        if expanded and not self._shut:
            self.refresh_journal()

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
        """A server broadcast: trust, the run fence, connections, the SLM
        reinit and the loops update at once; any of them asks for the journal
        again shortly.  Ignored after shutdown."""
        if self._shut or not isinstance(payload, dict):
            return
        kind = payload.get("type")
        if kind == "trust":
            self.state["trust"] = payload.get("trust")
        elif kind == "run_pending":
            self.state["run_pending"] = payload.get("run_pending")
        elif kind == "connections" and isinstance(payload.get("connections"), dict):
            self.state["connections"] = payload["connections"]
        elif kind == "slm_reinit":
            self.state["slm_reinit"] = payload.get("slm_reinit")
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
        self._render_connections()
        self._render_loops()

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

    def _render_connections(self) -> None:
        snap = self.state.get("connections") if self.reachable else None
        snap = snap if isinstance(snap, dict) else {}
        for key in list(self.conn_rows):
            if key not in snap:
                chip, line = self.conn_rows.pop(key)
                chip.deleteLater()
                line.deleteLater()
        for key, c in snap.items():
            c = c or {}
            if key not in self.conn_rows:
                chip = _chip()
                self.conn_chips.addWidget(chip)
                line = _line()
                self.conn_box.addWidget(line)
                self.conn_rows[key] = (chip, line)
            chip, line = self.conn_rows[key]
            state = str(c.get("state") or "?")
            label = str(c.get("label") or key)
            chip.setText(f"{label}: {state}")
            set_style_if_changed(chip, chip_css(_CONN_LEVEL.get(state, "unknown")))
            detail = c.get("detail") or ""
            text = (label + (f" -- {detail}" if detail else "")
                    + (f" (since {_clock(c.get('since'), True)})" if c.get("since") else ""))
            line.setText(f"{state}: {text}")
            tip = str(c.get("tooltip") or "")
            chip.setToolTip(text + (f"\n{tip}" if tip else ""))
            line.setToolTip(tip)
        self.no_conn.setText("" if snap or not self.reachable else
                             "This server holds no connections.")
        self.no_conn.setVisible(bool(self.reachable and not snap))

        slm = self.state.get("slm_reinit") if self.reachable else None
        if isinstance(slm, dict):
            text = slm_text(slm)
            self.slm_line.setText(text)
            bad = str(slm.get("state")) in ("failed", "unreachable")
            set_style_if_changed(self.slm_line,
                                 f"color: {ERR_TEXT if bad else theme.FG}; font-size: 11px;")
            self.slm_line.show()
            self.slm_chip.setText(f"SLM: {slm.get('state') or '?'}")
            set_style_if_changed(self.slm_chip, chip_css(
                "hazard" if bad else "partial" if slm.get("reinit_due") else "off"))
            self.slm_chip.setToolTip(text)
            self.slm_chip.show()
        else:
            self.slm_line.hide()
            self.slm_chip.hide()

    def _render_loops(self) -> None:
        loops = self.state.get("run_loops") if self.reachable else None
        loops = loops if isinstance(loops, dict) else {}
        for key in list(self.loop_chips):
            if key not in loops:
                self.loop_chips.pop(key).deleteLater()
        for key, info in loops.items():
            info = info or {}
            chip = self.loop_chips.get(key)
            if chip is None:
                chip = _chip()
                self.loop_chips_box.addWidget(chip)
                self.loop_chips[key] = chip
            state = str(info.get("state") or "?")
            chip.setText(f"{info.get('title') or key}: {state}")
            set_style_if_changed(chip, chip_css(_LOOP_LEVEL.get(state, "off")))
            chip.setToolTip(str(info.get("text") or "")
                            + "\nStart / Stop: open the details (the arrow at the left).")

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
        if self._own_runner:
            self.runner.shutdown()


__all__ = ["MonitorStatePanel", "fence_text", "monitor_button_css", "monitor_text", "slm_text"]
