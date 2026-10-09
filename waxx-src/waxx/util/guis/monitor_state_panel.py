"""The monitor server's state at a glance: the "State" tab beside the run
queue panel (:mod:`waxx.util.guis.run_queue_panel`) in the Server Dashboard's
monitor panel (:mod:`waxx.util.guis.monitor_panel`, over the network)
and in the monitor server's own window when it runs as a GUI.

:class:`MonitorStatePanel` is fed like the queue panel -- :meth:`set_state`
with the server's ``status_json`` dict (None when it does not answer),
:meth:`on_broadcast` with its broadcasts -- and acts through an injected
``requester(request_dict) -> reply_dict`` (run off the GUI thread by a
:class:`~waxx.util.guis.request_runner.RequestRunner`).  It never discovers
the server or opens a socket.  It shows:

* the monitor experiment: READY / LOADING / NOT_READY, its sub-state and
  reason, since when, its pid;
* the trust flag: a red banner while the device state is untrusted (the
  reason in the tooltip and on the line below);
* the run fence: which run announced itself, from where, since when and for
  how long, and whether the monitor is READY (a fence lapses on its TTL only
  while it is);
* the run loops, with Start / Stop and their output -- the Sequences tab's own
  cards (:class:`~waxx.util.guis.sequences_panel.SequencesPanel`, sending
  through the same requester; a person's Start carries ``owner`` "person");
* the host-side connections the server holds between runs (the tweezer AWG:
  held, released for a run, failed) and the SLM's reinit lease (what blocks it,
  when it is due);
* the last ops-journal records, from the server's ``get_journal`` request
  (the records the server keeps in memory since it started; at most
  :data:`JOURNAL_LINES`; the client never names a file).

Requests: ``get_journal`` when the tab is shown, every
:data:`JOURNAL_POLL_MS` while it is visible, and shortly after a broadcast;
the loops' own requests (start, stop, output) as the Sequences tab sends
them.  Nothing else -- the host feeds ``status_json``.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.device_state import connections as conns
from waxx.util.device_state.op_journal import describe_entry
from waxx.util.device_state.run_queue_client import default_by
from waxx.util.guis.qt_upkeep import set_style_if_changed
from waxx.util.guis.request_runner import RequestRunner, is_unknown_request
from waxx.util.guis.run_queue_panel import ERR_TEXT, OK_TEXT, WARN_TEXT, _clock, _dur, _pill_css

#: Journal records asked for (the newest).
JOURNAL_LINES = 60
#: How often the journal is asked for while the tab is visible.
JOURNAL_POLL_MS = 5000
#: Delay between a broadcast and the journal request it causes (coalesced).
JOURNAL_DEBOUNCE_MS = 1000
#: The fence's "held N s" refresh.
TICK_MS = 1000

_MONITOR_LEVEL = {"READY": "on", "LOADING": "partial", "NOT_READY": "hazard"}
_CONN_LEVEL = {conns.CONNECTED: "on", conns.CONNECTING: "partial",
               conns.DISCONNECTED: "off", conns.FAILED: "hazard"}


def _section(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("state_section")
    frame.setStyleSheet(f"QFrame#state_section {{ background: {theme.BG_CARD};"
                        f" border: 1px solid {theme.BORDER}; border-radius: 8px; }}")
    box = QVBoxLayout(frame)
    box.setContentsMargins(10, 6, 10, 6)
    box.setSpacing(4)
    head = QLabel(title)
    font = QFont()
    font.setBold(True)
    head.setFont(font)
    head.setStyleSheet(f"color: {theme.FG_STRONG};")
    box.addWidget(head)
    return frame, box


def _line(color: str = theme.FG) -> QLabel:
    label = QLabel("")
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setStyleSheet(f"color: {color}; font-size: 12px;")
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


class MonitorStatePanel(QWidget):
    """The monitor server's state (see the module docstring).

    ``requester``: ``request_dict -> reply_dict``; ``synchronous``: call it on
    the GUI thread (tests); ``runner``: an existing RequestRunner to share;
    ``by``: who is clicking (default ``user@host``)."""

    def __init__(self, requester: Callable[[dict], Any] | None = None, *,
                 synchronous: bool = False, runner: RequestRunner | None = None,
                 by: str | None = None, parent=None):
        super().__init__(parent)
        from waxx.util.guis.sequences_panel import SequencesPanel  # noqa: PLC0415
        self.by = by or default_by()
        self._own_runner = runner is None
        self.runner = runner or RequestRunner(requester, synchronous=synchronous, parent=self)
        self.state: dict = {}
        self.reachable = False
        self.journal_supported = True
        self._journal_fetching = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        page = QWidget()
        scroll.setWidget(page)
        box = QVBoxLayout(page)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(8)

        frame, sec = _section("Monitor experiment")
        row = QHBoxLayout()
        self.monitor_pill = QLabel("?")
        row.addWidget(self.monitor_pill)
        self.monitor_line = _line()
        row.addWidget(self.monitor_line, 1)
        sec.addLayout(row)
        self.expt_line = _line(theme.FG_MUTED)
        sec.addWidget(self.expt_line)
        box.addWidget(frame)

        frame, sec = _section("Device state trust")
        self.trust_banner = QLabel("")
        self.trust_banner.setWordWrap(True)
        sec.addWidget(self.trust_banner)
        self.trust_line = _line(theme.FG_MUTED)
        sec.addWidget(self.trust_line)
        box.addWidget(frame)

        frame, sec = _section("Run fence")
        self.fence_line = _line()
        sec.addWidget(self.fence_line)
        box.addWidget(frame)

        frame, sec = _section("Run loops")
        # the Sequences tab's loop cards; their requests go through this
        # panel's runner (no discovery, no socket)
        self.sequences = SequencesPanel(start_sender=False, show_hold=False, show_queue=False,
                                        runner=self.runner, by=self.by)
        sec.addWidget(self.sequences)
        box.addWidget(frame)

        frame, sec = _section("Connections held by the server")
        self.conn_box = QVBoxLayout()
        self.conn_box.setSpacing(2)
        sec.addLayout(self.conn_box)
        self.conn_rows: dict[str, tuple[QLabel, QLabel]] = {}
        self.no_conn = _line(theme.FG_MUTED)
        sec.addWidget(self.no_conn)
        self.slm_line = _line()
        sec.addWidget(self.slm_line)
        box.addWidget(frame)

        frame, sec = _section("Ops journal (newest last)")
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
        self.journal.setMinimumHeight(220)
        self.journal.setPlaceholderText("No journal records yet.")
        sec.addWidget(self.journal)
        box.addWidget(frame)
        box.addStretch(1)

        self._journal_timer = QTimer(self)
        self._journal_timer.timeout.connect(self._poll_journal)
        self._journal_timer.start(JOURNAL_POLL_MS)
        self._journal_soon = QTimer(self)
        self._journal_soon.setSingleShot(True)
        self._journal_soon.timeout.connect(self.refresh_journal)
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(TICK_MS)
        self._render()

    # -- inputs ---------------------------------------------------------------------------

    def set_state(self, state: dict | None) -> None:
        """The server's ``status_json`` (None: it did not answer)."""
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
        again shortly."""
        if not isinstance(payload, dict):
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
            self.sequences.on_run_loop(payload.get("loop"))
        elif kind in ("state_update", "op_result"):
            pass                                      # frequent; the journal poll covers them
        self._render()
        if kind not in ("state_update",):
            self.journal_soon()

    # -- rendering -----------------------------------------------------------------------------

    def _render(self) -> None:
        s = self.state
        if not self.reachable:
            self.monitor_pill.setText("?")
            set_style_if_changed(self.monitor_pill, _pill_css("unknown"))
            self.monitor_line.setText("The monitor server is not answering.")
        else:
            name = str(s.get("state_name") or "?")
            self.monitor_pill.setText(name.replace("_", " "))
            set_style_if_changed(self.monitor_pill, _pill_css(_MONITOR_LEVEL.get(name, "unknown")))
            self.monitor_line.setText(monitor_text(s))
        self.expt_line.setText(f"monitor experiment: {s.get('expt_path') or '?'}"
                               if self.reachable else "")

        trust = s.get("trust") if self.reachable else None
        if isinstance(trust, dict) and not trust.get("trusted", True):
            self.trust_banner.setText("Device state UNTRUSTED: the device tabs may not match "
                                      "the hardware.")
            self.trust_banner.setStyleSheet("QLabel { background: #c62828; color: white;"
                                            " border-radius: 6px; padding: 4px 8px; }")
            self.trust_banner.setToolTip(str(trust.get("reason") or ""))
            self.trust_line.setText(f"why: {trust.get('reason') or '?'} "
                                    f"(since {_clock(trust.get('since'), True)})")
        elif isinstance(trust, dict):
            self.trust_banner.setText("Device state trusted.")
            self.trust_banner.setStyleSheet(f"QLabel {{ color: {OK_TEXT}; }}")
            self.trust_banner.setToolTip("")
            self.trust_line.setText(f"{trust.get('reason') or ''}"
                                    + (f" (since {_clock(trust.get('since'), True)})"
                                       if trust.get("since") else ""))
        else:
            self.trust_banner.setText("")
            self.trust_banner.setStyleSheet("")
            self.trust_line.setText("" if not self.reachable else "The server reports no trust "
                                                                  "flag.")
        self._render_fence()
        self._render_connections()

    def _render_fence(self) -> None:
        if not self.reachable:
            self.fence_line.setText("")
            return
        pending = self.state.get("run_pending")
        self.fence_line.setText(fence_text(pending, str(self.state.get("state_name") or "?")))
        set_style_if_changed(self.fence_line, f"color: {WARN_TEXT if pending else theme.FG_MUTED};"
                                              " font-size: 12px;")

    def _render_connections(self) -> None:
        snap = self.state.get("connections") if self.reachable else None
        snap = snap if isinstance(snap, dict) else {}
        for key in list(self.conn_rows):
            if key not in snap:
                pill, line = self.conn_rows.pop(key)
                pill.parentWidget().deleteLater()
        for key, c in snap.items():
            if key not in self.conn_rows:
                holder = QWidget()
                row = QHBoxLayout(holder)
                row.setContentsMargins(0, 0, 0, 0)
                pill = QLabel("")
                line = _line()
                row.addWidget(pill)
                row.addWidget(line, 1)
                self.conn_box.addWidget(holder)
                self.conn_rows[key] = (pill, line)
            pill, line = self.conn_rows[key]
            state = str((c or {}).get("state") or "?")
            pill.setText(state)
            set_style_if_changed(pill, _pill_css(_CONN_LEVEL.get(state, "unknown")))
            detail = (c or {}).get("detail") or ""
            line.setText(f"{(c or {}).get('label') or key}"
                         + (f" -- {detail}" if detail else "")
                         + (f" (since {_clock(c.get('since'), True)})" if c.get("since") else ""))
            line.setToolTip(str((c or {}).get("tooltip") or ""))
        self.no_conn.setText("" if snap or not self.reachable else
                             "This server holds no connections.")
        self.no_conn.setVisible(bool(self.reachable and not snap))

        slm = self.state.get("slm_reinit") if self.reachable else None
        if isinstance(slm, dict):
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
            self.slm_line.setText(" ".join(parts))
            color = ERR_TEXT if str(slm.get("state")) in ("failed", "unreachable") else theme.FG
            set_style_if_changed(self.slm_line, f"color: {color}; font-size: 12px;")
            self.slm_line.show()
        else:
            self.slm_line.hide()

    def _on_tick(self) -> None:
        if self.isVisible() and self.reachable:
            self._render_fence()
            self.monitor_line.setText(monitor_text(self.state))

    # -- the journal ------------------------------------------------------------------------------

    def showEvent(self, event):                                 # noqa: N802
        super().showEvent(event)
        self.refresh_journal()

    def _poll_journal(self) -> None:
        if self.isVisible():
            self.refresh_journal()

    def journal_soon(self) -> None:
        if not self._journal_soon.isActive():
            self._journal_soon.start(JOURNAL_DEBOUNCE_MS)

    def journal_request(self) -> dict:
        return {"type": "get_journal", "n": JOURNAL_LINES}

    def refresh_journal(self) -> None:
        if self._journal_fetching or not self.journal_supported:
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
        self.journal_note.setText(f"last {len(lines)} records the server keeps in memory"
                                  + (f"; on disk: {reply['path']}" if reply.get("path") else ""))

    # -- shutdown -------------------------------------------------------------------------------

    def shutdown(self) -> None:
        self._journal_timer.stop()
        self._journal_soon.stop()
        self._tick.stop()
        self.sequences.shutdown()                     # leaves the shared runner alone
        if self._own_runner:
            self.runner.shutdown()


__all__ = ["MonitorStatePanel", "fence_text", "monitor_text"]
