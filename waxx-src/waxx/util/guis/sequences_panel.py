"""Sequences tab of the Device Control GUI.

The experiments the monitor server runs itself, one card each: its run loops
(:mod:`waxx.util.device_state.run_loop`; kexp: the BEC TOF loop) and its
reset experiment (:mod:`waxx.util.device_state.state_reset`; kexp: MOT
Observe).  A card is one row -- Start / Stop (Run for the reset) on the left,
the title, a state pill and what it is doing, and ▾ on the right, which opens
the terminal output of its runs.  The server keeps the last lines
(:class:`~waxx.util.device_state.output_log.OutputLog`); an open log asks for
the new ones every second while this tab is visible, and nothing is polled
otherwise.

Everything shown is the server's: every GUI sees the same loop and reset.
The reset's Run emits ``reset_requested``; the host GUI confirms and sends it
(the same dialog as the untrusted banner's button).
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QToolButton,
    QVBoxLayout, QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.guis.composite_panel import (
    CARD_GAP, ERR_TEXT, OK_TEXT, WARN_TEXT, _OpSender, _card_css, _clock, _label_pill_css,
    _pill_button_css, _small,
)
from waxx.util.guis.device_summary import reset_title

#: How often an open log asks the server for new lines.
LOG_POLL_MS = 1000
#: Lines an open log keeps (the server keeps its own last lines too).
LOG_MAX_LINES = 5000
LOG_HEIGHT = 240

#: loop state -> (pill text, pill level)
_LOOP_PILL = {"idle": ("idle", "unknown"), "running": ("RUNNING", "on"),
              "stopping": ("stopping", "partial"), "stopped": ("stopped", "off"),
              "latched": ("LATCHED OFF", "warn")}
#: reset state -> (pill text, pill level)
_RESET_PILL = {"idle": ("idle", "unknown"), "running": ("RUNNING", "on"),
               "done": ("done", "off"), "failed": ("FAILED", "hazard")}

_NO_OUTPUT = ("This monitor server keeps no output (older code): restart the monitor "
              "server to get the log.")


class SequenceCard(QFrame):
    """One sequence: its buttons, title, pill and status on one row, and its
    terminal output under ▾.  ``kind`` is ``"run_loop"`` (``key`` names the
    loop) or ``"reset"``."""

    def __init__(self, panel: "SequencesPanel", kind: str, key: str = ""):
        super().__init__()
        self.panel = panel
        self.kind = kind
        self.key = key
        self.info: dict = {}
        self.message = ""
        #: Number of the last output line shown (OutputLog numbering).
        self.after = 0
        self.fetching = False
        self.unsupported = False
        self._log_error = ""
        self.setObjectName("composite_card")
        self.setStyleSheet(_card_css(theme.ACCENT))
        box = QVBoxLayout(self)
        box.setContentsMargins(10, 5, 6, 5)
        box.setSpacing(5)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.start_button = QPushButton("Start" if kind == "run_loop" else "Run")
        self.stop_button = QPushButton("Stop") if kind == "run_loop" else None
        for b in (self.start_button, self.stop_button):
            if b is None:
                continue
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setStyleSheet(_pill_button_css(theme.FG, theme.BORDER))
            row.addWidget(b)
        if kind == "run_loop":
            self.start_button.clicked.connect(lambda _=False: panel.start_loop(key))
            self.stop_button.clicked.connect(lambda _=False: panel.stop_loop(key))
            self.stop_button.setToolTip("Let the run in progress finish and save, then end "
                                        "the loop and start the monitor.")
        else:
            self.start_button.clicked.connect(lambda _=False: panel.reset_requested.emit())
        row.addSpacing(4)
        self.title = QLabel(key)
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        self.title.setFont(font)
        self.title.setStyleSheet(f"color: {theme.FG_STRONG};")
        row.addWidget(self.title)
        self.pill = QLabel("")
        row.addWidget(self.pill)
        self.status = _small("")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.status, 1)
        self.toggle = QToolButton()
        self.toggle.setText("▾")
        self.toggle.setCheckable(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.setToolTip("Show the terminal output of its runs")
        self.toggle.setStyleSheet(
            f"QToolButton {{ color: {theme.FG_MUTED}; border: 0; border-radius: 4px;"
            f" font-size: 14px; padding: 0px 6px; }}"
            f"QToolButton:hover {{ color: {theme.FG_STRONG}; background: {theme.BG_BUTTON_HOVER}; }}"
            f"QToolButton:checked {{ color: {theme.FG_STRONG}; }}")
        self.toggle.toggled.connect(self._on_toggled)
        row.addWidget(self.toggle)
        box.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(LOG_MAX_LINES)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setFixedHeight(LOG_HEIGHT)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.log.setFont(mono)
        self.log.setStyleSheet(f"QPlainTextEdit {{ background: {theme.BG_SUNKEN};"
                               f" color: {theme.FG}; border: 1px solid {theme.BORDER};"
                               f" border-radius: 4px; }}")
        self.log.setPlaceholderText("No output yet.")
        self.log.hide()
        box.addWidget(self.log)
        self.hide()

    # -- state ------------------------------------------------------------------

    @property
    def log_open(self) -> bool:
        return self.toggle.isChecked()

    def set_info(self, info: dict | None) -> None:
        """The server's info for this sequence (``RunLoop.info()`` /
        ``StateReset.info()``); None hides the card."""
        self.info = dict(info) if isinstance(info, dict) else {}
        self.setVisible(bool(self.info))
        if self.info:
            self._show()

    def set_message(self, text: str) -> None:
        """A local note (a refused Start) shown until the next change."""
        self.message = text
        self._show()

    def _show(self) -> None:
        info = self.info
        if self.kind == "run_loop":
            self.title.setText(info.get("title") or self.key)
            about = info.get("about") or ""
            self.title.setToolTip("\n\n".join(p for p in (about, info.get("path") or "") if p))
            state = info.get("state") or "idle"
            text, level = _LOOP_PILL.get(state, (state, "unknown"))
            parts = [info.get("text") or ""]
            runs = int(info.get("runs") or 0)
            if state in ("running", "stopping"):
                parts.append(f"{runs} saved since {_clock(info.get('started'))}")
            elif state in ("stopped", "latched") and info.get("ended"):
                parts.append(f"{runs} saved · ended {_clock(info.get('ended'))}")
            last = info.get("last") or {}
            if last.get("outcome") == "saved" and state not in ("running", "stopping"):
                parts.append(f"last saved: run {last.get('run_id')}")
            color = WARN_TEXT if state == "latched" else (OK_TEXT if state == "running"
                                                          else theme.FG_MUTED)
        else:
            self.title.setText(reset_title(info))
            self.title.setToolTip(str(info.get("about") or ""))
            state = info.get("state") or "idle"
            text, level = _RESET_PILL.get(state, (state, "unknown"))
            who = "@".join(p for p in (info.get("operator"), info.get("client")) if p)
            if state == "running":
                parts = [f"started {_clock(info.get('started'))}" + (f" by {who}" if who else ""),
                         info.get("text") or ""]
            elif state in ("done", "failed"):
                parts = [info.get("text") or "", f"ended {_clock(info.get('ended'))}"]
            else:
                parts = ["not run since the monitor server started"]
            color = ERR_TEXT if state == "failed" else (OK_TEXT if state == "running"
                                                        else theme.FG_MUTED)
        self.pill.setText(text)
        self.pill.setStyleSheet(_label_pill_css(level))
        if self.message:
            parts.insert(0, self.message)
            self.message = ""
        self.status.setText(" · ".join(p for p in parts if p))
        self.status.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.refresh_buttons()

    def refresh_buttons(self) -> None:
        reachable = self.panel.reachable
        state = self.info.get("state") or "idle"
        unreachable = "" if reachable else "the monitor server is unreachable"
        if self.kind == "run_loop":
            self.start_button.setEnabled(reachable and state not in ("running", "stopping"))
            self.start_button.setToolTip(unreachable)
            self.stop_button.setEnabled(reachable and state == "running")
            self.stop_button.setText("Stopping…" if state == "stopping" else "Stop")
        else:
            running = state == "running"
            self.start_button.setText("Running…" if running else "Run")
            self.start_button.setEnabled(reachable and not running)
            self.start_button.setToolTip(
                unreachable or f"Run {self.info.get('expt')}.py through the monitor server. It "
                               "takes the core; its end state marks the device state trusted.")

    # -- the log ----------------------------------------------------------------

    def _on_toggled(self, open_: bool) -> None:
        self.toggle.setText("▴" if open_ else "▾")
        self.toggle.setToolTip("Hide the terminal output" if open_
                               else "Show the terminal output of its runs")
        self.log.setVisible(open_)
        if open_:
            self.panel.fetch_output(self)

    def output_request(self) -> dict:
        obj = {"type": "output", "kind": self.kind, "after": self.after}
        if self.kind == "run_loop":
            obj["key"] = self.key
        return obj

    def add_output(self, reply: dict) -> bool:
        """Append one ``output`` reply; True when the server has more
        waiting."""
        if reply.get("status") != "ok":
            msg = str(reply.get("msg") or "no reply")
            if "unknown type" in msg:
                self.unsupported = True
                text = _NO_OUTPUT
                tail = self.info.get("tail")
                if tail:
                    text += "\n\nLast lines of its last failed run:\n" + "\n".join(tail)
                self.log.setPlainText(text)
            elif msg != self._log_error:
                self._append([f"(output not available: {msg})"])
            self._log_error = msg
            return False
        self._log_error = ""
        try:
            first, nxt = int(reply.get("first", 0)), int(reply.get("next", 0))
        except (TypeError, ValueError):
            return False
        if nxt <= self.after:
            # The server restarted (its numbering started again): start over.
            self.log.clear()
            self.after = 0
            return True
        lines = [str(x) for x in (reply.get("lines") or [])]
        if lines and first > self.after + 1:
            lines.insert(0, "(earlier lines were not kept)" if self.after == 0
                         else f"({first - self.after - 1} lines were not kept)")
        self._append(lines)
        self.after = max(self.after, nxt - 1)
        return bool(reply.get("more"))

    def _append(self, lines: list[str]) -> None:
        if not lines:
            return
        bar = self.log.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 2
        self.log.appendPlainText("\n".join(lines))
        if at_end:
            bar.setValue(bar.maximum())


class SequencesPanel(QWidget):
    """The Sequences tab: one :class:`SequenceCard` per run loop the monitor
    server offers, then its reset experiment.

    ``log_line(text)`` records a loop's end in the host GUI's changes log.
    The host GUI feeds in the server's state (:meth:`set_reachable`,
    :meth:`set_loops` / :meth:`on_run_loop`, :meth:`set_reset`)."""

    #: The reset card's Run: the host GUI confirms and asks the server.
    reset_requested = pyqtSignal()

    def __init__(self, log_line: Callable[[str], None] | None = None, parent=None,
                 start_sender: bool = True):
        super().__init__(parent)
        self._log_line = log_line
        self._reachable = False
        self._req = 0
        self._requests: dict[int, Callable[[dict], None]] = {}
        self.loop_cards: dict[str, SequenceCard] = {}
        box = QVBoxLayout(self)
        box.setContentsMargins(CARD_GAP, 10, CARD_GAP, CARD_GAP)
        box.setSpacing(8)
        self.empty = _small("")
        box.addWidget(self.empty)
        self._cards = QVBoxLayout()
        self._cards.setSpacing(8)
        box.addLayout(self._cards)
        self.reset_card = SequenceCard(self, "reset")
        self._cards.addWidget(self.reset_card)
        box.addStretch(1)

        self._sender = _OpSender(self)
        self._sender.requested.connect(self._on_requested)
        if start_sender:
            self._sender.start()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.poll_logs)
        self._timer.start(LOG_POLL_MS)
        self._refresh_empty()

    # -- inputs from the host GUI -------------------------------------------------

    @property
    def reachable(self) -> bool:
        return self._reachable

    def cards(self) -> list[SequenceCard]:
        return list(self.loop_cards.values()) + [self.reset_card]

    def set_reachable(self, reachable: bool) -> None:
        if bool(reachable) != self._reachable:
            self._reachable = bool(reachable)
            for card in self.cards():
                card.refresh_buttons()
            self._refresh_empty()

    def set_loops(self, loops: dict | None) -> None:
        """The server's loops (``status_json`` ``run_loops``)."""
        for info in (loops or {}).values():
            self._update_loop(info)

    def on_run_loop(self, info: dict | None) -> None:
        """A loop's change, as the server broadcast it (or replied)."""
        if not isinstance(info, dict) or not info.get("key"):
            return
        card = self.loop_cards.get(info["key"])
        before = card.info.get("state") if card is not None else None
        self._update_loop(info)
        if info.get("state") in ("stopped", "latched") and before != info.get("state") \
                and self._log_line is not None:
            self._log_line(f"[loop] {info.get('title')}: {info.get('state')} -- "
                           f"{info.get('text')}")

    def _update_loop(self, info: dict | None) -> None:
        if not isinstance(info, dict) or not info.get("key"):
            return
        key = info["key"]
        card = self.loop_cards.get(key)
        if card is None:
            card = SequenceCard(self, "run_loop", key)
            self._cards.insertWidget(len(self.loop_cards), card)
            self.loop_cards[key] = card
        card.set_info(info)
        self._refresh_empty()

    def set_reset(self, info: dict | None) -> None:
        """The server's reset experiment (``StateReset.info()``; None: it has
        none)."""
        self.reset_card.set_info(info)
        self._refresh_empty()

    def _refresh_empty(self) -> None:
        shown = any(not c.isHidden() for c in self.cards())
        if not self._reachable:
            text = "Waiting for the monitor server…"
        else:
            text = ("The monitor server runs no sequences (it has no run loops and no reset "
                    "experiment configured).")
        self.empty.setText(text)
        self.empty.setVisible(not shown)

    # -- run loops ----------------------------------------------------------------------

    def start_loop(self, key: str) -> bool:
        card = self.loop_cards.get(key)
        info = card.info if card is not None else {}
        title = info.get("title") or key
        lines = [info["about"]] if info.get("about") else []
        lines.append("It runs back to back on the monitor server, whether or not this window "
                     "stays open, until Stop. It stops by itself (latched off) on an Abort in "
                     "liveOD, a run that fails or saves incomplete, someone else's run, or the "
                     "monitor being started. Stop lets the run in progress finish and save, "
                     "then starts the monitor.")
        if info.get("path"):
            lines.append(f"File: {info['path']}")
        if not self.confirm(title, "\n\n".join(lines), verb=f"Start {title}"):
            return False

        def done(reply):
            if reply.get("status") == "ok":
                self.on_run_loop(reply.get("loop"))
            elif card is not None:
                card.set_message(f"✕ not started: {reply.get('msg')}")
        self.send_request({"type": "run_loop", "action": "start", "loop": key}, done)
        return True

    def stop_loop(self, key: str) -> None:
        card = self.loop_cards.get(key)

        def done(reply):
            if reply.get("status") == "ok":
                self.on_run_loop(reply.get("loop"))
            elif card is not None:
                card.set_message(f"✕ Stop: {reply.get('msg')}")
        self.send_request({"type": "run_loop", "action": "stop", "loop": key}, done)

    # -- logs -------------------------------------------------------------------------

    def poll_logs(self) -> None:
        """Ask for new output for every open log -- only while this tab is
        on screen."""
        if not self.isVisible():
            return
        for card in self.cards():
            if card.log_open and not card.isHidden():
                self.fetch_output(card)

    def fetch_output(self, card: SequenceCard) -> None:
        if card.fetching or card.unsupported or not self._reachable:
            return
        card.fetching = True

        def done(reply):
            card.fetching = False
            if card.add_output(reply) and card.log_open:
                self.fetch_output(card)
        self.send_request(card.output_request(), done)

    # -- plumbing ---------------------------------------------------------------------

    def send_request(self, obj: dict, on_reply: Callable[[dict], None] | None = None) -> int:
        self._req += 1
        req = self._req
        obj = dict(obj)
        obj.setdefault("client", self._sender.client_name)
        if on_reply is not None:
            self._requests[req] = on_reply
        self._sender.request(req, obj)
        return req

    def _on_requested(self, req: int, reply: dict) -> None:
        callback = self._requests.pop(req, None)
        if callback is not None:
            callback(reply)

    def confirm(self, title: str, text: str, verb: str = "Send") -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(text)
        yes = box.addButton(verb, QMessageBox.ButtonRole.AcceptRole)
        no = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(no)
        box.exec()
        return box.clickedButton() is yes

    def shutdown(self) -> None:
        self._timer.stop()
        self._sender.stop()
        self._sender.wait(2000)
