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

A pick loop's card (kexp: "Experiment loop") opens a file dialog at Start; the
file goes to the server relative to its experiments folder (:func:`server_path`),
which checks it and describes it from its own copy before the confirm.

A loop with scan settings (:mod:`waxx.util.device_state.loop_scan`; kexp: the
BEC TOF loop's t_tof) has ⚙ left of ▾: start, stop (blank: repeat the start
value), points and repeats, sent to the server, which uses them from the next
run.  The card's status line shows the current scan.

Everything shown is the server's: every GUI sees the same loop and reset.
The reset's Run emits ``reset_requested``; the host GUI confirms and sends it
(the same dialog as the untrusted banner's button).

Above the cards: the person hold (:mod:`~waxx.util.device_state.person_hold`).
"Hold — a person has the machine" asks for a reason and puts it on at the
server; while it is on the row says since when, by whom and why, and the same
button releases it.  The server also puts it on by itself on every Reset a
person presses in liveOD, whatever run it lands on (a Reset sent by the queue,
by an agent's own reset_liveod.py or by liveOD itself never does).  Under it,
one line on the run
queue (:meth:`SequencesPanel.set_queue`, from ``status_json`` alone): its
state and how many jobs are queued; the queue itself is shown and driven in
the monitor server's own panel (Server Dashboard).

Requests go through the GUI's own sender (a MonitorClient found by
discovery) -- or, when the host passes ``requester`` (the monitor server's
own window: a direct call into the server), through that callable on a
worker thread (:class:`~waxx.util.guis.request_runner.RequestRunner`), with
no discovery and no socket.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.device_state import loop_scan
from waxx.util.device_state.person_hold import describe as describe_hold
from waxx.util.guis.composite_panel import (
    CARD_GAP, ERR_TEXT, OK_TEXT, WARN_TEXT, _OpSender, _card_css, _clock, _label_pill_css,
    _pill_button_css, _small,
)
from waxx.util.guis.device_summary import reset_title
from waxx.util.guis.qt_upkeep import delete_later, set_style_if_changed
from waxx.util.guis.request_runner import RequestRunner
from waxx.util.device_state.run_queue_client import default_by
from waxx.util.guis.run_queue_panel import QueueSummaryLine

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


def server_path(local: str, root: str) -> str | None:
    """A file chosen in this GUI's dialog as the monitor server's pick loop
    wants it: relative to ``root`` (the server's folder).  On the server's own
    machine the file is inside ``root``; on another lab PC the same repo
    layout is assumed -- the part after the root's last two folders (e.g.
    ``kexp/experiments``).  None when neither fits."""
    picked = Path(local)
    try:
        return picked.resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, ValueError):
        pass
    tail = [p.lower() for p in Path(root).parts[-2:]]
    parts = list(picked.parts)
    lowered = [p.lower() for p in parts]
    n = len(tail)
    for i in range(len(parts) - n, -1, -1):
        if n and lowered[i:i + n] == tail and len(parts) > i + n:
            return Path(*parts[i + n:]).as_posix()
    return None


def _local_start(path, root: str) -> str:
    """Where the dialog opens: the loop's last file or its root, when this
    machine has them."""
    for candidate in (path, root):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return ""


class ScanSettingsDialog(QDialog):
    """A loop's scan: start, stop (blank: the start value alone), points
    between them (grayed out without a stop value), repeats.  Values are shown
    in the spec's unit; :meth:`settings` returns SI.  OK is enabled only for
    settings the server would accept (the same check, loop_scan.normalize)."""

    def __init__(self, title: str, scan: dict, running: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title}: scan settings")
        self.spec = loop_scan.ScanSpec(
            xvar=str(scan.get("xvar") or "value"), unit=str(scan.get("unit") or ""),
            scale=float(scan.get("scale") or 1.0), minimum=scan.get("minimum"),
            maximum=scan.get("maximum"))
        current = scan.get("settings") or {}
        unit = f" ({self.spec.unit})" if self.spec.unit else ""
        box = QVBoxLayout(self)
        form = QFormLayout()
        self.start = QLineEdit(self._shown(current.get("start")))
        self.stop = QLineEdit(self._shown(current.get("stop")))
        self.stop.setPlaceholderText("blank: repeat the start value")
        self.points = QSpinBox()
        self.points.setRange(2, int(scan.get("max_points") or loop_scan.MAX_POINTS))
        self.points.setValue(max(2, int(current.get("n") or 2)))
        self.repeats = QSpinBox()
        self.repeats.setRange(1, int(scan.get("max_repeats") or loop_scan.MAX_REPEATS))
        self.repeats.setValue(int(current.get("repeats") or 1))
        form.addRow(f"{self.spec.xvar} start{unit}", self.start)
        form.addRow(f"{self.spec.xvar} stop{unit}", self.stop)
        form.addRow("points", self.points)
        form.addRow("repeats", self.repeats)
        box.addLayout(form)
        self.summary = _small("")
        self.summary.setWordWrap(True)
        box.addWidget(self.summary)
        if running:
            box.addWidget(_small("The loop is running: new settings apply from its next run."))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        box.addWidget(self.buttons)
        for w in (self.start, self.stop):
            w.textChanged.connect(self._update)
        for w in (self.points, self.repeats):
            w.valueChanged.connect(self._update)
        self._update()

    def _shown(self, v) -> str:
        return "" if v is None else f"{float(v) / self.spec.scale:g}"

    def _si(self, text: str):
        text = text.strip()
        if not text:
            return None
        try:
            return float(text) * self.spec.scale
        except ValueError:
            return text             # normalize says it is not a number

    def settings(self) -> dict | None:
        """The checked settings (SI), or None when they are refused."""
        try:
            return loop_scan.normalize(self.spec, self._raw())
        except loop_scan.ScanSettingsError:
            return None

    def _raw(self) -> dict:
        start = self._si(self.start.text())
        return {"start": "" if start is None else start, "stop": self._si(self.stop.text()),
                "n": self.points.value(), "repeats": self.repeats.value()}

    def _update(self) -> None:
        self.points.setEnabled(bool(self.stop.text().strip()))
        try:
            s = loop_scan.normalize(self.spec, self._raw())
        except loop_scan.ScanSettingsError as exc:
            s, text, color = None, f"✕ {exc}", ERR_TEXT
        else:
            text, color = loop_scan.describe(self.spec, s), theme.FG_MUTED
        self.summary.setText(text)
        self.summary.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(s is not None)


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
        #: ⚙: the loop's scan settings (shown when the server offers them).
        self.settings_button = QToolButton()
        self.settings_button.setText("⚙")
        self.settings_button.setAutoRaise(True)
        self.settings_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_button.setStyleSheet(
            f"QToolButton {{ color: {theme.FG_MUTED}; border: 0; border-radius: 4px;"
            f" font-size: 14px; padding: 0px 6px; }}"
            f"QToolButton:hover {{ color: {theme.FG_STRONG}; background: {theme.BG_BUTTON_HOVER}; }}"
            f"QToolButton:disabled {{ color: {theme.BORDER}; }}")
        self.settings_button.clicked.connect(lambda _=False: panel.configure_loop(key))
        self.settings_button.hide()
        row.addWidget(self.settings_button)
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
            title = info.get("title") or self.key
            if info.get("pick"):
                title += f": {info['expt']}" if info.get("expt") else " (no file chosen yet)"
            self.title.setText(title)
            about = info.get("about") or ""
            self.title.setToolTip("\n\n".join(p for p in (about, info.get("path") or "") if p))
            state = info.get("state") or "idle"
            text, level = _LOOP_PILL.get(state, (state, "unknown"))
            parts = [info.get("text") or ""]
            runs = int(info.get("runs") or 0)
            if state in ("running", "stopping"):
                parts.append(f"{runs} saved since {_clock(info.get('started'))}")
                if info.get("owner"):
                    parts.append(f"owner {info['owner']}")
            elif state in ("stopped", "latched") and info.get("ended"):
                parts.append(f"{runs} saved · ended {_clock(info.get('ended'))}")
            last = info.get("last") or {}
            if last.get("outcome") == "saved" and state not in ("running", "stopping"):
                parts.append(f"last saved: run {last.get('run_id')}")
            scan = info.get("scan") or {}
            if scan.get("text"):
                parts.append(f"scan: {scan['text']}")
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
        set_style_if_changed(self.pill, _label_pill_css(level))
        if self.message:
            parts.insert(0, self.message)
            self.message = ""
        self.status.setText(" · ".join(p for p in parts if p))
        set_style_if_changed(self.status, f"color: {color}; font-size: 11px;")
        self.refresh_buttons()

    def refresh_buttons(self) -> None:
        reachable = self.panel.reachable
        state = self.info.get("state") or "idle"
        unreachable = "" if reachable else "the monitor server is unreachable"
        if self.kind == "run_loop":
            pick = bool(self.info.get("pick"))
            self.start_button.setText("Start…" if pick else "Start")
            self.start_button.setEnabled(reachable and state not in ("running", "stopping"))
            self.start_button.setToolTip(
                unreachable or ("Choose an experiment file in "
                                f"{self.info.get('root') or 'the experiments folder'}, "
                                "then run it back to back" if pick else ""))
            self.stop_button.setEnabled(reachable and state == "running")
            self.stop_button.setText("Stopping…" if state == "stopping" else "Stop")
            scan = self.info.get("scan")
            self.settings_button.setVisible(bool(scan))
            self.settings_button.setEnabled(reachable)
            self.settings_button.setToolTip(
                unreachable or "Scan settings: "
                + str((scan or {}).get("text") or "") + "\n(a change applies from the next run)")
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


HOLD_TEXT = "Hold — a person has the machine"
RELEASE_TEXT = "Release hold"


class HoldRow(QFrame):
    """The person hold: one button that puts it on (asking for a reason) or,
    while it is on, releases it; and a line saying since when, by whom and
    why.  Hidden while the server reports no ``person_hold`` (older server)."""

    def __init__(self, panel: "SequencesPanel"):
        super().__init__()
        self.panel = panel
        self.info: dict = {}
        self.setObjectName("composite_card")
        self.setStyleSheet(_card_css(theme.ACCENT))
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 5, 6, 5)
        row.setSpacing(8)
        self.button = QPushButton(HOLD_TEXT)
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button.setStyleSheet(_pill_button_css(theme.FG, theme.BORDER))
        self.button.clicked.connect(lambda _=False: panel.toggle_hold())
        row.addWidget(self.button)
        self.pill = QLabel("")
        row.addWidget(self.pill)
        self.status = _small("")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        self.hide()

    @property
    def held(self) -> bool:
        return bool(self.info.get("active"))

    def set_info(self, info: dict | None) -> None:
        self.info = dict(info) if isinstance(info, dict) else {}
        self.setVisible(isinstance(info, dict))
        self._show()

    def set_message(self, text: str) -> None:
        self._show(text)

    def _show(self, message: str = "") -> None:
        if self.held:
            text, level = "HELD", "warn"
            line = describe_hold(self.info)
            line = line[0].upper() + line[1:] + " — agents' runs and the run loops wait."
            color = WARN_TEXT
        else:
            text, level = "free", "off"
            line = "No hold: agents may use the machine when it is free."
            color = theme.FG_MUTED
        if message:
            line = f"{message} · {line}"
        self.pill.setText(text)
        set_style_if_changed(self.pill, _label_pill_css(level))
        self.status.setText(line)
        set_style_if_changed(self.status, f"color: {color}; font-size: 11px;")
        self.refresh_buttons()

    def refresh_buttons(self) -> None:
        reachable = self.panel.reachable
        self.button.setText(RELEASE_TEXT if self.held else HOLD_TEXT)
        self.button.setEnabled(reachable)
        self.button.setToolTip(
            "the monitor server is unreachable" if not reachable else
            ("Release the hold: agents' queued runs and the run loops may run again."
             if self.held else
             "Put a hold on the machine: agents' runs wait and the run loops stop after "
             "their run in progress, until someone releases it. It never expires by itself."))


class SequencesPanel(QWidget):
    """The Sequences tab: one :class:`SequenceCard` per run loop the monitor
    server offers, then its reset experiment.

    ``log_line(text)`` records a loop's end in the host GUI's changes log.
    The host GUI feeds in the server's state (:meth:`set_reachable`,
    :meth:`set_loops` / :meth:`on_run_loop`, :meth:`set_reset`)."""

    #: The reset card's Run: the host GUI confirms and asks the server.
    reset_requested = pyqtSignal()

    def __init__(self, log_line: Callable[[str], None] | None = None, parent=None,
                 start_sender: bool = True, requester: Callable[[dict], dict] | None = None,
                 synchronous_requests: bool = False, show_hold: bool = True,
                 show_queue: bool = True, runner: RequestRunner | None = None,
                 by: str | None = None):
        super().__init__(parent)
        #: who is clicking (``user@host``): a loop's start / stop / configure
        #: carry it as ``by`` and the user as ``operator`` (the server names
        #: who started or stopped a loop from operator@client)
        self.by = by or default_by()
        self._operator = self.by.split("@", 1)[0]
        self._log_line = log_line
        self._reachable = False
        self._req = 0
        self._requests: dict[int, Callable[[dict], None]] = {}
        self._show_hold = bool(show_hold)
        self._show_queue = bool(show_queue)
        self.loop_cards: dict[str, SequenceCard] = {}
        #: The last file chosen for a pick loop (the dialog opens there).
        self._last_pick = ""
        box = QVBoxLayout(self)
        box.setContentsMargins(CARD_GAP, 10, CARD_GAP, CARD_GAP)
        box.setSpacing(8)
        self.hold_row = HoldRow(self)
        box.addWidget(self.hold_row)
        #: "Queue: <state> (<n> queued) -- ..." (status_json only)
        self.queue_line = QueueSummaryLine()
        box.addWidget(self.queue_line)
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
        #: with a requester (or a host's runner), requests go through it -- no
        #: discovery, no socket
        self._own_runner = runner is None and requester is not None
        self._runner = runner if runner is not None else (
            RequestRunner(requester, synchronous=synchronous_requests, parent=self)
            if requester is not None else None)
        if start_sender and self._runner is None:
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
            self.hold_row.refresh_buttons()
            self._refresh_empty()

    # -- the person hold ----------------------------------------------------------------

    def set_hold(self, info: dict | None) -> None:
        """The server's person hold (``status_json`` ``person_hold``, or its
        ``person_hold`` broadcast); None: the server has none (older code)."""
        before = self.hold_row.held
        self.hold_row.set_info(info)
        if not self._show_hold:
            self.hold_row.hide()
        if isinstance(info, dict) and bool(info.get("active")) != before \
                and self._log_line is not None:
            self._log_line("[hold] " + (describe_hold(info) if info.get("active")
                                        else "person hold released"))

    def toggle_hold(self) -> bool:
        """The hold button: put the hold on (asking for a reason) or release it."""
        if self.hold_row.held:
            request = {"type": "run_queue", "action": "release", "owner": "person",
                       "by": self._hold_by()}
        else:
            reason = self.ask_hold_reason()
            if reason is None:
                return False
            request = {"type": "run_queue", "action": "hold", "reason": reason, "owner": "person",
                       "by": self._hold_by()}

        def done(reply):
            if reply.get("status") == "ok":
                self.set_hold(reply.get("person_hold"))
            else:
                self.hold_row.set_message(f"✕ {reply.get('msg')}")
        self.send_request(request, done)
        return True

    def _hold_by(self) -> str:
        host = self._sender.client_name
        return f"Device Control GUI on {host}" if host else "Device Control GUI"

    def ask_hold_reason(self) -> str | None:
        """The reason dialog (tests replace it): the text, or None (cancelled)."""
        text, ok = QInputDialog.getText(
            self, HOLD_TEXT,
            "Why (shown to everyone, and to agents waiting for the machine):",
            QLineEdit.EchoMode.Normal, "a person has the machine")
        if not ok:
            return None
        return text.strip() or "a person has the machine"

    def set_queue(self, info: dict | None) -> None:
        """The run queue's summary (``status_json`` ``run_queue``, or its
        ``run_queue`` broadcast); None: the server has none (older code)."""
        self.queue_line.set_info(info)
        if not self._show_queue:
            self.queue_line.hide()

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
        if info.get("pick"):
            return self._pick_and_start(key, card, info)
        return self._confirm_and_start(key, card, info)

    def _pick_and_start(self, key: str, card, info: dict) -> bool:
        """A pick loop: choose the file here, have the server check it (and
        describe it from its own copy), confirm, start."""
        root = str(info.get("root") or "")
        start_at = self._last_pick or _local_start(info.get("path"), root)
        chosen = self.choose_file(f"{info.get('title') or key}: choose an experiment", start_at)
        if not chosen:
            return False
        self._last_pick = chosen
        rel = server_path(chosen, root)
        if rel is None:
            if card is not None:
                card.set_message(f"✕ not started: {chosen} is not inside the experiments "
                                 f"folder ({root})")
            return False

        def described(reply):
            if reply.get("status") != "ok":
                if card is not None:
                    card.set_message(f"✕ not started: {reply.get('msg')}")
                return
            self._confirm_and_start(key, card, dict(info, about=reply.get("about") or "",
                                                    path=reply.get("path") or rel,
                                                    expt=reply.get("expt") or ""),
                                    path=rel)
        self.send_request({"type": "run_loop", "action": "describe", "loop": key,
                           "path": rel}, described)
        return True

    def choose_file(self, caption: str, start_at: str) -> str:
        """The file dialog (tests replace it)."""
        chosen, _ = QFileDialog.getOpenFileName(self, caption, start_at,
                                                "Python experiments (*.py)")
        return chosen

    def _confirm_and_start(self, key: str, card, info: dict, path: str | None = None) -> bool:
        title = info.get("title") or key
        if path is not None and info.get("expt"):
            title = f"{title}: {info['expt']}"
        lines = [info["about"]] if info.get("about") else []
        if path is not None:
            lines.append("The file is read again at every run: an edit takes effect at the "
                         "loop's next run. Its runs must save through liveOD (save_data=True), "
                         "or the loop stops after the first.")
        lines.append("It runs back to back on the monitor server, whether or not this window "
                     "stays open, until Stop. It stops by itself (latched off) on an Abort in "
                     "liveOD, a run that fails or saves incomplete, someone else's run, or the "
                     "monitor being started. Stop lets the run in progress finish and save, "
                     "then starts the monitor.")
        if info.get("path"):
            lines.append(f"File{' (on the monitor server)' if path is not None else ''}: "
                         f"{info['path']}")
        if not self.confirm(title, "\n\n".join(lines), verb=f"Start {title}"):
            return False

        def done(reply):
            if reply.get("status") == "ok":
                self.on_run_loop(reply.get("loop"))
            elif card is not None:
                card.set_message(f"✕ not started: {reply.get('msg')}")
        # a person is clicking: the loop is a person's (the run queue stops
        # only agent / queue / idle-started loops for its jobs)
        request = {"type": "run_loop", "action": "start", "loop": key, "owner": "person"}
        if path is not None:
            request["path"] = path
        self.send_request(request, done)
        return True

    def configure_loop(self, key: str) -> bool:
        """⚙: edit the loop's scan settings and send them to the server."""
        card = self.loop_cards.get(key)
        info = card.info if card is not None else {}
        scan = info.get("scan")
        if not scan:
            return False
        settings = self.ask_scan(info.get("title") or key, scan,
                                 running=info.get("state") in ("running", "stopping"))
        if settings is None:
            return False

        def done(reply):
            if reply.get("status") == "ok":
                self.on_run_loop(reply.get("loop"))
            elif card is not None:
                card.set_message(f"✕ scan not set: {reply.get('msg')}")
        self.send_request({"type": "run_loop", "action": "configure", "loop": key,
                           "scan": settings}, done)
        return True

    def ask_scan(self, title: str, scan: dict, running: bool = False) -> dict | None:
        """The settings dialog (tests replace it): new settings (SI), or None."""
        dialog = ScanSettingsDialog(title, scan, running=running, parent=self)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            return dialog.settings()
        finally:
            delete_later(dialog)

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
        if obj.get("type") == "run_loop" and obj.get("action") in ("start", "stop", "configure"):
            obj.setdefault("operator", self._operator)
            obj.setdefault("by", self.by)
        if self._runner is not None:
            self._runner.send(obj, on_reply)
            return req
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
        clicked = box.clickedButton()
        delete_later(box)
        return clicked is yes

    def shutdown(self) -> None:
        self._timer.stop()
        if self._runner is not None and self._own_runner:
            self._runner.shutdown()
        self._sender.stop()
        self._sender.wait(2000)
