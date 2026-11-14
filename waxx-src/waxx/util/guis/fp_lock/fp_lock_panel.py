"""FPLockPanel: dashboard panel for the FP peak-separation lock server.

Laid out for a small dock: one header line (lock state, the peak separation
against its setpoint, the plot picker), the server's warnings (first peak near
the sweep start, second peak near the sweep end, HOLD, rail, no data, COM), a
compact grid of the latest readings (names spelled out in tooltips), one row
of buttons, and one plot showing the separation, the error, the output or the
two peak positions (edge bands shaded) over the last ``history_s`` seconds
(60 by default; the box beside the plot picker changes it).  Controls: lock on / off, reset
setpoint (setp = the measured separation), clear integrator, a settings dialog
(gains, window, peak finding), and a console for raw firmware commands (hidden
until its button is pressed; Up recalls the commands sent).  Only reset setpoint
and applying settings ask first; the override button sets the output to the
voltage in the box beside it (the firmware holds its PI frozen) and disables the
lock button while on; the state pill reads HOLD under override.

Construction never blocks: the client is discovered on a worker thread, polls
and actions run off the GUI thread (SnapshotPoller, RequestRunner).
Machine-agnostic: the host passes the server id.
"""

from __future__ import annotations

import collections
import html
import logging
import math
import time
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy,
    QSpinBox, QVBoxLayout,
    QWidget,
)

from waxx.util.dashboard.embed_helpers import WidgetPanelBase, auto_cleanup_timers
from waxx.util.guis.fp_lock import fp_lock_protocol as proto
from waxx.util.guis.qt_upkeep import set_style_if_changed
from waxx.util.guis.request_runner import RequestRunner

__all__ = ["FPLockPanel", "SettingsDialog"]

_LOG = logging.getLogger("waxx.fp_lock.panel")

HISTORY_POINTS = 3000      # one row per ~1.2 s: about an hour
HISTORY_S_DEFAULT = 60.0   # s: the plot shows the last this many seconds
HISTORY_S_MAX = 3600
OUT_WARN_V = 8.0          # V: the "out" readout turns orange beyond this (the firmware rail)
# values that come from the last pair: blanked in the plot, tinted, when the pair is stale
_PAIR_KEYS = ("sep", "err", "tfirst", "t2")

_PILL = {
    "LOCKED": ("#2e8b57", "LOCKED"),
    "HOLD": ("#d4a017", "HOLD"),
    "OFF": ("#777777", "LOCK OFF"),
    "NO DATA": ("#b22222", "NO DATA"),
    "NO LINK": ("#b22222", "NO LINK"),
    "CONNECTING": ("#888888", "connecting…"),
}
_LEVEL_COLOR = {"error": "#b22222", "warn": "#b8860b", "info": "#555555"}
_VAL_CSS = "font-family: Consolas, monospace; font-size: 11px;"
_VAL_STYLE = "QLabel { " + _VAL_CSS + " }"
_BTN_STYLE = "QPushButton { padding: 1px 4px; font-size: 11px; }"
_SPIN_STYLE = "QDoubleSpinBox { font-size: 11px; padding: 0px 2px; }"
OVOLT_MAX_V = 10.0        # V: the firmware clamps the forced output to this (kDacMax)
# plot views: combo label, the history keys drawn
_VIEWS = {"sep": ("sep (ms)", ("sep",)),
          "err": ("err (ms)", ("err",)),
          "output": ("output (V)", ("feedback", "integ")),
          "peaks": ("peaks (ms)", ("tfirst", "t2"))}


class _CommandLine(QLineEdit):
    """Console input: Up / Down step back and forward through the commands sent."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history: list[str] = []
        self._pos: Optional[int] = None      # index into history while stepping, else None

    def remember(self, line: str) -> None:
        if not self.history or self.history[-1] != line:
            self.history.append(line)
        self._pos = None

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Up and self.history:
            self._pos = len(self.history) - 1 if self._pos is None else max(0, self._pos - 1)
            self.setText(self.history[self._pos])
            return
        if e.key() == Qt.Key.Key_Down and self._pos is not None:
            self._pos += 1
            if self._pos >= len(self.history):
                self._pos = None
                self.clear()
            else:
                self.setText(self.history[self._pos])
            return
        self._pos = None
        super().keyPressEvent(e)


def _ask(parent, title: str, text: str) -> bool:
    return QMessageBox.question(parent, title, text,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes


def _fmt(v, spec: str = ".3f", unit: str = "") -> str:
    if not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    return f"{v:{spec}}{(' ' + unit) if unit else ''}"


def _window_spin(parent, seconds: float, on_change: Callable[[float], None]) -> QSpinBox:
    """The plot's time-window box (s)."""
    spin = QSpinBox(parent)
    spin.setRange(5, HISTORY_S_MAX)
    spin.setSingleStep(30)
    spin.setValue(int(round(seconds)))
    spin.setSuffix(" s")
    spin.setToolTip("plot window: the last this many seconds")
    spin.setFixedWidth(66)
    spin.setStyleSheet("QSpinBox { font-size: 11px; padding: 0px 2px; }")
    spin.valueChanged.connect(lambda v: on_change(float(v)))
    return spin


class _PollSource:
    """What the poller calls (pool thread): the snapshot, plus the rows and
    console lines that are new since the last poll."""

    def __init__(self, client):
        self.client = client
        self.timeout = getattr(client, "timeout", None)
        self.seq = 0
        self.text_seq = 0

    def _rediscover(self, timeout: float = 0.5):
        fn = getattr(self.client, "_rediscover", None)
        return fn(timeout=timeout) if callable(fn) and not getattr(self.client, "_fixed_address", False) else False

    def get_snapshot(self) -> dict:
        snap = self.client.get_snapshot()
        seq = int(snap.get("seq") or 0)
        if seq < self.seq:                      # server restarted
            self.seq = 0
        if seq > self.seq:
            hist = self.client.get_history(since=self.seq)
            snap["_new_rows"] = hist.get("rows", [])
            self.seq = int(hist.get("seq", seq))
        text_seq = int(snap.get("text_seq") or 0)
        if text_seq < self.text_seq:
            self.text_seq = 0
        if text_seq > self.text_seq:
            text = self.client.get_text(since=self.text_seq)
            snap["_new_text"] = [t["line"] for t in text.get("lines", [])]
            self.text_seq = int(text.get("seq", text_seq))
        return snap


class _ClientBuild(QThread):
    done = pyqtSignal(object, object)

    def __init__(self, factory, parent=None):
        super().__init__(parent)
        self._factory = factory

    def run(self) -> None:
        try:
            self.done.emit(self._factory(), None)
        except BaseException as exc:  # noqa: BLE001
            self.done.emit(None, exc)


class SettingsDialog(QDialog):
    """Edit the firmware settings; ``changes()`` is {name: text} of the edited ones."""

    def __init__(self, settings: dict, missing: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FP lock settings")
        self._initial = dict(settings)
        self._edits: dict[str, Any] = {}
        form = QFormLayout()
        for s in proto.SETTINGS:
            if s.name == "lock":          # the panel's Lock button
                continue
            value = settings.get(s.name)
            if s.kind == "switch":
                w = QCheckBox(self)
                w.setChecked(bool(value))
            else:
                w = QLineEdit(self)
                w.setText("" if value is None else (f"{int(value)}" if s.kind == "int" else f"{value:g}"))
            w.setToolTip(s.help)
            if s.name in missing:
                w.setEnabled(False)
                w.setToolTip("the firmware does not have this setting")
            label = f"{s.name}" + (f" ({s.unit})" if s.unit else "")
            form.addRow(label, w)
            self._edits[s.name] = w
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply
                               | QDialogButtonBox.StandardButton.Cancel, self)
        apply_btn = box.button(QDialogButtonBox.StandardButton.Apply)
        apply_btn.setDefault(True)          # Enter in any field accepts
        apply_btn.clicked.connect(self.accept)
        box.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        note = QLabel("Only edited values are sent. Settings live on the board until it is "
                      "power-cycled or reflashed.", self)
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addLayout(form)
        lay.addWidget(box)

    def edit(self, name: str):
        return self._edits[name]

    def changes(self) -> dict:
        out = {}
        for name, w in self._edits.items():
            if not w.isEnabled():
                continue
            old = self._initial.get(name)
            if isinstance(w, QCheckBox):
                new = 1 if w.isChecked() else 0
                if old is None or int(old) != new:
                    out[name] = str(new)
                continue
            text = w.text().strip()
            if not text:
                continue
            try:
                same = old is not None and math.isclose(float(text), float(old), rel_tol=1e-9,
                                                        abs_tol=1e-12)
            except ValueError:
                same = False
            if not same:
                out[name] = text
        return out


class FPLockPanel(WidgetPanelBase):
    """The FP lock panel (module docstring)."""

    def __init__(self, server_id: str, *, title: str = "FP lock",
                 client: Any = None, client_factory: Optional[Callable[[], Any]] = None,
                 confirm: Optional[Callable[[str, str], bool]] = None,
                 synchronous: bool = False, plots: bool = True,
                 history_s: float = HISTORY_S_DEFAULT, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._server_id = server_id
        self._title = title
        self._history_s = float(history_s)
        self._client = None
        self._poller = None
        self._source: Optional[_PollSource] = None
        self._confirm = confirm or (lambda t, txt: _ask(self, t, txt))
        self._runner = RequestRunner(None, synchronous=synchronous, parent=self)
        self._snap: dict = {}
        self._hist = {k: collections.deque(maxlen=HISTORY_POINTS)
                      for k in ("t", "sep", "err", "feedback", "integ", "tfirst", "t2")}
        self._plots: dict = {}
        self._bands: list = []

        self._view = "peaks"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(6)
        self._pill = QLabel(self)
        self._pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pill.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        top.addWidget(self._pill, 0)
        self._sep_big = QLabel("sep --", self)
        self._sep_big.setToolTip("peak separation (latest) and its setpoint")
        self._sep_big.setStyleSheet("QLabel { font-family: Consolas, monospace; font-size: 13px;"
                                    " font-weight: 700; }")
        top.addWidget(self._sep_big, 0)
        top.addStretch(1)
        self._view_box = None
        if plots:
            self._view_box = QComboBox(self)
            for key, (label, _) in _VIEWS.items():
                self._view_box.addItem(label, key)
            self._view_box.addItem("no plot", "none")
            self._view_box.setCurrentIndex(self._view_box.findData(self._view))
            self._view_box.setToolTip("what the plot shows (time axis: s, 0 = now)")
            self._view_box.currentIndexChanged.connect(
                lambda _i: self._set_view(self._view_box.currentData()))
            top.addWidget(self._view_box, 0)
            self._window_spin = _window_spin(self, self._history_s, self._set_history_s)
            top.addWidget(self._window_spin, 0)
        outer.addLayout(top)

        self._warn = QLabel("", self)
        self._warn.setWordWrap(True)
        self._warn.setTextFormat(Qt.TextFormat.RichText)
        self._warn.setStyleSheet("QLabel { font-size: 11px; }")
        self._warn.hide()
        outer.addWidget(self._warn)

        # sep / setp are in the header; the rail state is a warning
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(0)
        self._vals: dict[str, QLabel] = {}
        cells = (("err", "err", "error = sep - setp"),
                 ("feedback", "out", "lock output"),
                 ("integ", "int", "integral term"),
                 ("period", "per", "measured sweep period"),
                 ("tfirst", "pk1", "1st peak position in the sweep"),
                 ("t2", "pk2", "2nd peak position in the sweep"),
                 ("h1", "h1", "1st peak height"),
                 ("h2", "h2", "2nd peak height"))
        for i, (key, label, tip) in enumerate(cells):
            r, c = divmod(i, 4)
            name = QLabel(label, self)
            name.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
            name.setToolTip(tip)
            val = QLabel("--", self)
            val.setStyleSheet(_VAL_STYLE)
            val.setToolTip(tip)
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(name, r, 2 * c)
            grid.addWidget(val, r, 2 * c + 1)
            self._vals[key] = val
        for c in range(4):
            grid.setColumnStretch(2 * c + 1, 1)
        outer.addLayout(grid)

        buttons = QHBoxLayout()
        buttons.setSpacing(2)
        buttons.setContentsMargins(0, 0, 0, 0)
        self._lock_btn = QPushButton("Lock …", self)
        self._lock_btn.clicked.connect(self._on_lock_clicked)
        self._reset_btn = QPushButton("Reset setp", self)
        self._reset_btn.setToolTip("setp = the separation the lock measures now (asks first)")
        self._reset_btn.clicked.connect(self._on_reset_setpoint)
        self._clear_btn = QPushButton("Clear int", self)
        self._clear_btn.setToolTip("zero the integral and the output (0 V); the state is kept")
        self._clear_btn.clicked.connect(self._on_clear)
        self._override_btn = QPushButton("Override", self)
        self._override_btn.clicked.connect(self._on_override_clicked)
        self._ovolt_spin = QDoubleSpinBox(self)
        self._ovolt_spin.setRange(-OVOLT_MAX_V, OVOLT_MAX_V)
        self._ovolt_spin.setDecimals(3)
        self._ovolt_spin.setSingleStep(0.05)
        self._ovolt_spin.setSuffix(" V")
        self._ovolt_spin.setToolTip("the output while override is on (sent to the lock as ovolt)")
        self._ovolt_spin.setFixedWidth(76)
        self._ovolt_spin.setStyleSheet(_SPIN_STYLE)
        # arrows and typing are sent after a short pause, so one click is one command
        self._ovolt_timer = QTimer(self)
        self._ovolt_timer.setSingleShot(True)
        self._ovolt_timer.setInterval(400)
        self._ovolt_timer.timeout.connect(self._send_ovolt)
        self._ovolt_spin.valueChanged.connect(lambda _v: self._ovolt_timer.start())
        self._settings_btn = QPushButton("Settings…", self)
        self._settings_btn.setToolTip("gains, window, peak finding")
        self._settings_btn.clicked.connect(self._on_settings)
        self._console_btn = QPushButton("Console", self)
        self._console_btn.setCheckable(True)
        self._console_btn.setToolTip("raw firmware commands")
        self._console_btn.toggled.connect(self._on_console_toggled)
        for b in (self._lock_btn, self._reset_btn, self._clear_btn, self._override_btn,
                  self._settings_btn, self._console_btn):
            b.setStyleSheet(_BTN_STYLE)
        buttons.addWidget(self._lock_btn)
        buttons.addWidget(self._reset_btn)
        buttons.addWidget(self._clear_btn)
        buttons.addWidget(self._override_btn)
        buttons.addWidget(self._ovolt_spin)
        buttons.addWidget(self._settings_btn)
        buttons.addWidget(self._console_btn)
        outer.addLayout(buttons)

        self._msg = QLabel("", self)
        self._msg.setWordWrap(True)
        self._msg.setStyleSheet("QLabel { color: #444; font-size: 11px; }")
        self._msg.hide()
        outer.addWidget(self._msg)

        self._plot_box = None
        if plots:
            self._plot_box = QVBoxLayout()
            self._plot_box.setContentsMargins(0, 0, 0, 0)
            outer.addLayout(self._plot_box, 1)
            self._init_plots()

        self._console = QPlainTextEdit(self)
        self._console.setReadOnly(True)
        self._console.setMaximumBlockCount(500)
        self._console.setStyleSheet("QPlainTextEdit { font-family: Consolas, monospace;"
                                    " font-size: 11px; }")
        self._console.setMaximumHeight(140)
        self._console_in = _CommandLine(self)
        self._console_in.setPlaceholderText("command, e.g. help or tmin 1.5 (Enter sends, Up recalls)")
        self._console_in.returnPressed.connect(self._on_console_send)
        outer.addWidget(self._console)
        outer.addWidget(self._console_in)
        self._console.hide()
        self._console_in.hide()

        self._set_controls_enabled(False)
        self._set_pill("CONNECTING")

        if client is not None:
            self._on_client_built(client, None)
        else:
            self._factory = client_factory or self._default_factory
            QTimer.singleShot(0, self._build_client)

    # ------------------------------------------------------------------ #
    # Client
    # ------------------------------------------------------------------ #

    def _default_factory(self):
        from waxx.util.guis.fp_lock.fp_lock_client import FPLockClient  # noqa: PLC0415
        return FPLockClient(self._server_id, discovery_timeout=3.0)

    def _build_client(self) -> None:
        self._builder = _ClientBuild(self._factory, self)
        self._builder.done.connect(self._on_client_built)
        self._builder.start()

    def _on_client_built(self, client, exc) -> None:
        if exc is not None:
            self._set_pill("NO LINK", f"server {self._server_id} not found")
            QTimer.singleShot(5000, self._build_client)
            return
        self._client = client
        self._source = _PollSource(client)
        from waxx.util.dashboard.snapshot_poller import SnapshotPoller  # noqa: PLC0415
        self._poller = SnapshotPoller(self._source, panel_id=self._server_id, parent=self)
        self._poller.snapshot_received.connect(self.apply_snapshot)
        self._poller.conn_changed.connect(self._on_conn_changed)
        self._poller.start()

    def _on_conn_changed(self, status: str, detail: str) -> None:
        if status != "connected":
            self._set_pill("NO LINK", f"server: {detail}" if detail else "server unreachable")
            self._set_controls_enabled(False)

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #

    def apply_snapshot(self, snap: dict) -> None:
        self._snap = snap
        row = snap.get("latest") or {}
        settings = snap.get("settings") or {}
        com = snap.get("com") or {}
        warnings = snap.get("warnings") or []
        codes = {w.get("code") for w in warnings}

        if com.get("status") != "connected":
            self._set_pill("NO LINK", f"COM {com.get('port', '?')}: {com.get('status', '?')}")
        elif codes & {"no_data", "stale"}:
            self._set_pill("NO DATA")
        elif settings.get("override") == 1:
            # override holds the output at ovolt: HOLD, whatever state older firmware reports
            self._set_pill("HOLD")
        else:
            self._set_pill(row.get("state_name") or "CONNECTING")
        label = snap.get("label") or ""
        self._sep_big.setToolTip("peak separation (latest) and its setpoint\n"
                                 f"{self._title} · {com.get('port', '')}"
                                 + (f" · {label}" if label else ""))

        if warnings:
            html = "<br>".join(
                f"<span style='color:{_LEVEL_COLOR.get(w['level'], '#555')}'>"
                f"{'⚠ ' if w['level'] != 'info' else ''}{w['text']}</span>" for w in warnings)
            self._warn.setText(html)
            self._warn.show()
        else:
            self._warn.hide()

        v = self._vals
        self._sep_big.setText(f"sep {_fmt(row.get('sep'), '.3f', 'ms')}  "
                              f"(setp {_fmt(settings.get('setp'), '.3f')})")
        v["err"].setText(_fmt(row.get("err"), "+.3f", "ms"))
        override_on = settings.get("override") == 1
        out_v = settings.get("ovolt") if override_on else row.get("feedback")
        v["feedback"].setText(_fmt(out_v, "+.4f", "V"))
        self._tint(v["feedback"], isinstance(out_v, (int, float)) and abs(out_v) > OUT_WARN_V)
        # the box follows the board, except while it is being edited or a value is pending
        ovolt = settings.get("ovolt")
        if (isinstance(ovolt, (int, float)) and not self._ovolt_timer.isActive()
                and not self._ovolt_spin.hasFocus()):
            self._ovolt_spin.blockSignals(True)
            self._ovolt_spin.setValue(float(ovolt))
            self._ovolt_spin.blockSignals(False)
        v["integ"].setText(_fmt(row.get("integ"), "+.4f", "V"))
        v["tfirst"].setText(_fmt(row.get("tfirst"), ".2f", "ms"))
        v["t2"].setText(_fmt(row.get("t2"), ".2f", "ms"))
        v["period"].setText(_fmt(row.get("period"), ".1f", "ms"))
        v["h1"].setText(_fmt(row.get("h1"), ".2f", "V"))
        v["h2"].setText(_fmt(row.get("h2"), ".2f", "V"))
        edge = snap.get("edge") or {}
        # orange when a peak sits within the edge margin of the sweep ends
        margin, right = edge.get("margin_ms", 10.0), edge.get("right_edge_ms", 120.0)
        tf, t2 = row.get("tfirst"), row.get("t2")
        stale = proto.pair_is_stale(row)   # no new pair since the previous line
        self._tint(v["tfirst"], stale or (isinstance(tf, (int, float)) and tf < margin))
        self._tint(v["t2"], stale or (isinstance(t2, (int, float)) and t2 > right - margin))
        self._tint(v["err"], stale)
        for k in ("h1", "h2"):
            self._tint(v[k], stale)

        lock_on = settings.get("lock")
        self._lock_btn.setText("Lock ON" if lock_on == 1 else
                               ("Lock OFF" if lock_on == 0 else "Lock …"))
        self._lock_btn.setToolTip("click to turn the lock off" if lock_on == 1 else
                                  "click to turn the lock on")
        self._override_btn.setText("Override ON" if override_on else "Override")
        self._override_btn.setToolTip(
            f"output forced to {_fmt(settings.get('ovolt'), '.3f', 'V')}; "
            "click to stop overriding" if override_on else
            f"force the output to {_fmt(settings.get('ovolt'), '.3f', 'V')} (the box beside it); "
            "the lock is disabled while on")
        set_style_if_changed(self._override_btn,
                             "QPushButton { padding: 2px 6px; background-color: #b8860b; color: white;"
                             " font-weight: 600; }" if override_on else _BTN_STYLE)
        set_style_if_changed(self._lock_btn,
                             "QPushButton { padding: 2px 6px; background-color: #2e8b57; color: white;"
                             " font-weight: 600; }" if lock_on == 1 else _BTN_STYLE)
        self._set_controls_enabled(com.get("status") == "connected")

        for row_ in snap.get("_new_rows", []):
            self._append_history(row_)
        self._redraw(edge, settings.get("setp"))
        for line in snap.get("_new_text", []):
            self._console.appendPlainText(line)

    def _append_history(self, r: dict) -> None:
        h = self._hist
        h["t"].append(r.get("t", time.time()))
        # a row with no new pair repeats an older sweep's pair: a gap, not a point
        stale = proto.pair_is_stale(r)
        for k in ("sep", "err", "feedback", "integ", "tfirst", "t2"):
            val = r.get(k)
            if stale and k in _PAIR_KEYS:
                val = None
            h[k].append(float(val) if isinstance(val, (int, float)) else float("nan"))

    def _redraw(self, edge: dict, setp=None) -> None:
        if not self._plots or not self._hist["t"]:
            return
        import numpy as np  # noqa: PLC0415
        t = np.asarray(self._hist["t"]) - time.time()
        keep = t >= -self._history_s
        for key, curve in self._plots["curves"].items():
            curve.setData(t[keep], np.asarray(self._hist[key], dtype=float)[keep], connect="finite")
        self._plots["plot"].setXRange(-self._history_s, 0, padding=0)
        self._setp_ok = isinstance(setp, (int, float)) and math.isfinite(setp)
        if self._setp_ok:
            self._plots["setp"].setValue(setp)
        self._plots["setp"].setVisible(self._setp_ok and self._view == "sep")
        right, margin = edge.get("right_edge_ms"), edge.get("margin_ms")
        if isinstance(right, (int, float)) and isinstance(margin, (int, float)):
            lo, hi = self._bands
            lo.setRegion((0.0, margin))
            hi.setRegion((right - margin, right))
            self._right_edge = right
            if self._view == "peaks":
                self._plots["plot"].setYRange(0, right, padding=0.02)

    def _init_plots(self) -> None:
        try:
            import pyqtgraph as pg  # noqa: PLC0415
        except Exception:
            self._plot_box.addWidget(QLabel("pyqtgraph not available"))
            return
        lay = pg.GraphicsLayoutWidget()
        lay.setBackground("w")
        lay.setMinimumHeight(80)
        lay.ci.setContentsMargins(0, 2, 2, 0)
        lay.setToolTip("time axis: s, 0 = now")
        plot = lay.addPlot(row=0, col=0)
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setXRange(-self._history_s, 0, padding=0)
        plot.setMouseEnabled(x=False, y=True)
        plot.getAxis("left").setWidth(40)
        plot.getAxis("bottom").setHeight(18)
        font = self.font()
        font.setPointSize(7)
        for ax in ("left", "bottom"):
            try:
                plot.getAxis(ax).setTickFont(font)
            except AttributeError:                  # older pyqtgraph
                pass
        legend = plot.addLegend(offset=(2, 2), labelTextSize="7pt", verSpacing=-6)
        curves = {
            "sep": plot.plot(pen=pg.mkPen("#000000", width=1.2)),
            "err": plot.plot(pen=pg.mkPen("#1f77b4", width=1.2)),
            "feedback": plot.plot(pen=pg.mkPen("#d62728", width=1.2)),
            "integ": plot.plot(pen=pg.mkPen("#ff9896", width=1.0, style=Qt.PenStyle.DashLine)),
            "tfirst": plot.plot(pen=pg.mkPen("#2ca02c", width=1.2)),
            "t2": plot.plot(pen=pg.mkPen("#9467bd", width=1.2)),
        }
        for _ in range(2):
            band = pg.LinearRegionItem(values=(0, 0), orientation="horizontal", movable=False,
                                       brush=pg.mkBrush(178, 34, 34, 40), pen=pg.mkPen(None))
            plot.addItem(band)
            self._bands.append(band)
        setp_line = pg.InfiniteLine(angle=0, movable=False,
                                    pen=pg.mkPen("#d62728", width=1.0, style=Qt.PenStyle.DashLine),
                                    label="setp", labelOpts={"position": 0.03, "color": "#d62728"})
        plot.addItem(setp_line)
        setp_line.hide()
        self._plots = {"layout": lay, "curves": curves, "plot": plot, "legend": legend,
                       "setp": setp_line}
        self._setp_ok = False
        self._right_edge = None
        self._plot_box.addWidget(lay)
        self._set_view(self._view)

    _LEGEND_NAMES = {"feedback": "output", "integ": "integ", "tfirst": "1st", "t2": "2nd"}

    def _set_view(self, view: str) -> None:
        """Show one view (``_VIEWS``) in the plot, or hide the plot (``"none"``)."""
        self._view = view
        if not self._plots:
            return
        keys = _VIEWS.get(view, ("", ()))[1]
        legend = self._plots["legend"]
        legend.clear()
        for key, curve in self._plots["curves"].items():
            curve.setVisible(key in keys)
            if key in keys and len(keys) > 1:
                legend.addItem(curve, self._LEGEND_NAMES.get(key, key))
        for band in self._bands:
            band.setVisible(view == "peaks")
        self._plots["setp"].setVisible(view == "sep" and self._setp_ok)
        plot = self._plots["plot"]
        if view == "peaks" and self._right_edge is not None:
            plot.setYRange(0, self._right_edge, padding=0.02)
        else:
            plot.enableAutoRange(axis="y")
        self._plots["layout"].setVisible(view != "none")

    def _set_history_s(self, seconds: float) -> None:
        """Show the last ``seconds`` of history in the plot."""
        self._history_s = float(seconds)
        self._redraw((self._snap.get("edge") or {}), (self._snap.get("settings") or {}).get("setp"))

    def _set_pill(self, key: str, detail: str = "") -> None:
        # the pill stays short (a long detail would widen the dock); the detail
        # goes on the warnings line, which the next snapshot rewrites
        color, text = _PILL.get(key, ("#888888", key))
        self._pill.setText(text)
        self._pill.setToolTip(text + (f" — {detail}" if detail else ""))
        if detail:
            self._warn.setText(f"<span style='color:{color}'>⚠ {text}: {html.escape(detail)}</span>")
            self._warn.show()
        set_style_if_changed(self._pill, f"QLabel {{ background-color: {color}; color: white;"
                                         " font-size: 13px; font-weight: 700; padding: 1px 8px;"
                                         " border-radius: 4px; }")

    @staticmethod
    def _tint(label: QLabel, on: bool) -> None:
        set_style_if_changed(label, "QLabel { " + _VAL_CSS
                                    + (" color: #b8860b; font-weight: 700;" if on else "") + " }")

    def _set_controls_enabled(self, on: bool) -> None:
        override_on = ((self._snap.get("settings") or {}).get("override") == 1)
        for b in (self._reset_btn, self._clear_btn, self._override_btn,
                  self._ovolt_spin, self._settings_btn, self._console_in):
            b.setEnabled(bool(on))
        # override takes the output, so the lock is disabled while it is on
        self._lock_btn.setEnabled(bool(on) and not override_on)
        self._lock_btn.setToolTip("disabled while override is on" if override_on else
                                  self._lock_btn.toolTip())

    def _send_ovolt(self) -> None:
        value = self._ovolt_spin.value()
        self._run(f"ovolt {value:g} V", lambda: self._client.set_value("ovolt", f"{value:g}"))

    def _say(self, text: str, error: bool = False) -> None:
        self._msg.setText(text)
        self._msg.setVisible(bool(text))
        set_style_if_changed(self._msg, "QLabel { color: %s; font-size: 11px; }"
                             % ("#b22222" if error else "#444"))

    def _on_console_toggled(self, on: bool) -> None:
        self._console.setVisible(on)
        self._console_in.setVisible(on)

    # ------------------------------------------------------------------ #
    # Actions (worker thread; reply on the GUI thread)
    # ------------------------------------------------------------------ #

    def _run(self, what: str, fn: Callable[[], dict], on_ok: Optional[Callable[[dict], str]] = None):
        if self._client is None:
            return

        def done(reply):
            if not isinstance(reply, dict):
                self._say(f"{what}: no reply from the server", error=True)
                return
            if reply.get("status") == "ok":
                self._say(on_ok(reply) if on_ok else f"{what}: done")
            else:
                self._say(f"{what}: {reply.get('message') or reply.get('status')}", error=True)

        self._say(f"{what}…")
        self._runner.call(fn, done)

    def _on_lock_clicked(self) -> None:
        on = (self._snap.get("settings") or {}).get("lock") == 1
        if on:
            self._run("lock off", lambda: self._client.lock(False))
        else:
            self._run("lock on", lambda: self._client.lock(True))

    def _on_reset_setpoint(self) -> None:
        row = self._snap.get("latest") or {}
        old = (self._snap.get("settings") or {}).get("setp")
        if not self._confirm("Reset the setpoint?",
                             f"Set setp to the separation measured now (about "
                             f"{_fmt(row.get('sep'), '.3f', 'ms')}; currently "
                             f"{_fmt(old, '.3f', 'ms')}).\n\nThe lock then holds the laser where "
                             "it is."):
            return
        self._run("reset setpoint", lambda: self._client.reset_setpoint(),
                  lambda r: f"setpoint {_fmt(r.get('old'), '.3f')} → {_fmt(r.get('new'), '.3f')} ms")

    def _on_clear(self) -> None:
        self._run("clear integrator", lambda: self._client.clear())

    def _on_override_clicked(self) -> None:
        on = (self._snap.get("settings") or {}).get("override") == 1
        self._run("override off" if on else "override on",
                  lambda: self._client.set_value("override", "0" if on else "1"))

    def _on_settings(self) -> None:
        dlg = SettingsDialog(self._snap.get("settings") or {}, self._snap.get("missing_settings") or [],
                             self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.apply_settings(dlg.changes())

    def apply_settings(self, changes: dict) -> None:
        if not changes:
            self._say("settings: nothing changed")
            return
        listing = "\n".join(f"  {k} = {v}" for k, v in changes.items())
        if not self._confirm("Apply settings?", f"Send to the lock:\n{listing}"):
            return
        client = self._client

        def send_all():
            done, failed = [], []
            for name, value in changes.items():
                reply = client.set_value(name, value)
                (done if reply.get("status") == "ok" else failed).append(
                    f"{name} = {reply.get('value')}" if reply.get("status") == "ok"
                    else f"{name}: {reply.get('message')}")
            return {"status": "ok" if not failed else "error",
                    "message": "; ".join(failed), "done": done}

        self._run("settings", send_all, lambda r: "set " + ", ".join(r["done"]))

    def _on_console_send(self) -> None:
        line = self._console_in.text().strip()
        if not line:
            return
        self._console_in.remember(line)
        self._console_in.clear()
        self._console.appendPlainText(f"> {line}")
        self._run(f"'{line}'", lambda: self._client.raw(line),
                  lambda r: f"'{line}': {len(r.get('lines', []))} line(s) back")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        try:
            if self._poller is not None:
                self._poller.stop()
            self._runner.shutdown(timeout=1.0)
        except Exception:
            pass
        auto_cleanup_timers(self)
