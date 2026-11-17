"""IntensityPIDPanel: dashboard panel for a two-loop intensity PID server.

Laid out for a small dock: a header (state pill; Hold / Release, Clear,
Settings…), a grid of the measured and set voltages of both loops with their
difference and the output, a grey line with the gains and the line rate, the
server's warnings, and one plot of measured (solid) and set (dashed) for both
loops, loop 1 on the left axis (upper band) and loop 2 on the right (lower band,
a quarter of the plot height below; each band spans at least 0.1 V), 100 ms means, over the
last ``history_s`` seconds (60 by default; the box in the header changes it).  Settings… edits p1 i1 p2 i2 (asks before sending); the server
remembers them and re-applies them if the board resets.

Construction never blocks: the client is discovered on a worker thread, polls
and actions run off the GUI thread.  Machine-agnostic: the host passes the
server id, title and loop names.
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
    QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from waxx.util.dashboard.embed_helpers import WidgetPanelBase, auto_cleanup_timers
from waxx.util.guis.intensity_pid import intensity_pid_protocol as proto
from waxx.util.guis.qt_upkeep import set_style_if_changed
from waxx.util.guis.request_runner import RequestRunner

__all__ = ["IntensityPIDPanel", "GainsDialog"]

_LOG = logging.getLogger("waxx.intensity_pid.panel")

HISTORY_S_DEFAULT = 60.0   # s: the plot shows the last this many seconds
HISTORY_S_MAX = 3600
HISTORY_POINTS = HISTORY_S_MAX * 10 + 50    # 100 ms means
POLL_INTERVAL_MS = 500

_PILL = {
    "ACTIVE": ("#2e8b57", "ACTIVE"),
    "HELD": ("#b8860b", "HELD"),
    "OVERRIDE": ("#b8860b", "OVERRIDE"),
    "NO DATA": ("#b22222", "NO DATA"),
    "NO LINK": ("#b22222", "NO LINK"),
    "CONNECTING": ("#888888", "connecting…"),
}
_LEVEL_COLOR = {"error": "#b22222", "warn": "#b8860b", "info": "#555555"}
_MONO = "font-family: Consolas, monospace;"
_NUM_CSS = _MONO + " font-size: 12px;"
_BIG_CSS = _MONO + " font-size: 13px; font-weight: 700;"
_HEAD_CSS = "QLabel { color: #777; font-size: 10px; }"
_BTN_STYLE = "QPushButton { padding: 2px 8px; font-size: 11px; }"
_HELD_BTN_STYLE = ("QPushButton { padding: 2px 8px; font-size: 11px; background-color: #b8860b;"
                   " color: white; font-weight: 600; }")
_LOOP_COLORS = ("#1f77b4", "#d2691e")    # loop 1 left axis, loop 2 right axis
# y layout of the twin axes, as fractions of the plot height from the bottom: loop 1's lines
# fill the upper band, loop 2's the lower, a quarter of the height apart (top margin for the legend)
PLOT_MIN_SPAN_V = 0.025       # a band shows at least this many volts (flat lines stay flat)
_BAND_GAP = 0.05
_BAND_MARGIN_TOP, _BAND_MARGIN_BOTTOM = 0.05, 0.05
_BAND_H = (1.0 - _BAND_GAP - _BAND_MARGIN_TOP - _BAND_MARGIN_BOTTOM) / 2
_BANDS = {1: (_BAND_MARGIN_BOTTOM + _BAND_H + _BAND_GAP, 1.0 - _BAND_MARGIN_TOP),
          2: (_BAND_MARGIN_BOTTOM, _BAND_MARGIN_BOTTOM + _BAND_H)}


def band_y_range(lo: float, hi: float, band: tuple[float, float],
                 min_span: float = PLOT_MIN_SPAN_V) -> tuple[float, float]:
    """The view y range that puts data in [lo, hi] (widened about its middle to at
    least ``min_span``) into ``band`` (fractions of the plot height from the bottom)."""
    mid, span = 0.5 * (lo + hi), max(hi - lo, min_span)
    f_lo, f_hi = band
    height = span / (f_hi - f_lo)
    y0 = (mid - 0.5 * span) - f_lo * height
    return y0, y0 + height


def _ask(parent, title: str, text: str) -> bool:
    return QMessageBox.question(parent, title, text,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes


def _num(v) -> Optional[float]:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _fmt(v, spec: str = ".4f") -> str:
    x = _num(v)
    return "--" if x is None else f"{x:{spec}}"


class _PollSource:
    """What the poller calls (pool thread): the snapshot plus the 100 ms means
    that are new since the last poll."""

    def __init__(self, client):
        self.client = client
        self.timeout = getattr(client, "timeout", None)
        self.seq = 0

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


class GainsDialog(QDialog):
    """Edit p1 i1 p2 i2; ``changes()`` is {name: text} of the edited ones."""

    def __init__(self, settings: dict, loop_names=("loop 1", "loop 2"), parent=None):
        super().__init__(parent)
        self.setWindowTitle("PID gains")
        self._initial = dict(settings)
        self._edits: dict[str, QLineEdit] = {}
        lay = QVBoxLayout(self)
        note = QLabel("Only edited values are sent. The server remembers them and sends them "
                      "again if the board resets or is reflashed (with a warning).", self)
        note.setWordWrap(True)
        note.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        lay.addWidget(note)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for col, head in enumerate(("", "P", "I (per sample)")):
            h = QLabel(head, self)
            h.setStyleSheet(_HEAD_CSS)
            grid.addWidget(h, 0, col)
        for loop in (1, 2):
            grid.addWidget(QLabel(loop_names[loop - 1], self), loop, 0)
            for col, kind in ((1, "p"), (2, "i")):
                name = f"{kind}{loop}"
                w = QLineEdit(self)
                v = settings.get(name)
                w.setText("" if v is None else f"{v:g}")
                w.setToolTip(proto.SETTABLE[name].help)
                w.setFixedWidth(90)
                grid.addWidget(w, loop, col)
                self._edits[name] = w
        lay.addLayout(grid)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply
                               | QDialogButtonBox.StandardButton.Cancel, self)
        apply_btn = box.button(QDialogButtonBox.StandardButton.Apply)
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def edit(self, name: str) -> QLineEdit:
        return self._edits[name]

    def changes(self) -> dict:
        out = {}
        for name, w in self._edits.items():
            text = w.text().strip()
            if not text:
                continue
            old = self._initial.get(name)
            try:
                same = old is not None and math.isclose(float(text), float(old), rel_tol=1e-9,
                                                        abs_tol=1e-12)
            except ValueError:
                same = False
            if not same:
                out[name] = text
        return out


class IntensityPIDPanel(WidgetPanelBase):
    """The intensity PID panel (module docstring)."""

    def __init__(self, server_id: str, *, title: str = "Intensity PID",
                 loop_names: tuple[str, str] = ("loop 1", "loop 2"),
                 client: Any = None, client_factory: Optional[Callable[[], Any]] = None,
                 confirm: Optional[Callable[[str, str], bool]] = None,
                 synchronous: bool = False, plots: bool = True,
                 history_s: float = HISTORY_S_DEFAULT, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._server_id = server_id
        self._title = title
        self._history_s = float(history_s)
        self._loop_names = tuple(loop_names)
        self._client = None
        self._poller = None
        self._source: Optional[_PollSource] = None
        self._confirm = confirm or (lambda t, txt: _ask(self, t, txt))
        self._runner = RequestRunner(None, synchronous=synchronous, parent=self)
        self._snap: dict = {}
        self._hist = {k: collections.deque(maxlen=HISTORY_POINTS)
                      for k in ("t", "meas1", "set1", "meas2", "set2")}
        self._plots: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(3)

        # header: pill + actions
        top = QHBoxLayout()
        top.setSpacing(4)
        self._pill = QLabel(self)
        self._pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pill.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        top.addWidget(self._pill, 0)
        top.addStretch(1)
        if plots:
            spin = QSpinBox(self)
            spin.setRange(5, HISTORY_S_MAX)
            spin.setSingleStep(30)
            spin.setValue(int(round(self._history_s)))
            spin.setSuffix(" s")
            spin.setToolTip("plot window: the last this many seconds")
            spin.setFixedWidth(66)
            spin.setStyleSheet("QSpinBox { font-size: 11px; padding: 0px 2px; }")
            spin.valueChanged.connect(lambda v: self._set_history_s(float(v)))
            self._window_spin = spin
            top.addWidget(spin, 0)
        self._hold_btn = QPushButton("Hold", self)
        self._hold_btn.clicked.connect(self._on_hold_clicked)
        self._clear_btn = QPushButton("Clear", self)
        self._clear_btn.setToolTip("zero both integrators")
        self._clear_btn.clicked.connect(self._on_clear)
        self._settings_btn = QPushButton("Settings…", self)
        self._settings_btn.setToolTip("P and I gains of both loops")
        self._settings_btn.clicked.connect(self._on_settings)
        for b in (self._hold_btn, self._clear_btn, self._settings_btn):
            b.setStyleSheet(_BTN_STYLE)
            top.addWidget(b, 0)
        outer.addLayout(top)

        # readings
        grid = QGridLayout()
        grid.setContentsMargins(2, 0, 2, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(0)
        heads = (("meas", "photodiode voltage"), ("set", "setpoint voltage (analog input)"),
                 ("meas − set", "error the loop corrects"), ("out", "control voltage sent out"))
        for c, (text, tip) in enumerate(heads, start=1):
            h = QLabel(text + (" (V)" if c == 1 else ""), self)
            h.setStyleSheet(_HEAD_CSS)
            h.setToolTip(tip)
            h.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(h, 0, c)
        self._cells: dict[str, QLabel] = {}
        for loop in (1, 2):
            name = QLabel(self._loop_names[loop - 1], self)
            name.setStyleSheet("QLabel { font-size: 11px; font-weight: 600; }")
            grid.addWidget(name, loop, 0)
            for c, key in enumerate((f"meas{loop}", f"set{loop}", f"err{loop}", f"out{loop}"), start=1):
                lab = QLabel("--", self)
                lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                lab.setStyleSheet("QLabel { " + (_BIG_CSS if c == 1 else _NUM_CSS) + " }")
                lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                lab.setToolTip("mean of the last 100 ms of status lines")
                grid.addWidget(lab, loop, c)
                self._cells[key] = lab
        grid.setColumnStretch(0, 0)
        for c in range(1, 5):
            grid.setColumnStretch(c, 1)
        outer.addLayout(grid)

        # override: per loop, a button (on / off) and the voltage
        orow = QHBoxLayout()
        orow.setSpacing(4)
        olabel = QLabel("override", self)
        olabel.setStyleSheet(_HEAD_CSS)
        olabel.setToolTip("force an output to a voltage instead of the PID")
        orow.addWidget(olabel)
        self._ov_btns: dict[int, QPushButton] = {}
        self._ov_spins: dict[int, QDoubleSpinBox] = {}
        self._ov_timers: dict[int, QTimer] = {}
        for loop in (1, 2):
            btn = QPushButton(self._loop_names[loop - 1], self)
            btn.setStyleSheet(_BTN_STYLE)
            btn.clicked.connect(lambda _c=False, n=loop: self._on_override_clicked(n))
            spin = QDoubleSpinBox(self)
            spin.setRange(0.0, 9.9)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setSuffix(" V")
            spin.setFixedWidth(76)
            spin.setStyleSheet("QDoubleSpinBox { font-size: 11px; padding: 0px 2px; }")
            spin.setToolTip(f"{self._loop_names[loop - 1]} output while overridden")
            # arrows and typing are sent after a short pause, so one click is one command
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(400)
            timer.timeout.connect(lambda n=loop: self._send_ovolt(n))
            spin.valueChanged.connect(lambda _v, t=timer: t.start())
            orow.addWidget(btn)
            orow.addWidget(spin)
            if loop == 1:
                orow.addSpacing(8)
            self._ov_btns[loop], self._ov_spins[loop], self._ov_timers[loop] = btn, spin, timer
        orow.addStretch(1)
        outer.addLayout(orow)

        self._gains_line = QLabel("", self)
        self._gains_line.setStyleSheet("QLabel { color: #777; font-size: 10px; " + _MONO + " }")
        self._gains_line.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self._gains_line)

        self._warn = QLabel("", self)
        self._warn.setWordWrap(True)
        self._warn.setTextFormat(Qt.TextFormat.RichText)
        self._warn.setStyleSheet("QLabel { font-size: 11px; }")
        self._warn.hide()
        outer.addWidget(self._warn)

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
        else:
            outer.addStretch(1)

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
        from waxx.util.guis.intensity_pid.intensity_pid_client import IntensityPIDClient  # noqa: PLC0415
        return IntensityPIDClient(self._server_id, discovery_timeout=3.0)

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

        class _Poller(SnapshotPoller):
            NORMAL_INTERVAL_MS = POLL_INTERVAL_MS

        self._poller = _Poller(self._source, panel_id=self._server_id, parent=self)
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
        row = snap.get("mean") or snap.get("latest") or {}
        settings = snap.get("settings") or {}
        com = snap.get("com") or {}
        warnings = snap.get("warnings") or []
        codes = {w.get("code") for w in warnings}
        held = snap.get("held")
        limits = snap.get("limits") or {}
        connected = com.get("status") == "connected"

        if not connected:
            self._set_pill("NO LINK", f"COM {com.get('port', '?')}: {com.get('status', '?')}")
        elif codes & {"no_data", "stale"}:
            self._set_pill("NO DATA")
        elif held:
            self._set_pill("HELD")
        elif any(settings.get(f"override{i}") == 1 for i in (1, 2)):
            self._set_pill("OVERRIDE")
        else:
            self._set_pill("ACTIVE")
        label = snap.get("label") or ""
        self._pill.setToolTip(f"{self._title} · {com.get('port', '')}" + (f" · board {label}" if label else ""))

        # the pill and the override row say these
        hide_hold_warning = [w for w in warnings
                             if w.get("code") not in ("hold", "override1", "override2")]
        if hide_hold_warning:
            self._warn.setText("<br>".join(
                f"<span style='color:{_LEVEL_COLOR.get(w['level'], '#555')}'>"
                f"{'⚠ ' if w['level'] != 'info' else ''}{html.escape(w['text'])}</span>"
                for w in hide_hold_warning))
            self._warn.show()
        else:
            self._warn.hide()

        for loop in (1, 2):
            meas, setp, out = row.get(f"meas{loop}"), row.get(f"set{loop}"), row.get(f"out{loop}")
            self._cells[f"meas{loop}"].setText(_fmt(meas))
            self._cells[f"set{loop}"].setText(_fmt(setp))
            m, s = _num(meas), _num(setp)
            self._cells[f"err{loop}"].setText("--" if m is None or s is None else f"{m - s:+.4f}")
            self._cells[f"out{loop}"].setText(_fmt(out, ".3f"))
            o = _num(out)
            overridden = settings.get(f"override{loop}") == 1
            at_rail = (not held and o is not None
                       and (o >= limits.get("high", 9.9) or o <= limits.get("low", 0.0)))
            set_style_if_changed(self._cells[f"out{loop}"], "QLabel { " + _NUM_CSS
                                 + (" color: #b8860b; font-weight: 700;"
                                    if at_rail or overridden else "") + " }")
            btn, spin = self._ov_btns[loop], self._ov_spins[loop]
            btn.setText(f"{self._loop_names[loop - 1]} ON" if overridden else self._loop_names[loop - 1])
            btn.setToolTip(f"{self._loop_names[loop - 1]}: output forced to the voltage beside it; "
                           "click to give it back to the PID" if overridden else
                           f"force the {self._loop_names[loop - 1]} output to the voltage beside it")
            set_style_if_changed(btn, _HELD_BTN_STYLE if overridden else _BTN_STYLE)
            # the box follows the board, except while it is being edited or a value is pending
            ov = _num(settings.get(f"ovolt{loop}"))
            if ov is not None and not self._ov_timers[loop].isActive() and not spin.hasFocus():
                spin.blockSignals(True)
                spin.setValue(ov)
                spin.blockSignals(False)
            sd = _num(row.get(f"meas{loop}_sd"))
            self._cells[f"meas{loop}"].setToolTip(
                "photodiode, mean of the last 100 ms of status lines"
                + ("" if sd is None else f" (SD {sd * 1e3:.2f} mV over {row.get('n')} lines)"))

        rate = _num(snap.get("rate_hz"))
        self._gains_line.setText(
            "  ".join(f"{k} {_fmt(settings.get(k), 'g')}" for k in ("p1", "i1"))
            + "  ·  " + "  ".join(f"{k} {_fmt(settings.get(k), 'g')}" for k in ("p2", "i2"))
            + ("" if rate is None else f"  ·  {rate:.0f} lines/s"))

        self._hold_btn.setText("Release" if held else "Hold")
        self._hold_btn.setToolTip("let both loops run again (each integrator restarts at its held "
                                  "output)" if held else
                                  "fix both outputs at their current values; the loops stop correcting")
        set_style_if_changed(self._hold_btn, _HELD_BTN_STYLE if held else _BTN_STYLE)
        self._set_controls_enabled(connected)

        for r in snap.get("_new_rows", []):
            self._append_history(r)
        self._redraw()

    def _append_history(self, r: dict) -> None:
        h = self._hist
        h["t"].append(r.get("t") or time.time())
        for k in ("meas1", "set1", "meas2", "set2"):
            v = _num(r.get(k))
            h[k].append(float("nan") if v is None else v)

    def _set_history_s(self, seconds: float) -> None:
        """Show the last ``seconds`` of history in the plot."""
        self._history_s = float(seconds)
        self._redraw()

    def _redraw(self) -> None:
        if not self._plots:
            return
        self._plots["plot"].setXRange(-self._history_s, 0, padding=0)
        if not self._hist["t"]:
            return
        import numpy as np  # noqa: PLC0415
        t = np.asarray(self._hist["t"]) - time.time()
        keep = t >= -self._history_s
        shown = {}
        for key, curve in self._plots["curves"].items():
            shown[key] = np.asarray(self._hist[key], dtype=float)[keep]
            curve.setData(t[keep], shown[key], connect="finite")
        for loop, vb in ((1, self._plots["plot"].vb), (2, self._plots["right_vb"])):
            y = np.concatenate((shown[f"meas{loop}"], shown[f"set{loop}"]))
            y = y[np.isfinite(y)]
            if y.size:
                vb.setYRange(*band_y_range(float(y.min()), float(y.max()), _BANDS[loop]), padding=0)

    def _init_plots(self) -> None:
        """One plot: loop 1 on the left axis, loop 2 on the right (its own
        view box); solid = meas, dashed = set, one colour per loop.  ``_redraw``
        sets both y ranges (loop 1 upper band, loop 2 lower band, ``_BANDS``), so
        the mouse does not zoom y."""
        try:
            import pyqtgraph as pg  # noqa: PLC0415
        except Exception:
            self._plot_box.addWidget(QLabel("pyqtgraph not available"))
            return
        lay = pg.GraphicsLayoutWidget()
        lay.setBackground("w")
        lay.setMinimumHeight(110)
        lay.ci.setContentsMargins(0, 2, 0, 0)
        lay.setToolTip("100 ms means; time axis s, 0 = now; solid = meas, dashed = set; "
                       f"{self._loop_names[0]} left axis, {self._loop_names[1]} right axis")
        font = self.font()
        font.setPointSize(7)
        plot = lay.addPlot(row=0, col=0)
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.setXRange(-self._history_s, 0, padding=0)
        plot.showAxis("right")
        right_vb = pg.ViewBox()
        right_vb.setMouseEnabled(x=False, y=False)
        plot.scene().addItem(right_vb)
        plot.getAxis("right").linkToView(right_vb)
        right_vb.setXLink(plot)

        def _sync():
            right_vb.setGeometry(plot.vb.sceneBoundingRect())
            right_vb.linkedViewChanged(plot.vb, right_vb.XAxis)

        plot.vb.sigResized.connect(_sync)
        for ax, color in (("left", _LOOP_COLORS[0]), ("right", _LOOP_COLORS[1]), ("bottom", "#555")):
            axis = plot.getAxis(ax)
            axis.setPen(pg.mkPen(color))
            axis.setTextPen(pg.mkPen(color))
            try:
                axis.setTickFont(font)
            except AttributeError:
                pass
        plot.getAxis("left").setWidth(42)
        plot.getAxis("right").setWidth(42)
        plot.getAxis("right").setGrid(False)        # one grid (the left axis's) is enough
        plot.getAxis("bottom").setHeight(16)
        legend = plot.addLegend(offset=(2, 2), labelTextSize="7pt", verSpacing=-6)
        curves = {}
        for loop, vb in ((1, plot.vb), (2, right_vb)):
            color = _LOOP_COLORS[loop - 1]
            set_curve = pg.PlotDataItem(pen=pg.mkPen(color, width=1.0, style=Qt.PenStyle.DashLine))
            meas_curve = pg.PlotDataItem(pen=pg.mkPen(color, width=1.3))
            vb.addItem(set_curve)
            vb.addItem(meas_curve)
            legend.addItem(meas_curve, self._loop_names[loop - 1])
            curves[f"set{loop}"] = set_curve
            curves[f"meas{loop}"] = meas_curve
        self._plots = {"layout": lay, "plot": plot, "right_vb": right_vb, "curves": curves,
                       "sync": _sync}
        self._plot_box.addWidget(lay)
        _sync()

    def _set_pill(self, key: str, detail: str = "") -> None:
        color, text = _PILL.get(key, ("#888888", key))
        self._pill.setText(text)
        self._pill.setToolTip(text + (f" — {detail}" if detail else ""))
        if detail:
            self._warn.setText(f"<span style='color:{color}'>⚠ {text}: {html.escape(detail)}</span>")
            self._warn.show()
        set_style_if_changed(self._pill, f"QLabel {{ background-color: {color}; color: white;"
                                         " font-size: 12px; font-weight: 700; padding: 1px 8px;"
                                         " border-radius: 4px; }")

    def _set_controls_enabled(self, on: bool) -> None:
        held = bool(self._snap.get("held"))
        self._hold_btn.setEnabled(bool(on))
        self._settings_btn.setEnabled(bool(on))
        for w in (*self._ov_btns.values(), *self._ov_spins.values()):
            w.setEnabled(bool(on))
        # while held a clear would be overwritten by the release (bumpless restart)
        self._clear_btn.setEnabled(bool(on) and not held)
        self._clear_btn.setToolTip("release first: on release each integrator restarts at its held "
                                   "output" if held else "zero both integrators")

    def _say(self, text: str, error: bool = False) -> None:
        self._msg.setText(text)
        self._msg.setVisible(bool(text))
        set_style_if_changed(self._msg, "QLabel { color: %s; font-size: 11px; }"
                             % ("#b22222" if error else "#444"))

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

    def _on_hold_clicked(self) -> None:
        held = bool(self._snap.get("held"))
        # explicit on/off, not a toggle: a double click cannot invert it
        self._run("release" if held else "hold", lambda: self._client.hold(not held),
                  lambda r: "held: both outputs fixed" if r.get("on") else "released: loops running")

    def _on_override_clicked(self, loop: int) -> None:
        on = (self._snap.get("settings") or {}).get(f"override{loop}") == 1
        name = self._loop_names[loop - 1]
        if on:
            self._run(f"{name} override off", lambda: self._client.override(loop, False),
                      lambda r: f"{name}: the PID drives the output again")
        else:
            self._ov_timers[loop].stop()
            volt = self._ov_spins[loop].value()
            self._run(f"{name} override on", lambda: self._client.override(loop, True, volt),
                      lambda r: f"{name}: output forced to {_fmt(r.get('volt'), '.3f')} V")

    def _send_ovolt(self, loop: int) -> None:
        volt = self._ov_spins[loop].value()
        self._run(f"ovolt{loop} {volt:g} V", lambda: self._client.set_value(f"ovolt{loop}", volt))

    def _on_clear(self) -> None:
        self._run("clear", lambda: self._client.clear(),
                  lambda r: r.get("message") or "integrators cleared")

    def _on_settings(self) -> None:
        dlg = GainsDialog(self._snap.get("settings") or {}, self._loop_names, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.apply_settings(dlg.changes())

    def apply_settings(self, changes: dict) -> None:
        if not changes:
            self._say("gains: nothing changed")
            return
        for name, value in changes.items():
            try:
                proto.format_value(proto.SETTABLE[name], value)
            except (ValueError, KeyError):
                self._say(f"gains: {name} = {value!r} is not a number; nothing sent", error=True)
                return
        listing = "\n".join(f"  {k} = {v}" for k, v in changes.items())
        if not self._confirm("Apply gains?", f"Send to the board:\n{listing}"):
            return
        client = self._client

        def send_all():
            done, failed = [], []
            for name, value in changes.items():
                reply = client.set_value(name, value)
                if reply.get("status") == "ok":
                    done.append(f"{name} = {reply.get('value')}")
                else:
                    failed.append(f"{name}: {reply.get('message')}")
            return {"status": "ok" if not failed else "error",
                    "message": "; ".join(failed), "done": done}

        self._run("gains", send_all, lambda r: "set " + ", ".join(r["done"]))

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
