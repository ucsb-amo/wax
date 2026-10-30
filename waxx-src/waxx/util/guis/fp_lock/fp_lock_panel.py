"""FPLockPanel: dashboard panel for the FP peak-separation lock server.

Shows the lock state, the peak separation against its setpoint, the server's warnings (first peak near the sweep start,
second peak near the sweep end, HOLD, rail, no data, COM), the latest
readings, and plots of the error, the output and the two peak positions with
the edge bands shaded.  Controls: lock on / off, reset setpoint (setp = the
measured separation), clear integrator, a settings dialog (gains, window,
peak finding), and a console for raw firmware commands.  Every control that
moves the laser asks first.

Construction never blocks: the client is discovered on a worker thread, polls
and actions run off the GUI thread (SnapshotPoller, RequestRunner).
Machine-agnostic: the host passes the server id.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from waxx.util.dashboard.embed_helpers import WidgetPanelBase, auto_cleanup_timers
from waxx.util.dashboard.widgets import CollapsibleGroupBox
from waxx.util.guis.fp_lock import fp_lock_protocol as proto
from waxx.util.guis.qt_upkeep import set_style_if_changed
from waxx.util.guis.request_runner import RequestRunner

__all__ = ["FPLockPanel", "SettingsDialog"]

_LOG = logging.getLogger("waxx.fp_lock.panel")

HISTORY_POINTS = 3000

_PILL = {
    "LOCKED": ("#2e8b57", "LOCKED"),
    "HOLD": ("#d4a017", "HOLD"),
    "OFF": ("#777777", "LOCK OFF"),
    "NO DATA": ("#b22222", "NO DATA"),
    "NO LINK": ("#b22222", "NO LINK"),
    "CONNECTING": ("#888888", "connecting…"),
}
_LEVEL_COLOR = {"error": "#b22222", "warn": "#b8860b", "info": "#555555"}


def _ask(parent, title: str, text: str) -> bool:
    return QMessageBox.question(parent, title, text,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes


def _fmt(v, spec: str = ".3f", unit: str = "") -> str:
    if not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    return f"{v:{spec}}{(' ' + unit) if unit else ''}"


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
        box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
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
                 synchronous: bool = False, plots: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._server_id = server_id
        self._title = title
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        top = QHBoxLayout()
        self._pill = QLabel(self)
        self._pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pill.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        top.addWidget(self._pill, 0)
        self._sep_big = QLabel("sep --", self)
        self._sep_big.setToolTip("peak separation (latest) and its setpoint")
        self._sep_big.setStyleSheet("QLabel { font-family: Consolas, monospace; font-size: 15px;"
                                    " font-weight: 700; padding: 0 8px; }")
        top.addWidget(self._sep_big, 0)
        self._board = QLabel("", self)
        self._board.setStyleSheet("QLabel { color: #666; }")
        top.addWidget(self._board, 1)
        outer.addLayout(top)

        self._warn = QLabel("", self)
        self._warn.setWordWrap(True)
        self._warn.setTextFormat(Qt.TextFormat.RichText)
        self._warn.hide()
        outer.addWidget(self._warn)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(1)
        self._vals: dict[str, QLabel] = {}
        cells = (("sep", "sep"), ("setp", "setp"), ("err", "err"),
                 ("feedback", "output"), ("integ", "integ"), ("rail", "rail"),
                 ("tfirst", "1st peak"), ("t2", "2nd peak"), ("period", "period"),
                 ("h1", "h1"), ("h2", "h2"), ("bad", "bad"),
                 ("why", "skipped"), ("unlocks", "unlocks"), ("sats", "rail hits"))
        for i, (key, label) in enumerate(cells):
            r, c = divmod(i, 3)
            name = QLabel(label, self)
            name.setStyleSheet("QLabel { color: #666; }")
            val = QLabel("--", self)
            val.setStyleSheet("QLabel { font-family: Consolas, monospace; }")
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(name, r, 2 * c)
            grid.addWidget(val, r, 2 * c + 1)
            self._vals[key] = val
        outer.addLayout(grid)

        buttons = QHBoxLayout()
        self._lock_btn = QPushButton("Lock …", self)
        self._lock_btn.clicked.connect(self._on_lock_clicked)
        self._reset_btn = QPushButton("Reset setpoint", self)
        self._reset_btn.setToolTip("setp = the separation the lock measures now")
        self._reset_btn.clicked.connect(self._on_reset_setpoint)
        self._clear_btn = QPushButton("Clear integrator", self)
        self._clear_btn.setToolTip("zero the integral and the output (0 V); the state is kept")
        self._clear_btn.clicked.connect(self._on_clear)
        self._settings_btn = QPushButton("Settings…", self)
        self._settings_btn.clicked.connect(self._on_settings)
        for b in (self._lock_btn, self._reset_btn, self._clear_btn, self._settings_btn):
            buttons.addWidget(b)
        outer.addLayout(buttons)

        self._msg = QLabel("", self)
        self._msg.setWordWrap(True)
        self._msg.setStyleSheet("QLabel { color: #444; }")
        outer.addWidget(self._msg)

        if plots:
            self._plot_group = CollapsibleGroupBox("Separation · error · output · peak positions",
                                                   expanded=True)
            outer.addWidget(self._plot_group, 1)
            self._init_plots()

        self._console_group = CollapsibleGroupBox("Console (raw firmware commands)", expanded=False)
        self._console = QPlainTextEdit(self)
        self._console.setReadOnly(True)
        self._console.setMaximumBlockCount(500)
        self._console.setStyleSheet("QPlainTextEdit { font-family: Consolas, monospace; }")
        self._console_in = QLineEdit(self)
        self._console_in.setPlaceholderText("command, e.g. help or tmin 1.5 (Enter sends)")
        self._console_in.returnPressed.connect(self._on_console_send)
        self._console_group.addWidget(self._console)
        self._console_group.addWidget(self._console_in)
        outer.addWidget(self._console_group)

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
        else:
            self._set_pill(row.get("state_name") or "CONNECTING")
        label = snap.get("label") or ""
        self._board.setText(f"{self._title} · {com.get('port', '')}" + (f" · {label}" if label else ""))

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
        v["sep"].setText(_fmt(row.get("sep"), ".3f", "ms"))
        v["setp"].setText(_fmt(settings.get("setp"), ".3f", "ms"))
        v["err"].setText(_fmt(row.get("err"), "+.3f", "ms"))
        v["feedback"].setText(_fmt(row.get("feedback"), "+.4f", "V"))
        v["integ"].setText(_fmt(row.get("integ"), "+.4f", "V"))
        v["rail"].setText("AT RAIL" if row.get("saturated") else ("-" if row else "--"))
        v["tfirst"].setText(_fmt(row.get("tfirst"), ".2f", "ms"))
        v["t2"].setText(_fmt(row.get("t2"), ".2f", "ms"))
        v["period"].setText(_fmt(row.get("period"), ".1f", "ms"))
        v["h1"].setText(_fmt(row.get("h1"), ".2f", "V"))
        v["h2"].setText(_fmt(row.get("h2"), ".2f", "V"))
        v["bad"].setText(_fmt(row.get("bad"), "d"))
        if "bad_none" in row:
            v["why"].setText(f"{row['bad_none']} none / {row['bad_one']} 1pk / {row['bad_win']} win")
        v["unlocks"].setText(_fmt(row.get("unlocks"), "d"))
        v["sats"].setText(_fmt(row.get("sats"), "d"))
        edge = snap.get("edge") or {}
        self._tint(v["tfirst"], "first_near_start" in codes)
        self._tint(v["t2"], "second_near_end" in codes)

        lock_on = settings.get("lock")
        self._lock_btn.setText("Lock ON — turn off…" if lock_on == 1 else
                               ("Lock OFF — turn on" if lock_on == 0 else "Lock …"))
        set_style_if_changed(self._lock_btn,
                             "QPushButton { background-color: #2e8b57; color: white; font-weight: 600; }"
                             if lock_on == 1 else "")
        self._set_controls_enabled(com.get("status") == "connected")

        for row_ in snap.get("_new_rows", []):
            self._append_history(row_)
        self._redraw(edge, settings.get("setp"))
        for line in snap.get("_new_text", []):
            self._console.appendPlainText(line)

    def _append_history(self, r: dict) -> None:
        h = self._hist
        h["t"].append(r.get("t", time.time()))
        for k in ("sep", "err", "feedback", "integ", "tfirst", "t2"):
            val = r.get(k)
            h[k].append(float(val) if isinstance(val, (int, float)) else float("nan"))

    def _redraw(self, edge: dict, setp=None) -> None:
        if not self._plots or not self._hist["t"]:
            return
        import numpy as np  # noqa: PLC0415
        t = np.asarray(self._hist["t"]) - time.time()
        for key, curve in self._plots["curves"].items():
            curve.setData(t, np.asarray(self._hist[key], dtype=float), connect="finite")
        if isinstance(setp, (int, float)) and math.isfinite(setp):
            self._plots["setp"].setValue(setp)
            self._plots["setp"].show()
        else:
            self._plots["setp"].hide()
        right, margin = edge.get("right_edge_ms"), edge.get("margin_ms")
        if isinstance(right, (int, float)) and isinstance(margin, (int, float)):
            lo, hi = self._bands
            lo.setRegion((0.0, margin))
            hi.setRegion((right - margin, right))
            self._plots["pos"].setYRange(0, right, padding=0.02)

    def _init_plots(self) -> None:
        try:
            import pyqtgraph as pg  # noqa: PLC0415
        except Exception:
            self._plot_group.addWidget(QLabel("pyqtgraph not available"))
            return
        lay = pg.GraphicsLayoutWidget()
        lay.setBackground("w")
        p_sep = lay.addPlot(row=0, col=0)
        p_err = lay.addPlot(row=1, col=0)
        p_out = lay.addPlot(row=2, col=0)
        p_pos = lay.addPlot(row=3, col=0)
        for p, label in ((p_sep, "sep (ms)"), (p_err, "err (ms)"), (p_out, "output (V)"),
                         (p_pos, "peak (ms)")):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("left", label)
            p.getAxis("left").setWidth(48)
        p_err.setXLink(p_sep)
        p_out.setXLink(p_sep)
        p_pos.setXLink(p_sep)
        p_sep.addLegend(offset=(5, 5))
        p_pos.setLabel("bottom", "time (s, 0 = now)")
        p_out.addLegend(offset=(5, 5))
        p_pos.addLegend(offset=(5, 5))
        curves = {
            "sep": p_sep.plot(pen=pg.mkPen("#000000", width=1.2), name="sep"),
            "err": p_err.plot(pen=pg.mkPen("#1f77b4", width=1.2)),
            "feedback": p_out.plot(pen=pg.mkPen("#d62728", width=1.2), name="output"),
            "integ": p_out.plot(pen=pg.mkPen("#ff9896", width=1.0, style=Qt.PenStyle.DashLine),
                                name="integ"),
            "tfirst": p_pos.plot(pen=pg.mkPen("#2ca02c", width=1.2), name="1st"),
            "t2": p_pos.plot(pen=pg.mkPen("#9467bd", width=1.2), name="2nd"),
        }
        for _ in range(2):
            band = pg.LinearRegionItem(values=(0, 0), orientation="horizontal", movable=False,
                                       brush=pg.mkBrush(178, 34, 34, 40), pen=pg.mkPen(None))
            p_pos.addItem(band)
            self._bands.append(band)
        setp_line = pg.InfiniteLine(angle=0, movable=False,
                                    pen=pg.mkPen("#d62728", width=1.0, style=Qt.PenStyle.DashLine),
                                    label="setp", labelOpts={"position": 0.03, "color": "#d62728"})
        p_sep.addItem(setp_line)
        setp_line.hide()
        self._plots = {"layout": lay, "curves": curves, "pos": p_pos, "sep": p_sep,
                       "setp": setp_line}
        self._plot_group.addWidget(lay)

    def _set_pill(self, key: str, detail: str = "") -> None:
        color, text = _PILL.get(key, ("#888888", key))
        self._pill.setText(text + (f" — {detail}" if detail else ""))
        self._pill.setToolTip(self._pill.text())
        set_style_if_changed(self._pill, f"QLabel {{ background-color: {color}; color: white;"
                                         " font-size: 15px; font-weight: 700; padding: 4px 12px;"
                                         " border-radius: 4px; }")

    @staticmethod
    def _tint(label: QLabel, on: bool) -> None:
        set_style_if_changed(label, "QLabel { font-family: Consolas, monospace;"
                                    + (" color: #b8860b; font-weight: 700;" if on else "") + " }")

    def _set_controls_enabled(self, on: bool) -> None:
        for b in (self._lock_btn, self._reset_btn, self._clear_btn, self._settings_btn,
                  self._console_in):
            b.setEnabled(bool(on))

    def _say(self, text: str, error: bool = False) -> None:
        self._msg.setText(text)
        set_style_if_changed(self._msg, "QLabel { color: %s; }" % ("#b22222" if error else "#444"))

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
            if not self._confirm("Turn the lock off?",
                                 "The output goes to 0 V and the integral is cleared: the laser "
                                 "moves to where 0 V puts it."):
                return
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
        if not self._confirm("Clear the integrator?",
                             "The integral and the output go to 0 V now (the laser moves); "
                             "the lock state is kept."):
            return
        self._run("clear integrator", lambda: self._client.clear())

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
