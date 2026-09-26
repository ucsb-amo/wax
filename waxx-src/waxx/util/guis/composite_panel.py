"""Composite tab of the Device Control GUI.

Renders one card per :class:`~waxx.util.device_state.composite.CompositeDevice`
and sends its ops through the monitor server (see that module for the whole
path).  Nothing here knows about a particular machine: the lab passes its
device definitions, scenes and telemetry providers in.

What a card shows is read from the device-state JSON the Device Control GUI
already keeps up to date -- lamps, the state pill and each field's readback
-- so it describes the hardware as the monitor last set it, whichever tab (or
experiment) set it.  Measured values (telemetry: a supply's output current,
the interlock state) are separate chips next to them, dashed, never in place
of a setpoint.

Fields: a value typed but not yet applied is shown with an orange fill and
is not overwritten by incoming state; it stays so until an op that sends it
*succeeds* (Esc reverts it to the hardware value).  The wheel only changes a
field that has focus, so scrolling the tab never edits a value.

Ops are refused before sending when a value is outside its hard limits (the
reason is shown on the card, no pop-up), and confirmed -- with a button that
says what will happen -- when it is outside its soft limits, when the op asks
for a confirmation, or when it is marked dangerous.  Each card's footer says
what happened to its last op, whichever GUI sent it.

Cards are grouped (``CompositeDevice.group``), groups are laid out in balanced
columns (card_layout.MasonryLayout), and every card can be collapsed to its
header.  Every card starts collapsed, and opening or collapsing one never
moves a card to another column (the arrangement is pinned; see card_layout).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import socket
import threading
import time
from collections import deque
from typing import Any, Callable

from PyQt6.QtCore import QEvent, QObject, QSettings, QSignalBlocker, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QSizePolicy,
    QTableWidget, QToolButton, QVBoxLayout, QWidget, QWidgetAction,
)

from waxx.util.comms_server.comm_client import MonitorClient
from waxx.util.comms_server.comm_server import STATES
from waxx.util.dashboard import theme
from waxx.util.device_state import composite as cmp
from waxx.util.guis.card_layout import FlowLayout, MasonryLayout

_LOG = logging.getLogger("waxx.device_control.composite")

#: No result this long after sending: tell the operator the outcome is unknown.
#: Generous because a host step (tweezer AWG connect, with its retries) or a
#: long ramp playing out can legitimately take a while.
OP_RESULT_TIMEOUT_S = 60.0
#: Ask the server for a result if its broadcast has not arrived by then (UDP
#: is lossy); repeated every tick until the result is in or the timeout.
OP_STATUS_POLL_AFTER_S = 2.0
PENDING_TICK_MS = 1000
FLASH_MS = 1500
#: Incoming state arrives in bursts (a write-back is one broadcast per
#: channel): card refreshes are coalesced into one after this long.
REFRESH_COALESCE_MS = 60
#: Watchdog grace after the warning, before the server sends the safe op.
WATCHDOG_GRACE_S = 120.

# Card geometry.  Groups of cards are laid out by MasonryLayout
# (card_layout.py), which picks the number of columns: never narrower than
# CARD_MIN_WIDTH (or the widest card's own minimum), narrower than
# CARD_PREFERRED_WIDTH only when the extra column earns it, never wider than
# CARD_MAX_WIDTH (then centred).
CARD_MIN_WIDTH = 360
CARD_PREFERRED_WIDTH = 430
CARD_MAX_WIDTH = 540
CARD_GAP = 12

# Colours.  Text on the dark cards needs brighter variants of the theme's
# state colours (theme.OK / ERR are fills; as text they are hard to read).
OK_TEXT = "#5fd38d"
WARN_TEXT = "#f0c14b"
ERR_TEXT = "#ff6b6b"
HAZARD = "#ff5252"
HAZARD_FILL = "#5c1f1f"
DIRTY_FILL = "#4d3a1f"
ON_FILL = "#2f4a3a"
CHIP_BG = "#2e2e2e"
CHIP_BORDER = "#4c4c4c"

_LEVEL_COLOR = {
    "on": OK_TEXT, "ok": OK_TEXT,
    "off": theme.FG_MUTED,
    "partial": WARN_TEXT, "warn": WARN_TEXT,
    "hazard": HAZARD,
    "err": ERR_TEXT, "error": ERR_TEXT,
    "unknown": theme.FG_DISABLED, "stale": theme.FG_DISABLED,
}
_PILL_BG = {
    "on": theme.OK, "off": "#4d4d4d", "partial": "#8a6a12", "warn": "#8a6a12",
    "hazard": "#c62828", "unknown": "#454545",
}
_ACCENT = {
    "on": theme.OK, "partial": theme.WARN, "warn": theme.WARN, "hazard": HAZARD,
}

_SETTINGS_ORG = "waxx"
_SETTINGS_APP = "device_control_gui"


def _settings() -> QSettings:
    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


def _setting(key: str, default=None):
    try:
        value = _settings().value(key, default)
    except Exception:
        return default
    return default if value is None else value


def _save_setting(key: str, value) -> None:
    try:
        _settings().setValue(key, value)
    except Exception:
        pass


def _small(text: str = "", color: str = theme.FG_MUTED) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {color}; font-size: 11px;")
    label.setWordWrap(True)
    return label


def _fmt_s(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    if seconds < 60:
        return f"{seconds} s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def _button_css(color: str = theme.FG, border: str = theme.BORDER,
                bold: bool = False, height: int = 24, background: str = theme.BG_BUTTON,
                radius: int = 4) -> str:
    weight = "600" if bold else "normal"
    return (f"QPushButton {{ color: {color}; background: {background};"
            f" border: 1px solid {border}; border-radius: {radius}px; padding: 1px 10px;"
            f" min-height: {height - 4}px; font-weight: {weight}; }}"
            f"QPushButton:hover {{ background: {theme.BG_BUTTON_HOVER}; }}"
            f"QPushButton:disabled {{ color: {theme.FG_DISABLED}; border-color: #3a3a3a;"
            f" background: #303030; }}")


def _chip_css(color: str = theme.FG, border: str = CHIP_BORDER, dashed: bool = False) -> str:
    style = "dashed" if dashed else "solid"
    return (f"QLabel {{ color: {color}; background: {CHIP_BG}; border: 1px {style} {border};"
            f" border-radius: 10px; padding: 2px 9px; font-size: 12px; }}")


def _pill_button_css(color: str, border: str, background: str = theme.BG_BUTTON) -> str:
    return (f"QPushButton {{ color: {color}; background: {background};"
            f" border: 1px solid {border}; border-radius: 12px; padding: 3px 12px;"
            f" min-height: 18px; }}"
            f"QPushButton:hover {{ background: {theme.BG_BUTTON_HOVER}; }}"
            f"QPushButton:disabled {{ color: {theme.FG_DISABLED}; border-color: #3a3a3a; }}")


def _state_pill_css(level: str) -> str:
    bg = _PILL_BG.get(level, "#454545")
    return (f"QLabel {{ background: {bg}; color: white; border-radius: 9px;"
            f" padding: 2px 10px; font-size: 12px; font-weight: 600; }}")


def _card_css(accent: str) -> str:
    return (f"QFrame#composite_card {{ background: {theme.BG_CARD};"
            f" border: 1px solid {theme.BORDER}; border-left: 4px solid {accent};"
            f" border-radius: 8px; }}")


def _hline() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {theme.BORDER}; border: none;")
    return line


def _clipboard(text: str) -> None:
    app = QGuiApplication.instance()
    if app is not None:
        app.clipboard().setText(text)


def _search_terms(query: str) -> list[str]:
    return [t for t in re.split(r"\s+", (query or "").strip().lower()) if t]


# --- network worker ------------------------------------------------------------

class _OpSender(QThread):
    """Sends op requests, status queries and other requests off the GUI
    thread, in order."""

    replied = pyqtSignal(int, dict)           # request id, reply (ops)
    status_replied = pyqtSignal(int, dict)    # seq, reply
    requested = pyqtSignal(int, dict)         # request id, reply (anything else)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._jobs: deque = deque()
        self._running = True
        self._client: MonitorClient | None = None
        try:
            self.client_name = socket.gethostname()
        except Exception:
            self.client_name = ""

    def submit(self, req_id: int, op: str, sig: str, args: dict, payload: dict,
               operator: str = "") -> None:
        with self._cond:
            self._jobs.append(("op", req_id, op, sig, args, payload, operator))
            self._cond.notify()

    def query(self, seq: int) -> None:
        with self._cond:
            self._jobs.append(("status", seq))
            self._cond.notify()

    def request(self, req_id: int, obj: dict) -> None:
        with self._cond:
            self._jobs.append(("request", req_id, obj))
            self._cond.notify()

    def _get_client(self) -> MonitorClient | None:
        if self._client is None:
            try:
                self._client = MonitorClient(discovery_timeout=0.5)
            except Exception:
                return None
        return self._client

    def run(self):
        while True:
            with self._cond:
                while self._running and not self._jobs:
                    self._cond.wait(0.5)
                if not self._running:
                    return
                job = self._jobs.popleft()
            client = self._get_client()
            if job[0] == "op":
                _, req_id, op, sig, args, payload, operator = job
                reply = None
                if client is not None:
                    reply = client.send_op(op, sig, args, payload, client=self.client_name,
                                           operator=operator)
                if reply is None:
                    self._client = None
                    reply = {"status": "error", "msg": "monitor server unreachable"}
                self.replied.emit(req_id, reply)
            elif job[0] == "status":
                _, seq = job
                reply = client.op_status(seq) if client is not None else None
                if reply is None:
                    self._client = None
                    continue
                self.status_replied.emit(seq, reply)
            else:
                _, req_id, obj = job
                reply = client.request(obj) if client is not None else None
                if reply is None:
                    self._client = None
                    reply = {"status": "error", "msg": "monitor server unreachable"}
                self.requested.emit(req_id, reply)

    def stop(self):
        with self._cond:
            self._running = False
            self._cond.notify_all()


# --- editors --------------------------------------------------------------------

class _WheelGuard(QObject):
    """Wheel events reach a spinbox/combo only while it has focus; otherwise
    they go to the scroll area.  Scrolling the tab never edits a value."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return False


_WHEEL_GUARD = None


def _wheel_guard() -> _WheelGuard:
    global _WHEEL_GUARD
    if _WHEEL_GUARD is None:
        _WHEEL_GUARD = _WheelGuard()
    return _WHEEL_GUARD


class _FieldEditor(QWidget):
    """Spinbox (or combo for a choice) for one :class:`~composite.Arg`, with a
    default button.  Orange fill while edited and not yet applied; follows the
    hardware readback otherwise.  ``label`` is a separate widget: the card
    puts it in its own grid column so every field of a card lines up."""

    edited = pyqtSignal()
    submitted = pyqtSignal()    # Enter pressed

    def __init__(self, device_key: str, arg: cmp.Arg, parent=None):
        super().__init__(parent)
        self.arg = arg
        self._settings_key = f"composite/{device_key}/{arg.name}"
        self._dirty = False
        self._readback: float | None = None
        self._clean: float | None = None      # last value known to be applied
        self._level = "ok"
        self._highlight = False
        self._default_ctx_fn: Callable[[], Any] | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        self.label = QLabel(arg.title)
        self.label.setStyleSheet(f"color: {theme.FG_MUTED};")
        if arg.tooltip:
            self.label.setToolTip(arg.tooltip)

        self._css = ""
        if arg.kind in (cmp.KIND_CHOICE, cmp.KIND_BOOL):
            self.combo = QComboBox()
            choices = arg.choices or (("off", 0.0), ("on", 1.0))
            for text, value in choices:
                self.combo.addItem(text, float(value))
            self.combo.currentIndexChanged.connect(self._on_user_edit)
            self.spin = None
            self.input = self.combo
        else:
            self.combo = None
            self.spin = QDoubleSpinBox()
            decimals = 0 if arg.kind == cmp.KIND_INT else arg.decimals
            self.spin.setDecimals(decimals)
            lo = arg.to_display(arg.minimum) if arg.minimum is not None else -1e12
            hi = arg.to_display(arg.maximum) if arg.maximum is not None else 1e12
            if arg.scale < 0:
                lo, hi = hi, lo
            self.spin.setRange(min(lo, hi), max(lo, hi))
            if arg.step is not None:
                self.spin.setSingleStep(arg.step)
            if arg.unit:
                self.spin.setSuffix(f" {arg.unit}")
            self.spin.setKeyboardTracking(False)
            self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
            self.spin.setMinimumWidth(96)
            self.spin.valueChanged.connect(self._on_user_edit)
            self.spin.lineEdit().returnPressed.connect(self.submitted.emit)
            self.input = self.spin
        self.input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.input.installEventFilter(_wheel_guard())
        self.input.installEventFilter(self)
        if self.spin is not None:
            self.spin.lineEdit().installEventFilter(self)
        self.input.setMinimumHeight(24)
        row.addWidget(self.input, 1)

        self.default_button = QPushButton("↺")
        self.default_button.setFixedSize(24, 24)
        self.default_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.default_button.setStyleSheet(
            f"QPushButton {{ color: {theme.FG_MUTED}; background: transparent;"
            f" border: 1px solid transparent; border-radius: 4px; }}"
            f"QPushButton:hover {{ color: {theme.FG}; border-color: {theme.BORDER};"
            f" background: {theme.BG_BUTTON}; }}")
        self.default_button.clicked.connect(self._on_default)
        # A field without a default keeps the space, so every row's editor
        # ends at the same x.
        policy = self.default_button.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.default_button.setSizePolicy(policy)
        self.default_button.setVisible(arg.default is not None)
        row.addWidget(self.default_button)
        self._restyle()

    # -- values ---------------------------------------------------------------

    def value(self) -> float | None:
        if self.combo is not None:
            data = self.combo.currentData()
            return None if data is None else float(data)
        return self.arg.from_display(self.spin.value())

    def set_value(self, si: float | None, dirty: bool = False) -> None:
        if si is None:
            return
        if self.combo is not None:
            idx = self.combo.findData(float(si))
            if idx < 0:
                return
            with QSignalBlocker(self.combo):
                self.combo.setCurrentIndex(idx)
        else:
            with QSignalBlocker(self.spin):
                self.spin.setValue(self.arg.to_display(float(si)))
        self._dirty = dirty
        if not dirty:
            self._clean = self.value()
        self._restyle()

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        """The value was applied: follow the readback again."""
        self._dirty = False
        self._clean = self.value()
        self._save()
        self._restyle()

    def revert(self) -> None:
        """Esc: back to the hardware value (or the last applied one)."""
        value = self._readback if self._readback is not None else self._clean
        if value is not None:
            self.set_value(value)
        else:
            self._dirty = False
            self._restyle()
        self.edited.emit()

    def set_readback(self, si: float | None) -> None:
        self._readback = si
        if si is None or self._dirty or self.input.hasFocus():
            self._restyle()
            return
        self.set_value(si)

    def has_readback(self) -> bool:
        return self.arg.readback is not None

    def load_initial(self, ctx: cmp.Context) -> None:
        value = self.arg.readback_value(ctx)
        if value is None and not self.has_readback():
            saved = _setting(self._settings_key)
            try:
                value = float(saved) if saved is not None else None
            except (TypeError, ValueError):
                value = None
        if value is None:
            value = self.arg.default_value(ctx)
        if value is None and self.arg.minimum is not None:
            value = self.arg.minimum
        self.set_value(value if value is not None else 0.0)

    def set_default_source(self, ctx_fn: Callable[[], Any]) -> None:
        self._default_ctx_fn = ctx_fn

    def refresh_default_tooltip(self, ctx: cmp.Context) -> None:
        value = self.arg.default_value(ctx)
        text = "Default: --" if value is None else f"Default: {self.arg.format(value)}"
        self.default_button.setToolTip(f"{text}\nFills the field; press an op to send it.")

    def set_level(self, level: str) -> None:
        if level != self._level:
            self._level = level
            self._restyle()

    def set_highlight(self, on: bool) -> None:
        """The op under the mouse sends this field."""
        if on != self._highlight:
            self._highlight = on
            self._restyle()

    # -- internals ---------------------------------------------------------------

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape \
                and self._dirty:
            self.revert()
            return True
        return False

    def _save(self) -> None:
        if not self.has_readback():
            value = self.value()
            if value is not None:
                _save_setting(self._settings_key, value)

    def _on_user_edit(self, *_):
        self._dirty = True
        self._restyle()
        self.edited.emit()

    def _on_default(self):
        if self._default_ctx_fn is None:
            return
        value = self.arg.default_value(self._default_ctx_fn())
        if value is not None:
            self.set_value(value, dirty=True)
            self.edited.emit()

    def _restyle(self) -> None:
        if self._level == "error":
            border = ERR_TEXT
        elif self._level == "warn":
            border = WARN_TEXT
        elif self._highlight:
            border = theme.ACCENT
        else:
            border = theme.BORDER
        background = DIRTY_FILL if self._dirty else theme.BG_SUNKEN
        kind = "QComboBox" if self.combo is not None else "QDoubleSpinBox"
        css = (f"{kind} {{ border: 1px solid {border}; border-radius: 4px;"
               f" padding: 1px 4px; background: {background}; color: {theme.FG_STRONG}; }}")
        if css != self._css:
            # Refreshes arrive in bursts; re-polishing unchanged widgets is
            # the expensive part.
            self._css = css
            self.input.setStyleSheet(css)
        tip = []
        if self.arg.tooltip:
            tip.append(self.arg.tooltip)
        if self._readback is not None:
            tip.append(f"Hardware (last set): {self.arg.format(self._readback)}")
        if self._dirty:
            tip.append("Edited -- not applied yet (Esc reverts).")
        self.input.setToolTip("\n".join(tip))


class _TableEditor(QWidget):
    """Rows of values (e.g. tweezer traps), sent as an op payload."""

    edited = pyqtSignal()

    def __init__(self, device_key: str, table: cmp.Table, parent=None):
        super().__init__(parent)
        self.table = table
        self._settings_key = f"composite/{device_key}/table/{table.name}"
        self._dirty = False
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(3)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        title = QLabel(table.label)
        title.setStyleSheet(f"color: {theme.FG}; font-weight: 600;")
        if table.tooltip:
            title.setToolTip(table.tooltip)
        head.addWidget(title)
        head.addStretch(1)
        self.add_button = QPushButton("+ Add")
        self.defaults_button = QPushButton("Defaults")
        for b in (self.add_button, self.defaults_button):
            b.setStyleSheet(_pill_button_css(theme.FG, theme.BORDER))
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            head.addWidget(b)
        self.add_button.clicked.connect(self._add_row_clicked)
        box.addLayout(head)

        headers = [f"{c.title} ({c.unit})" if c.unit else c.title for c in table.columns]
        self._info_col = len(headers) if table.info is not None else None
        if table.info is not None:
            headers.append(table.info_label or "")
        self._remove_col = len(headers)
        headers.append("")
        self.grid = QTableWidget(0, len(headers))
        self.grid.setHorizontalHeaderLabels(headers)
        self.grid.verticalHeader().setVisible(False)
        self.grid.verticalHeader().setDefaultSectionSize(24)
        self.grid.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.grid.setShowGrid(False)
        self.grid.setAlternatingRowColors(True)
        self.grid.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._grid_css = ""
        self._style_grid(theme.BORDER)
        hdr = self.grid.horizontalHeader()
        for col in range(len(headers)):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self._remove_col, QHeaderView.ResizeMode.Fixed)
        self.grid.setColumnWidth(self._remove_col, 26)
        self.grid.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        box.addWidget(self.grid)
        self._fit_height()
        self.message = _small()
        self.message.hide()
        box.addWidget(self.message)

    def _style_grid(self, border: str) -> None:
        css = (f"QTableWidget {{ background: {theme.BG_SUNKEN};"
               f" alternate-background-color: {theme.BG_ALT}; color: {theme.FG};"
               f" border: 1px solid {border}; border-radius: 4px; }}"
               f"QHeaderView::section {{ background: {theme.BG_RAISED}; color: {theme.FG_MUTED};"
               f" border: none; border-bottom: 1px solid {theme.BORDER}; padding: 2px 6px;"
               f" font-size: 11px; }}"
               f"QDoubleSpinBox {{ background: transparent; color: {theme.FG_STRONG};"
               f" border: none; padding-left: 4px; }}")
        if css != self._grid_css:
            self._grid_css = css
            self.grid.setStyleSheet(css)

    # -- rows -------------------------------------------------------------------

    def rows(self) -> list[list[float]]:
        out = []
        for r in range(self.grid.rowCount()):
            row = []
            for c, col in enumerate(self.table.columns):
                spin = self.grid.cellWidget(r, c)
                row.append(col.from_display(spin.value()))
            out.append(row)
        return out

    def set_rows(self, rows, dirty: bool = False) -> None:
        self.grid.setRowCount(0)
        for row in rows:
            self._append(row)
        self._fit_height()
        self._dirty = dirty
        self.edited.emit()

    def _fit_height(self) -> None:
        """As tall as its rows (one empty row minimum), scrolling beyond ~8."""
        rows = min(max(self.grid.rowCount(), 1), 8)
        header = self.grid.horizontalHeader().sizeHint().height()
        row_h = self.grid.verticalHeader().defaultSectionSize()
        self.grid.setFixedHeight(header + rows * row_h + 2 * self.grid.frameWidth() + 2)

    def load_initial(self, ctx: cmp.Context) -> None:
        saved = _setting(self._settings_key)
        rows = None
        if saved:
            try:
                rows = [list(map(float, r)) for r in json.loads(saved)]
            except Exception:
                rows = None
        self.set_rows(rows if rows is not None else self.table.default_rows(ctx))
        self._dirty = False

    def mark_clean(self) -> None:
        self._dirty = False
        _save_setting(self._settings_key, json.dumps(self.rows()))

    @property
    def dirty(self) -> bool:
        return self._dirty

    def _append(self, values) -> None:
        r = self.grid.rowCount()
        self.grid.insertRow(r)
        for c, col in enumerate(self.table.columns):
            spin = QDoubleSpinBox()
            spin.setDecimals(col.decimals)
            lo = col.to_display(col.minimum) if col.minimum is not None else -1e12
            hi = col.to_display(col.maximum) if col.maximum is not None else 1e12
            spin.setRange(min(lo, hi), max(lo, hi))
            if col.step is not None:
                spin.setSingleStep(col.step)
            spin.setKeyboardTracking(False)
            spin.setFrame(False)
            spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            spin.installEventFilter(_wheel_guard())
            value = values[c] if c < len(values) else (col.default_value(None) or 0.0)
            spin.setValue(col.to_display(float(value)))
            spin.valueChanged.connect(self._on_edit)
            self.grid.setCellWidget(r, c, spin)
        if self._info_col is not None:
            info = QLabel("")
            info.setStyleSheet(f"color: {theme.FG_MUTED}; padding-left: 4px;")
            self.grid.setCellWidget(r, self._info_col, info)
        remove = QPushButton("✕")
        remove.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        remove.setStyleSheet(
            f"QPushButton {{ color: {theme.FG_MUTED}; background: transparent; border: none; }}"
            f"QPushButton:hover {{ color: {ERR_TEXT}; }}")
        remove.setToolTip("Remove this row (sent with the next apply)")
        remove.clicked.connect(lambda _=False, b=remove: self._remove(b))
        self.grid.setCellWidget(r, self._remove_col, remove)

    def _remove(self, button) -> None:
        for r in range(self.grid.rowCount()):
            if self.grid.cellWidget(r, self._remove_col) is button:
                self.grid.removeRow(r)
                break
        self._fit_height()
        self._on_edit()

    def _add_row_clicked(self) -> None:
        if self.grid.rowCount() >= self.table.max_rows:
            return
        # A new row copies the last one, shifted by one step in the first
        # column, so adding traps side by side is one click each.
        rows = self.rows()
        if rows:
            new = list(rows[-1])
            step = self.table.columns[0].step
            if step:
                new[0] = new[0] + self.table.columns[0].from_display(step)
        else:
            new = [c.default_value(None) or 0.0 for c in self.table.columns]
        self._append(new)
        self._fit_height()
        self._on_edit()

    def _on_edit(self, *_):
        self._dirty = True
        self.edited.emit()

    def refresh(self, ctx: cmp.Context) -> list[cmp.Check]:
        rows = self.rows()
        if self._info_col is not None:
            for r, row in enumerate(rows):
                label = self.grid.cellWidget(r, self._info_col)
                try:
                    text = self.table.info(row, ctx)
                except Exception:
                    text = ""
                label.setText(text or "")
        checks = self.table.checks(rows, ctx)
        shown = [c for c in checks if c.is_error or self._dirty]
        if shown:
            level = cmp.worst(shown)
            self.message.setStyleSheet(f"color: {_LEVEL_COLOR[level]}; font-size: 11px;")
            self.message.setText("\n".join(c.message for c in shown))
            self.message.show()
        else:
            self.message.hide()
        self._style_grid(theme.PENDING if self._dirty else theme.BORDER)
        return checks


class _Lamp(QLabel):
    """A status chip: coloured dot, name, state word."""

    def __init__(self, text: str = "", parent=None, dashed: bool = False):
        super().__init__(text, parent)
        self._css = ""
        self._dashed = dashed
        self._set_css(CHIP_BORDER)

    def _set_css(self, border: str) -> None:
        css = _chip_css(theme.FG_MUTED, border, self._dashed)
        if css != self._css:
            self._css = css
            self.setStyleSheet(css)

    def set_state(self, label: str, on: bool | None, on_text: str, off_text: str,
                  on_level: str, tooltip: str = "") -> None:
        if on is None:
            color, word = theme.FG_DISABLED, "?"
        elif on:
            color, word = _LEVEL_COLOR.get(on_level, OK_TEXT), on_text
        else:
            color, word = theme.OFF, off_text
        word_color = color if on else theme.FG_MUTED
        self.setText(f"<span style='color:{color}'>●</span> {label} "
                     f"<span style='color:{word_color}'>{word}</span>")
        self._set_css(color if on else CHIP_BORDER)
        self.setToolTip(tooltip)

    def set_measured(self, label: str, text: str, level: str, tooltip: str = "") -> None:
        """A telemetry chip: dashed, so it never reads as a setpoint."""
        color = _LEVEL_COLOR.get(level, theme.FG_MUTED)
        value_color = theme.FG_STRONG if level == "ok" else color
        self.setText(f"<span style='color:{theme.FG_MUTED}'>{label}</span> "
                     f"<span style='color:{value_color}'>{text}</span>")
        self._set_css(color if level in ("warn", "hazard", "error") else CHIP_BORDER)
        self.setToolTip(tooltip)


# --- card ------------------------------------------------------------------------

class _HoverWatcher(QObject):
    """Highlights the fields an op sends while the mouse is on its button."""

    def __init__(self, card: "CompositeCard", names: tuple, parent=None):
        super().__init__(parent)
        self._card = card
        self._names = names

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Enter:
            self._card.highlight_fields(self._names, True)
        elif event.type() == QEvent.Type.Leave:
            self._card.highlight_fields(self._names, False)
        return False


class CompositeCard(QFrame):
    """One device: header (collapse, title, state pill, ⋯), chips, rows,
    watchdog line, footer."""

    def __init__(self, panel: "CompositePanel", device: cmp.CompositeDevice):
        super().__init__()
        self.panel = panel
        self.device = device
        self.setObjectName("composite_card")
        self._accent = ""
        self._set_accent(theme.BORDER)
        self.editors: dict[str, _FieldEditor] = {}
        self.tables: dict[str, _TableEditor] = {}
        self.op_buttons: dict[str, list[QPushButton]] = {}
        self.menu_actions: dict[str, QPushButton] = {}
        self.lamps: list[tuple[cmp.Lamp, _Lamp]] = []
        self.readouts: list[tuple[cmp.Readout, QLabel]] = []
        self.measured: list[tuple[cmp.Measured, _Lamp]] = []
        self.infos: list[tuple[cmp.Info, QLabel]] = []
        self.toggles: list[tuple[cmp.ChannelToggle, QPushButton]] = []
        self.row_messages: list[tuple[tuple, QLabel]] = []
        self._pending_ops: dict[str, tuple[float, str]] = {}   # key -> (t0, text)
        self.last_sent: tuple[str, dict] | None = None          # (op key, args)
        self.status = cmp.Status("unknown", "")
        self.hazard: str | None = None
        self.watchdog: dict | None = None
        self._collapsed = True              # every card starts collapsed
        self._build()
        self.set_collapsed(True)

    # -- construction -------------------------------------------------------------

    def _set_accent(self, color: str) -> None:
        """The card's left stripe follows its state."""
        if color != self._accent:
            self._accent = color
            self.setStyleSheet(_card_css(color))

    def _build(self) -> None:
        """Header, then a body with status chips and the layout rows in
        definition order -- consecutive field rows share one grid so labels,
        editors and buttons line up; consecutive toggles and small buttons
        share one wrapping row of pills; every Menu goes into the header's ⋯.
        The footer (last op's outcome) sits under a hairline."""
        d = self.device
        box = QVBoxLayout(self)
        box.setContentsMargins(12, 9, 12, 9)
        box.setSpacing(7)

        head = QHBoxLayout()
        head.setSpacing(6)
        self.chevron = QToolButton()
        self.chevron.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chevron.setStyleSheet(f"QToolButton {{ color: {theme.FG_MUTED}; border: none;"
                                   f" background: transparent; font-size: 12px; }}"
                                   f"QToolButton:hover {{ color: {theme.FG}; }}")
        self.chevron.clicked.connect(self.toggle_collapsed)
        head.addWidget(self.chevron)
        self.title = QLabel(d.title)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.title.setFont(font)
        self.title.setStyleSheet(f"color: {theme.FG_STRONG};")
        self.title.setToolTip((d.doc + "\n\n" if d.doc else "") + "Click to collapse / expand.")
        self.title.setCursor(Qt.CursorShape.PointingHandCursor)
        self.title.mousePressEvent = lambda e: self.toggle_collapsed()
        head.addWidget(self.title)
        head.addStretch(1)
        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {theme.FG_MUTED}; font-size: 11px;")
        head.addWidget(self.summary)
        self.state_pill = QLabel("")
        self.state_pill.setStyleSheet(_state_pill_css("unknown"))
        head.addWidget(self.state_pill)
        head.addWidget(self._menu_button())
        box.addLayout(head)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 2, 0, 0)
        body.setSpacing(8)
        box.addWidget(self.body)

        if d.lamps or d.readouts or d.measured:
            chips = FlowLayout(hspacing=5, vspacing=5)
            for lamp in d.lamps:
                w = _Lamp(lamp.label)
                chips.addWidget(w)
                self.lamps.append((lamp, w))
            for readout in d.readouts:
                w = QLabel("")
                w.setStyleSheet(_chip_css(theme.FG))
                w.setToolTip(readout.tooltip)
                chips.addWidget(w)
                self.readouts.append((readout, w))
            for m in d.measured:
                w = _Lamp("", dashed=True)
                chips.addWidget(w)
                self.measured.append((m, w))
            body.addLayout(chips)

        self._form: QGridLayout | None = None
        self._actions: FlowLayout | None = None
        for layout_row in d.layout:
            self._build_row(body, layout_row)

        if d.max_on_s:
            self._build_watchdog(body)

        box.addStretch(1)
        self._footer_line = _hline()
        box.addWidget(self._footer_line)
        self.footer = _small("")
        self.footer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self.footer)
        self._footer_line.hide()
        self.footer.hide()

        ctx = self.ctx()
        for editor in self.editors.values():
            editor.load_initial(ctx)
        for table in self.tables.values():
            table.load_initial(ctx)

    def _form_grid(self, box: QVBoxLayout) -> QGridLayout:
        if self._form is None:
            self._form = QGridLayout()
            self._form.setHorizontalSpacing(8)
            self._form.setVerticalSpacing(5)
            self._form.setColumnStretch(1, 1)
            self._form_rows = 0
            box.addLayout(self._form)
        return self._form

    def _action_row(self, box: QVBoxLayout) -> FlowLayout:
        if self._actions is None:
            self._actions = FlowLayout(hspacing=6, vspacing=5)
            box.addLayout(self._actions)
        return self._actions

    def _build_row(self, box: QVBoxLayout, row) -> None:
        d = self.device
        if isinstance(row, cmp.Menu):
            return                      # in the header's ⋯ button
        if not isinstance(row, cmp.FieldRow):
            self._form = None
        if not (isinstance(row, cmp.ChannelToggle)
                or (isinstance(row, cmp.Buttons) and not row.main)):
            self._actions = None

        if isinstance(row, cmp.Buttons) and row.main:
            line = QHBoxLayout()
            line.setSpacing(8)
            for key in row.ops:
                line.addWidget(self._op_button(key, main=True), 1)
            box.addLayout(line)
        elif isinstance(row, cmp.Buttons):
            flow = self._action_row(box)
            for key in row.ops:
                flow.addWidget(self._op_button(key, pill=True))
        elif isinstance(row, cmp.ChannelToggle):
            button = QPushButton(row.label)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setToolTip(row.tooltip or f"{row.dtype}.{row.name}")
            button.setProperty("css", "")
            button.clicked.connect(lambda _=False, r=row: self._toggle_channel(r))
            self._action_row(box).addWidget(button)
            self.toggles.append((row, button))
        elif isinstance(row, cmp.FieldRow):
            grid = self._form_grid(box)
            for j, name in enumerate(row.fields):
                editor = _FieldEditor(d.key, d.get_field(name))
                editor.set_default_source(self.ctx)
                editor.edited.connect(self.refresh)
                if row.ops:
                    first = row.ops[0]
                    editor.submitted.connect(lambda k=first: self.trigger(k))
                self.editors[name] = editor
                r = self._form_rows
                grid.addWidget(editor.label, r, 0, Qt.AlignmentFlag.AlignVCenter)
                grid.addWidget(editor, r, 1)
                if j == 0 and row.ops:
                    ops = QHBoxLayout()
                    ops.setSpacing(4)
                    for key in row.ops:
                        ops.addWidget(self._op_button(key))
                    ops.addStretch(1)
                    grid.addLayout(ops, r, 2)
                self._form_rows += 1
            msg = _small()
            msg.hide()
            grid.addWidget(msg, self._form_rows, 1, 1, 2)
            self._form_rows += 1
            self.row_messages.append((tuple(row.fields), msg))
        elif isinstance(row, cmp.TableRow):
            table = _TableEditor(d.key, d.get_table(row.table))
            table.defaults_button.clicked.connect(
                lambda _=False, t=table: t.set_rows(t.table.default_rows(self.ctx()), dirty=True))
            table.edited.connect(self.refresh)
            self.tables[row.table] = table
            box.addWidget(table)
            if row.ops:
                line = QHBoxLayout()
                line.setSpacing(6)
                line.addStretch(1)
                for key in row.ops:
                    line.addWidget(self._op_button(key))
                box.addLayout(line)
        elif isinstance(row, cmp.Info):
            label = _small()
            label.setStyleSheet(f"color: {theme.FG_MUTED}; font-size: 11px; font-style: italic;")
            box.addWidget(label)
            self.infos.append((row, label))

    def _build_watchdog(self, box: QVBoxLayout) -> None:
        line = QHBoxLayout()
        line.setSpacing(6)
        self.watchdog_label = _small("")
        line.addWidget(self.watchdog_label, 1)
        self.watchdog_arm = QPushButton("Arm watchdog")
        self.watchdog_extend = QPushButton("Keep on")
        self.watchdog_disarm = QPushButton("Disarm")
        for b in (self.watchdog_arm, self.watchdog_extend, self.watchdog_disarm):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setStyleSheet(_pill_button_css(theme.FG, theme.BORDER))
            line.addWidget(b)
        self.watchdog_arm.setToolTip(
            f"The monitor server warns every GUI after {_fmt_s(self.device.max_on_s)} and, "
            f"{_fmt_s(WATCHDOG_GRACE_S)} later without 'Keep on', sends "
            f"'{self.device.get_op(self.device.safe_op).label}' itself -- whether or not a "
            f"GUI is open.")
        self.watchdog_arm.clicked.connect(lambda: self.panel.arm_watchdog(self))
        self.watchdog_extend.clicked.connect(lambda: self.panel.watchdog_request(
            self, "extend_watchdog"))
        self.watchdog_disarm.clicked.connect(lambda: self.panel.watchdog_request(
            self, "disarm_watchdog"))
        box.addLayout(line)

    def _menu_button(self) -> QToolButton:
        """The header's ⋯: every Menu op (danger ops in red), then copy-as-code
        and adopt-as-default."""
        d = self.device
        keys = [key for row in d.layout if isinstance(row, cmp.Menu) for key in row.ops]
        tool = QToolButton()
        tool.setText("⋯")
        tool.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        tool.setToolTip("More operations")
        tool.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tool.setStyleSheet(f"QToolButton {{ color: {theme.FG}; background: transparent;"
                           f" border: 1px solid transparent; border-radius: 4px;"
                           f" padding: 0px 6px; font-size: 15px; font-weight: 600; }}"
                           f"QToolButton:hover {{ background: {theme.BG_BUTTON};"
                           f" border-color: {theme.BORDER}; }}"
                           f"QToolButton::menu-indicator {{ image: none; }}")
        menu = QMenu(tool)
        menu.setStyleSheet(f"QMenu {{ background: {theme.BG_RAISED};"
                           f" border: 1px solid {theme.BORDER_STRONG}; padding: 4px 0px; }}")

        def item(text, color, bold, tip, callback):
            button = QPushButton(text)
            button.setStyleSheet(
                f"QPushButton {{ text-align: left; color: {color}; background: transparent;"
                f" border: none; padding: 5px 18px; font-weight: {'600' if bold else 'normal'}; }}"
                f"QPushButton:hover {{ background: {theme.HIGHLIGHT}; }}"
                f"QPushButton:disabled {{ color: {theme.FG_DISABLED}; }}")
            button.setToolTip(tip)
            action = QWidgetAction(menu)
            action.setDefaultWidget(button)
            menu.addAction(action)
            button.clicked.connect(lambda _=False: (menu.close(), callback()))
            return button

        for key in keys:
            op = d.get_op(key)
            self.menu_actions[key] = item(op.label, ERR_TEXT if op.danger else theme.FG,
                                          op.danger, op.tooltip,
                                          lambda k=key: self.trigger(k))
        if keys:
            menu.addSeparator()
        item("Copy last op as experiment code", theme.FG_MUTED, False,
             "The last op sent from this card, with its values, as it would read in an "
             "experiment's kernel (right-click any op button for that op).",
             self.copy_last_as_code)
        if any(f.param for f in d.fields):
            item("Copy 'adopt as ExptParams default' diff", theme.FG_MUTED, False,
                 "The ExptParams lines these fields mirror, with the card's values, as a "
                 "diff on the clipboard. Nothing is written.", self.copy_adopt_diff)
        tool.setMenu(menu)
        return tool

    def _op_button(self, key: str, main: bool = False, pill: bool = False) -> QPushButton:
        op = self.device.get_op(key)
        button = QPushButton(op.label)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if pill:
            color = ERR_TEXT if op.danger else theme.FG
            css = _pill_button_css(color, ERR_TEXT if op.danger else theme.BORDER)
        elif main and op.danger:
            css = _button_css(color=ERR_TEXT, border=ERR_TEXT, bold=True, height=34, radius=6)
        elif main:
            css = _button_css(bold=True, height=34, radius=6)
        elif op.danger:
            css = _button_css(color=ERR_TEXT, border=ERR_TEXT, bold=True)
        else:
            css = _button_css()
        button.setStyleSheet(css)
        button.setProperty("base_css", css)
        button.setProperty("main", main)
        sends = [self.device.get_field(a).title for a in op.args
                 if a in {f.name for f in self.device.fields}]
        tip = op.tooltip
        if sends:
            tip = (tip + "\n\n" if tip else "") + "Sends: " + ", ".join(sends)
        button.setProperty("tip", tip)
        button.setToolTip(tip)
        button.clicked.connect(lambda _=False, k=key: self.trigger(k))
        button.installEventFilter(_HoverWatcher(self, tuple(op.args), button))
        button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        button.customContextMenuRequested.connect(
            lambda pos, k=key, b=button: self._op_context_menu(k, b, pos))
        self.op_buttons.setdefault(key, []).append(button)
        return button

    def _op_context_menu(self, key: str, button: QPushButton, pos) -> None:
        menu = QMenu(button)
        action = menu.addAction("Copy as experiment code")
        action.triggered.connect(lambda: self.copy_as_code(key))
        menu.exec(button.mapToGlobal(pos))

    # -- state --------------------------------------------------------------------

    def values(self) -> dict:
        return {name: e.value() for name, e in self.editors.items()}

    def ctx(self) -> cmp.Context:
        return self.panel.context(self.device.key, self.values() if self.editors else {})

    def highlight_fields(self, names, on: bool) -> None:
        for name in names:
            editor = self.editors.get(name)
            if editor is not None:
                editor.set_highlight(on)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = bool(collapsed)
        self.body.setVisible(not self._collapsed)
        self.chevron.setText("▸" if self._collapsed else "▾")
        self.summary.setVisible(self._collapsed)
        self.updateGeometry()
        self.panel.relayout()

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def search_text(self) -> str:
        d = self.device
        parts = [d.key, d.title, d.group, d.doc]
        parts += [op.label for op in d.ops] + [f.title for f in d.fields]
        return " ".join(parts).lower()

    def refresh(self) -> None:
        ctx = self.ctx()
        status = self.device.status(ctx)
        self.status = status
        self.hazard = self.device.hazard_text(ctx)
        level = "hazard" if self.hazard else status.level
        text = status.text or ("" if status.level == "unknown" else status.level)
        self.state_pill.setText(text)
        self.state_pill.setToolTip(status.detail or text)
        pill = _state_pill_css(level)
        if self.state_pill.styleSheet() != pill:
            self.state_pill.setStyleSheet(pill)
        self.state_pill.setVisible(bool(text))
        self._set_accent(_ACCENT.get(level, theme.BORDER))
        for lamp, widget in self.lamps:
            on = ctx.is_on(lamp.dtype, lamp.name)
            widget.set_state(lamp.label, on, lamp.on_text, lamp.off_text, lamp.on_level,
                             lamp.tooltip or f"{lamp.dtype}.{lamp.name}")
        for readout, widget in self.readouts:
            try:
                value = readout.value(ctx)
            except Exception as e:
                value = f"error: {e!r}"
            widget.setText(f"{readout.label} {value if value is not None else '--'}")
        self.refresh_measured(ctx)
        for editor in self.editors.values():
            editor.set_readback(editor.arg.readback_value(ctx))
            editor.refresh_default_tooltip(ctx)
        ctx = self.ctx()
        for fields, label in self.row_messages:
            checks = []
            for name in fields:
                editor = self.editors[name]
                value = editor.value()
                found = editor.arg.checks(value, ctx) if value is not None else []
                editor.set_level(cmp.worst(found))
                # Spell out warnings only for what the operator typed; a value
                # that is just the hardware readback keeps the coloured
                # border, and the text again on sending.
                checks.extend(c for c in found if c.is_error or editor.dirty)
            if checks:
                lvl = cmp.worst(checks)
                label.setStyleSheet(f"color: {_LEVEL_COLOR[lvl]}; font-size: 11px;")
                label.setText("\n".join(c.message for c in checks))
                label.show()
            else:
                label.hide()
        for table in self.tables.values():
            table.refresh(ctx)
        for info, label in self.infos:
            try:
                text = info.text(ctx)
            except Exception as e:
                text = f"error: {e!r}"
            label.setText(text or "")
            label.setVisible(bool(text))
        for toggle, button in self.toggles:
            on = ctx.is_on(toggle.dtype, toggle.name)
            word = "?" if on is None else (toggle.on_text if on else toggle.off_text)
            color = _LEVEL_COLOR.get(toggle.on_level, WARN_TEXT)
            button.setText(f"{toggle.label}: {word}")
            css = _pill_button_css(color, color) if on else _pill_button_css(theme.FG, theme.BORDER)
            if button.property("css") != css:
                button.setProperty("css", css)
                button.setStyleSheet(css)
        self._refresh_main_buttons(level)
        self._refresh_summary()
        self._refresh_watchdog()
        self._refresh_enabled()

    def refresh_measured(self, ctx: cmp.Context | None = None) -> None:
        if not self.measured:
            return
        ctx = ctx or self.ctx()
        for (m, sample, level), (_, widget) in zip(self.device.measured_values(ctx),
                                                   self.measured):
            if sample is None:
                raw = ctx.telemetry.get(m.source)
                why = "no reading"
                if isinstance(raw, cmp.Sample):
                    why = raw.error or f"{raw.age_s:.0f} s old"
                widget.set_measured(m.label, "--", "stale",
                                    f"{m.tooltip}\n\nNo fresh reading ({why}).".strip())
            else:
                widget.set_measured(m.label, m.format(sample, ctx), level,
                                    f"{m.tooltip}\n\nRead {sample.age_s:.1f} s ago.".strip())
        self._refresh_summary()

    def _refresh_main_buttons(self, level: str) -> None:
        """The main button matching the state is lit: On while on, Off while
        off -- the buttons read as the state at a glance."""
        lit = {"on": "on", "hazard": "on", "off": "off"}.get(level)
        for key, buttons in self.op_buttons.items():
            for b in buttons:
                if not b.property("main"):
                    continue
                base = b.property("base_css")
                if key == lit == "on":
                    css = _button_css(color=theme.FG_STRONG, border=OK_TEXT, bold=True,
                                      height=34, background=ON_FILL, radius=6)
                elif key == lit == "off":
                    css = _button_css(color=theme.FG_STRONG, border=theme.BORDER_STRONG,
                                      bold=True, height=34, background="#404040", radius=6)
                else:
                    css = base
                if b.property("lit_css") != css and not b.property("flashing"):
                    b.setProperty("lit_css", css)
                    b.setStyleSheet(css)

    def _refresh_summary(self) -> None:
        """What a collapsed card still shows: the measured values."""
        parts = []
        for m, widget in self.measured:
            if m.unit:
                parts.append(re.sub(r"<[^>]+>", "", widget.text()))
        self.summary.setText("  ".join(parts[:1]))

    def _refresh_watchdog(self) -> None:
        if not self.device.max_on_s:
            return
        dog = self.watchdog
        hazardous = bool(self.hazard)
        if dog is None:
            seen = self.panel.hazard_age(self.device.key)
            text = "Watchdog off"
            color = theme.FG_MUTED
            if hazardous and seen is not None:
                text += f" · on (seen) for {_fmt_s(seen)}"
                if seen > self.device.max_on_s:
                    color = WARN_TEXT
                    text += f" -- over {_fmt_s(self.device.max_on_s)}"
            self.watchdog_label.setText(text)
            self.watchdog_label.setStyleSheet(f"color: {color}; font-size: 11px;")
            self.watchdog_arm.setVisible(True)
            self.watchdog_arm.setEnabled(hazardous and self.panel.ops_allowed()[0])
            self.watchdog_extend.setVisible(False)
            self.watchdog_disarm.setVisible(False)
            return
        fires = dog.get("fires_in_s")
        warned = dog.get("warned")
        if warned:
            text = f"WATCHDOG: '{self.device.get_op(self.device.safe_op).label}' in " \
                   f"{_fmt_s(fires or 0)} unless kept on"
            color = HAZARD
        else:
            text = f"Watchdog armed · acts in {_fmt_s(fires or 0)}"
            color = OK_TEXT
        self.watchdog_label.setText(text)
        self.watchdog_label.setStyleSheet(f"color: {color}; font-size: 11px;"
                                          f" font-weight: {'600' if warned else 'normal'};")
        self.watchdog_arm.setVisible(False)
        self.watchdog_extend.setVisible(True)
        self.watchdog_disarm.setVisible(True)

    def set_watchdog(self, info: dict | None) -> None:
        self.watchdog = dict(info) if info else None
        self._refresh_watchdog()

    def _refresh_enabled(self) -> None:
        allowed, why = self.panel.ops_allowed()
        for key, buttons in self.op_buttons.items():
            for b in buttons:
                b.setEnabled(allowed and key not in self._pending_ops)
                tip = b.property("tip") or ""
                b.setToolTip(f"{tip}\n\n{why}".strip() if not allowed else tip)
        for key, item in self.menu_actions.items():
            item.setEnabled(allowed and key not in self._pending_ops)

    # -- ops ------------------------------------------------------------------------

    def collect(self, key: str, overrides: dict | None = None):
        """(args, payload, checks) for op ``key`` from the fields (or
        ``overrides``), with every finding."""
        op = self.device.get_op(key)
        ctx = self.ctx()
        overrides = dict(overrides or {})
        args, checks = {}, []
        for name in op.args:
            editor = self.editors.get(name)
            arg = self.device.get_field(name)
            if name in overrides:
                value = overrides[name]
            else:
                value = editor.value() if editor is not None else arg.default_value(ctx)
            if value is None:
                checks.append(cmp.Check.error(f"{arg.title}: no value"))
                continue
            args[name] = float(value)
            checks.extend(arg.checks(value, ctx))
        payload = {}
        for name in op.payload:
            rows = self.tables[name].rows()
            payload[name] = rows
            checks.extend(self.tables[name].table.checks(rows, ctx))
        if op.check is not None and not any(c.is_error for c in checks):
            try:
                checks.extend(cmp._as_checks(op.check(args, ctx)))
            except Exception as e:
                checks.append(cmp.Check.warn(f"check failed: {e!r}"))
        return args, payload, checks

    def describe_args(self, key: str, args: dict) -> str:
        op = self.device.get_op(key)
        parts = []
        for name in op.args:
            spec = self.device.get_field(name)
            if name in args and not (spec.replay is not None and float(args[name]) < 0):
                parts.append(f"{spec.title} {spec.format(args[name])}")
        return ", ".join(parts)

    def trigger(self, key: str, overrides: dict | None = None, confirmed: bool = False) -> bool:
        """Collect, check, confirm and send op ``key``.  Returns whether sent.
        Refusals are shown on the card, not in a pop-up."""
        op = self.device.get_op(key)
        entry = self.panel.table.get(f"{self.device.key}.{key}")
        allowed, why = self.panel.ops_allowed()
        if not allowed:
            self.set_footer(f"✕ {op.label}: {why}", "error")
            return False
        args, payload, checks = self.collect(key, overrides)
        errors = [c for c in checks if c.is_error]
        if errors:
            text = "; ".join(c.message for c in errors)
            self.set_footer(f"✕ {op.label} not sent: {text}", "error")
            return False
        warnings = [c.message for c in checks if c.level == "warn"]
        if not confirmed and (warnings or op.confirm or op.danger):
            lines = []
            if op.confirm:
                lines.append(op.confirm)
            if warnings:
                lines.append("\n".join(f"• {w}" for w in warnings))
            detail = self.describe_args(key, args)
            verb = f"{op.label}" + (f" ({detail})" if detail else "")
            if not self.panel.confirm(f"{self.device.title}: {op.label}", "\n\n".join(lines)
                                      or f"{self.device.title}: {verb}?",
                                      verb=verb, danger=op.danger):
                self.set_footer(f"{op.label}: cancelled", "off")
                return False
        self.last_sent = (key, dict(args))
        self.panel.send_op(self, entry, args, payload)
        return True

    def op_sent(self, entry: cmp.OpEntry) -> None:
        key = entry.op.key
        self._pending_ops[key] = (time.monotonic(), "sending")
        for b in self.op_buttons.get(key, []):
            b.setText(f"{entry.op.label} …")
        self.set_footer(f"… {entry.op.label}: sending", "partial")
        self._refresh_enabled()

    def op_accepted(self, entry: cmp.OpEntry, seq: int) -> None:
        key = entry.op.key
        t0 = self._pending_ops.get(key, (time.monotonic(), ""))[0]
        self._pending_ops[key] = (t0, f"queued (#{seq})")
        self.tick_pending(time.monotonic())

    def tick_pending(self, now: float) -> None:
        if not self._pending_ops:
            return
        key, (t0, state) = next(iter(self._pending_ops.items()))
        busy = self.panel.busy_left()
        extra = f" · hardware busy ~{busy:.0f} s" if busy > 0.5 else ""
        self.set_footer(f"… {self.device.get_op(key).label}: {state} · {now - t0:.0f} s{extra}",
                        "partial")

    def op_finished(self, entry: cmp.OpEntry, ok: bool, text: str,
                    sent_args: dict | None = None, sent_payload: dict | None = None) -> None:
        key = entry.op.key
        self._pending_ops.pop(key, None)
        if ok:
            # Fields stay orange until what they sent has actually happened --
            # and only if they still hold what was sent.
            for name, value in (sent_args or {}).items():
                editor = self.editors.get(name)
                if editor is not None and editor.value() is not None \
                        and abs(editor.value() - float(value)) <= 1e-12 * max(1., abs(value)):
                    editor.mark_clean()
            for name, rows in (sent_payload or {}).items():
                table = self.tables.get(name)
                if table is not None and table.rows() == rows:
                    table.mark_clean()
        for b in self.op_buttons.get(key, []):
            b.setText(entry.op.label)
            current = b.property("lit_css") or b.property("base_css") or ""
            color = OK_TEXT if ok else ERR_TEXT
            b.setProperty("flashing", True)
            b.setStyleSheet(current + f"QPushButton {{ border: 1px solid {color}; }}")

            def unflash(b=b):
                b.setProperty("flashing", False)
                b.setStyleSheet(b.property("lit_css") or b.property("base_css") or "")
            QTimer.singleShot(FLASH_MS, unflash)
        stamp = time.strftime("%H:%M:%S")
        if ok:
            self.set_footer(f"✓ {entry.op.label} · {stamp} · {text}", "ok")
        else:
            self.set_footer(f"✕ {entry.op.label} · {stamp} · {text}", "error")
        self.refresh()

    def other_result(self, result: dict) -> None:
        """An op sent by another GUI, a scene or a watchdog."""
        op_key = str(result.get("op", "")).split(".", 1)[-1]
        try:
            label = self.device.get_op(op_key).label
        except KeyError:
            label = op_key
        who = [w for w in (result.get("operator"), result.get("client")) if w]
        origin = result.get("origin")
        src = f" · {origin}" if origin and origin != "gui" else ""
        src += f" · from {' @ '.join(who)}" if who else ""
        stamp = time.strftime("%H:%M:%S")
        ok = bool(result.get("ok"))
        text = str(result.get("text") or ("done" if ok else "failed"))
        self.set_footer(f"{'✓' if ok else '✕'} {label} · {stamp} · {text}{src}",
                        "ok" if ok else "error")

    def set_footer(self, text: str, level: str) -> None:
        self.footer.setText(text)
        self.footer.setToolTip(text)
        self.footer.setStyleSheet(f"color: {_LEVEL_COLOR.get(level, theme.FG_MUTED)};"
                                  f" font-size: 11px;")
        self.footer.setVisible(bool(text))
        self._footer_line.setVisible(bool(text))

    def _toggle_channel(self, toggle: cmp.ChannelToggle) -> None:
        on = self.ctx().is_on(toggle.dtype, toggle.name)
        new = 0 if on else 1
        key = "ttl_state" if toggle.dtype == "ttl" else "sw_state"
        self.panel.send_channel(toggle.dtype, toggle.name, {key: new})
        self.refresh()

    # -- clipboard ------------------------------------------------------------------

    def copy_as_code(self, key: str) -> str:
        op = self.device.get_op(key)
        args, _, _ = self.collect(key)
        text = self._code_text(op, args, "the card's current values")
        _clipboard(text)
        self.set_footer(f"Copied '{op.label}' as experiment code.", "off")
        return text

    def copy_last_as_code(self) -> str:
        if self.last_sent is None:
            self.set_footer("Nothing sent from this card yet -- right-click an op button "
                            "to copy it with the current values.", "off")
            return ""
        key, args = self.last_sent
        op = self.device.get_op(key)
        text = self._code_text(op, args, "the values it was last sent with")
        _clipboard(text)
        self.set_footer(f"Copied the last op ('{op.label}') as experiment code.", "off")
        return text

    def _code_text(self, op: cmp.Op, args: dict, source: str) -> str:
        lines = [f"# {self.device.title}: {op.label} -- from the Device Control Composite tab,",
                 f"# {source}, {time.strftime('%Y-%m-%d %H:%M')}.  Inside a @kernel method."]
        if op.host is not None:
            name = getattr(op.host, "__qualname__", repr(op.host))
            lines.append(f"# NOTE: in the monitor this op also runs a host step ({name}) "
                         f"{'after' if op.host_after else 'before'} the code; not included.")
        lines.append(cmp.literal_code(self.device, op, args))
        return "\n".join(lines) + "\n"

    def copy_adopt_diff(self) -> str:
        text = adopt_diff(self, self.panel.params)
        _clipboard(text)
        self.set_footer("Copied the ExptParams diff to the clipboard -- nothing was written.",
                        "off")
        return text


def adopt_diff(card: CompositeCard, params) -> str:
    """The ExptParams lines ``card``'s fields mirror (``Arg.param``), as a
    diff with the card's values -- the lab convention: the old line commented
    out, the new one tagged.  Only text; nothing is written anywhere."""
    stamp = time.strftime("%Y-%m-%d")
    head = [f"# Adopt as default -- {card.device.title}, Device Control {time.strftime('%Y-%m-%d %H:%M')}.",
            "# NOT applied: paste by hand after checking. Values are the card's fields as shown."]
    try:
        path = inspect.getsourcefile(type(params))
        with open(path, encoding="utf-8") as f:
            source = f.read().splitlines()
    except Exception:
        path, source = None, []
    body = []
    for arg in card.device.fields:
        if not arg.param:
            continue
        editor = card.editors.get(arg.name)
        value = editor.value() if editor is not None else None
        if value is None:
            continue
        old = getattr(params, arg.param, None)
        state = "typed, NOT applied" if editor.dirty else "applied"
        try:
            same = old is not None and abs(float(old) - value) <= 1e-12 * max(1., abs(value))
        except (TypeError, ValueError):
            same = False
        if same:
            body.append(f"# {arg.param}: unchanged ({arg.format(value)})")
            continue
        pattern = re.compile(rf"^(\s*)self\.{re.escape(arg.param)}\s*=")
        idx = next((i for i, line in enumerate(source) if pattern.match(line)), None)
        if idx is None:
            body.append(f"# {arg.param}: no 'self.{arg.param} = ...' line found in "
                        f"{path or 'the ExptParams source'}; value {value!r} ({state})")
            continue
        line = source[idx]
        indent = pattern.match(line).group(1)
        body += [f"@@ {os.path.basename(path)}:{idx + 1}  ({arg.title}, {state})",
                 f"-{line}",
                 f"+{indent}# {line.strip()}",
                 f"+{indent}self.{arg.param} = {value!r} # Device Control, {stamp}"]
    if not body:
        body.append("# (no field of this card mirrors an ExptParams attribute)")
    return "\n".join(head + ([f"--- {path}"] if path else []) + body) + "\n"


# --- groups and scenes -----------------------------------------------------------

class GroupSection(QWidget):
    """A labelled column of cards: one masonry item, so a group's cards stay
    together."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(CARD_GAP - 4)
        label = QLabel(title.upper())
        label.setStyleSheet(f"color: {theme.FG_MUTED}; font-size: 10px; font-weight: 700;"
                            f" letter-spacing: 1px; padding-left: 4px;")
        box.addWidget(label)
        self._box = box
        self.cards: list[QWidget] = []

    def add(self, card: QWidget) -> None:
        self.cards.append(card)
        self._box.addWidget(card)

    def refresh_visibility(self) -> None:
        self.setVisible(any(not c.isHidden() for c in self.cards))


class ScenesCard(QFrame):
    """Scenes: several ops in order, run by the monitor server, with a cleanup
    that always runs."""

    def __init__(self, panel: "CompositePanel", scenes):
        super().__init__()
        self.panel = panel
        self.scenes = tuple(scenes)
        self.setObjectName("composite_card")
        self.setStyleSheet(_card_css(theme.ACCENT))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 9, 12, 9)
        outer.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.chevron = QToolButton()
        self.chevron.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chevron.setStyleSheet(f"QToolButton {{ color: {theme.FG_MUTED}; border: none;"
                                   f" background: transparent; font-size: 12px; }}"
                                   f"QToolButton:hover {{ color: {theme.FG}; }}")
        self.chevron.clicked.connect(self.toggle_collapsed)
        head.addWidget(self.chevron)
        title = QLabel("Scenes")
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        title.setFont(font)
        title.setStyleSheet(f"color: {theme.FG_STRONG};")
        title.setToolTip("Run by the monitor server one op after the other; the cleanup "
                         "runs when the steps finish, fail or are cancelled, and closing "
                         "this window does not stop it.\n\nClick to collapse / expand.")
        title.setCursor(Qt.CursorShape.PointingHandCursor)
        title.mousePressEvent = lambda e: self.toggle_collapsed()
        head.addWidget(title)
        head.addStretch(1)
        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {WARN_TEXT}; font-size: 11px;")
        head.addWidget(self.summary)
        outer.addLayout(head)
        self.body = QWidget()
        box = QVBoxLayout(self.body)
        box.setContentsMargins(0, 2, 0, 0)
        box.setSpacing(8)
        outer.addWidget(self.body)
        self.editors: dict[tuple[str, str], _FieldEditor] = {}
        self.run_buttons: dict[str, QPushButton] = {}
        for scene in self.scenes:
            head = QHBoxLayout()
            name = QLabel(scene.title)
            name.setStyleSheet(f"color: {theme.FG};")
            name.setWordWrap(True)
            name.setToolTip(scene.doc)
            head.addWidget(name, 1)
            run = QPushButton("Run")
            run.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            run.setStyleSheet(_pill_button_css(theme.FG, theme.BORDER))
            run.setToolTip(scene.doc)
            run.clicked.connect(lambda _=False, s=scene: self.panel.run_scene(s))
            head.addWidget(run)
            self.run_buttons[scene.key] = run
            box.addLayout(head)
            if scene.fields:
                grid = QGridLayout()
                grid.setHorizontalSpacing(8)
                grid.setColumnStretch(1, 1)
                for r, f in enumerate(scene.fields):
                    editor = _FieldEditor(f"scene.{scene.key}", f)
                    editor.set_default_source(lambda: self.panel.context(None, {}))
                    editor.load_initial(self.panel.context(None, {}))
                    grid.addWidget(editor.label, r, 0)
                    grid.addWidget(editor, r, 1)
                    self.editors[(scene.key, f.name)] = editor
                box.addLayout(grid)
        line = QHBoxLayout()
        self.progress = _small("No scene running.")
        line.addWidget(self.progress, 1)
        self.cancel_button = QPushButton("Cancel scene")
        self.cancel_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cancel_button.setStyleSheet(_pill_button_css(ERR_TEXT, ERR_TEXT))
        self.cancel_button.setToolTip("Stop after the op in flight and run the cleanup.")
        self.cancel_button.clicked.connect(self.panel.cancel_scene)
        self.cancel_button.hide()
        line.addWidget(self.cancel_button)
        box.addWidget(_hline())
        box.addLayout(line)
        self.running: dict | None = None
        self._collapsed = True
        self.set_collapsed(True)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = bool(collapsed)
        self.body.setVisible(not self._collapsed)
        self.chevron.setText("▸" if self._collapsed else "▾")
        self.summary.setVisible(self._collapsed)
        self.updateGeometry()
        self.panel.relayout()

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def values(self, scene: cmp.Scene) -> dict:
        return {f.name: self.editors[(scene.key, f.name)].value() for f in scene.fields}

    def mark_sent(self, scene: cmp.Scene) -> None:
        for f in scene.fields:
            self.editors[(scene.key, f.name)].mark_clean()

    def show_progress(self, info: dict | None, last: dict | None = None) -> None:
        self.running = info
        if info:
            phase = "cleanup" if info.get("phase") == "finally" else "step"
            text = f"▶ {info.get('title')}: {phase} {info.get('step')}/{info.get('n')}"
            if info.get("label"):
                text += f" · {info['label']}"
            if info.get("hold_left_s") is not None:
                text += f" · {_fmt_s(info['hold_left_s'])} left"
            if info.get("cancel"):
                text += " · cancelling"
            self.progress.setText(text)
            self.progress.setStyleSheet(f"color: {WARN_TEXT}; font-size: 11px;")
            self.cancel_button.show()
            self.summary.setText(f"▶ {info.get('title')} running")
        else:
            self.cancel_button.hide()
            self.summary.setText("")
            if last:
                ok = last.get("state") == "done"
                self.progress.setText(f"{'✓' if ok else '✕'} {last.get('title')}: "
                                      f"{last.get('text')}")
                self.progress.setStyleSheet(
                    f"color: {OK_TEXT if ok else ERR_TEXT}; font-size: 11px;")
        allowed = self.panel.ops_allowed()[0]
        for b in self.run_buttons.values():
            b.setEnabled(allowed and not info)


# --- panel -----------------------------------------------------------------------

class CompositePanel(QWidget):
    """The Composite tab: header (op-path status, operator, Ping), then the
    cards in groups, then the scenes.

    ``channel_sender(dtype, name, changes)`` is how single-channel toggles
    are sent (the host GUI's ordinary update path); ``log_line(text)`` records
    op outcomes in the host GUI's changes log.  ``telemetry`` is a
    :class:`~waxx.util.device_state.telemetry.TelemetryHub` (or None); the
    host GUI feeds its samples in with :meth:`set_telemetry`.
    """

    hazards_changed = pyqtSignal()

    def __init__(self, devices, params=None, frames=None,
                 channel_sender: Callable | None = None,
                 log_line: Callable[[str], None] | None = None,
                 parent=None, start_sender: bool = True, scenes=()):
        super().__init__(parent)
        self.params = params
        self.frames = frames
        self.config: dict = {}
        self.device_state: dict = {}
        self.telemetry: dict = {}
        self.trust: dict = {"trusted": True, "reason": ""}
        self.run_pending: dict | None = None
        self._busy_until = 0.0
        self._channel_sender = channel_sender
        self._log_line = log_line
        self._monitor_state: int | None = None
        self._monitor_ops: dict | None = None
        self._reachable = False
        self._req = 0
        self._by_req: dict[int, tuple] = {}
        self._by_seq: dict[int, tuple] = {}
        self._requests: dict[int, Callable[[dict], None]] = {}
        self._ping_req: int | None = None
        self._hazard_seen: dict[str, float] = {}
        self._last_hazards: tuple = ()
        self.cards: list[CompositeCard] = []
        self.sections: list[GroupSection] = []
        self.definition_error = ""
        self._relayout_pending = False

        box = QVBoxLayout(self)
        box.setContentsMargins(CARD_GAP, 10, CARD_GAP, CARD_GAP)
        box.setSpacing(10)

        # Header bar: where the op path stands, who is operating, Ping.
        bar = QFrame()
        bar.setObjectName("composite_header")
        bar.setStyleSheet(f"QFrame#composite_header {{ background: {theme.BG_RAISED};"
                          f" border: 1px solid {theme.BORDER}; border-radius: 6px; }}")
        head = QHBoxLayout(bar)
        head.setContentsMargins(10, 5, 6, 5)
        head.setSpacing(8)
        self.path_dot = QLabel("●")
        head.addWidget(self.path_dot)
        self.path_label = _small("Composite ops: waiting for the monitor server…")
        head.addWidget(self.path_label, 1)
        self.operator = QLineEdit(str(_setting("composite/operator", "") or ""))
        self.operator.setPlaceholderText("operator")
        self.operator.setToolTip("Your name: recorded with every op in the monitor "
                                 "server's journal and shown to other GUIs.")
        self.operator.setMaximumWidth(120)
        self.operator.setStyleSheet(f"QLineEdit {{ background: {theme.BG_SUNKEN};"
                                    f" color: {theme.FG}; border: 1px solid {theme.BORDER};"
                                    f" border-radius: 4px; padding: 1px 6px; }}")
        self.operator.editingFinished.connect(
            lambda: _save_setting("composite/operator", self.operator.text().strip()))
        head.addWidget(self.operator)
        self.ping_button = QPushButton("Ping")
        self.ping_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.ping_button.setStyleSheet(_pill_button_css(theme.FG, theme.BORDER))
        self.ping_button.setToolTip("Send a no-op through the whole op path (server -> "
                                    "monitor kernel -> back) and time it. Touches no hardware.")
        self.ping_button.clicked.connect(self.ping)
        head.addWidget(self.ping_button)
        self.collapse_button = QPushButton("Expand all")
        self.collapse_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.collapse_button.setStyleSheet(_pill_button_css(theme.FG, theme.BORDER))
        self.collapse_button.clicked.connect(self._toggle_all)
        head.addWidget(self.collapse_button)
        box.addWidget(bar)

        try:
            self.table = cmp.OpTable(devices)
            self.scenes = tuple(scenes or ())
            cmp.validate_scenes(self.scenes, self.table)
        except cmp.DefinitionError as e:
            self.table = cmp.OpTable(())
            self.scenes = ()
            self.definition_error = str(e)
            err = _small(f"Composite device definitions are invalid, nothing is shown:\n{e}",
                         ERR_TEXT)
            box.addWidget(err)

        # Groups of cards in balanced columns (card_layout.MasonryLayout).
        self.cards_host = QWidget()
        self.masonry = MasonryLayout(self.cards_host, min_column=CARD_MIN_WIDTH,
                                     preferred_column=CARD_PREFERRED_WIDTH,
                                     max_column=CARD_MAX_WIDTH, spacing=CARD_GAP)
        box.addWidget(self.cards_host)
        box.addStretch(1)

        by_group: dict[str, GroupSection] = {}
        for device in self.table.devices:
            group = device.group or "Devices"
            if group not in by_group:
                by_group[group] = GroupSection(group)
                self.sections.append(by_group[group])
            card = CompositeCard(self, device)
            self.cards.append(card)
            by_group[group].add(card)
        self.scenes_card = None
        if self.scenes:
            section = GroupSection("Scenes")
            self.scenes_card = ScenesCard(self, self.scenes)
            section.add(self.scenes_card)
            self.sections.append(section)
        for section in self.sections:
            self.masonry.addWidget(section)

        self._sender = _OpSender(self)
        self._sender.replied.connect(self._on_replied)
        self._sender.status_replied.connect(self._on_status_replied)
        self._sender.requested.connect(self._on_requested)
        if start_sender:
            self._sender.start()

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(PENDING_TICK_MS)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_header()
        self.refresh()

    # -- context ------------------------------------------------------------------

    def context(self, device_key: str | None, fields: dict) -> cmp.Context:
        host = (self.device_state.get(device_key, {}) if device_key
                else dict(self.device_state))
        return cmp.Context(self.config, self.params, self.frames, fields=fields,
                           host_state=host, telemetry=self.telemetry, trust=self.trust)

    # -- inputs from the host GUI -------------------------------------------------

    def set_config(self, config: dict) -> None:
        self.config = config if config is not None else {}
        self.request_refresh()

    def request_refresh(self) -> None:
        """Coalesce bursts of state changes into one refresh."""
        if not self._refresh_timer.isActive():
            self._refresh_timer.start(REFRESH_COALESCE_MS)

    def set_device_state(self, state: dict | None) -> None:
        self.device_state = dict(state or {})
        self.request_refresh()

    def set_telemetry(self, samples: dict) -> None:
        self.telemetry = dict(samples or {})
        for card in self.cards:
            card.refresh_measured()
        self._update_hazards()

    def set_trust(self, trust: dict | None) -> None:
        trust = dict(trust or {"trusted": True, "reason": ""})
        if trust != self.trust:
            self.trust = trust
            self.request_refresh()

    def set_run_pending(self, pending: dict | None) -> None:
        if pending != self.run_pending:
            self.run_pending = pending
            self._refresh_header()
            self._refresh_allowed()
            self.request_refresh()

    def set_busy(self, seconds: float) -> None:
        self._busy_until = time.monotonic() + max(float(seconds), 0.)

    def busy_left(self) -> float:
        return max(self._busy_until - time.monotonic(), 0.)

    def set_monitor_state(self, state: int | None, reachable: bool = True) -> None:
        changed = state != self._monitor_state or reachable != self._reachable
        self._monitor_state = state
        self._reachable = reachable
        if changed:
            self._refresh_header()
            self._refresh_allowed()
            self.request_refresh()

    def set_monitor_detail(self, detail: dict | None) -> None:
        detail = detail or {}
        ops = detail.get("composite_ops")
        if ops != self._monitor_ops:
            self._monitor_ops = ops if isinstance(ops, dict) else None
            self._refresh_header()
            self._refresh_allowed()
            self.request_refresh()
        if "trust" in detail:
            self.set_trust(detail.get("trust"))
        if "run_pending" in detail:
            self.set_run_pending(detail.get("run_pending"))
        if isinstance(detail.get("runner"), dict):
            self.set_runner(detail["runner"])

    def set_runner(self, runner: dict) -> None:
        dogs = runner.get("watchdogs") or {}
        for card in self.cards:
            card.set_watchdog(dogs.get(card.device.key))
        if self.scenes_card is not None:
            self.scenes_card.show_progress(runner.get("scene"), runner.get("last_scene"))

    def on_scene(self, payload: dict) -> None:
        if self.scenes_card is None:
            return
        state = payload.get("state")
        if state == "running":
            self.scenes_card.show_progress(payload)
        else:
            self.scenes_card.show_progress(None, payload)
            if self._log_line is not None:
                self._log_line(f"[scene] {payload.get('title')}: {state} -- {payload.get('text')}")

    def on_watchdog(self, payload: dict) -> None:
        card = self._card(payload.get("device"))
        state = payload.get("state")
        if card is not None:
            if state in ("disarmed", "fired", "failed"):
                card.set_watchdog(None)
            if state == "warning":
                card.set_watchdog({"fires_in_s": payload.get("fires_in_s"), "warned": True})
            if state == "extended":
                card.set_watchdog({"fires_in_s": payload.get("fires_in_s"), "warned": False})
            if state in ("fired", "failed", "warning"):
                card.set_footer(f"Watchdog {state}: {payload.get('text') or ''}".strip(),
                                "error" if state != "warning" else "warn")
        if self._log_line is not None and state in ("warning", "fired", "failed"):
            self._log_line(f"[watchdog] {payload.get('title') or payload.get('device')}: "
                           f"{state} {payload.get('text') or ''}".strip())

    def refresh(self) -> None:
        for card in self.cards:
            card.refresh()
        if self.scenes_card is not None:
            self.scenes_card.show_progress(self.scenes_card.running)
        self._update_hazards()

    def _refresh_allowed(self) -> None:
        """Whether ops may be sent changes the buttons at once (not coalesced)."""
        for card in self.cards:
            card._refresh_enabled()
        if self.scenes_card is not None:
            self.scenes_card.show_progress(self.scenes_card.running)

    def relayout(self) -> None:
        """Heights changed (a card collapsed or opened): restack.  The masonry
        keeps every card in its column."""
        if getattr(self, "masonry", None) is not None:
            self.masonry.invalidate()

    # -- hazards ------------------------------------------------------------------

    def _update_hazards(self) -> None:
        now = time.monotonic()
        current = []
        for card in self.cards:
            if card.hazard:
                self._hazard_seen.setdefault(card.device.key, now)
                current.append((card.device.key, card.hazard))
            else:
                self._hazard_seen.pop(card.device.key, None)
        current = tuple(current)
        if current != self._last_hazards:
            self._last_hazards = current
            self.hazards_changed.emit()

    def hazards(self) -> list[dict]:
        """Every hazardous device: key, title, text, seen for, max on."""
        now = time.monotonic()
        out = []
        for card in self.cards:
            if not card.hazard:
                continue
            seen = self._hazard_seen.get(card.device.key)
            out.append({"key": card.device.key, "title": card.device.title,
                        "text": card.hazard, "seen_s": None if seen is None else now - seen,
                        "max_on_s": card.device.max_on_s,
                        "watchdog": card.watchdog,
                        "safe": card.device.safe_op})
        return out

    def hazard_age(self, key: str) -> float | None:
        seen = self._hazard_seen.get(key)
        return None if seen is None else time.monotonic() - seen

    def safe_plan(self, key: str) -> tuple[str, dict, str] | None:
        """(op key, args, description) of a device's safe op, or None."""
        card = self._card(key)
        if card is None or not card.device.safe_op:
            return None
        try:
            op_key, args = card.device.safe_request(card.ctx())
        except cmp.DefinitionError:
            return None
        op = card.device.get_op(op_key)
        detail = card.describe_args(op_key, args)
        return op_key, args, op.label + (f" ({detail})" if detail else "")

    def make_safe(self, keys) -> list[str]:
        """Send each device's safe op (already confirmed by the caller).
        Returns the devices whose op was refused before sending."""
        refused = []
        for key in keys:
            plan = self.safe_plan(key)
            card = self._card(key)
            if plan is None or card is None:
                refused.append(key)
                continue
            op_key, args, _ = plan
            if not card.trigger(op_key, overrides=args, confirmed=True):
                refused.append(key)
        return refused

    def show_device(self, key: str) -> None:
        card = self._card(key)
        if card is None:
            return
        if card.is_collapsed():
            card.set_collapsed(False)
        w = self.parentWidget()
        while w is not None and not hasattr(w, "ensureWidgetVisible"):
            w = w.parentWidget()
        if w is not None:
            w.ensureWidgetVisible(card)

    # -- search / collapse ----------------------------------------------------------

    def apply_search(self, query: str) -> None:
        terms = _search_terms(query)
        first = None
        for card in self.cards:
            match = all(t in card.search_text() for t in terms)
            card.setHidden(bool(terms) and not match)
            if terms and match and first is None:
                first = card
        if self.scenes_card is not None:
            text = " ".join(["scenes"] + [s.title.lower() for s in self.scenes])
            self.scenes_card.setHidden(bool(terms) and not all(t in text for t in terms))
        for section in self.sections:
            section.refresh_visibility()
        self.relayout()
        if first is not None and first.is_collapsed():
            first.set_collapsed(False)

    def _collapsibles(self) -> list:
        return list(self.cards) + ([self.scenes_card] if self.scenes_card is not None else [])

    def _toggle_all(self) -> None:
        collapse = not all(c.is_collapsed() for c in self._collapsibles())
        for card in self._collapsibles():
            card.set_collapsed(collapse)
        self.collapse_button.setText("Expand all" if collapse else "Collapse all")

    # -- op path status ---------------------------------------------------------------

    def ops_allowed(self) -> tuple[bool, str]:
        if self.definition_error:
            return False, "the composite definitions are invalid"
        if not self._reachable:
            return False, "the monitor server is unreachable"
        if self._monitor_state != STATES.READY:
            return False, "the monitor is not running -- composite ops need it"
        if self._monitor_ops is not None and not self._monitor_ops.get("registered"):
            return False, ("the running monitor registered no composite ops -- an older "
                           "monitor experiment, or its composite ops were rejected or did "
                           "not compile (see the monitor log)")
        if self.run_pending:
            return False, (f"run {self.run_pending.get('run_id')} is starting -- ops are "
                           "refused until it ends")
        return True, ""

    def _refresh_header(self) -> None:
        allowed, why = self.ops_allowed()
        info = self._monitor_ops or {}
        if not allowed:
            text, color = f"Composite ops unavailable: {why}.", WARN_TEXT
        elif info.get("hash") and info.get("hash") != self.table.hash:
            text = (f"Monitor ready, but it was built from different composite definitions "
                    f"({info.get('hash')} vs this GUI's {self.table.hash}): ops whose code "
                    f"changed are refused -- restart the monitor to load these.")
            color = WARN_TEXT
        else:
            n = info.get("count", len(self.table)) - 1
            text = f"Composite ops ready · {n} ops · definitions {self.table.hash}"
            color = OK_TEXT
        self.path_label.setText(text)
        self.path_label.setStyleSheet(f"color: {theme.FG}; font-size: 11px;")
        self.path_dot.setStyleSheet(f"color: {color}; font-size: 13px;")
        self.ping_button.setEnabled(allowed)

    # -- sending ---------------------------------------------------------------------

    def operator_name(self) -> str:
        return self.operator.text().strip()

    def send_op(self, card: CompositeCard | None, entry: cmp.OpEntry,
                args: dict, payload: dict) -> int:
        self._req += 1
        req = self._req
        self._by_req[req] = (card, entry, time.monotonic(), dict(args), dict(payload))
        if card is not None:
            card.op_sent(entry)
        self._sender.submit(req, entry.name, entry.signature, args, payload,
                            operator=self.operator_name())
        return req

    def send_request(self, obj: dict, on_reply: Callable[[dict], None] | None = None) -> int:
        self._req += 1
        req = self._req
        obj = dict(obj)
        obj.setdefault("client", self._sender.client_name)
        obj.setdefault("operator", self.operator_name())
        if on_reply is not None:
            self._requests[req] = on_reply
        self._sender.request(req, obj)
        return req

    def ping(self) -> None:
        entry = self.table.get(cmp.PING_OP)
        self._ping_req = self.send_op(None, entry, {}, {})
        self.path_label.setText("Ping sent…")

    def send_channel(self, dtype: str, name: str, changes: dict) -> None:
        if self._channel_sender is not None:
            self._channel_sender(dtype, name, changes)

    def confirm(self, title: str, text: str, verb: str = "Send", danger: bool = False) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning if danger else QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(text)
        yes = box.addButton(verb, QMessageBox.ButtonRole.AcceptRole)
        no = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        if danger:
            yes.setStyleSheet(f"color: {ERR_TEXT}; font-weight: 600;")
        box.setDefaultButton(no)
        box.exec()
        return box.clickedButton() is yes

    # -- watchdogs and scenes -------------------------------------------------------

    def arm_watchdog(self, card: CompositeCard) -> None:
        d = card.device
        try:
            op_key, args = d.safe_request(card.ctx(), deferred=True)
        except cmp.DefinitionError as e:
            card.set_footer(f"✕ watchdog not armed: {e}", "error")
            return
        entry = self.table.get(f"{d.key}.{op_key}")
        obj = {"type": "arm_watchdog", "device": d.key, "title": d.title,
               "max_on_s": d.max_on_s, "grace_s": WATCHDOG_GRACE_S,
               "op": entry.name, "sig": entry.signature, "args": args}

        def done(reply):
            if reply.get("status") == "ok":
                card.set_watchdog({"fires_in_s": reply.get("fires_in_s"), "warned": False})
                card.set_footer(f"Watchdog armed: '{d.get_op(op_key).label}' after "
                                f"{_fmt_s(reply.get('fires_in_s') or 0)} unless kept on.", "ok")
            else:
                card.set_footer(f"✕ watchdog not armed: {reply.get('msg')}", "error")
        self.send_request(obj, done)

    def watchdog_request(self, card: CompositeCard, mtype: str) -> None:
        def done(reply):
            if reply.get("status") != "ok":
                card.set_footer(f"✕ {mtype.replace('_', ' ')}: {reply.get('msg')}", "error")
            elif mtype == "disarm_watchdog":
                card.set_watchdog(None)
            else:
                card.set_watchdog({"fires_in_s": reply.get("fires_in_s"), "warned": False})
        self.send_request({"type": mtype, "device": card.device.key}, done)

    def run_scene(self, scene: cmp.Scene) -> bool:
        if self.scenes_card is None:
            return False
        allowed, why = self.ops_allowed()
        if not allowed:
            self.scenes_card.progress.setText(f"✕ {scene.title}: {why}")
            return False
        values = self.scenes_card.values(scene)
        problems, warnings = [], []
        for f in scene.fields:
            for c in f.checks(values.get(f.name), self.context(None, values)):
                (problems if c.is_error else warnings).append(f"{f.title}: {c.message}")
        request = None
        if not problems:
            try:
                request = self._resolve_with_cards(scene, values)
            except Exception as e:
                problems.append(str(e))
        if request is not None:
            # every step is checked as if its card sent it
            for step in request["steps"] + request["finally"]:
                if "op" not in step:
                    continue
                device_key, op_key = step["op"].split(".", 1)
                card = self._card(device_key)
                if card is None:
                    continue
                _, _, checks = card.collect(op_key, step["args"])
                for c in checks:
                    (problems if c.is_error else warnings).append(f"{step['label']}: {c.message}")
        if problems:
            self.scenes_card.progress.setText(f"✕ {scene.title} not started: "
                                              + "; ".join(problems))
            self.scenes_card.progress.setStyleSheet(f"color: {ERR_TEXT}; font-size: 11px;")
            return False
        lines = [scene.doc] if scene.doc else []
        if scene.confirm:
            lines.append(scene.confirm)
        if scene.leaves_on:
            lines.append(f"Leaves on: {scene.leaves_on}")
        steps = [s.get("label") or s.get("op") for s in request["steps"]]
        cleanup = [s.get("label") or s.get("op") for s in request["finally"]]
        lines.append("Steps: " + " → ".join(steps))
        if cleanup:
            lines.append("Always afterwards: " + " → ".join(cleanup))
        if warnings:
            lines.append("\n".join(f"• {w}" for w in warnings))
        if not self.confirm(scene.title, "\n\n".join(lines), verb=f"Run '{scene.title}'"):
            return False

        def done(reply):
            if reply.get("status") == "ok":
                self.scenes_card.mark_sent(scene)
                self.scenes_card.progress.setText(f"▶ {scene.title}: started")
            else:
                self.scenes_card.progress.setText(f"✕ {scene.title}: {reply.get('msg')}")
                self.scenes_card.progress.setStyleSheet(f"color: {ERR_TEXT}; font-size: 11px;")
        self.send_request(dict(request, type="run_scene"), done)
        return True

    def _resolve_with_cards(self, scene: cmp.Scene, values: dict) -> dict:
        """The scene's request, with every argument a step does not give
        taken from its device field's default in *that card's* context (a
        default may read the device's host state, telemetry or trust)."""
        request = scene.resolve(self.table, values, self.context(None, values))
        defs = list(scene.steps) + list(scene.finally_)
        resolved = request["steps"] + request["finally"]
        for spec, step in zip(defs, resolved):
            if not isinstance(spec, cmp.Step):
                continue
            card = self._card(step["op"].split(".", 1)[0])
            if card is None:
                continue
            ctx = card.ctx()
            entry = self.table.get(step["op"])
            for name in entry.arg_names:
                if name in (spec.args or {}):
                    continue
                value = entry.device.get_field(name).default_value(ctx)
                if value is not None:
                    step["args"][name] = float(value)
        return request

    def cancel_scene(self) -> None:
        running = self.scenes_card.running if self.scenes_card is not None else None
        self.send_request({"type": "cancel_scene",
                           "id": (running or {}).get("id")})

    # -- replies -----------------------------------------------------------------------

    def _card(self, key) -> CompositeCard | None:
        for card in self.cards:
            if card.device.key == key:
                return card
        return None

    def _on_requested(self, req: int, reply: dict) -> None:
        callback = self._requests.pop(req, None)
        if callback is not None:
            callback(reply)

    def _on_replied(self, req: int, reply: dict) -> None:
        item = self._by_req.pop(req, None)
        if item is None:
            return
        card, entry, t0, args, payload = item
        if reply.get("status") == "ok" and "seq" in reply:
            seq = int(reply["seq"])
            self._by_seq[seq] = (card, entry, t0, req, args, payload)
            if card is not None:
                card.op_accepted(entry, seq)
            return
        msg = str(reply.get("msg", "refused"))
        self._finish(card, entry, req, ok=False, text=f"not queued: {msg}")

    def on_op_result(self, result: dict) -> None:
        """An ``op_result`` broadcast (or a status reply's result)."""
        try:
            seq = int(result.get("seq"))
        except (TypeError, ValueError):
            return
        device = result.get("device")
        if device and isinstance(result.get("state"), dict):
            self.device_state[device] = dict(result["state"])
        item = self._by_seq.pop(seq, None)
        if item is None:
            # Someone else's op (another GUI, a scene, a watchdog).
            op = str(result.get("op", ""))
            card = self._card(op.split(".", 1)[0]) if "." in op else None
            if card is not None:
                card.other_result(result)
            if self._log_line is not None and op:
                who = f" (from {result.get('client')})" if result.get("client") else ""
                self._log_line(f"[op] {op} #{seq}: {result.get('text', '')}{who}")
            self.request_refresh()
            return
        card, entry, t0, req, args, payload = item
        elapsed = result.get("elapsed")
        if result.get("ok"):
            text = f"{elapsed:.2f} s" if isinstance(elapsed, (int, float)) else "done"
        else:
            text = str(result.get("text") or result.get("status_text") or "failed")
        self._finish(card, entry, req, ok=bool(result.get("ok")), text=text, seq=seq,
                     args=args, payload=payload)

    def _on_status_replied(self, seq: int, reply: dict) -> None:
        if reply.get("state") == "done" and isinstance(reply.get("result"), dict):
            self.on_op_result(reply["result"])
        elif reply.get("state") == "unknown" and seq in self._by_seq:
            card, entry, t0, req, args, payload = self._by_seq.pop(seq)
            self._finish(card, entry, req, ok=False,
                         text="the server no longer knows this op (was it restarted?) -- "
                              "outcome unknown")

    def _finish(self, card, entry, req, ok: bool, text: str, seq: int | None = None,
                args: dict | None = None, payload: dict | None = None) -> None:
        if req == self._ping_req:
            self._ping_req = None
            self._refresh_header()
            self.path_label.setText(
                f"Ping: {text}" if not ok else f"Ping round trip {text} ✓ (op path works)")
            return
        if card is not None:
            card.op_finished(entry, ok, text, args, payload)
        if self._log_line is not None:
            tag = f" #{seq}" if seq is not None else ""
            self._log_line(f"[op] {entry.name}{tag}: {'done, ' + text if ok else text}")

    def _on_tick(self) -> None:
        now = time.monotonic()
        for seq, (card, entry, t0, req, args, payload) in list(self._by_seq.items()):
            age = now - t0
            timeout = OP_RESULT_TIMEOUT_S + self.busy_left()
            if age > timeout:
                self._by_seq.pop(seq, None)
                self._finish(card, entry, req, ok=False,
                             text=f"no result after {age:.0f} s -- outcome unknown; check "
                                  "the monitor log")
            elif age > OP_STATUS_POLL_AFTER_S:
                self._sender.query(seq)
        for card in self.cards:
            card.tick_pending(now)
            if card.device.max_on_s:
                card._refresh_watchdog()

    def shutdown(self) -> None:
        self._tick.stop()
        self._sender.stop()
        self._sender.wait(2000)
