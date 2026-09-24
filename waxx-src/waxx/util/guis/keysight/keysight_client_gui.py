"""Client-side Keysight monitor GUI.

This widget talks ONLY to ``KeysightServer`` over TCP — it never opens a
direct VXI11 connection to the supplies.  Multiple dashboards can run
this widget concurrently without spamming the hardware.

All network I/O (discovery, snapshot polling, action RPCs) is dispatched
to ``QThreadPool`` so the GUI thread never stalls; that keeps window
moves smooth even when the server is missing or slow.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import math
import time

import pyqtgraph as pg
from PyQt6.QtCore import QRunnable, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.guis.keysight.keysight_client import KeysightClient

T_UPDATE_MS = 250
FONTSIZE_PT = 12          # value buttons
CAPTION_PT = 9           # the small '170 A' caption next to each value

# Rolling current-history plot.  Samples are whatever the server's poll
# worker read.  The buffer holds ``PLOT_RANGE_MAX_S``; the visible window
# is the "plot range" spinbox (``PLOT_RANGE_DEFAULT_S`` at start).
PLOT_RANGE_DEFAULT_S = 240
PLOT_RANGE_MIN_S = 10
PLOT_RANGE_MAX_S = 3600
PLOT_MIN_HEIGHT_PX = 90   # the plot grows with the dock; this is the floor
PLOT_PENS: dict[int, str] = {170: "#4fc3f7", 500: "#ffab40"}

# Stay quiet for this long after construction before surfacing a
# "server not found" message — the supervised server subprocess often
# takes a few seconds to spool up after the dashboard launches.
STARTUP_GRACE_S = 15.0

# When the server is unreachable, throttle reconnect attempts so we
# don't spawn a discovery thread every 500 ms while it's down.
# Polling stays at ``T_UPDATE_MS`` once connected; this only governs
# how often we *try* to (re)build the client.
RECONNECT_INTERVAL_S = 3.0

# Per-supply over-current alert thresholds (A), keyed by ``max_current``.
ALERT_THRESHOLDS: dict[int, float] = {500: 100, 170: 50}

# If any supply stays above its alert threshold (button red) for longer
# than this, the window flashes yellow and the taskbar entry flashes
# until the current drops back below threshold.
OVERCURRENT_FLASH_AFTER_S = 60.0
FLASH_PERIOD_MS = 500
FLASH_COLOR = "#ffd600"


class _BgCall(QRunnable):
    """Run ``func()`` on a thread-pool thread and report back via a callable.

    ``on_done`` receives ``(result, exception)``; exactly one is non-None.
    It is invoked from the worker thread — connect through a pyqtSignal if
    the callback needs to touch the GUI.
    """

    def __init__(self, func: Callable[[], object],
                 on_done: Callable[[object, Optional[BaseException]], None]) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._func = func
        self._on_done = on_done

    def run(self) -> None:  # noqa: D401 - QRunnable hook
        try:
            res = self._func()
        except BaseException as exc:  # noqa: BLE001
            self._on_done(None, exc)
            return
        self._on_done(res, None)


class _StatusDecoder:
    """Decode the QUEStionable condition register into a short label."""

    _BITS = {
        0: "OV", 1: "OC", 2: "PF", 3: "CP", 4: "OT",
        5: "MSP", 6: "", 7: "", 8: "", 9: "INH", 10: "UNR",
    }

    def decode(self, status: int) -> str:
        out = []
        for bit, name in self._BITS.items():
            if name and ((status >> bit) & 1):
                out.append(name)
        return " ".join(out)


class _SupplyRow(QWidget):
    """One row: label + value/action button.

    The button text and click handler change with the supply state — it
    doubles as the status indicator and the "fix it" action button.
    """

    def __init__(self, client: KeysightClient, ip: str, max_current: int,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._client = client
        self._ip = ip
        self._max_current = int(max_current)
        self._alert_threshold = ALERT_THRESHOLDS.get(self._max_current)
        self._decoder = _StatusDecoder()
        self._connected = False
        self._output_on: Optional[bool] = None
        self._status = 0
        self._err_str = ""
        # True while the last reading was above the alert threshold.
        self.alert = False
        self._build_ui()

    def _build_ui(self) -> None:
        self.value_btn = QPushButton("…")
        self.value_btn.clicked.connect(self._on_click)
        self.value_btn.setToolTip(
            f"{self._max_current} A supply at {self._ip}\n"
            "Click: reconnect when disconnected, turn on when OFF, "
            "clear protection when faulted."
        )
        font = QFont()
        font.setPointSize(FONTSIZE_PT)
        font.setBold(True)
        fixed_w = QFontMetrics(font).horizontalAdvance("000.00 A") + 18
        self.value_btn.setFixedWidth(fixed_w)

        # Small caption instead of "170 A supply current = ": the number is
        # what people read, the caption only says which supply it belongs to.
        caption = QLabel(f"{self._max_current} A")
        caption.setStyleSheet(f"font-size: {CAPTION_PT}pt; color: {theme.FG_MUTED};")
        caption.setToolTip(f"{self._max_current} A supply, {self._ip}")

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.layout.addWidget(caption)
        self.layout.addWidget(self.value_btn)

    # ------------------------------------------------------------------ #

    def apply_snapshot(self, snap: dict) -> None:
        self._connected = bool(snap.get("connected"))
        self._output_on = snap.get("output_on")
        self._status = int(snap.get("status") or 0)
        current = snap.get("current_a")
        self.alert = False

        if not self._connected:
            self._set_value("CXN_ERR", "orange")
            return
        if self._status:
            self._err_str = self._decoder.decode(self._status)
            self._set_value(self._err_str or f"STAT 0x{self._status:X}", "")
            return
        if self._output_on is False:
            self._set_value("OFF", "orange")
            return
        if current is None:
            self._set_value("…", "")
            return
        self.alert = (
            self._alert_threshold is not None
            and float(current) > self._alert_threshold
        )
        self._set_value(f"{float(current):1.2f} A", "red" if self.alert else "")

    def _set_value(self, text: str, bg: str) -> None:
        self.value_btn.setText(text)
        self.value_btn.setStyleSheet(
            f"font-weight: bold; font-size: {FONTSIZE_PT}pt; "
            f"text-align: right; padding: 2px 8px 2px 4px; "
            + (f"background-color: {bg};" if bg else "")
        )

    def _on_click(self) -> None:
        client = self._client
        ip = self._ip
        if not self._connected:
            func = lambda: client.reconnect(ip)
        elif self._output_on is False:
            func = lambda: client.turn_on(ip)
        else:
            # Connected + on: assume the click is to clear protection.
            func = lambda: client.clear_protect(ip)

        def _done(result, exc):
            if exc is not None:
                print(f"[Keysight] RPC failed for {ip}: {exc}")

        QThreadPool.globalInstance().start(_BgCall(func, _done))


class _CurrentPlot(QWidget):
    """Collapsible rolling plot of every supply's measured current.

    Fed from the snapshot the window is already polling, so it adds no
    hardware traffic.  Repeats of a cached sample are dropped using the
    server's ``seq`` counter; a ``None`` reading becomes a gap (NaN).
    Expanded by default; the toggle collapses it to just the header row.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._t: dict[str, deque] = {}
        self._y: dict[str, deque] = {}
        self._last_seq: dict[str, int] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._dirty = False

        self._toggle = QToolButton()
        self._toggle.setText("history")
        self._toggle.setToolTip("Show / hide the rolling current plot")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setArrowType(pg.QtCore.Qt.ArrowType.DownArrow)
        self._toggle.setToolButtonStyle(
            pg.QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle.setAutoRaise(True)
        self._toggle.toggled.connect(self._on_toggled)

        # "plot range = [  240] s" -- only shown while expanded.
        self._range = QSpinBox()
        self._range.setRange(PLOT_RANGE_MIN_S, PLOT_RANGE_MAX_S)
        self._range.setValue(PLOT_RANGE_DEFAULT_S)
        self._range.setSuffix(" s")
        self._range.setSingleStep(30)
        self._range.setKeyboardTracking(False)
        self._range.valueChanged.connect(self._on_range_changed)
        self._range.setToolTip("Visible time span of the history plot")
        self._range.setMaximumWidth(78)
        # Only shown while the plot is expanded; lives on the header row.
        self._range_row = self._range

        self._plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        # Grow with the dock instead of a fixed 160 px strip.
        self._plot.setMinimumHeight(PLOT_MIN_HEIGHT_PX)
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._plot.setLabel("left", "A")
        self._plot.getAxis("left").setWidth(40)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        # Top-left: the newest samples always sit at the right edge, so a
        # right-side legend would cover them.
        self._plot.addLegend(offset=(5, 5))
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.getPlotItem().setContentsMargins(0, 0, 4, 0)

        # This widget *is* the plot; the toggle + range are handed to the
        # window's header row via ``header_widgets()``.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._plot)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def header_widgets(self) -> tuple[QWidget, QWidget]:
        """(toggle button, range spinbox) for the caller to place inline."""
        return self._toggle, self._range_row

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    @property
    def range_s(self) -> float:
        return float(self._range.value())

    # ------------------------------------------------------------------ #

    def add_sample(self, ip: str, max_current: int, t: Optional[float],
                   current: Optional[float], seq: Optional[int]) -> None:
        if seq is not None and self._last_seq.get(ip) == seq:
            return  # cached snapshot unchanged since last poll
        if seq is not None:
            self._last_seq[ip] = seq
        if t is None:
            t = time.time()  # pre-``t`` server: fall back to receive time
        if ip not in self._t:
            self._t[ip] = deque()
            self._y[ip] = deque()
            pen = pg.mkPen(PLOT_PENS.get(int(max_current), "#ffffff"), width=1.5)
            self._curves[ip] = self._plot.plot(
                pen=pen, name=f"{max_current} A", connect="finite",
            )
        self._t[ip].append(float(t))
        self._y[ip].append(math.nan if current is None else float(current))
        cutoff = t - PLOT_RANGE_MAX_S
        while self._t[ip] and self._t[ip][0] < cutoff:
            self._t[ip].popleft()
            self._y[ip].popleft()
        self._dirty = True
        if self._plot.isVisible():
            self._redraw()

    def _redraw(self) -> None:
        if not self._dirty:
            return
        t_last = None
        for ip, curve in self._curves.items():
            curve.setData(list(self._t[ip]), list(self._y[ip]))
            if self._t[ip]:
                t_last = self._t[ip][-1] if t_last is None else max(t_last, self._t[ip][-1])
        if t_last is not None:
            self._plot.setXRange(t_last - self.range_s, t_last, padding=0.0)
        self._dirty = False

    def _on_range_changed(self, _value: int) -> None:
        self._dirty = True
        if self._plot.isVisible():
            self._redraw()

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(
            pg.QtCore.Qt.ArrowType.DownArrow if checked
            else pg.QtCore.Qt.ArrowType.RightArrow
        )
        self._range_row.setVisible(checked)
        self._plot.setVisible(checked)
        if checked:
            self._dirty = True
            self._redraw()


class KeysightClientWindow(QWidget):
    """Client GUI: discovers ``KeysightServer`` and renders one row per supply."""

    # Cross-thread signals so the worker-thread completion handler can
    # marshal results back onto the GUI thread.
    _snapshot_ready = pyqtSignal(object)   # list[dict]
    _snapshot_failed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._client: Optional[KeysightClient] = None
        self._rows: dict[str, _SupplyRow] = {}
        self._start_time = time.monotonic()
        self._ever_connected = False
        self._poll_in_flight = False
        self._connect_in_flight = False
        # Last time we *attempted* a reconnect.  Used to back off so we
        # don't spawn a discovery thread on every 500 ms tick while the
        # server is down.  ``-inf`` so the first tick always tries.
        self._last_reconnect_attempt = float("-inf")
        # Small, low-key status label — the dashboard's own server
        # indicator is the loud one.  We only fill this in if a real
        # problem persists past the startup grace window.
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(f"font-size: {FONTSIZE_PT - 4}pt; color: gray;")
        self._error_label.setWordWrap(True)
        self._error_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._error_label.setMinimumWidth(0)

        # Layout: one header row holding every supply (caption + value) on
        # the left and the history toggle + range on the right, then the
        # plot filling whatever height the dock gives us.  The old stacked
        # layout left most of the panel empty.
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(6, 4, 6, 4)
        self._root.setSpacing(4)
        self._header = QHBoxLayout()
        self._header.setSpacing(12)
        self._header.addStretch(1)
        self._plot = _CurrentPlot(self)
        toggle, rng = self._plot.header_widgets()
        self._header.addWidget(toggle)
        self._header.addWidget(rng)
        self._root.addLayout(self._header)
        self._root.addWidget(self._plot, 1)
        self._root.addWidget(self._error_label)
        self._error_label.hide()
        # Supply rows are inserted before the header stretch as they appear.
        self._n_rows = 0

        self._snapshot_ready.connect(self._on_snapshot_ready)
        self._snapshot_failed.connect(self._on_snapshot_failed)

        # Over-current attention flash.  ``_alert_since`` holds, per supply,
        # the monotonic time its reading first went above threshold; it is
        # dropped as soon as a reading comes back below (or the supply is
        # off / faulted / disconnected).  A missing server keeps the last
        # known state -- better to over-warn than to go quiet.
        self._alert_since: dict[str, float] = {}
        self._flashing = False
        self._flash_on = False
        self.setObjectName("keysightClientWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(FLASH_PERIOD_MS)
        self._flash_timer.timeout.connect(self._flash_tick)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(T_UPDATE_MS)
        # Kick once at startup so the user sees data quickly.
        QTimer.singleShot(50, self._refresh)

    # ------------------------------------------------------------------ #

    def _ensure_client_async(self) -> None:
        """Start a background ``KeysightClient`` construction if needed."""
        if self._client is not None or self._connect_in_flight:
            return
        # Throttle reconnects: once we've tried and failed, wait
        # ``RECONNECT_INTERVAL_S`` before trying again.  The poll timer
        # keeps ticking at ``T_UPDATE_MS`` but most ticks return early.
        now = time.monotonic()
        if (now - self._last_reconnect_attempt) < RECONNECT_INTERVAL_S:
            return
        self._last_reconnect_attempt = now
        self._connect_in_flight = True

        def _build():
            return KeysightClient(timeout_s=2.0, discovery_timeout=0.5)

        def _done(result, exc):
            # Worker thread — bounce result to GUI thread via signal.
            if exc is not None:
                self._snapshot_failed.emit(f"keysight server not found: {exc}")
            else:
                # Stash the client; main thread will pick it up next tick.
                self._client = result  # type: ignore[assignment]
                self._snapshot_failed.emit("")  # clear error on GUI thread
            # Clearing the flag from the worker thread is safe; it's just a bool.
            self._connect_in_flight = False

        QThreadPool.globalInstance().start(_BgCall(_build, _done))

    def _refresh(self) -> None:
        if self._client is None:
            self._ensure_client_async()
            return
        if self._poll_in_flight:
            return  # Don't pile up requests if the network is slow.
        self._poll_in_flight = True
        client = self._client

        def _fetch():
            return client.get_snapshot()

        def _done(result, exc):
            if exc is not None:
                self._snapshot_failed.emit(f"keysight server unreachable: {exc}")
            else:
                self._snapshot_ready.emit(result)

        QThreadPool.globalInstance().start(_BgCall(_fetch, _done))

    # ----- GUI-thread slots -------------------------------------------- #

    def _on_snapshot_ready(self, snapshot) -> None:
        self._poll_in_flight = False
        self._ever_connected = True
        self._hide_error()
        if isinstance(snapshot, list):
            self._apply(snapshot)

    def _on_snapshot_failed(self, msg: str) -> None:
        self._poll_in_flight = False
        if not msg:
            self._hide_error()
            return
        # Drop the client so the next tick reconnects from scratch.
        self._client = None
        self._maybe_show_error(msg)

    def _apply(self, snapshot: list) -> None:
        # Lazily build a row per supply on first snapshot.
        for snap in snapshot:
            if not isinstance(snap, dict):
                continue
            ip = str(snap.get("ip"))
            max_current = int(snap.get("max_current", 0))
            row = self._rows.get(ip)
            if row is None:
                row = _SupplyRow(self._client, ip, max_current, self)
                self._rows[ip] = row
                # Keep supplies left of the stretch, in arrival order.
                self._header.insertLayout(self._n_rows, row.layout)
                self._n_rows += 1
            row.apply_snapshot(snap)
            if snap.get("connected"):
                self._plot.add_sample(
                    ip, max_current, snap.get("t"), snap.get("current_a"),
                    snap.get("seq"),
                )
            if row.alert:
                self._alert_since.setdefault(ip, time.monotonic())
            else:
                self._alert_since.pop(ip, None)
        self._update_flash()

    # ----- over-current flash ------------------------------------------- #

    def overcurrent_ips(self) -> list[str]:
        """Supplies above threshold for longer than ``OVERCURRENT_FLASH_AFTER_S``."""
        now = time.monotonic()
        return [ip for ip, t0 in self._alert_since.items()
                if now - t0 > OVERCURRENT_FLASH_AFTER_S]

    def _update_flash(self) -> None:
        want = bool(self.overcurrent_ips())
        if want and not self._flashing:
            self._flashing = True
            self._flash_timer.start()
            self._flash_tick()
        elif not want and self._flashing:
            self._flashing = False
            self._flash_timer.stop()
            self._flash_on = False
            self.setStyleSheet("")

    def _flash_tick(self) -> None:
        self._flash_on = not self._flash_on
        self.setStyleSheet(
            f"QWidget#keysightClientWindow {{ background-color: {FLASH_COLOR}; }}"
            if self._flash_on else ""
        )
        if self._flash_on:
            # Flash the taskbar entry of whatever top-level window we live
            # in (the dashboard when embedded).  Re-issued every period with
            # a short duration so it stops soon after the alert clears.
            top = self.window()
            if top is not None:
                QApplication.alert(top, 2 * FLASH_PERIOD_MS)

    def _maybe_show_error(self, msg: str) -> None:
        # Stay silent until we've actually been connected at least once.
        # The dashboard's server-status LED is the loud indicator; this
        # in-panel label is only useful to flag a *new* disconnect.
        if not self._ever_connected:
            self._hide_error()
            return
        self._error_label.setText(msg)
        self._error_label.show()

    def _hide_error(self) -> None:
        if self._error_label.isVisible():
            self._error_label.hide()

    # ------------------------------------------------------------------ #

    def closeEvent(self, event):  # noqa: N802 - Qt-style
        self._timer.stop()
        self._flash_timer.stop()
        event.accept()


__all__ = ["KeysightClientWindow"]
