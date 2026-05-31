"""Client-side Keysight monitor GUI.

This widget talks ONLY to ``KeysightServer`` over TCP — it never opens a
direct VXI11 connection to the supplies.  Multiple dashboards can run
this widget concurrently without spamming the hardware.

All network I/O (discovery, snapshot polling, action RPCs) is dispatched
to ``QThreadPool`` so the GUI thread never stalls; that keeps window
moves smooth even when the server is missing or slow.
"""
from __future__ import annotations

from typing import Callable, Optional

import time

from PyQt6.QtCore import QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from waxx.util.guis.keysight.keysight_client import KeysightClient

T_UPDATE_MS = 500
FONTSIZE_PT = 14

# Stay quiet for this long after construction before surfacing a
# "server not found" message — the supervised server subprocess often
# takes a few seconds to spool up after the dashboard launches.
STARTUP_GRACE_S = 15.0

# Per-supply over-current alert thresholds (A), keyed by ``max_current``.
ALERT_THRESHOLDS: dict[int, float] = {500: 100, 170: 50}


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
        self._build_ui()

    def _build_ui(self) -> None:
        self.value_btn = QPushButton("…")
        self.value_btn.clicked.connect(self._on_click)
        font = QFont()
        font.setPointSize(FONTSIZE_PT)
        font.setBold(True)
        fixed_w = QFontMetrics(font).horizontalAdvance("000.00 A") + 20
        self.value_btn.setFixedWidth(fixed_w)

        text_label = QLabel(f"{self._max_current} A supply current = ")
        text_label.setStyleSheet(f"font-size: {FONTSIZE_PT}pt;")

        self.layout = QHBoxLayout()
        self.layout.addWidget(text_label)
        self.layout.addWidget(self.value_btn)

    # ------------------------------------------------------------------ #

    def apply_snapshot(self, snap: dict) -> None:
        self._connected = bool(snap.get("connected"))
        self._output_on = snap.get("output_on")
        self._status = int(snap.get("status") or 0)
        current = snap.get("current_a")

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
        alert = (
            self._alert_threshold is not None
            and float(current) > self._alert_threshold
        )
        self._set_value(f"{float(current):1.2f} A", "red" if alert else "")

    def _set_value(self, text: str, bg: str) -> None:
        self.value_btn.setText(text)
        self.value_btn.setStyleSheet(
            f"font-weight: bold; font-size: {FONTSIZE_PT}pt; "
            f"text-align: right; padding-right: 10px; "
            f"background-color: {bg};"
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
        # Small, low-key status label — the dashboard's own server
        # indicator is the loud one.  We only fill this in if a real
        # problem persists past the startup grace window.
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(f"font-size: {FONTSIZE_PT - 4}pt; color: gray;")

        self._root = QVBoxLayout(self)
        self._root.addWidget(self._error_label)
        self._error_label.hide()

        self._snapshot_ready.connect(self._on_snapshot_ready)
        self._snapshot_failed.connect(self._on_snapshot_failed)

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
            row = self._rows.get(ip)
            if row is None:
                row = _SupplyRow(self._client, ip, int(snap.get("max_current", 0)), self)
                self._rows[ip] = row
                self._root.addLayout(row.layout)
            row.apply_snapshot(snap)

    def _maybe_show_error(self, msg: str) -> None:
        # Suppress the message during the startup grace period so the
        # GUI doesn't shout while the server subprocess is still booting.
        if not self._ever_connected:
            if (time.monotonic() - self._start_time) < STARTUP_GRACE_S:
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
        event.accept()


__all__ = ["KeysightClientWindow"]
