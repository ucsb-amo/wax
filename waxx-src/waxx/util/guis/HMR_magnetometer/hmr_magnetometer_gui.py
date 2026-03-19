#!/usr/bin/env python3
"""HMR2300 Magnetometer GUI — network client.

Connects to a running hmr_magnetometer_server and displays real-time
magnetic field data.  No serial port required on the GUI machine.

Usage::

    python hmr_magnetometer_gui.py [--host localhost] [--port 50000]
"""

import argparse
import csv
import math
import queue
import sys
import threading
import time
from collections import deque

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from waxx.util.guis.HMR_magnetometer.hmr_magnetometer_client import HMRClient

DEFAULT_SERVER_HOST = "localhost"
DEFAULT_SERVER_PORT = 50000
DEFAULT_POLL_INTERVAL = 0.10   # seconds between GET_SINCE calls
DEFAULT_TIME_WINDOW = 200.0
PLOT_BUFFER_MAXLEN = 2000
STATS_WINDOW = 200
DEFAULT_STATS_WINDOW_S = 30.0

class FixedPrecisionAxisItem(pg.AxisItem):
    def __init__(self, orientation, decimals=4, **kwargs):
        super().__init__(orientation=orientation, **kwargs)
        self.decimals = decimals

    def tickStrings(self, values, scale, spacing):
        return [f"{(value * scale): .{self.decimals}f}" for value in values]


class MagnetometerGUI(QtWidgets.QMainWindow):
    def __init__(
        self,
        server_host: str = DEFAULT_SERVER_HOST,
        server_port: int = DEFAULT_SERVER_PORT,
    ):
        super().__init__()
        self.setWindowTitle("HMR2300 Magnetometer")
        self.resize(1220, 900)
        self.setMinimumSize(980, 700)

        # State
        self.running = False
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.data_queue: queue.Queue = queue.Queue()
        self.log_data = []
        self.session_id = 0
        self.internal_ylim_update = False
        self.manual_ylim: dict = {}

        # Data buffers
        self.time_buffer = deque(maxlen=PLOT_BUFFER_MAXLEN)
        self.x_buffer = deque(maxlen=PLOT_BUFFER_MAXLEN)
        self.y_buffer = deque(maxlen=PLOT_BUFFER_MAXLEN)
        self.z_buffer = deque(maxlen=PLOT_BUFFER_MAXLEN)
        self.mag_buffer = deque(maxlen=PLOT_BUFFER_MAXLEN)
        self.x_stats = deque(maxlen=STATS_WINDOW)
        self.y_stats = deque(maxlen=STATS_WINDOW)
        self.z_stats = deque(maxlen=STATS_WINDOW)
        self.mag_stats = deque(maxlen=STATS_WINDOW)

        # Settings
        self.server_host = server_host
        self.server_port = server_port
        self.client = HMRClient(host=server_host, port=server_port)
        self.poll_interval = DEFAULT_POLL_INTERVAL
        self.window_s = DEFAULT_TIME_WINDOW
        self.stats_window_s = DEFAULT_STATS_WINDOW_S

        # Settings dialog widgets (None when dialog is closed)
        self.settings_dialog = None
        self.server_host_edit = None
        self.server_port_spin = None
        self.poll_spin = None

        # Reference lines (set by "Set Reference" button)
        self.ref_lines: dict = {}   # key -> pg.InfiniteLine or None
        self.ref_values: dict = {}  # key -> float or None

        # Plot handles
        self.plot_items: dict = {}
        self.curves: dict = {}
        self.overlay_items: dict = {}

        self._build_ui()

        self.queue_timer = QtCore.QTimer(self)
        self.queue_timer.timeout.connect(self._process_queue)
        self.queue_timer.start(100)

        QtCore.QTimer.singleShot(200, self.start_monitor)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- toolbar ---
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        root.addLayout(top)

        top.addWidget(QtWidgets.QLabel("Window (s):"))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.1, 1e6)
        self.window_spin.setDecimals(2)
        self.window_spin.setValue(self.window_s)
        self.window_spin.setSingleStep(5.0)
        self.window_spin.valueChanged.connect(self._on_window_changed)
        top.addWidget(self.window_spin)


        top.addWidget(QtWidgets.QLabel("Stats (s):"))
        self.stats_spin = QtWidgets.QDoubleSpinBox()
        self.stats_spin.setRange(0.1, 1e6)
        self.stats_spin.setDecimals(2)
        self.stats_spin.setValue(self.stats_window_s)
        self.stats_spin.setSingleStep(5.0)
        self.stats_spin.valueChanged.connect(self._on_stats_window_changed)
        top.addWidget(self.stats_spin)

        self.toggle_button = QtWidgets.QPushButton("Start")
        self.toggle_button.clicked.connect(self.toggle_monitor)
        top.addWidget(self.toggle_button)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_data)
        top.addWidget(clear_btn)

        save_btn = QtWidgets.QPushButton("Save CSV")
        save_btn.clicked.connect(self.save_log)
        top.addWidget(save_btn)

        settings_btn = QtWidgets.QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.open_settings_window)
        top.addWidget(settings_btn)

        ref_btn = QtWidgets.QPushButton("Set Reference")
        ref_btn.clicked.connect(self._set_reference)
        top.addWidget(ref_btn)

        clear_ref_btn = QtWidgets.QPushButton("Clear Reference")
        clear_ref_btn.clicked.connect(self._clear_reference)
        top.addWidget(clear_ref_btn)
        top.addStretch(1)

        # --- status bar ---
        status_row = QtWidgets.QHBoxLayout()
        status_row.setSpacing(12)
        root.addLayout(status_row)

        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.status_label.setMinimumWidth(420)
        status_row.addWidget(self.status_label)

        self.points_label = QtWidgets.QLabel("Points: 0")
        self.points_label.setMinimumWidth(90)
        status_row.addWidget(self.points_label)

        self.x_value_label = QtWidgets.QLabel("X = --- G")
        self.y_value_label = QtWidgets.QLabel("Y = --- G")
        self.z_value_label = QtWidgets.QLabel("Z = --- G")
        self.mag_value_label = QtWidgets.QLabel("|B| = --- G")

        self.x_value_label.setStyleSheet("color: #d62728; font-weight: 700;")
        self.y_value_label.setStyleSheet("color: #2ca02c; font-weight: 700;")
        self.z_value_label.setStyleSheet("color: #1f77b4; font-weight: 700;")
        self.mag_value_label.setStyleSheet("color: #6a3d9a; font-weight: 700;")

        status_row.addWidget(self.x_value_label)
        status_row.addWidget(self.y_value_label)
        status_row.addWidget(self.z_value_label)
        status_row.addWidget(self.mag_value_label)
        status_row.addStretch(1)

        # --- plots ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        root.addWidget(self.plot_widget, stretch=1)

        plot_defs = [
            ("x",   "#d62728", "Bx (G)"),
            ("y",   "#2ca02c", "By (G)"),
            ("z",   "#1f77b4", "Bz (G)"),
            ("mag", "#6a3d9a", "|B| (G)"),
        ]

        prev_plot = None
        for idx, (key, color, ylabel) in enumerate(plot_defs):
            p = self.plot_widget.addPlot(
                row=idx,
                col=0,
                axisItems={"left": FixedPrecisionAxisItem(orientation="left", decimals=4)},
            )
            p.setLabel("left", ylabel)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.getViewBox().setMouseEnabled(x=True, y=True)
            p.getViewBox().setDefaultPadding(0.02)
            p.getViewBox().invertX(True)

            if prev_plot is not None:
                p.setXLink(prev_plot)
            prev_plot = p

            if idx < len(plot_defs) - 1:
                p.hideAxis("bottom")
            else:
                p.setLabel("bottom", "Seconds ago (s)")

            curve = p.plot([], [], pen=pg.mkPen(color=color, width=1.8))
            overlay = pg.TextItem(
                html=self._format_overlay_html(),
                anchor=(0, 0),
                border=pg.mkPen(color=color, width=1.2),
                fill=pg.mkBrush(253, 253, 253, 235),
            )
            p.addItem(overlay)

            vb = p.getViewBox()
            vb.sigYRangeChanged.connect(lambda *_a, k=key: self._on_y_range_changed(k))
            vb.sigRangeChanged.connect(lambda *_a, k=key: self._position_overlay(k))

            self.plot_items[key] = p
            self.curves[key] = curve
            self.overlay_items[key] = overlay
            self.manual_ylim[key] = False
            self.ref_lines[key] = None
            self.ref_values[key] = None

        self._reset_display()

    # ------------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------------

    def _set_reference(self):
        """Persist and plot a reference from mean values over the stats window."""
        now_s = time.time()
        cutoff = now_s - max(self.stats_window_s, 1e-9)
        timestamps = list(self.time_buffer)
        channels = {
            "x": list(self.x_buffer),
            "y": list(self.y_buffer),
            "z": list(self.z_buffer),
            "mag": list(self.mag_buffer),
        }

        means = {}
        for key, data in channels.items():
            values = [v for ts, v in zip(timestamps, data) if ts >= cutoff]
            if not values:
                self._set_status("Failed to set reference: no samples in stats window")
                return
            means[key] = sum(values) / len(values)

        try:
            response = self.client._set_reference_values(
                means["x"],
                means["y"],
                means["z"],
                means["mag"],
                timeout=2.0,
            )
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Server returned error"))
            ref = response["reference"]
        except Exception as exc:
            self._set_status(f"Failed to set reference: {exc}")
            return

        latest = {
            "x":   float(ref["Bx"]),
            "y":   float(ref["By"]),
            "z":   float(ref["Bz"]),
            "mag": float(ref["Btot"]),
        }
        colors = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4", "mag": "#6a3d9a"}

        for key, val in latest.items():
            if val is None:
                continue
            # Remove old line if present
            if self.ref_lines[key] is not None:
                self.plot_items[key].removeItem(self.ref_lines[key])

            line = pg.InfiniteLine(
                pos=val,
                angle=0,
                pen=pg.mkPen(
                    color=colors[key],
                    width=1.5,
                    style=QtCore.Qt.PenStyle.DashLine,
                ),
                movable=False,
            )
            line.setOpacity(0.7)
            self.plot_items[key].addItem(line)
            self.ref_lines[key] = line
            self.ref_values[key] = val

        self._set_status(
            "Reference set: "
            + "  ".join(
                f"{k.upper()}={v:+.6f} G"
                for k, v in latest.items()
                if v is not None
            )
        )

    def _clear_reference(self):
        """Remove reference lines from all plots."""
        for key in list(self.ref_lines):
            if self.ref_lines[key] is not None:
                self.plot_items[key].removeItem(self.ref_lines[key])
                self.ref_lines[key] = None
                self.ref_values[key] = None
        self._set_status("Reference cleared")

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_overlay_html(value=None, mean=None, std_ug=None):
        value_s = "---" if value is None else f"{value:+.6f}"
        mean_s  = "---" if mean  is None else f"{mean:+.6f}"
        std_s   = "---" if std_ug is None else f"{std_ug:.1f}"
        return (
            "<span style='font-family: Consolas, \"Courier New\", monospace; font-size: 10pt;'>"
            "<table cellspacing='0' cellpadding='0'>"
            f"<tr><td style='text-align:left; padding-right:8px;'>value</td>"
            f"<td style='text-align:right; min-width:110px;'>{value_s}</td><td>&nbsp;G</td></tr>"
            f"<tr><td style='text-align:left; padding-right:8px;'>mean</td>"
            f"<td style='text-align:right; min-width:110px;'>{mean_s}</td><td>&nbsp;G</td></tr>"
            f"<tr><td style='text-align:left; padding-right:8px;'>std</td>"
            f"<td style='text-align:right; min-width:110px;'>{std_s}</td><td>&nbsp;uG</td></tr>"
            "</table></span>"
        )

    def _position_overlay(self, key):
        p = self.plot_items.get(key)
        overlay = self.overlay_items.get(key)
        if p is None or overlay is None:
            return
        xr, yr = p.viewRange()
        overlay.setPos(max(xr), max(yr))

    def _on_y_range_changed(self, key):
        if not self.internal_ylim_update:
            self.manual_ylim[key] = True
        self._position_overlay(key)

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------

    def open_settings_window(self):
        if self.settings_dialog is not None:
            self.settings_dialog.show()
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setModal(False)
        dlg.resize(380, 180)

        layout = QtWidgets.QGridLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        title = QtWidgets.QLabel("Server Connection")
        title.setStyleSheet("font-weight: 700;")
        layout.addWidget(title, 0, 0, 1, 4)

        layout.addWidget(QtWidgets.QLabel("Host:"), 1, 0)
        self.server_host_edit = QtWidgets.QLineEdit(self.server_host)
        layout.addWidget(self.server_host_edit, 1, 1)

        layout.addWidget(QtWidgets.QLabel("Port:"), 1, 2)
        self.server_port_spin = QtWidgets.QSpinBox()
        self.server_port_spin.setRange(1, 65535)
        self.server_port_spin.setValue(self.server_port)
        layout.addWidget(self.server_port_spin, 1, 3)

        layout.addWidget(QtWidgets.QLabel("Poll interval (s):"), 2, 0)
        self.poll_spin = QtWidgets.QDoubleSpinBox()
        self.poll_spin.setRange(0.05, 5.0)
        self.poll_spin.setDecimals(3)
        self.poll_spin.setSingleStep(0.05)
        self.poll_spin.setValue(self.poll_interval)
        layout.addWidget(self.poll_spin, 2, 1)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self._close_settings_window)
        layout.addWidget(close_btn, 3, 3)

        dlg.finished.connect(self._on_settings_closed)
        self.settings_dialog = dlg
        dlg.show()

    def _close_settings_window(self):
        if self.settings_dialog is not None:
            self._sync_settings()
            self.settings_dialog.close()

    def _on_settings_closed(self, _result):
        self._sync_settings()
        self.settings_dialog = None
        self.server_host_edit = None
        self.server_port_spin = None
        self.poll_spin = None

    def _sync_settings(self):
        self.window_s = float(self.window_spin.value())
        if self.server_host_edit is not None:
            txt = self.server_host_edit.text().strip()
            if txt:
                self.server_host = txt
                self.client.host = txt
        if self.server_port_spin is not None:
            self.server_port = int(self.server_port_spin.value())
            self.client.port = self.server_port
        if self.poll_spin is not None:
            self.poll_interval = max(float(self.poll_spin.value()), 0.05)

    # ------------------------------------------------------------------
    # Monitor start / stop
    # ------------------------------------------------------------------

    def toggle_monitor(self):
        if self.running:
            self.stop_monitor()
        else:
            self.start_monitor()

    def start_monitor(self):
        if self.running or (self.worker_thread and self.worker_thread.is_alive()):
            return

        self._sync_settings()
        self.stop_event.clear()
        self._reset_display()

        # Quick connectivity check before starting the worker thread
        try:
            self.client._ping(timeout=2.0)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Connection Error",
                f"Cannot reach server at {self.server_host}:{self.server_port}\n\n{exc}",
            )
            self._set_status("Connection failed")
            return

        self.running = True
        self.session_id += 1
        self.worker_thread = threading.Thread(
            target=self._read_loop,
            args=(self.session_id,),
            daemon=True,
        )
        self.worker_thread.start()
        self.toggle_button.setText("Stop")
        self._set_status(f"Connected to {self.server_host}:{self.server_port}")

    def stop_monitor(self):
        self.running = False
        self.stop_event.set()
        self.session_id += 1
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        self.worker_thread = None
        self.toggle_button.setText("Start")
        self._set_status("Stopped")

    # ------------------------------------------------------------------
    # Worker thread — polls server with GET_SINCE
    # ------------------------------------------------------------------

    def _read_loop(self, session_id: int):
        last_t = 0.0
        consec_errors = 0

        while not self.stop_event.is_set() and session_id == self.session_id:
            try:
                result = self.client._get_since(last_t, timeout=3.0)
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "Server returned error"))

                for reading in result.get("readings", []):
                    if self.stop_event.is_set() or session_id != self.session_id:
                        return
                    last_t = max(last_t, reading["t"])
                    self.data_queue.put({**reading, "session_id": session_id})

                consec_errors = 0

            except Exception as exc:
                if self.stop_event.is_set() or session_id != self.session_id:
                    return
                consec_errors += 1
                self.data_queue.put({
                    "session_id": session_id,
                    "warning": f"Server poll error ({consec_errors}): {exc}",
                })
                if consec_errors >= 10:
                    self.data_queue.put({
                        "session_id": session_id,
                        "error": "Too many consecutive server errors — stopping.",
                    })
                    return

            self.stop_event.wait(self.poll_interval)

    # ------------------------------------------------------------------
    # Queue processing (Qt timer, 100 ms)
    # ------------------------------------------------------------------

    def _process_queue(self):
        updated = False
        try:
            while not self.data_queue.empty():
                item = self.data_queue.get_nowait()
                if item.get("session_id") != self.session_id:
                    continue

                if "warning" in item:
                    self._set_status(item["warning"])
                    continue

                if "error" in item:
                    self._set_status(f"Error: {item['error']}")
                    self.stop_monitor()
                    break

                x    = item["Bx"]
                y    = item["By"]
                z    = item["Bz"]
                btot = item["Btot"]
                t    = item["t"]

                self.log_data.append({"timestamp_s": t, "x_G": x, "y_G": y,
                                      "z_G": z, "btot_G": btot})
                self.time_buffer.append(t)
                self.x_buffer.append(x)
                self.y_buffer.append(y)
                self.z_buffer.append(z)
                self.mag_buffer.append(btot)
                self.x_value_label.setText(f"X = {x:+.6f} G")
                self.y_value_label.setText(f"Y = {y:+.6f} G")
                self.z_value_label.setText(f"Z = {z:+.6f} G")
                self.mag_value_label.setText(f"|B| = {btot:+.6f} G")
                self.points_label.setText(f"Points: {len(self.log_data)}")
                updated = True

            self._update_stats()
            if updated:
                self._update_plot()

        except Exception as exc:
            self._set_status(f"GUI error: {exc}")

    # ------------------------------------------------------------------
    # Plot & stats updates
    # ------------------------------------------------------------------

    def _update_stats(self):
        if len(self.time_buffer) < 2:
            for key in ("x", "y", "z", "mag"):
                self.overlay_items[key].setHtml(self._format_overlay_html())
                self._position_overlay(key)
            return

        now_s = time.time()
        cutoff = now_s - max(self.stats_window_s, 1e-9)
        timestamps = list(self.time_buffer)
        all_data = {
            "x":   list(self.x_buffer),
            "y":   list(self.y_buffer),
            "z":   list(self.z_buffer),
            "mag": list(self.mag_buffer),
        }
        latest = {
            "x":   self.x_buffer[-1]   if self.x_buffer   else None,
            "y":   self.y_buffer[-1]   if self.y_buffer   else None,
            "z":   self.z_buffer[-1]   if self.z_buffer   else None,
            "mag": self.mag_buffer[-1] if self.mag_buffer else None,
        }

        for key in ("x", "y", "z", "mag"):
            values = [v for ts, v in zip(timestamps, all_data[key]) if ts >= cutoff]
            if len(values) < 2:
                self.overlay_items[key].setHtml(self._format_overlay_html())
                self._position_overlay(key)
                continue
            mean_v = sum(values) / len(values)
            std_ug = (sum((v - mean_v) ** 2 for v in values) / len(values)) ** 0.5 * 1e6
            val    = latest[key] if latest[key] is not None else mean_v
            self.overlay_items[key].setHtml(self._format_overlay_html(val, mean_v, std_ug))
            self._position_overlay(key)

    def _update_plot(self):
        if not self.time_buffer or self.isMinimized():
            return

        self._sync_settings()
        window = max(float(self.window_s), 1e-9)

        timestamps = list(self.time_buffer)
        data = [list(self.x_buffer), list(self.y_buffer),
                list(self.z_buffer), list(self.mag_buffer)]

        now_s = time.time()
        ages  = [now_s - ts for ts in timestamps]

        # trim to window
        i0 = 0
        while i0 < len(ages) and ages[i0] > window:
            i0 += 1
        ages = ages[i0:]
        data = [d[i0:] for d in data]

        self.internal_ylim_update = True
        try:
            for key, values in zip(("x", "y", "z", "mag"), data):
                p = self.plot_items[key]
                self.curves[key].setData(ages, values)
                p.setXRange(0, window, padding=0)

                if not self.manual_ylim[key] and values:
                    ymin, ymax = min(values), max(values)
                    pad = 0.05 if abs(ymax - ymin) < 1e-12 else 0.1 * (ymax - ymin)
                    if abs(ymax - ymin) < 1e-12 and abs(ymin) >= 1:
                        pad = abs(ymin) * 0.05
                    p.setYRange(ymin - pad, ymax + pad, padding=0)
                self._position_overlay(key)
        finally:
            self.internal_ylim_update = False

    def _on_window_changed(self, value):
        self.window_s = float(value)
        self._update_plot()

    def _on_stats_window_changed(self, value):
        self.stats_window_s = float(value)
        self._update_stats()

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def _reset_display(self):
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        for buf in (self.time_buffer, self.x_buffer, self.y_buffer, self.z_buffer,
                    self.mag_buffer):
            buf.clear()
        self.log_data.clear()

        self.internal_ylim_update = True
        for key, p in self.plot_items.items():
            self.curves[key].setData([], [])
            p.setXRange(0, 1, padding=0)
            p.setYRange(-1, 1, padding=0)
            self.manual_ylim[key] = False
            self.overlay_items[key].setHtml(self._format_overlay_html())
            self._position_overlay(key)
        self.internal_ylim_update = False

        self.points_label.setText("Points: 0")
        self.x_value_label.setText("X = --- G")
        self.y_value_label.setText("Y = --- G")
        self.z_value_label.setText("Z = --- G")
        self.mag_value_label.setText("|B| = --- G")

    def clear_data(self):
        was_running = self.running
        poll = self.poll_interval
        if was_running:
            self.stop_monitor()
        self._reset_display()
        if was_running:
            self.poll_interval = poll
            self.start_monitor()
        self._set_status("Data cleared")

    def save_log(self):
        if not self.log_data:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to save.")
            return

        default_name = f"hmr2300_log_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Log", default_name, "CSV files (*.csv);;All files (*.*)"
        )
        if not path:
            return

        fields = ["timestamp_s", "x_G", "y_G", "z_G", "btot_G"]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in self.log_data:
                    writer.writerow({k: row[k] for k in fields})
            self._set_status(f"Saved: {path}")
            QtWidgets.QMessageBox.information(self, "Saved", f"Log saved to:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(exc))

    def _set_status(self, text: str):
        self.status_label.setText(f"Status: {text}")

    def shutdown(self):
        self.stop_event.set()
        self.running = False
        self._close_settings_window()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)