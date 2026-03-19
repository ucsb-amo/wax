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
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from waxx.util.guis.HMR_magnetometer.hmr_magnetometer_client import HMRClient

DEFAULT_SERVER_HOST = "localhost"
DEFAULT_SERVER_PORT = 50000
DEFAULT_POLL_INTERVAL = 0.10   # seconds between GET_SINCE calls
DEFAULT_TIME_WINDOW = 60.0
PLOT_BUFFER_MAXLEN = 2000
STATS_WINDOW = 200
DEFAULT_STATS_WINDOW_S = 10.0
READOUT_PANEL_WIDTH = 300

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
        reference_csv_path=None,
    ):
        super().__init__()
        self.setWindowTitle("HMR2300 Magnetometer")
        # self.resize(1220, 900)
        # self.setMinimumSize(980, 700)

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

        # Settings
        self.server_host = server_host
        self.server_port = server_port
        self.client = HMRClient(host=server_host, port=server_port)
        self.poll_interval = DEFAULT_POLL_INTERVAL
        self.window_s = DEFAULT_TIME_WINDOW
        self.stats_window_s = DEFAULT_STATS_WINDOW_S
        self.reference_csv_path = reference_csv_path

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
        self.readout_labels: dict = {}

        # Popup log widgets
        self.log_dialog = None
        self.log_text = None
        self.log_lines = deque(maxlen=2000)

        # Reference summary label
        self.reference_info_label = None

        self._build_ui()
        self._auto_load_latest_reference()

        self.queue_timer = QtCore.QTimer(self)
        self.queue_timer.timeout.connect(self._process_queue)
        self.queue_timer.start(100)

        QtCore.QTimer.singleShot(200, self.start_monitor)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.setStyleSheet(
            "QMainWindow { background-color: #f3f5f7; }"
            "QLabel { color: #20262e; }"
            "QPushButton { background: #ffffff; border: 1px solid #c7cfd9; border-radius: 6px; padding: 5px 10px; }"
            "QPushButton:hover { background: #eef3ff; border-color: #8aa2ff; }"
            "QGroupBox { border: 1px solid #cfd6df; border-radius: 8px; margin-top: 8px; font-weight: 600; }"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; top: 1px; padding: 0 4px 0 4px; }"
        )

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- toolbar ---
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(10)
        top.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(top)

        time_group = QtWidgets.QGroupBox("Time")
        time_group.setStyleSheet(
            "QGroupBox { border: 1px solid #cfd6df; border-radius: 8px; margin-top: 8px; font-weight: 700; }"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; top: 1px; padding: 0 4px 0 4px; color: #506070; }"
        )
        time_layout = QtWidgets.QGridLayout(time_group)
        time_layout.setContentsMargins(10, 10, 10, 8)
        time_layout.setHorizontalSpacing(8)
        time_layout.setVerticalSpacing(6)

        time_layout.addWidget(QtWidgets.QLabel("Window (s):"), 0, 0)
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.1, 1e6)
        self.window_spin.setDecimals(2)
        self.window_spin.setValue(self.window_s)
        self.window_spin.setSingleStep(5.0)
        self.window_spin.valueChanged.connect(self._on_window_changed)
        time_layout.addWidget(self.window_spin, 0, 1)

        time_layout.addWidget(QtWidgets.QLabel("Statistics (s):"), 1, 0)
        self.stats_spin = QtWidgets.QDoubleSpinBox()
        self.stats_spin.setRange(0.1, 1e6)
        self.stats_spin.setDecimals(2)
        self.stats_spin.setValue(self.stats_window_s)
        self.stats_spin.setSingleStep(5.0)
        self.stats_spin.valueChanged.connect(self._on_stats_window_changed)
        time_layout.addWidget(self.stats_spin, 1, 1)
        top.addWidget(time_group, 1)

        action_group = QtWidgets.QGroupBox("Run & Data")
        action_group.setStyleSheet(
            "QGroupBox { border: 1px solid #cfd6df; border-radius: 8px; margin-top: 8px; font-weight: 700; }"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; top: 1px; padding: 0 4px 0 4px; color: #506070; }"
        )
        action_layout = QtWidgets.QHBoxLayout(action_group)
        action_layout.setContentsMargins(10, 10, 10, 8)
        action_layout.setSpacing(6)

        self.toggle_button = QtWidgets.QPushButton("Start")
        self.toggle_button.clicked.connect(self.toggle_monitor)
        action_layout.addWidget(self.toggle_button)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_data)
        action_layout.addWidget(clear_btn)

        save_btn = QtWidgets.QPushButton("Save CSV")
        save_btn.clicked.connect(self.save_log)
        action_layout.addWidget(save_btn)

        log_btn = QtWidgets.QPushButton("Open Log")
        log_btn.clicked.connect(self._open_log_dialog)
        action_layout.addWidget(log_btn)
        top.addWidget(action_group, 1)

        ref_group = QtWidgets.QGroupBox("Reference")
        ref_group_layout = QtWidgets.QVBoxLayout(ref_group)
        ref_group_layout.setContentsMargins(8, 12, 8, 8)
        ref_group_layout.setSpacing(6)

        ref_group_layout.addSpacing(2)

        ref_row = QtWidgets.QHBoxLayout()
        ref_row.setSpacing(4)
        ref_group_layout.addLayout(ref_row)

        ref_btn = QtWidgets.QPushButton("Set")
        ref_btn.clicked.connect(self._set_reference)
        ref_btn.setFixedHeight(24)
        ref_btn.setMinimumWidth(48)
        ref_row.addWidget(ref_btn)

        load_ref_btn = QtWidgets.QPushButton("Load")
        load_ref_btn.clicked.connect(self._open_load_reference_dialog)
        load_ref_btn.setFixedHeight(24)
        load_ref_btn.setMinimumWidth(48)
        ref_row.addWidget(load_ref_btn)

        clear_ref_btn = QtWidgets.QPushButton("Clear")
        clear_ref_btn.clicked.connect(self._clear_reference)
        clear_ref_btn.setFixedHeight(24)
        clear_ref_btn.setMinimumWidth(48)
        ref_row.addWidget(clear_ref_btn)

        self.reference_info_label = QtWidgets.QLabel("loaded reference: --")
        self.reference_info_label.setStyleSheet("color: #6f7a87; font-size: 11px;")
        self.reference_info_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        ref_group_layout.addWidget(self.reference_info_label)

        top.addWidget(ref_group, 1)

        settings_btn = QtWidgets.QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.open_settings_window)
        settings_btn.setFixedHeight(
            max(
                time_group.sizeHint().height(),
                action_group.sizeHint().height(),
                ref_group.sizeHint().height(),
            )
        )
        settings_btn.setStyleSheet(
            "QPushButton { background: #ffffff; border: 1px solid #bcc7d6; border-radius: 7px; padding: 5px 12px; font-weight: 600; }"
            "QPushButton:hover { background: #eef3ff; border-color: #8aa2ff; }"
        )
        top.addWidget(settings_btn, 0, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # --- plots + left readout cards (one row per channel) ---
        rows = QtWidgets.QVBoxLayout()
        rows.setSpacing(8)
        root.addLayout(rows, stretch=1)

        plot_defs = [
            ("x",   "#d62728", "Bx (G)"),
            ("y",   "#2ca02c", "By (G)"),
            ("z",   "#1f77b4", "Bz (G)"),
            ("mag", "#6a3d9a", "|B| (G)"),
        ]

        prev_plot = None
        for idx, (key, color, ylabel) in enumerate(plot_defs):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(8)
            rows.addLayout(row, stretch=1)

            card = QtWidgets.QGroupBox(ylabel.replace(" (G)", ""))
            card.setFixedWidth(READOUT_PANEL_WIDTH)
            card.setStyleSheet(
                f"QGroupBox {{ border: 1px solid {color}; border-radius: 8px; margin-top: 8px; font-weight: 700; }}"
                f"QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 10px; top: 1px; padding: 0 4px 0 4px; color: {color}; }}"
            )
            card_layout = QtWidgets.QGridLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setHorizontalSpacing(8)
            card_layout.setVerticalSpacing(2)

            def add_stat_row(name, unit, grid_row, value_style=""):
                name_label = QtWidgets.QLabel(name)
                value_label = QtWidgets.QLabel("---")
                unit_label = QtWidgets.QLabel(unit)
                value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                unit_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                if value_style:
                    value_label.setStyleSheet(value_style)
                else:
                    value_label.setStyleSheet("color: #4c596a;")
                unit_label.setStyleSheet("color: #6f7a87;")
                card_layout.addWidget(name_label, grid_row, 0)
                card_layout.addWidget(value_label, grid_row, 1)
                card_layout.addWidget(unit_label, grid_row, 2)
                return value_label

            val_label = add_stat_row("Value", "G", 0, f"color: {color}; font-weight: 700;")
            mean_label = add_stat_row("Mean", "G", 1)
            std_label = add_stat_row("σ", "uG", 2)
            delta_label = add_stat_row("Δ<sub>ref</sub>", "mG", 3)

            row.addWidget(card, stretch=0)

            p = pg.PlotWidget(axisItems={"left": FixedPrecisionAxisItem(orientation="left", decimals=4)})
            p.getAxis("left").enableAutoSIPrefix(False)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.getViewBox().setMouseEnabled(x=True, y=True)
            p.getViewBox().setDefaultPadding(0.02)
            p.getViewBox().invertX(True)
            row.addWidget(p, stretch=1)

            if prev_plot is not None:
                p.setXLink(prev_plot)
            prev_plot = p

            if idx < len(plot_defs) - 1:
                p.hideAxis("bottom")

            curve = p.plot([], [], pen=pg.mkPen(color=color, width=1.8))

            vb = p.getViewBox()
            vb.sigYRangeChanged.connect(lambda *_a, k=key: self._on_y_range_changed(k))

            self.plot_items[key] = p
            self.curves[key] = curve
            self.manual_ylim[key] = False
            self.ref_lines[key] = None
            self.ref_values[key] = None

            self.readout_labels[key] = {
                "value": val_label,
                "mean": mean_label,
                "std": std_label,
                "delta": delta_label,
            }

        x_label_row = QtWidgets.QHBoxLayout()
        x_label_row.setSpacing(8)
        root.addLayout(x_label_row)

        left_spacer = QtWidgets.QWidget()
        left_spacer.setFixedWidth(READOUT_PANEL_WIDTH)
        x_label_row.addWidget(left_spacer)

        x_label = QtWidgets.QLabel("Seconds ago")
        x_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        x_label.setStyleSheet("color: #4c596a; font-weight: 600;")
        x_label_row.addWidget(x_label, stretch=1)

        self._reset_display()

    # ------------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------------

    @staticmethod
    def _reference_value_map(reference):
        return {
            "x": float(reference["Bx"]),
            "y": float(reference["By"]),
            "z": float(reference["Bz"]),
            "mag": float(reference["Btot"]),
        }

    def _apply_reference_to_plots(self, reference, status_prefix="Reference loaded"):
        latest = self._reference_value_map(reference)
        colors = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4", "mag": "#6a3d9a"}

        for key, val in latest.items():
            if val is None:
                continue

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

        ts = reference.get("timestamp_s")
        dt_text = self._format_reference_datetime(reference)
        self._set_reference_info_text(reference)
        suffix = f" ({dt_text})" if dt_text else ""
        if ts is not None:
            suffix += f" [t={float(ts):.3f}]"

        self._set_status(
            status_prefix
            + suffix
            + ": "
            + "  ".join(
                f"{k.upper()}={v:+.6f} G"
                for k, v in latest.items()
                if v is not None
            )
        )
        self._update_plot()

    @staticmethod
    def _format_reference_datetime(reference):
        iso_text = str(reference.get("datetime_iso", "")).strip()
        if iso_text:
            try:
                return datetime.fromisoformat(iso_text).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        t_val = reference.get("timestamp_s")
        try:
            if t_val is not None and math.isfinite(float(t_val)):
                return datetime.fromtimestamp(float(t_val)).strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError, OSError):
            return ""
        return ""

    def _set_reference_info_text(self, reference=None):
        if self.reference_info_label is None:
            return
        if reference is None:
            self.reference_info_label.setText("loaded reference: --")
            return

        iso_text = str(reference.get("datetime_iso", "")).strip()
        if iso_text:
            try:
                dt = datetime.fromisoformat(iso_text)
                self.reference_info_label.setText(
                    f"loaded reference: {dt.strftime('%Y-%m-%d %H:%M')}"
                )
                return
            except ValueError:
                pass

        t_val = reference.get("timestamp_s")
        try:
            if t_val is not None and math.isfinite(float(t_val)):
                self.reference_info_label.setText(
                    f"loaded reference: {datetime.fromtimestamp(float(t_val)).strftime('%Y-%m-%d %H:%M')}"
                )
                return
        except (TypeError, ValueError, OSError):
            pass

        self.reference_info_label.setText("loaded reference: unknown")

    def _load_references_from_csv(self):
        if not self.reference_csv_path:
            raise RuntimeError("Reference CSV path is not configured")
        if not os.path.exists(self.reference_csv_path):
            raise FileNotFoundError(self.reference_csv_path)

        refs = []
        with open(self.reference_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ref = {
                        "datetime_iso": row.get("datetime_iso", ""),
                        "timestamp_s": float(row["timestamp_s"]),
                        "Bx": float(row["Bx"]),
                        "By": float(row["By"]),
                        "Bz": float(row["Bz"]),
                        "Btot": float(row["Btot"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
                refs.append(ref)

        refs.sort(key=lambda r: r["timestamp_s"], reverse=True)
        return refs

    def _write_references_to_csv(self, references):
        if not self.reference_csv_path:
            raise RuntimeError("Reference CSV path is not configured")
        os.makedirs(os.path.dirname(os.path.abspath(self.reference_csv_path)), exist_ok=True)
        with open(self.reference_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["datetime_iso", "timestamp_s", "Bx", "By", "Bz", "Btot"],
            )
            writer.writeheader()
            for ref in references:
                writer.writerow(
                    {
                        "datetime_iso": ref.get("datetime_iso", ""),
                        "timestamp_s": float(ref["timestamp_s"]),
                        "Bx": float(ref["Bx"]),
                        "By": float(ref["By"]),
                        "Bz": float(ref["Bz"]),
                        "Btot": float(ref["Btot"]),
                    }
                )

    def _open_load_reference_dialog(self):
        try:
            references = self._load_references_from_csv()
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(
                self,
                "Reference CSV Missing",
                f"Could not find reference CSV:\n{self.reference_csv_path}",
            )
            return
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load Reference Error", str(exc))
            return

        if not references:
            QtWidgets.QMessageBox.information(
                self,
                "No References",
                "No valid references found in CSV.",
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Load Reference")
        dlg.setModal(True)
        dlg.resize(640, 420)

        layout = QtWidgets.QVBoxLayout(dlg)
        help_label = QtWidgets.QLabel(
            "Select a saved reference to load. Use X to delete an entry from the CSV."
        )
        layout.addWidget(help_label)

        table = QtWidgets.QTableWidget(dlg)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Date", "|B| (G)", ""])
        table.verticalHeader().setVisible(False)
        sel_behavior = getattr(
            QtWidgets.QAbstractItemView,
            "SelectionBehavior",
            QtWidgets.QAbstractItemView,
        )
        sel_mode = getattr(
            QtWidgets.QAbstractItemView,
            "SelectionMode",
            QtWidgets.QAbstractItemView,
        )
        edit_trigger = getattr(
            QtWidgets.QAbstractItemView,
            "EditTrigger",
            QtWidgets.QAbstractItemView,
        )
        table.setSelectionBehavior(getattr(sel_behavior, "SelectRows"))
        table.setSelectionMode(getattr(sel_mode, "SingleSelection"))
        table.setEditTriggers(getattr(edit_trigger, "NoEditTriggers"))
        table.horizontalHeader().setStretchLastSection(False)
        header_resize_mode = getattr(QtWidgets.QHeaderView, "ResizeMode", QtWidgets.QHeaderView)
        table.horizontalHeader().setSectionResizeMode(0, getattr(header_resize_mode, "ResizeToContents"))
        table.horizontalHeader().setSectionResizeMode(1, getattr(header_resize_mode, "Stretch"))
        table.horizontalHeader().setSectionResizeMode(2, getattr(header_resize_mode, "ResizeToContents"))
        layout.addWidget(table)

        def _refresh_table():
            table.setRowCount(0)
            for idx, ref in enumerate(references):
                table.insertRow(idx)

                dt_text = self._format_reference_datetime(ref) or "(unknown date)"
                date_item = QtWidgets.QTableWidgetItem(dt_text)
                date_item.setData(QtCore.Qt.ItemDataRole.UserRole, ref)
                table.setItem(idx, 0, date_item)

                mag_item = QtWidgets.QTableWidgetItem(f"{float(ref['Btot']):+.6f}")
                mag_item.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                mag_item.setData(QtCore.Qt.ItemDataRole.UserRole, ref)
                table.setItem(idx, 1, mag_item)

                del_btn = QtWidgets.QPushButton("X")
                del_btn.setFixedWidth(28)
                del_btn.setStyleSheet(
                    "QPushButton { background: #fdeaea; border: 1px solid #efb1b1; color: #a33a3a; border-radius: 5px; padding: 2px 0; }"
                    "QPushButton:hover { background: #f9d8d8; border-color: #e28c8c; }"
                )

                def _delete_ref(_checked=False, ts=ref["timestamp_s"], dt=ref.get("datetime_iso", "")):
                    for i, r in enumerate(list(references)):
                        if float(r["timestamp_s"]) == float(ts) and str(r.get("datetime_iso", "")) == str(dt):
                            references.pop(i)
                            break
                    self._write_references_to_csv(references)
                    _refresh_table()

                del_btn.clicked.connect(_delete_ref)
                table.setCellWidget(idx, 2, del_btn)

            if references:
                table.selectRow(0)

        _refresh_table()

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        load_btn = QtWidgets.QPushButton("Load")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_row.addWidget(load_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def _load_selected_reference():
            row = table.currentRow()
            if row < 0:
                QtWidgets.QMessageBox.information(dlg, "No Selection", "Select a reference first.")
                return
            item = table.item(row, 0)
            if item is None:
                QtWidgets.QMessageBox.information(dlg, "No Selection", "Select a reference first.")
                return
            ref = item.data(QtCore.Qt.ItemDataRole.UserRole)
            self._apply_reference_to_plots(ref)
            dlg.accept()

        load_btn.clicked.connect(_load_selected_reference)
        cancel_btn.clicked.connect(dlg.reject)
        table.itemDoubleClicked.connect(lambda _item: _load_selected_reference())

        if hasattr(dlg, "exec"):
            dlg.exec()
        else:
            dlg.exec_()

    def _auto_load_latest_reference(self):
        try:
            references = self._load_references_from_csv()
        except (FileNotFoundError, RuntimeError):
            return
        except Exception as exc:
            self._set_status(f"Could not auto-load reference: {exc}")
            return

        if not references:
            return

        try:
            self._apply_reference_to_plots(references[0], status_prefix="Reference auto-loaded")
        except Exception as exc:
            self._set_status(f"Could not apply auto-loaded reference: {exc}")

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

        self._apply_reference_to_plots(ref, status_prefix="Reference set")

    def _clear_reference(self):
        """Remove reference lines from all plots."""
        for key in list(self.ref_lines):
            if self.ref_lines[key] is not None:
                self.plot_items[key].removeItem(self.ref_lines[key])
                self.ref_lines[key] = None
                self.ref_values[key] = None
        self._set_reference_info_text(None)
        self._set_status("Reference cleared")
        self._update_plot()

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def _set_readout(self, key, value=None, mean=None, std_ug=None, delta_mg=None):
        labels = self.readout_labels.get(key)
        if not labels:
            return
        labels["value"].setText("---" if value is None else f"{value:+.6f}")
        labels["mean"].setText("---" if mean is None else f"{mean:+.6f}")
        labels["std"].setText("---" if std_ug is None else f"{std_ug:.1f}")
        labels["delta"].setText("" if delta_mg is None else f"{delta_mg:+.3f}")

    def _position_overlay(self, key):
        # Kept for compatibility with existing signal wiring.
        return

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
                self._set_readout(key)
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
                self._set_readout(key)
                continue
            mean_v = sum(values) / len(values)
            std_ug = (sum((v - mean_v) ** 2 for v in values) / len(values)) ** 0.5 * 1e6
            val    = latest[key] if latest[key] is not None else mean_v
            ref_v = self.ref_values.get(key)
            delta_mg = None
            if ref_v is not None and math.isfinite(float(ref_v)):
                delta_mg = (val - float(ref_v)) * 1e3
            self._set_readout(key, val, mean_v, std_ug, delta_mg)

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

                if not self.manual_ylim[key]:
                    y_candidates = list(values)
                    ref_val = self.ref_values.get(key)
                    if ref_val is not None and math.isfinite(float(ref_val)):
                        y_candidates.append(float(ref_val))

                    if y_candidates:
                        ymin, ymax = min(y_candidates), max(y_candidates)
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
            self._set_readout(key)
        self.internal_ylim_update = False

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

    def _open_log_dialog(self):
        if self.log_dialog is not None:
            self.log_dialog.show()
            self.log_dialog.raise_()
            self.log_dialog.activateWindow()
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Event Log")
        dlg.setModal(False)
        dlg.resize(840, 420)

        layout = QtWidgets.QVBoxLayout(dlg)
        text = QtWidgets.QPlainTextEdit(dlg)
        text.setReadOnly(True)
        line_wrap_mode = getattr(QtWidgets.QPlainTextEdit, "LineWrapMode", QtWidgets.QPlainTextEdit)
        text.setLineWrapMode(getattr(line_wrap_mode, "NoWrap"))
        text.setStyleSheet(
            "QPlainTextEdit { background: #11161d; color: #d5dde8; border-radius: 6px; font-family: Consolas, 'Courier New'; }"
        )
        text.setPlainText("\n".join(self.log_lines))
        text.verticalScrollBar().setValue(text.verticalScrollBar().maximum())
        layout.addWidget(text)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        clear_btn = QtWidgets.QPushButton("Clear Log")
        close_btn = QtWidgets.QPushButton("Close")
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        def _clear_log():
            self.log_lines.clear()
            text.clear()

        clear_btn.clicked.connect(_clear_log)
        close_btn.clicked.connect(dlg.close)
        dlg.finished.connect(self._on_log_dialog_closed)

        self.log_dialog = dlg
        self.log_text = text
        dlg.show()

    def _on_log_dialog_closed(self, _result):
        self.log_dialog = None
        self.log_text = None

    def _set_status(self, text: str):
        stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
        self.log_lines.append(stamped)
        if self.log_text is not None:
            self.log_text.appendPlainText(stamped)
            sb = self.log_text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def shutdown(self):
        self.stop_event.set()
        self.running = False
        self._close_settings_window()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)