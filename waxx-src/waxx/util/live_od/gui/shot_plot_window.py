"""
Pop-out plot window for derived quantities vs. independent variables.

Each ``ShotPlotWindow`` listens to a ``new_shot_data`` signal emitted by
the viewer window and updates its plot after every shot. Multiple windows
can coexist, each plotting different derived quantities.

Up to **two** derived quantities can be selected simultaneously; they are
drawn in distinct colours on separate left / right y-axes.
"""

import time
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QSpinBox, QCheckBox, QLabel, QGroupBox,
    QListWidget, QListWidgetItem, QPushButton, QMenu,
    QWidgetAction, QSizePolicy,
)
from PyQt6.QtGui import QFont, QColor, QAction, QStandardItem, QStandardItemModel
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

import pyqtgraph as pg


# ======================================================================
#  Available derived quantities
# ======================================================================

DERIVED_QUANTITIES = {
    # key: display_name
    "integrated_od":       "Integrated OD",
    "atom_number":         "Atom number (integrated OD)",
    "atom_number_fit_x":   "Atom number (fit area x)",
    "atom_number_fit_y":   "Atom number (fit area y)",
    "fit_sigma_x":         "σ_x  (Gaussian width x)",
    "fit_sigma_y":         "σ_y  (Gaussian width y)",
    "fit_center_x":        "Center x",
    "fit_center_y":        "Center y",
    "fit_amplitude_x":     "Amplitude x",
    "fit_amplitude_y":     "Amplitude y",
    "fit_offset_x":        "Offset x",
    "fit_offset_y":        "Offset y",
    "fit_area_x":          "Fit area x",
    "fit_area_y":          "Fit area y",
    "sum_od_peak_x":       "Sum OD peak x",
    "sum_od_peak_y":       "Sum OD peak y",
}

INDEPENDENT_VARIABLES = {
    "shot_index":   "Shot index",
    "timestamp":    "Timestamp (s)",
}

# Colours for the two y-axis curves
_COLORS = [
    (50, 100, 220),   # blue  – left axis
    (220, 80, 50),    # red   – right axis
]


# ======================================================================
#  Checkable combo box (max 2 selections)
# ======================================================================

class _CheckableCombo(QComboBox):
    """
    Drop-down whose items have check-boxes.  At most ``max_checked``
    items may be checked at once — checking a new one un-checks the
    oldest.  The closed combo displays a summary of the selections.

    Emits ``selectionChanged`` whenever the set of checked keys changes.
    """

    selectionChanged = pyqtSignal()  # fired when checked set changes

    def __init__(self, max_checked: int = 2, parent=None):
        super().__init__(parent)
        self._max = max_checked
        self._order: list[int] = []          # row indices in check order
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self._model.itemChanged.connect(self._on_item_changed)
        # keep popup open on click
        self.view().pressed.connect(self._handle_press)

    # ---- public API -------------------------------------------------

    def add_item(self, text: str, data=None):
        item = QStandardItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        item.setData(data, Qt.ItemDataRole.UserRole)
        self._model.appendRow(item)

    def checked_keys(self) -> list[str]:
        """Return data (keys) of currently checked items, in check order."""
        out = []
        for row in self._order:
            item = self._model.item(row)
            if item is not None:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def checked_labels(self) -> list[str]:
        out = []
        for row in self._order:
            item = self._model.item(row)
            if item is not None:
                out.append(item.text())
        return out

    # ---- internals --------------------------------------------------

    def _handle_press(self, index):
        """Toggle check on click while keeping dropdown open."""
        item = self._model.itemFromIndex(index)
        if item is None:
            return
        new_state = (
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        item.setCheckState(new_state)

    def _on_item_changed(self, item: QStandardItem):
        row = item.row()
        if item.checkState() == Qt.CheckState.Checked:
            if row not in self._order:
                self._order.append(row)
            # enforce max
            while len(self._order) > self._max:
                oldest = self._order.pop(0)
                old_item = self._model.item(oldest)
                if old_item is not None:
                    self._model.itemChanged.disconnect(self._on_item_changed)
                    old_item.setCheckState(Qt.CheckState.Unchecked)
                    self._model.itemChanged.connect(self._on_item_changed)
        else:
            if row in self._order:
                self._order.remove(row)

        self._update_display_text()
        self.selectionChanged.emit()

    def _update_display_text(self):
        labels = self.checked_labels()
        if not labels:
            self.setCurrentIndex(-1)
            self.setEditText("(none)")
        else:
            # Put summary in the line-edit area
            self.setEditable(True)
            self.lineEdit().setReadOnly(True)
            self.lineEdit().setText(" · ".join(labels))

    def hidePopup(self):
        # Allow normal hide
        super().hidePopup()

    def showPopup(self):
        super().showPopup()


# ======================================================================
#  ShotPlotWindow
# ======================================================================

class ShotPlotWindow(QWidget):
    """
    Pop-out window that plots up to two user-chosen derived quantities
    versus an independent variable, updated live after every shot.

    The two quantities are drawn on separate y-axes (left / right) in
    distinct colours.

    Parameters
    ----------
    window_id : int
        A running counter used for the window title.
    xvar_names : list[str]
        Names of the current scan xvars (added to the independent
        variable combo box).
    """

    closed = pyqtSignal(object)  # emitted on close so parent can de-register

    def __init__(self, window_id: int = 0, xvar_names: list | None = None):
        super().__init__()
        self._id = window_id
        self.setWindowTitle(f"Plot #{window_id}")
        self.resize(580, 440)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # ---- accumulated data ----
        self._shot_data: list[dict] = []

        # ---- autorange state ----
        self._autorange_left = True
        self._autorange_right = True

        # ---- widgets ----
        self._build_controls(xvar_names or [])
        self._build_plot()
        self._build_layout()

    # ------------------------------------------------------------------
    #  UI construction
    # ------------------------------------------------------------------

    def _build_controls(self, xvar_names: list[str]):
        # Derived quantity selector (checkable, max 2)
        self.qty_combo = _CheckableCombo(max_checked=2)
        for key, label in DERIVED_QUANTITIES.items():
            self.qty_combo.add_item(label, data=key)
        self.qty_combo.selectionChanged.connect(self._replot)

        # Independent variable selector
        self.indep_combo = QComboBox()
        self.indep_combo.addItem("Shot index", userData="shot_index")
        self.indep_combo.addItem("Timestamp (s)", userData="timestamp")
        for name in xvar_names:
            self.indep_combo.addItem(f"xvar: {name}", userData=f"xvar:{name}")
        self.indep_combo.currentIndexChanged.connect(self._on_indep_changed)
        self.indep_combo.currentIndexChanged.connect(self._replot)

        # "Plot last N" controls (only for shot_index / timestamp)
        self.plot_last_check = QCheckBox("Plot last")
        self.plot_last_check.setChecked(False)
        self.plot_last_check.stateChanged.connect(self._replot)

        self.plot_last_spin = QSpinBox()
        self.plot_last_spin.setRange(1, 100000)
        self.plot_last_spin.setValue(50)
        self.plot_last_spin.setSuffix(" shots")
        self.plot_last_spin.valueChanged.connect(self._replot)

        # Auto-range button
        self.autorange_button = QPushButton("⟳ Auto Range")
        self.autorange_button.setMaximumWidth(120)
        self.autorange_button.setStyleSheet(
            "font-size: 11px; font-weight: bold;"
        )
        self.autorange_button.clicked.connect(self._enable_autorange)

        # colour legend labels (updated in _replot)
        self._legend_label_1 = QLabel("")
        self._legend_label_1.setStyleSheet(
            f"color: rgb{_COLORS[0]}; font-weight: bold; border: none;"
        )
        self._legend_label_2 = QLabel("")
        self._legend_label_2.setStyleSheet(
            f"color: rgb{_COLORS[1]}; font-weight: bold; border: none;"
        )

    def _build_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # --- left-axis curve (quantity 1) ---
        self._curve_left = self.plot_widget.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush(*_COLORS[0], 200),
            symbolPen=pg.mkPen("k", width=0.5),
        )

        # --- right-axis view & curve (quantity 2) ---
        self._vb_right = pg.ViewBox()
        self.plot_widget.scene().addItem(self._vb_right)
        self.plot_widget.getAxis("right").linkToView(self._vb_right)
        self._vb_right.setXLink(self.plot_widget)
        self.plot_widget.getAxis("right").show()

        self._curve_right = pg.ScatterPlotItem(
            size=7,
            brush=pg.mkBrush(*_COLORS[1], 200),
            pen=pg.mkPen("k", width=0.5),
        )
        self._vb_right.addItem(self._curve_right)

        # keep right ViewBox geometry in sync
        self.plot_widget.getViewBox().sigResized.connect(self._sync_right_viewbox)

        # Detect manual user interaction → disable autorange
        self.plot_widget.getViewBox().sigRangeChangedManually.connect(
            self._on_left_range_manual
        )
        self._vb_right.sigRangeChangedManually.connect(
            self._on_right_range_manual
        )

    def _sync_right_viewbox(self):
        self._vb_right.setGeometry(self.plot_widget.getViewBox().sceneBoundingRect())
        self._vb_right.linkedViewChanged(self.plot_widget.getViewBox(), self._vb_right.XAxis)

    def _build_layout(self):
        form = QFormLayout()
        form.addRow("Quantities (≤ 2):", self.qty_combo)
        form.addRow("vs:", self.indep_combo)

        last_row = QHBoxLayout()
        last_row.addWidget(self.plot_last_check)
        last_row.addWidget(self.plot_last_spin)
        last_row.addStretch()
        form.addRow("", last_row)

        legend_row = QHBoxLayout()
        legend_row.addWidget(self._legend_label_1)
        legend_row.addWidget(self._legend_label_2)
        legend_row.addStretch()
        legend_row.addWidget(self.autorange_button)
        form.addRow("", legend_row)

        controls = QGroupBox("Settings")
        controls.setLayout(form)
        controls.setMaximumHeight(160)

        layout = QVBoxLayout()
        layout.addWidget(controls)
        layout.addWidget(self.plot_widget, stretch=1)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    #  Slot: receive new shot data
    # ------------------------------------------------------------------

    @pyqtSlot(dict)
    def on_new_shot(self, shot: dict):
        """Append one shot result dict and refresh the plot."""
        self._shot_data.append(shot)
        self._replot()

    @pyqtSlot()
    def on_run_started(self):
        """Clear accumulated data when a new run begins."""
        self._shot_data.clear()
        self._curve_left.setData([], [])
        self._curve_right.setData([], [])
        self._autorange_left = True
        self._autorange_right = True

    def update_xvar_names(self, names: list[str]):
        """Replace xvar entries in the independent combo."""
        to_remove = []
        for i in range(self.indep_combo.count()):
            if str(self.indep_combo.itemData(i)).startswith("xvar:"):
                to_remove.append(i)
        for i in reversed(to_remove):
            self.indep_combo.removeItem(i)
        for name in names:
            self.indep_combo.addItem(f"xvar: {name}", userData=f"xvar:{name}")

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _on_indep_changed(self):
        key = self.indep_combo.currentData()
        is_sequential = key in ("shot_index", "timestamp")
        self.plot_last_check.setVisible(is_sequential)
        self.plot_last_spin.setVisible(is_sequential)

    def _build_x(self, data: list[dict], indep_key: str):
        """Return (x_vals, x_label) for the given independent variable."""
        if indep_key == "shot_index":
            x = np.array([d.get("shot_index", i) for i, d in enumerate(data)],
                         dtype=float)
            return x, "Shot index"
        elif indep_key == "timestamp":
            ts = np.array([d.get("timestamp", 0.0) for d in data], dtype=float)
            if len(ts):
                ts = ts - ts[0]
            return ts, "Time (s)"
        elif indep_key.startswith("xvar:"):
            xvar_name = indep_key.split(":", 1)[1]
            x = np.array([d.get("xvars", {}).get(xvar_name, np.nan)
                          for d in data], dtype=float)
            return x, xvar_name
        else:
            return np.arange(len(data), dtype=float), "index"

    def _on_left_range_manual(self):
        """User manually panned/zoomed the left axis."""
        self._autorange_left = False

    def _on_right_range_manual(self):
        """User manually panned/zoomed the right axis."""
        self._autorange_right = False

    def _enable_autorange(self):
        """Re-enable autorange for both axes and replot."""
        self._autorange_left = True
        self._autorange_right = True
        self._replot()

    def _replot(self):
        qty_keys = self.qty_combo.checked_keys()
        indep_key = self.indep_combo.currentData()

        if not self._shot_data or not qty_keys:
            self._curve_left.setData([], [])
            self._curve_right.setData([], [])
            self._legend_label_1.setText("")
            self._legend_label_2.setText("")
            return

        data = self._shot_data

        # Apply "plot last N" filter
        is_sequential = indep_key in ("shot_index", "timestamp")
        if is_sequential and self.plot_last_check.isChecked():
            n = self.plot_last_spin.value()
            data = data[-n:]

        x_vals, x_label = self._build_x(data, indep_key)

        # ---- quantity 1 → left axis -----------------------------------
        key1 = qty_keys[0]
        label1 = DERIVED_QUANTITIES.get(key1, key1)
        y1 = np.array([d.get(key1, np.nan) for d in data], dtype=float)
        mask1 = np.isfinite(x_vals) & np.isfinite(y1)
        self._curve_left.setData(x_vals[mask1], y1[mask1])
        self.plot_widget.setLabel("left", label1, color=pg.mkColor(*_COLORS[0]))
        self.plot_widget.getAxis("left").setPen(pg.mkPen(*_COLORS[0]))
        self._legend_label_1.setText(f"● {label1}")

        if self._autorange_left:
            self.plot_widget.enableAutoRange(axis=pg.ViewBox.XYAxes)

        # ---- quantity 2 → right axis (if selected) --------------------
        if len(qty_keys) >= 2:
            key2 = qty_keys[1]
            label2 = DERIVED_QUANTITIES.get(key2, key2)
            y2 = np.array([d.get(key2, np.nan) for d in data], dtype=float)
            mask2 = np.isfinite(x_vals) & np.isfinite(y2)
            self._curve_right.setData(x=x_vals[mask2], y=y2[mask2])
            self.plot_widget.setLabel("right", label2, color=pg.mkColor(*_COLORS[1]))
            self.plot_widget.getAxis("right").setPen(pg.mkPen(*_COLORS[1]))
            self.plot_widget.getAxis("right").show()
            self._legend_label_2.setText(f"● {label2}")
            # auto-range the right ViewBox only if not manually overridden
            if self._autorange_right and np.any(mask2):
                y2f = y2[mask2]
                pad = (y2f.max() - y2f.min()) * 0.05 or 1.0
                self._vb_right.setYRange(y2f.min() - pad, y2f.max() + pad)
        else:
            self._curve_right.setData([], [])
            self.plot_widget.setLabel("right", "")
            self.plot_widget.getAxis("right").setPen(pg.mkPen(200, 200, 200))
            self._legend_label_2.setText("")

        self.plot_widget.setLabel("bottom", x_label)

        # ---- window title ---------------------------------------------
        labels = self.qty_combo.checked_labels()
        title_qty = " & ".join(labels) if labels else "–"
        self.setWindowTitle(f"Plot #{self._id}  —  {title_qty}  vs  {x_label}")

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.closed.emit(self)
        super().closeEvent(event)
