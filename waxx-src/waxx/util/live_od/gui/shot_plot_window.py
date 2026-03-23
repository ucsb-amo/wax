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
    QWidget,
    QDialog,
    QFormLayout,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QLabel,
    QFrame,
    QPushButton,
    QSizePolicy,
)
from PyQt6.QtGui import QFont, QStandardItem, QStandardItemModel
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
    "atom_number_apd_up":  "Atom number (APD up)",
    "atom_number_apd_down":"Atom number (APD down)",
    "atom_number_apd_total":"Atom number (APD total)",
}

CAMERA_DERIVED_QUANTITY_KEYS = {
    "integrated_od",
    "atom_number",
    "atom_number_fit_x",
    "atom_number_fit_y",
    "fit_sigma_x",
    "fit_sigma_y",
    "fit_center_x",
    "fit_center_y",
    "fit_amplitude_x",
    "fit_amplitude_y",
    "fit_offset_x",
    "fit_offset_y",
    "fit_area_x",
    "fit_area_y",
    "sum_od_peak_x",
    "sum_od_peak_y",
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

_COLOR_PRESETS = {
    "Blue": (50, 100, 220),
    "Red": (220, 80, 50),
    "Green": (60, 160, 90),
    "Amber": (225, 145, 40),
    "Purple": (155, 90, 210),
    "Teal": (55, 150, 165),
    "Slate": (95, 110, 135),
}


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

    def has_key(self, key: str) -> bool:
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == key:
                return True
        return False

    def remove_key(self, key: str):
        for row in range(self._model.rowCount() - 1, -1, -1):
            item = self._model.item(row)
            if item is None or item.data(Qt.ItemDataRole.UserRole) != key:
                continue
            if row in self._order:
                self._order.remove(row)
            self._model.removeRow(row)
            self._order = [idx - 1 if idx > row else idx for idx in self._order]
            self._update_display_text()
            self.selectionChanged.emit()
            return

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
            summary = " · ".join(labels)
            if len(summary) > 36:
                summary = summary[:33] + "..."
            self.lineEdit().setText(summary)

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

    def __init__(self,
                 window_id: int = 0,
                 xvar_names: list | None = None,
                 data_field_names: list | None = None,
                 camera_enabled: bool = True,
                 embedded: bool = False):
        super().__init__()
        self._id = window_id
        self._camera_enabled = camera_enabled
        self._embedded = embedded
        if not self._embedded:
            self.setWindowTitle(f"Plot #{window_id}")
            self.resize(580, 440)
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setStyleSheet(
            "QWidget { background: #0c131b; color: #e8f1fb; }"
            "QFrame#controlsPanel {"
            "  background: #111c28;"
            "  border: 1px solid #2e4256;"
            "  border-radius: 10px;"
            "}"
            "QComboBox, QSpinBox {"
            "  background: #162434; border: 1px solid #35516c; border-radius: 7px;"
            "  padding: 2px 6px; min-height: 12px;"
            "}"
            "QLabel#sectionLabel { color: #9fb8d1; font-size: 11px; font-weight: 700; }"
            "QCheckBox { color: #d8e7f8; }"
            "QPushButton {"
            "  background: #1d3550; border: 1px solid #4f7ca8; border-radius: 8px;"
            "  color: #edf6ff; padding: 5px 10px; font-weight: 700;"
            "}"
            "QPushButton:hover { background: #26476a; }"
            "QLabel { color: #d8e7f8; }"
        )

        # ---- accumulated data ----
        self._shot_data: list[dict] = []
        self._dynamic_qty_labels: dict[str, str] = {}
        self._data_field_names: list[str] = []

        # ---- autorange state ----
        self._autorange_left = True
        self._autorange_right = True
        self._series_colors = [tuple(_COLORS[0]), tuple(_COLORS[1])]

        # ---- widgets ----
        self._build_controls(xvar_names or [])
        self._build_plot()
        self._build_options_dialog()
        self.update_data_field_names(data_field_names or [])
        self.set_camera_enabled(camera_enabled)
        self._build_layout()

    # ------------------------------------------------------------------
    #  UI construction
    # ------------------------------------------------------------------

    def _build_controls(self, xvar_names: list[str]):
        # Derived quantity selector (checkable, max 2)
        self.qty_combo = _CheckableCombo(max_checked=2)
        self.qty_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.qty_combo.setMinimumWidth(100)
        for key, label in DERIVED_QUANTITIES.items():
            self.qty_combo.add_item(label, data=key)
        self.qty_combo.selectionChanged.connect(self._replot)

        # Independent variable selector
        self.indep_combo = QComboBox()
        self.indep_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
        self.plot_last_spin.setMaximumWidth(130)
        self.plot_last_spin.valueChanged.connect(self._replot)

        # Plot style controls (hosted in options dialog)
        self.connect_points_check = QCheckBox("Connect points")
        self.connect_points_check.setChecked(False)
        self.connect_points_check.stateChanged.connect(self._on_style_options_changed)

        self.symbol_size_spin = QSpinBox()
        self.symbol_size_spin.setRange(3, 20)
        self.symbol_size_spin.setValue(7)
        self.symbol_size_spin.setSuffix(" px")
        self.symbol_size_spin.valueChanged.connect(self._on_style_options_changed)

        self.show_grid_check = QCheckBox("Show grid")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.stateChanged.connect(self._on_style_options_changed)

        self.left_color_combo = QComboBox()
        self.right_color_combo = QComboBox()
        for name, rgb in _COLOR_PRESETS.items():
            self.left_color_combo.addItem(name, userData=rgb)
            self.right_color_combo.addItem(name, userData=rgb)
        self.left_color_combo.setCurrentText("Blue")
        self.right_color_combo.setCurrentText("Red")
        self.left_color_combo.currentIndexChanged.connect(self._on_style_options_changed)
        self.right_color_combo.currentIndexChanged.connect(self._on_style_options_changed)

        self.options_button = QPushButton("⚙")
        self.options_button.setToolTip("Plot options")
        self.options_button.setFixedWidth(32)
        self.options_button.clicked.connect(self._toggle_options_dialog)

        # Auto-range button
        self.autorange_button = QPushButton("⟳")
        self.autorange_button.setToolTip("Auto range")
        self.autorange_button.setFixedWidth(32)
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
        self._update_legend_styles()

    def _build_options_dialog(self):
        self._options_dialog = QDialog(self)
        self._options_dialog.setWindowTitle("Plot options")
        self._options_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self._options_dialog.setMinimumWidth(280)

        form = QFormLayout(self._options_dialog)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        form.addRow("Window", self.plot_last_check)
        form.addRow("Shots", self.plot_last_spin)
        form.addRow("Style", self.connect_points_check)
        form.addRow("Marker size", self.symbol_size_spin)
        form.addRow("Left color", self.left_color_combo)
        form.addRow("Right color", self.right_color_combo)
        form.addRow("Grid", self.show_grid_check)

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

        self._curve_right = pg.PlotDataItem(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush(*_COLORS[1], 200),
            symbolPen=pg.mkPen("k", width=0.5),
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
        self._apply_plot_style()

    def _sync_right_viewbox(self):
        self._vb_right.setGeometry(self.plot_widget.getViewBox().sceneBoundingRect())
        self._vb_right.linkedViewChanged(self.plot_widget.getViewBox(), self._vb_right.XAxis)

    def _build_layout(self):
        self._controls_panel = QFrame()
        self._controls_panel.setObjectName("controlsPanel")

        self._controls_grid = QGridLayout()
        self._controls_grid.setContentsMargins(3, 3, 3, 3)
        self._controls_grid.setHorizontalSpacing(6)
        self._controls_grid.setVerticalSpacing(4)

        self._y_label = QLabel("Y")
        self._y_label.setObjectName("sectionLabel")
        self._x_label = QLabel("X")
        self._x_label.setObjectName("sectionLabel")

        self._legend_widget = QWidget()
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(0, 0, 0, 0)
        legend_row.setSpacing(14)
        legend_row.addWidget(self._legend_label_1)
        legend_row.addWidget(self._legend_label_2)
        legend_row.addStretch()
        self._legend_widget.setLayout(legend_row)

        self._controls_panel.setLayout(self._controls_grid)

        # Single persistent layout row; compact mode intentionally removed.
        self._controls_grid.addWidget(self._y_label, 0, 0)
        self._controls_grid.addWidget(self.qty_combo, 0, 1)
        self._controls_grid.addWidget(self._x_label, 0, 2)
        self._controls_grid.addWidget(self.indep_combo, 0, 3)
        self._controls_grid.addWidget(self.options_button, 0, 4)
        self._controls_grid.addWidget(self.autorange_button, 0, 5)
        self._controls_grid.addWidget(self._legend_widget, 1, 0, 1, 6)
        self._controls_grid.setColumnStretch(1, 1)
        self._controls_grid.setColumnStretch(3, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(3,3,3,3)
        layout.setSpacing(4)
        layout.addWidget(self._controls_panel)
        layout.addWidget(self.plot_widget, stretch=1)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    #  Slot: receive new shot data
    # ------------------------------------------------------------------

    @pyqtSlot(dict)
    def on_new_shot(self, shot: dict):
        """Append one shot result dict and refresh the plot."""
        self._register_dynamic_quantities(shot)
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

    def set_camera_enabled(self, enabled: bool):
        self._camera_enabled = bool(enabled)
        if self._camera_enabled:
            for key, label in DERIVED_QUANTITIES.items():
                if key not in CAMERA_DERIVED_QUANTITY_KEYS:
                    continue
                if not self.qty_combo.has_key(key):
                    self.qty_combo.add_item(label, data=key)
        else:
            for key in CAMERA_DERIVED_QUANTITY_KEYS:
                self.qty_combo.remove_key(key)

    def update_xvar_names(self, names: list[str]):
        """Replace xvar entries in the independent combo."""
        to_remove = []
        for i in range(self.indep_combo.count()):
            if str(self.indep_combo.itemData(i)).startswith("xvar:"):
                to_remove.append(i)
        for i in reversed(to_remove):
            self.indep_combo.removeItem(i)
        for name in names:
            label_prefix = "data" if str(name) in self._data_field_names else "xvar"
            self.indep_combo.addItem(f"{label_prefix}: {name}", userData=f"xvar:{name}")

    def update_data_field_names(self, names: list[str]):
        """Register data fields as selectable y-quantities while running."""
        normalized = [str(name) for name in names]
        self._data_field_names = normalized

        # If data-field metadata arrives after selector items were created,
        # relabel those existing entries from xvar:* to data:*.
        for i in range(self.indep_combo.count()):
            item_data = self.indep_combo.itemData(i)
            if not isinstance(item_data, str) or not item_data.startswith("xvar:"):
                continue
            name = item_data.split(":", 1)[1]
            if name in self._data_field_names:
                self.indep_combo.setItemText(i, f"data: {name}")

        # Remove stale data-field quantities so selector options match the run.
        current_data_keys = {
            f"xvar.{name}" for name in self._data_field_names
        }
        to_remove = []
        for row in range(self.qty_combo.model().rowCount()):
            item = self.qty_combo.model().item(row)
            if item is None:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(key, str) and key.startswith("xvar.") and key not in current_data_keys:
                to_remove.append(key)
        for key in to_remove:
            self.qty_combo.remove_key(key)
            self._dynamic_qty_labels.pop(key, None)

        for name in self._data_field_names:
            key = f"xvar.{name}"
            label = f"data: {name}"
            self._dynamic_qty_labels[key] = label
            if self.qty_combo.has_key(key):
                for row in range(self.qty_combo.model().rowCount()):
                    item = self.qty_combo.model().item(row)
                    if item is not None and item.data(Qt.ItemDataRole.UserRole) == key:
                        item.setText(label)
                        break
            else:
                self.qty_combo.add_item(label, data=key)

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _on_indep_changed(self):
        key = self.indep_combo.currentData()
        is_sequential = key in ("shot_index", "timestamp")
        self.plot_last_check.setEnabled(is_sequential)
        self.plot_last_spin.setEnabled(is_sequential)
        self.plot_last_check.setToolTip(
            "Available for shot-index or timestamp x-axis"
            if not is_sequential
            else "Only plot the last N points"
        )

    def _toggle_options_dialog(self):
        if self._options_dialog.isVisible():
            self._options_dialog.hide()
            return
        anchor = self.options_button.mapToGlobal(self.options_button.rect().bottomLeft())
        self._options_dialog.move(anchor)
        self._options_dialog.show()
        self._options_dialog.raise_()
        self._options_dialog.activateWindow()

    def _on_style_options_changed(self):
        left_rgb = self.left_color_combo.currentData()
        right_rgb = self.right_color_combo.currentData()
        if isinstance(left_rgb, tuple) and len(left_rgb) == 3:
            self._series_colors[0] = left_rgb
        if isinstance(right_rgb, tuple) and len(right_rgb) == 3:
            self._series_colors[1] = right_rgb
        self._apply_plot_style()
        self._replot()

    def _apply_plot_style(self):
        marker_size = self.symbol_size_spin.value()
        left_color = self._series_colors[0]
        right_color = self._series_colors[1]
        connect_lines = self.connect_points_check.isChecked()

        self._curve_left.setSymbolSize(marker_size)
        self._curve_right.setSymbolSize(marker_size)
        self._curve_left.setSymbolBrush(pg.mkBrush(*left_color, 200))
        self._curve_right.setSymbolBrush(pg.mkBrush(*right_color, 200))
        self._curve_left.setPen(pg.mkPen(*left_color, width=2) if connect_lines else None)
        self._curve_right.setPen(pg.mkPen(*right_color, width=2) if connect_lines else None)
        self.plot_widget.showGrid(x=self.show_grid_check.isChecked(), y=self.show_grid_check.isChecked(), alpha=0.3)
        self._update_legend_styles()

    def _update_legend_styles(self):
        c1 = self._series_colors[0]
        c2 = self._series_colors[1]
        self._legend_label_1.setStyleSheet(
            f"color: rgb{c1}; font-weight: bold; border: none;"
        )
        self._legend_label_2.setStyleSheet(
            f"color: rgb{c2}; font-weight: bold; border: none;"
        )

    def _register_dynamic_quantities(self, shot: dict):
        for key, value in shot.items():
            if key in DERIVED_QUANTITIES or key in ("shot_index", "timestamp", "xvars", "data_fields"):
                continue
            if isinstance(value, (dict, list, tuple)):
                continue
            try:
                arr = np.asarray(value, dtype=float)
            except Exception:
                continue
            if arr.ndim > 1:
                continue
            if self.qty_combo.has_key(key):
                continue
            if key.startswith("xvar.") and key.split(".", 1)[1] in self._data_field_names:
                label = f"data: {key.split('.', 1)[1]}"
            else:
                label = key.replace("_", " ")
            self._dynamic_qty_labels[key] = label
            self.qty_combo.add_item(label, data=key)

    def _quantity_label(self, key: str) -> str:
        return DERIVED_QUANTITIES.get(key, self._dynamic_qty_labels.get(key, key))

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
        label1 = self._quantity_label(key1)
        y1 = np.array([d.get(key1, np.nan) for d in data], dtype=float)
        mask1 = np.isfinite(x_vals) & np.isfinite(y1)
        self._curve_left.setData(x_vals[mask1], y1[mask1])
        self.plot_widget.setLabel("left", label1, color=pg.mkColor(*self._series_colors[0]))
        self.plot_widget.getAxis("left").setPen(pg.mkPen(*self._series_colors[0]))
        self._legend_label_1.setText(f"● {label1}")

        if self._autorange_left:
            self.plot_widget.enableAutoRange(axis=pg.ViewBox.XYAxes)

        # ---- quantity 2 → right axis (if selected) --------------------
        if len(qty_keys) >= 2:
            key2 = qty_keys[1]
            label2 = self._quantity_label(key2)
            y2 = np.array([d.get(key2, np.nan) for d in data], dtype=float)
            mask2 = np.isfinite(x_vals) & np.isfinite(y2)
            self._curve_right.setData(x=x_vals[mask2], y=y2[mask2])
            self.plot_widget.setLabel("right", label2, color=pg.mkColor(*self._series_colors[1]))
            self.plot_widget.getAxis("right").setPen(pg.mkPen(*self._series_colors[1]))
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
        if not self._embedded:
            self.setWindowTitle(f"Plot #{self._id}  —  {title_qty}  vs  {x_label}")

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if not self._embedded:
            self.closed.emit(self)
        super().closeEvent(event)
