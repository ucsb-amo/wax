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
    QLineEdit,
    QSlider,
    QTreeView,
)
from PyQt6.QtGui import QFont, QStandardItem, QStandardItemModel, QIntValidator, QDoubleValidator
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer

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

_QUANTITY_GROUP_LABELS = {
    "image": "Image-derived",
    "data": "Data containers",
}

_GROUP_ROLE = int(Qt.ItemDataRole.UserRole) + 1


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
        self._order: list[str] = []
        self._group_items: dict[str, QStandardItem] = {}
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        tree = QTreeView(self)
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(True)
        tree.setItemsExpandable(True)
        tree.setUniformRowHeights(True)
        self.setView(tree)
        self._model.itemChanged.connect(self._on_item_changed)
        self.view().pressed.connect(self._handle_press)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self._update_display_text()

    # ---- public API -------------------------------------------------

    def add_item(self, text: str, data=None, group: str | None = None):
        parent = self._ensure_group_item(group) if group is not None else self._model.invisibleRootItem()
        item = QStandardItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        item.setData(data, Qt.ItemDataRole.UserRole)
        item.setData(group, _GROUP_ROLE)
        parent.appendRow(item)

    def set_item_text(self, key: str, text: str, group: str | None = None):
        item = self._item_for_key(key)
        if item is None:
            return
        item.setText(text)

    def checked_keys(self) -> list[str]:
        """Return data (keys) of currently checked items, in check order."""
        return list(self._order)

    def checked_labels(self) -> list[str]:
        out = []
        for key in self._order:
            item = self._item_for_key(key)
            if item is not None:
                out.append(item.text())
        return out

    def has_key(self, key: str) -> bool:
        return self._item_for_key(key) is not None

    def remove_key(self, key: str):
        item = self._item_for_key(key)
        if item is None:
            return
        if key in self._order:
            self._order.remove(key)
        parent = item.parent() or self._model.invisibleRootItem()
        parent.removeRow(item.row())
        self._remove_empty_group_items()
        self._update_display_text()
        self.selectionChanged.emit()

    def _ensure_group_item(self, group: str):
        if group in self._group_items:
            return self._group_items[group]
        label = _QUANTITY_GROUP_LABELS.get(group, group.title())
        item = QStandardItem(label)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setData(group, _GROUP_ROLE)
        self._model.appendRow(item)
        self._group_items[group] = item
        return item

    def _remove_empty_group_items(self):
        for group, item in list(self._group_items.items()):
            if item.rowCount() == 0:
                self._model.removeRow(item.row())
                del self._group_items[group]

    def _item_for_key(self, key: str):
        for group_item in self._group_items.values():
            for row in range(group_item.rowCount()):
                child = group_item.child(row)
                if child is not None and child.data(Qt.ItemDataRole.UserRole) == key:
                    return child
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == key:
                return item
        return None

    # ---- internals --------------------------------------------------

    def _handle_press(self, index):
        """Toggle check on click while keeping dropdown open."""
        item = self._model.itemFromIndex(index)
        if item is None or item.hasChildren():
            return
        new_state = (
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        item.setCheckState(new_state)

    def _on_item_changed(self, item: QStandardItem):
        if item.hasChildren():
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            if key not in self._order:
                self._order.append(key)
            # enforce max
            while len(self._order) > self._max:
                oldest = self._order.pop(0)
                old_item = self._item_for_key(oldest)
                if old_item is not None:
                    self._model.itemChanged.disconnect(self._on_item_changed)
                    old_item.setCheckState(Qt.CheckState.Unchecked)
                    self._model.itemChanged.connect(self._on_item_changed)
        else:
            if key in self._order:
                self._order.remove(key)

        self._update_display_text()
        self.selectionChanged.emit()

    def _update_display_text(self):
        labels = self.checked_labels()
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        if not labels:
            self.setCurrentIndex(-1)
            self.setEditText("(none)")
        else:
            summary = " · ".join(labels)
            if len(summary) > 36:
                summary = summary[:33] + "..."
            self.lineEdit().setText(summary)

    def hidePopup(self):
        # Allow normal hide
        super().hidePopup()

    def showPopup(self):
        if isinstance(self.view(), QTreeView):
            self.view().expandAll()
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
        self._multi_value_count = 1
        self._index_controls_updating = False

        # ---- autorange state ----
        self._autorange_left = True
        self._autorange_right = True
        self._pending_autorange = False
        self._series_colors = [tuple(_COLORS[0]), tuple(_COLORS[1])]
        self._left_curves: list[pg.PlotDataItem] = []
        self._right_curves: list[pg.PlotDataItem] = []
        self._array_legend_lines: list[str] = []
        self._override_controls_updating = False
        self._replot_queued = False
        self._queued_autorange = False

        # ---- widgets ----
        self._build_controls(xvar_names or [])
        self._build_plot()
        self._build_options_dialog()
        self.update_data_field_names(data_field_names or [])
        self.set_camera_enabled(camera_enabled)
        self._build_layout()
        self._sync_axis_override_control_states()

    # ------------------------------------------------------------------
    #  UI construction
    # ------------------------------------------------------------------

    def _build_controls(self, xvar_names: list[str]):
        # Derived quantity selector (checkable, max 2)
        self.qty_combo = _CheckableCombo(max_checked=2)
        self.qty_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.qty_combo.setMinimumWidth(100)
        for key, label in DERIVED_QUANTITIES.items():
            self.qty_combo.add_item(label, data=key, group="image")
        self.qty_combo.selectionChanged.connect(self._on_quantity_selection_changed)

        # Independent variable selector
        self.indep_combo = QComboBox()
        self.indep_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.indep_combo.addItem("Shot index", userData="shot_index")
        self.indep_combo.addItem("Timestamp (s)", userData="timestamp")
        for name in xvar_names:
            self.indep_combo.addItem(f"xvar: {name}", userData=f"xvar:{name}")
        self.indep_combo.currentIndexChanged.connect(self._on_indep_changed)
        self.indep_combo.currentIndexChanged.connect(self._request_replot_no_autorange)

        # "Plot last N" controls (only for shot_index / timestamp)
        self.plot_last_check = QCheckBox("Plot last")
        self.plot_last_check.setChecked(False)
        self.plot_last_check.stateChanged.connect(self._request_replot_no_autorange)

        self.plot_last_spin = QSpinBox()
        self.plot_last_spin.setRange(1, 100000)
        self.plot_last_spin.setValue(50)
        self.plot_last_spin.setSuffix(" shots")
        self.plot_last_spin.setMaximumWidth(130)
        self.plot_last_spin.valueChanged.connect(self._request_replot_no_autorange)

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

        self.values_per_shot_label = QLabel("1")
        self.values_per_shot_label.setStyleSheet("color: #e8f1fb; font-weight: 700;")

        self.index_lower_slider = QSlider(Qt.Orientation.Horizontal)
        self.index_upper_slider = QSlider(Qt.Orientation.Horizontal)
        for slider in (self.index_lower_slider, self.index_upper_slider):
            slider.setRange(0, 0)
            slider.setSingleStep(1)
            slider.setPageStep(1)
            slider.valueChanged.connect(self._on_index_slider_changed)

        validator = QIntValidator(0, 0, self)
        self.index_lower_edit = QLineEdit("0")
        self.index_upper_edit = QLineEdit("0")
        for edit in (self.index_lower_edit, self.index_upper_edit):
            edit.setValidator(validator)
            edit.setFixedWidth(40)
            edit.editingFinished.connect(self._on_index_edit_changed)

        self.left_color_combo = QComboBox()
        self.right_color_combo = QComboBox()
        for name, rgb in _COLOR_PRESETS.items():
            self.left_color_combo.addItem(name, userData=rgb)
            self.right_color_combo.addItem(name, userData=rgb)
        self.left_color_combo.setCurrentText("Blue")
        self.right_color_combo.setCurrentText("Red")
        self.left_color_combo.currentIndexChanged.connect(self._on_style_options_changed)
        self.right_color_combo.currentIndexChanged.connect(self._on_style_options_changed)

        float_validator = QDoubleValidator(self)
        self.xlim_enable_check = QCheckBox("Enable")
        self.left_ylim_enable_check = QCheckBox("Enable")
        self.right_ylim_enable_check = QCheckBox("Enable")
        self.xlim_enable_check.stateChanged.connect(self._on_axis_override_changed)
        self.left_ylim_enable_check.stateChanged.connect(self._on_axis_override_changed)
        self.right_ylim_enable_check.stateChanged.connect(self._on_axis_override_changed)
        self.xlim_min_edit = QLineEdit()
        self.xlim_max_edit = QLineEdit()
        self.left_ylim_min_edit = QLineEdit()
        self.left_ylim_max_edit = QLineEdit()
        self.right_ylim_min_edit = QLineEdit()
        self.right_ylim_max_edit = QLineEdit()
        for edit in (
            self.xlim_min_edit,
            self.xlim_max_edit,
            self.left_ylim_min_edit,
            self.left_ylim_max_edit,
            self.right_ylim_min_edit,
            self.right_ylim_max_edit,
        ):
            edit.setValidator(float_validator)
            edit.setPlaceholderText("auto")
            edit.setFixedWidth(72)
            edit.editingFinished.connect(self._on_axis_override_changed)

        self.options_autoscale_button = QPushButton("Autoscale")
        self.options_autoscale_button.clicked.connect(self._autoscale_now_from_dialog)

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
        self._options_dialog.setWindowTitle(f"Plot #{self._id} options")
        self._options_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self._options_dialog.setMinimumWidth(280)

        form = QFormLayout(self._options_dialog)
        form.setContentsMargins(8, 8, 8, 8)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(6)
        form.addRow("Window", self.plot_last_check)
        form.addRow("Shots", self.plot_last_spin)
        form.addRow("Style", self.connect_points_check)
        form.addRow("Marker size", self.symbol_size_spin)
        form.addRow("Left color", self.left_color_combo)
        form.addRow("Right color", self.right_color_combo)
        form.addRow("Grid", self.show_grid_check)
        form.addRow("Values / shot", self.values_per_shot_label)

        index_range_widget = QWidget()
        index_range_layout = QVBoxLayout()
        index_range_layout.setContentsMargins(0, 0, 0, 0)
        index_range_layout.setSpacing(6)

        lower_row = QHBoxLayout()
        lower_row.setContentsMargins(0, 0, 0, 0)
        lower_row.setSpacing(6)
        lower_row.addWidget(QLabel("Start"))
        lower_row.addWidget(self.index_lower_slider, stretch=1)
        lower_row.addWidget(self.index_lower_edit)

        upper_row = QHBoxLayout()
        upper_row.setContentsMargins(0, 0, 0, 0)
        upper_row.setSpacing(6)
        upper_row.addWidget(QLabel("End"))
        upper_row.addWidget(self.index_upper_slider, stretch=1)
        upper_row.addWidget(self.index_upper_edit)

        index_range_layout.addLayout(lower_row)
        index_range_layout.addLayout(upper_row)
        index_range_widget.setLayout(index_range_layout)
        form.addRow("Indices", index_range_widget)

        xlim_widget = QWidget()
        xlim_row = QHBoxLayout()
        xlim_row.setContentsMargins(0, 0, 0, 0)
        xlim_row.setSpacing(6)
        xlim_row.addWidget(self.xlim_enable_check)
        xlim_row.addWidget(self.xlim_min_edit)
        xlim_row.addWidget(QLabel("to"))
        xlim_row.addWidget(self.xlim_max_edit)
        xlim_widget.setLayout(xlim_row)
        form.addRow("X limits", xlim_widget)

        left_ylim_widget = QWidget()
        left_ylim_row = QHBoxLayout()
        left_ylim_row.setContentsMargins(0, 0, 0, 0)
        left_ylim_row.setSpacing(6)
        left_ylim_row.addWidget(self.left_ylim_enable_check)
        left_ylim_row.addWidget(self.left_ylim_min_edit)
        left_ylim_row.addWidget(QLabel("to"))
        left_ylim_row.addWidget(self.left_ylim_max_edit)
        left_ylim_widget.setLayout(left_ylim_row)
        form.addRow("Left Y", left_ylim_widget)

        right_ylim_widget = QWidget()
        right_ylim_row = QHBoxLayout()
        right_ylim_row.setContentsMargins(0, 0, 0, 0)
        right_ylim_row.setSpacing(6)
        right_ylim_row.addWidget(self.right_ylim_enable_check)
        right_ylim_row.addWidget(self.right_ylim_min_edit)
        right_ylim_row.addWidget(QLabel("to"))
        right_ylim_row.addWidget(self.right_ylim_max_edit)
        right_ylim_widget.setLayout(right_ylim_row)
        form.addRow("Right Y", right_ylim_widget)
        form.addRow("Axes", self.options_autoscale_button)

    def _build_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._array_legend_item = pg.TextItem(anchor=(0, 1))
        self._array_legend_item.setZValue(1000)
        self.plot_widget.addItem(self._array_legend_item)
        self._array_legend_item.hide()

        # --- right-axis view ---
        self._vb_right = pg.ViewBox()
        self.plot_widget.scene().addItem(self._vb_right)
        self.plot_widget.getAxis("right").linkToView(self._vb_right)
        self._vb_right.setXLink(self.plot_widget)
        self.plot_widget.getAxis("right").show()

        # keep right ViewBox geometry in sync
        self.plot_widget.getViewBox().sigResized.connect(self._sync_right_viewbox)

        # Detect manual user interaction → disable autorange
        self.plot_widget.getViewBox().sigRangeChangedManually.connect(
            self._on_left_range_manual
        )
        self.plot_widget.getViewBox().sigRangeChanged.connect(self._position_array_legend)
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
        self._controls_grid.setContentsMargins(2, 2, 2, 2)
        self._controls_grid.setHorizontalSpacing(4)
        self._controls_grid.setVerticalSpacing(2)

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

        self._controls_grid.addWidget(self._y_label, 0, 0)
        self._controls_grid.addWidget(self.qty_combo, 0, 1, 1, 3)
        self._controls_grid.addWidget(self.options_button, 0, 4)
        self._controls_grid.addWidget(self.autorange_button, 0, 5)
        self._controls_grid.addWidget(self._x_label, 1, 0)
        self._controls_grid.addWidget(self.indep_combo, 1, 1, 1, 5)
        self._controls_grid.addWidget(self._legend_widget, 2, 0, 1, 6)
        self._controls_grid.setColumnStretch(1, 1)
        self._controls_grid.setColumnStretch(3, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)
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
        self._update_value_index_controls()
        self._request_replot(
            trigger_autorange=self._autorange_left or self._autorange_right or self._pending_autorange
        )

    @pyqtSlot()
    def on_run_started(self):
        """Clear accumulated data when a new run begins."""
        self._shot_data.clear()
        self._clear_curve_pool(self._left_curves)
        self._clear_curve_pool(self._right_curves)
        self._autorange_left = True
        self._autorange_right = True
        self._pending_autorange = False
        self._multi_value_count = 1
        self._update_value_index_controls(reset_range=True)
        self._update_array_series_legend([])
        self._sync_axis_override_control_states()

    def set_camera_enabled(self, enabled: bool):
        self._camera_enabled = bool(enabled)
        if self._camera_enabled:
            for key, label in DERIVED_QUANTITIES.items():
                if key not in CAMERA_DERIVED_QUANTITY_KEYS:
                    continue
                if not self.qty_combo.has_key(key):
                    self.qty_combo.add_item(label, data=key, group="image")
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
        to_remove = [
            key for key in list(self._dynamic_qty_labels)
            if key.startswith("xvar.") and key not in current_data_keys
        ]
        for key in to_remove:
            self.qty_combo.remove_key(key)
            self._dynamic_qty_labels.pop(key, None)

        for name in self._data_field_names:
            key = f"xvar.{name}"
            label = f"data: {name}"
            self._dynamic_qty_labels[key] = label
            if self.qty_combo.has_key(key):
                self.qty_combo.set_item_text(key, label, group="data")
            else:
                self.qty_combo.add_item(label, data=key, group="data")

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
        self._update_value_index_controls()
        self._request_replot(trigger_autorange=False)

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
        self._request_replot(trigger_autorange=False)

    def _on_quantity_selection_changed(self):
        self._update_value_index_controls(reset_range=True)
        self._sync_axis_override_control_states()
        self._request_replot(trigger_autorange=False)

    def _on_index_slider_changed(self):
        if self._index_controls_updating:
            return
        lower = self.index_lower_slider.value()
        upper = self.index_upper_slider.value()
        if lower > upper:
            sender = self.sender()
            if sender is self.index_lower_slider:
                upper = lower
            else:
                lower = upper
        self._set_index_controls(lower, upper)
        self._request_replot(trigger_autorange=False)

    def _on_index_edit_changed(self):
        if self._index_controls_updating:
            return
        lower = self._parse_index_edit(self.index_lower_edit, self.index_lower_slider.value())
        upper = self._parse_index_edit(self.index_upper_edit, self.index_upper_slider.value())
        if lower > upper:
            sender = self.sender()
            if sender is self.index_lower_edit:
                upper = lower
            else:
                lower = upper
        self._set_index_controls(lower, upper)
        self._request_replot(trigger_autorange=False)

    def _on_axis_override_changed(self):
        if self._override_controls_updating:
            return
        sender = self.sender()
        if sender is self.xlim_enable_check and self.xlim_enable_check.isChecked():
            self._seed_override_from_current_range("x")
        elif sender is self.left_ylim_enable_check and self.left_ylim_enable_check.isChecked():
            self._seed_override_from_current_range("left")
        elif sender is self.right_ylim_enable_check and self.right_ylim_enable_check.isChecked():
            self._seed_override_from_current_range("right")
        self._apply_axis_overrides()

    def _apply_plot_style(self):
        self.plot_widget.showGrid(x=self.show_grid_check.isChecked(), y=self.show_grid_check.isChecked(), alpha=0.3)
        self._update_legend_styles()

    def _parse_float_edit(self, edit: QLineEdit):
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _seed_override_from_current_range(self, axis: str):
        x_range, y_range = self.plot_widget.getViewBox().viewRange()
        right_y_range = self._vb_right.viewRange()[1]
        self._override_controls_updating = True
        if axis == "x":
            if not self.xlim_min_edit.text().strip():
                self.xlim_min_edit.setText(f"{x_range[0]:.6g}")
            if not self.xlim_max_edit.text().strip():
                self.xlim_max_edit.setText(f"{x_range[1]:.6g}")
        elif axis == "left":
            if not self.left_ylim_min_edit.text().strip():
                self.left_ylim_min_edit.setText(f"{y_range[0]:.6g}")
            if not self.left_ylim_max_edit.text().strip():
                self.left_ylim_max_edit.setText(f"{y_range[1]:.6g}")
        else:
            if not self.right_ylim_min_edit.text().strip():
                self.right_ylim_min_edit.setText(f"{right_y_range[0]:.6g}")
            if not self.right_ylim_max_edit.text().strip():
                self.right_ylim_max_edit.setText(f"{right_y_range[1]:.6g}")
        self._override_controls_updating = False

    def _apply_axis_overrides(self):
        vb = self.plot_widget.getViewBox()
        if self.xlim_enable_check.isChecked():
            xmin = self._parse_float_edit(self.xlim_min_edit)
            xmax = self._parse_float_edit(self.xlim_max_edit)
            if xmin is not None and xmax is not None and xmin < xmax:
                vb.setXRange(xmin, xmax, padding=0)
                self._autorange_left = False
        if self.left_ylim_enable_check.isChecked():
            ymin = self._parse_float_edit(self.left_ylim_min_edit)
            ymax = self._parse_float_edit(self.left_ylim_max_edit)
            if ymin is not None and ymax is not None and ymin < ymax:
                vb.setYRange(ymin, ymax, padding=0)
                self._autorange_left = False
        if self.right_ylim_enable_check.isChecked() and len(self.qty_combo.checked_keys()) >= 2:
            ymin = self._parse_float_edit(self.right_ylim_min_edit)
            ymax = self._parse_float_edit(self.right_ylim_max_edit)
            if ymin is not None and ymax is not None and ymin < ymax:
                self._vb_right.setYRange(ymin, ymax, padding=0)
                self._autorange_right = False
        self.plot_widget.update()
        self._vb_right.update()

    def _capture_current_ranges_to_overrides(self):
        x_range, y_range = self.plot_widget.getViewBox().viewRange()
        right_y_range = self._vb_right.viewRange()[1]
        self._override_controls_updating = True
        if not self.xlim_enable_check.isChecked():
            self.xlim_min_edit.setText(f"{x_range[0]:.6g}")
            self.xlim_max_edit.setText(f"{x_range[1]:.6g}")
        if not self.left_ylim_enable_check.isChecked():
            self.left_ylim_min_edit.setText(f"{y_range[0]:.6g}")
            self.left_ylim_max_edit.setText(f"{y_range[1]:.6g}")
        if not self.right_ylim_enable_check.isChecked() and len(self.qty_combo.checked_keys()) >= 2:
            self.right_ylim_min_edit.setText(f"{right_y_range[0]:.6g}")
            self.right_ylim_max_edit.setText(f"{right_y_range[1]:.6g}")
        self._override_controls_updating = False

    def _autoscale_now_from_dialog(self):
        self._override_controls_updating = True
        self.xlim_enable_check.setChecked(False)
        self.left_ylim_enable_check.setChecked(False)
        self.right_ylim_enable_check.setChecked(False)
        self._override_controls_updating = False
        self._enable_autorange(immediate=True)

    def _sync_axis_override_control_states(self):
        has_right_axis = len(self.qty_combo.checked_keys()) >= 2
        for widget in (
            self.right_ylim_enable_check,
            self.right_ylim_min_edit,
            self.right_ylim_max_edit,
        ):
            widget.setEnabled(has_right_axis)
        if not has_right_axis:
            self._override_controls_updating = True
            self.right_ylim_enable_check.setChecked(False)
            self._override_controls_updating = False

    def _request_replot(self, trigger_autorange: bool = False):
        self._queued_autorange = self._queued_autorange or bool(trigger_autorange)
        if self._replot_queued:
            return
        self._replot_queued = True
        QTimer.singleShot(0, self._flush_replot)

    def _request_replot_no_autorange(self, *args):
        self._request_replot(trigger_autorange=False)

    def _flush_replot(self):
        trigger_autorange = self._queued_autorange
        self._queued_autorange = False
        self._replot_queued = False
        self._replot(trigger_autorange=trigger_autorange)

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
                group = "data"
            else:
                label = key.replace("_", " ")
                group = "image"
            self._dynamic_qty_labels[key] = label
            self.qty_combo.add_item(label, data=key, group=group)

    def _value_length(self, value) -> int:
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return 1
        return max(1, int(arr.size))

    def _scalar_value(self, value):
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return np.nan
        if arr.size == 0:
            return np.nan
        return float(arr[0] if arr.size == 1 else np.nanmean(arr))

    def _value_at_index(self, value, index: int):
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return np.nan
        if arr.size == 0 or index < 0 or index >= arr.size:
            return np.nan
        return float(arr[index])

    def _raw_quantity_value(self, shot: dict, key: str):
        if key.startswith("xvar."):
            field_name = key.split(".", 1)[1]
            if field_name in shot.get("xvars", {}):
                return shot["xvars"][field_name]
        return shot.get(key, np.nan)

    def _selected_index_bounds(self, count: int | None = None):
        max_count = max(1, self._multi_value_count if count is None else int(count))
        lower = min(self.index_lower_slider.value(), max_count - 1)
        upper = min(self.index_upper_slider.value(), max_count - 1)
        if lower > upper:
            lower = upper
        return lower, upper

    def _parse_index_edit(self, edit: QLineEdit, fallback: int) -> int:
        text = edit.text().strip()
        if not text:
            return fallback
        try:
            return int(text)
        except ValueError:
            return fallback

    def _set_index_controls(self, lower: int, upper: int):
        max_index = max(0, self._multi_value_count - 1)
        lower = max(0, min(int(lower), max_index))
        upper = max(0, min(int(upper), max_index))
        if lower > upper:
            lower = upper
        self._index_controls_updating = True
        self.index_lower_slider.setValue(lower)
        self.index_upper_slider.setValue(upper)
        self.index_lower_edit.setText(str(lower))
        self.index_upper_edit.setText(str(upper))
        self._index_controls_updating = False

    def _update_value_index_controls(self, reset_range: bool = False):
        prev_count = self._multi_value_count
        count = 1
        for key in self.qty_combo.checked_keys():
            for shot in reversed(self._shot_data):
                raw = self._raw_quantity_value(shot, key)
                count = max(count, self._value_length(raw))
                if count > 1:
                    break
        self._multi_value_count = max(1, count)
        max_index = self._multi_value_count - 1
        self.values_per_shot_label.setText(str(self._multi_value_count))
        self.index_lower_slider.setRange(0, max_index)
        self.index_upper_slider.setRange(0, max_index)
        validator = QIntValidator(0, max_index, self)
        self.index_lower_edit.setValidator(validator)
        self.index_upper_edit.setValidator(validator)
        enabled = self._multi_value_count > 1
        for widget in (
            self.index_lower_slider,
            self.index_upper_slider,
            self.index_lower_edit,
            self.index_upper_edit,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            lower = 0
            upper = 0
        elif reset_range or prev_count != self._multi_value_count or prev_count <= 1:
            lower = 0
            upper = max_index
        else:
            lower, upper = self._selected_index_bounds(self._multi_value_count)
        self._set_index_controls(lower, upper)

    def _clear_curve_pool(self, curves: list[pg.PlotDataItem]):
        for curve in curves:
            curve.setData([], [])
            curve.hide()

    def _ensure_curve_pool(self, curves: list[pg.PlotDataItem], count: int, right_axis: bool = False):
        while len(curves) < count:
            curve = pg.PlotDataItem(
                pen=None,
                symbol="o",
                symbolSize=self.symbol_size_spin.value(),
                symbolPen=pg.mkPen("k", width=0.5),
            )
            if right_axis:
                self._vb_right.addItem(curve)
            else:
                self.plot_widget.addItem(curve)
            curves.append(curve)
        for index, curve in enumerate(curves):
            if index < count:
                curve.show()
            else:
                curve.setData([], [])
                curve.hide()
        return curves[:count]

    def _series_color(self, base_color: tuple[int, int, int], order: int, total: int):
        if total <= 1:
            return base_color
        factors = np.linspace(0.7, 1.15, total)
        factor = float(factors[min(order, total - 1)])
        return tuple(int(np.clip(channel * factor, 0, 255)) for channel in base_color)

    def _style_curve(self, curve: pg.PlotDataItem, color: tuple[int, int, int]):
        connect_lines = self.connect_points_check.isChecked()
        curve.setSymbolSize(self.symbol_size_spin.value())
        curve.setSymbolBrush(pg.mkBrush(*color, 200))
        curve.setPen(pg.mkPen(*color, width=2) if connect_lines else None)

    def _extract_quantity_series(self, data: list[dict], key: str):
        raw_values = [self._raw_quantity_value(shot, key) for shot in data]
        count = max(self._value_length(value) for value in raw_values) if raw_values else 1
        if count <= 1:
            y = np.array([self._scalar_value(value) for value in raw_values], dtype=float)
            return [(None, y)]
        lower, upper = self._selected_index_bounds(count)
        series = []
        for index in range(lower, upper + 1):
            y = np.array([self._value_at_index(value, index) for value in raw_values], dtype=float)
            series.append((index, y))
        return series

    def _series_label(self, key: str, index):
        base = self._quantity_label(key)
        if index is None:
            return base
        return f"{base}[{index}]"

    def _legend_summary(self, key: str, series: list[tuple[int | None, np.ndarray]]):
        if not series:
            return ""
        if len(series) == 1:
            return f"● {self._series_label(key, series[0][0])}"
        first = series[0][0]
        last = series[-1][0]
        return f"● {self._quantity_label(key)}[{first}:{last}]"

    def _html_color(self, color: tuple[int, int, int]):
        return f"rgb({color[0]}, {color[1]}, {color[2]})"

    def _update_array_series_legend(self, lines: list[str]):
        self._array_legend_lines = list(lines)
        if not self._array_legend_lines:
            self._array_legend_item.setHtml("")
            self._array_legend_item.hide()
            return
        html = "<div style='background: rgba(255,255,255,0.75); padding: 2px 4px; font-size: 8pt;'>" + "<br>".join(self._array_legend_lines) + "</div>"
        self._array_legend_item.setHtml(html)
        self._array_legend_item.show()
        self._position_array_legend()

    def _position_array_legend(self, *args):
        if not self._array_legend_lines:
            return
        x_range, y_range = self.plot_widget.getViewBox().viewRange()
        self._array_legend_item.setPos(x_range[0], y_range[1])

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
            raw_values = [d.get("xvars", {}).get(xvar_name, np.nan) for d in data]
            count = max(self._value_length(value) for value in raw_values) if raw_values else 1
            if count > 1:
                index, _ = self._selected_index_bounds(count)
                x = np.array([self._value_at_index(value, index) for value in raw_values], dtype=float)
                prefix = "data" if xvar_name in self._data_field_names else "xvar"
                return x, f"{prefix}: {xvar_name}[{index}]"
            x = np.array([self._scalar_value(value) for value in raw_values], dtype=float)
            prefix = "data" if xvar_name in self._data_field_names else "xvar"
            return x, f"{prefix}: {xvar_name}"
        else:
            return np.arange(len(data), dtype=float), "index"

    def _on_left_range_manual(self):
        """User manually panned/zoomed the left axis."""
        self._autorange_left = False
        self._pending_autorange = False

    def _on_right_range_manual(self):
        """User manually panned/zoomed the right axis."""
        self._autorange_right = False
        self._pending_autorange = False

    def _enable_autorange(self, immediate: bool = True):
        """Re-enable autorange for both axes.

        When ``immediate`` is false, the next arriving data point triggers the
        autoscale instead of the current settings change doing it.
        """
        self._autorange_left = True
        self._autorange_right = True
        self._pending_autorange = not immediate
        if immediate:
            self._request_replot(trigger_autorange=True)

    def _replot(self, *args, trigger_autorange: bool = False):
        qty_keys = self.qty_combo.checked_keys()
        indep_key = self.indep_combo.currentData()
        do_autorange = bool(trigger_autorange or self._pending_autorange)

        if not self._shot_data or not qty_keys:
            self._clear_curve_pool(self._left_curves)
            self._clear_curve_pool(self._right_curves)
            self._legend_label_1.setText("")
            self._legend_label_2.setText("")
            self._update_array_series_legend([])
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
        left_series = self._extract_quantity_series(data, key1)
        left_curves = self._ensure_curve_pool(self._left_curves, len(left_series), right_axis=False)
        array_legend_lines = []
        for order, ((index, y_vals), curve) in enumerate(zip(left_series, left_curves)):
            color = self._series_color(self._series_colors[0], order, len(left_series))
            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            curve.setData(x_vals[mask], y_vals[mask])
            self._style_curve(curve, color)
            if index is not None:
                array_legend_lines.append(
                    f"<span style='color:{self._html_color(color)}'>● {self._series_label(key1, index)}</span>"
                )
        self.plot_widget.setLabel("left", self._quantity_label(key1), color=pg.mkColor(*self._series_colors[0]))
        self.plot_widget.getAxis("left").setPen(pg.mkPen(*self._series_colors[0]))
        self._legend_label_1.setText(self._legend_summary(key1, left_series))

        if self._autorange_left and do_autorange:
            x_auto = not self.xlim_enable_check.isChecked()
            y_auto = not self.left_ylim_enable_check.isChecked()
            if x_auto or y_auto:
                self.plot_widget.getViewBox().enableAutoRange(x=x_auto, y=y_auto)
                self.plot_widget.getViewBox().autoRange()

        # ---- quantity 2 → right axis (if selected) --------------------
        if len(qty_keys) >= 2:
            key2 = qty_keys[1]
            right_series = self._extract_quantity_series(data, key2)
            right_curves = self._ensure_curve_pool(self._right_curves, len(right_series), right_axis=True)
            right_finite = []
            for order, ((index, y_vals), curve) in enumerate(zip(right_series, right_curves)):
                color = self._series_color(self._series_colors[1], order, len(right_series))
                mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                right_finite.append(y_vals[mask])
                curve.setData(x=x_vals[mask], y=y_vals[mask])
                self._style_curve(curve, color)
                if index is not None:
                    array_legend_lines.append(
                        f"<span style='color:{self._html_color(color)}'>● {self._series_label(key2, index)}</span>"
                    )
            self.plot_widget.setLabel("right", self._quantity_label(key2), color=pg.mkColor(*self._series_colors[1]))
            self.plot_widget.getAxis("right").setPen(pg.mkPen(*self._series_colors[1]))
            self.plot_widget.getAxis("right").show()
            self._legend_label_2.setText(self._legend_summary(key2, right_series))
            # auto-range the right ViewBox only if not manually overridden
            if self._autorange_right and do_autorange and not self.right_ylim_enable_check.isChecked():
                finite = np.concatenate([vals for vals in right_finite if len(vals)]) if any(len(vals) for vals in right_finite) else np.array([])
                if finite.size:
                    pad = (finite.max() - finite.min()) * 0.05 or 1.0
                    self._vb_right.setYRange(finite.min() - pad, finite.max() + pad)
        else:
            self._clear_curve_pool(self._right_curves)
            self.plot_widget.setLabel("right", "")
            self.plot_widget.getAxis("right").setPen(pg.mkPen(200, 200, 200))
            self._legend_label_2.setText("")
            self.plot_widget.getAxis("right").hide()

        self.plot_widget.setLabel("bottom", x_label)
        self._update_array_series_legend(array_legend_lines)
        self._sync_axis_override_control_states()
        self._apply_axis_overrides()
        if do_autorange:
            self._pending_autorange = False
            self._capture_current_ranges_to_overrides()
        self.plot_widget.update()
        self._vb_right.update()

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
