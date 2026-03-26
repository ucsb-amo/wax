import os
import sys
import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSplitter,
    QFrame,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QDoubleSpinBox,
    QToolButton,
    QMenu,
    QSizePolicy,
)
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QTimer, pyqtSignal


class SuppressPrints:
    def __init__(self, suppress=True):
        self.suppress = suppress
        self._original_stdout = None

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress and self._original_stdout:
            sys.stdout.close()
            sys.stdout = self._original_stdout


class LiveODViewer(QWidget):
    frame_changed = pyqtSignal(int)
    recompute_derived_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.Nimg = 0

        self._sumodx_scale = 1.0
        self._sumody_scale = 1.0

        self._first_image_received = 0
        self._first_image_minmax = {}
        self._autoscale_ready = False
        self._autoscale_buffer = []

        self._cmap_name = "viridis"
        self._od_min = 0.0
        self._od_max = 2.5
        self._od_slider_min = 0.0
        self._od_slider_max = 6.0

        self._syncing_image_views = False
        self._syncing_all_views = False

        # Raw image views are always synchronized with each other.
        self._lock_raw_group = True
        # Optional lock between raw group and OD view.
        self._lock_fov_between_raw_and_od = True

        self._last_od = None
        self._last_sumodx = None
        self._last_sumody = None
        self._last_od_shape = None

        self._current_shot_count = 0
        self._total_shot_count = 0

        self._frame_history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._frame_index = -1
        self._auto_follow_frames = True

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(
            "QWidget { background: #0d131b; color: #dbe7f2; }"
            "QToolButton, QPushButton {"
            "  background: #1c2733; border: 1px solid #2f3f50;"
            "  border-radius: 6px; padding: 3px 7px;"
            "}"
            "QToolButton:hover, QPushButton:hover { background: #243445; }"
            "QPlainTextEdit {"
            "  background: #0d1218; border: 1px solid #2a3745;"
            "  border-radius: 6px;"
            "}"
        )

        self.reset_zoom_button = QPushButton("Reset zoom")
        self.clear_button = QPushButton("Clear")
        self.log_button = QPushButton("Log")
        self.new_plot_button = QPushButton("New Plot")
        self.counts_label = QLabel("Shots 0/0   |   Images 0/0")
        self.prev_image_button = QPushButton("<")
        self.next_image_button = QPushButton(">")
        self.auto_follow_button = QToolButton()
        self.auto_follow_button.setCheckable(True)
        self.auto_follow_button.setChecked(True)
        self.auto_follow_button.setText("Follow")
        self.auto_follow_button.setToolTip(
            "Auto-follow newest frame\n"
            "Shortcuts: Left = previous, Right = next, F = toggle follow"
        )
        self.frame_index_label = QLabel("Frame 0/0")

        for btn in [
            self.reset_zoom_button,
            self.clear_button,
            self.log_button,
            self.new_plot_button,
            self.prev_image_button,
            self.next_image_button,
        ]:
            btn.setFixedHeight(34)

        self.prev_image_button.setFixedWidth(40)
        self.next_image_button.setFixedWidth(40)
        self.frame_index_label.setStyleSheet(
            "QLabel {"
            "  background: #0f1c2b;"
            "  border: 1px solid #30485f;"
            "  border-radius: 8px;"
            "  padding: 3px 8px;"
            "  color: #dbe9f8;"
            "  font-weight: 600;"
            "  font-size: 11px;"
            "}"
        )
        self.frame_index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_index_label.setToolTip(
            "Frame navigation\n"
            "Left Arrow: previous frame\n"
            "Right Arrow: next frame\n"
            "F: toggle auto-follow"
        )

        self.log_button.setStyleSheet(
            "QPushButton { background: #2b3f55; color: #e9f4ff; border: 1px solid #49617a; border-radius: 8px; font-weight: 600; }"
            "QPushButton:hover { background: #35516e; }"
        )
        self.clear_button.setStyleSheet(
            "QPushButton { background: #4a3a20; color: #fff2db; border: 1px solid #7a6136; border-radius: 8px; font-weight: 600; }"
            "QPushButton:hover { background: #5b4828; }"
        )
        self.reset_zoom_button.setStyleSheet(
            "QPushButton { background: #294433; color: #e9ffef; border: 1px solid #49715d; border-radius: 8px; font-weight: 600; }"
            "QPushButton:hover { background: #335a44; }"
        )
        self.new_plot_button.setStyleSheet(
            "QPushButton { background: #203754; color: #eaf5ff; border: 1px solid #4d78a8; border-radius: 8px; font-weight: 700; }"
            "QPushButton:hover { background: #29466a; }"
        )

        self.counts_label.setStyleSheet(
            "QLabel {"
            "  background: #0f1c2b;"
            "  border: 1px solid #30485f;"
            "  border-radius: 8px;"
            "  padding: 3px 8px;"
            "  color: #dbe9f8;"
            "  font-weight: 600;"
            "  font-size: 11px;"
            "}"
        )
        self.counts_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.output_window = QPlainTextEdit()
        self.output_window.setReadOnly(True)
        self.output_window.setMinimumSize(720, 340)

        self.log_dialog = QDialog(self)
        self.log_dialog.setWindowTitle("LiveOD Log")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.addWidget(self.output_window)
        self.log_dialog.setLayout(log_layout)
        self.log_button.clicked.connect(self._show_log_dialog)

        self.lock_fov_button = QToolButton()
        self.lock_fov_button.setCheckable(True)
        self.lock_fov_button.setChecked(True)
        self.lock_fov_button.clicked.connect(self._on_lock_fov_toggled)
        self._refresh_lock_button_label()

        self.display_menu_button = QToolButton()
        self.display_menu_button.setText("Display")
        self.display_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.display_menu_button.setMenu(self._build_display_menu())
        self.display_menu_button.setFixedHeight(34)
        self.lock_fov_button.setFixedHeight(34)
        self.lock_fov_button.hide()

        controls_bar = QFrame()
        controls_bar.setStyleSheet(
            "QFrame {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #111d2a, stop:1 #152436);"
            "  border: 1px solid #2d4156;"
            "  border-radius: 8px;"
            "}"
        )

        top_controls = QHBoxLayout()
        self._top_controls = top_controls
        top_controls.setContentsMargins(10, 6, 10, 6)
        top_controls.setSpacing(8)

        self._external_status_container = QWidget()
        self._external_status_layout = QHBoxLayout()
        self._external_status_layout.setContentsMargins(0, 0, 0, 0)
        self._external_status_layout.setSpacing(8)
        self._external_status_container.setLayout(self._external_status_layout)

        self._external_trailing_container = QWidget()
        self._external_trailing_layout = QHBoxLayout()
        self._external_trailing_layout.setContentsMargins(0, 0, 0, 0)
        self._external_trailing_layout.setSpacing(8)
        self._external_trailing_container.setLayout(self._external_trailing_layout)

        top_controls.addWidget(self._external_status_container)
        top_controls.addWidget(self.display_menu_button)
        top_controls.addWidget(self.log_button)
        top_controls.addWidget(self.new_plot_button)
        top_controls.addWidget(self.prev_image_button)
        top_controls.addWidget(self.next_image_button)
        top_controls.addWidget(self.auto_follow_button)
        top_controls.addWidget(self.frame_index_label)
        top_controls.addStretch()
        top_controls.addWidget(self.counts_label)
        top_controls.addWidget(self._external_trailing_container)
        controls_bar.setLayout(top_controls)
        self._controls_bar = controls_bar

        self.img_atoms_view = pg.ImageView()
        self.img_light_view = pg.ImageView()
        self.img_dark_view = pg.ImageView()

        for v in [self.img_atoms_view, self.img_light_view, self.img_dark_view]:
            v.ui.histogram.hide()
            v.ui.roiBtn.hide()
            v.ui.menuBtn.hide()
            self.set_pg_colormap(v, self._cmap_name)
            v.getView().setAspectLocked(False)

        self.img_atoms_view.getView().sigRangeChanged.connect(
            lambda *args: self._on_raw_range_changed(self.img_atoms_view)
        )
        self.img_light_view.getView().sigRangeChanged.connect(
            lambda *args: self._on_raw_range_changed(self.img_light_view)
        )
        self.img_dark_view.getView().sigRangeChanged.connect(
            lambda *args: self._on_raw_range_changed(self.img_dark_view)
        )

        raw_row = QHBoxLayout()
        raw_row.setContentsMargins(0, 0, 0, 0)
        raw_row.setSpacing(0)
        raw_row.addWidget(self._with_label(self.img_atoms_view, "Atoms + Light"), stretch=1)
        raw_row.addWidget(self._with_label(self.img_light_view, "Light only"), stretch=1)
        raw_row.addWidget(self._with_label(self.img_dark_view, "Dark"), stretch=1)

        raw_container = QWidget()
        raw_container.setLayout(raw_row)
        raw_container.setMinimumHeight(170)
        raw_container.setMaximumHeight(280)

        self.od_plot = pg.PlotWidget()
        self.od_img_item = pg.ImageItem()
        self.od_plot.addItem(self.od_img_item)
        self.od_img_item.setZValue(-10)
        self.set_pg_colormap(self.od_img_item, self._cmap_name)
        self.od_plot.hideAxis("right")
        self.od_plot.hideAxis("top")
        self.od_plot.getAxis("left").setStyle(showValues=False)
        self.od_plot.getAxis("bottom").setStyle(showValues=False)
        self.od_plot.showGrid(x=False, y=False)
        self.od_plot.setMenuEnabled(True)
        self.od_plot.setMouseEnabled(x=True, y=True)
        self.od_plot.setXRange(0, 512, padding=0)
        self.od_plot.setYRange(0, 512, padding=0)

        self.sumody_panel = pg.PlotWidget()
        self.sumody_panel.setMenuEnabled(False)
        self.sumody_panel.setMouseEnabled(x=True, y=True)
        self.sumody_panel.hideAxis("top")
        self.sumody_panel.getAxis("bottom").setStyle(showValues=False)
        self.sumody_panel.showGrid(x=False, y=False)

        self.sumodx_panel = pg.PlotWidget()
        self.sumodx_panel.setMenuEnabled(False)
        self.sumodx_panel.setMouseEnabled(x=True, y=True)
        self.sumodx_panel.hideAxis("right")
        self.sumodx_panel.getAxis("left").setStyle(showValues=False)
        self.sumodx_panel.showGrid(x=False, y=False)

        # Keep projections aligned with OD field of view.
        self.sumody_panel.setYLink(self.od_plot)
        self.sumodx_panel.setXLink(self.od_plot)

        self.od_plot.getViewBox().sigRangeChanged.connect(self.sync_sumod_panels)
        self.od_plot.getViewBox().sigRangeChanged.connect(
            lambda *args: self._on_od_range_changed_by_user()
        )

        od_grid = QGridLayout()
        od_grid.setContentsMargins(0, 0, 0, 0)
        od_grid.setHorizontalSpacing(0)
        od_grid.setVerticalSpacing(0)

        od_grid.addWidget(QWidget(), 0, 0)
        od_grid.addWidget(self.sumody_panel, 0, 1)
        od_grid.addWidget(self.od_plot, 0, 2)
        od_grid.addWidget(QWidget(), 1, 1)
        od_grid.addWidget(self.sumodx_panel, 1, 2)

        od_grid.setColumnStretch(0, 0)
        od_grid.setColumnStretch(1, 1)
        od_grid.setColumnStretch(2, 5)
        od_grid.setRowStretch(0, 6)
        od_grid.setRowStretch(1, 2)

        od_container = QWidget()
        od_container.setLayout(od_grid)

        main_vsplit = QSplitter(Qt.Orientation.Vertical)
        main_vsplit.addWidget(raw_container)
        main_vsplit.addWidget(od_container)
        main_vsplit.setSizes([220, 740])
        main_vsplit.setChildrenCollapsible(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(controls_bar)
        layout.addWidget(main_vsplit, stretch=1)
        self.setLayout(layout)

        self.clear_button.clicked.connect(self.clear_plots)
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        self.prev_image_button.clicked.connect(self.show_prev_frame)
        self.next_image_button.clicked.connect(self.show_next_frame)
        self.auto_follow_button.toggled.connect(self._on_auto_follow_toggled)

        self._shortcut_prev = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._shortcut_prev.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_prev.activated.connect(self.show_prev_frame)

        self._shortcut_next = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._shortcut_next.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_next.activated.connect(self.show_next_frame)

        self._shortcut_follow = QShortcut(QKeySequence("F"), self)
        self._shortcut_follow.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_follow.activated.connect(
            lambda: self.auto_follow_button.setChecked(not self.auto_follow_button.isChecked())
        )
        self._update_frame_controls()
        QTimer.singleShot(0, self._apply_minimum_width_from_controls)

    def _apply_minimum_width_from_controls(self):
        self._controls_bar.layout().activate()
        controls_width = self._controls_bar.sizeHint().width()
        margins = self.layout().contentsMargins()
        frame_width = controls_width + margins.left() + margins.right()
        self.setMinimumWidth(frame_width)
        self.resize(frame_width, self.height())

    def set_external_status_widgets(self, widgets):
        """Place caller-owned widgets in the left side of the top control bar."""
        while self._external_status_layout.count():
            item = self._external_status_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        for w in widgets:
            self._external_status_layout.addWidget(w)

    def set_external_trailing_widgets(self, widgets):
        """Place caller-owned widgets at the far right of the top control bar."""
        while self._external_trailing_layout.count():
            item = self._external_trailing_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        for w in widgets:
            self._external_trailing_layout.addWidget(w)

    def _build_display_menu(self):
        menu = QMenu(self)

        cmap_menu = menu.addMenu("Colormap")
        for cmap_name in ["viridis", "plasma", "magma", "inferno", "cividis", "gray"]:
            action = cmap_menu.addAction(cmap_name)
            action.triggered.connect(
                lambda checked=False, name=cmap_name: self.set_all_colormaps(name)
            )

        menu.addSeparator()
        limits_action = menu.addAction("Set OD min/max...")
        limits_action.triggered.connect(self._show_od_limits_dialog)

        menu.addSeparator()
        clear_action = menu.addAction("Clear")
        clear_action.triggered.connect(self.clear_plots)

        reset_zoom_action = menu.addAction("Reset zoom")
        reset_zoom_action.triggered.connect(self.reset_zoom)

        self._toggle_fov_action = menu.addAction("")
        self._toggle_fov_action.triggered.connect(self._toggle_fov_from_menu)

        menu.addSeparator()
        recompute_action = menu.addAction("Recompute derived from ROI/view")
        recompute_action.triggered.connect(self.recompute_derived_requested.emit)

        menu.addSeparator()
        reset_action = menu.addAction("Reset colormap to viridis")
        reset_action.triggered.connect(lambda: self.set_all_colormaps("viridis"))

        self._refresh_lock_button_label()
        return menu

    def _toggle_fov_from_menu(self):
        self._on_lock_fov_toggled(not self._lock_fov_between_raw_and_od)

    def _show_log_dialog(self):
        self.log_dialog.show()
        self.log_dialog.raise_()
        self.log_dialog.activateWindow()

    def _refresh_lock_button_label(self):
        if self._lock_fov_between_raw_and_od:
            self.lock_fov_button.setText("Unlock FOV")
            self.lock_fov_button.setToolTip("OD and raw images pan/zoom together")
            if hasattr(self, "_toggle_fov_action"):
                self._toggle_fov_action.setText("Unlock FOV")
        else:
            self.lock_fov_button.setText("Lock FOV")
            self.lock_fov_button.setToolTip("OD and raw images pan/zoom independently")
            if hasattr(self, "_toggle_fov_action"):
                self._toggle_fov_action.setText("Lock FOV")

    def _on_lock_fov_toggled(self, checked):
        self._lock_fov_between_raw_and_od = bool(checked)
        self._refresh_lock_button_label()
        if self._lock_fov_between_raw_and_od:
            ref_range = self.img_atoms_view.getView().viewRange()
            self._sync_all_views(ref_range, exclude="atoms")

    def _on_raw_range_changed(self, source_view):
        if self._syncing_image_views:
            return

        self._syncing_image_views = True
        src_range = source_view.getView().viewRange()

        for v in [self.img_atoms_view, self.img_light_view, self.img_dark_view]:
            if v is not source_view:
                v.getView().setRange(xRange=src_range[0], yRange=src_range[1], padding=0)

        self._syncing_image_views = False

        if self._lock_fov_between_raw_and_od and not self._syncing_all_views:
            self._sync_all_views(src_range, exclude=None)

    def _on_od_range_changed_by_user(self):
        if self._lock_fov_between_raw_and_od and not self._syncing_all_views:
            ref_range = self.od_plot.getViewBox().viewRange()
            self._sync_all_views(ref_range, exclude="od")

    def _sync_all_views(self, ref_range, exclude=None):
        self._syncing_all_views = True

        if exclude != "atoms":
            self.img_atoms_view.getView().setRange(
                xRange=ref_range[0], yRange=ref_range[1], padding=0
            )
        if exclude != "light":
            self.img_light_view.getView().setRange(
                xRange=ref_range[0], yRange=ref_range[1], padding=0
            )
        if exclude != "dark":
            self.img_dark_view.getView().setRange(
                xRange=ref_range[0], yRange=ref_range[1], padding=0
            )
        if exclude != "od":
            self.od_plot.getViewBox().setRange(
                xRange=ref_range[0], yRange=ref_range[1], padding=0
            )

        self._syncing_all_views = False

    def _with_label(self, imgview, label):
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        title = QLabel(label)
        title.setStyleSheet("font-weight: 600; color: #9eb2c5;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(imgview)
        container.setLayout(layout)
        return container

    def _show_od_limits_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("OD range")

        form = QFormLayout()
        min_spin = QDoubleSpinBox()
        min_spin.setDecimals(2)
        min_spin.setSingleStep(0.1)
        min_spin.setRange(self._od_slider_min, self._od_slider_max)
        min_spin.setValue(self._od_min)

        max_spin = QDoubleSpinBox()
        max_spin.setDecimals(2)
        max_spin.setSingleStep(0.1)
        max_spin.setRange(self._od_slider_min, self._od_slider_max)
        max_spin.setValue(self._od_max)

        form.addRow("Min OD", min_spin)
        form.addRow("Max OD", max_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        vmin = min_spin.value()
        vmax = max_spin.value()
        if vmax < vmin:
            vmax = vmin
        self._od_min = vmin
        self._od_max = vmax

        if self._last_od is not None:
            self.plot_od(self._last_od, self._last_sumodx, self._last_sumody)

    def set_pg_colormap(self, imgitem, cmap_name):
        import matplotlib

        lut = (
            matplotlib.colormaps[cmap_name](np.linspace(0, 1, 256))[:, :3] * 255
        ).astype(np.uint8)
        if hasattr(imgitem, "imageItem"):
            imgitem.imageItem.setLookupTable(lut)
            imgitem.imageItem.lut = lut
        else:
            imgitem.setLookupTable(lut)
            imgitem.lut = lut

    def set_all_colormaps(self, cmap_name):
        self._cmap_name = cmap_name
        for v in [self.img_atoms_view, self.img_light_view, self.img_dark_view, self.od_img_item]:
            self.set_pg_colormap(v, cmap_name)

    def clear_plots(self):
        self.img_atoms_view.clear()
        self.img_light_view.clear()
        self.img_dark_view.clear()
        self.od_img_item.clear()
        self.sumodx_panel.clear()
        self.sumody_panel.clear()

        self._last_sumodx = None
        self._last_sumody = None
        self._last_od_shape = None
        self._last_od = None

        self._sumodx_scale = 1.0
        self._sumody_scale = 1.0
        self._first_image_received = 0
        self._first_image_minmax = {}
        self._autoscale_ready = False
        self._autoscale_buffer = []
        self._frame_history = []
        self._frame_index = -1
        self._update_frame_controls()

        if hasattr(self, "_last_view_ranges"):
            del self._last_view_ranges

    def update_image_count(self, count, total):
        self._current_image_count = count
        self._total_image_count = total
        self._refresh_counts_label()

    def update_shot_count(self, current, total):
        self._current_shot_count = current
        self._total_shot_count = total
        self._refresh_counts_label()

    def _refresh_counts_label(self):
        current_img = getattr(self, "_current_image_count", 0)
        total_img = getattr(self, "_total_image_count", 0)
        self.counts_label.setText(
            f"Shots {self._current_shot_count}/{self._total_shot_count}\n"
            f"Images {current_img}/{total_img}"
        )

    def get_img_number(self, N_img, N_shots, N_pwa_per_shot, run_id=None):
        self.Nimg = N_img
        if run_id is not None:
            self._current_run_id = run_id

    def plot_images(self, atoms, light, dark):
        is_first = not hasattr(self, "_last_view_ranges") or self._first_image_received == 0

        atoms_range = self.img_atoms_view.getView().viewRange()
        light_range = self.img_light_view.getView().viewRange()
        dark_range = self.img_dark_view.getView().viewRange()
        self._last_view_ranges = {
            "atoms": atoms_range,
            "light": light_range,
            "dark": dark_range,
        }

        if not self._autoscale_ready:
            self._autoscale_buffer.append((atoms, light, dark))
            if len(self._autoscale_buffer) >= 1:
                a, l, d = self._autoscale_buffer[0]
                self._first_image_minmax["atoms_light"] = (
                    float(np.min(a)),
                    float(np.max(a)),
                )
                self._first_image_minmax["light"] = (float(np.min(l)), float(np.max(l)))
                self._first_image_minmax["dark"] = (float(np.min(d)), float(np.max(d)))
                self._autoscale_ready = True

        atoms_min, atoms_max = self._first_image_minmax.get("atoms_light", (None, None))
        dark_min, dark_max = self._first_image_minmax.get("dark", (None, None))

        if self._autoscale_ready:
            self.img_atoms_view.setImage(
                atoms.T,
                autoRange=False,
                autoLevels=False,
                levels=(atoms_min, atoms_max),
            )
            self.img_light_view.setImage(
                light.T,
                autoRange=False,
                autoLevels=False,
                levels=(atoms_min, atoms_max),
            )
            self.img_dark_view.setImage(
                dark.T,
                autoRange=False,
                autoLevels=False,
                levels=(dark_min, dark_max),
            )
        else:
            self.img_atoms_view.setImage(atoms.T, autoRange=False, autoLevels=True)
            self.img_light_view.setImage(light.T, autoRange=False, autoLevels=True)
            self.img_dark_view.setImage(dark.T, autoRange=False, autoLevels=True)

        if is_first:
            shape = atoms.T.shape
            self.img_atoms_view.getView().setRange(xRange=[0, shape[0]], yRange=[0, shape[1]], padding=0)
            self.img_light_view.getView().setRange(xRange=[0, shape[0]], yRange=[0, shape[1]], padding=0)
            self.img_dark_view.getView().setRange(xRange=[0, shape[0]], yRange=[0, shape[1]], padding=0)
            self._first_image_received = 1
        else:
            self.img_atoms_view.getView().setRange(
                xRange=self._last_view_ranges["atoms"][0],
                yRange=self._last_view_ranges["atoms"][1],
                padding=0,
            )
            self.img_light_view.getView().setRange(
                xRange=self._last_view_ranges["light"][0],
                yRange=self._last_view_ranges["light"][1],
                padding=0,
            )
            self.img_dark_view.getView().setRange(
                xRange=self._last_view_ranges["dark"][0],
                yRange=self._last_view_ranges["dark"][1],
                padding=0,
            )

    def sync_sumod_panels(self):
        od_vb = self.od_plot.getViewBox()
        x_range, y_range = od_vb.viewRange()

        self.od_plot.setYRange(*y_range, padding=0)
        self.od_plot.setXRange(*x_range, padding=0)

        if self._last_od is None:
            return

        od = self._last_od
        x0 = max(int(np.floor(x_range[0])), 0)
        x1 = min(int(np.ceil(x_range[1])), od.shape[1])
        y0 = max(int(np.floor(y_range[0])), 0)
        y1 = min(int(np.ceil(y_range[1])), od.shape[0])

        cropped = od[y0:y1, x0:x1]
        if cropped.size == 0:
            self.sumodx_panel.clear()
            self.sumody_panel.clear()
            return

        sumodx = np.sum(cropped, axis=0)
        sumody = np.sum(cropped, axis=1)

        x_axis = np.arange(x0, x1)
        y_axis = np.arange(y0, y1)

        self._plot_sumodx(sumodx, x_axis)
        self._plot_sumody(sumody, y_axis)

    def _plot_sumodx(self, sumodx, x_axis):
        self.sumodx_panel.clear()
        if sumodx is not None and len(sumodx) > 0:
            self.sumodx_panel.plot(x_axis, sumodx, pen=pg.mkPen("#59c8ff", width=2))

    def _plot_sumody(self, sumody, y_axis):
        self.sumody_panel.clear()
        if sumody is not None and len(sumody) > 0:
            self.sumody_panel.plot(sumody, y_axis, pen=pg.mkPen("#ffbe55", width=2))

    def reset_zoom(self):
        self.od_plot.setXRange(0, 512, padding=0)
        self.od_plot.setYRange(0, 512, padding=0)

        for v in [self.img_atoms_view, self.img_light_view, self.img_dark_view]:
            v.getView().setRange(xRange=[0, 512], yRange=[0, 512], padding=0)

        self.sync_sumod_panels()

    def plot_od(self, od, sumodx, sumody, min_od=None, max_od=None):
        if getattr(self, "_last_od_shape", None) is None and od is not None:
            self.od_plot.setXRange(0, od.shape[1], padding=0)
            self.od_plot.setYRange(0, od.shape[0], padding=0)

        if min_od is None or max_od is None:
            min_od = self._od_min
            max_od = self._od_max

        self.od_img_item.setImage(od.T, autoLevels=False, levels=(min_od, max_od))

        self._last_od = od
        self._last_sumodx = sumodx
        self._last_sumody = sumody
        self._last_od_shape = od.shape

        self.sync_sumod_panels()

    def handle_plot_data(self, to_plot):
        img_atoms, img_light, img_dark, od, sum_od_x, sum_od_y = to_plot
        frame = (
            np.asarray(img_atoms).copy(),
            np.asarray(img_light).copy(),
            np.asarray(img_dark).copy(),
            np.asarray(od).copy(),
            np.asarray(sum_od_x).copy(),
            np.asarray(sum_od_y).copy(),
        )
        self._frame_history.append(frame)
        if self._auto_follow_frames or self._frame_index < 0:
            self._frame_index = len(self._frame_history) - 1
            self._render_frame(self._frame_index)
        self._update_frame_controls()

    def _render_frame(self, idx: int):
        if idx < 0 or idx >= len(self._frame_history):
            return
        img_atoms, img_light, img_dark, od, sum_od_x, sum_od_y = self._frame_history[idx]
        self.plot_images(img_atoms, img_light, img_dark)
        self.plot_od(od, sum_od_x, sum_od_y)

    def show_prev_frame(self):
        if self._frame_index <= 0:
            return
        self._frame_index -= 1
        self._auto_follow_frames = False
        self.auto_follow_button.blockSignals(True)
        self.auto_follow_button.setChecked(False)
        self.auto_follow_button.blockSignals(False)
        self._render_frame(self._frame_index)
        self._update_frame_controls()

    def show_next_frame(self):
        if self._frame_index >= len(self._frame_history) - 1:
            return
        self._frame_index += 1
        if self._frame_index < len(self._frame_history) - 1:
            self._auto_follow_frames = False
            self.auto_follow_button.blockSignals(True)
            self.auto_follow_button.setChecked(False)
            self.auto_follow_button.blockSignals(False)
        self._render_frame(self._frame_index)
        self._update_frame_controls()

    def _on_auto_follow_toggled(self, checked: bool):
        self._auto_follow_frames = bool(checked)
        if self._auto_follow_frames and self._frame_history:
            self._frame_index = len(self._frame_history) - 1
            self._render_frame(self._frame_index)
            self._update_frame_controls()

    def _update_frame_controls(self):
        n_frames = len(self._frame_history)
        has_frames = n_frames > 0
        self.prev_image_button.setEnabled(has_frames and self._frame_index > 0)
        self.next_image_button.setEnabled(has_frames and self._frame_index < n_frames - 1)
        if has_frames:
            self.frame_index_label.setText(f"Frame {self._frame_index + 1}/{n_frames}")
        else:
            self.frame_index_label.setText("Frame 0/0")
        self.frame_changed.emit(self._frame_index)

    def get_od_view_range(self):
        try:
            od_vb = self.od_plot.getViewBox()
            x_range, y_range = od_vb.viewRange()
            return x_range, y_range
        except Exception:
            return [0, 512], [0, 512]
