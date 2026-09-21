import os
import sys
import json
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
                             QSizePolicy, QDoubleSpinBox, QSpinBox, QCheckBox, QFrame,
                             QToolButton, QMenu, QWidgetAction, QDialog, QDialogButtonBox,
                             QFormLayout)
from PyQt6.QtGui import QAction, QKeySequence, QPainterPath, QShortcut
from PyQt6.QtCore import Qt, QEvent, QRectF, pyqtSignal

from waxx.util.live_od.gui import theme
from waxx.util.live_od.gui.log_panel import LogPanel
from waxx.util.live_od.log import get_logger
from waxx.util.live_od.marker_store import clean_markers
from waxx.util.live_od.gui.markers import MarkerItem, MarkerDialog, MarkerPanel, random_marker_color
from waxa.units import mult_for, unit_for_param

logger = get_logger("viewer")

MARKER_SIZE_VIEW_FRACTION = 1 / 25  # a new marker's size, as a fraction of the view width
ROI_EDGE_GRAB_PX = 5            # screen pixels either side of the ROI's outline that grab it

PROFILE_FRACTION = 0.2          # height of the sumOD overlays, as a fraction of the view
DEFAULT_OD_LEVELS = (0.0, 2.5)
DEFAULT_OD_LIMITS = (-2.0, 10.0)    # changeable in the settings dialog
OD_STEP = 0.1
# offered in the images' right-click menu; one colormap for the raw frames and the OD
COLORMAPS = ("viridis", "magma", "inferno", "plasma", "cividis", "turbo", "gray", "RdBu_r")
HISTORY_MAX_SHOTS = 50
HISTORY_MAX_BYTES = 256e6       # a 2 MP Basler shot is ~30 MB of arrays
SATURATION_LEVELS = (255, 1023, 4095, 16383, 65535)
SATURATION_MIN_PIXELS = 5
AUTO_ROI_DEFAULT_SHOTS = 10


class SuppressPrints:
    def __init__(self, suppress=True):
        self.suppress = suppress
        self._original_stdout = None
    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress and self._original_stdout:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def _vline():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class _OverlayLabel(QLabel):
    """Text floating over the OD plot: bold, light on a translucent dark plate.
    Invisible while empty, and never in the way of the mouse."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 170); color: #ffffff; "
                           "font-weight: bold; font-size: 10pt; "
                           "padding: 3px 8px; border-radius: 4px; }")
        self.hide()

    def fix_width_for(self, widest_html):
        """Size the plate once, for the widest thing it will ever say, so it does
        not jump about as the numbers (or the units) change."""
        self.setText(widest_html)
        self.setFixedWidth(self.sizeHint().width())
        self.setText("")

    def show_text(self, html_text):
        self.setText(html_text)
        self.setVisible(bool(html_text))
        self.adjustSize()


def _fit_to_text(button, padding=18):
    """Fix a button's width to its text."""
    button.setFixedWidth(button.fontMetrics().horizontalAdvance(button.text()) + padding)


def _label_value(label, value):
    """'<dim>label</dim> value' for an overlay: the label recedes, the number stands out."""
    return f'<span style="color:#b0b0b0; font-weight:normal">{label}</span>&nbsp;{value}'


def _format_number(value, digits=3):
    """3 significant figures (or ``digits``), '3.51e5' rather than '3.51e+05'."""
    if not np.isfinite(value):
        return "–"
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{int(exponent)}"
    return text


def _get_colormap(cmap_name):
    try:
        return pg.colormap.get(cmap_name)
    except Exception:
        return pg.colormap.get(cmap_name, source='matplotlib')


class _PassRightClickRectROI(pg.ROI):
    """The crop rectangle: resized by its corner handles, moved by dragging its
    edges. The inside is not part of it, so dragging there pans the image and a
    marker inside can still be grabbed.

    It also leaves right-clicks to the plot's context menu: pg.ROI.hoverEvent claims
    every mouse button while hovered, so a right-click over the box went to the ROI
    (which has no menu) and was dropped. Here only left drags/clicks are claimed,
    plus right while dragging so it still cancels the move."""

    def __init__(self, pos, size, **kwargs):
        super().__init__(pos, size, **kwargs)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def _edge_band(self):
        """Half the width of the grabbable edge, in the item's coordinates."""
        try:
            px_w, px_h = self.pixelSize()
        except Exception:
            px_w = px_h = 0.0
        return ROI_EDGE_GRAB_PX * (px_w or 0.0), ROI_EDGE_GRAB_PX * (px_h or 0.0)

    def _rect(self):
        return QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()

    def boundingRect(self):
        bx, by = self._edge_band()
        return self._rect().adjusted(-bx, -by, bx, by)

    def shape(self):
        """A ring along the outline (odd-even fill: the inner rectangle is a hole)."""
        bx, by = self._edge_band()
        rect = self._rect()
        path = QPainterPath()
        path.addRect(rect.adjusted(-bx, -by, bx, by))
        inner = rect.adjusted(bx, by, -bx, -by)
        if inner.width() > 0 and inner.height() > 0:
            path.addRect(inner)
        return path

    def viewTransformChanged(self):
        self.prepareGeometryChange()        # the band is in screen pixels

    def hoverEvent(self, ev):
        hover = False
        if not ev.isExit():
            if self.translatable and ev.acceptDrags(Qt.MouseButton.LeftButton):
                hover = True
            if ev.acceptClicks(Qt.MouseButton.LeftButton):
                hover = True
            if self.isMoving:
                ev.acceptClicks(Qt.MouseButton.RightButton)
        if hover:
            self.setMouseHover(True)
            self.sigHoverEvent.emit(self)
        else:
            self.setMouseHover(False)


class LiveODViewer(QWidget):
    """The image area shared by the acquisition window and the remote viewer: a log
    panel, a toolbar, the three raw frames, and the OD image with its sumOD
    profiles drawn along the bottom and left edges.

    Keys: R reset zoom, P profiles, L live plot, Space pause, Left/Right step
    through recent shots, M add a marker, Delete/Backspace delete the marker under
    the cursor. Double-click the image to add a marker there.
    """

    live_plot_requested = pyqtSignal()   # emitted when the Live Plot button is clicked
    fk_tof_requested = pyqtSignal()      # emitted when the FK TOF button is clicked
    # The user moved / added / reshaped / deleted a pin: (camera_key, markers). The
    # window passes it to the liveOD server, which keeps the markers per camera.
    markers_changed = pyqtSignal(str, list)
    markers_updated = pyqtSignal()       # the markers shown changed, from any source (for the panel)

    def __init__(self):
        super().__init__()
        self.Nimg = 0
        self._first_image_received = 0
        self._first_image_minmax = {}
        self._autoscale_ready = False
        self._cmap_name = 'viridis'
        self._od_min = 0.0
        self._od_max = 2.5
        self._od_limits = DEFAULT_OD_LIMITS     # how far the colour scale may be pushed
        self._lock_views = False
        self._syncing_all_views = False
        self._syncing_image_views = False
        self._camera_key = ""
        self._roi_item = None           # pg.RectROI when active, else None
        self._last_od = None            # the latest shot's OD
        self._displayed_od = None       # what is on screen (may be an average, or an older shot)
        self._last_od_shape = None
        self._last_sumodx = None
        self._last_sumody = None
        self._px_size_m = None          # object-plane pixel size, once known
        self._fit = None                # the shown shot's scalars (fit geometry), None until fitted
        # history entries: [to_plot, shot_idx, scalars or None]
        self._history = deque(maxlen=HISTORY_MAX_SHOTS)
        self._shown_entry = None        # the history entry on screen
        self._early_scalars = {}        # shot_idx -> scalars that beat their image here
        self._shot_xvars = {}           # shot_idx -> that shot's xvar values (SI)
        self._latest_xvars = {}         # the last shot's, for when no image is up
        self._xvar_ranges = {}          # xvar name -> (min, max) of its scan, from INIT_RUN
        self._xvar_units = {}           # xvar name -> (unit, mult), fixed for the run
        self._history_pos = None        # index into _history while paused, None = live
        self._frames_seen = 0
        self._settings = None           # QSettings, if the window attached one
        self._marker_items = []         # MarkerItems on the OD image
        self._marker_panel = None       # MarkerPanel, made when first opened
        self._cmap_actions = []         # the colormap entries of every image's context menu
        self._last_right_click = None   # image coords of the last right-click, for "Add marker here"
        self._last_cursor = None        # image coords under the mouse, for the M key
        self.init_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def init_ui(self):
        self.output_window = LogPanel()     # `output_window`: the name its callers know

        self._build_toolbar()
        self._build_raw_images()
        self._build_od_plot()

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(self._raw_images_widget)
        self.main_splitter.addWidget(self.od_plot)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([100, 660])

        controls_container = QWidget()
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(2)
        controls_layout.addLayout(self._toolbar_row)
        controls_layout.addWidget(self.main_splitter, 1)
        controls_container.setLayout(controls_layout)

        self.top_splitter = QSplitter(Qt.Orientation.Vertical)
        self.top_splitter.addWidget(self.output_window)
        self.top_splitter.addWidget(controls_container)
        self.top_splitter.setStretchFactor(0, 0)
        self.top_splitter.setStretchFactor(1, 1)
        self.top_splitter.setSizes([70, 1000])
        self.top_splitter.setCollapsible(0, False)     # the log is never dragged shut

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.top_splitter)
        self.setLayout(layout)

        for key, slot in (("R", self.reset_zoom),
                          ("P", self.profiles_checkbox.toggle),
                          ("L", self.live_plot_requested.emit),
                          ("Space", self.pause_button.toggle),
                          ("M", self.add_marker),
                          ("Left", self.show_previous_shot),
                          ("Right", self.show_next_shot)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        # anywhere in the window: the mouse over a marker is what says which one. A
        # text box or spin box with the focus still gets its own Backspace/Delete.
        for key in ("Delete", "Backspace"):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(self.delete_hovered_marker)

        self._on_lock_views_changed(True)
        self._update_history_label()
        self._sync_roi_controls()
        self._apply_units()
        theme.notifier().changed.connect(self._restyle)

    def _build_toolbar(self):
        # --- view ---
        self.reset_zoom_button = QPushButton('Fit')
        self.reset_zoom_button.setToolTip("Reset the zoom: show the whole image (R)")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)

        self.lock_views_checkbox = QCheckBox("Lock views")
        self.lock_views_checkbox.setToolTip("Pan and zoom the raw frames and the OD image together")
        self.lock_views_checkbox.setChecked(True)
        self.lock_views_checkbox.stateChanged.connect(self._on_lock_views_changed)

        self.profiles_checkbox = QCheckBox("Profiles")
        self.profiles_checkbox.setToolTip(
            "sumOD along x (bottom edge) and y (left edge) over the ROI, or over the view "
            "if there is no ROI; orange: the Gaussian fits (P)")
        self.profiles_checkbox.setChecked(True)
        self.profiles_checkbox.stateChanged.connect(lambda _: self._update_marginals())

        # (`um_checkbox`: it was a checkbox in the View menu before it was a button)
        self.um_checkbox = QPushButton("µm")
        self.um_checkbox.setCheckable(True)
        self.um_checkbox.setFixedWidth(34)
        self.um_checkbox.setToolTip(
            "Show distances in µm at the atoms -- axes, cursor, widths, centre, ROI -- "
            "instead of camera pixels: pixel size / magnification from the run's camera "
            "parameters. Greyed out until a run has supplied them.")
        self.um_checkbox.setEnabled(False)
        self.um_checkbox.toggled.connect(lambda _: self._apply_units())

        # --- OD levels ---
        self.od_min_spinner = QDoubleSpinBox()
        self.od_max_spinner = QDoubleSpinBox()
        for spinner, value in ((self.od_min_spinner, self._od_min), (self.od_max_spinner, self._od_max)):
            spinner.setDecimals(1)
            spinner.setSingleStep(0.1)
            spinner.setRange(*self._od_limits)
            spinner.setValue(value)
            spinner.setFixedWidth(54)
        self.od_min_spinner.setToolTip("Bottom of the OD colour scale")
        self.od_max_spinner.setToolTip("Top of the OD colour scale")
        for spinner in (self.od_min_spinner, self.od_max_spinner):
            spinner.valueChanged.connect(self._on_od_spinner_changed)
        self.od_auto_button = QPushButton('Auto')
        self.od_auto_button.setToolTip("Set the OD colour scale from the shot on screen")
        self.od_auto_button.clicked.connect(self.auto_od_levels)

        # --- ROI ---
        self.roi_button = QPushButton('ROI')
        self.roi_button.setCheckable(True)
        self.roi_button.toggled.connect(self._on_roi_toggled)

        # --- other windows (FK TOF's button is gone while nobody uses it; the
        # fk_tof_requested signal stays so the windows' wiring need not change) ---
        self.live_plot_button = QPushButton('Plot')
        self.live_plot_button.setToolTip("Per-shot atom number and fit results against the scan (L)")
        self.live_plot_button.clicked.connect(self.live_plot_requested)
        self._windows_layout = QHBoxLayout()
        self._windows_layout.setSpacing(4)
        self._windows_layout.addWidget(self.live_plot_button)

        # --- shot history ---
        self.prev_shot_button = QPushButton('◀')
        self.next_shot_button = QPushButton('▶')
        self.pause_button = QPushButton('❚❚')     # not U+23F8: Windows draws that as a colour emoji
        for button, tip in ((self.prev_shot_button, "Previous shot (Left)"),
                            (self.next_shot_button, "Next shot (Right)"),
                            (self.pause_button, "Hold the display while the run carries on; "
                                                "new shots are still kept (Space)")):
            button.setFixedWidth(26)
            button.setToolTip(tip)
        self.prev_shot_button.clicked.connect(self.show_previous_shot)
        self.next_shot_button.clicked.connect(self.show_next_shot)
        self.pause_button.setFixedWidth(40)     # room for "Live", which it says when held
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._on_pause_toggled)
        self.history_label = QLabel()
        self.image_count_label = QLabel('Shot count: 0/0')

        # --- the things set once and left alone live in the View menu ---
        self.avg_spinner = QSpinBox()
        self.avg_spinner.setRange(1, 20)
        self.avg_spinner.setPrefix("Average last ")
        self.avg_spinner.setSuffix(" shot(s)")
        self.avg_spinner.setToolTip("Show the mean OD of the last N shots (1: off). Display only.")
        self.avg_spinner.valueChanged.connect(lambda _: self._redisplay())
        self.clear_button = QPushButton('Clear images')
        self.clear_button.setToolTip("Clear the images and the shot history")
        self.clear_button.clicked.connect(self.clear_plots)

        # Auto ROI: one row, the button and how many shots it looks at
        auto_roi_tip = ("Set the ROI around the atoms in the last N shots (up to the one on "
                        "screen): waxa's auto-ROI detector on the atoms and light frames")
        self.auto_roi_button = QPushButton('Auto ROI')
        self.auto_roi_button.setToolTip(auto_roi_tip)
        self.auto_roi_button.clicked.connect(lambda: self.auto_roi())
        self.auto_roi_spinner = QSpinBox()
        self.auto_roi_spinner.setRange(1, HISTORY_MAX_SHOTS)
        self.auto_roi_spinner.setValue(AUTO_ROI_DEFAULT_SHOTS)
        self.auto_roi_spinner.setPrefix("last ")
        self.auto_roi_spinner.setToolTip(auto_roi_tip)
        self.auto_roi_row = QWidget()
        auto_roi_layout = QHBoxLayout()
        auto_roi_layout.setContentsMargins(0, 0, 0, 0)
        auto_roi_layout.setSpacing(4)
        auto_roi_layout.addWidget(self.auto_roi_button)
        auto_roi_layout.addWidget(self.auto_roi_spinner)
        self.auto_roi_row.setLayout(auto_roi_layout)

        self.view_button = QToolButton()
        self.view_button.setText("View  ")    # room for the menu arrow
        self.view_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.view_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.view_button.setToolTip("Lock views, profiles (P), shot averaging, auto ROI, clear")
        view_menu = QMenu(self.view_button)
        self.markers_button = QPushButton('Markers')
        self.markers_button.setToolTip(
            "The markers on this camera's image: add, label, reshape, resize, recolor, hide, "
            "delete. Hover a row to light its marker up.\n"
            "On the image: double-click, right-click for 'Add marker here', or press M; drag a marker to move "
            "it, right-click a marker to edit it, Delete or Backspace over a marker to delete it. "
            "Markers are kept per camera by the liveOD server.")
        self.markers_button.clicked.connect(self.open_marker_panel)
        for widget in (self.lock_views_checkbox, self.profiles_checkbox,
                       None, self.avg_spinner, self.auto_roi_row,
                       None, self.clear_button):
            if widget is None:
                view_menu.addSeparator()
                continue
            holder = QWidget()
            holder_layout = QHBoxLayout()
            holder_layout.setContentsMargins(10, 3, 10, 3)
            holder_layout.addWidget(widget)
            holder.setLayout(holder_layout)
            action = QWidgetAction(view_menu)
            action.setDefaultWidget(holder)
            view_menu.addAction(action)
        self.clear_button.clicked.connect(view_menu.close)
        self.auto_roi_button.clicked.connect(view_menu.close)
        self.view_button.setMenu(view_menu)

        self.settings_button = QToolButton()
        self.settings_button.setText("⚙")
        self.settings_button.setToolTip("Settings: OD scale limits, dark mode")
        self.settings_button.clicked.connect(self.open_settings_dialog)

        # Qt gives a push button a 75 px minimum whatever it says; a toolbar of short
        # words is a third narrower with each sized to its text.
        for button in (self.reset_zoom_button, self.roi_button, self.od_auto_button,
                       self.live_plot_button, self.markers_button, self.auto_roi_button):
            _fit_to_text(button)
        self.auto_roi_spinner.setFixedWidth(self.auto_roi_spinner.sizeHint().width())

        # One row, left to right: the image, its colour scale, which shot; the other
        # windows and the settings sit at the right-hand end.
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        # The text of these two changes from shot to shot; at a fixed width that
        # cannot move anything else or widen the window.
        self.history_label.setFixedWidth(50)
        self.image_count_label.setFixedWidth(104)
        # (no "OD" caption: the two boxes' tooltips and the colour bar say what they are)
        for widget in (self.reset_zoom_button, self.view_button, self.roi_button,
                       self.um_checkbox, self.markers_button, _vline(),
                       self.od_min_spinner, self.od_max_spinner,
                       self.od_auto_button, _vline(),
                       self.prev_shot_button, self.next_shot_button, self.pause_button,
                       self.history_label, self.image_count_label):
            row.addWidget(widget)
        row.addStretch()
        row.addLayout(self._windows_layout)
        row.addWidget(self.settings_button)
        self._toolbar_row = row

    def add_window_button(self, button):
        """Put a window's own button (Adjust, ...) next to Live Plot."""
        self._windows_layout.addWidget(button)

    def _build_raw_images(self):
        self.img_atoms_view = pg.ImageView()
        self.img_light_view = pg.ImageView()
        self.img_dark_view = pg.ImageView()
        self._raw_titles = {}
        self._raw_labels = {}

        self._raw_images_widget = QSplitter(Qt.Orientation.Horizontal)
        for name, view, title in (('atoms', self.img_atoms_view, 'Atoms + light'),
                                  ('light', self.img_light_view, 'Light only'),
                                  ('dark', self.img_dark_view, 'Dark')):
            view.ui.histogram.hide(); view.ui.roiBtn.hide(); view.ui.menuBtn.hide()
            view.setMinimumHeight(60)
            self.set_pg_colormap(view, 'viridis')
            # title and max-count readout drawn in the corner of the frame itself
            label = pg.LabelItem(justify='left')
            label.setParentItem(view.getView())
            label.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(4, 2))
            self._raw_titles[name] = title
            self._raw_labels[name] = label
            self._set_raw_label(name)
            self._raw_images_widget.addWidget(view)
            self._extend_image_menu(view.getView(), view.scene)
            view.getView().sigRangeChanged.connect(
                lambda *args, v=view: self._sync_image_views(v))
            view.getView().sigRangeChanged.connect(
                lambda *args, n=name: self._on_any_view_range_changed(n))

    def _build_od_plot(self):
        self.od_plot = pg.PlotWidget()
        self.od_plot.setXRange(0, 512, padding=0)
        self.od_plot.setYRange(0, 512, padding=0)
        self.od_plot.setAspectLocked(True)
        self.od_plot.setMouseEnabled(x=True, y=True)
        self.od_plot.setMenuEnabled(True)
        self.od_plot.hideAxis('right'); self.od_plot.hideAxis('top')
        self.od_plot.showGrid(x=False, y=False)

        self.od_img_item = pg.ImageItem()
        self.od_img_item.setZValue(-10)
        self.od_plot.addItem(self.od_img_item)

        # colour scale: drag its ends (or the whole bar) to set the OD levels
        self.od_colorbar = pg.ColorBarItem(values=(self._od_min, self._od_max),
                                           colorMap=_get_colormap(self._cmap_name),
                                           interactive=True, rounding=OD_STEP, width=14,
                                           limits=self._od_limits,
                                           colorMapMenu=False)  # the images' own menu sets it for all
        # pyqtgraph reserves 45 px for the bar's numbers; "10.0" needs about half that
        self.od_colorbar.axis.setWidth(26)
        self.od_colorbar.axis.setStyle(tickTextOffset=2, tickLength=-3)
        self.od_colorbar.layout.setContentsMargins(0, 0, 0, 0)
        self.od_plot.getPlotItem().layout.setContentsMargins(1, 7, 1, 1)    # top: the bar's top number
        self.od_colorbar.setImageItem(self.od_img_item, insert_in=self.od_plot.getPlotItem())
        self.od_colorbar.sigLevelsChanged.connect(self._on_colorbar_levels_changed)
        self.od_colorbar.sigLevelsChangeFinished.connect(self._on_colorbar_drag_finished)

        # sumOD profiles along the bottom and left edges, with their fits
        profile_pen = pg.mkPen((255, 255, 255, 230), width=1.5)
        fit_pen = pg.mkPen((255, 120, 0, 255), width=1.5)
        self._marginal_items = []
        def curve(pen, z):
            item = pg.PlotCurveItem(pen=pen)
            item.setZValue(z)
            self.od_plot.addItem(item, ignoreBounds=True)
            self._marginal_items.append(item)
            return item
        self._sumx_curve, self._sumx_base = curve(profile_pen, 5), curve(None, 5)
        self._sumy_curve, self._sumy_base = curve(profile_pen, 5), curve(None, 5)
        self._fitx_curve, self._fity_curve = curve(fit_pen, 6), curve(fit_pen, 6)
        for a, b in ((self._sumx_curve, self._sumx_base), (self._sumy_curve, self._sumy_base)):
            fill = pg.FillBetweenItem(a, b, brush=(255, 255, 255, 50))
            fill.setZValue(4)
            self.od_plot.addItem(fill, ignoreBounds=True)
            self._marginal_items.append(fill)

        # per-shot numbers top right of the image, cursor readout bottom right: bold
        # on a translucent plate, so they read over any part of the image
        self.readout_label = _OverlayLabel(self.od_plot)
        self.cursor_label = _OverlayLabel(self.od_plot)
        # one width each, whatever the numbers and whichever units
        self.readout_label.fix_width_for(" &nbsp;&nbsp; ".join((
            _label_value("ΣOD", "8.88e8"), _label_value("σ", "888.8 × 888.8 µm"),
            _label_value("center", "8888, 8888 µm"))))
        self.cursor_label.fix_width_for(" &nbsp;&nbsp; ".join((
            _label_value("x, y", "8888, 8888 µm"), _label_value("OD", "−8.88"))))
        # which units the axes, readouts and ROI are in: bottom left
        self.units_label = _OverlayLabel(self.od_plot)
        # the shown shot's xvar values: top left
        self.xvar_label = _OverlayLabel(self.od_plot)
        self.od_plot.installEventFilter(self)

        # markers and the colormap in the image's right-click menu (M also adds a marker)
        self._extend_image_menu(self.od_plot.getViewBox(), self.od_plot.scene())
        self.od_plot.getViewBox().sigResized.connect(lambda *_: self._place_overlays())

        vb = self.od_plot.getViewBox()
        vb.sigRangeChanged.connect(self.sync_sumod_panels)
        vb.sigRangeChanged.connect(lambda *args: self._on_any_view_range_changed('od'))
        self._mouse_proxy = pg.SignalProxy(self.od_plot.scene().sigMouseMoved,
                                           rateLimit=30, slot=self._on_mouse_moved)

    # ------------------------------------------------------------------
    # View locking
    # ------------------------------------------------------------------

    def _on_lock_views_changed(self, state):
        self._lock_views = bool(state)
        if self._lock_views:
            # When locking, bring the raw frames to what the OD plot shows
            ref_range = self.od_plot.getViewBox().viewRange()
            self._sync_all_views(ref_range, exclude='od')

    def _on_any_view_range_changed(self, source):
        if not self._lock_views or self._syncing_all_views:
            return
        views = {'atoms': self.img_atoms_view, 'light': self.img_light_view,
                 'dark': self.img_dark_view, 'od': self.od_plot}
        widget = views.get(source)
        # Only the view the user is dragging or wheeling leads. The views keep
        # their aspect ratio and have different shapes, so each one adjusts the
        # range it is given and reports that a moment later; following those
        # reports too would walk every view away from what the user chose.
        if widget is None or not widget.underMouse():
            return
        self._sync_all_views(widget.getView().viewRange() if source != 'od'
                             else self.od_plot.getViewBox().viewRange(), exclude=source)

    def _sync_all_views(self, ref_range, exclude=None):
        self._syncing_all_views = True
        try:
            # ref_range: [xRange, yRange]
            for name, view in (('atoms', self.img_atoms_view.getView()),
                               ('light', self.img_light_view.getView()),
                               ('dark', self.img_dark_view.getView()),
                               ('od', self.od_plot.getViewBox())):
                if name != exclude:
                    view.setRange(xRange=ref_range[0], yRange=ref_range[1], padding=0)
        finally:
            self._syncing_all_views = False
        self._update_marginals()

    def _sync_image_views(self, source_view):
        if self._syncing_image_views:
            return
        self._syncing_image_views = True
        try:
            src_range = source_view.getView().viewRange()
            for v in (self.img_atoms_view, self.img_light_view, self.img_dark_view):
                if v is not source_view:
                    v.getView().setRange(xRange=src_range[0], yRange=src_range[1], padding=0)
        finally:
            self._syncing_image_views = False

    # ------------------------------------------------------------------
    # Colours and levels
    # ------------------------------------------------------------------

    def set_pg_colormap(self, imgitem, cmap_name):
        import matplotlib
        lut = (matplotlib.colormaps[cmap_name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        if hasattr(imgitem, 'imageItem'):
            imgitem.imageItem.setLookupTable(lut)
            imgitem.imageItem.lut = lut
        else:
            imgitem.setLookupTable(lut)
            imgitem.lut = lut

    def set_all_colormaps(self, cmap_name):
        """One colormap for the three raw frames, the OD image and its colour bar
        (their levels stay separate). Chosen from any image's right-click menu."""
        try:
            colormap = _get_colormap(cmap_name)
            for v in [self.img_atoms_view, self.img_light_view, self.img_dark_view]:
                self.set_pg_colormap(v, cmap_name)
        except Exception as exc:
            logger.warning(f"Unknown colormap {cmap_name!r}: {exc}")
            return
        self._cmap_name = cmap_name
        self.od_colorbar.setColorMap(colormap)
        for action in self._cmap_actions:
            action.setChecked(action.text() == cmap_name)
        if self._settings is not None:
            self._settings.setValue("viewer/colormap", cmap_name)

    def set_od_levels(self, od_min, od_max):
        """Set the OD colour scale from anywhere but the colour bar itself
        (spinboxes, Auto, remembered levels)."""
        lo_lim, hi_lim = self._od_limits
        od_min = min(max(float(od_min), lo_lim), hi_lim)
        od_max = min(max(float(od_max), lo_lim), hi_lim)
        if od_max - od_min < OD_STEP:
            # a collapsed scale (e.g. levels remembered from outside the limits):
            # nothing would be visible, so start again from the default
            od_min, od_max = (max(DEFAULT_OD_LEVELS[0], lo_lim), min(DEFAULT_OD_LEVELS[1], hi_lim))
            if od_max - od_min < OD_STEP:
                od_min, od_max = lo_lim, hi_lim
        self._show_od_levels(od_min, od_max)
        self.od_colorbar.blockSignals(True)
        self.od_colorbar.setLevels((od_min, od_max))    # also sets the image's levels
        self.od_colorbar.blockSignals(False)
        self._save_od_levels()

    def set_od_limits(self, lo_lim, hi_lim):
        """How far the OD colour scale can be taken, by dragging the bar or in the
        spinboxes. The current levels are pulled inside."""
        lo_lim, hi_lim = float(lo_lim), float(hi_lim)
        if hi_lim - lo_lim < 2 * OD_STEP:
            return
        self._od_limits = (lo_lim, hi_lim)
        for spinner in (self.od_min_spinner, self.od_max_spinner):
            spinner.blockSignals(True)
            spinner.setRange(lo_lim, hi_lim)
            spinner.blockSignals(False)
        # ColorBarItem takes its limits only at construction; these are the attributes
        # its drag handler and setLevels clip against
        self.od_colorbar.lo_lim, self.od_colorbar.hi_lim = lo_lim, hi_lim
        self.set_od_levels(self._od_min, self._od_max)
        if self._settings is not None:
            self._settings.setValue("viewer/od_limits", [lo_lim, hi_lim])

    def _show_od_levels(self, od_min, od_max):
        self._od_min, self._od_max = od_min, od_max
        for spinner, value in ((self.od_min_spinner, od_min), (self.od_max_spinner, od_max)):
            spinner.blockSignals(True)
            spinner.setValue(value)
            spinner.blockSignals(False)

    def _on_od_spinner_changed(self, _value):
        self.set_od_levels(self.od_min_spinner.value(), self.od_max_spinner.value())

    def _on_colorbar_levels_changed(self, *_):
        # Mid-drag. The bar has already applied these levels to itself and the image,
        # so only follow it. Do NOT call od_colorbar.setLevels here: that also resets
        # the level the drag is measured from, so every mouse move would add the
        # handle's whole displacement again and the level runs away.
        self._show_od_levels(*(float(v) for v in self.od_colorbar.levels()))

    def _on_colorbar_drag_finished(self, *_):
        # pyqtgraph reports a release twice (snapping the handles back counts as one)
        levels = (self._od_min, self._od_max)
        if levels != getattr(self, '_od_levels_at_last_release', None):
            self._od_levels_at_last_release = levels
            self._save_od_levels()

    def auto_od_levels(self):
        """0 to the 99.9th percentile of the OD in the crop region."""
        od = self._displayed_od
        if od is None:
            return
        x0, x1, y0, y1 = self.get_crop_bounds(od.shape)
        region = od[y0:y1, x0:x1]
        region = region[np.isfinite(region)]
        if region.size == 0:
            return
        self.set_od_levels(0.0, max(0.1, round(float(np.percentile(region, 99.9)), 1)))

    # ------------------------------------------------------------------
    # Clearing / run boundaries
    # ------------------------------------------------------------------

    def clear_plots(self):
        self.img_atoms_view.clear()
        self.img_light_view.clear()
        self.img_dark_view.clear()
        self.od_img_item.clear()
        self._last_od = None
        self._displayed_od = None
        self._last_sumodx = None
        self._last_sumody = None
        self._last_od_shape = None
        self._first_image_received = 0
        self._first_image_minmax = {}
        self._autoscale_ready = False
        self._fit = None
        self._early_scalars.clear()
        self.readout_label.show_text("")
        self._clear_xvars()
        for name in self._raw_labels:
            self._set_raw_label(name)
        self._clear_history()
        self._update_marginals()
        if hasattr(self, '_last_view_ranges'):
            del self._last_view_ranges

    def on_new_run(self):
        """A run is starting: the shot history is per run. The last image stays up
        until the new run's first shot replaces it."""
        self._clear_history()
        self._fit = None
        self._early_scalars.clear()
        self.readout_label.show_text("")
        self._clear_xvars()
        # a new series: the raw frames' levels are taken afresh from its first shot
        self._autoscale_ready = False
        self._first_image_minmax = {}

    def update_image_count(self, count, total):
        self.image_count_label.setText(f'Shot count: {count}/{total}')

    def get_img_number(self, N_img, N_shots, N_pwa_per_shot, run_id=None):
        self.Nimg = N_img
        if run_id is not None:
            self._current_run_id = run_id

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def handle_plot_data(self, to_plot):
        """Slot for LiveODPlotter: (atoms, light, dark, od, sum_od_x, sum_od_y)."""
        self._frames_seen += 1
        shot_idx = getattr(to_plot, 'shot_idx', None)
        entry = [to_plot, shot_idx, None]
        if shot_idx is not None:
            entry[2] = self._early_scalars.pop(shot_idx, None)
            # anything older is for a shot the plotter dropped
            for idx in [k for k in self._early_scalars if k < shot_idx]:
                del self._early_scalars[idx]
        self._remember(entry)
        if self._history_pos is None:
            self._display_latest(entry)
        self._update_history_label()

    def _display(self, entry, od=None):
        img_atoms, img_light, img_dark, shot_od, sum_od_x, sum_od_y = entry[0]
        # this shot's fit, or none until it is done: never an older shot's
        self._shown_entry = entry
        self._fit = entry[2]
        self._render_readout()
        self._render_xvars()
        self._syncing_image_views = True # workaround for not having image sizes reset on replotting new images
        try:
            self.plot_images(img_atoms, img_light, img_dark)
        finally:
            self._syncing_image_views = False
        self.plot_od(shot_od if od is None else od, sum_od_x, sum_od_y)

    def plot_images(self, atoms, light, dark):
        # Determine if this is the first image after a clear
        is_first = not hasattr(self, '_last_view_ranges') or self._first_image_received == 0

        views = (self.img_atoms_view, self.img_light_view, self.img_dark_view)
        self._last_view_ranges = [v.getView().viewRange() for v in views]

        if not self._autoscale_ready:
            # Levels are set by the first shot of a series (a run, or after Clear) and
            # then held, so shot-to-shot changes in brightness stay visible. Atoms and
            # light share one range, covering both, so they can be compared by eye;
            # the dark frame has its own.
            self._first_image_minmax['atoms_light'] = (
                float(min(np.min(atoms), np.min(light))), float(max(np.max(atoms), np.max(light))))
            self._first_image_minmax['dark'] = (float(np.min(dark)), float(np.max(dark)))
            self._autoscale_ready = True
        atoms_levels = self._first_image_minmax['atoms_light']
        dark_levels = self._first_image_minmax['dark']
        self.img_atoms_view.setImage(atoms.T, autoLevels=False, levels=atoms_levels)
        self.img_light_view.setImage(light.T, autoLevels=False, levels=atoms_levels)
        self.img_dark_view.setImage(dark.T, autoLevels=False, levels=dark_levels)
        self._last_atoms = atoms
        self._last_light = light
        self._last_dark = dark
        for name, img in (('atoms', atoms), ('light', light), ('dark', dark)):
            self._set_raw_label(name, img)

        if is_first:
            shape = atoms.T.shape
            for v in views:
                v.getView().setRange(xRange=[0, shape[0]], yRange=[0, shape[1]], padding=0)
            self._first_image_received = 1
        else:
            # setImage resets the view; put it back where the user had it
            for v, (x_range, y_range) in zip(views, self._last_view_ranges):
                v.getView().setRange(xRange=x_range, yRange=y_range, padding=0)

    def _set_raw_label(self, name, img=None):
        text = self._raw_titles[name]
        if img is not None and np.size(img):
            peak = np.max(img)
            text += f" · max {peak:.0f}"
            if name != 'dark' and int(peak) in SATURATION_LEVELS \
                    and np.count_nonzero(img == peak) >= SATURATION_MIN_PIXELS:
                text += ' · <span style="color:#ff5252; font-weight:bold">SATURATED</span>'
        self._raw_labels[name].setText(text, color='#dddddd', size='8pt')

    def plot_od(self, od, sumodx, sumody, min_od=None, max_od=None):
        # If this is the first OD after a clear, set axes to match its shape
        if self._last_od_shape is None and od is not None:
            self.od_plot.setXRange(0, od.shape[1], padding=0)
            self.od_plot.setYRange(0, od.shape[0], padding=0)
        if min_od is None or max_od is None:
            min_od, max_od = self._od_min, self._od_max
        self.od_img_item.setImage(od.T, autoLevels=False, levels=(min_od, max_od))
        self._last_od_shape = od.shape
        self._last_od = od
        self._displayed_od = od
        self._last_sumodx = sumodx
        self._last_sumody = sumody
        self._update_marginals()

    def reset_zoom(self):
        shape = self._last_od_shape if self._last_od_shape is not None else (512, 512)
        self.od_plot.setXRange(0, shape[1], padding=0)
        self.od_plot.setYRange(0, shape[0], padding=0)
        if self._lock_views:
            self._sync_all_views([[0, shape[1]], [0, shape[0]]], exclude='od')
        self._update_marginals()

    # ------------------------------------------------------------------
    # Crop region and sumOD profiles
    # ------------------------------------------------------------------

    def get_od_view_range(self):
        """(x_range, y_range) of the OD plot, each [min, max]."""
        try:
            x_range, y_range = self.od_plot.getViewBox().viewRange()
            return x_range, y_range
        except Exception:
            return [0, 512], [0, 512]

    def get_crop_bounds(self, od_shape):
        """(x0, x1, y0, y1): the pixels that atom number, profiles and fits are
        taken over -- the drawn ROI if there is one, else what the OD plot shows.
        The Analyzer crops with this too, so both always mean the same region."""
        h, w = od_shape[0], od_shape[1]
        rect = self.get_od_roi_rect()
        if rect is not None:
            x0, x1 = max(0, int(round(rect[0]))), min(w, int(round(rect[2])))
            y0, y1 = max(0, int(round(rect[1]))), min(h, int(round(rect[3])))
            if x1 > x0 and y1 > y0:
                return x0, x1, y0, y1
            # drawn rect is degenerate or off the image: fall through to the view
        x_range, y_range = self.get_od_view_range()
        x0, x1 = max(0, int(round(x_range[0]))), min(w, int(round(x_range[1])))
        y0, y1 = max(0, int(round(y_range[0]))), min(h, int(round(y_range[1])))
        return x0, max(x0, x1), y0, max(y0, y1)

    def sync_sumod_panels(self, *_):
        """Redraw the profiles for the current view (name kept from when they were
        separate panels)."""
        self._update_marginals()

    def _update_marginals(self):
        od = self._displayed_od
        show = od is not None and self.profiles_checkbox.isChecked()
        if show:
            x0, x1, y0, y1 = self.get_crop_bounds(od.shape)
            show = x1 - x0 > 1 and y1 - y0 > 1
        for item in self._marginal_items:
            item.setVisible(bool(show))
        if not show:
            return

        crop = np.nan_to_num(od[y0:y1, x0:x1], nan=0.0, posinf=0.0, neginf=0.0)
        sum_x, sum_y = crop.sum(axis=0), crop.sum(axis=1)
        vb = self.od_plot.getViewBox()
        (vx0, vx1), (vy0, vy1) = vb.viewRange()
        # The part of the image on screen: the plot keeps the image's aspect ratio,
        # so a wide window shows empty margins, and the profiles belong on the
        # image's edge, not out at the margin's.
        vx0, vx1 = max(vx0, 0.0), min(vx1, float(od.shape[1]))
        vy0, vy1 = max(vy0, 0.0), min(vy1, float(od.shape[0]))
        if vx1 <= vx0 or vy1 <= vy0:
            for item in self._marginal_items:
                item.setVisible(False)
            return
        # the bottom edge is the high end of y if the axis is flipped
        y_base, y_dir = (vy1, -1.0) if vb.yInverted() else (vy0, 1.0)
        x_base, x_dir = (vx1, -1.0) if vb.xInverted() else (vx0, 1.0)
        scale_x = PROFILE_FRACTION * (vy1 - vy0) / sum_x.max() if sum_x.max() > 0 else 0.0
        scale_y = PROFILE_FRACTION * (vx1 - vx0) / sum_y.max() if sum_y.max() > 0 else 0.0
        cols = np.arange(x0, x1) + 0.5
        rows = np.arange(y0, y1) + 0.5

        self._sumx_curve.setData(cols, y_base + y_dir * scale_x * sum_x)
        self._sumx_base.setData(cols, np.full(cols.size, y_base))
        self._sumy_curve.setData(x_base + x_dir * scale_y * sum_y, rows)
        self._sumy_base.setData(np.full(rows.size, x_base), rows)

        fit = self._fit
        fit_matches = (fit is not None
                       and tuple(fit.get('crop_origin_px', ())) == (x0, y0)
                       and tuple(fit.get('crop_shape_px', ())) == (x1 - x0, y1 - y0))
        for axis, curve_item, pixels in (('x', self._fitx_curve, cols), ('y', self._fity_curve, rows)):
            model = self._fit_model(fit, axis, pixels.size) if fit_matches else None
            if model is None:
                curve_item.setVisible(False)
            elif axis == 'x':
                curve_item.setData(pixels, y_base + y_dir * scale_x * model)
            else:
                curve_item.setData(x_base + x_dir * scale_y * model, pixels)

    @staticmethod
    def _fit_model(fit, axis, n):
        """The Gaussian the Analyzer fitted to this axis's profile, sampled at the
        crop's n pixels; None if that fit failed."""
        try:
            dx = float(fit['px_size_m'])
            amp, sigma = float(fit[f'fit_amp_{axis}']), float(fit[f'fit_sd_{axis}'])
            center, offset = float(fit[f'fit_center_{axis}']), float(fit[f'fit_offset_{axis}'])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite([dx, amp, sigma, center, offset])) or sigma == 0:
            return None
        axis_m = dx * np.arange(n)
        return offset + amp * np.exp(-(axis_m - center) ** 2 / (2 * sigma ** 2))

    # ------------------------------------------------------------------
    # Per-shot numbers
    # ------------------------------------------------------------------

    def on_shot_scalars(self, scalars):
        """Slot for ``Analyzer.shot_scalars_signal`` (or the remote viewer's copy):
        atom number, widths and centre in the corner of the OD image, and the fits
        over the profiles. Kept with the shot they belong to, so they come back when
        that shot is stepped to."""
        if not isinstance(scalars, dict):
            return
        if scalars.get('px_calibrated', False):
            self.set_pixel_size_m(scalars.get('px_size_m'))
        shot_idx = scalars.get('shot_idx')
        tagged = not self._history or self._history[-1][1] is not None
        entry = None
        if shot_idx is not None and tagged:
            entry = next((e for e in reversed(self._history) if e[1] == shot_idx), None)
            if entry is None:
                # its image has not reached us yet (or was dropped by the plotter)
                self._early_scalars[shot_idx] = scalars
                return
        elif self._history:
            # images without shot_idx (a server that predates it): the latest shot
            entry = self._history[-1]
        if entry is not None:
            entry[2] = scalars
            if entry is not self._shown_entry:
                return
        self._fit = scalars
        self._render_readout()
        self._update_marginals()

    def _render_readout(self):
        """N, widths and centre from the latest shot's scalars, in the units the µm
        button selects."""
        scalars = self._fit
        if not scalars:
            self.readout_label.show_text("")
            return
        nan = float('nan')
        calibrated = bool(scalars.get('px_calibrated', False))
        number = float(scalars.get('atom_number', nan))
        is_atoms = calibrated and np.isfinite(scalars.get('atom_cross_section_m2', nan))
        parts = [_label_value("N" if is_atoms else "ΣOD", _format_number(number))]

        # The fits are in their own axis: px_size_m * (index within the crop), with
        # px_size_m = 1 when there is no pixel calibration. Back to pixels first.
        dx = float(scalars.get('px_size_m', 1.0) or 1.0)
        unit, scale = self._length_unit()
        sd_x = float(scalars.get('fit_sd_x', nan)) / dx * scale
        sd_y = float(scalars.get('fit_sd_y', nan)) / dx * scale
        parts.append(_label_value("σ", f"{sd_x:.1f} × {sd_y:.1f} {unit}"))
        origin = scalars.get('crop_origin_px')
        if origin is not None:
            cx = (origin[0] + float(scalars.get('fit_center_x', nan)) / dx) * scale
            cy = (origin[1] + float(scalars.get('fit_center_y', nan)) / dx) * scale
            parts.append(_label_value("center", f"{cx:.0f}, {cy:.0f} {unit}"))
        self.readout_label.show_text(" &nbsp;&nbsp; ".join(parts))
        self._place_overlays()

    # ------------------------------------------------------------------
    # xvars (top left of the OD image)
    # ------------------------------------------------------------------

    def set_xvar_ranges(self, ranges):
        """The run's scan ranges, ``{name: (min, max)}`` in SI, from INIT_RUN. Each
        xvar's display unit is picked from its range, so it does not change from
        shot to shot. Call after ``on_new_run``."""
        self._xvar_ranges = {}
        for name, bounds in dict(ranges or {}).items():
            try:
                lo, hi = (float(v) for v in bounds)
            except (TypeError, ValueError):
                continue
            self._xvar_ranges[str(name)] = (lo, hi)
        self._xvar_units = {}

    def set_shot_xvars(self, shot_idx, xvar_values):
        """Slot for the server's shot progress: this shot's xvar values (SI), kept
        with the shot so they come back when it is stepped to."""
        if not xvar_values:
            return
        xvar_values = dict(xvar_values)
        self._latest_xvars = xvar_values
        if shot_idx is not None:
            self._shot_xvars[int(shot_idx)] = xvar_values
            for idx in sorted(self._shot_xvars)[:-4 * HISTORY_MAX_SHOTS]:
                del self._shot_xvars[idx]
        shown = self._shown_entry
        if shown is None or shown[1] is None or shown[1] == shot_idx:
            self._render_xvars()

    def _clear_xvars(self):
        self._shot_xvars.clear()
        self._latest_xvars = {}
        self._xvar_ranges = {}
        self._xvar_units = {}
        self.xvar_label.setMinimumWidth(0)
        self.xvar_label.show_text("")

    def _xvar_unit(self, name, value):
        """(unit, mult) for an xvar, fixed for the run: from its scan range when
        INIT_RUN gave one, else from the first value seen."""
        if name not in self._xvar_units:
            lo, hi = self._xvar_ranges.get(name, (value, value))
            try:
                magnitude = max(abs(float(lo)), abs(float(hi)))
            except (TypeError, ValueError):
                return "", 1.0      # not a number: shown as it is
            unit = unit_for_param(name, [magnitude])
            self._xvar_units[name] = (unit, mult_for(unit) if unit else 1.0)
        return self._xvar_units[name]

    def _format_xvar(self, name, value):
        unit, mult = self._xvar_unit(name, value)
        try:
            text = _format_number(float(value) * mult, digits=4)
        except (TypeError, ValueError):
            text = str(value)
        return _label_value(name, f"{text} {unit}".rstrip())

    def _render_xvars(self):
        """The xvars of the shot on screen; before a run's first image, the latest."""
        shown = self._shown_entry
        if shown is None or shown[1] is None:
            xvars = self._latest_xvars
        else:
            xvars = self._shot_xvars.get(shown[1])
            if xvars is None and shown[2]:
                xvars = shown[2].get('xvar_values')     # a remote viewer's scalars carry them
            if xvars is None:
                return          # its progress message is still on the way: leave what is up
        if not xvars:
            self.xvar_label.show_text("")
            return
        self.xvar_label.show_text("<br>".join(self._format_xvar(str(k), v) for k, v in xvars.items()))
        # grows to the widest it has been this run, never back: no jumping about
        self.xvar_label.setMinimumWidth(max(self.xvar_label.minimumWidth(),
                                            self.xvar_label.sizeHint().width()))
        self._place_overlays()

    def set_pixel_size_m(self, px_size_m):
        """Size of one camera pixel at the atoms (pixel size / magnification), which
        is what the µm button converts with. None: unknown, so pixels only."""
        try:
            px_size_m = float(px_size_m)
            if not np.isfinite(px_size_m) or px_size_m <= 0:
                px_size_m = None
        except (TypeError, ValueError):
            px_size_m = None
        if px_size_m == self._px_size_m:
            return
        self._px_size_m = px_size_m
        self.um_checkbox.setEnabled(px_size_m is not None)
        self._apply_units()

    def _length_unit(self):
        """('µm', µm per pixel) when the µm button is on and the pixel size is known,
        else ('px', 1.0)."""
        if self.um_checkbox.isChecked() and self._px_size_m is not None:
            return "µm", self._px_size_m * 1e6
        return "px", 1.0

    def _apply_units(self):
        """Everything that shows a distance: the axes, the per-shot readout, the ROI
        tooltip (the cursor readout follows on the next mouse move)."""
        unit, scale = self._length_unit()
        for name in ('bottom', 'left'):
            axis = self.od_plot.getAxis(name)
            axis.setScale(scale)
            axis.showLabel(False)       # the units are on the plate in the corner instead
        self.units_label.show_text(unit)
        self.markers_updated.emit()         # the markers panel shows sizes in these units
        self._render_readout()
        self._sync_roi_controls()
        self.cursor_label.show_text("")
        self._place_overlays()

    def _place_overlays(self):
        """Keep the overlays in the corners of the image area (inside the axes, left
        of the colour bar): per-shot numbers top right, cursor bottom right, units
        bottom left, xvars top left."""
        rect = self.od_plot.mapFromScene(self.od_plot.getViewBox().sceneBoundingRect()).boundingRect()
        margin = 6
        for label, at_top, at_right in ((self.readout_label, True, True),
                                        (self.cursor_label, False, True),
                                        (self.units_label, False, False),
                                        (self.xvar_label, True, False)):
            x = rect.right() - label.width() - margin if at_right else rect.left() + margin
            y = rect.top() + margin if at_top else rect.bottom() - label.height() - margin
            label.move(int(max(rect.left(), x)), int(y))

    # ------------------------------------------------------------------
    # Markers (pins). The viewer only draws and edits them: the window hands it the
    # current camera's markers and takes `markers_changed` to the liveOD server,
    # which is where they are kept.
    # ------------------------------------------------------------------

    def get_markers(self) -> list:
        return [item.to_dict() for item in self._marker_items]

    def set_markers(self, markers):
        """Show these markers. Does not emit ``markers_changed``. Ignored while the
        user is dragging one, so a broadcast cannot pull it out of their hand."""
        markers = clean_markers(markers)
        if any(item.moving for item in self._marker_items):
            return
        if markers == self.get_markers():
            return
        for item in self._marker_items:
            self.od_plot.removeItem(item)
        self._marker_items = []
        for marker in markers:
            item = MarkerItem(marker)
            item.sigMoved.connect(lambda *_: self._emit_markers())
            item.sigEditRequested.connect(self._open_marker_dialog)
            self.od_plot.addItem(item, ignoreBounds=True)
            self._marker_items.append(item)
        self.markers_updated.emit()

    def add_marker(self, pos=None, **fields):
        """A new marker at ``pos`` (image pixels); default: under the cursor, else
        the middle of what is on screen. ``fields``: shape, size (image pixels),
        color (default: a random one), label."""
        if pos is None:
            pos = self._last_cursor
        if pos is None:
            (x0, x1), (y0, y1) = self.get_od_view_range()
            pos = ((x0 + x1) / 2, (y0 + y1) / 2)
        if "size" not in fields:
            # a sensible size for what is on screen: a fraction of the view width
            (x0, x1), _ = self.get_od_view_range()
            fields["size"] = max(4.0, round(abs(x1 - x0) * MARKER_SIZE_VIEW_FRACTION))
        if "color" not in fields:
            fields["color"] = random_marker_color(avoid=[m["color"] for m in self.get_markers()])
        self.set_markers(self.get_markers() + [{"x": pos[0], "y": pos[1], **fields}])
        self._emit_markers()

    def update_marker(self, index: int, **fields):
        """Change one marker's shape / size / color / label / position / hidden."""
        markers = self.get_markers()
        if not 0 <= index < len(markers):
            return
        markers[index].update(fields)
        self.set_markers(markers)       # items are cheap: rebuilt rather than mutated
        self._emit_markers()

    def delete_marker(self, index: int):
        markers = self.get_markers()
        if 0 <= index < len(markers):
            del markers[index]
            self.set_markers(markers)
            self._emit_markers()

    def delete_hovered_marker(self):
        """Delete / Backspace: delete the marker under the cursor, no questions asked."""
        for index, item in enumerate(self._marker_items):
            if item.is_hovered() and not item.moving:
                self.delete_marker(index)
                return

    def clear_markers(self):
        if self._marker_items:
            self.set_markers([])
            self._emit_markers()

    def highlight_marker(self, index):
        """Light one marker up (from the markers panel); None for none."""
        for i, item in enumerate(self._marker_items):
            item.set_highlighted(i == index)

    def _emit_markers(self):
        self.markers_updated.emit()
        self.markers_changed.emit(self._camera_key, self.get_markers())

    def _open_marker_dialog(self, item, screen_pos=None):
        if item not in self._marker_items:
            return
        dialog = MarkerDialog(self, self._marker_items.index(item))
        if screen_pos is not None:
            point = screen_pos.toPoint() if hasattr(screen_pos, "toPoint") else screen_pos
            dialog.move(point.x() + 12, point.y() + 12)
        dialog.exec()

    def open_marker_panel(self):
        if self._marker_panel is None:
            self._marker_panel = MarkerPanel(self)
        self._marker_panel.refresh()
        self._marker_panel.show()
        self._marker_panel.raise_()

    def _on_image_clicked(self, view_box, ev):
        """Remember where the context menu was opened (for "Add marker here"), and
        add a marker where the image is double-clicked, unless that is on a marker
        already or off the image (the colour bar, the axes)."""
        if ev.button() == Qt.MouseButton.RightButton:
            point = view_box.mapSceneToView(ev.scenePos())
            self._last_right_click = (point.x(), point.y())
        elif ev.button() == Qt.MouseButton.LeftButton and ev.double():
            if not view_box.sceneBoundingRect().contains(ev.scenePos()):
                return
            if any(item.is_hovered() for item in self._marker_items):
                return
            point = view_box.mapSceneToView(ev.scenePos())
            self.add_marker((point.x(), point.y()))

    def _extend_image_menu(self, view_box, scene):
        """An image's right-click menu is markers and the colormap, nothing else. The
        raw frames and the OD image share pixel coordinates and one colormap, so the
        menu is the same on all four.

        pyqtgraph's own entries are taken out: the ViewBox's (View All, X axis, Y
        axis, Mouse Mode), and what it adds from the items above it (the plot's Plot
        Options, the scene's Export...), by popping the menu up without asking them."""
        scene.sigMouseClicked.connect(lambda ev, vb=view_box: self._on_image_clicked(vb, ev))
        menu = view_box.menu
        if menu is None:
            return
        # hidden, not removed: removing the X / Y axis entries lets Qt delete the combo
        # boxes inside them, which pyqtgraph keeps updating (ViewBox.updateViewLists)
        for action in menu.actions():
            action.setVisible(False)
        view_box.raiseContextMenu = lambda ev, m=menu: m.popup(ev.screenPos().toPoint())
        add_action = QAction("Add marker here", menu)
        add_action.triggered.connect(lambda: self.add_marker(self._last_right_click))
        panel_action = QAction("Markers…", menu)
        panel_action.triggered.connect(self.open_marker_panel)
        cmap_menu = QMenu("Colormap", menu)
        for name in COLORMAPS:
            action = cmap_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == self._cmap_name)
            action.triggered.connect(lambda _checked, n=name: self.set_all_colormaps(n))
            self._cmap_actions.append(action)
        menu.addAction(add_action)
        menu.addAction(panel_action)
        menu.addMenu(cmap_menu)

    def eventFilter(self, watched, event):
        if watched is self.od_plot and event.type() == QEvent.Type.Resize:
            self._place_overlays()
        return super().eventFilter(watched, event)

    def _on_mouse_moved(self, event):
        pos = event[0]
        vb = self.od_plot.getViewBox()
        od = self._displayed_od
        if od is None or not vb.sceneBoundingRect().contains(pos):
            self.cursor_label.show_text("")
            return
        point = vb.mapSceneToView(pos)
        col, row = int(np.floor(point.x())), int(np.floor(point.y()))
        if not (0 <= row < od.shape[0] and 0 <= col < od.shape[1]):
            self.cursor_label.show_text("")
            self._last_cursor = None
            return
        self._last_cursor = (point.x(), point.y())
        unit, scale = self._length_unit()
        where = f"{col * scale:.0f}, {row * scale:.0f} {unit}"
        self.cursor_label.show_text(" &nbsp;&nbsp; ".join((
            _label_value("x, y", where), _label_value("OD", f"{od[row, col]:.2f}"))))
        self._place_overlays()

    # ------------------------------------------------------------------
    # Shot history
    # ------------------------------------------------------------------

    def _remember(self, entry):
        nbytes = sum(getattr(a, 'nbytes', 0) for a in entry[0])
        maxlen = int(np.clip(HISTORY_MAX_BYTES // max(1, nbytes), 5, HISTORY_MAX_SHOTS))
        if maxlen != self._history.maxlen:
            self._history = deque(self._history, maxlen=maxlen)
        if self._history_pos is not None and len(self._history) == self._history.maxlen:
            # the oldest shot is about to fall off: keep pointing at the same one
            self._history_pos = max(0, self._history_pos - 1)
        self._history.append(entry)

    def _clear_history(self):
        self._history.clear()
        self._shown_entry = None
        self._history_pos = None
        self._frames_seen = 0
        if self.pause_button.isChecked():
            self.pause_button.blockSignals(True)
            self.pause_button.setChecked(False)
            self.pause_button.blockSignals(False)
        self._update_history_label()

    def _redisplay(self):
        """Show the history entry we are on (the latest when live), averaged over
        the last N shots if asked."""
        if not self._history:
            return
        pos = len(self._history) - 1 if self._history_pos is None else self._history_pos
        entry = self._history[pos]
        n_avg = self.avg_spinner.value()
        od = None
        if n_avg > 1:
            ods = [self._history[i][0][3] for i in range(max(0, pos - n_avg + 1), pos + 1)
                   if self._history[i][0][3].shape == entry[0][3].shape]
            od = np.mean(ods, axis=0)
        self._display(entry, od=od)
        self._update_history_label()

    def _display_latest(self, entry):
        self._redisplay() if self.avg_spinner.value() > 1 else self._display(entry)

    def _on_pause_toggled(self, paused):
        if paused:
            if self._history_pos is None and self._history:
                self._history_pos = len(self._history) - 1
        else:
            self._history_pos = None
            self._redisplay()
        self._update_history_label()

    def _step_history(self, step):
        if not self._history:
            return
        pos = len(self._history) - 1 if self._history_pos is None else self._history_pos
        self._history_pos = int(np.clip(pos + step, 0, len(self._history) - 1))
        if not self.pause_button.isChecked():
            self.pause_button.blockSignals(True)
            self.pause_button.setChecked(True)
            self.pause_button.blockSignals(False)
        self._redisplay()

    def show_previous_shot(self):
        self._step_history(-1)

    def show_next_shot(self):
        self._step_history(+1)

    def _update_history_label(self):
        # short, in a fixed-width label; the tooltip spells it out
        if not self._history:
            text, tip = "–", "no shots yet"
        elif self._history_pos is None:
            text, tip = f"{self._frames_seen}", f"shot {self._frames_seen} · live"
        else:
            shown = self._frames_seen - (len(self._history) - 1 - self._history_pos)
            text = f"{shown}/{self._frames_seen}"
            tip = f"shot {shown} of {self._frames_seen} · paused (the display is held)"
        self.history_label.setText(text)
        self.history_label.setToolTip(tip)
        # The pause button is also the way back: while an older shot is held (by
        # pausing, or by stepping with the arrows) it turns into a lit "Live" button.
        held = self._history_pos is not None
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(held)
        self.pause_button.blockSignals(False)
        self.pause_button.setText("Live" if held else "❚❚")
        self.pause_button.setToolTip(
            "Not live: the display is held on an older shot. Click (or Space) to go back to live."
            if held else
            "Hold the display while the run carries on; new shots are still kept (Space)")
        self._restyle()

    # ------------------------------------------------------------------
    # ROI rectangle (drawn on OD image, persisted per camera)
    # ------------------------------------------------------------------

    _STATE_DIR = os.path.join(os.path.expanduser("~"), ".waxx")

    def _rect_state_path(self, camera_key: str) -> str:
        return os.path.join(self._STATE_DIR, f"live_od_{camera_key}_rect.json")

    def set_camera_key(self, camera_key: str):
        """Called on each new run. Loads the saved ROI rect and OD levels for *camera_key*."""
        if camera_key != self._camera_key:
            # another camera's pins mean nothing on this image; the window hands
            # over this camera's own (set_markers) from the server's store
            self.set_markers([])
        self._camera_key = camera_key
        # Remove any existing ROI first so we start fresh.
        self._remove_roi_item()
        if camera_key:
            self._load_rect()
            self._load_od_levels()
        self._sync_roi_controls()

    def get_od_roi_rect(self):
        """Return (x1, y1, x2, y2) in OD pixel coords if a drawn ROI is active, else None."""
        if self._roi_item is None:
            return None
        try:
            pos = self._roi_item.pos()
            size = self._roi_item.size()
            x1 = pos.x()
            y1 = pos.y()
            x2 = x1 + size.x()
            y2 = y1 + size.y()
            # Ensure correct orientation (handles may be dragged to negative size)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def _remove_roi_item(self):
        if self._roi_item is not None:
            try:
                self.od_plot.removeItem(self._roi_item)
            except Exception:
                pass
            self._roi_item = None

    def _show_roi(self, x1: float, y1: float, x2: float, y2: float):
        """Create (or replace) the RectROI overlay on the OD plot."""
        self._remove_roi_item()
        self._roi_item = _PassRightClickRectROI(
            [x1, y1], [x2 - x1, y2 - y1],
            pen=pg.mkPen('r', width=2),
            handlePen=pg.mkPen('r', width=2),
        )
        self._roi_item.setZValue(10)
        # one handle per corner, each scaling about the opposite corner
        self._roi_item.addScaleHandle([1, 1], [0, 0])
        self._roi_item.addScaleHandle([0, 0], [1, 1])
        self._roi_item.addScaleHandle([1, 0], [0, 1])
        self._roi_item.addScaleHandle([0, 1], [1, 0])
        self.od_plot.addItem(self._roi_item, ignoreBounds=True)
        self._roi_item.sigRegionChanged.connect(self._on_roi_moving)
        self._roi_item.sigRegionChangeFinished.connect(self._on_roi_changed)

    def _on_roi_toggled(self, on):
        if on:
            if self._roi_item is None:
                rect = self._read_rect_file().get("rect")
                if rect is None:
                    shape = self._last_od_shape
                    if shape is not None:
                        h, w = shape
                        rect = [w // 4, h // 4, w - w // 4, h - h // 4]
                    else:
                        rect = [50.0, 50.0, 250.0, 250.0]
                self._show_roi(*(float(v) for v in rect))
            self._save_rect()
        else:
            # keep the rectangle on file, switched off, so it comes back as it was
            rect = self.get_od_roi_rect()
            self._remove_roi_item()
            self._save_rect(rect=rect, enabled=False)
        self._sync_roi_controls()
        self._update_marginals()

    def auto_roi(self, n_shots=None):
        """Set the ROI around the atoms in the last ``n_shots`` shots (default: the
        View menu's spinbox), counting back from the one on screen. Uses waxa's
        auto-ROI detector on the raw atoms and light frames, as the analysis does.
        Returns the (x1, y1, x2, y2) set, or None if nothing was found."""
        if not self._history:
            logger.warning("Auto ROI: no shots yet")
            return None
        if n_shots is None:
            n_shots = self.auto_roi_spinner.value()
        pos = len(self._history) - 1 if self._history_pos is None else self._history_pos
        shape = np.shape(self._history[pos][0][0])
        entries = [self._history[i][0] for i in range(max(0, pos - int(n_shots) + 1), pos + 1)
                   if np.shape(self._history[i][0][0]) == shape]
        from waxa.image_processing.auto_roi import suggest_roi
        try:
            result = suggest_roi(atoms=np.stack([e[0] for e in entries]),
                                 light=np.stack([e[1] for e in entries]))
        except Exception as exc:
            logger.warning(f"Auto ROI failed: {exc}")
            return None
        if not result.valid:
            logger.warning(f"Auto ROI: no ROI set over {len(entries)} shot(s): {result.reason}")
            return None
        rect = (float(result.roix[0]), float(result.roiy[0]),
                float(result.roix[1]), float(result.roiy[1]))
        self._show_roi(*rect)
        self._save_rect()
        self._sync_roi_controls()
        self._update_marginals()
        logger.info(f"Auto ROI over {len(entries)} shot(s): x {rect[0]:.0f}–{rect[2]:.0f}, "
                    f"y {rect[1]:.0f}–{rect[3]:.0f} px")
        return rect

    def _on_roi_moving(self):
        self._sync_roi_controls()
        self._update_marginals()

    def _on_roi_changed(self):
        """Save the ROI to file whenever the user finishes dragging/resizing."""
        self._save_rect()

    def _sync_roi_controls(self):
        self.roi_button.blockSignals(True)
        self.roi_button.setChecked(self._roi_item is not None)
        self.roi_button.blockSignals(False)
        tip = ("Integration region for atom number, profiles and fits. Off: the visible part "
               "of the image is used. Remembered per camera.")
        rect = self.get_od_roi_rect()
        if rect is not None:
            x1, y1, x2, y2 = rect
            unit, scale = self._length_unit()
            tip += (f"\nNow: {(x2 - x1) * scale:.0f}×{(y2 - y1) * scale:.0f} {unit} "
                    f"at ({x1 * scale:.0f}, {y1 * scale:.0f})")
        self.roi_button.setToolTip(tip)

    def _read_rect_file(self) -> dict:
        if not self._camera_key:
            return {}
        path = self._rect_state_path(self._camera_key)
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                return dict(json.load(f))
        except Exception as exc:
            logger.warning(f"Could not load ROI rect: {exc}")
            return {}

    def _save_rect(self, rect=None, enabled=True):
        if not self._camera_key:
            return
        if rect is None:
            rect = self.get_od_roi_rect()
        if rect is None:
            return
        try:
            os.makedirs(self._STATE_DIR, exist_ok=True)
            with open(self._rect_state_path(self._camera_key), 'w') as f:
                json.dump({"rect": [float(v) for v in rect], "enabled": bool(enabled)}, f)
        except Exception as exc:
            logger.warning(f"Could not save ROI rect: {exc}")

    def _load_rect(self):
        data = self._read_rect_file()
        # files written before the ROI could be switched off have no "enabled"
        if data.get("rect") is not None and data.get("enabled", True):
            x1, y1, x2, y2 = data["rect"]
            self._show_roi(float(x1), float(y1), float(x2), float(y2))

    # ------------------------------------------------------------------
    # Remembered UI state (only if the window attaches a QSettings)
    # ------------------------------------------------------------------

    def attach_settings(self, settings):
        """Restore the layout and toggles from ``settings`` (a QSettings) and keep
        them up to date in it. Without this call nothing is stored."""
        self._settings = settings
        try:
            theme.apply_theme(str(settings.value("ui/dark_mode", "true")).lower() in ("true", "1"))
            colormap = settings.value("viewer/colormap")
            if colormap and str(colormap) != self._cmap_name:
                self.set_all_colormaps(str(colormap))
            limits = settings.value("viewer/od_limits")
            if limits is not None:
                self.set_od_limits(float(limits[0]), float(limits[1]))
            for key, splitter in (("viewer/main_splitter", self.main_splitter),
                                  ("viewer/top_splitter", self.top_splitter),
                                  ("viewer/raw_splitter", self._raw_images_widget)):
                state = settings.value(key)
                if state is not None:
                    splitter.restoreState(state)
            # a layout saved with the log dragged shut: open it again
            sizes = self.top_splitter.sizes()
            if sizes[0] < self.output_window.minimumSizeHint().height():
                self.top_splitter.setSizes([70, max(sizes[1] - 70, 1)])
            for key, checkbox in (("viewer/lock_views", self.lock_views_checkbox),
                                  ("viewer/profiles", self.profiles_checkbox),
                                  ("viewer/um_axes", self.um_checkbox)):
                value = settings.value(key)
                if value is not None:
                    checkbox.setChecked(str(value).lower() in ("true", "1"))
            self.output_window.set_collapsed(
                str(settings.value("viewer/log_collapsed", "false")).lower() in ("true", "1"))
            value = settings.value("viewer/auto_roi_shots")
            if value is not None:
                self.auto_roi_spinner.setValue(int(value))
        except Exception as exc:
            logger.warning(f"Could not restore the viewer layout: {exc}")

    def save_settings(self):
        """Call from the window's closeEvent."""
        settings = self._settings
        if settings is None:
            return
        settings.setValue("viewer/main_splitter", self.main_splitter.saveState())
        settings.setValue("viewer/top_splitter", self.top_splitter.saveState())
        settings.setValue("viewer/raw_splitter", self._raw_images_widget.saveState())
        settings.setValue("viewer/lock_views", self.lock_views_checkbox.isChecked())
        settings.setValue("viewer/profiles", self.profiles_checkbox.isChecked())
        settings.setValue("viewer/um_axes", self.um_checkbox.isChecked())
        settings.setValue("viewer/auto_roi_shots", self.auto_roi_spinner.value())
        settings.setValue("viewer/log_collapsed", self.output_window.is_collapsed())

    def open_settings_dialog(self):
        """OD scale limits and the theme. Applied on OK; stored if the window
        attached a QSettings."""
        dialog = QDialog(self)
        dialog.setWindowTitle("LiveOD settings")
        lo_spin, hi_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, value in ((lo_spin, self._od_limits[0]), (hi_spin, self._od_limits[1])):
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setValue(value)
        dark_box = QCheckBox("Dark mode")
        dark_box.setChecked(theme.is_dark())
        form = QFormLayout()
        form.addRow("Lowest OD the colour scale can reach", lo_spin)
        form.addRow("Highest OD the colour scale can reach", hi_spin)
        form.addRow(dark_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.setLayout(form)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.apply_settings(od_limits=(lo_spin.value(), hi_spin.value()), dark=dark_box.isChecked())

    def apply_settings(self, od_limits=None, dark=None):
        if od_limits is not None:
            self.set_od_limits(*od_limits)
        if dark is not None and bool(dark) != theme.is_dark():
            theme.apply_theme(dark)
        if dark is not None and self._settings is not None:
            self._settings.setValue("ui/dark_mode", bool(dark))

    def _restyle(self, *_):
        """Colours this widget sets itself, for the current theme."""
        held = self._history_pos is not None
        self.history_label.setStyleSheet(
            f"color: {theme.color('warning')}; font-weight: bold;" if held else "")
        self.pause_button.setStyleSheet(
            "background-color: #fb8c00; color: black; font-weight: bold;" if held else "")

    def _save_od_levels(self):
        if self._settings is not None and self._camera_key:
            self._settings.setValue(f"od_levels/{self._camera_key}", [self._od_min, self._od_max])

    def _load_od_levels(self):
        if self._settings is None:
            return
        levels = self._settings.value(f"od_levels/{self._camera_key}")
        try:
            if levels is not None:
                self.set_od_levels(float(levels[0]), float(levels[1]))
        except (TypeError, ValueError, IndexError):
            pass
