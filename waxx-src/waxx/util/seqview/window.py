"""The viewer window.

Layout: toolbar / overview strip / lane stack (label column + x-linked
plots) with the time axis under the last lane / status bar; docks for the
code pane (right), the pulse table and warnings (bottom), lane visibility
(left, hidden by default).

Interaction (all navigation is plain mouse; measuring takes a modifier):
    wheel               zoom about the cursor        Shift+wheel: pan
    right-click         zoom out about the cursor    Ctrl+wheel: lane height
    left-drag           pan                          left-click: select
    right-drag          box zoom                     double-click: zoom to pulse
                                                     (empty space: fit shot)
    Shift+left-drag     measure a span (cursors A/B snap to edges; Alt: no snap)
    A / B               drop cursor A / B at the mouse
    Home / F            fit current shot / fit all   Z: zoom to selection
    + / -  ← / →        zoom / pan (Shift: faster)   [ ]: previous / next edge
    , / .  0-9          previous / next shot, jump to shot
    P                   show / hide the physical port lanes
    G                   toggle gap dimensions        L: toggle labels
    Esc                 clear selection and cursors  Ctrl+C: copy readout
    Ctrl+E              export PNG                   Ctrl+Shift+E: export CSV
"""

import html
import json
import os

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import (QAction, QColor, QKeySequence, QCursor, QPen, QBrush,
                         QGuiApplication)
from PyQt6.QtNetwork import QTcpServer, QHostAddress
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QToolBar, QComboBox, QDockWidget,
                             QTableView, QListWidget, QListWidgetItem,
                             QStatusBar, QToolTip, QFileDialog, QCheckBox,
                             QLineEdit, QPushButton, QSizePolicy, QMessageBox)
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel

from waxx.util.seqview.bundle import Bundle
from waxx.util.seqview.items import (PulseBarItem, DigitalTraceItem,
                                     AnalogTraceItem, EventMarkerItem, BandItem,
                                     GapDimensionItem, CursorLine, GridItem,
                                     qcolor)
from waxx.util.seqview.timefmt import TimeAxis, fmt_duration, fmt_time, unit_for
from waxx.util.seqview.code_pane import CodePane
from waxx.util.seqview import launch as _launch

LABEL_COL_W = 170
BG = '#1b1d21'
CURSOR_COLORS = {'A': '#ffd166', 'B': '#7ae582'}


# ---------------------------------------------------------------------------
# view box with the interaction model
# ---------------------------------------------------------------------------

class LaneViewBox(pg.ViewBox):
    sigClicked = pyqtSignal(object, object)        # (viewbox, scene pos)
    sigDoubleClicked = pyqtSignal(object, object)
    sigRightClicked = pyqtSignal(object, object)
    sigBoxZoom = pyqtSignal(float, float)
    sigMeasure = pyqtSignal(float, float)
    sigHeightWheel = pyqtSignal(object, int)       # (viewbox, delta)
    sigPanWheel = pyqtSignal(float)                # fraction of the view

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setMouseMode(self.PanMode)
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)
        self.enableAutoRange(x=False, y=False)
        self._rb = None

    def _rubber(self):
        if self._rb is None:
            self._rb = pg.LinearRegionItem(movable=False,
                                           brush=QBrush(QColor(255, 255, 255, 40)),
                                           pen=QPen(QColor(255, 255, 255, 160)))
            self._rb.setZValue(50)
        return self._rb

    def mouseDragEvent(self, ev, axis=None):
        mods = ev.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if ev.button() == Qt.MouseButton.RightButton or (
                ev.button() == Qt.MouseButton.LeftButton and shift):
            ev.accept()
            x0 = self.mapToView(ev.buttonDownPos()).x()
            x1 = self.mapToView(ev.pos()).x()
            rb = self._rubber()
            if rb.scene() is None:
                self.addItem(rb, ignoreBounds=True)
            rb.setRegion((min(x0, x1), max(x0, x1)))
            rb.show()
            if ev.isFinish():
                rb.hide()
                if abs(x1 - x0) > 0:
                    if ev.button() == Qt.MouseButton.RightButton:
                        self.sigBoxZoom.emit(min(x0, x1), max(x0, x1))
                    else:
                        self.sigMeasure.emit(x0, x1)
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            if ev.double():
                self.sigDoubleClicked.emit(self, ev.scenePos())
            else:
                self.sigClicked.emit(self, ev.scenePos())
            return
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self.sigRightClicked.emit(self, ev.scenePos())
            return
        super().mouseClickEvent(ev)

    def wheelEvent(self, ev, axis=None):
        mods = ev.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            ev.accept()
            self.sigHeightWheel.emit(self, 1 if ev.delta() > 0 else -1)
            return
        if mods & Qt.KeyboardModifier.ShiftModifier:
            ev.accept()
            self.sigPanWheel.emit(-0.15 if ev.delta() > 0 else 0.15)
            return
        super().wheelEvent(ev, axis=axis)


# ---------------------------------------------------------------------------
# one lane
# ---------------------------------------------------------------------------

class Lane:
    def __init__(self, meta, bundle, t_end):
        self.meta = meta
        self.id = meta['id']
        self.kind = meta['kind']
        self.group = meta['group']
        self.visible = bool(meta.get('visible', True))
        self.height = float(meta.get('height', 1.0))
        self.vb = LaneViewBox()
        self.plot = pg.PlotItem(viewBox=self.vb)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.hideButtons()
        self.plot.setContentsMargins(0, 0, 0, 0)
        self.vb.setLimits(xMin=-0.02 * t_end, xMax=1.02 * t_end, minXRange=8.)
        self.label = pg.LabelItem(justify='left')
        self.bars = None
        self.trace = None
        self.analog = None
        self.events = None
        self.gaps = None
        self.bands = None
        self.cursors = {}
        self._build(bundle, t_end)
        self.set_label()

    def set_label(self, extra=''):
        m = self.meta
        col = m.get('color', '#cccccc')
        sub = m.get('note', '')
        txt = (f"<span style='color:{col};font-size:9pt'><b>{html.escape(m['label'])}"
               f"</b></span>")
        if extra:
            txt += f"<br><span style='color:#bbbbbb;font-size:7pt'>{html.escape(extra)}</span>"
        elif sub and self.kind in ('digital', 'analog'):
            short = sub if len(sub) < 40 else sub[:38] + '…'
            txt += f"<br><span style='color:#8f8f8f;font-size:7pt'>{html.escape(short)}</span>"
        self.label.setText(txt)
        self.label.setToolTip(sub)

    def _build(self, bundle, t_end):
        m = self.meta
        pulses = [bundle.pulses[i] for i in m.get('pulses', [])]
        events = [bundle.events[i] for i in m.get('events', [])]
        self.bands = BandItem(bundle.meta['shots'], bundle.meta['framing'], t_end)
        self.vb.addItem(self.bands, ignoreBounds=True)
        if self.kind == 'digital':
            edges = bundle.array(m['edges']) if m.get('edges') else np.zeros(0)
            self.trace = DigitalTraceItem(edges, m.get('level0', 0), t_end,
                                          color=m.get('color', '#9a9a9a'),
                                          high_label=m.get('high_label', ''),
                                          low_label=m.get('low_label', ''))
            self.vb.addItem(self.trace, ignoreBounds=True)
            self.bars = PulseBarItem(pulses, t_end, style='marker', rows=False,
                                     labels=False, y0=0.72, y1=0.98)
            self.vb.addItem(self.bars, ignoreBounds=True)
            self.vb.setYRange(0., 1., padding=0)
        elif self.kind == 'analog':
            samples = bundle.array(m['samples']) if m.get('samples') else np.zeros(1)
            self.analog = AnalogTraceItem(samples, bundle.meta.get('dt_ns', 1.0),
                                          color=m.get('color', '#f0e442'))
            self.vb.addItem(self.analog, ignoreBounds=True)
            a = self.analog.pyr.amax or 1.
            self.vb.setYRange(-1.15 * a, 1.15 * a, padding=0)
            # bars (latch pulses) drawn as markers over the top edge
            self.bars = PulseBarItem(pulses, t_end, style='marker', rows=False,
                                     labels=False, y0=0.72, y1=0.98)
            # marker item works in the 0..1 frame: wrap in a transform
            self.bars.setTransform(pg.QtGui.QTransform().scale(1., 2.3 * a)
                                   .translate(0., -0.5))
            self.vb.addItem(self.bars, ignoreBounds=True)
        else:
            self.bars = PulseBarItem(pulses, t_end, style='bar')
            self.vb.addItem(self.bars, ignoreBounds=True)
            self.vb.setYRange(0., 1., padding=0)
            if self.id != 'steps':
                self.gaps = GapDimensionItem(self.bars, t_end, y=0.5)
                self.vb.addItem(self.gaps, ignoreBounds=True)
        if events:
            self.events = EventMarkerItem(events, t_end)
            self.vb.addItem(self.events, ignoreBounds=True)

    def y_frac_to_view(self, f):
        r = self.vb.viewRect()
        return r.top() + f * r.height()

    def value_text(self, t):
        if self.kind == 'digital' and self.trace is not None:
            lev = self.trace.level_at(t)
            lab = self.meta.get('high_label' if lev else 'low_label') or str(lev)
            return f"{lab}"
        if self.kind == 'analog' and self.analog is not None:
            v = self.analog.pyr.value_at(t)
            return f"{v:+.4f} V" if v is not None else ''
        return ''

    def edges(self):
        out = []
        if self.trace is not None:
            out.append(self.trace.edges)
        if self.bars is not None and len(self.bars.t0):
            out.append(self.bars.t0)
            out.append(self.bars.t1)
        if self.events is not None:
            out.append(self.events.t)
        if not out:
            return np.zeros(0)
        return np.unique(np.concatenate(out))


# ---------------------------------------------------------------------------
# pulse table model
# ---------------------------------------------------------------------------

class PulseTableModel(QAbstractTableModel):
    COLS = ('lane', 'what', 'start', 'end', 'length', 'shot', 'macro',
            'parameter', 'src line', 'QUA line', 'note')

    def __init__(self, pulses, lanes, origin_fn):
        super().__init__()
        self.pulses = pulses
        self.lane_label = {ln['id']: ln['label'] for ln in lanes}
        self.origin_fn = origin_fn

    def rowCount(self, parent=QModelIndex()):
        return len(self.pulses)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLS)

    def headerData(self, s, o, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and o == Qt.Orientation.Horizontal:
            return self.COLS[s]
        return None

    def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
        if not idx.isValid():
            return None
        p = self.pulses[idx.row()]
        c = idx.column()
        if role == Qt.ItemDataRole.DisplayRole:
            o = self.origin_fn()
            if c == 0:
                return self.lane_label.get(p['lane'], p['lane'])
            if c == 1:
                return p.get('text') or p.get('pulse_name') or ''
            if c == 2:
                return fmt_time(p['t0'] - o, 'µs')
            if c == 3:
                return fmt_time(p['t1'] - o, 'µs')
            if c == 4:
                return fmt_duration(p['t1'] - p['t0'])
            if c == 5:
                return '' if p.get('shot') is None else str(p['shot'])
            if c == 6:
                return p.get('macro', '')
            if c == 7:
                return p.get('value_str', '')
            if c == 8:
                return '' if p.get('src_line') is None else str(p['src_line'])
            if c == 9:
                return '' if p.get('qua_line') is None else str(p['qua_line'])
            if c == 10:
                return p.get('note', '')
        if role == Qt.ItemDataRole.UserRole:
            if c in (2, 3):
                return float(p['t0'] if c == 2 else p['t1'])
            if c == 4:
                return float(p['t1'] - p['t0'])
            if c == 5:
                return -1 if p.get('shot') is None else int(p['shot'])
            return self.data(idx, Qt.ItemDataRole.DisplayRole)
        if role == Qt.ItemDataRole.ForegroundRole:
            return QColor(p.get('color', '#dddddd'))
        return None


class _SortProxy(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.setSortRole(Qt.ItemDataRole.UserRole)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterKeyColumn(-1)


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------

class SeqViewWindow(QMainWindow):

    def __init__(self, bundle=None, serve=False):
        super().__init__()
        self.setWindowTitle('seqview')
        self.resize(1500, 900)
        self.bundle = None
        self.lanes = []
        self.lane_by_id = {}
        self.pulse_by_id = {}
        self.t_end = 1.0
        self.selected = None          # pulse id
        self.selected_event = None
        self.cursor_x = {'A': None, 'B': None}
        self.origin_mode = 'shot'
        self.origin_t = 0.0
        self.show_physical = False
        self.show_gaps = True
        self.show_labels = True
        self._hover_last = None
        self._server = None
        self._pending_paths = []
        self._build_ui()
        if serve:
            self._start_server()
        if bundle is not None:
            self.load_bundle(bundle, keep_view=False)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        pg.setConfigOptions(antialias=False, useOpenGL=False, background=BG,
                            foreground='#d0d0d0')
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # toolbar
        tb = QToolBar('view')
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_fit_shot = QAction('Fit shot', self, triggered=self.fit_shot)
        self.act_fit_all = QAction('Fit all', self, triggered=self.fit_all)
        tb.addAction(self.act_fit_shot)
        tb.addAction(self.act_fit_all)
        tb.addAction(QAction('－', self, triggered=lambda: self.zoom_by(2.0)))
        tb.addAction(QAction('＋', self, triggered=lambda: self.zoom_by(0.5)))
        tb.addSeparator()
        tb.addWidget(QLabel(' shot '))
        self.shot_combo = QComboBox()
        self.shot_combo.currentIndexChanged.connect(self._shot_combo_changed)
        tb.addWidget(self.shot_combo)
        tb.addSeparator()
        tb.addWidget(QLabel(' origin '))
        self.origin_combo = QComboBox()
        self.origin_combo.addItems(['shot start', 'absolute', 'cursor A'])
        self.origin_combo.currentTextChanged.connect(self._origin_changed)
        tb.addWidget(self.origin_combo)
        tb.addSeparator()
        self.chk_physical = QCheckBox('physical ports (P)')
        self.chk_physical.toggled.connect(self.set_physical_visible)
        tb.addWidget(self.chk_physical)
        self.chk_gaps = QCheckBox('gaps (G)')
        self.chk_gaps.setChecked(True)
        self.chk_gaps.toggled.connect(self.set_gaps_visible)
        tb.addWidget(self.chk_gaps)
        self.chk_labels = QCheckBox('labels (L)')
        self.chk_labels.setChecked(True)
        self.chk_labels.toggled.connect(self.set_labels_visible)
        tb.addWidget(self.chk_labels)
        tb.addSeparator()
        tb.addAction(QAction('Export PNG', self, triggered=self.export_png))
        tb.addAction(QAction('Export CSV', self, triggered=self.export_csv))
        tb.addAction(QAction('Reload', self, triggered=self.reload_file))
        tb.addAction(QAction('?', self, triggered=self.show_help))
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        self.title_label = QLabel('')
        self.title_label.setStyleSheet('color:#dddddd; padding-right:8px;')
        tb.addWidget(self.title_label)

        # overview strip
        self.overview = pg.PlotWidget()
        self.overview.setFixedHeight(46)
        self.overview.hideAxis('left')
        self.overview.hideAxis('bottom')
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.setMenuEnabled(False)
        self.overview.hideButtons()
        self.overview.plotItem.setContentsMargins(0, 0, 0, 0)
        self.region = pg.LinearRegionItem(brush=QBrush(QColor(255, 255, 255, 35)),
                                          pen=QPen(QColor('#ffffff')))
        self.region.setZValue(30)
        self.region.sigRegionChanged.connect(self._region_moved)
        self._sync = False
        ovl = QHBoxLayout()
        ovl.setContentsMargins(0, 0, 0, 0)
        ovl.setSpacing(0)
        self.ov_label = QLabel('  overview')
        self.ov_label.setFixedWidth(LABEL_COL_W)
        self.ov_label.setStyleSheet('color:#8f8f8f; font-size:8pt;')
        ovl.addWidget(self.ov_label)
        ovl.addWidget(self.overview)
        lay.addLayout(ovl)
        self.overview.scene().sigMouseClicked.connect(self._overview_clicked)

        # lane stack
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.layout.setSpacing(0)
        self.glw.ci.setContentsMargins(0, 0, 0, 0)
        self.glw.ci.layout.setColumnFixedWidth(0, LABEL_COL_W)
        lay.addWidget(self.glw, 1)
        self.setCentralWidget(central)
        self.glw.scene().sigMouseMoved.connect(self._mouse_moved)

        # status bar
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.hover_label = QLabel('')
        self.cursor_label = QLabel('')
        self.cursor_label.setStyleSheet('color:#ffd166;')
        self.warn_button = QPushButton('')
        self.warn_button.setFlat(True)
        self.warn_button.clicked.connect(lambda: self.warn_dock.setVisible(
            not self.warn_dock.isVisible()))
        sb.addWidget(self.hover_label, 1)
        sb.addPermanentWidget(self.cursor_label)
        sb.addPermanentWidget(self.warn_button)

        # docks
        self.code = CodePane()
        self.code.lineClicked.connect(self._code_line_clicked)
        self.code.lineDoubleClicked.connect(self._code_line_double_clicked)
        self.code.shotClicked.connect(self.zoom_to_shot)
        self.code_dock = QDockWidget('code', self)
        self.code_dock.setWidget(self.code)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.code_dock)
        self.code_dock.setMinimumWidth(380)

        tw = QWidget()
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(2, 2, 2, 2)
        self.table_filter = QLineEdit()
        self.table_filter.setPlaceholderText('filter pulses…')
        tl.addWidget(self.table_filter)
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        tl.addWidget(self.table)
        self.table_dock = QDockWidget('pulses', self)
        self.table_dock.setWidget(tw)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.table_dock)
        self.table_dock.hide()

        self.warn_list = QListWidget()
        self.warn_dock = QDockWidget('diagnostics', self)
        self.warn_dock.setWidget(self.warn_list)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.warn_dock)
        self.warn_dock.hide()

        self.lane_list = QListWidget()
        self.lane_list.itemChanged.connect(self._lane_item_changed)
        self.lane_dock = QDockWidget('lanes', self)
        self.lane_dock.setWidget(self.lane_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.lane_dock)
        self.lane_dock.hide()

        view_menu = self.menuBar().addMenu('View')
        for d in (self.code_dock, self.table_dock, self.warn_dock, self.lane_dock):
            view_menu.addAction(d.toggleViewAction())
        help_menu = self.menuBar().addMenu('Help')
        help_menu.addAction(QAction('Keys and mouse', self, triggered=self.show_help))

        # hover throttle
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(16)
        self._hover_timer.timeout.connect(self._do_hover)
        self._hover_pos = None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load_bundle(self, bundle, keep_view=True, path=None):
        bundle = Bundle.coerce(bundle)
        state = self._capture_state() if (keep_view and self.bundle) else None
        self._clear_stack()
        self.bundle = bundle
        self.bundle_path = path
        m = bundle.meta
        self.t_end = float(m.get('t_end_ns') or 1.0)
        self.pulse_by_id = {p['id']: p for p in m['pulses']}
        self.event_by_id = {e['id']: e for e in m['events']}
        self.setWindowTitle(f"seqview — {m.get('title', '')}")
        self.title_label.setText(m.get('title', ''))
        self.title_label.setToolTip(m.get('subtitle', ''))

        # header lane: shots
        self.header_vb = LaneViewBox()
        self.header_plot = pg.PlotItem(viewBox=self.header_vb)
        self.header_plot.hideAxis('left')
        self.header_plot.hideAxis('bottom')
        self.header_plot.hideButtons()
        self.header_vb.setLimits(xMin=-0.02 * self.t_end, xMax=1.02 * self.t_end,
                                 minXRange=8.)
        self.header_vb.setYRange(0., 1., padding=0)
        self.axis = TimeAxis(orientation='bottom')
        self.header_vb.addItem(GridItem(self._tick_levels, self.t_end, header=True),
                               ignoreBounds=True)
        self.header_bands = BandItem(m['shots'], m['framing'], self.t_end, labels=True)
        self.header_vb.addItem(self.header_bands, ignoreBounds=True)
        hl = pg.LabelItem("<span style='color:#bbbbbb;font-size:8pt'><b>shots</b></span>",
                          justify='left')
        self.glw.ci.addItem(hl, row=0, col=0)
        self.glw.ci.addItem(self.header_plot, row=0, col=1)
        self.glw.ci.layout.setRowFixedHeight(0, 22)
        self._wire_vb(self.header_vb)
        self.master = self.header_vb

        self.lanes = []
        self.lane_by_id = {}
        for i, ln in enumerate(m['lanes']):
            lane = Lane(ln, bundle, self.t_end)
            self.lanes.append(lane)
            self.lane_by_id[lane.id] = lane
            lane.vb.setXLink(self.master)
            self._wire_vb(lane.vb)
            lane.grid = GridItem(self._tick_levels, self.t_end, odd=bool(i % 2))
            lane.vb.addItem(lane.grid, ignoreBounds=True)
            if lane.bars is not None:
                lane.bars.labels_enabled = self.show_labels and lane.kind not in (
                    'digital', 'analog')
            if lane.gaps is not None:
                lane.gaps.enabled = self.show_gaps
        self._layout_lanes()
        self.master.sigXRangeChanged.connect(self._x_range_changed)

        # edges for snapping
        self.all_edges = np.unique(np.concatenate(
            [ln.edges() for ln in self.lanes] + [np.array([0., self.t_end])]))

        self._build_overview()
        self._build_shot_combo()
        self._build_table()
        self._build_warnings()
        self._build_lane_list()
        self.code.load(m)
        self._build_chips()
        self._cursor_lines = {}

        if state is not None:
            self._restore_state(state)
        else:
            self.show_physical = False
            self.chk_physical.setChecked(False)
            self.set_physical_visible(False)
            self.fit_shot(0)
        self._x_range_changed()

    def _clear_stack(self):
        try:
            self.master.sigXRangeChanged.disconnect(self._x_range_changed)
        except Exception:
            pass
        self.glw.ci.clear()
        self.lanes = []
        self.selected = None
        self.selected_event = None
        self.cursor_x = {'A': None, 'B': None}
        self._cursor_lines = {}
        self.overview.clear()

    def _wire_vb(self, vb):
        vb.sigClicked.connect(self._vb_clicked)
        vb.sigDoubleClicked.connect(self._vb_double_clicked)
        vb.sigRightClicked.connect(self._vb_right_clicked)
        vb.sigBoxZoom.connect(self.set_x_range)
        vb.sigMeasure.connect(self._measured)
        vb.sigHeightWheel.connect(self._height_wheel)
        vb.sigPanWheel.connect(self.pan_by)

    def _tick_levels(self, l, r, px):
        """Tick positions of the time axis for the grid lines."""
        return [(sp, vals) for sp, vals in self.axis.tickValues(l, r, px)]

    def _layout_lanes(self):
        """(Re)place the visible lanes in the layout, time axis under the
        last one."""
        ci = self.glw.ci
        # remove all lane rows (keep header at row 0)
        for lane in self.lanes:
            for it in (lane.label, lane.plot):
                if it.scene() is ci.scene() and it in ci.items:
                    ci.removeItem(it)
        if self.axis.scene() is ci.scene():
            try:
                ci.removeItem(self.axis)
            except Exception:
                pass
        row = 1
        vis = [ln for ln in self.lanes if self._lane_shown(ln)]
        for lane in vis:
            ci.addItem(lane.label, row=row, col=0)
            ci.addItem(lane.plot, row=row, col=1)
            ci.layout.setRowStretchFactor(row, max(int(lane.height * 100), 10))
            ci.layout.setRowMinimumHeight(row, 26)
            row += 1
        # time axis in its own row, linked to the master
        self.axis.linkToView(self.master)
        ci.addItem(self.axis, row=row, col=1)
        ci.layout.setRowFixedHeight(row, 34)
        ci.layout.setRowStretchFactor(row, 0)
        for lane in self.lanes:
            shown = lane in vis
            lane.plot.setVisible(shown)
            lane.label.setVisible(shown)

    def _lane_shown(self, lane):
        if not lane.visible:
            return False
        if lane.group == 'physical' and not self.show_physical:
            return False
        return True

    # ------------------------------------------------------------------
    # overview / shots / table / warnings / lane list / chips
    # ------------------------------------------------------------------
    def _build_overview(self):
        ov = self.overview.plotItem
        ov.vb.setLimits(xMin=0, xMax=self.t_end)
        ov.setXRange(0., self.t_end, padding=0)
        ov.setYRange(0., 1., padding=0)
        m = self.bundle.meta
        bands = BandItem(m['shots'], m['framing'], self.t_end, labels=False)
        ov.addItem(bands, ignoreBounds=True)
        sem = [ln for ln in self.lanes if ln.group == 'semantic' and ln.id != 'steps']
        n = max(len(sem), 1)
        for i, lane in enumerate(sem):
            pulses = [self.bundle.pulses[k] for k in lane.meta.get('pulses', [])]
            if not pulses:
                continue
            y1 = 1. - i / n
            y0 = 1. - (i + 1) / n
            bars = PulseBarItem(pulses, self.t_end, style='bar', rows=False,
                                labels=False, y0=y0 + 0.02, y1=y1 - 0.02)
            ov.addItem(bars, ignoreBounds=True)
        self.region.setBounds((0., self.t_end))
        ov.addItem(self.region, ignoreBounds=True)

    def _build_shot_combo(self):
        self.shot_combo.blockSignals(True)
        self.shot_combo.clear()
        for s in self.bundle.meta['shots']:
            self.shot_combo.addItem(s.get('label', f"shot {s['index']}"), s['index'])
        self.shot_combo.blockSignals(False)

    def _build_table(self):
        self.table_model = PulseTableModel(self.bundle.pulses, self.bundle.lanes,
                                           lambda: self.origin_t)
        self.table_proxy = _SortProxy()
        self.table_proxy.setSourceModel(self.table_model)
        self.table.setModel(self.table_proxy)
        self.table.selectionModel().currentRowChanged.connect(self._table_row)
        self.table.doubleClicked.connect(self._table_double)
        try:
            self.table_filter.textChanged.disconnect()
        except Exception:
            pass
        self.table_filter.textChanged.connect(self.table_proxy.setFilterFixedString)
        self.table.resizeColumnsToContents()

    def _build_warnings(self):
        self.warn_list.clear()
        warns = self.bundle.meta.get('warnings', [])
        n_err = sum(1 for w in warns if w.get('level') == 'error')
        n_warn = sum(1 for w in warns if w.get('level') == 'warning')
        for w in warns:
            it = QListWidgetItem(f"[{w.get('level', 'info')}] {w.get('text', '')}")
            col = {'error': '#ff6b6b', 'warning': '#ffb74d'}.get(w.get('level'), '#a0a0a0')
            it.setForeground(QColor(col))
            self.warn_list.addItem(it)
        if not warns:
            self.warn_button.setText('no warnings')
            self.warn_button.setStyleSheet('color:#8f8f8f;')
        else:
            parts = []
            if n_err:
                parts.append(f"{n_err} error{'s' if n_err > 1 else ''}")
            if n_warn:
                parts.append(f"{n_warn} warning{'s' if n_warn > 1 else ''}")
            if not parts:
                parts.append(f"{len(warns)} note{'s' if len(warns) > 1 else ''}")
            self.warn_button.setText(', '.join(parts))
            self.warn_button.setStyleSheet(
                'color:#ff6b6b;' if n_err else 'color:#ffb74d;')
        if n_err:
            self.warn_dock.show()

    def _build_lane_list(self):
        self.lane_list.blockSignals(True)
        self.lane_list.clear()
        for lane in self.lanes:
            it = QListWidgetItem(lane.meta['label'])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if lane.visible
                             else Qt.CheckState.Unchecked)
            it.setForeground(QColor(lane.meta.get('color', '#dddddd')))
            it.setData(Qt.ItemDataRole.UserRole, lane.id)
            self.lane_list.addItem(it)
        self.lane_list.blockSignals(False)

    def _lane_item_changed(self, item):
        lid = item.data(Qt.ItemDataRole.UserRole)
        lane = self.lane_by_id.get(lid)
        if lane is None:
            return
        lane.visible = item.checkState() == Qt.CheckState.Checked
        self._layout_lanes()

    def _build_chips(self):
        seq_chips, qua_chips = {}, {}
        for p in self.bundle.pulses:
            col = p.get('color', '#cccccc')
            for line in p.get('src_lines') or ([p['src_line']] if p.get('src_line') else []):
                seq_chips.setdefault(int(line), [])
                if col not in seq_chips[int(line)]:
                    seq_chips[int(line)].append(col)
            if p.get('qua_line'):
                qua_chips.setdefault(int(p['qua_line']), [])
                if col not in qua_chips[int(p['qua_line'])]:
                    qua_chips[int(p['qua_line'])].append(col)
        for e in self.bundle.events:
            for line in e.get('src_lines') or []:
                seq_chips.setdefault(int(line), []).append('#ff7f7f')
            if e.get('qua_line'):
                qua_chips.setdefault(int(e['qua_line']), []).append('#ff7f7f')
        self.code.set_chips('seq', seq_chips)
        self.code.set_chips('qua', qua_chips)

    # ------------------------------------------------------------------
    # view state
    # ------------------------------------------------------------------
    def _capture_state(self):
        return {
            'xrange': tuple(self.master.viewRange()[0]),
            'visible': {ln.id: ln.visible for ln in self.lanes},
            'heights': {ln.id: ln.height for ln in self.lanes},
            'physical': self.show_physical,
            'selected': self._selection_key(),
            'cursors': dict(self.cursor_x),
            'origin_mode': self.origin_mode,
        }

    def _selection_key(self):
        p = self.pulse_by_id.get(self.selected) if self.selected is not None else None
        if p is None:
            return None
        return (p['lane'], p.get('src_line'), p.get('shot'), p.get('macro'),
                p.get('op'))

    def _restore_state(self, st):
        for ln in self.lanes:
            if ln.id in st['visible']:
                ln.visible = st['visible'][ln.id]
            if ln.id in st['heights']:
                ln.height = st['heights'][ln.id]
        self.show_physical = st['physical']
        self.chk_physical.blockSignals(True)
        self.chk_physical.setChecked(self.show_physical)
        self.chk_physical.blockSignals(False)
        self._build_lane_list()
        self._layout_lanes()
        x0, x1 = st['xrange']
        self.set_x_range(x0, min(x1, self.t_end * 1.02))
        key = st.get('selected')
        if key is not None:
            for p in self.bundle.pulses:
                if (p['lane'], p.get('src_line'), p.get('shot'), p.get('macro'),
                        p.get('op')) == key:
                    self.select_pulse(p['id'], scroll_code=False)
                    break
        for name, x in (st.get('cursors') or {}).items():
            if x is not None:
                self.place_cursor(name, x, snap=False)
        self._origin_changed(self.origin_combo.currentText())

    # ------------------------------------------------------------------
    # navigation
    # ------------------------------------------------------------------
    def set_x_range(self, x0, x1):
        if x1 <= x0:
            return
        self.master.setXRange(float(x0), float(x1), padding=0)

    def view_range(self):
        return self.master.viewRange()[0]

    def fit_all(self):
        self.set_x_range(0., self.t_end)

    def current_shot(self):
        """The shot under the view centre (or the nearest)."""
        shots = self.bundle.meta['shots']
        if not shots:
            return None
        x0, x1 = self.view_range()
        c = 0.5 * (x0 + x1)
        for s in shots:
            if s['t0'] <= c <= s['t1']:
                return s
        return min(shots, key=lambda s: min(abs(s['t0'] - c), abs(s['t1'] - c)))

    def fit_shot(self, index=None):
        shots = self.bundle.meta['shots']
        if not shots:
            return self.fit_all()
        s = None
        if isinstance(index, (int, np.integer)) and not isinstance(index, bool):
            s = next((x for x in shots if x['index'] == int(index)), None)
        if s is None:
            s = self.current_shot()
        pad = 0.04 * (s['t1'] - s['t0'])
        self.set_x_range(s['t0'] - pad, s['t1'] + pad)

    def zoom_to_shot(self, index):
        self.fit_shot(int(index))

    def zoom_to_pulse(self, pid):
        p = self.pulse_by_id.get(pid)
        if p is None:
            return
        w = max(p['t1'] - p['t0'], 16.)
        self.set_x_range(p['t0'] - 0.5 * w, p['t1'] + 0.5 * w)

    def zoom_by(self, factor, about=None):
        x0, x1 = self.view_range()
        if about is None:
            about = 0.5 * (x0 + x1)
        self.set_x_range(about - (about - x0) * factor, about + (x1 - about) * factor)

    def pan_by(self, frac):
        x0, x1 = self.view_range()
        d = (x1 - x0) * frac
        self.set_x_range(x0 + d, x1 + d)

    def step_shot(self, delta):
        s = self.current_shot()
        if s is None:
            return
        self.fit_shot(int(np.clip(s['index'] + delta, 0,
                                  len(self.bundle.meta['shots']) - 1)))

    def jump_edge(self, direction):
        """Centre the view on the previous / next edge (any lane)."""
        x0, x1 = self.view_range()
        c = 0.5 * (x0 + x1)
        e = self.all_edges
        if direction > 0:
            k = int(np.searchsorted(e, c + 1e-6))
        else:
            k = int(np.searchsorted(e, c - 1e-6)) - 1
        if 0 <= k < e.size:
            w = x1 - x0
            self.set_x_range(e[k] - 0.5 * w, e[k] + 0.5 * w)

    def _x_range_changed(self, *a):
        x0, x1 = self.view_range()
        if not self._sync:
            self._sync = True
            self.region.setRegion((max(x0, 0.), min(x1, self.t_end)))
            self._sync = False
        px = max(self.master.width(), 1.)
        for lane in self.lanes:
            if lane.analog is not None and self._lane_shown(lane):
                lane.analog.refresh(x0, x1, px)
        if self.origin_mode == 'shot':
            s = self.current_shot()
            t = s['t0'] if s else 0.
            if t != self.origin_t:
                self.origin_t = t
                self.axis.set_origin(t, f"shot {s['index']}" if s else '')
                self._update_cursor_label()
        self.shot_combo.blockSignals(True)
        s = self.current_shot()
        if s is not None:
            self.shot_combo.setCurrentIndex(
                max(self.shot_combo.findData(s['index']), 0))
        self.shot_combo.blockSignals(False)

    def _region_moved(self):
        if self._sync:
            return
        self._sync = True
        x0, x1 = self.region.getRegion()
        self.master.setXRange(x0, x1, padding=0)
        self._sync = False

    def _overview_clicked(self, ev):
        pos = self.overview.plotItem.vb.mapSceneToView(ev.scenePos())
        x0, x1 = self.view_range()
        w = x1 - x0
        self.set_x_range(pos.x() - w / 2., pos.x() + w / 2.)

    def _shot_combo_changed(self, idx):
        if idx >= 0:
            self.fit_shot(self.shot_combo.itemData(idx))

    def _origin_changed(self, text):
        self.origin_mode = {'shot start': 'shot', 'absolute': 'abs',
                            'cursor A': 'A'}.get(text, 'shot')
        if self.origin_mode == 'abs':
            self.origin_t = 0.
            self.axis.set_origin(0., '')
        elif self.origin_mode == 'A':
            self.origin_t = self.cursor_x['A'] or 0.
            self.axis.set_origin(self.origin_t, 'cursor A')
        else:
            s = self.current_shot()
            self.origin_t = s['t0'] if s else 0.
            self.axis.set_origin(self.origin_t, f"shot {s['index']}" if s else '')
        self._update_cursor_label()
        if hasattr(self, 'table_model'):
            self.table_model.layoutChanged.emit()

    def _height_wheel(self, vb, delta):
        for lane in self.lanes:
            if lane.vb is vb:
                lane.height = float(np.clip(lane.height * (1.15 if delta > 0 else 1 / 1.15),
                                            0.3, 4.0))
                self._layout_lanes()
                return

    # ------------------------------------------------------------------
    # toggles
    # ------------------------------------------------------------------
    def set_physical_visible(self, on):
        self.show_physical = bool(on)
        if self.chk_physical.isChecked() != self.show_physical:
            self.chk_physical.blockSignals(True)
            self.chk_physical.setChecked(self.show_physical)
            self.chk_physical.blockSignals(False)
        self._layout_lanes()
        self._x_range_changed()

    def set_gaps_visible(self, on):
        self.show_gaps = bool(on)
        for lane in self.lanes:
            if lane.gaps is not None:
                lane.gaps.enabled = self.show_gaps
                lane.gaps.update()

    def set_labels_visible(self, on):
        self.show_labels = bool(on)
        for lane in self.lanes:
            if lane.bars is not None and lane.kind not in ('digital', 'analog'):
                lane.bars.labels_enabled = self.show_labels
                lane.bars.update()

    # ------------------------------------------------------------------
    # hover / click / selection
    # ------------------------------------------------------------------
    def _lane_at(self, scene_pos):
        for lane in self.lanes:
            if not self._lane_shown(lane):
                continue
            if lane.vb.sceneBoundingRect().contains(scene_pos):
                return lane
        if self.header_vb.sceneBoundingRect().contains(scene_pos):
            return None
        return None

    def _mouse_moved(self, pos):
        self._hover_pos = pos
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _do_hover(self):
        pos = self._hover_pos
        if pos is None or self.bundle is None:
            return
        lane = self._lane_at(pos)
        if lane is None:
            if self.header_vb.sceneBoundingRect().contains(pos):
                t = self.header_vb.mapSceneToView(pos).x()
                self.hover_label.setText(self._time_readout(t))
            self._set_hover(None, None)
            return
        t = lane.vb.mapSceneToView(pos).x()
        tol = 4. * lane.vb.viewPixelSize()[0]
        pid = lane.bars.hit(t, tol) if lane.bars is not None else None
        eid = None
        if lane.events is not None:
            eid = lane.events.hit(t, 5. * lane.vb.viewPixelSize()[0])
        if eid is not None and (pid is None or lane.kind in ('analog', 'digital')):
            pid = None
        self._set_hover(pid, eid)
        # status readout
        parts = [self._time_readout(t)]
        v = lane.value_text(t)
        if v:
            parts.append(f"{lane.meta['label']}: {v}")
        e = lane.edges()
        if e.size:
            k = int(np.searchsorted(e, t))
            cands = e[max(k - 1, 0):k + 1]
            ne = float(cands[np.argmin(np.abs(cands - t))])
            parts.append(f"nearest edge {fmt_time(ne - self.origin_t, self._unit())}"
                         f" (Δ {fmt_duration(t - ne) if t >= ne else '-' + fmt_duration(ne - t)})")
        self.hover_label.setText('   |   '.join(parts))
        if pid is not None:
            QToolTip.showText(QCursor.pos(), self._pulse_tooltip(pid), self.glw)
        elif eid is not None:
            QToolTip.showText(QCursor.pos(), self._event_tooltip(eid), self.glw)
        else:
            QToolTip.hideText()

    def _set_hover(self, pid, eid):
        if (pid, eid) == self._hover_last:
            return
        self._hover_last = (pid, eid)
        for lane in self.lanes:
            if lane.bars is not None:
                lane.bars.hovered = pid
                lane.bars.update()
            if lane.events is not None:
                lane.events.hovered = eid
                lane.events.update()

    def _unit(self):
        x0, x1 = self.view_range()
        return unit_for(x1 - x0)[0]

    def _time_readout(self, t):
        u = self._unit()
        s = f"t = {fmt_time(t - self.origin_t, u)}"
        if self.origin_t:
            s += f"  (abs {fmt_time(t, u)})"
        return s

    def _pulse_tooltip(self, pid):
        p = self.pulse_by_id[pid]
        lane = self.lane_by_id.get(p['lane'])
        rows = []
        head = p.get('text') or p.get('pulse_name') or ''
        rows.append(f"<b style='color:{p.get('color', '#fff')}'>{html.escape(head)}</b>"
                    f" &nbsp;<span style='color:#999'>{html.escape(lane.meta['label'] if lane else p['lane'])}</span>")
        u = 'µs' if (p['t1'] - p['t0']) < 2e6 else 'ms'
        rows.append(f"start {fmt_time(p['t0'] - self.origin_t, u)} · end "
                    f"{fmt_time(p['t1'] - self.origin_t, u)} · length "
                    f"<b>{fmt_duration(p['t1'] - p['t0'])}</b>")
        if p.get('requested_ns') is not None and p.get('rounded'):
            rows.append(f"<span style='color:#ffb74d'>requested "
                        f"{fmt_duration(p['requested_ns'])} → rounded to the 4 ns grid</span>")
        if p.get('truncated'):
            rows.append("<span style='color:#ff6b6b'>extends past the simulated window</span>")
        if p.get('value_str'):
            rows.append(f"parameter: <b>{html.escape(p['value_str'])}</b>")
        meta = []
        if p.get('macro'):
            meta.append(f"ctx.{p['macro']}" if p.get('phase') == 'body' else p['macro'])
        if p.get('element'):
            meta.append(f"element {p['element']}")
        if p.get('op'):
            meta.append(f"op '{p['op']}'")
        if p.get('shot') is not None:
            meta.append(f"shot {p['shot']}")
        meta.append(p.get('phase', ''))
        rows.append(html.escape(' · '.join(x for x in meta if x)))
        if p.get('src_line'):
            src = self.bundle.meta['sources'].get('seq', {})
            rows.append(f"source: {html.escape(src.get('title', ''))}:{p['src_line']}"
                        + (f" &nbsp;QUA line {p['qua_line']}" if p.get('qua_line') else ''))
            text = src.get('text', '')
            first = int(src.get('first_line', 1))
            lines = text.splitlines()
            k = int(p['src_line']) - first
            if 0 <= k < len(lines):
                rows.append(f"<code style='color:#dcdcaa'>{html.escape(lines[k].strip())}</code>")
        elif p.get('qua_line'):
            rows.append(f"QUA line {p['qua_line']} (framework framing, no sequence line)")
        if p.get('note'):
            rows.append(f"<span style='color:#aaa'>{html.escape(p['note'])}</span>")
        return "<div style='white-space:nowrap'>" + '<br>'.join(rows) + "</div>"

    def _event_tooltip(self, eid):
        e = self.event_by_id[eid]
        rows = [f"<b style='color:#ffb3c1'>{html.escape(e.get('label', e['kind']))}</b>",
                f"t = {fmt_time(e['t'] - self.origin_t, self._unit())}"
                + (" (inferred from the program)" if e.get('approx') else '')]
        if e.get('detail'):
            rows.append(html.escape(e['detail']))
        if e.get('src_line'):
            rows.append(f"source line {e['src_line']}"
                        + (f" · QUA line {e['qua_line']}" if e.get('qua_line') else ''))
        return '<br>'.join(rows)

    def _vb_clicked(self, vb, scene_pos):
        lane = next((ln for ln in self.lanes if ln.vb is vb), None)
        if lane is None:
            self.clear_selection()
            return
        t = lane.vb.mapSceneToView(scene_pos).x()
        tol = 4. * lane.vb.viewPixelSize()[0]
        eid = lane.events.hit(t, 5. * lane.vb.viewPixelSize()[0]) if lane.events else None
        pid = lane.bars.hit(t, tol) if lane.bars is not None else None
        if eid is not None and (pid is None or lane.kind in ('analog', 'digital')):
            self.select_event(eid)
        elif pid is not None:
            self.select_pulse(pid)
        else:
            self.clear_selection()

    def _vb_double_clicked(self, vb, scene_pos):
        lane = next((ln for ln in self.lanes if ln.vb is vb), None)
        pid = None
        if lane is not None and lane.bars is not None:
            t = lane.vb.mapSceneToView(scene_pos).x()
            pid = lane.bars.hit(t, 4. * lane.vb.viewPixelSize()[0])
        if pid is not None:
            self.select_pulse(pid)
            self.zoom_to_pulse(pid)
        else:
            self.fit_shot()          # empty space: back to the shot

    def _vb_right_clicked(self, vb, scene_pos):
        # a plain right-click zooms out about the cursor (right-drag = box zoom)
        t = vb.mapSceneToView(scene_pos).x()
        self.zoom_by(2.0, about=t)

    def select_pulse(self, pid, scroll_code=True):
        p = self.pulse_by_id.get(pid)
        if p is None:
            return
        self.selected = pid
        self.selected_event = None
        line = p.get('src_line')
        related = set()
        if line is not None:
            related = {q['id'] for q in self.bundle.pulses
                       if q.get('src_line') == line and q['id'] != pid}
        for lane in self.lanes:
            if lane.bars is not None:
                lane.bars.selected = {pid}
                lane.bars.related = related
                lane.bars.dim_others = True
                lane.bars.update()
            if lane.events is not None:
                lane.events.selected = set()
                lane.events.update()
        if line is not None:
            self.code.highlight('seq', current=[line],
                                related=[x for x in p.get('src_lines', []) if x != line],
                                scroll=scroll_code)
        else:
            self.code.highlight('seq', current=[], related=[], scroll=False)
        if p.get('qua_line'):
            self.code.highlight('qua', current=[p['qua_line']], scroll=scroll_code)
        else:
            self.code.highlight('qua', current=[], scroll=False)
        self._sync_table_selection(pid)
        self._update_cursor_label()

    def select_event(self, eid):
        e = self.event_by_id.get(eid)
        if e is None:
            return
        self.selected = None
        self.selected_event = eid
        for lane in self.lanes:
            if lane.bars is not None:
                lane.bars.selected = set()
                lane.bars.related = set()
                lane.bars.dim_others = False
                lane.bars.update()
            if lane.events is not None:
                lane.events.selected = {eid}
                lane.events.update()
        if e.get('src_line'):
            self.code.highlight('seq', current=[e['src_line']],
                                related=e.get('src_lines', []))
        if e.get('qua_line'):
            self.code.highlight('qua', current=[e['qua_line']])
        self._update_cursor_label()

    def clear_selection(self):
        self.selected = None
        self.selected_event = None
        for lane in self.lanes:
            if lane.bars is not None:
                lane.bars.selected = set()
                lane.bars.related = set()
                lane.bars.dim_others = False
                lane.bars.update()
            if lane.events is not None:
                lane.events.selected = set()
                lane.events.update()
        self.code.clear_highlights()
        self._update_cursor_label()

    def _code_line_clicked(self, key, line):
        field = 'src_line' if key == 'seq' else ('qua_line' if key == 'qua' else None)
        if field is None:
            return
        ids = [p['id'] for p in self.bundle.pulses if p.get(field) == line]
        if key == 'seq':
            ids += [p['id'] for p in self.bundle.pulses
                    if line in (p.get('src_lines') or []) and p['id'] not in ids]
        eids = [e['id'] for e in self.bundle.events if e.get(field) == line]
        if not ids and not eids:
            self.clear_selection()
            return
        self.selected = None
        for lane in self.lanes:
            if lane.bars is not None:
                lane.bars.selected = set()
                lane.bars.related = set(ids)
                lane.bars.dim_others = True
                lane.bars.update()
            if lane.events is not None:
                lane.events.selected = set(eids)
                lane.events.update()
        self.code.highlight(key, current=[line], scroll=False)
        # bring one into view if none is visible
        x0, x1 = self.view_range()
        ts = [self.pulse_by_id[i]['t0'] for i in ids] + [self.event_by_id[i]['t'] for i in eids]
        if ts and not any(x0 <= t <= x1 for t in ts):
            w = x1 - x0
            self.set_x_range(ts[0] - 0.1 * w, ts[0] + 0.9 * w)
        self.hover_label.setText(f"{len(ids)} pulse(s), {len(eids)} event(s) from line {line}")

    def _code_line_double_clicked(self, key, line):
        field = 'src_line' if key == 'seq' else 'qua_line'
        ids = [p['id'] for p in self.bundle.pulses if p.get(field) == line]
        if ids:
            ps = [self.pulse_by_id[i] for i in ids]
            t0 = min(p['t0'] for p in ps)
            t1 = max(p['t1'] for p in ps)
            w = max(t1 - t0, 16.)
            self.set_x_range(t0 - 0.2 * w, t1 + 0.2 * w)

    def _sync_table_selection(self, pid):
        if not hasattr(self, 'table_model'):
            return
        row = next((i for i, p in enumerate(self.bundle.pulses) if p['id'] == pid), None)
        if row is None:
            return
        src = self.table_model.index(row, 0)
        prox = self.table_proxy.mapFromSource(src)
        sm = self.table.selectionModel()
        sm.blockSignals(True)
        self.table.setCurrentIndex(prox)
        self.table.scrollTo(prox)
        sm.blockSignals(False)

    def _table_row(self, current, previous):
        if not current.isValid():
            return
        src = self.table_proxy.mapToSource(current)
        p = self.bundle.pulses[src.row()]
        self.select_pulse(p['id'])
        x0, x1 = self.view_range()
        if not (x0 <= p['t0'] <= x1 or x0 <= p['t1'] <= x1):
            w = x1 - x0
            c = 0.5 * (p['t0'] + p['t1'])
            if p['t1'] - p['t0'] > w:
                self.zoom_to_pulse(p['id'])
            else:
                self.set_x_range(c - w / 2., c + w / 2.)

    def _table_double(self, idx):
        src = self.table_proxy.mapToSource(idx)
        self.zoom_to_pulse(self.bundle.pulses[src.row()]['id'])

    # ------------------------------------------------------------------
    # cursors
    # ------------------------------------------------------------------
    def place_cursor(self, name, x, snap=True):
        if snap:
            k = int(np.searchsorted(self.all_edges, x))
            tol = 6. * self.master.viewPixelSize()[0]
            cands = [e for e in self.all_edges[max(k - 1, 0):k + 1] if abs(e - x) <= tol]
            if cands:
                x = float(min(cands, key=lambda e: abs(e - x)))
        self.cursor_x[name] = float(x)
        lines = self._cursor_lines.setdefault(name, {})
        for lane in self.lanes + [None]:
            vb = lane.vb if lane is not None else self.header_vb
            key = lane.id if lane is not None else '__header__'
            ln = lines.get(key)
            if ln is None:
                ln = CursorLine(name, CURSOR_COLORS.get(name, '#ffffff'),
                                self.all_edges,
                                lambda xx, n=name: self._cursor_dragged(n, xx))
                vb.addItem(ln, ignoreBounds=True)
                lines[key] = ln
            ln.blockSignals(True)
            ln.setValue(x)
            ln.blockSignals(False)
            ln.show()
        if self.origin_mode == 'A' and name == 'A':
            self._origin_changed('cursor A')
        self._update_cursor_label()

    def _cursor_dragged(self, name, x):
        self.place_cursor(name, x, snap=False)

    def clear_cursors(self):
        for name, lines in self._cursor_lines.items():
            for ln in lines.values():
                ln.hide()
        self.cursor_x = {'A': None, 'B': None}
        self._update_cursor_label()

    def _measured(self, x0, x1):
        self.place_cursor('A', x0)
        self.place_cursor('B', x1)

    def _update_cursor_label(self):
        a, b = self.cursor_x['A'], self.cursor_x['B']
        u = self._unit() if self.bundle else 'µs'
        parts = []
        if a is not None:
            parts.append(f"A = {fmt_time(a - self.origin_t, u)}")
        if b is not None:
            parts.append(f"B = {fmt_time(b - self.origin_t, u)}")
        if a is not None and b is not None:
            d = b - a
            parts.append(f"Δ = {fmt_duration(abs(d))}"
                         + (f" ({1e3 / abs(d):.4g} MHz)" if d else ''))
        if self.selected is not None:
            p = self.pulse_by_id[self.selected]
            parts.append(f"selected: {p.get('text') or p.get('pulse_name')} "
                         f"[{fmt_time(p['t0'] - self.origin_t, u)} → "
                         f"{fmt_time(p['t1'] - self.origin_t, u)}]")
        self.cursor_label.setText('   '.join(parts))

    def copy_readout(self):
        text = self.cursor_label.text() + '\n' + self.hover_label.text()
        QGuiApplication.clipboard().setText(text.strip())

    # ------------------------------------------------------------------
    # keyboard
    # ------------------------------------------------------------------
    def keyPressEvent(self, ev):
        k = ev.key()
        mods = ev.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if ctrl and k == Qt.Key.Key_C:
            self.copy_readout()
        elif ctrl and k == Qt.Key.Key_E:
            self.export_csv() if shift else self.export_png()
        elif k in (Qt.Key.Key_Home, Qt.Key.Key_F) and not ctrl:
            self.fit_all() if k == Qt.Key.Key_F else self.fit_shot()
        elif k == Qt.Key.Key_Z:
            if self.selected is not None:
                self.zoom_to_pulse(self.selected)
            elif self.cursor_x['A'] is not None and self.cursor_x['B'] is not None:
                a, b = sorted((self.cursor_x['A'], self.cursor_x['B']))
                w = max(b - a, 16.)
                self.set_x_range(a - 0.1 * w, b + 0.1 * w)
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(0.5)
        elif k in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom_by(2.0)
        elif k == Qt.Key.Key_Left:
            self.pan_by(-0.5 if shift else -0.1)
        elif k == Qt.Key.Key_Right:
            self.pan_by(0.5 if shift else 0.1)
        elif k == Qt.Key.Key_BracketLeft:
            self.jump_edge(-1)
        elif k == Qt.Key.Key_BracketRight:
            self.jump_edge(+1)
        elif k == Qt.Key.Key_Comma:
            self.step_shot(-1)
        elif k == Qt.Key.Key_Period:
            self.step_shot(+1)
        elif Qt.Key.Key_0 <= k <= Qt.Key.Key_9 and not ctrl:
            self.fit_shot(int(k - Qt.Key.Key_0))
        elif k in (Qt.Key.Key_A, Qt.Key.Key_B) and not ctrl:
            x = self._mouse_x()
            if x is not None:
                self.place_cursor('A' if k == Qt.Key.Key_A else 'B', x)
        elif k == Qt.Key.Key_Escape:
            self.clear_selection()
            self.clear_cursors()
        elif k == Qt.Key.Key_P:
            self.set_physical_visible(not self.show_physical)
        elif k == Qt.Key.Key_G:
            self.chk_gaps.setChecked(not self.show_gaps)
        elif k == Qt.Key.Key_L:
            self.chk_labels.setChecked(not self.show_labels)
        elif k == Qt.Key.Key_T:
            self.table_dock.setVisible(not self.table_dock.isVisible())
        elif k == Qt.Key.Key_Question or (k == Qt.Key.Key_H and not ctrl):
            self.show_help()
        else:
            super().keyPressEvent(ev)

    def _mouse_x(self):
        pos = self.glw.mapFromGlobal(QCursor.pos())
        scene_pos = self.glw.mapToScene(pos)
        lane = self._lane_at(scene_pos)
        vb = lane.vb if lane is not None else self.master
        if not vb.sceneBoundingRect().contains(scene_pos) and lane is None:
            x0, x1 = self.view_range()
            return 0.5 * (x0 + x1)
        return vb.mapSceneToView(scene_pos).x()

    # ------------------------------------------------------------------
    # export / reload / help
    # ------------------------------------------------------------------
    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export PNG', 'seqview.png',
                                              'PNG (*.png)')
        if not path:
            return
        from pyqtgraph.exporters import ImageExporter
        ex = ImageExporter(self.glw.ci)
        ex.parameters()['width'] = max(self.glw.width() * 2, 1600)
        ex.export(path)
        self.hover_label.setText(f"saved {path}")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export pulses (visible range) CSV',
                                              'seqview_pulses.csv', 'CSV (*.csv)')
        if not path:
            return
        x0, x1 = self.view_range()
        cols = ['lane', 'text', 'element', 'op', 'pulse_name', 't0_ns', 't1_ns',
                'length_ns', 'requested_ns', 'shot', 'phase', 'macro', 'label',
                'value', 'src_line', 'qua_line', 'note']
        with open(path, 'w', encoding='utf-8') as f:
            f.write(','.join(cols) + '\n')
            for p in self.bundle.pulses:
                if p['t1'] < x0 or p['t0'] > x1:
                    continue
                row = [p['lane'], p.get('text', ''), p.get('element') or '',
                       p.get('op') or '', p.get('pulse_name') or '',
                       f"{p['t0']:.1f}", f"{p['t1']:.1f}",
                       f"{p['t1'] - p['t0']:.1f}",
                       '' if p.get('requested_ns') is None else f"{p['requested_ns']:.3f}",
                       '' if p.get('shot') is None else str(p['shot']),
                       p.get('phase', ''), p.get('macro', ''), p.get('label', ''),
                       '' if p.get('value') is None else repr(p['value']),
                       '' if p.get('src_line') is None else str(p['src_line']),
                       '' if p.get('qua_line') is None else str(p['qua_line']),
                       p.get('note', '')]
                f.write(','.join('"' + str(x).replace('"', '""') + '"' for x in row) + '\n')
        self.hover_label.setText(f"saved {path} (pulses in the visible range, times in ns)")

    def reload_file(self):
        if getattr(self, 'bundle_path', None):
            try:
                self.load_bundle(Bundle.load(self.bundle_path), keep_view=True,
                                 path=self.bundle_path)
            except Exception as e:
                QMessageBox.warning(self, 'reload failed', str(e))

    def show_help(self):
        text = __doc__.split('Interaction')[1]
        QMessageBox.information(self, 'seqview — keys and mouse',
                                '<pre>Interaction' + html.escape(text) + '</pre>')

    # ------------------------------------------------------------------
    # reload server (re-simulate in place)
    # ------------------------------------------------------------------
    def _start_server(self):
        self._server = QTcpServer(self)
        if not self._server.listen(QHostAddress.SpecialAddress.LocalHost, 0):
            self._server = None
            return
        _launch.write_port(self._server.serverPort())
        self._server.newConnection.connect(self._on_connection)

    def _on_connection(self):
        while self._server and self._server.hasPendingConnections():
            sock = self._server.nextPendingConnection()
            sock.readyRead.connect(lambda s=sock: self._on_ready(s))

    def _on_ready(self, sock):
        data = bytes(sock.readAll()).decode('utf-8', 'replace')
        for line in data.splitlines():
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            path = msg.get('open')
            if path:
                try:
                    self.load_bundle(Bundle.load(path), keep_view=True, path=path)
                    sock.write(b'ok\n')
                    self.raise_window()
                except Exception as e:
                    sock.write(f'error {e}\n'.encode('utf-8'))
        sock.flush()
        sock.disconnectFromHost()

    def raise_window(self):
        self.show()
        self.raise_()
        self.activateWindow()
        if os.name == 'nt':
            try:
                import ctypes
                hwnd = int(self.winId())
                SWP = 0x0001 | 0x0002 | 0x0040   # NOSIZE | NOMOVE | SHOWWINDOW
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, SWP)
                ctypes.windll.user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, SWP)
            except Exception:
                pass

    def closeEvent(self, ev):
        if self._server is not None:
            port, pid = _launch.read_port()
            if pid == os.getpid():
                _launch.clear_port()
            self._server.close()
        super().closeEvent(ev)
