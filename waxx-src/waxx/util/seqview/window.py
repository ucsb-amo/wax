"""The viewer window.

Layout: toolbar / overview strip / lane stack (label column + x-linked
plots) with the time axis under the last lane / status bar; docks for the
code pane (right), the pulse table and warnings (bottom), lane visibility
(left, hidden by default).

Interaction (all navigation is plain mouse; measuring takes a modifier):
    wheel               zoom about the cursor        Shift+wheel: pan
    right-click         zoom out about the cursor    Ctrl+wheel: lane height
    left-drag           pan                          left-click: select
    right-drag          box zoom                     Ctrl+click: add / remove
    double-click        zoom to pulse (empty space: fit shot)
    Shift+left-drag     measure a span (cursors A/B snap to edges; Alt: no snap)
    A / B               drop cursor A / B at the mouse
    Home / F            fit current shot / fit all   Z: zoom to selection
    + / -  ← / →        zoom / pan (Shift: faster)   [ ]: previous / next edge
    , / .  0-9          previous / next shot, jump to shot
    P / G / L           port lanes / gap dimensions / labels    T: pulse table
    Esc                 clear selection and cursors  Ctrl+C: copy readout
    F5                  reload                       Ctrl+E: export PNG
                                                     Ctrl+Shift+E: export CSV

Pulse table: click selects a row, Ctrl+click adds / removes one, Shift+click
selects a range, Ctrl+Shift+click adds a range; double-click zooms to it.
Code pane: click a line to select everything it emitted, double-click to zoom
to it. With "Snap view" on, a code or table click moves the view to the
pulses (the nearest shot's) when none of them is on screen.
"""

import html
import json
import os

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QPointF, QRect, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import (QAction, QColor, QKeySequence, QCursor, QPen, QBrush,
                         QGuiApplication)
from PyQt6.QtNetwork import QTcpServer, QHostAddress
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QToolBar, QComboBox, QDockWidget,
                             QTableView, QListWidget, QListWidgetItem,
                             QStatusBar, QToolTip, QFileDialog,
                             QLineEdit, QPushButton, QSizePolicy, QMessageBox)
from PyQt6.QtCore import (QAbstractTableModel, QModelIndex, QSortFilterProxyModel,
                          QItemSelection, QItemSelectionModel)

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
# Qt expires a tooltip after ~10 s even with the cursor still on it; 24 h is
# "never" -- the hover tip goes when the mouse leaves the pulse / event
# (or on a click / wheel, which Qt handles itself)
TIP_SHOW_MS = 24 * 3600 * 1000
EVENT_COLOR = '#ff7f7f'
TOOLBAR_CSS = """
QToolBar { spacing: 3px; padding: 2px 4px; border: none; }
QToolBar QToolButton { padding: 2px 8px; border: 1px solid transparent;
                       border-radius: 3px; }
QToolBar QToolButton[toggle="true"] { border-color: #3d424b; color: #b0b0b0; }
QToolBar QToolButton:hover { border-color: #6a707b; }
QToolBar QToolButton:checked { background: #2f4f86; border-color: #4f7fd0;
                               color: #ffffff; }
QToolBar QLabel { color: #9a9a9a; padding-left: 4px; }
QToolBar::separator { background: #3d424b; width: 1px; margin: 4px 6px; }
"""


def _pulse_key(p):
    """What identifies a pulse across a re-simulation (with its occurrence
    count, see SeqViewWindow._selection_state)."""
    return (p['lane'], p.get('src_line'), p.get('shot'), p.get('macro'), p.get('op'))


def _event_key(e):
    return (e['lane'], e.get('src_line'), e.get('shot'), e.get('kind'))


def _occurrence_keys(items, keyf):
    """id -> (key, n): the n-th item with that key, in bundle order."""
    seen, out = {}, {}
    for it in items:
        k = keyf(it)
        n = seen.get(k, 0)
        seen[k] = n + 1
        out[it['id']] = (k, n)
    return out


# ---------------------------------------------------------------------------
# view box with the interaction model
# ---------------------------------------------------------------------------

class LaneViewBox(pg.ViewBox):
    sigClicked = pyqtSignal(object, object, object)  # (viewbox, scene pos, modifiers)
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
                self.sigClicked.emit(self, ev.scenePos(), ev.modifiers())
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
        self.sel_pulses = []          # selected pulse ids, in selection order
        self.sel_events = []
        self.selected = None          # the pulse acted on last (readout, Z)
        self.selected_event = None
        self.cursor_x = {'A': None, 'B': None}
        self.origin_mode = 'shot'
        self.origin_t = 0.0
        self.show_physical = False
        self.show_gaps = True
        self.show_labels = True
        self.snap_view = True         # code / table clicks bring pulses on screen
        self._ov_bars = []
        self._table_syncing = False
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

        # toolbar: navigation | shot, time origin | what is drawn | snap |
        # panels (added with the docks below) ... title
        tb = QToolBar('view')
        tb.setMovable(False)
        tb.setStyleSheet(TOOLBAR_CSS)
        self.addToolBar(tb)
        self.act_fit_shot = self._action('Fit shot', self.fit_shot,
                                         tip='Fit the shot under the view (Home)')
        self.act_fit_all = self._action('Fit all', self.fit_all,
                                        tip='Show the whole program (F)')
        tb.addAction(self.act_fit_shot)
        tb.addAction(self.act_fit_all)
        tb.addAction(self._action('−', lambda: self.zoom_by(2.0),
                                  tip='Zoom out (-, right-click)'))
        tb.addAction(self._action('+', lambda: self.zoom_by(0.5),
                                  tip='Zoom in (+, wheel)'))
        tb.addSeparator()
        tb.addWidget(QLabel('shot'))
        self.shot_combo = QComboBox()
        self.shot_combo.setToolTip('Jump to a shot (, and . step, 0-9 jump)')
        self.shot_combo.currentIndexChanged.connect(self._shot_combo_changed)
        tb.addWidget(self.shot_combo)
        tb.addWidget(QLabel('t = 0 at'))
        self.origin_combo = QComboBox()
        self.origin_combo.addItems(['shot start', 'absolute', 'cursor A'])
        self.origin_combo.setToolTip('What the time axis and the readouts count from')
        self.origin_combo.currentTextChanged.connect(self._origin_changed)
        tb.addWidget(self.origin_combo)
        tb.addSeparator()
        self.act_physical = self._toggle(
            'Ports', 'Physical port lanes\tP', False, self.set_physical_visible,
            'Show the raw digital / analog port lanes under the semantic ones (P)')
        self.act_gaps = self._toggle(
            'Gaps', 'Gap dimensions\tG', True, self.set_gaps_visible,
            'Draw the <-- gap --> dimensions between pulses (G)')
        self.act_labels = self._toggle(
            'Labels', 'Pulse labels\tL', True, self.set_labels_visible,
            'Write the pulse text on bars wide enough to hold it (L)')
        for a in (self.act_physical, self.act_gaps, self.act_labels):
            tb.addAction(a)
        tb.addSeparator()
        self.act_snap = self._toggle(
            'Snap view', 'Snap view to code / table clicks', True, self.set_snap_view,
            'On: clicking a code line or a pulse-table row moves the view to its '
            'pulses when none of them is on screen.\nOff: those clicks only '
            'select; double-click still zooms.')
        tb.addAction(self.act_snap)

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
        # click / Ctrl+click / Shift+click / Ctrl+Shift+click, as in a file list
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
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

        # panels: the dock toggles, in the toolbar and the View menu
        panels = ((self.code_dock, 'Code', 'Code panel',
                   'Sequence source, generated QUA, parameters'),
                  (self.table_dock, 'Pulses', 'Pulse table\tT',
                   'Every pulse as a sortable, filterable table (T)'),
                  (self.lane_dock, 'Lanes', 'Lane list', 'Show / hide single lanes'),
                  (self.warn_dock, 'Diagnostics', 'Diagnostics',
                   'Warnings from matching the program to the simulation'))
        tb.addSeparator()
        for dock, short, text, tip in panels:
            a = dock.toggleViewAction()
            a.setText(text)
            a.setIconText(short)
            a.setToolTip(tip)
            if dock is not self.warn_dock:        # that one has the status-bar badge
                tb.addAction(a)
        for a in tb.actions():                    # outline the toggles (TOOLBAR_CSS)
            btn = tb.widgetForAction(a) if a.isCheckable() else None
            if btn is not None:
                btn.setProperty('toggle', True)
                btn.style().unpolish(btn)
                btn.style().polish(btn)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        self.title_label = QLabel('')
        self.title_label.setStyleSheet('color:#dddddd; padding-right:8px;')
        tb.addWidget(self.title_label)

        file_menu = self.menuBar().addMenu('File')
        file_menu.addAction(self._action('Reload', self.reload_file, shortcut='F5'))
        file_menu.addSeparator()
        file_menu.addAction(self._action('Export PNG…', self.export_png,
                                         shortcut='Ctrl+E'))
        file_menu.addAction(self._action('Export CSV (pulses in view)…',
                                         self.export_csv, shortcut='Ctrl+Shift+E'))
        view_menu = self.menuBar().addMenu('View')
        for a in (self.act_physical, self.act_gaps, self.act_labels):
            view_menu.addAction(a)
        view_menu.addSeparator()
        view_menu.addAction(self.act_snap)
        view_menu.addSeparator()
        for dock, *_ in panels:
            view_menu.addAction(dock.toggleViewAction())
        help_menu = self.menuBar().addMenu('Help')
        help_menu.addAction(self._action('Keys and mouse\t?', self.show_help))

        # hover throttle
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(16)
        self._hover_timer.timeout.connect(self._do_hover)
        self._hover_pos = None

    def _action(self, text, slot, tip='', shortcut=None):
        a = QAction(text, self)
        a.triggered.connect(slot)
        if tip:
            a.setToolTip(tip)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        return a

    def _toggle(self, short, text, checked, slot, tip):
        """A checkable action: `short` on the toolbar button, `text` in the
        View menu (after a tab: the key, shown but handled in keyPressEvent)."""
        a = QAction(text, self)
        a.setIconText(short)
        a.setToolTip(tip)
        a.setCheckable(True)
        a.setChecked(checked)
        a.toggled.connect(slot)
        return a

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
        self.sel_pulses = []
        self.sel_events = []
        self.selected = None
        self.selected_event = None
        self.cursor_x = {'A': None, 'B': None}
        self._cursor_lines = {}
        self.overview.clear()
        self._ov_bars = []

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
            self._ov_bars.append(bars)       # shows the selection across the run
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
        self._row_of = {p['id']: i for i, p in enumerate(self.bundle.pulses)}
        self.table.selectionModel().selectionChanged.connect(
            self._table_selection_changed)
        try:
            self.table.doubleClicked.disconnect()
        except Exception:
            pass
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
            'selection': self._selection_state(),
            'cursors': dict(self.cursor_x),
            'origin_mode': self.origin_mode,
        }

    def _selection_state(self):
        """The selection as bundle-independent keys: (key, n) = the n-th
        pulse / event with that key, so a re-simulation maps it back."""
        pk = _occurrence_keys(self.bundle.pulses, _pulse_key)
        ek = _occurrence_keys(self.bundle.events, _event_key)
        return {'pulses': [pk[i] for i in self.sel_pulses],
                'events': [ek[i] for i in self.sel_events],
                'current': pk.get(self.selected)}

    def _restore_selection(self, sel):
        pid_of = {v: k for k, v in _occurrence_keys(self.bundle.pulses, _pulse_key).items()}
        eid_of = {v: k for k, v in _occurrence_keys(self.bundle.events, _event_key).items()}
        pids = [pid_of[k] for k in sel.get('pulses', []) if k in pid_of]
        eids = [eid_of[k] for k in sel.get('events', []) if k in eid_of]
        if pids or eids:
            self.set_selection(pids, eids, current=pid_of.get(sel.get('current')),
                               scroll_code=False)

    def _restore_state(self, st):
        for ln in self.lanes:
            if ln.id in st['visible']:
                ln.visible = st['visible'][ln.id]
            if ln.id in st['heights']:
                ln.height = st['heights'][ln.id]
        self.set_physical_visible(st['physical'])
        self._build_lane_list()
        self._layout_lanes()
        x0, x1 = st['xrange']
        self.set_x_range(x0, min(x1, self.t_end * 1.02))
        if st.get('selection'):
            self._restore_selection(st['selection'])
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
            name = f"shot {s['index']}" if s else ''
            if t != self.origin_t or name != self.axis.origin_name:
                self.origin_t = t
                self.axis.set_origin(t, name)
                self._update_cursor_label()
                self._refresh_table_times()
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
        self._refresh_table_times()

    def _refresh_table_times(self):
        """The start / end columns count from the origin: repaint them.
        (dataChanged, not layoutChanged -- a bare layoutChanged scrambles
        the proxy's selection.)"""
        m = getattr(self, 'table_model', None)
        if m is not None and m.rowCount():
            m.dataChanged.emit(m.index(0, 2), m.index(m.rowCount() - 1, 3),
                               [Qt.ItemDataRole.DisplayRole])

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
    # Each setter can be called directly or from its action; a direct call
    # sets the action, whose toggled signal comes back here once.
    def set_physical_visible(self, on):
        on = bool(on)
        if self.act_physical.isChecked() != on:
            self.act_physical.setChecked(on)
            return
        self.show_physical = on
        if self.bundle is None:
            return
        self._layout_lanes()
        self._x_range_changed()

    def set_gaps_visible(self, on):
        on = bool(on)
        if self.act_gaps.isChecked() != on:
            self.act_gaps.setChecked(on)
            return
        self.show_gaps = on
        for lane in self.lanes:
            if lane.gaps is not None:
                lane.gaps.enabled = self.show_gaps
                lane.gaps.update()

    def set_labels_visible(self, on):
        on = bool(on)
        if self.act_labels.isChecked() != on:
            self.act_labels.setChecked(on)
            return
        self.show_labels = on
        for lane in self.lanes:
            if lane.bars is not None and lane.kind not in ('digital', 'analog'):
                lane.bars.labels_enabled = self.show_labels
                lane.bars.update()

    def set_snap_view(self, on):
        on = bool(on)
        if self.act_snap.isChecked() != on:
            self.act_snap.setChecked(on)
            return
        self.snap_view = on

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
            QToolTip.hideText()
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
            QToolTip.showText(QCursor.pos(), self._pulse_tooltip(pid), self.glw,
                              QRect(), TIP_SHOW_MS)
        elif eid is not None:
            QToolTip.showText(QCursor.pos(), self._event_tooltip(eid), self.glw,
                              QRect(), TIP_SHOW_MS)
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

    def _vb_clicked(self, vb, scene_pos, mods=Qt.KeyboardModifier.NoModifier):
        """Click selects what is under the mouse (empty space clears);
        Ctrl+click adds it to the selection or takes it out."""
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        lane = next((ln for ln in self.lanes if ln.vb is vb), None)
        pid = eid = None
        if lane is not None:
            t = lane.vb.mapSceneToView(scene_pos).x()
            px = lane.vb.viewPixelSize()[0]
            eid = lane.events.hit(t, 5. * px) if lane.events else None
            pid = lane.bars.hit(t, 4. * px) if lane.bars is not None else None
            if eid is not None and (pid is None or lane.kind in ('analog', 'digital')):
                pid = None
            else:
                eid = None
        if not ctrl:
            self.set_selection([] if pid is None else [pid],
                               [] if eid is None else [eid], source='timeline')
            return
        pids, eids = list(self.sel_pulses), list(self.sel_events)
        if pid is not None:
            pids = [x for x in pids if x != pid] if pid in pids else pids + [pid]
        elif eid is not None:
            eids = [x for x in eids if x != eid] if eid in eids else eids + [eid]
        else:
            return                   # Ctrl+click on empty space keeps the selection
        self.set_selection(pids, eids, current=pid, source='timeline')

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

    # --- the selection: one state, shown by the lanes, the overview, the
    # --- pulse table and the code pane

    def set_selection(self, pids=(), eids=(), current=None, source=None,
                      code_line=None, scroll_code=True):
        """The one place the selection changes.

        pids / eids: selected pulse / event ids, in selection order.
        current: the pulse acted on last (status readout, Z, which code
        line to scroll to); default the last one. source: 'timeline',
        'table' or 'code' -- the widget the click came from is not written
        back to (that is what made the table fight the user). code_line:
        (source key, line) of a code click, highlighted along with the
        pulses' own lines."""
        if self.bundle is None:
            return
        pids = [int(i) for i in dict.fromkeys(pids) if int(i) in self.pulse_by_id]
        eids = [int(i) for i in dict.fromkeys(eids) if int(i) in self.event_by_id]
        self.sel_pulses, self.sel_events = pids, eids
        if current not in pids:
            current = pids[-1] if pids else None
        self.selected = current
        self.selected_event = eids[-1] if eids else None
        psel, esel = set(pids), set(eids)
        # pulses from the same emitting line: dotted outline
        lines = {self.pulse_by_id[i].get('src_line') for i in pids} - {None}
        related = ({q['id'] for q in self.bundle.pulses
                    if q.get('src_line') in lines} - psel) if lines else set()
        dim = bool(psel or esel)
        for bars in [ln.bars for ln in self.lanes if ln.bars is not None] + self._ov_bars:
            bars.selected = psel
            bars.related = related
            bars.dim_others = dim
            bars.update()
        for lane in self.lanes:
            if lane.events is not None:
                lane.events.selected = esel
                lane.events.update()
        self._highlight_code(pids, eids, current, code_line,
                             scroll=scroll_code and source != 'code')
        if source != 'table':
            self._sync_table_selection(pids, current)
        self._update_cursor_label()

    def select_pulse(self, pid, scroll_code=True):
        self.set_selection([pid], current=pid, scroll_code=scroll_code)

    def select_event(self, eid):
        self.set_selection([], [eid])

    def clear_selection(self):
        self.set_selection([], [])

    def _highlight_code(self, pids, eids, current, code_line, scroll):
        """Tint the lines the selection was emitted from with their lane
        colour; the call chain above them gets a grey band."""
        items = ([self.pulse_by_id[i] for i in pids]
                 + [self.event_by_id[i] for i in eids])
        cur = {'seq': set(), 'qua': set()}
        colors = {'seq': {}, 'qua': {}}
        chain = set()
        for it in items:
            col = it.get('color') or EVENT_COLOR
            for key, field in (('seq', 'src_line'), ('qua', 'qua_line')):
                if it.get(field):
                    ln = int(it[field])
                    cur[key].add(ln)
                    colors[key].setdefault(ln, col)
            chain.update(int(x) for x in it.get('src_lines') or [])
        if code_line is not None and code_line[0] in cur:
            cur[code_line[0]].add(int(code_line[1]))
        focus = (self.pulse_by_id.get(current) if current is not None else
                 (self.event_by_id.get(eids[-1]) if eids else None)) or {}
        for key, field in (('seq', 'src_line'), ('qua', 'qua_line')):
            self.code.highlight(key, cur[key], chain if key == 'seq' else (),
                                scroll=scroll and bool(focus.get(field)),
                                colors=colors[key], scroll_line=focus.get(field))

    def _items_from_line(self, key, line):
        """Pulse and event ids emitted by a code line (for the sequence
        source: also by calls that pass through it)."""
        field = {'seq': 'src_line', 'qua': 'qua_line'}.get(key)
        if field is None:
            return None, None

        def hit(it):
            return it.get(field) == line or (
                key == 'seq' and line in (it.get('src_lines') or []))
        return ([p['id'] for p in self.bundle.pulses if hit(p)],
                [e['id'] for e in self.bundle.events if hit(e)])

    def _code_line_clicked(self, key, line):
        ids, eids = self._items_from_line(key, line)
        if ids is None:
            return
        if not ids and not eids:
            self.clear_selection()
            self.hover_label.setText(f"line {line}: emitted nothing")
            return
        self.set_selection(ids, eids, current=ids[0] if ids else None,
                           source='code', code_line=(key, line))
        if self.snap_view:
            self._bring_into_view(ids, eids)
        self.hover_label.setText(f"line {line}: {len(ids)} pulse(s), "
                                 f"{len(eids)} event(s)")

    def _code_line_double_clicked(self, key, line):
        ids, eids = self._items_from_line(key, line)
        if ids or eids:
            self._fit_items(ids, eids)

    # --- moving the view to a selection

    def _spans(self, pids=(), eids=()):
        ps = [self.pulse_by_id[i] for i in pids]
        es = [self.event_by_id[i] for i in eids]
        return ([(p['t0'], p['t1'], p.get('shot')) for p in ps]
                + [(e['t'], e['t'], e.get('shot')) for e in es])

    def _nearest_group(self, spans):
        """The spans in the shot of the one nearest the view centre (a line
        in the shot body emits once per shot: show one shot's worth)."""
        x0, x1 = self.view_range()
        c = 0.5 * (x0 + x1)
        near = min(spans, key=lambda s: 0. if s[0] <= c <= s[1]
                   else min(abs(s[0] - c), abs(s[1] - c)))
        if near[2] is None:
            return [near]
        return [s for s in spans if s[2] == near[2]]

    def _fit_span(self, a, b):
        w = max(b - a, 16.)
        self.set_x_range(a - 0.15 * w, b + 0.15 * w)

    def _fit_items(self, pids=(), eids=(), group=True):
        spans = self._spans(pids, eids)
        if spans:
            grp = self._nearest_group(spans) if group else spans
            self._fit_span(min(s[0] for s in grp), max(s[1] for s in grp))

    def _bring_into_view(self, pids=(), eids=()):
        """Leave the view alone when any of the items is on screen; else
        centre the nearest shot's worth of them at the current zoom, or
        zoom out to fit them when they do not fit."""
        spans = self._spans(pids, eids)
        if not spans:
            return
        x0, x1 = self.view_range()
        if any(a <= x1 and b >= x0 for a, b, _ in spans):
            return
        grp = self._nearest_group(spans)
        a, b = min(s[0] for s in grp), max(s[1] for s in grp)
        w = x1 - x0
        if b - a <= 0.8 * w:
            c = 0.5 * (a + b)
            self.set_x_range(c - 0.5 * w, c + 0.5 * w)
        else:
            self._fit_span(a, b)

    # --- pulse table

    def _proxy_index(self, pid):
        row = self._row_of.get(pid) if hasattr(self, '_row_of') else None
        if row is None:
            return QModelIndex()
        return self.table_proxy.mapFromSource(self.table_model.index(row, 0))

    def _sync_table_selection(self, pids, current=None):
        """Show the selection in the table, without it echoing back."""
        if not hasattr(self, 'table_model'):
            return
        rows = sorted(ix.row() for ix in map(self._proxy_index, pids) if ix.isValid())
        sel = QItemSelection()
        ncol = self.table_proxy.columnCount() - 1
        k = 0
        while k < len(rows):            # contiguous runs -> one range each
            j = k
            while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1:
                j += 1
            sel.select(self.table_proxy.index(rows[k], 0),
                       self.table_proxy.index(rows[j], ncol))
            k = j + 1
        sm = self.table.selectionModel()
        F = QItemSelectionModel.SelectionFlag
        self._table_syncing = True
        try:
            sm.select(sel, F.ClearAndSelect | F.Rows)
            ix = self._proxy_index(current) if current is not None else QModelIndex()
            if ix.isValid():
                sm.setCurrentIndex(ix, F.NoUpdate)
                self.table.scrollTo(ix)
        finally:
            self._table_syncing = False

    def _table_selection_changed(self, *a):
        if self._table_syncing:
            return
        sm = self.table.selectionModel()
        rows = sorted(sm.selectedRows(), key=lambda ix: ix.row())
        pids = [self.bundle.pulses[self.table_proxy.mapToSource(ix).row()]['id']
                for ix in rows]
        cur = sm.currentIndex()
        current = None
        if cur.isValid() and sm.isRowSelected(cur.row(), QModelIndex()):
            current = self.bundle.pulses[self.table_proxy.mapToSource(cur).row()]['id']
        self.set_selection(pids, [], current=current, source='table')
        if current is not None and self.snap_view:
            self._bring_into_view([current])

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
        n_p, n_e = len(self.sel_pulses), len(self.sel_events)
        if n_p + n_e > 1:
            spans = self._spans(self.sel_pulses, self.sel_events)
            what = ', '.join(x for x in (f"{n_p} pulses" if n_p else '',
                                         f"{n_e} events" if n_e else '') if x)
            parts.append(f"selected: {what} "
                         f"[{fmt_time(min(s[0] for s in spans) - self.origin_t, u)} → "
                         f"{fmt_time(max(s[1] for s in spans) - self.origin_t, u)}]")
        elif self.selected is not None:
            p = self.pulse_by_id[self.selected]
            parts.append(f"selected: {p.get('text') or p.get('pulse_name')} "
                         f"[{fmt_time(p['t0'] - self.origin_t, u)} → "
                         f"{fmt_time(p['t1'] - self.origin_t, u)}]")
        elif self.selected_event is not None:
            e = self.event_by_id[self.selected_event]
            parts.append(f"selected: {e.get('label', e['kind'])} "
                         f"@ {fmt_time(e['t'] - self.origin_t, u)}")
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
        elif k in (Qt.Key.Key_Home, Qt.Key.Key_F) and not ctrl:
            self.fit_all() if k == Qt.Key.Key_F else self.fit_shot()
        elif k == Qt.Key.Key_Z:
            if len(self.sel_pulses) == 1 and not self.sel_events:
                self.zoom_to_pulse(self.sel_pulses[0])
            elif self.sel_pulses or self.sel_events:
                self._fit_items(self.sel_pulses, self.sel_events, group=False)
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
            self.set_gaps_visible(not self.show_gaps)
        elif k == Qt.Key.Key_L:
            self.set_labels_visible(not self.show_labels)
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
