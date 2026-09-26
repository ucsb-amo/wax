"""Offline tests for waxx.util.seqview: bundle schema, edge encoding,
launcher plumbing, and an offscreen window smoke test (hover, select,
cursors, code linking, reload in place).

Run from the workspace root with the root venv:
    .venv/Scripts/python.exe -m pytest wax/waxx-src/tests/test_seqview.py
"""

import json
import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from waxx.util.seqview.bundle import (Bundle, empty_meta, edges_from_samples,  # noqa: E402
                                      level_at)
from waxx.util.seqview import launch                                           # noqa: E402


def synthetic_bundle(t_end=100_000.):
    """A small hand-made bundle: two semantic lanes, a digital and an
    analog lane, three pulses, one event, two shots."""
    m = empty_meta('synthetic')
    m['t_end_ns'] = t_end
    m['lanes'] = [
        {'id': 'steps', 'label': 'steps', 'kind': 'semantic', 'group': 'semantic',
         'color': '#d9d9d9', 'element': None, 'port': None, 'role': None,
         'note': '', 'high_label': '', 'low_label': '', 'samples': None,
         'edges': None, 'level0': 0, 'pulses': [0], 'events': [],
         'visible': True, 'height': 1.0},
        {'id': 'light:x', 'label': 'x light ON', 'kind': 'semantic', 'group': 'semantic',
         'color': '#e69f00', 'element': 'x_switch', 'port': None, 'role': 'x',
         'note': '', 'high_label': '', 'low_label': '', 'samples': None,
         'edges': None, 'level0': 0, 'pulses': [1, 2], 'events': [],
         'visible': True, 'height': 1.0},
        {'id': 'D1', 'label': 'D1 x_switch', 'kind': 'digital', 'group': 'physical',
         'color': '#888888', 'element': 'x_switch', 'port': '1', 'role': 'x',
         'note': 'HIGH = blocked', 'high_label': 'BLOCK', 'low_label': 'pass',
         'samples': None, 'edges': 'edges:D1', 'level0': 0, 'pulses': [],
         'events': [], 'visible': True, 'height': 0.6},
        {'id': 'A1', 'label': 'A1 drive', 'kind': 'analog', 'group': 'physical',
         'color': '#f0e442', 'element': 'drive', 'port': '1', 'role': 'x',
         'note': '', 'high_label': '', 'low_label': '', 'samples': 'analog:A1',
         'edges': None, 'level0': 0, 'pulses': [], 'events': [0],
         'visible': True, 'height': 0.9},
    ]
    def pulse(i, lane, t0, t1, **kw):
        p = {'id': i, 'lane': lane, 'element': 'x_switch', 'op': 'pass',
             'pulse_name': 'pass_pulse', 't0': t0, 't1': t1, 'shot': 0,
             'phase': 'body', 'macro': 'x_pulse', 'label': 't_x', 'value': (t1 - t0) * 1e-9,
             'value_str': f't_x = {(t1 - t0) / 1e3:g} µs', 'requested_ns': t1 - t0,
             'actual_ns': t1 - t0, 'rounded': False, 'src_line': 12,
             'src_lines': [12], 'qua_line': 5, 'note': '', 'data_key': '',
             'kind': 'exposure', 'color': '#e69f00', 'text': f'x · {(t1 - t0) / 1e3:g} µs',
             'truncated': False}
        p.update(kw)
        return p
    m['pulses'] = [pulse(0, 'steps', 1000., 9000., kind='play', text='step'),
                   pulse(1, 'light:x', 1000., 5000.),
                   pulse(2, 'light:x', 8000., 9000., shot=1, src_line=13, src_lines=[13])]
    m['events'] = [{'id': 0, 'lane': 'A1', 't': 6000., 'kind': 'reset_if_phase',
                    'label': 'phase reset', 'detail': '', 'src_line': 11,
                    'src_lines': [11], 'qua_line': 4, 'approx': False, 'shot': 0}]
    m['shots'] = [{'index': 0, 't0': 0., 't1': 7000., 'label': 'shot 0', 'params': {}},
                  {'index': 1, 't0': 7000., 't1': 12000., 'label': 'shot 1', 'params': {}}]
    m['framing'] = [{'t0': 0., 't1': 1000., 'shot': 0, 'kind': 'handoff', 'label': 'h'}]
    src = '\n'.join(f'line {i}' for i in range(1, 30))
    m['sources'] = {'seq': {'title': 'seq.py', 'path': '', 'text': src,
                            'first_line': 1, 'language': 'python'},
                    'qua': {'title': 'QUA', 'path': '', 'text': src,
                            'first_line': 1, 'language': 'python'}}
    m['params'] = {'names': ['t_x'], 'columns': {'t_x': [4e-6, 1e-6]},
                   'display': {'t_x': ['4 µs', '1 µs']}}
    m['warnings'] = [{'level': 'warning', 'text': 'a warning'}]
    edges = np.array([1000., 5000., 8000., 9000.])
    t = np.arange(int(t_end))
    analog = (0.25 * np.sin(2 * np.pi * 0.08 * t)).astype(np.float32)
    return Bundle(m, {'edges:D1': edges, 'analog:A1': analog}).validate()


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

def test_edges_from_samples_and_level_at():
    s = np.array([0, 0, 1, 1, 1, 0, 1])
    level0, idx = edges_from_samples(s)
    assert level0 == 0 and list(idx) == [2, 5, 6]
    assert level_at(idx, level0, 0) == 0
    assert level_at(idx, level0, 3) == 1
    assert level_at(idx, level0, 5) == 0
    assert level_at(idx, level0, 6.5) == 1


def test_bundle_validate_catches_dangling_ids():
    b = synthetic_bundle()
    b.meta['pulses'][0]['lane'] = 'nope'
    with pytest.raises(ValueError):
        b.validate()
    b = synthetic_bundle()
    b.meta['lanes'][2]['edges'] = 'missing'
    with pytest.raises(ValueError):
        b.validate()


def test_bundle_save_load(tmp_path):
    b = synthetic_bundle()
    path = tmp_path / 'x.npz'
    b.save(path)
    back = Bundle.load(path)
    assert back.meta == json.loads(json.dumps(b.meta))
    assert np.array_equal(back.arrays['edges:D1'], b.arrays['edges:D1'])
    assert back.arrays['analog:A1'].dtype == np.float32
    assert not [f for f in os.listdir(tmp_path) if f.startswith('.seqview-')]


# ---------------------------------------------------------------------------
# launcher plumbing (no process is started)
# ---------------------------------------------------------------------------

def test_send_to_viewer_without_server(tmp_path, monkeypatch):
    monkeypatch.setenv('WAXX_SEQVIEW_DIR', str(tmp_path))
    assert launch.read_port() == (None, None)
    assert launch.send_to_viewer(str(tmp_path / 'b.npz')) is False
    launch.write_port(1)          # nothing listens on port 1
    assert launch.send_to_viewer(str(tmp_path / 'b.npz')) is False
    launch.clear_port()
    assert launch.read_port() == (None, None)


def test_show_writes_bundle_and_spawns(tmp_path, monkeypatch):
    monkeypatch.setenv('WAXX_SEQVIEW_DIR', str(tmp_path))
    spawned = []
    monkeypatch.setattr(launch, 'spawn_viewer', lambda p: spawned.append(p))
    path = launch.show(synthetic_bundle(), name='syn thetic')
    assert os.path.exists(path) and path.endswith('.npz')
    assert spawned == [path]
    assert 'syn_thetic' in os.path.basename(path)


# ---------------------------------------------------------------------------
# window (offscreen)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_window_smoke(qapp):
    from PyQt6.QtCore import QPointF
    from waxx.util.seqview.window import SeqViewWindow
    b = synthetic_bundle()
    w = SeqViewWindow(b, serve=False)
    w.resize(1200, 700)
    w.show()
    qapp.processEvents()
    # semantic lanes shown, physical hidden by default
    shown = [ln.id for ln in w.lanes if w._lane_shown(ln)]
    assert shown == ['steps', 'light:x']
    w.set_physical_visible(True)
    qapp.processEvents()
    assert [ln.id for ln in w.lanes if w._lane_shown(ln)] == ['steps', 'light:x', 'D1', 'A1']
    # initial view: shot 0
    x0, x1 = w.view_range()
    assert x0 < 0. and 7000. < x1 < 9000.
    # select a pulse: code pane highlights its line, related = same line
    w.select_pulse(1)
    assert w.code.views['seq'].current == {12}
    assert w.lane_by_id['light:x'].bars.selected == {1}
    assert w.lane_by_id['light:x'].bars.related == {0}     # steps bar shares line 12
    # cursors: snapped to the pulse edges
    w.place_cursor('A', 1002.)      # within 6 px of the 1000 edge
    w.place_cursor('B', 5000.)
    assert w.cursor_x == {'A': 1000., 'B': 5000.}
    assert 'Δ = 4 µs' in w.cursor_label.text()
    # hover readout over the pulse
    lane = w.lane_by_id['light:x']
    w._hover_pos = lane.vb.mapViewToScene(QPointF(3000., 0.5))
    w._do_hover()
    assert 't = ' in w.hover_label.text()
    assert 'x · 4 µs' in w._pulse_tooltip(1)
    # code line click selects every pulse from that line and, snap view on,
    # brings it on screen (pulse 2 is in shot 1, off screen)
    w._code_line_clicked('seq', 13)
    assert w.lane_by_id['light:x'].bars.selected == {2}
    assert w.code.views['seq'].current == {13}
    x0, x1 = w.view_range()
    assert x0 <= 8000. and 9000. <= x1
    # event tooltip
    assert 'phase reset' in w._event_tooltip(0)
    # analog value readout
    assert 'V' in w.lane_by_id['A1'].value_text(1234.)
    assert w.lane_by_id['D1'].value_text(3000.) == 'BLOCK'
    # reload keeps the view and the physical toggle
    w.set_x_range(2000., 6000.)
    w.select_pulse(1)
    w.load_bundle(b, keep_view=True)
    qapp.processEvents()
    x0, x1 = w.view_range()
    assert (round(x0), round(x1)) == (2000, 6000)
    assert w.show_physical is True
    assert w.selected == 1
    # zoom to shot 1
    w.fit_shot(1)
    x0, x1 = w.view_range()
    assert x0 < 7000. < 12000. < x1
    # warning badge
    assert 'warning' in w.warn_button.text()
    w.close()


# ---------------------------------------------------------------------------
# selection: pulse table, timeline, code pane
# ---------------------------------------------------------------------------

def _many_pulse_bundle():
    """synthetic_bundle plus eight short pulses on light:x in shot 0, each
    emitted from its own source line (20..27)."""
    b = synthetic_bundle()
    m = b.meta
    lane = m['lanes'][1]
    for k in range(8):
        pid = 3 + k
        m['pulses'].append(dict(m['pulses'][1], id=pid, t0=5200. + 200. * k,
                                t1=5300. + 200. * k, src_line=20 + k,
                                src_lines=[20 + k]))
        lane['pulses'].append(pid)
    return b.validate()


def _table_window(qapp):
    from PyQt6.QtCore import Qt
    from waxx.util.seqview.window import SeqViewWindow
    w = SeqViewWindow(_many_pulse_bundle(), serve=False)
    w.resize(1400, 900)
    w.show()
    w.table_dock.show()
    w.table.sortByColumn(2, Qt.SortOrder.AscendingOrder)     # by start time
    qapp.processEvents()
    return w


def _row_of_pid(w):
    px = w.table_proxy
    return {w.bundle.pulses[px.mapToSource(px.index(r, 0)).row()]['id']: r
            for r in range(px.rowCount())}


def _table_rows(w):
    return sorted(ix.row() for ix in w.table.selectionModel().selectedRows())


def _click_row(qapp, w, r, mods=None):
    """Click table row r; returns the selected rows after checking that the
    lanes show the same pulses."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    ix = w.table_proxy.index(r, 1)
    QTest.mouseClick(w.table.viewport(), Qt.MouseButton.LeftButton,
                     mods or Qt.KeyboardModifier.NoModifier,
                     w.table.visualRect(ix).center())
    qapp.processEvents()
    rows = _table_rows(w)
    row_of = _row_of_pid(w)
    assert sorted(row_of[p] for p in w.sel_pulses) == rows
    # every lane holds the whole selection and draws its own pulses of it
    assert w.lane_by_id['light:x'].bars.selected == set(w.sel_pulses)
    return rows


def test_table_selection_modifiers(qapp):
    """Click selects one row, Ctrl+click adds / removes one, Shift+click a
    range, Ctrl+Shift+click adds a range; the lanes follow every step."""
    from PyQt6.QtCore import Qt
    M = Qt.KeyboardModifier
    w = _table_window(qapp)
    assert _click_row(qapp, w, 1) == [1]
    assert _click_row(qapp, w, 3, M.ShiftModifier) == [1, 2, 3]
    assert _click_row(qapp, w, 5, M.ControlModifier) == [1, 2, 3, 5]
    assert _click_row(qapp, w, 7, M.ControlModifier | M.ShiftModifier) == [1, 2, 3, 5, 6, 7]
    assert _click_row(qapp, w, 2, M.ControlModifier) == [1, 3, 5, 6, 7]
    assert _click_row(qapp, w, 0) == [0]
    assert _click_row(qapp, w, 0, M.ControlModifier) == []
    assert w.lane_by_id['light:x'].bars.dim_others is False
    w.close()


def test_selection_sync_highlight_and_snap(qapp):
    from PyQt6.QtCore import Qt, QPointF
    w = _table_window(qapp)
    row_of = _row_of_pid(w)
    # a timeline selection shows in the table
    w.select_pulse(5)
    assert _table_rows(w) == [row_of[5]]
    # the code pane tints the emitting line with the lane colour
    assert w.code.views['seq'].current == {22}
    assert w.code.views['seq'].colors == {22: '#e69f00'}
    # Ctrl+click on the timeline adds, and again removes
    lane = w.lane_by_id['light:x']
    at = lane.vb.mapViewToScene(QPointF(3000., 0.5))          # pulse 1
    w._vb_clicked(lane.vb, at, Qt.KeyboardModifier.ControlModifier)
    assert w.sel_pulses == [5, 1]
    assert _table_rows(w) == sorted([row_of[5], row_of[1]])
    w._vb_clicked(lane.vb, at, Qt.KeyboardModifier.ControlModifier)
    assert w.sel_pulses == [5]
    # a plain click replaces the selection
    w._vb_clicked(lane.vb, at, Qt.KeyboardModifier.NoModifier)
    assert w.sel_pulses == [1]
    # snap off: a code click selects but leaves the view alone
    w.set_x_range(0., 3000.)
    w.set_snap_view(False)
    assert not w.act_snap.isChecked() and not w.snap_view
    w._code_line_clicked('seq', 13)                           # pulse 2, shot 1
    assert w.sel_pulses == [2]
    assert tuple(round(x) for x in w.view_range()) == (0, 3000)
    # snap on (from the toolbar): the same click brings it on screen
    w.act_snap.trigger()
    assert w.snap_view
    w._code_line_clicked('seq', 13)
    x0, x1 = w.view_range()
    assert x0 <= 8000. and 9000. <= x1
    # a line that emitted nothing clears the selection and the highlight
    w._code_line_clicked('seq', 2)
    assert w.sel_pulses == [] and w.code.views['seq'].current == set()
    assert _table_rows(w) == []
    w.close()


def test_code_click_vs_text_drag(qapp):
    """A click on a code line selects its pulses; a drag that selects text
    (to copy it) does not."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QTextCursor
    from PyQt6.QtTest import QTest
    w = _table_window(qapp)
    w.code.setCurrentWidget(w.code.views['seq'])
    v = w.code.views['seq']
    qapp.processEvents()
    got = []
    v.lineClicked.connect(got.append)

    def pos(line):
        return v.cursorRect(QTextCursor(v.document().findBlockByNumber(line - 1))).center()
    QTest.mouseClick(v.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     pos(22))
    assert got == [22]
    assert w.sel_pulses == [5]
    QTest.mousePress(v.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     pos(3))
    QTest.mouseMove(v.viewport(), pos(6))
    QTest.mouseRelease(v.viewport(), Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, pos(6))
    assert v.textCursor().hasSelection()
    assert got == [22] and w.sel_pulses == [5]
    w.close()


def test_reload_keeps_multiselection_and_toggles(qapp):
    w = _table_window(qapp)
    w.set_selection([4, 6, 2], current=6)
    w.set_gaps_visible(False)
    assert not w.act_gaps.isChecked() and not w.show_gaps
    w.act_physical.trigger()
    assert w.show_physical and w.act_physical.isChecked()
    w.load_bundle(_many_pulse_bundle(), keep_view=True)
    qapp.processEvents()
    assert w.sel_pulses == [4, 6, 2] and w.selected == 6
    row_of = _row_of_pid(w)
    assert _table_rows(w) == sorted(row_of[p] for p in (4, 6, 2))
    assert w.show_physical and w.act_physical.isChecked()
    assert not w.show_gaps and not w.act_gaps.isChecked()
    w.close()
