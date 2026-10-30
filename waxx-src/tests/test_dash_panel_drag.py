"""Dashboard: a header drag only rearranges the dock - it never floats a panel
or pops it out; the pop-out button alone moves a panel into its own window.
Offscreen Qt; QSettings is redirected to an INI file in tmp_path (never the registry)."""
import os
import time

import pytest
from PyQt6.QtCore import QPoint, QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDockWidget, QLabel

from waxx.util.dashboard import dashboard_window as dw
from waxx.util.dashboard.panel_container import ClientPanel


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ini(tmp_path, monkeypatch):
    path = str(tmp_path / "dash.ini")
    real = QSettings
    monkeypatch.setattr(dw, "QSettings", lambda *_a, **_k: real(path, real.Format.IniFormat))
    monkeypatch.setattr(dw.DashboardMainWindow, "_probe_data_dir", lambda self: None)
    return path


def _pump(qapp, cond=lambda: False, rounds=300):
    for _ in range(rounds):
        qapp.processEvents()
        if cond():
            return
        time.sleep(0.01)


@pytest.fixture
def win(qapp, ini):
    panels = [
        dw.PanelPlacement(ClientPanel(pid, pid.upper(), lambda pid=pid: QLabel(pid)), area="right")
        for pid in ("alpha", "beta")
    ]
    w = dw.DashboardMainWindow("client", "Test dashboard", panels, host_ip="test")
    w.move(0, 0)
    w.resize(900, 600)
    w.show()
    _pump(qapp, lambda: w._layout_ready)
    assert w._layout_ready
    yield w
    for pid in list(w._popped):
        w.return_panel(w._by_id[pid])
    w._layout_ready = False
    w.hide()
    w.deleteLater()


def _drag_header(qapp, panel, offset: QPoint):
    hdr = panel.header()
    start = QPoint(40, hdr.height() // 2)          # on the title, clear of the buttons
    QTest.mousePress(hdr, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for i in range(1, 11):
        QTest.mouseMove(panel, hdr.mapTo(panel, start) + offset * i / 10)
        qapp.processEvents()
    QTest.mouseRelease(panel, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       panel.mapFromGlobal(panel.mapToGlobal(QPoint(0, 0))))
    qapp.processEvents()


def test_panels_are_not_floatable(qapp, win):
    for panel in win._panels:
        assert not panel.features() & QDockWidget.DockWidgetFeature.DockWidgetFloatable
        assert panel.features() & QDockWidget.DockWidgetFeature.DockWidgetMovable


def test_drag_out_of_dock_area_does_not_pop_out(qapp, win):
    panel = win._by_id["alpha"]
    _drag_header(qapp, panel, QPoint(-700, 300))
    _pump(qapp, lambda: not panel.isFloating())
    assert not panel.is_popped_out()
    assert not panel.isFloating() and panel.isVisible()   # re-docked, not stranded
    assert not win._popped


def test_popout_button_pops_out(qapp, win):
    panel = win._by_id["alpha"]
    panel.header()._popout_btn.click()
    _pump(qapp, lambda: panel.is_popped_out())
    assert panel.is_popped_out() and win._popped["alpha"].isVisible()
    win.return_panel(panel)
    assert not panel.is_popped_out() and not panel.isFloating() and panel.isVisible()


def test_double_click_does_not_float(qapp, win):
    panel = win._by_id["beta"]
    hdr = panel.header()
    QTest.mouseDClick(hdr, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      QPoint(40, hdr.height() // 2))
    _pump(qapp, rounds=10)
    assert not panel.isFloating() and not panel.is_popped_out()
