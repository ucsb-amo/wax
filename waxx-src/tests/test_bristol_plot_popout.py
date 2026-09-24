"""The Bristol client panel's detuning plot lives in a pop-out window.

Offscreen Qt only; the network poller is stubbed so no discovery traffic
is sent and the tests never touch a server.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

import waxx.util.guis.bristol.bristol_wavemeter_client_gui as mod  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _NoServer:
    """Stand-in for the GUI client: discovery always fails, fast."""

    def __init__(self, *a, **k):
        raise RuntimeError("no server in tests")


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setattr(mod, "BristolWavemeterGuiClient", _NoServer)
    w = mod.BristolDetuningWidget()
    w._timer.stop()  # drive _update() by hand
    yield w
    w.stop()
    qapp.processEvents()


def test_panel_has_no_collapsible_plot_and_window_starts_hidden(widget):
    from waxx.util.dashboard.widgets import CollapsibleGroupBox

    assert widget.findChildren(CollapsibleGroupBox) == []
    # The plot is not inside the panel at all ...
    assert widget._plot not in widget.findChildren(type(widget._plot))
    # ... it is in a parentless top-level window that is not yet shown.
    win = widget._plot_win
    assert win is not None
    assert win.parent() is None
    assert win.isWindow()
    assert not win.isVisible()
    assert win.centralWidget().isAncestorOf(widget._plot)


def test_plot_button_shows_the_window_with_the_recorded_history(widget, qapp):
    f0 = widget._f0_spin.value()
    widget._set_state(f0 + 1e-3, reachable=True)  # +1.000 GHz
    widget._update()
    widget._update()
    assert len(widget._detunings_ghz) == 2

    widget._plot_btn.click()
    qapp.processEvents()
    win = widget._plot_win
    assert win.isVisible()

    x, y = widget._curve.getData()
    assert len(y) == 2
    assert y[0] == pytest.approx(1.0, abs=1e-6)


def test_closing_the_window_hides_it_and_keeps_history(widget, qapp):
    widget._set_state(widget._f0_spin.value(), reachable=True)
    widget._update()
    widget.show_plot_window()
    qapp.processEvents()
    win = widget._plot_win

    win.close()
    qapp.processEvents()
    assert not win.isVisible()
    assert widget._plot_win is win          # not destroyed
    assert len(widget._detunings_ghz) == 1  # history intact

    # More samples arrive while hidden: recorded, and drawn on re-open.
    widget._update()
    widget.show_plot_window()
    qapp.processEvents()
    assert win.isVisible()
    _, y = widget._curve.getData()
    assert len(y) == 2


def test_stop_closes_the_window_for_real(widget, qapp):
    widget.show_plot_window()
    qapp.processEvents()
    win = widget._plot_win

    widget.stop()
    qapp.processEvents()
    assert widget._plot_win is None
    assert not win.isVisible()
    # A late click after teardown is a no-op rather than a crash.
    widget.show_plot_window()


def test_panel_cleanup_stops_widget_and_closes_window(qapp, monkeypatch):
    monkeypatch.setattr(mod, "BristolWavemeterGuiClient", _NoServer)
    from kexp.util.guis.wavemeter_monitor.bristol.bristol_panel import BristolServerPanel

    panel = BristolServerPanel()
    w = panel._gui._widget
    w.show_plot_window()
    qapp.processEvents()
    win = w._plot_win
    assert win.isVisible()

    panel.cleanup()
    qapp.processEvents()
    assert w._stop_event.is_set()
    assert w._plot_win is None
    assert not win.isVisible()
    assert not w._timer.isActive()
