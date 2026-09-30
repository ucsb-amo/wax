"""The Bristol client panel's detuning plot: inline (▸/▾) or popped out (⧉).

Offscreen Qt only; the network poller is stubbed so no discovery traffic
is sent and the tests never touch a server.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
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


def _in_window(widget) -> bool:
    return widget._plot_win.centralWidget().isAncestorOf(widget._plot)


def _in_panel(widget) -> bool:
    return widget.isAncestorOf(widget._plot)


def test_plot_starts_embedded_and_collapsed(widget):
    widget.show()
    assert _in_panel(widget)
    assert not widget._plot.isVisible()
    assert widget._plot_toggle.arrowType() == Qt.ArrowType.RightArrow
    win = widget._plot_win
    assert win is not None
    assert win.parent() is None
    assert win.isWindow()
    assert not win.isVisible()


def test_arrow_expands_plot_inline_with_history(widget, qapp):
    widget.show()
    widget._set_state(widget._f0_spin.value() + 1e-3, reachable=True)  # +1 GHz
    widget._update()
    widget._update()

    widget._plot_toggle.click()
    qapp.processEvents()
    assert widget._plot_toggle.arrowType() == Qt.ArrowType.DownArrow
    assert _in_panel(widget)
    assert widget._plot.isVisible()
    assert not widget._plot_win.isVisible()
    _, y = widget._curve.getData()
    assert len(y) == 2
    assert y[0] == pytest.approx(1.0, abs=1e-6)

    widget._plot_toggle.click()
    qapp.processEvents()
    assert not widget._plot.isVisible()


def test_popout_button_moves_plot_into_window(widget, qapp):
    widget.show()
    widget._set_state(widget._f0_spin.value() + 1e-3, reachable=True)
    widget._update()
    widget._update()

    widget._popout_btn.click()
    qapp.processEvents()
    win = widget._plot_win
    assert win.isVisible()
    assert _in_window(widget) and not _in_panel(widget)
    assert widget._plot.isVisible()
    assert not widget._plot_toggle.isEnabled()  # arrow is moot while popped out

    _, y = widget._curve.getData()
    assert len(y) == 2
    assert y[0] == pytest.approx(1.0, abs=1e-6)


def test_popout_button_again_docks_plot_back(widget, qapp):
    widget.show()
    widget._popout_btn.click()
    qapp.processEvents()
    win = widget._plot_win

    widget._popout_btn.click()
    qapp.processEvents()
    assert not win.isVisible()
    assert widget._plot_win is win
    assert _in_panel(widget)
    assert widget._plot.isVisible()  # docked back expanded
    assert widget._plot_toggle.isEnabled()
    assert widget._plot_toggle.arrowType() == Qt.ArrowType.DownArrow


def test_closing_window_docks_plot_back_and_keeps_history(widget, qapp):
    widget.show()
    widget._set_state(widget._f0_spin.value(), reachable=True)
    widget._update()
    widget.show_plot_window()
    qapp.processEvents()
    win = widget._plot_win

    win.close()
    qapp.processEvents()
    assert not win.isVisible()
    assert widget._plot_win is win          # not destroyed
    assert _in_panel(widget)
    assert widget._plot.isVisible()
    assert len(widget._detunings_ghz) == 1  # history intact

    # More samples arrive; recorded, and drawn on the next pop-out.
    widget._update()
    widget.show_plot_window()
    qapp.processEvents()
    assert win.isVisible()
    assert _in_window(widget)
    _, y = widget._curve.getData()
    assert len(y) == 2


def test_stop_closes_the_window_for_real_and_keeps_the_plot(widget, qapp):
    widget.show_plot_window()
    qapp.processEvents()
    win = widget._plot_win

    widget.stop()
    qapp.processEvents()
    assert widget._plot_win is None
    assert not win.isVisible()
    # The plot came back to the panel, so a late _update() cannot hit a
    # deleted widget; a late pop-out click is a no-op rather than a crash.
    assert _in_panel(widget)
    widget._update()
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
