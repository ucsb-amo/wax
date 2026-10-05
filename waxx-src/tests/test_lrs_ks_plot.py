"""Supply-current history plot: one redraw per snapshot, visible window only.

It used to redraw once per supply per 250 ms snapshot, each time handing
pyqtgraph the whole hour of history (~14,400 points a curve) and scanning
all of it in Python for the y range.  The window's client is never built:
snapshots are applied directly.  No network.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.util.guis.keysight import keysight_client_gui as ks


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch):
    # Keep the window from ever trying to find the server.
    monkeypatch.setattr(ks.KeysightClientWindow, "_refresh", lambda self: None)
    w = ks.KeysightClientWindow()
    w._timer.stop()
    w.show()
    qapp.processEvents()
    yield w
    w.close()
    w.deleteLater()
    qapp.processEvents()


def _snapshot(t, seq, i170=12.5, i500=80.0):
    return [
        {"ip": "10.0.0.1", "max_current": 170, "connected": True, "output_on": True,
         "status": 0, "current_a": i170, "t": t, "seq": seq},
        {"ip": "10.0.0.2", "max_current": 500, "connected": True, "output_on": True,
         "status": 0, "current_a": i500, "t": t, "seq": seq},
    ]


def test_one_redraw_per_snapshot_with_two_supplies(qapp, window, monkeypatch):
    plot = window._plot
    counts = {"redraw": 0}
    real = plot._redraw

    def _counting():
        counts["redraw"] += 1
        real()

    monkeypatch.setattr(plot, "_redraw", _counting)
    for k in range(5):
        window._apply(_snapshot(1_000_000.0 + k, seq=k))
    assert counts["redraw"] == 5


def test_only_the_visible_window_is_handed_to_pyqtgraph(qapp, window):
    plot = window._plot
    t0 = 2_000_000.0
    n = int(ks.PLOT_RANGE_MAX_S * 4)                 # one hour at 4 Hz
    window._apply(_snapshot(t0, seq=0, i170=400.0))   # creates the rows/curves
    for k in range(1, n - 1):
        # Old samples carry a huge current; only the last range_s are visible.
        i170 = 400.0 if k < n - 4 * 300 else 12.5
        plot.add_sample("10.0.0.1", 170, t0 + 0.25 * k, i170, k)
        plot.add_sample("10.0.0.2", 500, t0 + 0.25 * k, 80.0, k)
    window._apply(_snapshot(t0 + 0.25 * (n - 1), seq=n - 1))
    t_last = t0 + 0.25 * (n - 1)
    range_s = plot.range_s
    assert range_s == ks.PLOT_RANGE_DEFAULT_S
    for ip, curve in plot._curves.items():
        xs, _ys = curve.getOriginalDataset()
        assert len(plot._t[ip]) > 3 * len(xs)    # the buffer holds far more
        # From one sample before the window start to the newest sample.
        assert xs[-1] == pytest.approx(t_last)
        assert xs[1] >= t_last - range_s
        assert xs[0] < t_last - range_s
    # y range comes from the visible window (old 400 A samples are out of it).
    (_x_lo, _x_hi), (y_lo, y_hi) = plot._plot.getViewBox().viewRange()
    assert y_lo == pytest.approx(0.0)
    assert y_hi == pytest.approx(max(ks.PLOT_Y_MAX_FLOOR_A / 1.05, 80.0) * 1.05)


def test_value_button_restyled_only_on_change(qapp, window, monkeypatch):
    window._apply(_snapshot(3_000_000.0, seq=1))
    row = window._rows["10.0.0.1"]
    calls = []
    real = row.value_btn.setStyleSheet
    monkeypatch.setattr(row.value_btn, "setStyleSheet", lambda css: (calls.append(css), real(css)))
    for k in range(2, 6):
        window._apply(_snapshot(3_000_000.0 + k, seq=k, i170=12.0 + k))
    assert calls == []
    window._apply(_snapshot(3_000_010.0, seq=10, i170=999.0))   # alert: red
    assert len(calls) == 1 and "red" in calls[0]
