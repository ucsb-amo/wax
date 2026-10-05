"""Magnetometer GUI (embedded in the dashboard): no network on the GUI thread,
no redraw while hidden, bounded Save-CSV log.

The client class is replaced by a fake before the window is built, so the
window never discovers or talks to the real server running on this PC.
"""

from __future__ import annotations

import threading
import time

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.util.guis.HMR_magnetometer import hmr_magnetometer_gui as gui_mod


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _spin(app, cond, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.002)
    app.processEvents()
    return cond()


class _FakeClient:
    """Stands in for HMRClient: no sockets, no discovery."""

    status_delay_s = 0.3
    reachable = True
    instances: list["_FakeClient"] = []

    def __init__(self, discovery_timeout=None, timeout=2.0):
        self.host, self.port = "127.0.0.1", 1
        self.status_calls = 0
        self.status_threads: set[int] = set()
        self.in_status = 0
        self.max_in_status = 0
        self.lock = threading.Lock()
        _FakeClient.instances.append(self)

    def _get_serial_status(self, timeout=2.0):
        with self.lock:
            self.status_calls += 1
            self.in_status += 1
            self.max_in_status = max(self.max_in_status, self.in_status)
            self.status_threads.add(threading.get_ident())
        try:
            time.sleep(self.status_delay_s)
            if not _FakeClient.reachable:
                raise ConnectionRefusedError("fake server down")
            return {"ok": True, "connected": True}
        finally:
            with self.lock:
                self.in_status -= 1

    def _get_since(self, timestamp_s, timeout=5.0):
        return {"ok": True, "readings": []}

    def _rediscover(self, timeout=2.0):
        return False


@pytest.fixture
def gui(qapp, monkeypatch):
    _FakeClient.instances = []
    _FakeClient.reachable = True
    monkeypatch.setattr(gui_mod, "HMRClient", _FakeClient)
    w = gui_mod.MagnetometerGUI(reference_csv_path=None)
    yield w
    w.serial_status_timer.stop()
    w.queue_timer.stop()
    w.shutdown()
    _spin(qapp, lambda: not w._probe_in_flight, timeout_s=3.0)
    w.deleteLater()
    qapp.processEvents()


def test_status_probe_runs_off_the_gui_thread_without_overlap(qapp, gui):
    _spin(qapp, lambda: gui.client is not None and not gui._probe_in_flight)
    client = gui.client
    calls_before = client.status_calls
    t0 = time.monotonic()
    gui._refresh_serial_status()
    gui._refresh_serial_status()   # still in flight: must not start another
    gui._refresh_serial_status()
    assert time.monotonic() - t0 < 0.15, "the GUI thread waited for the network"
    assert _spin(qapp, lambda: not gui._probe_in_flight)
    assert client.status_calls == calls_before + 1
    assert client.max_in_status == 1
    assert threading.get_ident() not in client.status_threads
    assert gui.serial_button.text() == "Serial: Connected"
    # The successful probe started the monitor worker.
    assert gui.running


def test_start_monitor_does_not_block_and_reports_an_unreachable_server(qapp, gui):
    _spin(qapp, lambda: gui.running)
    gui.stop_monitor()
    _FakeClient.reachable = False
    _spin(qapp, lambda: not gui._probe_in_flight)
    t0 = time.monotonic()
    gui.start_monitor()
    assert time.monotonic() - t0 < 0.15
    assert _spin(qapp, lambda: any("Connection failed" in l for l in gui.log_lines))
    assert not gui.running
    # The client object is kept (it is re-pointed by _rediscover), never
    # dropped from under a worker.
    assert gui.client is not None


def test_sync_settings_with_no_client(qapp, gui):
    gui.client = None
    gui.open_settings_window()
    gui._sync_settings()           # used to raise AttributeError on None
    gui._close_settings_window()


def _feed(gui, t_values):
    for t in t_values:
        gui.data_queue.put({"session_id": gui.session_id, "t": float(t),
                            "Bx": 0.1, "By": 0.2, "Bz": 0.3, "Btot": 0.374})


def test_save_csv_log_keeps_only_the_last_span(qapp, gui, monkeypatch):
    gui.queue_timer.stop()
    monkeypatch.setattr(gui_mod, "LOG_SPAN_S", 10.0)
    _feed(gui, [1000.0 + 0.1 * i for i in range(300)])     # 30 s of readings
    gui._process_queue()
    assert len(gui.log_t) == len(gui.log_x) == len(gui.log_btot)
    assert gui.log_t[-1] - gui.log_t[0] <= 10.0
    assert gui.log_t[-1] == pytest.approx(1000.0 + 29.9)


def test_save_csv_says_what_span_it_holds(qapp, gui):
    from PyQt6.QtGui import QAction
    texts = [a.text() for a in gui.findChildren(QAction)]
    save = [t for t in texts if t.startswith("Save CSV")]
    assert save == ["Save CSV (last 1 h)…"]


def test_hidden_plots_are_not_redrawn_or_restatted(qapp, gui, monkeypatch):
    gui.queue_timer.stop()
    calls = {"plot": 0, "stats": 0}
    real_plot, real_stats = gui._update_plot, gui._update_stats
    monkeypatch.setattr(gui, "_update_plot", lambda: (calls.__setitem__("plot", calls["plot"] + 1), real_plot()))
    monkeypatch.setattr(gui, "_update_stats", lambda: (calls.__setitem__("stats", calls["stats"] + 1), real_stats()))

    assert not gui._plots_visible()        # never shown
    _feed(gui, [time.time() - 1.0, time.time()])
    gui._process_queue()
    assert calls == {"plot": 0, "stats": 0}
    assert gui._plot_dirty

    gui.show()
    _spin(qapp, lambda: gui._plots_visible(), timeout_s=2.0)
    gui._process_queue()                     # no new data, but the plot is dirty
    assert calls["plot"] == 1 and calls["stats"] == 1
    gui._process_queue()                     # nothing new, stats fresh: nothing to do
    assert calls["plot"] == 1 and calls["stats"] == 1
    gui.hide()
