"""Laser control panels embedded in the dashboard: no blocking network call
on the GUI thread for user actions, failures shown, no per-tick restyle,
bounded log view.

Both client classes are replaced by fakes before the windows are built: the
real servers on this PC are never discovered or contacted, and the fakes do
nothing to any hardware.
"""

from __future__ import annotations

import threading
import time

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.util.guis.als import als_control_gui as als_mod
from waxx.util.guis.precilaser import precilaser_control_gui as pre_mod


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


class _FakeAmpClient:
    host, port = "127.0.0.1", 1

    def __init__(self, *args, **kwargs):
        self.call_threads: list[int] = []

    def get_snapshot(self):
        return {
            "status": {"connection_state": "CONNECTED", "connected": True,
                       "power_enabled": False, "interlock_enabled": False,
                       "second_stage_enabled": False},
            "sequence": {"state": "IDLE"},
            "log_count": 0,
            "serial_port": "LRS-FAKE",
        }

    def get_logs_since(self, start_index):
        return {"messages": [], "next_index": start_index}

    def set_power_supply_on(self):
        self.call_threads.append(threading.get_ident())
        time.sleep(0.3)
        raise RuntimeError("lrs fake refused")


@pytest.fixture
def amp_gui(qapp, monkeypatch):
    monkeypatch.setattr(als_mod, "ALSGuiClient", _FakeAmpClient)
    w = als_mod.ALSControlGUI()
    yield w
    w.status_timer.stop()
    _spin(qapp, lambda: not w._remote_poll_in_flight and not w._remote_connect_in_flight, 3.0)
    w.close()
    w.deleteLater()
    qapp.processEvents()


def test_amp_action_runs_off_the_gui_thread_and_reports_failure(qapp, amp_gui):
    gui = amp_gui
    assert _spin(qapp, lambda: gui.status.connection_state == als_mod.ConnectionState.CONNECTED)
    client = gui.remote_client
    t0 = time.monotonic()
    gui._toggle_power_status()
    assert time.monotonic() - t0 < 0.1, "the GUI thread waited for the network"
    assert _spin(qapp, lambda: "lrs fake refused" in gui.statusBar().currentMessage())
    assert "failed" in gui.statusBar().currentMessage()
    assert client.call_threads and threading.get_ident() not in client.call_threads


def test_amp_status_pills_not_restyled_every_snapshot(qapp, amp_gui, monkeypatch):
    gui = amp_gui
    assert _spin(qapp, lambda: gui.status.connection_state == als_mod.ConnectionState.CONNECTED)
    gui.status_timer.stop()
    snap = _FakeAmpClient().get_snapshot()
    gui._apply_remote_snapshot(snap)
    widgets = [gui.power_status_dot, gui.interlock_status_dot, gui.second_stage_status_dot,
               gui.connect_button, gui.server_conn_button,
               *[s.indicator for s in gui.startup_steps + gui.shutdown_steps]]
    calls = []
    for w in widgets:
        real = w.setStyleSheet
        monkeypatch.setattr(w, "setStyleSheet", lambda css, _r=real: (calls.append(css), _r(css)))
    for _ in range(5):
        gui._apply_remote_snapshot(snap)
        gui._set_server_conn_button_state("connected")
    assert calls == []


class _FakeSeedClient:
    host, port = "127.0.0.1", 1

    def __init__(self, *args, **kwargs):
        pass

    def get_snapshot(self):
        return {"status": {"connection_state": "CONNECTED", "pd_ok": True,
                           "temperature_ok": True, "laser_enabled": False,
                           "power_stability_enabled": False},
                "sequence": {"state": "IDLE"}, "log_count": 0}

    def get_logs_since(self, start_index):
        return {"messages": [], "next_index": start_index}


def test_seed_log_view_is_bounded_and_pills_not_restyled(qapp, monkeypatch):
    monkeypatch.setattr(pre_mod, "PrecilaserGuiClient", _FakeSeedClient)
    gui = pre_mod.PrecilaserControlGUI()
    try:
        assert gui.log_text.maximumBlockCount() == 1000
        for i in range(1500):
            gui._append_log(f"lrs line {i}")
        assert gui.log_text.blockCount() <= 1000
        assert gui.log_text.toPlainText().splitlines()[-1] == "lrs line 1499"

        snap = _FakeSeedClient().get_snapshot()
        gui._apply_snapshot(snap)
        calls = []
        for w in (gui.pd_ok_dot, gui.temp_ok_dot, gui.laser_enable_dot,
                  gui.stability_dot, gui.serial_connect_button):
            real = w.setStyleSheet
            monkeypatch.setattr(w, "setStyleSheet", lambda css, _r=real: (calls.append(css), _r(css)))
        for _ in range(5):
            gui._apply_snapshot(snap)
        assert calls == []
    finally:
        gui.status_timer.stop()
        _spin(qapp, lambda: not gui._remote_poll_in_flight and not gui._remote_connect_in_flight, 3.0)
        gui.close()
        gui.deleteLater()
        qapp.processEvents()
