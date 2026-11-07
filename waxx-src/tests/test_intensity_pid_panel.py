"""Intensity PID panel, offscreen, against the real server and the fake Quarto
(fakes/fake_intensity_pid.py).  Confirmations are answered by the test;
actions run synchronously.  No real port, no discovery, no beacon (conftest)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
import fake_intensity_pid  # noqa: E402

from waxx.util.guis.intensity_pid import intensity_pid_server as iserver  # noqa: E402
from waxx.util.guis.intensity_pid.intensity_pid_client import IntensityPIDClient  # noqa: E402
from waxx.util.guis.intensity_pid.intensity_pid_panel import GainsDialog, IntensityPIDPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


@pytest.fixture
def rig(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(iserver.IntensityPIDLink, "BACKOFF_S", (0.05,))
    monkeypatch.setattr(iserver, "PLOT_BLOCK_S", 0.0)
    dev = fake_intensity_pid.FakeIntensityQuarto()
    srv = iserver.IntensityPIDServer("ipid_test", "COMFAKE16", host="127.0.0.1", port=0,
                                     serial_factory=dev.open, log_dir=None,
                                     gains_path=str(tmp_path / "gains.json"),
                                     loop_names=("80+ AOM", "150+ AOM"))
    srv.start()
    assert _until(lambda: srv.link.connected and srv.snapshot()["label"] is not None)
    client = IntensityPIDClient.at("127.0.0.1", srv.port, "ipid_test")
    answers = {"yes": True}
    asked = []

    def confirm(title, text):
        asked.append(title)
        return answers["yes"]

    panel = IntensityPIDPanel("ipid_test", title="Raman intensity", loop_names=("80+ AOM", "150+ AOM"),
                              client=client, confirm=confirm, synchronous=True)
    panel._poller.stop()                    # the test polls by hand

    def poll():
        panel.apply_snapshot(panel._source.get_snapshot())

    yield panel, dev, srv, poll, answers, asked
    panel.cleanup()
    panel.deleteLater()
    srv.stop()


def _rows(dev, srv, n=3, **kw):
    start = srv.snapshot()["seq"]
    for _ in range(n + 1):
        dev.emit_row(**kw)
    assert _until(lambda: srv.snapshot()["seq"] >= start + n)


def test_readings_and_pill(rig):
    panel, dev, srv, poll, *_ = rig
    poll()
    assert panel._pill.text() == "NO DATA"
    _rows(dev, srv, meas1=0.5012, set1=0.5, meas2=1.2, set2=1.25, out1=3.2, out2=9.9)
    poll()
    assert panel._pill.text() == "ACTIVE"
    c = panel._cells
    assert c["meas1"].text() == "0.5012" and c["set1"].text() == "0.5000"
    assert c["err1"].text() == "+0.0012" and c["out1"].text() == "3.200"
    assert c["err2"].text() == "-0.0500"
    assert "color: #b8860b" in c["out2"].styleSheet()          # at the 9.9 V limit
    assert "upper limit" in panel._warn.text()
    assert "p1 -0.055" in panel._gains_line.text()
    assert len(panel._hist["t"]) >= 3
    assert panel._hold_btn.text() == "Hold" and panel._clear_btn.isEnabled()


def test_hold_release_and_clear_buttons(rig):
    panel, dev, srv, poll, *_ = rig
    _rows(dev, srv)
    poll()
    panel._hold_btn.click()
    assert dev.hold == 1 and dev.received[-1] == "hold 1"
    _rows(dev, srv)
    poll()
    assert panel._pill.text() == "HELD" and panel._hold_btn.text() == "Release"
    assert not panel._clear_btn.isEnabled()                    # a clear would be overwritten
    panel._hold_btn.click()
    assert dev.hold == 0 and dev.received[-1] == "hold 0"
    _rows(dev, srv)
    poll()
    assert panel._pill.text() == "ACTIVE" and panel._clear_btn.isEnabled()
    dev.integ = [1.0, 1.0]
    panel._clear_btn.click()
    assert dev.integ == [0.0, 0.0] and "integrators cleared" in panel._msg.text()


def test_gains_dialog_and_apply(rig):
    panel, dev, srv, poll, answers, asked = rig
    poll()
    dlg = GainsDialog(panel._snap["settings"], ("80+ AOM", "150+ AOM"))
    assert dlg.edit("p1").text() == "-0.055"
    assert dlg.changes() == {}
    dlg.edit("i2").setText("-0.004")
    assert dlg.changes() == {"i2": "-0.004"}
    answers["yes"] = False
    panel.apply_settings(dlg.changes())
    assert dev.gains["i2"] == -0.006 and asked == ["Apply gains?"]
    answers["yes"] = True
    panel.apply_settings(dlg.changes())
    assert dev.gains["i2"] == -0.004 and "set i2" in panel._msg.text()
    panel.apply_settings({"p1": "abc"})
    assert "p1" in panel._msg.text() and dev.gains["p1"] == -0.055


def test_override_row(rig):
    panel, dev, srv, poll, *_ = rig
    _rows(dev, srv)
    poll()
    assert panel._ov_btns[1].text() == "80+ AOM"
    panel._ov_spins[1].setValue(2.5)
    panel._ov_timers[1].stop()                     # the button sends the box's value itself
    panel._ov_btns[1].click()
    assert dev.override["override1"] == 1 and dev.ovolt["ovolt1"] == 2.5
    _rows(dev, srv)
    poll()
    assert panel._pill.text() == "OVERRIDE"
    assert panel._ov_btns[1].text() == "80+ AOM ON"
    assert "#b8860b" in panel._ov_btns[1].styleSheet()
    assert panel._cells["out1"].text() == "2.500" and "#b8860b" in panel._cells["out1"].styleSheet()
    assert "overridden" not in panel._warn.text()  # the row says it
    panel._ov_spins[1].setValue(3.0)
    panel._send_ovolt(1)                           # what the debounce timer calls
    assert dev.ovolt["ovolt1"] == 3.0 and dev.override["override1"] == 1
    poll()
    panel._ov_btns[1].click()
    assert dev.override["override1"] == 0
    _rows(dev, srv)
    poll()
    assert panel._pill.text() == "ACTIVE" and panel._ov_btns[1].text() == "80+ AOM"


def test_plot_window(rig):
    panel, dev, srv, poll, *_ = rig
    now = time.time()
    for t in (now - 120, now - 30, now - 1):
        panel._append_history({"t": t, "meas1": 0.5, "set1": 0.5, "meas2": 1.0, "set2": 1.0})
    panel._redraw()
    xs, _ = panel._plots["curves"]["meas2"].getData()
    assert len(xs) == 2 and min(xs) > -60                       # default window 60 s
    assert panel._plots["plot"].vb.viewRange()[0] == pytest.approx([-60, 0])
    panel._window_spin.setValue(300)
    xs, _ = panel._plots["curves"]["meas1"].getData()
    assert len(xs) == 3
    assert panel._plots["plot"].vb.viewRange()[0] == pytest.approx([-300, 0])


def test_no_link_disables_controls(rig):
    panel, dev, srv, poll, *_ = rig
    srv.link.disconnect(hold=True)
    poll()
    assert panel._pill.text() == "NO LINK"
    assert not panel._hold_btn.isEnabled() and not panel._settings_btn.isEnabled()
