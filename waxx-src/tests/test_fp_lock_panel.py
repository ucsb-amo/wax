"""FP lock panel, offscreen, against the real server and the fake Quarto
(fakes/fake_fp_lock.py).  Confirmations are answered by the test; actions run
synchronously.  No real port, no discovery, no beacon (conftest)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QCheckBox, QLineEdit  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
import fake_fp_lock  # noqa: E402

from waxx.util.guis.fp_lock import fp_lock_server as fserver  # noqa: E402
from waxx.util.guis.fp_lock.fp_lock_client import FPLockClient  # noqa: E402
from waxx.util.guis.fp_lock.fp_lock_panel import FPLockPanel, SettingsDialog  # noqa: E402


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
    monkeypatch.setattr(fserver.FPLockLink, "BACKOFF_S", (0.05,))
    dev = fake_fp_lock.FakeQuarto()
    srv = fserver.FPLockServer("fp_lock_test", "COMFAKE35", host="127.0.0.1", port=0,
                               serial_factory=dev.open, log_dir=None)
    srv.start()
    assert _until(lambda: srv.link.connected and dev.values["fmt"] == 0
                  and srv.snapshot()["label"] is not None)
    client = FPLockClient.at("127.0.0.1", srv.port, "fp_lock_test")
    answers = {"yes": True}
    asked = []

    def confirm(title, text):
        asked.append(title)
        return answers["yes"]

    panel = FPLockPanel("fp_lock_test", title="Raman FP lock", client=client, confirm=confirm,
                        synchronous=True)
    panel._poller.stop()                    # the test polls by hand

    def poll():
        panel.apply_snapshot(panel._source.get_snapshot())

    yield panel, dev, srv, poll, answers, asked
    panel.cleanup()
    panel.deleteLater()
    srv.stop()


def test_display_states_warnings_and_separation_monitor(rig):
    panel, dev, srv, poll, _, _ = rig
    dev.emit_row(sep=6.40, state=2, tfirst=40.0, h1=9.0, h2=8.0, integ=-0.05)
    dev.emit_row(sep=6.42, state=2, tfirst=5.0, h1=9.0, h2=8.0, integ=-0.05)
    assert _until(lambda: srv.snapshot()["seq"] == 2)
    poll()
    assert panel._pill.text() == "LOCKED"
    assert "first peak at 5.00 ms" in panel._warn.text() and panel._warn.isVisibleTo(panel)
    assert panel._sep_big.text().startswith("sep 6.420 ms") and "setp 10.000" in panel._sep_big.text()
    assert panel._vals["t2"].text() == "11.42 ms"
    assert list(panel._hist["sep"]) == [6.40, 6.42]
    xs, ys = panel._plots["curves"]["sep"].getData()
    assert list(ys) == [6.40, 6.42]
    assert panel._plots["setp"].value() == pytest.approx(10.0)
    assert panel._bands[1].getRegion() == pytest.approx((110.0, 120.0))

    dev.emit_row(sep=6.42, state=1, tfirst=106.0, h1=9.0, h2=8.0)
    assert _until(lambda: srv.snapshot()["seq"] == 3)
    poll()
    assert panel._pill.text() == "HOLD"
    assert "second peak at 112.42 ms" in panel._warn.text()
    assert len(panel._hist["sep"]) == 3                 # only the new row was added


def test_plot_window(rig):
    panel, dev, srv, poll, _, _ = rig
    now = time.time()
    for t in (now - 120, now - 30, now - 1):
        panel._append_history({"t": t, "sep": 6.4, "err": 0.0, "feedback": 0.0, "integ": 0.0,
                               "tfirst": 40.0, "t2": 46.4})
    panel._redraw({}, None)
    xs, _ = panel._plots["curves"]["sep"].getData()
    assert len(xs) == 2 and min(xs) > -60                       # default window 60 s
    assert panel._plots["plot"].vb.viewRange()[0] == pytest.approx([-60, 0])
    panel._window_spin.setValue(300)
    xs, _ = panel._plots["curves"]["sep"].getData()
    assert len(xs) == 3
    assert panel._plots["plot"].vb.viewRange()[0] == pytest.approx([-300, 0])


def test_lock_toggle_does_not_ask(rig):
    panel, dev, srv, poll, answers, asked = rig
    poll()
    assert panel._lock_btn.text().startswith("Lock ON")
    panel._lock_btn.click()                              # turning off: no confirmation
    assert dev.values["lock"] == 0 and asked == []
    poll()
    assert panel._lock_btn.text().startswith("Lock OFF")
    panel._lock_btn.click()                              # turning on: no confirmation
    assert dev.values["lock"] == 1 and asked == []


def test_reset_setpoint_and_clear(rig):
    panel, dev, srv, poll, answers, asked = rig
    dev.values["sep"] = 6.437
    poll()
    panel._reset_btn.click()
    assert dev.values["setp"] == pytest.approx(6.437)
    assert "→ 6.437 ms" in panel._msg.text()
    asked.clear()
    dev.values["integ"] = 0.3
    panel._clear_btn.click()                             # clear: no confirmation
    assert dev.values["integ"] == 0.0 and asked == []


def test_override_button_toggles_the_override_setting(rig):
    panel, dev, srv, poll, answers, asked = rig
    panel._client.set_value("ovolt", "1.25")
    poll()
    assert panel._override_btn.text() == "Override"
    panel._override_btn.click()
    assert dev.values["override"] == 1 and asked == []
    poll()
    assert panel._override_btn.text() == "Override ON"
    assert "1.250 V" in panel._override_btn.toolTip()
    assert panel._vals["feedback"].text() == "+1.2500 V"     # ovolt shown in "out"
    panel._override_btn.click()
    assert dev.values["override"] == 0
    poll()
    assert panel._override_btn.text() == "Override"


def test_stale_pair_is_tinted_and_not_plotted(rig):
    panel, dev, srv, poll, answers, asked = rig
    dev.emit_row(sep=26.3, state=0, tfirst=29.8, h1=9.0, h2=7.0, nopair=0)
    dev.emit_row(sep=26.3, state=0, tfirst=29.8, h1=9.0, h2=7.0, nopair=60)
    assert _until(lambda: (srv.snapshot()["latest"] or {}).get("nopair") == 60)
    poll()
    assert "#b8860b" in panel._vals["err"].styleSheet()
    assert "#b8860b" in panel._vals["t2"].styleSheet()
    assert "no pair for 60 sweeps" in panel._warn.text()
    import math
    assert not math.isnan(panel._hist["sep"][-2]) and math.isnan(panel._hist["sep"][-1])
    assert math.isnan(panel._hist["t2"][-1]) and math.isnan(panel._hist["err"][-1])
    assert not math.isnan(panel._hist["feedback"][-1])                 # not a pair value
    dev.emit_row(sep=26.4, state=0, tfirst=29.8, h1=9.0, h2=7.0, nopair=0)
    assert _until(lambda: (srv.snapshot()["latest"] or {}).get("nopair") == 0)
    poll()
    assert "#b8860b" not in panel._vals["err"].styleSheet()


def test_override_shows_hold(rig):
    panel, dev, srv, poll, answers, asked = rig
    dev.emit_row(sep=6.40, state=2, tfirst=40.0, h1=9.0, h2=8.0)
    assert _until(lambda: (srv.snapshot()["latest"] or {}).get("state") == 2)
    poll()
    assert panel._pill.text() == "LOCKED"
    panel._client.set_value("ovolt", "0.5")
    panel._client.set_value("override", "1")
    dev.emit_row(sep=6.40, state=2, tfirst=40.0, h1=9.0, h2=8.0)   # the firmware reports HOLD
    assert _until(lambda: (srv.snapshot()["latest"] or {}).get("state") == 1)
    poll()
    assert panel._pill.text() == "HOLD"
    assert "override" in panel._warn.text() and "+0.500 V" in panel._warn.text()
    panel._client.set_value("override", "0")
    dev.emit_row(sep=6.40, state=2, tfirst=40.0, h1=9.0, h2=8.0)
    assert _until(lambda: (srv.snapshot()["latest"] or {}).get("state") == 2)
    poll()
    assert panel._pill.text() == "LOCKED"


def test_override_shows_hold_with_older_firmware(rig):
    panel, dev, srv, poll, answers, asked = rig
    dev.override_holds = False                           # older firmware: still reports LOCKED
    panel._client.set_value("override", "1")
    dev.emit_row(sep=6.40, state=2, tfirst=40.0, h1=9.0, h2=8.0)
    assert _until(lambda: (srv.snapshot()["latest"] or {}).get("state") == 2)
    poll()
    assert panel._pill.text() == "HOLD"


def test_out_turns_orange_beyond_8_volts_under_override(rig):
    panel, dev, srv, poll, answers, asked = rig
    panel._client.set_value("ovolt", "9.0")
    panel._client.set_value("override", "1")
    poll()
    assert "#b8860b" in panel._vals["feedback"].styleSheet()
    panel._client.set_value("ovolt", "7.5")
    poll()
    assert "#b8860b" not in panel._vals["feedback"].styleSheet()


def test_override_disables_the_lock_button(rig):
    panel, dev, srv, poll, answers, asked = rig
    poll()
    assert panel._lock_btn.isEnabled()
    panel._client.set_value("override", "1")
    poll()
    assert not panel._lock_btn.isEnabled()
    assert "disabled while override" in panel._lock_btn.toolTip()
    panel._client.set_value("override", "0")
    poll()
    assert panel._lock_btn.isEnabled()


def test_voltage_box_sends_ovolt_and_follows_the_board(rig):
    panel, dev, srv, poll, answers, asked = rig
    poll()
    panel._ovolt_spin.setValue(2.5)                      # starts the pause timer
    panel._ovolt_timer.stop()
    panel._send_ovolt()                                  # what the timer does after its pause
    assert dev.values["ovolt"] == pytest.approx(2.5)
    panel._client.set_value("ovolt", "-3.25")
    poll()
    assert panel._ovolt_spin.value() == pytest.approx(-3.25)


def test_console_up_recalls_last_commands(rig):
    panel, dev, srv, poll, _, _ = rig
    poll()                                               # enables the controls (COM connected)
    panel._console_btn.click()                           # show the console, as a user would
    line = panel._console_in
    for text in ("tmin 1.5", "help"):
        line.setText(text)
        line.returnPressed.emit()
    line.setText("partial")
    QTest.keyClick(line, Qt.Key.Key_Up)                  # last command
    assert line.text() == "help"
    QTest.keyClick(line, Qt.Key.Key_Up)
    assert line.text() == "tmin 1.5"
    QTest.keyClick(line, Qt.Key.Key_Up)                  # stops at the oldest
    assert line.text() == "tmin 1.5"
    QTest.keyClick(line, Qt.Key.Key_Down)
    assert line.text() == "help"
    QTest.keyClick(line, Qt.Key.Key_Down)
    assert line.text() == ""


def test_settings_dialog_sends_only_edits(rig):
    panel, dev, srv, poll, answers, asked = rig
    poll()
    dlg = SettingsDialog(panel._snap["settings"], [], panel)
    assert dlg.changes() == {}
    dlg.edit("p1").setText("-0.03")
    dlg.edit("hold").setChecked(False)
    assert isinstance(dlg.edit("hold"), QCheckBox) and isinstance(dlg.edit("vth"), QLineEdit)
    changes = dlg.changes()
    assert changes == {"p1": "-0.03", "hold": "0"}
    panel.apply_settings(changes)
    assert dev.values["p1"] == pytest.approx(-0.03) and dev.values["hold"] == 0
    panel.apply_settings({"nmin": "2.5"})
    assert "nmin" in panel._msg.text() and dev.values["nmin"] == 3
    dlg.deleteLater()


def test_console_and_link_loss(rig):
    panel, dev, srv, poll, answers, asked = rig
    poll()
    panel._console_in.setText("help")
    panel._console_in.returnPressed.emit()
    poll()
    assert "FP peak-separation lock" in panel._console.toPlainText()
    srv.link.disconnect(hold=True)
    poll()
    assert panel._pill.text().startswith("NO LINK") and not panel._reset_btn.isEnabled()
