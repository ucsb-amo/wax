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


def test_lock_toggle_asks_before_turning_off(rig):
    panel, dev, srv, poll, answers, asked = rig
    poll()
    assert panel._lock_btn.text().startswith("Lock ON")
    answers["yes"] = False
    panel._lock_btn.click()
    assert dev.values["lock"] == 1 and asked == ["Turn the lock off?"]
    answers["yes"] = True
    panel._lock_btn.click()
    assert dev.values["lock"] == 0
    poll()
    assert panel._lock_btn.text().startswith("Lock OFF")
    panel._lock_btn.click()                              # turning on does not ask
    assert dev.values["lock"] == 1 and asked == ["Turn the lock off?"] * 2


def test_reset_setpoint_and_clear(rig):
    panel, dev, srv, poll, answers, asked = rig
    dev.values["sep"] = 6.437
    poll()
    panel._reset_btn.click()
    assert dev.values["setp"] == pytest.approx(6.437)
    assert "→ 6.437 ms" in panel._msg.text()
    dev.values["integ"] = 0.3
    answers["yes"] = False
    panel._clear_btn.click()
    assert dev.values["integ"] == 0.3
    answers["yes"] = True
    panel._clear_btn.click()
    assert dev.values["integ"] == 0.0


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
