"""Long-run upkeep of the Device Control GUI pieces:

* the composite panel's op sender keeps at most one status query per op in
  flight (the 1 s tick asked again for every waiting op, and with a hung
  monitor server the queries piled up without bound);
* refresh paths restyle a widget only when its style sheet changes
  (``setStyleSheet`` re-polishes on every call);
* the per-click menu is deleted after use.

The monitor client is a fake; nothing goes on the network.  Offscreen Qt."""
import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _spin(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return cond()


class FakeMonitorClient:
    """Only op_status; it blocks until ``gate`` is set."""

    def __init__(self, reply):
        self.reply = reply
        self.gate = threading.Event()
        self.asked = []

    def op_status(self, seq):
        self.asked.append(seq)
        self.gate.wait(5)
        return self.reply


def test_a_waiting_op_has_one_status_query_in_flight(qapp):
    from waxx.util.guis.composite_panel import _OpSender
    sender = _OpSender()
    client = FakeMonitorClient({"status": "ok", "state": "running"})
    sender._client = client
    got = []
    sender.status_replied.connect(lambda seq, reply: got.append(seq))
    for _ in range(30):                     # 30 ticks while the server hangs
        sender.query(7)
        sender.query(8)
    assert len(sender._jobs) == 2           # not 60
    sender.start()
    try:
        assert _spin(lambda: client.asked == [7])
        for _ in range(10):                 # still in flight: not queued again
            sender.query(7)
        assert [j for j in sender._jobs if j == ("status", 7)] == []
        client.gate.set()
        assert _spin(lambda: sorted(got) == [7, 8])
        sender.query(7)                     # answered: the next tick may ask again
        assert _spin(lambda: got.count(7) == 2)
    finally:
        client.gate.set()
        sender.stop()
        sender.wait(3000)


def test_an_unanswered_query_can_be_asked_again(qapp):
    from waxx.util.guis.composite_panel import _OpSender
    sender = _OpSender()
    client = FakeMonitorClient(None)        # no reply (server unreachable)
    client.gate.set()
    sender._client = client
    sender.start()
    try:
        sender.query(3)
        assert _spin(lambda: client.asked == [3] and not sender._status_in_flight)
        # The sender dropped its client on the failure; give it the fake back.
        sender._client = client
        sender.query(3)
        assert _spin(lambda: client.asked == [3, 3])
    finally:
        sender.stop()
        sender.wait(3000)


class CountingLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.restyles = 0

    def setStyleSheet(self, css):
        self.restyles += 1
        super().setStyleSheet(css)


def test_set_style_if_changed_skips_identical_styles(qapp):
    from waxx.util.guis.qt_upkeep import set_style_if_changed
    w = CountingLabel()
    assert set_style_if_changed(w, "color: red;")
    for _ in range(10):
        assert not set_style_if_changed(w, "color: red;")
    assert set_style_if_changed(w, "color: blue;")
    assert w.restyles == 2 and w.styleSheet() == "color: blue;"


def test_delete_later_tolerates_stand_ins(qapp):
    from waxx.util.guis.qt_upkeep import delete_later
    delete_later(object())                  # a test fake without deleteLater
    w = QLabel()
    delete_later(w)                         # deferred: still usable here
    assert w.text() == ""


def test_the_slm_pill_restyles_only_on_change(qapp):
    from waxx.util.guis.slm_pill import SlmPill
    pill = SlmPill()
    calls = []
    orig = pill.setStyleSheet
    pill.setStyleSheet = lambda css: (calls.append(css), orig(css))
    for _ in range(5):
        pill.refresh()
    assert len(calls) <= 1                  # the first may change it, no more
    pill.deleteLater()


def test_the_device_control_status_pill_restyles_only_on_change(qapp):
    from waxx.util.guis import device_control_gui as dc
    holder = type("H", (), {})()
    holder.status_pill = CountingLabel()
    for _ in range(5):
        dc.DeviceStateGUI._style_pill(holder, "#123456")
    assert holder.status_pill.restyles == 1
    dc.DeviceStateGUI._style_pill(holder, "#654321")
    assert holder.status_pill.restyles == 2
