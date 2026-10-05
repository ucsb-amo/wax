"""SnapshotPoller: results handled on the GUI thread, throttle and recovery.

The poll runs on a QThreadPool thread.  Its result used to be handled there
too, so the 5th consecutive failure called QTimer.setInterval from the pool
thread; Qt refuses to restart a timer from another thread and polling stopped
for good.  A fake client fails, then recovers; the timer must stay active,
go 'normal -> throttled -> normal', and keep ticking.  No network.
"""

from __future__ import annotations

import threading
import time

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication, QLabel

from waxx.util.dashboard.restyle import set_style, set_tooltip
from waxx.util.dashboard.snapshot_poller import SnapshotPoller


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


class _FlakyClient:
    """Fails the first *n_fail* polls, then answers."""

    def __init__(self, n_fail: int):
        self.n_fail = n_fail
        self.calls = 0
        self.lock = threading.Lock()

    def get_snapshot(self):
        with self.lock:
            self.calls += 1
            n = self.calls
        if n <= self.n_fail:
            raise ConnectionRefusedError(f"poll {n} refused")
        return {"n": n}


class _FastPoller(SnapshotPoller):
    NORMAL_INTERVAL_MS = 20
    THROTTLED_INTERVAL_MS = 60


def test_poller_throttles_and_recovers_on_the_gui_thread(qapp):
    qt_messages = []

    def _record(_mode, _ctx, msg):
        qt_messages.append(msg)

    previous = qInstallMessageHandler(_record)
    client = _FlakyClient(n_fail=6)
    poller = _FastPoller(client, panel_id="lrs_test")
    main = threading.get_ident()
    handled_on = []
    snaps = []
    statuses = []
    intervals_seen = set()

    poller.conn_changed.connect(lambda s, _d: (statuses.append(s), handled_on.append(threading.get_ident())))
    poller.snapshot_received.connect(lambda s: (snaps.append(s), handled_on.append(threading.get_ident())))
    try:
        poller.start()
        # Throttled after the 5th failure.
        assert _spin(qapp, lambda: (intervals_seen.add(poller._timer.interval()) or
                                    poller._consecutive_failures >= 5))
        assert _spin(qapp, lambda: poller._timer.interval() == _FastPoller.THROTTLED_INTERVAL_MS)
        assert poller._timer.isActive()
        # Recovers: back to the normal interval, still active, still ticking.
        assert _spin(qapp, lambda: len(snaps) >= 1)
        assert poller._timer.interval() == _FastPoller.NORMAL_INTERVAL_MS
        assert poller._timer.isActive()
        n_after_recovery = len(snaps)
        assert _spin(qapp, lambda: len(snaps) >= n_after_recovery + 3)
    finally:
        poller.stop()
        _spin(qapp, lambda: not poller._in_flight, timeout_s=2.0)
        qInstallMessageHandler(previous)

    assert statuses[0] == "error" and "connected" in statuses
    assert poller._consecutive_failures == 0
    # Every handler ran on the GUI thread, and Qt complained about nothing
    # (e.g. "Timers cannot be started from another thread").
    assert set(handled_on) == {main}
    assert not [m for m in qt_messages if "thread" in m.lower()], qt_messages


def test_poller_late_result_after_delete_is_dropped(qapp):
    """A poll still running when the panel (and poller) is deleted must not
    raise in the pool thread."""
    gate = threading.Event()
    errors = []

    class _SlowClient:
        def get_snapshot(self):
            gate.wait(2.0)
            return {"ok": True}

    poller = _FastPoller(_SlowClient(), panel_id="lrs_late")
    old_hook = threading.excepthook
    threading.excepthook = lambda args: errors.append(args)
    try:
        poller._tick()
        poller.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        assert sip.isdeleted(poller)
        gate.set()
        _spin(qapp, lambda: False, timeout_s=0.3)
    finally:
        threading.excepthook = old_hook
    assert not errors


def test_poller_rediscovers_after_a_failed_poll(qapp):
    """A server restarted on a new port: the client follows its beacon."""

    class _MovedClient:
        def __init__(self):
            self.port = 1
            self.rediscovered = 0

        def get_snapshot(self):
            if self.port == 1:
                raise ConnectionRefusedError("old port")
            return {"port": self.port}

        def _rediscover(self, timeout):
            self.rediscovered += 1
            self.port = 2
            return True

    client = _MovedClient()
    poller = _FastPoller(client, panel_id="lrs_moved")
    snaps = []
    poller.snapshot_received.connect(snaps.append)
    try:
        poller.start()
        assert _spin(qapp, lambda: len(snaps) >= 1)
    finally:
        poller.stop()
        _spin(qapp, lambda: not poller._in_flight, timeout_s=2.0)
    assert client.rediscovered == 1
    assert snaps[0] == {"port": 2}


def test_restyle_helpers_only_set_on_change(qapp):
    label = QLabel()
    assert set_style(label, "color: red;") is True
    assert set_style(label, "color: red;") is False
    assert label.styleSheet() == "color: red;"
    assert set_style(label, "color: blue;") is True
    assert set_tooltip(label, "a") is True
    assert set_tooltip(label, "a") is False
    assert label.toolTip() == "a"
