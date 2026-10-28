"""ComLink: background connect with backoff, report_error, the hold, refusals,
beacon state.  Fake devices only -- no serial port is opened."""
from __future__ import annotations

import threading
import time

import pytest

from waxx.util.comms_server.com_link import ComLink, ComLinkNotConnected, ComLinkRefused


def _until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class FakeDev:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeServer:
    def __init__(self):
        self.states = []

    def set_com_state(self, state, port=None):
        self.states.append((state, port))


class FakeLink(ComLink):
    BACKOFF_S = (0.05, 0.05, 0.1)

    def __init__(self, fail_times=0, refusal=None, open_gate=None, **kw):
        super().__init__("COM99", baud=9600, **kw)
        self.fail_times = fail_times
        self.refusal = refusal
        self.open_gate = open_gate            # threading.Event the open waits on
        self.opens = 0
        self.devices = []
        self.connected_calls = 0

    def _open(self):
        self.opens += 1
        if self.open_gate is not None:
            self.open_gate.wait(5.0)
        if self.opens <= self.fail_times:
            raise OSError(f"could not open port 'COM99' (attempt {self.opens})")
        dev = FakeDev()
        self.devices.append(dev)
        return dev

    def _on_connected(self, device):
        self.connected_calls += 1

    def _disconnect_refusal(self):
        return self.refusal


@pytest.fixture
def links():
    made = []
    yield made
    for link in made:
        link.stop()


def test_start_returns_at_once_and_connects_in_the_background(links):
    gate = threading.Event()
    link = FakeLink(open_gate=gate)
    links.append(link)
    t0 = time.monotonic()
    link.start()
    assert time.monotonic() - t0 < 0.2, "start() must not wait for the port"
    assert _until(lambda: link.state == "connecting")
    assert link.device is None
    with pytest.raises(ComLinkNotConnected):
        link.require()
    gate.set()
    assert _until(lambda: link.state == "connected")
    assert link.require() is link.devices[0]
    assert link.connected_calls == 1


def test_failed_opens_retry_on_the_backoff_until_connected(links):
    server = FakeServer()
    link = FakeLink(fail_times=3, server=server)
    links.append(link)
    link.start()
    assert _until(lambda: link.state == "connected")
    assert link.opens == 4
    snap = link.snapshot()
    assert snap["status"] == "connected" and snap["last_error"] is None
    assert snap["reconnect_attempts"] == 0 and snap["held"] is False
    states = [s for s, _ in server.states]
    assert "error" in states and states[-1] == "connected"
    assert all(p == "COM99" for _, p in server.states)


def test_report_error_closes_and_reconnects(links):
    link = FakeLink()
    links.append(link)
    link.start()
    assert _until(lambda: link.state == "connected")
    first = link.device
    link.report_error(OSError("WriteFile failed"), where="poll")
    assert first.closed
    assert link.state in ("error", "connecting", "connected")
    assert _until(lambda: link.state == "connected" and link.device is not first)
    assert link.snapshot()["reconnects"] == 1
    assert link.connected_calls == 2


def test_a_device_that_fails_right_after_opening_backs_off_more_each_time(links):
    link = FakeLink()
    link.BACKOFF_S = (0.05, 0.2, 0.6)
    link.STABLE_S = 5.0
    links.append(link)
    link.start()
    waits = []
    for _ in range(3):
        assert _until(lambda: link.state == "connected")
        link.report_error(OSError("fails at once"))
        waits.append(link.snapshot()["next_retry_s"])
    assert waits[0] < waits[1] < waits[2], waits


def test_a_failure_after_a_good_run_starts_the_backoff_again(links):
    link = FakeLink()
    link.BACKOFF_S = (0.05, 5.0)
    link.STABLE_S = 0.1
    links.append(link)
    link.start()
    for _ in range(2):
        assert _until(lambda: link.state == "connected")
        time.sleep(0.15)                      # up longer than STABLE_S
        link.report_error(OSError("rare failure"))
        assert link.snapshot()["next_retry_s"] < 1.0


def test_report_error_twice_is_harmless(links):
    link = FakeLink()
    links.append(link)
    link.start()
    assert _until(lambda: link.state == "connected")
    link.report_error(OSError("x"))
    link.report_error(OSError("y"))          # device already gone: no-op
    assert _until(lambda: link.state == "connected")


def test_hold_stops_every_reconnect_until_released(links):
    link = FakeLink()
    links.append(link)
    link.start()
    assert _until(lambda: link.state == "connected")
    dev = link.device
    link.disconnect(hold=True)
    assert dev.closed and link.device is None
    assert link.state == "held" and link.held and link.snapshot()["held"]
    opens = link.opens
    link.report_error(OSError("late I/O failure"))   # must not schedule a reconnect
    time.sleep(0.4)                                    # several backoff periods
    assert link.opens == opens and link.state == "held"
    link.connect()
    assert _until(lambda: link.state == "connected")
    assert not link.held and link.opens == opens + 1


def test_disconnect_while_an_open_is_in_progress_ends_held(links):
    gate = threading.Event()
    link = FakeLink(open_gate=gate)
    links.append(link)
    link.start()
    assert _until(lambda: link.opens == 1)
    done = threading.Event()
    threading.Thread(target=lambda: (link.disconnect(hold=True), done.set()), daemon=True).start()
    time.sleep(0.05)
    gate.set()                                # the open finishes after the hold was set
    assert done.wait(3.0)
    assert link.state == "held" and link.device is None
    assert link.devices and link.devices[0].closed


def test_disconnect_does_not_wait_forever_behind_a_stuck_read(links):
    link = FakeLink()
    link.DISCONNECT_LOCK_WAIT_S = 0.1
    links.append(link)
    link.start()
    assert _until(lambda: link.state == "connected")
    dev = link.device
    reading = threading.Event()
    release = threading.Event()

    def stuck_read():                         # the server's I/O, holding the lock
        with link.io_lock:
            reading.set()
            release.wait(5.0)
    threading.Thread(target=stuck_read, daemon=True).start()
    assert reading.wait(1.0)
    t0 = time.monotonic()
    link.disconnect(hold=True)
    assert time.monotonic() - t0 < 1.0
    assert link.state == "held" and not dev.closed      # closed once the read ends
    release.set()
    assert _until(lambda: dev.closed and link.device is None)
    time.sleep(0.2)
    assert link.state == "held" and link.opens == 1      # and nothing reopened it


def test_refused_disconnect_leaves_the_device_open(links):
    link = FakeLink(refusal="magnets are enabled")
    links.append(link)
    link.start()
    assert _until(lambda: link.state == "connected")
    with pytest.raises(ComLinkRefused, match="magnets"):
        link.disconnect()
    assert link.state == "connected" and not link.device.closed
    link.disconnect(force=True)
    assert link.state == "held"


def test_connect_skips_the_backoff_wait(links):
    link = FakeLink(fail_times=1)
    link.BACKOFF_S = (5.0,)
    links.append(link)
    link.start()
    assert _until(lambda: link.state == "error")
    assert link.snapshot()["next_retry_s"] > 1.0
    link.connect()                            # a person's Connect: now, not in 5 s
    assert _until(lambda: link.state == "connected", timeout=1.5)


def test_no_auto_reconnect_gives_up_after_a_failure(links):
    link = FakeLink(fail_times=1, auto_reconnect=False)
    links.append(link)
    link.start()
    assert _until(lambda: link.state == "error")
    time.sleep(0.3)
    assert link.opens == 1
    link.connect()
    assert _until(lambda: link.state == "connected")


def test_stop_is_bounded_while_an_open_hangs():
    gate = threading.Event()
    link = FakeLink(open_gate=gate)
    link.STOP_TIMEOUT_S = 0.2
    link.start()
    assert _until(lambda: link.opens == 1)
    t0 = time.monotonic()
    link.stop()
    assert time.monotonic() - t0 < 1.5
    gate.set()
    assert _until(lambda: link.devices and link.devices[0].closed)
    assert link.device is None
