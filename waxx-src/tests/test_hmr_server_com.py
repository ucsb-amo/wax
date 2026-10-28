"""HMR magnetometer: the serial port through ComLink.

The sensor is a fake (HMR2300Reader replaced, serial.Serial made to fail the
test if touched); no beacon is sent (conftest); the server binds 127.0.0.1
port 0 only.  Covers: answering while the port fails, retry + connect, read
error / stuck sensor reconnects, the Disconnect hold (and the old race where a
disconnect landed between the read loop's check and its reopen), Connect,
the older commands, and the client's com_connect / com_disconnect.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

from waxx.util.guis.HMR_magnetometer import hmr_magnetometer_client as cli_mod
from waxx.util.guis.HMR_magnetometer import hmr_magnetometer_server as srv_mod


def _until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


class FakeSensor:
    """What the fake readers do: failed opens, a gate on open, read errors,
    stuck counts.  Counts opens and any read of a closed port."""

    def __init__(self, fail_opens=0):
        self.lock = threading.Lock()
        self.fail_opens = fail_opens
        self.opens = 0
        self.open_gate: threading.Event | None = None
        self.open_started = threading.Event()
        self.read_errors = 0
        self.stuck = False
        self.on_read = None             # called once inside read_one (serial lock held)
        self.reads = 0
        self.reads_on_closed = 0
        self.readers = []


def _reader_class(sensor: FakeSensor):
    class FakeReader:
        def __init__(self, port, baud, device_id, timeout=1.0):
            self.ser = None
            self.closed = False

        def open(self):
            with sensor.lock:
                sensor.opens += 1
                k = sensor.opens
            sensor.open_started.set()
            if sensor.open_gate is not None:
                sensor.open_gate.wait(5.0)
            if k <= sensor.fail_opens:
                raise OSError(f"could not open port 'LRS-FAKE' (fake attempt {k})")
            self.ser = object()
            sensor.readers.append(self)

        def setup(self):
            pass

        def close(self):
            self.closed = True
            self.ser = None

        def read_one(self):
            time.sleep(0.002)
            if self.closed:
                sensor.reads_on_closed += 1
            hook, sensor.on_read = sensor.on_read, None
            if hook is not None:
                hook()
            if sensor.read_errors > 0:
                sensor.read_errors -= 1
                raise OSError("ClearCommError failed (fake)")
            sensor.reads += 1
            if sensor.stuck:
                return (100, 200, 300)
            return (1000 + sensor.reads, -200, 300)

    return FakeReader


@pytest.fixture
def make_server(monkeypatch):
    def _no_serial(*a, **k):
        raise AssertionError("the test must not open a serial port")

    monkeypatch.setattr(srv_mod.serial, "Serial", _no_serial)
    monkeypatch.setattr(srv_mod, "_configure_logging", lambda log_path=None: None)
    made = []

    def _make(sensor: FakeSensor):
        monkeypatch.setattr(srv_mod, "HMR2300Reader", _reader_class(sensor))
        s = srv_mod.MagnetometerServer(serial_port="LRS-FAKE", poll_interval=0.005,
                                       server_host="127.0.0.1", server_port=0)
        s.link.BACKOFF_S = (0.05, 0.1)
        t = threading.Thread(target=s.run, daemon=True)
        made.append((s, t))
        t.start()
        assert _until(lambda: s._srv is not None and s.server_port != 0)
        return s

    yield _make
    for s, t in made:
        s.shutdown()
        t.join(timeout=3.0)


def _ask(s, command, timeout=3.0):
    return cli_mod._request(command, "127.0.0.1", s.server_port, timeout)


def _client(port):
    """HMRClient pointed at a port, without discovery (no UDP)."""
    c = cli_mod.HMRClient.__new__(cli_mod.HMRClient)
    c.host, c.port, c.timeout, c.last_read_failed = "127.0.0.1", port, 2.0, False
    return c


def _history_len(s):
    with s.history_lock:
        return len(s.history)


def test_server_answers_while_the_port_keeps_failing(make_server):
    sensor = FakeSensor(fail_opens=10**6)
    s = make_server(sensor)
    t0 = time.monotonic()
    assert _ask(s, "PING") == {"ok": True, "message": "pong"}
    assert time.monotonic() - t0 < 0.5
    assert _until(lambda: sensor.opens >= 3), "the link must keep retrying"
    snap = _ask(s, "GET_SNAPSHOT")
    assert snap["ok"] and snap["connected"] is False and snap["target_connected"] is True
    assert snap["com"]["status"] in ("error", "connecting")
    assert "could not open port" in (snap["com"]["last_error"] or "")
    assert snap["com"]["port"] == "LRS-FAKE" and snap["com"]["held"] is False
    assert _ask(s, "GET_FIELD") == {"ok": False, "error": "No data available yet"}
    assert sensor.readers == []


def test_retries_then_connects_and_reads(make_server):
    sensor = FakeSensor(fail_opens=2)
    s = make_server(sensor)
    assert _until(lambda: s.link.connected)
    assert sensor.opens == 3
    assert _until(lambda: _history_len(s) > 5)
    snap = _ask(s, "GET_SNAPSHOT")
    assert snap["connected"] is True and snap["com"]["status"] == "connected"
    assert snap["com"]["reconnect_attempts"] == 0 and snap["com"]["last_error"] is None
    assert snap["com"]["last_rx_seconds_ago"] is not None
    assert _ask(s, "GET_FIELD")["ok"] is True


def test_read_error_closes_and_the_link_reconnects(make_server):
    sensor = FakeSensor()
    s = make_server(sensor)
    assert _until(lambda: _history_len(s) > 3)
    first = sensor.readers[0]
    sensor.read_errors = 1
    assert _until(lambda: sensor.opens == 2 and s.link.connected)
    assert first.closed
    n = _history_len(s)
    assert _until(lambda: _history_len(s) > n + 3), "readings resume after the reconnect"
    assert s.link.snapshot()["reconnects"] == 1
    assert sensor.reads_on_closed == 0


def test_stuck_sensor_closes_and_the_link_reconnects(make_server):
    sensor = FakeSensor()
    s = make_server(sensor)
    assert _until(lambda: s.link.connected)
    sensor.stuck = True
    assert _until(lambda: sensor.opens >= 2)
    sensor.stuck = False
    assert sensor.readers[0].closed
    assert _until(lambda: s.link.connected)
    n = _history_len(s)
    assert _until(lambda: _history_len(s) > n + 3)
    assert sensor.reads_on_closed == 0


def test_com_disconnect_holds_until_com_connect(make_server):
    sensor = FakeSensor()
    s = make_server(sensor)
    assert _until(lambda: _history_len(s) > 3)
    reply = _ask(s, "COM_DISCONNECT")
    assert reply == {"ok": True, "connected": False}
    assert sensor.readers[0].closed and s.link.device is None
    opens = sensor.opens
    time.sleep(0.5)                                 # several backoff periods
    assert sensor.opens == opens, "nothing may reopen a held port"
    snap = _ask(s, "GET_SNAPSHOT")
    assert snap["connected"] is False and snap["target_connected"] is False
    assert snap["com"]["status"] == "held" and snap["com"]["held"] is True
    assert _ask(s, "GET_SERIAL_STATUS") == {"ok": True, "connected": False,
                                             "target_connected": False}

    reply = _ask(s, "COM_CONNECT")
    assert reply["ok"] is True and reply["message"]
    assert _until(lambda: s.link.connected)
    assert sensor.opens == opens + 1
    n = _history_len(s)
    assert _until(lambda: _history_len(s) > n + 3)
    assert _ask(s, "GET_SNAPSHOT")["target_connected"] is True


def test_disconnect_landing_while_a_reopen_is_in_progress_stays_held(make_server):
    """The old race: the read loop checked the flag, then reopened; a
    disconnect in between left the port open while the status said
    disconnected.  Now the reopen runs on the link and a hold set during it
    wins."""
    sensor = FakeSensor()
    s = make_server(sensor)
    assert _until(lambda: s.link.connected)
    gate = threading.Event()
    sensor.open_gate = gate
    sensor.open_started.clear()
    sensor.read_errors = 1                          # the read loop reports; the link reopens
    assert sensor.open_started.wait(3.0)            # the reopen has started, and waits
    done = []
    th = threading.Thread(target=lambda: done.append(_ask(s, "COM_DISCONNECT")), daemon=True)
    th.start()
    assert _until(lambda: s.link.held)               # the hold is set before the open ends
    gate.set()
    th.join(3.0)
    assert done == [{"ok": True, "connected": False}]
    assert s.link.device is None and s.link.state == "held"
    assert all(r.closed for r in sensor.readers), "the port opened under the hold is closed"
    opens = sensor.opens
    time.sleep(0.5)
    assert sensor.opens == opens and not s.link.connected
    assert _ask(s, "GET_SERIAL_STATUS")["connected"] is False


def test_disconnect_landing_between_a_read_error_and_the_reset_stays_held(make_server):
    sensor = FakeSensor()
    s = make_server(sensor)
    assert _until(lambda: _history_len(s) > 3)
    replies = []

    def _disconnect_now():                           # inside read_one, before it fails
        threading.Thread(target=lambda: replies.append(_ask(s, "COM_DISCONNECT")),
                         daemon=True).start()
        assert _until(lambda: s.link.held)

    sensor.on_read = _disconnect_now
    sensor.read_errors = 1
    assert _until(lambda: replies == [{"ok": True, "connected": False}])
    opens = sensor.opens
    time.sleep(0.5)
    assert sensor.opens == opens and s.link.state == "held" and s.link.device is None
    assert sensor.reads_on_closed == 0


def test_older_serial_commands_map_onto_the_link(make_server):
    sensor = FakeSensor()
    s = make_server(sensor)
    assert _until(lambda: s.link.connected)

    assert _ask(s, "SERIAL_DISCONNECT") == {"ok": True, "connected": False}
    assert s.link.held
    assert _ask(s, "GET_SERIAL_STATUS") == {"ok": True, "connected": False,
                                             "target_connected": False}
    assert _ask(s, "SERIAL_RECONNECT") == {"ok": True, "connected": True}
    assert not s.link.held

    before = sensor.readers[-1]
    opens = sensor.opens
    reply = _ask(s, "RESTART_SERIAL")
    assert reply == {"ok": True, "connected": True, "message": "Serial restarted successfully"}
    assert before.closed and sensor.opens == opens + 1

    sensor.fail_opens = 10**6                        # every open fails from now on
    reply = _ask(s, "RESTART_SERIAL")
    assert reply["ok"] is False and reply["connected"] is False
    assert reply["error"].startswith("Restart failed (") and "could not open port" in reply["error"]
    opens = sensor.opens
    assert _until(lambda: sensor.opens > opens), "not held: the link keeps retrying"

    reply = _ask(s, "SERIAL_RECONNECT")
    assert reply["ok"] is False and "could not open port" in reply["error"]


def test_client_com_connect_and_disconnect(make_server, monkeypatch):
    sensor = FakeSensor()
    s = make_server(sensor)
    assert _until(lambda: s.link.connected)
    c = _client(s.server_port)

    assert c.com_disconnect() == (True, "")
    assert s.link.held
    ok, msg = c.com_connect()
    assert ok is True and "connect" in msg
    assert _until(lambda: s.link.connected)

    monkeypatch.setattr(s.link, "_disconnect_refusal", lambda: "fake refusal for the test")
    assert c.com_disconnect() == (False, "Disconnect refused: fake refusal for the test")
    assert s.link.connected and not s.link.held

    # an older server that does not know the command
    old = _client(s.server_port)
    monkeypatch.setattr(old, "_request", lambda command, timeout: {
        "ok": False, "error": f"Unknown command: {command!r}"})
    assert old.com_connect() == (False, "Unknown command: 'COM_CONNECT'")


def test_client_com_calls_raise_on_network_failure():
    with socket.socket() as probe:                    # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    c = _client(port)
    with pytest.raises(OSError):
        c.com_connect()
    with pytest.raises(OSError):
        c.com_disconnect()


def test_shutdown_stops_the_link_and_closes_the_port(make_server):
    sensor = FakeSensor()
    s = make_server(sensor)
    assert _until(lambda: s.link.connected)
    s.shutdown()
    assert s.link.device is None and sensor.readers[0].closed
    assert s.link.state == "disconnected"
    opens = sensor.opens
    time.sleep(0.3)
    assert sensor.opens == opens
