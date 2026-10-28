"""Precilaser server on ComLink: background connect, reconnect after errors,
in-place frame resync, the person's hold (poll loop / resync / sequence never
reopen it), COM_CONNECT / COM_DISCONNECT, refusal during a sequence, and the
client's com_connect / com_disconnect.

Fake controllers only -- no serial port is opened.  The server binds
127.0.0.1 port 0; conftest makes the beacon a no-op.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from waxx.util.guis.precilaser import precilaser_server as srv_mod
from waxx.util.guis.precilaser.precilaser_controller import (
    PrecilaserStartupController,
    PrecilaserStatus,
)
from waxx.util.guis.precilaser.precilaser_gui_client import PrecilaserGuiClient


def _until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class Rig:
    """Shared state of the fake controllers (what the 'hardware' does)."""

    def __init__(self):
        self.fail_connects = 0          # next N connects raise
        self.always_fail = False
        self.query_errors: list[Exception] = []   # raised by the next queries, in order
        self.set_errors: list[Exception] = []     # raised by the next set_working_current
        self.instances: list["FakeLaser"] = []
        self.connects = 0
        self.current_a = 0.0
        self.enabled = True
        self.on_close = None            # callable(laser) run inside close()
        self.writes: list[tuple] = []   # every command that changes the laser


class FakeLaser:
    rig: Rig = None                      # set by the fixture

    def __init__(self, port="COM20", baudrate=115200, **_kw):
        self.port = port
        self.baudrate = baudrate
        self.is_open = False
        self.connect_calls = 0
        self.close_calls = 0
        self.rig.instances.append(self)

    def connect(self):
        rig = self.rig
        if self.is_open:
            return
        rig.connects += 1
        self.connect_calls += 1
        if rig.always_fail or rig.fail_connects > 0:
            rig.fail_connects = max(0, rig.fail_connects - 1)
            raise OSError(f"could not open port '{self.port}': FileNotFoundError")
        self.is_open = True

    def close(self):
        self.close_calls += 1
        self.is_open = False
        if self.rig.on_close is not None:
            self.rig.on_close(self)

    def _require(self):
        if not self.is_open:
            raise RuntimeError("Serial port is not open. Call connect() first.")

    def query_working_status(self):
        self._require()
        if self.rig.query_errors:
            raise self.rig.query_errors.pop(0)
        cur = self.rig.current_a
        return PrecilaserStatus(
            pd_ok=True, temperature_ok=True, laser_enabled=self.rig.enabled,
            power_stability_enabled=True, working_current_a=cur,
            stage_currents_a=[cur], pd_values=[1.0] * 5, temperatures_c=[25.0] * 4,
        )

    def set_working_current(self, value):
        self._require()
        if self.rig.set_errors:
            raise self.rig.set_errors.pop(0)
        self.rig.writes.append(("current", value))
        self.rig.current_a = value

    def set_laser_enable(self, enabled):
        self._require()
        self.rig.writes.append(("enable", enabled))
        self.rig.enabled = enabled

    def set_power_stability_mode(self, enabled):
        self._require()
        self.rig.writes.append(("stability", enabled))


class QuickStartup(PrecilaserStartupController):
    """The real sequence logic with short steps (0.1 A every 0.05 s)."""

    def __init__(self, laser, target_working_current_a=10.0, **_kw):
        super().__init__(laser, seed_stabilize_s=0.0, laser_stabilize_s=0.0,
                         ramp_step_a=0.1, ramp_wait_s=0.05,
                         target_working_current_a=target_working_current_a)


@pytest.fixture
def rig(monkeypatch):
    r = Rig()
    FakeLaser.rig = r
    monkeypatch.setattr(srv_mod, "PrecilaserController", FakeLaser)
    monkeypatch.setattr(srv_mod, "PrecilaserStartupController", QuickStartup)
    monkeypatch.setattr(srv_mod, "_SEQUENCE_RECONNECT_WAIT_S", 2.0)
    return r


@pytest.fixture
def make_server(rig):
    made = []

    def _make(**kw):
        server = srv_mod.PrecilaserLaserServer(
            host="127.0.0.1", port=0, serial_port="COM_TEST", poll_interval_s=0.05, **kw
        )
        server.link.BACKOFF_S = (0.05, 0.05, 0.1)
        made.append(server)
        return server

    yield _make
    for server in made:
        server.stop()


def _tcp(server, command: str) -> str:
    with socket.create_connection(("127.0.0.1", server.port), timeout=2.0) as sock:
        sock.sendall(f"{command}\n".encode())
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode().strip()


def _connected(server):
    return server.link.state == "connected" and server.laser is not None


# --------------------------------------------------------------------------- #


def test_starts_and_answers_while_the_port_fails(rig, make_server):
    rig.always_fail = True
    server = make_server()
    t0 = time.monotonic()
    server.start()
    assert time.monotonic() - t0 < 0.5, "start() must not wait for the COM port"
    assert _until(lambda: rig.connects >= 3), "the link keeps retrying"
    snap = json.loads(_tcp(server, "GET_SNAPSHOT"))
    assert snap["com"]["status"] in ("error", "connecting")
    assert snap["com"]["last_error"] and "COM_TEST" in snap["com"]["last_error"]
    assert snap["com"]["held"] is False and snap["com"]["port"] == "COM_TEST"
    assert snap["status"]["connection_state"] == "ERROR"
    assert snap["status"]["connected"] is False
    assert server.running
    # Commands that need the laser fail cleanly; nothing was written.
    assert _tcp(server, "SET_WORKING_CURRENT 1.0").startswith("ERROR")
    assert _tcp(server, "START_STARTUP").startswith("ERROR")
    assert rig.writes == []


def test_retries_and_connects_without_writing_to_the_laser(rig, make_server):
    rig.fail_connects = 2
    rig.current_a = 3.25
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    assert rig.connects == 3
    snap = server.get_snapshot()
    assert snap["com"]["status"] == "connected" and snap["com"]["last_error"] is None
    assert snap["status"]["connection_state"] == "CONNECTED"
    assert _until(lambda: server.get_snapshot()["status"]["working_current_a"] == 3.25)
    assert rig.writes == [], "a connect must not send anything that changes the laser"


def test_a_serial_error_reconnects_through_the_link(rig, make_server):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    first = server.laser
    rig.query_errors.append(OSError("WriteFile failed (PermissionError(13, 'Access is denied.'))"))
    assert _until(lambda: _connected(server) and server.laser is not first)
    assert first.close_calls >= 1 and not first.is_open
    assert server.link.snapshot()["reconnects"] == 1
    assert rig.writes == []


@pytest.mark.parametrize("error", [
    ValueError("Invalid frame tail: b'\\x00\\x00'"),
    ValueError("SUM checksum mismatch: expected 0x01, got 0x02"),
])
def test_frame_errors_resync_in_place(rig, make_server, error):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    laser = server.laser
    rig.query_errors.append(error)
    assert _until(lambda: laser.connect_calls == 2)
    assert server.laser is laser and laser.is_open
    assert server.link.state == "connected"
    assert len(rig.instances) == 1


def test_com_disconnect_holds_against_poll_and_errors(rig, make_server):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    laser = server.laser
    assert _tcp(server, "COM_DISCONNECT") == "OK"
    assert server.link.held and server.laser is None and not laser.is_open
    snap = server.get_snapshot()
    assert snap["com"]["status"] == "held" and snap["com"]["held"] is True
    assert snap["status"]["connection_state"] == "DISCONNECTED"
    connects = rig.connects
    server.poll_status()                       # a direct poll does nothing
    server.link.report_error(OSError("late failure"))
    time.sleep(0.5)                            # many poll periods and backoffs
    assert rig.connects == connects and server.laser is None
    assert server.link.state == "held"
    assert _tcp(server, "START_STARTUP").startswith("ERROR")
    assert rig.writes == []


def test_old_disconnect_holds_and_old_connect_releases(rig, make_server):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    assert _tcp(server, "DISCONNECT_SERIAL") == "OK"
    assert server.link.held
    connects = rig.connects
    time.sleep(0.3)
    assert rig.connects == connects
    assert _tcp(server, "CONNECT_SERIAL") == "OK"
    assert _until(lambda: _connected(server)) and not server.link.held


def test_com_connect_releases_the_hold(rig, make_server):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    assert _tcp(server, "COM_DISCONNECT") == "OK"
    connects = rig.connects
    assert _tcp(server, "COM_CONNECT") == "OK"
    assert _until(lambda: _connected(server))
    assert not server.link.held and rig.connects == connects + 1
    assert server.get_snapshot()["com"]["status"] == "connected"


def test_com_connect_without_auto_connect_starts_the_link(rig, make_server):
    server = make_server(auto_connect=False)
    server.start()
    time.sleep(0.2)
    assert rig.connects == 0 and server.link.state == "disconnected"
    assert _tcp(server, "COM_CONNECT") == "OK"
    assert _until(lambda: _connected(server))


def test_a_resync_never_reopens_a_port_held_meanwhile(rig, make_server, monkeypatch):
    monkeypatch.setattr(srv_mod, "_CHECKSUM_RESYNC_DELAY_S", 0.3)
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    laser = server.laser
    replies = []

    def disconnect_during_resync(_laser):
        rig.on_close = None
        # The person's Disconnect arrives while the resync sleeps (it waits
        # for the device lock the resync holds).
        threading.Thread(target=lambda: replies.append(_tcp(server, "COM_DISCONNECT")),
                         daemon=True).start()

    rig.on_close = disconnect_during_resync
    rig.query_errors.append(ValueError("XOR checksum mismatch: expected 0x01, got 0x02"))
    assert _until(lambda: replies == ["OK"])
    assert laser.connect_calls == 1, "the resync must not reopen a held port"
    assert server.laser is None and server.link.held and not laser.is_open
    time.sleep(0.3)
    assert rig.connects == 1


def _start_long_ramp(server, rig, endpoint=3.0):
    rig.current_a = 0.0
    rig.enabled = True
    server.set_ramp_up_endpoint(endpoint)       # 30 steps of 0.05 s
    assert _tcp(server, "START_STARTUP") == "OK"
    assert _until(lambda: len(rig.writes) >= 2)


def test_disconnect_is_refused_during_a_sequence(rig, make_server):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    laser = server.laser
    _start_long_ramp(server, rig)
    assert _tcp(server, "COM_DISCONNECT") == "ERROR: a sequence is running"
    assert _tcp(server, "DISCONNECT_SERIAL") == "ERROR: a sequence is running"
    assert server.laser is laser and laser.is_open and not server.link.held
    assert _tcp(server, "COM_CONNECT") == "OK"   # harmless: never replaces the port
    assert server.laser is laser
    assert _tcp(server, "INTERRUPT") == "OK"
    assert _until(lambda: server.sequence_state == srv_mod.SequenceState.INTERRUPTED)
    assert _tcp(server, "COM_DISCONNECT") == "OK"


def test_a_sequence_never_reopens_a_held_port(rig, make_server):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    _start_long_ramp(server, rig)
    connects = rig.connects
    server.link.disconnect(hold=True, force=True)     # forced: bypasses the refusal
    assert _until(lambda: server.sequence_state == srv_mod.SequenceState.INTERRUPTED)
    time.sleep(0.3)
    assert rig.connects == connects and server.laser is None and server.link.held
    assert rig.current_a < 3.0, "the ramp stopped"


def test_a_sequence_resumes_after_the_link_reconnects(rig, make_server):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    first = server.laser
    _start_long_ramp(server, rig, endpoint=1.0)
    rig.set_errors.append(OSError("WriteFile failed"))
    assert _until(lambda: server.sequence_state == srv_mod.SequenceState.COMPLETED, timeout=5.0)
    assert server.laser is not first and _connected(server)
    assert rig.current_a == pytest.approx(1.0)


def test_stop_closes_the_port_through_the_link(rig, make_server):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    laser = server.laser
    server.stop()
    assert not laser.is_open and server.laser is None
    assert server.link.state == "disconnected"
    connects = rig.connects
    time.sleep(0.2)
    assert rig.connects == connects


# --------------------------------------------------------------------------- #
# Client


def _bare_client(host="127.0.0.1", port=1):
    client = object.__new__(PrecilaserGuiClient)     # no discovery
    client.host = host
    client.port = port
    client.timeout_s = 2.0
    return client


@pytest.mark.parametrize("reply, expected", [
    ("OK", (True, "")),
    ("ERROR: a sequence is running", (False, "a sequence is running")),
    ("ERROR: unknown command COM_CONNECT", (False, "unknown command COM_CONNECT")),
])
def test_client_parses_com_replies(monkeypatch, reply, expected):
    client = _bare_client()
    sent = []
    monkeypatch.setattr(client, "_send_command", lambda cmd: (sent.append(cmd), reply)[1])
    assert client.com_connect() == expected
    assert client.com_disconnect() == expected
    assert sent == ["COM_CONNECT", "COM_DISCONNECT"]


def test_client_com_commands_raise_on_network_failure(monkeypatch):
    client = _bare_client()

    def boom(cmd):
        raise ConnectionRefusedError("no server")

    monkeypatch.setattr(client, "_send_command", boom)
    with pytest.raises(OSError):
        client.com_connect()
    with pytest.raises(OSError):
        client.com_disconnect()


class _ToggleClient:
    host, port = "127.0.0.1", 1

    def __init__(self, reply=(True, ""), old_ok=True):
        self.reply = reply
        self.old_ok = old_ok
        self.calls: list[str] = []

    def com_connect(self):
        self.calls.append("com_connect")
        return self.reply

    def com_disconnect(self):
        self.calls.append("com_disconnect")
        return self.reply

    def connect_serial(self):
        self.calls.append("connect_serial")
        return self.old_ok

    def disconnect_serial(self):
        self.calls.append("disconnect_serial")
        return self.old_ok


@pytest.fixture
def control_gui(monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from waxx.util.guis.precilaser import precilaser_control_gui as gui_mod

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui_mod, "PrecilaserGuiClient", lambda *a, **k: _ToggleClient())
    gui = gui_mod.PrecilaserControlGUI()
    gui.status_timer.stop()
    yield gui
    gui.close()
    gui.deleteLater()
    app.processEvents()


def test_gui_toggle_uses_the_link_commands_and_shows_a_refusal(control_gui):
    gui = control_gui
    gui.connection_state_value.setText("CONNECTED")
    gui.client = _ToggleClient(reply=(False, "a sequence is running"))
    gui._toggle_serial_connection()
    assert gui.client.calls == ["com_disconnect"]
    assert "a sequence is running" in gui.log_text.toPlainText()
    gui.connection_state_value.setText("DISCONNECTED")
    gui.client = _ToggleClient()
    gui._toggle_serial_connection()
    assert gui.client.calls == ["com_connect"]


def test_gui_toggle_falls_back_for_an_old_server(control_gui):
    gui = control_gui
    gui.connection_state_value.setText("DISCONNECTED")
    gui.client = _ToggleClient(reply=(False, "unknown command COM_CONNECT"))
    gui._toggle_serial_connection()
    assert gui.client.calls == ["com_connect", "connect_serial"]
    gui.connection_state_value.setText("CONNECTED")
    gui.client = _ToggleClient(reply=(False, "unknown command COM_DISCONNECT"))
    gui._toggle_serial_connection()
    assert gui.client.calls == ["com_disconnect", "disconnect_serial"]


def test_client_round_trip_against_the_server(rig, make_server):
    server = make_server()
    server.start()
    assert _until(lambda: _connected(server))
    client = _bare_client(port=server.port)
    assert client.com_disconnect() == (True, "")
    assert client.get_snapshot()["com"]["held"] is True
    assert client.com_connect() == (True, "")
    assert _until(lambda: _connected(server))
    _start_long_ramp(server, rig)
    assert client.com_disconnect() == (False, "a sequence is running")
    client.interrupt_sequence()
