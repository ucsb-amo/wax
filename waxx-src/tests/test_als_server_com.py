"""ALS server on the shared ComLink: it answers while the COM port fails, retries
and connects, reconnects after a failed poll, holds a person's Disconnect,
refuses while a sequence runs, and resumes a saved sequence only after the
first open at startup.

The laser controller is replaced by a fake before any server is built: no
serial port is opened, nothing is sent to any hardware, and no beacon goes out
(conftest).  The test servers listen on 127.0.0.1, port 0.
"""
from __future__ import annotations

import json
import socket
import time
from types import SimpleNamespace

import pytest

from waxx.util.guis.als import als_control_gui
from waxx.util.guis.als import als_server as als_mod
from waxx.util.guis.als.als_gui_client import ALSGuiClient

# What a connect / reconnect may send: open, handshake, status reads, close.
_READ_ONLY_CALLS = {"connect", "handshake", "close", "stop_and_read", "convert_frame",
                    "cmd_ask_power_consign", "cmd_ask_interlock_sts", "cmd_ask_secondstage_sts"}


def _until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class _Rig:
    """Knobs + records shared by every fake controller of one test."""

    def __init__(self):
        self.fail_opens = 0          # this many opens fail (handshake times out)
        self.opens = 0
        self.instances = []
        self.calls = []              # (instance index, method name)


@pytest.fixture
def rig(monkeypatch):
    r = _Rig()

    class FakeController:
        baudrate = 115200

        def __init__(self, port="COM6", **_kw):
            self.port = port
            self.idx = len(r.instances)
            self.closed = False
            self.fail_reads = False
            r.instances.append(self)

        def _rec(self, name):
            r.calls.append((self.idx, name))

        def connect(self):
            self._rec("connect")
            r.opens += 1

        def handshake(self):
            self._rec("handshake")
            if r.opens <= r.fail_opens:
                raise TimeoutError("Expected 1 bytes, received 0 bytes: ")

        def close(self):
            self._rec("close")
            self.closed = True

        def stop_and_read(self):
            self._rec("stop_and_read")
            if self.fail_reads:
                raise TimeoutError("Expected 1 bytes, received 0 bytes: ")
            return SimpleNamespace(statuses={"STS_RELAY_PSU": 0})

        def convert_frame(self, frame):
            self._rec("convert_frame")
            return SimpleNamespace(PMON_W=0.0, TACT_P=25.0, TSET_P=25.0, IMON_PA=0.0, LMON=0.0)

        def cmd_ask_power_consign(self):
            self._rec("cmd_ask_power_consign")
            return 0

        def cmd_ask_interlock_sts(self):
            self._rec("cmd_ask_interlock_sts")
            return 0

        def cmd_ask_secondstage_sts(self):
            self._rec("cmd_ask_secondstage_sts")
            return 0

    monkeypatch.setattr(als_mod, "ALSLaserController", FakeController)
    return r


@pytest.fixture
def make_server(tmp_path):
    made = []

    def _make(auto_connect=True, saved_state=None):
        state_file = tmp_path / f"als_state_{len(made)}.json"
        if saved_state is not None:
            state_file.write_text(json.dumps(saved_state))
        srv = als_mod.ALSLaserServer(host="127.0.0.1", port=0, serial_port="COMFAKE",
                                     poll_interval_s=0.05, auto_connect=auto_connect,
                                     state_file=str(state_file))
        srv._link.BACKOFF_S = (0.05, 0.05, 0.1)
        made.append(srv)
        return srv

    yield _make
    for srv in made:
        srv.sequence_state = als_mod.SequenceState.IDLE
        srv.stop()


def _tcp(srv, command: str) -> str:
    with socket.create_connection(("127.0.0.1", srv.port), timeout=2.0) as sock:
        sock.sendall(f"{command}\n".encode())
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode().strip()


def _snap(srv) -> dict:
    return json.loads(_tcp(srv, "GET_SNAPSHOT"))


def _client_for(srv) -> ALSGuiClient:
    """An ALSGuiClient aimed at the test server, without discovery."""
    client = ALSGuiClient.__new__(ALSGuiClient)
    client.host, client.port, client.timeout_s = "127.0.0.1", srv.port, 2.0
    client._rediscover = lambda timeout=None: False
    return client


def _connected(srv) -> bool:
    return srv._link.state == "connected" and \
        srv.status.connection_state == als_mod.ConnectionState.CONNECTED


def test_server_answers_and_keeps_retrying_while_the_port_fails(rig, make_server):
    rig.fail_opens = 10**6
    srv = make_server()
    t0 = time.monotonic()
    srv.start()
    assert time.monotonic() - t0 < 0.5, "start() must not wait for the COM port"
    snap = _snap(srv)                                    # the server answers at once
    assert snap["com"]["port"] == "COMFAKE"
    assert _until(lambda: rig.opens >= 3)                # retried on the backoff
    snap = _snap(srv)
    assert snap["com"]["status"] in ("error", "connecting")
    assert snap["com"]["last_error"] and snap["com"]["held"] is False
    assert snap["status"]["connection_state"] == "ERROR"
    assert snap["status"]["connected"] is False
    assert all(inst.closed for inst in rig.instances)    # a failed handshake closes the port
    assert srv.running


def test_retries_then_connects_sending_only_reads(rig, make_server):
    rig.fail_opens = 2
    srv = make_server()
    srv.start()
    assert _until(lambda: _connected(srv))
    assert rig.opens == 3
    snap = _snap(srv)
    assert snap["com"]["status"] == "connected" and snap["com"]["last_error"] is None
    assert snap["status"]["connection_state"] == "CONNECTED"
    assert {name for _, name in rig.calls} <= _READ_ONLY_CALLS


def test_a_failed_poll_reconnects(rig, make_server):
    srv = make_server()
    srv.start()
    assert _until(lambda: _connected(srv))
    first = srv.laser
    first.fail_reads = True                               # the next poll fails
    assert _until(lambda: srv.laser is not None and srv.laser is not first and _connected(srv))
    assert first.closed
    assert srv._link.snapshot()["reconnects"] == 1
    assert {name for _, name in rig.calls} <= _READ_ONLY_CALLS


@pytest.mark.parametrize("disconnect", ["COM_DISCONNECT", "DISCONNECT_SERIAL"])
def test_disconnect_holds_until_connect(rig, make_server, disconnect):
    srv = make_server()
    srv.start()
    assert _until(lambda: _connected(srv))
    dev = srv.laser
    assert _tcp(srv, disconnect) == "OK"
    assert dev.closed and srv.laser is None
    snap = _snap(srv)
    assert snap["com"]["status"] == "held" and snap["com"]["held"] is True
    assert snap["status"]["connection_state"] == "DISCONNECTED"
    opens = rig.opens
    time.sleep(0.5)                                       # many poll + backoff periods
    assert rig.opens == opens and srv.laser is None, "something reopened a held port"
    assert _tcp(srv, "START_STARTUP").startswith("ERROR")
    assert _tcp(srv, "SET_POWER_SUPPLY_ON").startswith("ERROR")
    assert rig.opens == opens

    assert _tcp(srv, "COM_CONNECT") == "OK"               # releases the hold
    assert _until(lambda: _connected(srv))
    assert rig.opens == opens + 1 and not srv._link.held


def test_old_connect_serial_releases_the_hold(rig, make_server):
    srv = make_server()
    srv.start()
    assert _until(lambda: _connected(srv))
    assert _tcp(srv, "COM_DISCONNECT") == "OK"
    assert _tcp(srv, "CONNECT_SERIAL") == "OK"
    assert _until(lambda: _connected(srv))


def test_connect_and_disconnect_refused_while_a_sequence_runs(rig, make_server):
    srv = make_server()
    srv.start()
    assert _until(lambda: _connected(srv))
    dev = srv.laser
    srv.sequence_state = als_mod.SequenceState.RUNNING   # stand-in: no real sequence
    for cmd in ("COM_DISCONNECT", "DISCONNECT_SERIAL", "COM_CONNECT", "CONNECT_SERIAL"):
        assert _tcp(srv, cmd) == "ERROR: a sequence is running", cmd
    assert srv.laser is dev and not dev.closed and srv._link.state == "connected"
    client = _client_for(srv)
    assert client.com_disconnect() == (False, "a sequence is running")
    assert client.com_connect() == (False, "a sequence is running")
    assert client.disconnect_serial() is False
    srv.sequence_state = als_mod.SequenceState.IDLE
    assert client.com_disconnect() == (True, "")
    assert srv._link.held and dev.closed
    assert client.com_connect() == (True, "")
    assert _until(lambda: _connected(srv))


_SAVED_RUNNING = {"sequence_type": "STARTUP", "sequence_state": "RUNNING",
                  "sequence_started_epoch": 0.0, "last_completed_step": 2,
                  "warmup_started_epoch": None}


def test_saved_sequence_resumes_after_the_first_open_only(rig, make_server):
    srv = make_server(saved_state=_SAVED_RUNNING)
    resumed = []
    srv._check_and_resume_sequence = lambda: resumed.append(time.monotonic())
    srv.start()
    assert _until(lambda: resumed)                        # the startup connect resumes it
    first = srv.laser
    first.fail_reads = True                               # a reconnect ...
    assert _until(lambda: srv.laser is not None and srv.laser is not first and _connected(srv))
    assert _tcp(srv, "COM_DISCONNECT") == "OK"            # ... and a person's reconnect
    assert _tcp(srv, "COM_CONNECT") == "OK"
    assert _until(lambda: _connected(srv) and srv._link.snapshot()["reconnects"] == 2)
    time.sleep(0.1)
    assert len(resumed) == 1, "the saved sequence was resumed on a reconnect"


def test_saved_sequence_not_resumed_when_the_startup_open_failed(rig, make_server):
    rig.fail_opens = 1
    srv = make_server(saved_state=_SAVED_RUNNING)
    resumed = []
    srv._check_and_resume_sequence = lambda: resumed.append(1)
    srv.start()
    assert _until(lambda: _connected(srv))                # connected on the retry
    time.sleep(0.1)
    assert resumed == []


def test_a_running_sequence_fails_on_a_reopened_port(rig, make_server):
    srv = make_server()
    srv.start()
    assert _until(lambda: _connected(srv))
    proxy = als_mod._LockedLaserProxy(srv, bound=srv.laser)
    proxy.cmd_ask_interlock_sts()                         # same handle: fine
    srv.laser.fail_reads = True
    first = srv.laser
    assert _until(lambda: srv.laser is not None and srv.laser is not first)
    with pytest.raises(RuntimeError, match="reopened"):
        proxy.cmd_ask_interlock_sts()


def test_stop_closes_the_port_and_stops_the_link(rig, make_server):
    srv = make_server()
    srv.start()
    assert _until(lambda: _connected(srv))
    dev = srv.laser
    srv.stop()
    assert dev.closed and srv.laser is None
    assert srv._link.state == "disconnected"
    opens = rig.opens
    time.sleep(0.2)
    assert rig.opens == opens


def test_client_com_replies_are_parsed(monkeypatch):
    client = ALSGuiClient.__new__(ALSGuiClient)
    client.timeout_s = 2.0
    replies = {"COM_CONNECT": "OK", "COM_DISCONNECT": "ERROR: a sequence is running"}
    sent = []
    monkeypatch.setattr(client, "_send_command",
                        lambda cmd: (sent.append(cmd), replies[cmd])[1], raising=False)
    assert client.com_connect() == (True, "")
    assert client.com_disconnect() == (False, "a sequence is running")
    assert sent == ["COM_CONNECT", "COM_DISCONNECT"]

    def _down(cmd):
        raise ConnectionRefusedError("server down")
    monkeypatch.setattr(client, "_send_command", _down, raising=False)
    with pytest.raises(OSError):
        client.com_connect()


def test_gui_toggle_raises_the_refusal_and_falls_back_for_an_old_server():
    class _Client:
        def __init__(self, reply, legacy=True):
            self.reply, self.legacy, self.calls = reply, legacy, []

        def com_connect(self):
            self.calls.append("com_connect")
            return self.reply

        com_disconnect = com_connect

        def connect_serial(self):
            self.calls.append("connect_serial")
            return self.legacy

        disconnect_serial = connect_serial

    als_control_gui._com_request(_Client((True, "")), connect=True)
    with pytest.raises(RuntimeError, match="a sequence is running"):
        als_control_gui._com_request(_Client((False, "a sequence is running")), connect=False)
    old = _Client((False, "unknown command COM_CONNECT"))
    als_control_gui._com_request(old, connect=True)
    assert old.calls == ["com_connect", "connect_serial"]
    with pytest.raises(RuntimeError):
        als_control_gui._com_request(_Client((False, "unknown command COM_DISCONNECT"),
                                             legacy=False), connect=False)
