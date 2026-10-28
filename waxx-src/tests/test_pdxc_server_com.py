"""PDXC_Server on a ComLink: listening before the port opens, background
retries, in-command recovery, the hold, refusal during a move, and the client's
com_connect / com_disconnect.

Fakes only: ``serial.Serial`` is replaced by a fake controller, the settings
file goes to a temp dir (the real one holds the stage's tracked position), the
socket binds 127.0.0.1 only, and conftest makes the beacon a no-op.
"""
from __future__ import annotations

import socket as _socket
import threading
import time
import types

import pytest
import serial

from waxx.control.misc import pdxc

_real_sleep = time.sleep

# Everything the link's open and the device's reopen may send; never a move.
_OPEN_COMMANDS = {"", "ERR?", "SW=1", "SW?", "FRQ=20000", "FRQ?"}


def _until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        _real_sleep(0.01)
    return pred()


class FakeController:
    """The controller behind every fake port: replies, failure injection, log."""

    def __init__(self):
        self.open_failures = 0          # opens still to refuse
        self.fail_writes = 0            # writes still to fail
        self.opens = 0                  # successful opens
        self.log = []                   # every command received
        self.serials = []
        self.freq = 20000
        self.move_gate = None           # Event a MOV write waits on
        self.moving = threading.Event()
        self.lock = threading.Lock()

    def make_serial(self, port=None, **kw):
        with self.lock:
            if self.open_failures > 0:
                self.open_failures -= 1
                raise serial.SerialException(f"could not open port '{port}': fake")
            self.opens += 1
        ser = FakeSerial(self, kw.get("timeout"))
        self.serials.append(ser)
        return ser

    def reply(self, cmd: str) -> bytes:
        if cmd in ("ERR?",):
            return b"0\r>"
        if cmd == "SW?":
            return b"1\r>"
        if cmd.startswith("FRQ="):
            self.freq = int(cmd.split("=")[1])
            return b">"
        if cmd == "FRQ?":
            return f"{self.freq}\r>".encode()
        if cmd.startswith(("MOVF=", "MOVB=")):
            self.moving.set()
            if self.move_gate is not None:
                self.move_gate.wait(5.0)
            return b">"
        return b">"

    def moves(self):
        return [c for c in self.log if c.startswith("MOV")]


class FakeSerial:
    def __init__(self, ctl: FakeController, timeout):
        self._ctl = ctl
        self.timeout = timeout
        self.is_open = True
        self._pending = b""

    def reset_input_buffer(self):
        self._pending = b""

    def write(self, data: bytes):
        if not self.is_open:
            raise serial.SerialException("Attempting to use a port that is not open")
        with self._ctl.lock:
            fail = self._ctl.fail_writes > 0
            if fail:
                self._ctl.fail_writes -= 1
        if fail:
            raise serial.SerialException("WriteFile failed (PermissionError(13, 'Access is denied.'))")
        cmd = data.decode("ascii").strip()
        self._ctl.log.append(cmd)
        self._pending += self._ctl.reply(cmd)
        return len(data)

    def read_until(self, term=b">"):
        out, self._pending = self._pending, b""
        return out

    def close(self):
        self.is_open = False


class _LoopbackSocket(_socket.socket):
    def bind(self, address):
        super().bind(("127.0.0.1", address[1]))   # never 0.0.0.0 in a test


@pytest.fixture
def rig(monkeypatch, tmp_path):
    ctl = FakeController()
    monkeypatch.setattr(pdxc.serial, "Serial", ctl.make_serial)
    monkeypatch.setattr(pdxc, "_DEFAULTS_FILE", str(tmp_path / "pdxc_server_defaults.json"))
    # initialize() sleeps 1.5 s, a move ~2.3 s per throw: run them fast.
    monkeypatch.setattr(pdxc, "time", types.SimpleNamespace(
        monotonic=time.monotonic, sleep=lambda s: _real_sleep(min(s, 0.01))))
    monkeypatch.setattr(pdxc.PDXC._open_port, "__defaults__", (0.1,))   # reopen window
    monkeypatch.setattr(pdxc._PDXCLink, "BACKOFF_S", (0.05, 0.05, 0.1))
    monkeypatch.setattr(pdxc, "socket", types.SimpleNamespace(
        socket=_LoopbackSocket, AF_INET=_socket.AF_INET, SOCK_STREAM=_socket.SOCK_STREAM,
        SOL_SOCKET=_socket.SOL_SOCKET, SO_REUSEADDR=_socket.SO_REUSEADDR,
        timeout=_socket.timeout, create_connection=_socket.create_connection))
    servers = []
    ctl.servers = servers
    yield ctl
    for s in servers:
        s.stop()
        s._close_device()
    if ctl.move_gate is not None:
        ctl.move_gate.set()


def _server(ctl, start_link=True):
    """A server with its link running, driven through _dispatch (no socket)."""
    s = pdxc.PDXC_Server(com_port="COM_TEST")
    ctl.servers.append(s)
    if start_link:
        s._running = True
        s._link.start()
    return s


def _client(port):
    c = pdxc.PDXC_Client.__new__(pdxc.PDXC_Client)     # no discovery
    c.host, c.port, c.timeout = "127.0.0.1", port, 2.0
    c._rediscover = lambda timeout=2.0: False
    return c


def _connected(s):
    return _until(lambda: s._link.state == "connected")


# --------------------------------------------------------------------------- #


def test_server_listens_and_answers_while_the_port_fails_then_connects(rig):
    rig.open_failures = 10 ** 6
    s = pdxc.PDXC_Server(com_port="COM_TEST")
    rig.servers.append(s)
    errors = []

    def run():
        try:
            s.start()
        except BaseException as exc:          # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert _until(lambda: s._srv is not None)
    client = _client(s._waxx_port)

    assert _until(lambda: s._link.state == "error")
    snap = client.get_snapshot()
    assert snap["com"]["status"] == "error" and snap["com"]["held"] is False
    assert "could not open port" in snap["com"]["last_error"]
    assert snap["com_port"] == "COM_TEST" and snap["position"] is None
    with pytest.raises(RuntimeError, match="not connected"):
        client.get_position_state()
    assert t.is_alive() and not errors

    rig.open_failures = 0                     # the port appears: the link retries
    assert _connected(s)
    assert client.get_position_state() == pdxc.POSITION_UNKNOWN
    assert client.get_snapshot()["com"]["status"] == "connected"
    assert rig.moves() == []

    s.stop()
    t.join(5.0)
    assert not t.is_alive() and not errors
    assert s._link.state == "disconnected" and s._link.device is None
    assert all(not ser.is_open for ser in rig.serials)


def test_a_serial_failure_mid_command_recovers_in_the_command(rig):
    s = _server(rig)
    assert _connected(s)
    dev = s._link.device
    rig.fail_writes = 1
    resp = s._dispatch({"method": "get_frequency", "args": {}})
    assert resp == {"ok": True, "result": 20000}
    assert s._link.device is dev and dev.reconnects == 1 and dev.link_error is None
    assert s._link.state == "connected"
    assert set(rig.log) <= _OPEN_COMMANDS


def test_an_unrecoverable_failure_goes_to_the_link_which_reconnects(rig):
    s = _server(rig)
    assert _connected(s)
    dev = s._link.device
    rig.fail_writes = 1
    rig.open_failures = 10 ** 6               # the in-command reopen fails too
    resp = s._dispatch({"method": "get_frequency", "args": {}})
    assert resp["ok"] is False
    assert s._link.device is None and s._link.state == "error"
    assert "not connected" in s._dispatch({"method": "get_step_size", "args": {}})["error"]
    rig.open_failures = 0
    assert _connected(s)
    assert s._link.device is not dev
    assert s._link.snapshot()["reconnects"] == 1
    assert s._dispatch({"method": "get_frequency", "args": {}})["ok"]
    assert rig.moves() == []


def test_remote_close_and_reconnect_are_refused(rig):
    s = _server(rig)
    assert _connected(s)
    ser = rig.serials[-1]
    for method in ("close", "reconnect", "_open_port", "_flush_line"):
        resp = s._dispatch({"method": method, "args": {}})
        assert resp["ok"] is False and "COM_CONNECT" in resp["error"], method
    assert ser.is_open and s._link.state == "connected"


def test_com_disconnect_holds_until_com_connect(rig):
    s = _server(rig)
    assert _connected(s)
    ser = rig.serials[-1]
    assert s._dispatch({"method": "COM_DISCONNECT"}) == {"ok": True,
                                                         "result": "disconnected (held)"}
    assert not ser.is_open and s._link.state == "held" and s._link.device is None
    assert s.get_snapshot()["com"]["held"] is True
    opens = rig.opens

    resp = s._dispatch({"method": "get_frequency", "args": {}})
    assert resp["ok"] is False and "held" in resp["error"]
    for method in ("move_to", "close", "reconnect", "initialize"):
        args = {"state": "in"} if method == "move_to" else {}
        assert s._dispatch({"method": method, "args": args})["ok"] is False
    _real_sleep(0.4)                          # several backoff periods
    assert rig.opens == opens and s._link.state == "held"

    assert s._dispatch({"method": "COM_CONNECT"}) == {"ok": True, "result": "connecting"}
    assert _connected(s)
    assert rig.opens == opens + 1 and not s._link.held
    assert s._dispatch({"method": "get_frequency", "args": {}})["ok"]
    assert rig.moves() == []


def test_a_command_waiting_when_the_hold_lands_never_reopens(rig):
    s = _server(rig)
    assert _connected(s)
    opens = rig.opens
    results = {}
    s._device_lock.acquire()                  # the command and the close queue
    try:
        cmd = threading.Thread(target=lambda: results.setdefault(
            "cmd", s._dispatch({"method": "get_frequency", "args": {}})), daemon=True)
        cmd.start()
        disc = threading.Thread(target=lambda: results.setdefault(
            "disc", s._dispatch({"method": "COM_DISCONNECT"})), daemon=True)
        disc.start()
        assert _until(lambda: s._link.held)
        rig.fail_writes = 1                   # if the command goes first, its I/O fails
    finally:
        s._device_lock.release()
    cmd.join(5.0)
    disc.join(5.0)
    assert results["disc"]["ok"] is True
    # Whichever ran first: no reopen, the port ends closed and held.
    assert rig.opens == opens
    assert s._link.state == "held" and s._link.device is None
    assert all(not ser.is_open for ser in rig.serials)
    _real_sleep(0.3)
    assert rig.opens == opens


def test_disconnect_is_refused_during_a_move(rig):
    s = _server(rig)
    assert _connected(s)
    rig.move_gate = threading.Event()
    out = {}
    mover = threading.Thread(target=lambda: out.setdefault(
        "move", s._dispatch({"method": "move_to", "args": {"state": "in", "force": True}})),
        daemon=True)
    mover.start()
    assert rig.moving.wait(3.0)
    t0 = time.monotonic()
    resp = s._dispatch({"method": "COM_DISCONNECT"})
    assert time.monotonic() - t0 < 0.5        # refused at once, not queued behind the move
    assert resp["ok"] is False and "move" in resp["error"]
    assert s._link.state == "connected" and rig.serials[-1].is_open
    rig.move_gate.set()
    mover.join(5.0)
    assert out["move"] == {"ok": True, "result": "moved in"}
    assert s._dispatch({"method": "COM_DISCONNECT"})["ok"] is True
    assert s._link.state == "held"


def test_connect_and_every_reconnect_send_no_move(rig):
    rig.open_failures = 2
    s = _server(rig)
    assert _connected(s)
    s._link.report_error(OSError("fake drop"), where="test")      # link reopen
    assert _until(lambda: s._link.state == "connected" and s._link.snapshot()["reconnects"] == 1)
    rig.fail_writes = 1                                              # in-command reopen
    assert s._dispatch({"method": "get_frequency", "args": {}})["ok"]
    assert s._dispatch({"method": "COM_DISCONNECT"})["ok"]
    assert s._dispatch({"method": "COM_CONNECT"})["ok"]
    assert _connected(s)
    assert rig.log and set(rig.log) <= _OPEN_COMMANDS
    assert rig.moves() == []


def test_snapshot_keeps_its_keys(rig):
    s = _server(rig)
    assert _connected(s)
    snap = s.get_snapshot()
    for key in ("com", "com_port", "server_port", "position", "move_in_progress"):
        assert key in snap
    assert snap["com"]["port"] == "COM_TEST" and snap["com"]["baud"] == 115200
    assert snap["com"]["status"] == "connected"


def test_client_com_connect_and_disconnect_parse_replies(rig):
    c = _client(1)
    replies = {"COM_CONNECT": {"ok": True, "result": "connecting"},
               "COM_DISCONNECT": {"ok": False, "error": "disconnect refused: a stage move"}}
    sent = []
    c._send = lambda method, **kw: (sent.append(method), replies[method])[1]
    assert c.com_connect() == (True, "connecting")
    assert c.com_disconnect() == (False, "disconnect refused: a stage move")
    assert sent == ["COM_CONNECT", "COM_DISCONNECT"]


def test_client_com_methods_end_to_end_and_raise_when_unreachable(rig):
    rig.open_failures = 10 ** 6
    s = pdxc.PDXC_Server(com_port="COM_TEST")
    rig.servers.append(s)
    t = threading.Thread(target=s.start, daemon=True)
    t.start()
    assert _until(lambda: s._srv is not None)
    client = _client(s._waxx_port)
    ok, msg = client.com_disconnect()
    assert ok and "held" in msg and s._link.state == "held"
    rig.open_failures = 0
    ok, msg = client.com_connect()
    assert ok and msg == "connecting"
    assert _connected(s)
    s.stop()
    t.join(5.0)

    probe = _socket.socket()
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(RuntimeError, match="unreachable"):
        _client(free_port).com_connect()
