"""UdpServer serves each connection on a thread of its own (2026-11-16).

Before, the accept loop served one connection at a time, reading each until a
newline or a 5 s timeout, with a listen backlog of 5: one client that connected
and sent nothing stalled every other client for 5 s, and the clients queued
behind it were refused (kq: "the monitor server did not answer the run queue's
tail request").  Here: an idle connection holds up nobody, many clients at once
are all answered, a handler that raises ends only its own connection, stop()
ends the loop, and the monitor server's plain-text commands run under its
request lock.

Everything runs on 127.0.0.1 with an OS-chosen port and no server_id, so no
beacon is ever sent; the monitor server is built (never run) with a recorder in
place of its broadcaster and every folder in a pytest temp dir.
"""
import json
import os
import socket
import threading
import time

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.util.comms_server.comm_server import UdpServer


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class EchoServer(UdpServer):
    CONN_TIMEOUT_S = 3.0

    def __init__(self):
        super().__init__(host="127.0.0.1", port=0, server_id=None)
        self._print_connections_bool = False
        self.seen = []

    def on_message_received(self, message):
        self.seen.append(message)

    def generate_reply(self, message):
        if message == "boom":
            raise ValueError("handler failed")
        if message.startswith("slow"):
            time.sleep(0.5)
        return "re:" + message


@pytest.fixture
def server(qapp):
    s = EchoServer()
    th = threading.Thread(target=s.run, daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while not s.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert s.running
    s.thread = th
    yield s
    s.running = False
    s.sock.close()
    th.join(timeout=3)


def ask(server, text, timeout=2.0):
    with socket.create_connection(("127.0.0.1", server.port), timeout=timeout) as c:
        c.sendall((text + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = c.recv(4096)
            if not chunk:
                break
            buf += chunk
    return buf.split(b"\n", 1)[0].decode()


def test_idle_connection_holds_up_nobody(server):
    idle = [socket.create_connection(("127.0.0.1", server.port)) for _ in range(3)]
    try:
        time.sleep(0.1)                           # the idle ones are accepted first
        t0 = time.monotonic()
        assert ask(server, "ping") == "re:ping"
        assert time.monotonic() - t0 < 1.0        # not behind the idle ones' timeout
    finally:
        for c in idle:
            c.close()


def test_many_clients_at_once_are_all_answered(server):
    idle = [socket.create_connection(("127.0.0.1", server.port)) for _ in range(3)]
    replies, errors = {}, []

    def one(i):
        try:
            replies[i] = ask(server, f"slow {i}", timeout=10.0)
        except Exception as exc:                  # noqa: BLE001
            errors.append(repr(exc))

    try:
        threads = [threading.Thread(target=one, args=(i,)) for i in range(40)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        elapsed = time.monotonic() - t0
    finally:
        for c in idle:
            c.close()
    assert errors == []
    assert replies == {i: f"re:slow {i}" for i in range(40)}
    # 40 x 0.5 s served one at a time would take 20 s; in parallel (32 slots) ~1 s
    assert elapsed < 5.0


def test_a_connection_that_sends_nothing_is_closed_after_its_timeout(server):
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as c:
        t0 = time.monotonic()
        assert c.recv(10) == b""                  # closed by the server, no reply
        assert server.CONN_TIMEOUT_S - 0.5 < time.monotonic() - t0 < server.CONN_TIMEOUT_S + 2


def test_a_handler_that_raises_ends_only_its_connection(server):
    assert ask(server, "boom") == ""              # closed without a reply
    assert ask(server, "ping") == "re:ping"       # the server goes on
    assert server.thread.is_alive()


def test_several_messages_on_one_connection_in_order(server):
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as c:
        c.sendall(b"a\nb\nc\n")
        buf = b""
        while buf.count(b"\n") < 3:
            buf += c.recv(4096)
    assert buf.decode().split("\n")[:3] == ["re:a", "re:b", "re:c"]


def test_stop_ends_the_accept_loop(qapp):
    s = EchoServer()
    th = threading.Thread(target=s.run, daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while not s.running and time.monotonic() < deadline:
        time.sleep(0.01)
    s.stop()
    th.join(timeout=3)
    assert not th.is_alive()


def test_listen_backlog_is_not_five(qapp, server):
    # 20 connections opened at once with nobody answering yet would overflow a
    # backlog of 5 (refused); with SOMAXCONN every one connects.
    conns = []
    try:
        for _ in range(20):
            conns.append(socket.create_connection(("127.0.0.1", server.port), timeout=2))
    finally:
        for c in conns:
            c.close()
    assert len(conns) == 20


# --- the monitor server -------------------------------------------------------------------


class Recorder:
    def __init__(self, *a, **k):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


@pytest.fixture
def monitor(qapp, monkeypatch, tmp_path):
    from waxx.util.guis import monitor_server_gui as msg
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"dds": {}, "dac": {}, "ttl": {}}))
    s = msg.MonitorUDPServer(config_file_path=str(path), journal_dir=str(tmp_path / "journal"),
                             run_queue_dir=str(tmp_path / "run_queue"))
    yield s
    s.sock.close()


@pytest.mark.parametrize("text", ["run complete", "reset"])
def test_monitor_plain_commands_run_under_the_request_lock(monitor, monkeypatch, text):
    held = []

    def busy():
        held.append(monitor._request_lock._is_owned())
        return "a job is in the slot"             # deferred: no signal, no monitor start

    monkeypatch.setattr(monitor.run_queue, "monitor_busy", busy)
    monkeypatch.setattr(monitor.run_queue, "defer_monitor", lambda *a, **k: None)
    monitor.on_message_received(text)
    assert held == [True]
    assert not monitor._request_lock._is_owned()  # released after


def test_monitor_status_polls_do_not_take_the_request_lock(monitor):
    got = threading.Event()

    def other():
        with monitor._request_lock:
            got.set()
            time.sleep(1.0)

    th = threading.Thread(target=other)
    th.start()
    got.wait(2)
    t0 = time.monotonic()
    monitor.on_message_received("status_json")
    reply = json.loads(monitor.generate_reply("status_json"))
    assert time.monotonic() - t0 < 0.5            # not behind the held lock
    assert "state" in reply
    th.join()
