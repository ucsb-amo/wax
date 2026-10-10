"""The monitor server's request lock: structured requests that may change
something are serialised (the TCP responder and the server window's panels,
through direct_request, can now ask at the same instant).

The server is built in-process with its broadcaster faked; the loops'
start() is a fake that takes a moment before the loop counts as running (the
window a check-then-act race needs) and never launches anything.  Nothing
binds, beacons or runs."""
import contextlib
import json
import os
import threading
import time

import pytest


class Broadcasts:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def server(qapp, monkeypatch, tmp_path):
    from waxx.util.device_state.run_loop import LoopSpec
    from waxx.util.guis import monitor_server_gui as msg
    folder = tmp_path / "experiments"
    folder.mkdir()
    for name in ("loop_a", "loop_b"):
        (folder / f"{name}.py").write_text(f'"""{name}."""\n')
    monkeypatch.setattr(msg, "StateBroadcaster", Broadcasts)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    s = msg.MonitorUDPServer(config_file_path=str(tmp_path / "state.json"),
                             journal_dir=str(tmp_path / "ops_journal"),
                             run_loops=[LoopSpec("loop_a", "Loop A", str(folder / "loop_a.py")),
                                        LoopSpec("loop_b", "Loop B", str(folder / "loop_b.py"))])
    s.started = []
    for key, loop in s.loops.items():
        def fake_start(loop=loop, key=key, **kw):
            time.sleep(0.2)                       # the window between check and act
            with loop._lock:
                loop._s["state"] = "running"
            s.started.append(key)
            return {"status": "ok", "loop": {"key": key, "state": "running"}}
        loop.start = fake_start
    yield s
    s.sock.close()


def _two_starts_at_once(server):
    barrier = threading.Barrier(2)
    replies = {}

    def ask(key):
        barrier.wait()
        replies[key] = json.loads(server.generate_reply(json.dumps(
            {"type": "run_loop", "action": "start", "loop": key, "owner": "person"})))
    threads = [threading.Thread(target=ask, args=(k,)) for k in ("loop_a", "loop_b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return replies


def test_two_loop_starts_at_the_same_instant_start_one_loop(server):
    replies = _two_starts_at_once(server)
    assert len(server.started) == 1
    refused = [r for r in replies.values() if r["status"] != "ok"]
    assert len(refused) == 1 and "one loop at a time" in refused[0]["msg"]


def test_without_the_lock_both_would_start(server):
    """The race is real: the same two requests with the lock replaced by a
    no-op start both loops (this is what the lock prevents)."""
    server._request_lock = contextlib.nullcontext()
    _two_starts_at_once(server)
    assert sorted(server.started) == ["loop_a", "loop_b"]


def test_reads_do_not_wait_for_the_lock(server):
    """status_json and the read requests are answered while another request
    holds the lock (a server restart waits up to 5 s for a loop to stop)."""
    held, release = threading.Event(), threading.Event()

    def hold():
        with server._request_lock:
            held.set()
            release.wait(5)
    t = threading.Thread(target=hold)
    t.start()
    try:
        assert held.wait(5)
        t0 = time.monotonic()
        assert json.loads(server.generate_reply("status_json"))["run_queue"] is not None
        for obj in ({"type": "run_queue", "action": "list"}, {"type": "get_version"},
                    {"type": "get_journal", "n": 5},
                    {"type": "run_loop", "action": "describe", "loop": "loop_a"}):
            assert json.loads(server.generate_reply(json.dumps(obj)))["status"] in ("ok",
                                                                                   "error")
        assert time.monotonic() - t0 < 2.0
    finally:
        release.set()
        t.join(5)

