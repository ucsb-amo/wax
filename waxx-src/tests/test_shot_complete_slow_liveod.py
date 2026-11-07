"""A liveOD that stalls for longer than the client's 5 s default must not kill
the run: SHOT_COMPLETE waits on the same request (never sent twice) and says how
long liveOD took. A liveOD that never answers still ends the run, with an
exception ARTIQ can map (runs 92593 / 92598, 2026-11-05: ConnectionError became
"AttributeError: module 'artiq.coredevice.exceptions' has no attribute
'ConnectionError'").

The fake server is a REP socket on 127.0.0.1 with a random port: no beacon, no
real liveOD."""
import pickle
import threading
import time

import pytest
import zmq


def _rep_server(delay_s, n_replies=1):
    """A REP socket answering SHOT_COMPLETE after ``delay_s``; returns (port,
    received, thread)."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    received = []

    def serve():
        try:
            for _ in range(n_replies):
                if not sock.poll(5000):
                    return
                received.append(pickle.loads(sock.recv()))
                time.sleep(delay_s)
                sock.send(pickle.dumps({"ok": True, "reset_requested": False,
                                        "adjust_values": {}}))
        finally:
            sock.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return port, received, t


def _client(port, timeout_ms):
    from waxx.util.live_od.live_od_client import LiveODClient
    c = LiveODClient.__new__(LiveODClient)
    c._ip, c._port, c._timeout_ms = "127.0.0.1", port, timeout_ms
    c._context = None
    c._socket = None
    c._run_token = ""
    c.last_reset_requested = False
    c.last_adjust_values = {}
    c._grab_failure_warned = False
    c._writer_fault_warned = False
    c._rediscover = lambda: None          # never the beacon
    return c


def test_a_reply_slower_than_the_default_timeout_is_waited_for(monkeypatch, capsys):
    from waxx.util.live_od import live_od_client as lc
    monkeypatch.setattr(lc, "SHOT_COMPLETE_TIMEOUT_MS", 3000)
    monkeypatch.setattr(lc, "SLOW_SHOT_COMPLETE_S", 0.3)
    port, received, t = _rep_server(delay_s=0.8)
    c = _client(port, timeout_ms=200)        # the default would give up at 0.2 s
    assert c.shot_complete(0, 10, {"x": 1.0}) is False
    t.join(2)
    assert len(received) == 1                # one request, never re-sent
    assert received[0]["tag"] == "SHOT_COMPLETE" and received[0]["shot_idx"] == 0
    out = capsys.readouterr().out
    assert "to answer SHOT_COMPLETE for shot 1/10" in out
    # the socket's own timeout is back to the default for the next request
    assert c._socket.getsockopt(zmq.RCVTIMEO) == 200


def test_a_fast_reply_prints_nothing(monkeypatch, capsys):
    from waxx.util.live_od import live_od_client as lc
    monkeypatch.setattr(lc, "SHOT_COMPLETE_TIMEOUT_MS", 3000)
    port, received, t = _rep_server(delay_s=0.0)
    c = _client(port, timeout_ms=200)
    assert c.shot_complete(0, 10, {}) is False
    t.join(2)
    assert "SHOT_COMPLETE" not in capsys.readouterr().out


def test_a_liveod_that_never_answers_still_raises(monkeypatch):
    from waxx.util.live_od import live_od_client as lc
    monkeypatch.setattr(lc, "SHOT_COMPLETE_TIMEOUT_MS", 300)
    port, received, t = _rep_server(delay_s=2.0)
    c = _client(port, timeout_ms=100)
    with pytest.raises(ConnectionError):
        c.shot_complete(0, 10, {})
    t.join(3)


def test_the_experiment_re_raises_a_lost_liveod_as_a_runtime_error():
    """RuntimeError is in artiq.coredevice.exceptions; ConnectionError is not."""
    import artiq.coredevice.exceptions as artiq_exc
    from waxx.base.expt import Expt
    assert hasattr(artiq_exc, "RuntimeError") and not hasattr(artiq_exc, "ConnectionError")

    class Gone:
        def shot_complete(self, *a, **k):
            raise ConnectionError("[LiveODClient] No response from liveOD server")

    e = Expt.__new__(Expt)
    e._shot_complete_count = 0
    e._N_shots_total = 5
    e.scan_xvars = []
    e.live_od_client = Gone()
    e._shot_conditions = lambda: {}
    with pytest.raises(RuntimeError, match="did not answer SHOT_COMPLETE for shot 1/5") as info:
        e._notify_shot_complete()
    assert isinstance(info.value.__cause__, ConnectionError)
