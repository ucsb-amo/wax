"""Bristol wavemeter running average: the server's GET_AVERAGE and the
per-shot BristolAverageReader.

Offline: the server object is built but never started (no beacon, no TCP
listener, no wavemeter); the reader talks to a throwaway localhost socket
with discovery monkeypatched, so nothing goes to UDP 50099.
"""
import json
import socket
import threading
import time

import pytest

from waxx.control.misc.bristol_wavemeter import _C_LIGHT
import waxx.util.guis.bristol.bristol_wavemeter_client as bclient
from waxx.util.guis.bristol.bristol_wavemeter_client import BristolAverageReader
from waxx.util.guis.bristol.bristol_wavemeter_server import BristolWavemeterServer


F0 = 389.3324e12  # Hz, a realistic Raman frequency


def _server(history):
    s = BristolWavemeterServer(wavemeter_host="0.0.0.0")
    s._history.extend(history)
    s._reading["connected"] = True
    return s


# ---------------------------------------------------------------------------
# server: GET_AVERAGE
# ---------------------------------------------------------------------------

def test_average_of_last_n_fresh_readings():
    now = time.time()
    hist = [(now - 5. + 0.2 * i, F0 + 1e6 * i) for i in range(25)]
    out = _server(hist).get_average(20, 30.)
    assert out["ok"] and out["n_used"] == 20 and out["n_requested"] == 20
    used = [f for _, f in hist[-20:]]
    mean = sum(u - F0 for u in used) / 20 + F0
    assert out["mean_hz"] == pytest.approx(mean, abs=1e-3)
    # sample std of 0..19 MHz offsets (ddof=1)
    import statistics
    assert out["std_hz"] == pytest.approx(statistics.stdev([1e6 * i for i in range(5, 25)]), rel=1e-12)
    assert out["t_last"] == hist[-1][0] and out["t_first"] == hist[5][0]
    assert 0. <= out["age_s"] < 1.


def test_stale_readings_dropped_and_partial_average_reported():
    now = time.time()
    hist = [(now - 100. + i, F0 - 1e9) for i in range(10)]   # old, wrong value
    hist += [(now - 1. + 0.1 * i, F0 + 2e6) for i in range(5)]
    out = _server(hist).get_average(20, 10.)
    assert out["ok"] and out["n_used"] == 5
    assert out["mean_hz"] == pytest.approx(F0 + 2e6, abs=1e-3)
    assert out["std_hz"] == 0.


def test_single_reading_std_zero():
    out = _server([(time.time(), F0)]).get_average(20, 10.)
    assert out["ok"] and out["n_used"] == 1 and out["std_hz"] == 0.


def test_no_fresh_readings_not_ok():
    out = _server([(time.time() - 60., F0)]).get_average(20, 10.)
    assert not out["ok"] and out["n_used"] == 0 and out["mean_hz"] is None
    assert "error" in out


def test_bad_arguments():
    s = _server([(time.time(), F0)])
    assert not s.get_average(0, 10.)["ok"]
    assert not s.get_average(5, 0.)["ok"]


def test_disconnect_clears_history():
    s = _server([(time.time(), F0)] * 3)
    s._disconnect_wavemeter()
    assert not s.get_average(20, 10.)["ok"]


class _FakeWavemeter:
    """Returns the given wavelengths, then stops the poll loop."""

    def __init__(self, server, wavelengths):
        self.server = server
        self.wl = list(wavelengths)
        self.calls = 0

    def get_wavelength(self):
        self.calls += 1
        wl = self.wl.pop(0)
        if not self.wl:
            self.server.running = False
        return wl

    def get_frequency(self):  # must not be used by the poll loop any more
        raise AssertionError("poll loop took a second measurement")


def test_poll_loop_one_measurement_per_reading(monkeypatch):
    s = BristolWavemeterServer(wavemeter_host="0.0.0.0", poll_interval_s=0.)
    wls = [770.1e-9, 770.2e-9, 770.3e-9]
    fake = _FakeWavemeter(s, wls)
    monkeypatch.setattr(s, "_connect_wavemeter", lambda: setattr(s, "_wavemeter", fake))
    s.running = True
    s._poll_loop()
    assert fake.calls == 3
    freqs = [f for _, f in s._history]
    assert freqs == [pytest.approx(_C_LIGHT / w) for w in wls]
    r = s.get_reading()
    assert r["frequency_thz"] == pytest.approx(_C_LIGHT / wls[-1] / 1e12)
    assert r["wavelength_nm"] == pytest.approx(wls[-1] * 1e9)


def test_zero_wavelength_not_averaged(monkeypatch):
    s = BristolWavemeterServer(wavemeter_host="0.0.0.0", poll_interval_s=0.)
    s._history.append((time.time(), F0))
    fake = _FakeWavemeter(s, [0.])
    monkeypatch.setattr(s, "_connect_wavemeter", lambda: setattr(s, "_wavemeter", fake))
    monkeypatch.setattr("time.sleep", lambda t: None)
    s.running = True
    s._poll_loop()
    assert len(s._history) == 0          # error path disconnects and clears
    assert not s.get_reading()["connected"]


def test_handler_parses_get_average():
    s = _server([(time.time(), F0)])
    a, b = socket.socketpair()
    try:
        a.sendall(b"get_average 20 10\n")
        s._handle_client(b, None)
        out = json.loads(a.makefile("rb").readline())
    finally:
        a.close()
    assert out["ok"] and out["n_used"] == 1


def test_handler_bad_get_average_args():
    s = _server([])
    a, b = socket.socketpair()
    try:
        a.sendall(b"GET_AVERAGE twenty\n")
        s._handle_client(b, None)
        out = json.loads(a.makefile("rb").readline())
    finally:
        a.close()
    assert not out["ok"] and out["n_used"] == 0


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

class _FakeEndpoint:
    """Localhost TCP endpoint answering every request with ``reply(line)``."""

    def __init__(self, reply):
        self.reply = reply
        self.lines = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.addr = self.sock.getsockname()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                line = conn.makefile("rb").readline().decode().strip()
                self.lines.append(line)
                out = self.reply(line)
                if out is not None:
                    conn.sendall((out + "\n").encode())

    def close(self):
        self.sock.close()


class _Clock:
    def __init__(self):
        self.t = 1000.

    def __call__(self):
        return self.t


@pytest.fixture
def addr_box(monkeypatch):
    box = {"addr": None}
    monkeypatch.setattr(bclient, "discover", lambda sid, timeout=0.: box["addr"])
    return box


def _reader(clock=None):
    r = BristolAverageReader(discovery_timeout=0.)
    if clock is not None:
        r.latch._clock = clock
        if r.latch.down:   # tripped in __init__ on the real clock
            r.latch._retry_at = clock() + r.latch.retry_after
    return r


def test_reader_returns_server_reply(addr_box):
    reply = {"ok": True, "n_used": 20, "mean_hz": F0, "std_hz": 1e6}
    ep = _FakeEndpoint(lambda line: json.dumps(reply))
    addr_box["addr"] = ep.addr
    try:
        r = _reader()
        assert r.get_average(20, 10.) == reply
        assert ep.lines == ["GET_AVERAGE 20 10"]
    finally:
        ep.close()


def test_reader_no_server_is_fast_and_latched(addr_box, capsys):
    clock = _Clock()
    r = _reader(clock)                      # not discovered: latch tripped
    assert r.latch.down
    t0 = time.monotonic()
    for _ in range(50):
        assert r.get_average(20, 10.) is None
    assert time.monotonic() - t0 < 0.1
    # after the cooldown it looks again and picks up a server that appeared
    ep = _FakeEndpoint(lambda line: json.dumps({"ok": True, "n_used": 1, "mean_hz": F0, "std_hz": 0.}))
    addr_box["addr"] = ep.addr
    try:
        assert r.get_average(20, 10.) is None   # still cooling down
        clock.t += 31.
        assert r.get_average(20, 10.)["n_used"] == 1
        assert not r.latch.down
    finally:
        ep.close()
    out = capsys.readouterr().out
    assert out.count("link down") == 1 and "link back" in out


def test_reader_hung_server_bounded_then_skipped(addr_box):
    ep = _FakeEndpoint(lambda line: (time.sleep(2.), None)[1])
    addr_box["addr"] = ep.addr
    try:
        r = _reader(_Clock())
        r.timeout_s = 0.2
        t0 = time.monotonic()
        assert r.get_average(20, 10.) is None
        assert time.monotonic() - t0 < 0.6
        t0 = time.monotonic()
        assert r.get_average(20, 10.) is None      # latched: no second wait
        assert time.monotonic() - t0 < 0.05
    finally:
        ep.close()


def test_reader_connection_refused(addr_box):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    addr_box["addr"] = s.getsockname()
    s.close()                                    # nothing listening there
    r = _reader(_Clock())
    assert r.get_average(20, 10.) is None
    assert r.latch.down


def test_reader_old_server_trips_latch(addr_box, capsys):
    ep = _FakeEndpoint(lambda line: json.dumps({"error": f"unknown command: {line!r}"}))
    addr_box["addr"] = ep.addr
    try:
        r = _reader(_Clock())
        assert r.get_average(20, 10.) is None
        assert r.latch.down
        assert "restart the Bristol server" in capsys.readouterr().out
    finally:
        ep.close()


def test_reader_no_fresh_readings_returned_not_latched(addr_box, capsys):
    reply = {"ok": False, "n_used": 0, "error": "no readings within 10 s", "connected": False}
    ep = _FakeEndpoint(lambda line: json.dumps(reply))
    addr_box["addr"] = ep.addr
    try:
        r = _reader(_Clock())
        assert r.get_average(20, 10.) == reply
        assert r.get_average(20, 10.) == reply
        assert not r.latch.down
        assert capsys.readouterr().out.count("no fresh readings") == 1
    finally:
        ep.close()


def test_reader_never_raises(addr_box, monkeypatch):
    r = _reader(_Clock())
    monkeypatch.setattr(r, "_get_average", lambda n, a: 1 / 0)
    assert r.get_average(20, 10.) is None
