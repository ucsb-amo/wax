"""Stuck-sensor logging is rate limited; detection, resets and data are not.

The read loop runs synchronously against a fake sensor (no COM port, no
socket, no beacon) and a fake clock.  The sensor returns the same counts for
a long stretch (one stuck episode, many resets), then changing counts.
"""

from __future__ import annotations

import logging
import types

import pytest

from waxx.util.guis.HMR_magnetometer import hmr_magnetometer_server as srv_mod

STUCK_READS = 21 * 1000         # ~2500 fake s stuck: several summary intervals
GOOD_READS = 700                # 84 fake s: longer than STUCK_RECOVERY_QUIET_S
READ_DT_S = 0.12                # fake seconds per poll, like the real loop


class _FakeSerial:
    is_open = True


def _make_fake_reader(server, clock, counters):
    class _FakeReader:
        def __init__(self, port, baud, device_id, timeout=1.0):
            self.ser = None

        def open(self):
            counters["opens"] += 1
            self.ser = _FakeSerial()

        def setup(self):
            pass

        def close(self):
            self.ser = None

        def read_one(self):
            n = counters["reads"]
            counters["reads"] += 1
            clock[0] += READ_DT_S
            if n >= STUCK_READS + GOOD_READS:
                server.stop_event.set()
            if n < STUCK_READS:
                return (1500, -300, 4500)            # frozen counts
            return (1500 + n % 7, -300, 4500 + n % 3)  # changing again

    return _FakeReader


@pytest.fixture
def server(monkeypatch):
    def _no_serial(*a, **k):
        raise AssertionError("the test must not open a serial port")

    monkeypatch.setattr(srv_mod.serial, "Serial", _no_serial)
    s = srv_mod.MagnetometerServer(serial_port="LRS-FAKE", poll_interval=0.0)
    monkeypatch.setattr(s, "_start_beacon", lambda: (_ for _ in ()).throw(AssertionError("no beacon")))
    return s


def test_stuck_episode_logs_first_event_summaries_and_recovery(server, monkeypatch, caplog):
    clock = [5000.0]
    counters = {"reads": 0, "opens": 0}
    import time as real_time
    fake_time = types.SimpleNamespace(
        time=real_time.time, monotonic=lambda: clock[0], sleep=lambda _s: None)
    monkeypatch.setattr(srv_mod, "time", fake_time)
    monkeypatch.setattr(srv_mod, "HMR2300Reader", _make_fake_reader(server, clock, counters))

    with caplog.at_level(logging.DEBUG, logger=srv_mod.__name__):
        server._read_loop()

    recs = [r for r in caplog.records if r.name == srv_mod.__name__]
    visible = [r for r in recs if r.levelno >= logging.INFO]     # what the dashboard gets
    msgs = [r.getMessage() for r in visible]

    # Detection and reset unchanged: a reset every MAX_STUCK_SAME_VALUES + 1
    # identical readings, one serial (re)open per reset plus the first open.
    n_stuck = STUCK_READS // (srv_mod.MAX_STUCK_SAME_VALUES + 1)
    assert counters["opens"] == 1 + n_stuck
    # Data unchanged: every reading except the one that trips each detection
    # goes into the history the server serves.
    total_reads = counters["reads"]
    assert len(server.history) == min(srv_mod.MAX_HISTORY, total_reads - n_stuck)

    # The first stuck event and the first reading after it: in full.
    assert sum("stuck for" in m and "resetting serial" in m for m in msgs) == 1
    assert sum(m.startswith("First reading after a stuck reset") for m in msgs) == 1
    # Later resets: DEBUG only, one per reset.
    debug_resets = [r for r in recs if r.levelno == logging.DEBUG and "stuck reset #" in r.getMessage()]
    assert len(debug_resets) == n_stuck - 1
    # Periodic summaries, one per interval of stuck time.
    stuck_time_s = STUCK_READS * READ_DT_S
    summaries = [m for m in msgs if m.startswith("Sensor still stuck")]
    expected = int(stuck_time_s // srv_mod.STUCK_SUMMARY_INTERVAL_S)
    assert expected >= 2
    assert abs(len(summaries) - expected) <= 1
    assert "(1500, -300, 4500)" in summaries[0]
    # Recovery is stated once, with the episode's size.
    recovered = [m for m in msgs if m.startswith("Sensor readings changing again")]
    assert len(recovered) == 1 and f"{n_stuck} stuck reset(s)" in recovered[0]
    assert server._stuck_episode is None
    # Routine reconnect chatter stays out of the dashboard log.
    assert sum(m.startswith("Sensor reconnected") for m in msgs) <= 2
    assert len(visible) < 20, msgs


def test_a_sensor_that_moves_a_little_after_each_reset_stays_one_episode(server, monkeypatch, caplog):
    """2026-10-05: after each reset the counts moved by a count or two for
    ~21 readings, then stuck again. That is one episode, not a recovery and a
    new episode every few seconds."""
    clock = [5000.0]
    counters = {"reads": 0, "opens": 0}
    cycle = 2 * (srv_mod.MAX_STUCK_SAME_VALUES + 1)     # changing, then frozen
    n_cycles = 400                                      # ~1.6 h of fake time

    class _FlapReader:
        def __init__(self, port, baud, device_id, timeout=1.0):
            self.ser = None

        def open(self):
            counters["opens"] += 1
            self.ser = _FakeSerial()

        def setup(self):
            pass

        def close(self):
            self.ser = None

        def read_one(self):
            n = counters["reads"]
            counters["reads"] += 1
            clock[0] += READ_DT_S
            if n >= cycle * n_cycles:
                server.stop_event.set()
            k = n % cycle
            if k < srv_mod.MAX_STUCK_SAME_VALUES + 1:
                return (-6660, 3499, 3343 + k % 3)     # moving a little
            return (-6660, 3499, 3343)                 # frozen

    import time as real_time
    monkeypatch.setattr(srv_mod, "time", types.SimpleNamespace(
        time=real_time.time, monotonic=lambda: clock[0], sleep=lambda _s: None))
    monkeypatch.setattr(srv_mod, "HMR2300Reader", _FlapReader)

    with caplog.at_level(logging.DEBUG, logger=srv_mod.__name__):
        server._read_loop()

    msgs = [r.getMessage() for r in caplog.records
            if r.name == srv_mod.__name__ and r.levelno >= logging.INFO]
    assert counters["opens"] > n_cycles // 2              # it kept resetting (unchanged)
    assert not any(m.startswith("Sensor readings changing again") for m in msgs), msgs[:10]
    assert sum("stuck for" in m and "resetting serial" in m for m in msgs) == 1
    assert len(msgs) < 30, msgs[:30]


def test_dashboard_stream_handler_is_info(monkeypatch, tmp_path):
    root = logging.getLogger()
    saved = list(root.handlers), root.level
    root.handlers[:] = []
    try:
        srv_mod._configure_logging(str(tmp_path / "lrs_server.log"))
        streams = [h for h in root.handlers
                   if type(h) is logging.StreamHandler]
        files = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert streams and all(h.level == logging.INFO for h in streams)
        assert files and all(h.level == logging.DEBUG for h in files)
    finally:
        for h in root.handlers:
            h.close()
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
