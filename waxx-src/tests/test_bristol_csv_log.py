"""Bristol server history log: one CSV row per window, mean/std/n, no row for an outage.

Offline: the server is built but never started (no beacon, no listener, no wavemeter).
"""
import csv
import statistics

import pytest

from waxx.util.guis.bristol.bristol_wavemeter_server import BristolWavemeterServer, LOG_HEADER

F0 = 389.3324e12  # Hz


def _server(tmp_path):
    return BristolWavemeterServer(wavemeter_host="0.0.0.0", log_dir=tmp_path, log_interval_s=1.0)


def _rows(tmp_path):
    files = sorted(tmp_path.glob("*.csv"))
    assert len(files) == 1
    return list(csv.reader(open(files[0])))


def test_window_row_is_mean_std_n(tmp_path):
    s = _server(tmp_path)
    t0 = 1_800_000_000.
    polls = [(t0 + 0.1 * i, F0 + 1e3 * i) for i in range(11)]   # spans 1.0 s
    for t, f in polls:
        s._log_poll(t, f)
    assert s._csv.flush() == 1
    header, row = _rows(tmp_path)
    assert tuple(header) == LOG_HEADER
    offs = [1e3 * i for i in range(11)]
    assert float(row[0]) == pytest.approx(t0 + 0.5, abs=1e-3)
    assert float(row[1]) * 1e12 == pytest.approx(F0 + statistics.fmean(offs), abs=1e3)
    assert float(row[2]) == pytest.approx(statistics.stdev(offs), abs=1.)
    assert int(row[3]) == 11


def test_partial_window_at_disconnect_is_dropped(tmp_path):
    s = _server(tmp_path)
    t0 = 1_800_000_000.
    for i in range(5):
        s._log_poll(t0 + 0.1 * i, F0)
    s._disconnect_wavemeter()
    assert s._log_window == []
    assert s._csv.flush() == 0


def test_no_log_dir_logs_nothing():
    s = BristolWavemeterServer(wavemeter_host="0.0.0.0")
    s._log_poll(1.0, F0)
    s._log_poll(3.0, F0)
    assert s._csv is None
