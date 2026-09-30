"""DailyCsvLog: append-only, one file per local day, rows survive a failed write."""

import csv
import time

from waxx.util.daily_csv import DailyCsvLog, local_day

HEADER = ["epoch", "value"]


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _epoch(y, mo, d, h, mi, s):
    return time.mktime((y, mo, d, h, mi, s, 0, 0, -1))


def test_add_does_no_io(tmp_path):
    log = DailyCsvLog(tmp_path / "hist", HEADER)
    log.add(_epoch(2026, 9, 30, 12, 0, 0), ["1", "a"])
    assert not (tmp_path / "hist").exists()
    assert log.pending == 1


def test_appends_across_flushes_with_one_header(tmp_path):
    log = DailyCsvLog(tmp_path, HEADER)
    t = _epoch(2026, 9, 30, 12, 0, 0)
    log.add(t, [t, "a"])
    assert log.flush() == 1
    log.add(t + 1, [t + 1, "b"])
    log.add(t + 2, [t + 2, "c"])
    assert log.flush() == 2
    rows = _read(tmp_path / "2026-09-30.csv")
    assert rows[0] == HEADER
    assert [r[1] for r in rows[1:]] == ["a", "b", "c"]
    assert log.pending == 0


def test_existing_file_is_appended_not_rewritten(tmp_path):
    path = tmp_path / "2026-09-30.csv"
    path.write_text("epoch,value\n0,earlier\n", encoding="utf-8")
    log = DailyCsvLog(tmp_path, HEADER)
    log.add(_epoch(2026, 9, 30, 12, 0, 0), ["1", "new"])
    log.flush()
    assert _read(path) == [HEADER, ["0", "earlier"], ["1", "new"]]


def test_midnight_splits_files(tmp_path):
    log = DailyCsvLog(tmp_path, HEADER)
    before = _epoch(2026, 9, 30, 23, 59, 59)
    after = _epoch(2026, 10, 1, 0, 0, 1)
    assert local_day(before) == "2026-09-30" and local_day(after) == "2026-10-01"
    log.add(before, ["x", "before"])
    log.add(after, ["y", "after"])
    assert log.flush() == 2
    assert _read(tmp_path / "2026-09-30.csv")[1:] == [["x", "before"]]
    assert _read(tmp_path / "2026-10-01.csv") == [HEADER, ["y", "after"]]


def test_failed_write_keeps_rows_in_order(tmp_path):
    blocker = tmp_path / "hist"
    blocker.write_text("not a folder")        # mkdir fails: a plain file is in the way
    log = DailyCsvLog(blocker, HEADER)
    t = _epoch(2026, 9, 30, 12, 0, 0)
    log.add(t, ["1", "a"])
    assert log.flush() == 0
    assert log.pending == 1
    log.add(t + 1, ["2", "b"])                # queued during the outage
    blocker.unlink()
    assert log.flush() == 2
    assert [r[1] for r in _read(blocker / "2026-09-30.csv")[1:]] == ["a", "b"]


def test_full_queue_drops_oldest_and_logs_count(tmp_path, caplog):
    log = DailyCsvLog(tmp_path, HEADER, max_pending=3)
    t = _epoch(2026, 9, 30, 12, 0, 0)
    for i in range(5):
        log.add(t + i, [str(i), "v"])
    assert log.pending == 3
    with caplog.at_level("WARNING"):
        assert log.flush() == 3
    assert "dropped the 2 oldest rows" in caplog.text
    assert [r[0] for r in _read(tmp_path / "2026-09-30.csv")[1:]] == ["2", "3", "4"]


def test_stop_does_a_final_flush(tmp_path):
    log = DailyCsvLog(tmp_path, HEADER)
    log.start(interval_s=3600.)
    log.add(_epoch(2026, 9, 30, 12, 0, 0), ["1", "last"])
    log.stop(timeout=5.)
    assert _read(tmp_path / "2026-09-30.csv")[1:] == [["1", "last"]]
