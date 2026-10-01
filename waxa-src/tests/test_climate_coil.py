"""waxa.climate.sender / waxa.climate.coil against a canned trapper.

No network: ``ZabbixSender._exchange`` is replaced by a fake that decodes
the request and answers like Zabbix 7 does.  History files go in tmp_path.
"""
from __future__ import annotations

import datetime as _dt
import json
import struct

import pytest

from waxa.climate import TrapperValue, ZabbixSender
from waxa.climate.coil import CoilTemperaturePusher, last_pushed_clock, latest_reading, read_day

HEADER = "epoch,local_time,temperature_plc_k,temperature_c,flow1_v,flow2_v,flow3_v,flow4_v,plc_tripped\n"


class FakeSender(ZabbixSender):
    """Records every entry sent; ``known_keys`` are the trapper items that exist."""

    def __init__(self, known_keys=("k.coil.temperature",), up=True):
        super().__init__("fake", 10051)
        self.known_keys = set(known_keys)
        self.up = up
        self.sent: list[dict] = []
        self.requests = 0

    def _exchange(self, packet: bytes) -> bytes:
        if not self.up:
            raise ConnectionError("down")
        assert packet[:5] == b"ZBXD\x01"
        (n, _) = struct.unpack("<II", packet[5:13])
        body = json.loads(packet[13:13 + n])
        assert body["request"] == "sender data"
        self.requests += 1
        ok = [e for e in body["data"] if e["key"] in self.known_keys]
        self.sent += ok
        info = (f"processed: {len(ok)}; failed: {len(body['data']) - len(ok)}; "
                f"total: {len(body['data'])}; seconds spent: 0.000100")
        return self.pack({"response": "success", "info": info})


def _row(epoch, temp_c):
    t = "" if temp_c is None else f"{temp_c + 273.0:.5f}"
    c = "" if temp_c is None else f"{temp_c:.3f}"
    local = _dt.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
    return f"{epoch:.3f},{local},{t},{c},5.4,6.9,5.6,7.1,0\n"


def _day_path(tmp_path, epoch):
    return tmp_path / f"{_dt.date.fromtimestamp(epoch).isoformat()}.csv"


T0 = _dt.datetime(2026, 9, 30, 12, 0, 0).timestamp()


# ---------------------------------------------------------------------------
# sender
# ---------------------------------------------------------------------------

def test_pack_unpack_roundtrip():
    body = {"request": "sender data", "data": [{"host": "h", "key": "k", "value": "1"}]}
    assert ZabbixSender.unpack(ZabbixSender.pack(body)) == body


def test_send_counts_and_clock():
    s = FakeSender()
    r = s.send([TrapperValue("K", "k.coil.temperature", 21.5, T0 + 0.25),
                TrapperValue("K", "no.such.item", 1)])
    assert (r.processed, r.failed, r.total, r.ok) == (1, 1, 2, False)
    e = s.sent[0]
    assert e["value"] == "21.5" and e["clock"] == int(T0) and e["ns"] == 250_000_000


def test_send_batches():
    s = FakeSender()
    r = s.send([TrapperValue("K", "k.coil.temperature", i, T0 + i) for i in range(600)])
    assert s.requests == 3 and r.processed == 600 and r.ok


def test_unpack_rejects_garbage():
    with pytest.raises(ValueError):
        ZabbixSender.unpack(b"HTTP/1.1 400")


# ---------------------------------------------------------------------------
# coil pusher
# ---------------------------------------------------------------------------

def test_read_day_and_latest_skip_blank_temperature(tmp_path):
    _day_path(tmp_path, T0).write_text(HEADER + _row(T0, 20.0) + _row(T0 + 1, None) + _row(T0 + 2, 21.0))
    rows = read_day(tmp_path, _dt.date.fromtimestamp(T0))
    assert [r.temperature_c for r in rows] == [20.0, 21.0]


def test_first_push_sends_only_newest_then_only_new(tmp_path):
    f = _day_path(tmp_path, T0)
    f.write_text(HEADER + "".join(_row(T0 + i * 20, 20.0 + i) for i in range(5)))
    s = FakeSender()
    p = CoilTemperaturePusher(s, tmp_path, host="K", min_interval_s=10)
    r = p.push_new(now=T0 + 100)
    assert r.processed == 1 and s.sent[-1]["value"] == "24.0"
    assert p.push_new(now=T0 + 110) is None
    with open(f, "a") as fh:
        fh.write(_row(T0 + 100, 30.0) + _row(T0 + 105, 31.0) + _row(T0 + 120, 32.0))
    p.push_new(now=T0 + 130)
    # 105 is within min_interval_s of 100 and is dropped
    assert [e["value"] for e in s.sent] == ["24.0", "30.0", "32.0"]


def test_backfill(tmp_path):
    _day_path(tmp_path, T0).write_text(HEADER + "".join(_row(T0 + i * 60, 20.0 + i) for i in range(10)))
    s = FakeSender()
    CoilTemperaturePusher(s, tmp_path, host="K", backfill_s=180).push_new(now=T0 + 600)
    assert [e["value"] for e in s.sent] == ["26.0", "27.0", "28.0", "29.0"]


def test_partial_last_line_waits(tmp_path):
    f = _day_path(tmp_path, T0)
    f.write_text(HEADER + _row(T0, 20.0))
    s = FakeSender()
    p = CoilTemperaturePusher(s, tmp_path, host="K")
    p.push_new(now=T0 + 1)
    full = _row(T0 + 30, 22.0)
    with open(f, "a") as fh:
        fh.write(full[:10])
    assert p.push_new(now=T0 + 31) is None
    with open(f, "a") as fh:
        fh.write(full[10:])
    p.push_new(now=T0 + 32)
    assert [e["value"] for e in s.sent] == ["20.0", "22.0"]


def test_midnight_rollover(tmp_path):
    late = _dt.datetime(2026, 9, 30, 23, 59, 0).timestamp()
    f1 = _day_path(tmp_path, late)
    f1.write_text(HEADER + _row(late, 20.0))
    s = FakeSender()
    p = CoilTemperaturePusher(s, tmp_path, host="K")
    p.push_new(now=late + 1)
    with open(f1, "a") as fh:
        fh.write(_row(late + 50, 21.0))          # written just before midnight
    f2 = _day_path(tmp_path, late + 120)
    f2.write_text(HEADER + _row(late + 120, 22.0))
    p.push_new(now=late + 130)
    assert [e["value"] for e in s.sent] == ["20.0", "21.0", "22.0"]


def test_unreachable_keeps_readings(tmp_path):
    f = _day_path(tmp_path, T0)
    f.write_text(HEADER + _row(T0, 20.0))
    s = FakeSender(up=False)
    p = CoilTemperaturePusher(s, tmp_path, host="K")
    with pytest.raises(ConnectionError):
        p.push_new(now=T0 + 1)
    with open(f, "a") as fh:
        fh.write(_row(T0 + 30, 21.0))
    s.up = True
    p.push_new(now=T0 + 31)
    assert [e["value"] for e in s.sent] == ["20.0", "21.0"]


def test_missing_item_warns_once(tmp_path, caplog):
    f = _day_path(tmp_path, T0)
    f.write_text(HEADER + _row(T0, 20.0))
    p = CoilTemperaturePusher(FakeSender(known_keys=()), tmp_path, host="K")
    r = p.push_new(now=T0 + 1)
    assert r.failed == 1
    with open(f, "a") as fh:
        fh.write(_row(T0 + 30, 21.0))
    p.push_new(now=T0 + 31)
    assert sum("trapper item" in m for m in caplog.messages) == 1


def test_latest_reading(tmp_path):
    today = _dt.datetime.combine(_dt.date.today(), _dt.time(1, 0)).timestamp()
    _day_path(tmp_path, today).write_text(HEADER + _row(today, 19.5))
    assert latest_reading(tmp_path).temperature_c == 19.5


def test_resume_after_sends_only_newer(tmp_path):
    _day_path(tmp_path, T0).write_text(HEADER + "".join(_row(T0 + i * 60, 20.0 + i) for i in range(6)))
    s = FakeSender()
    p = CoilTemperaturePusher(s, tmp_path, host="K", resume_after=T0 + 180)
    p.push_new(now=T0 + 400)
    assert [e["value"] for e in s.sent] == ["24.0", "25.0"]


def test_resume_after_nothing_newer(tmp_path):
    _day_path(tmp_path, T0).write_text(HEADER + _row(T0, 20.0))
    s = FakeSender()
    assert CoilTemperaturePusher(s, tmp_path, host="K", resume_after=T0 + 5).push_new(now=T0 + 10) is None


class _FakeAPI:
    def __init__(self, rows):
        self.rows = rows

    def call(self, method, params):
        assert method == "item.get" and params["filter"] == {"key_": "k.coil.temperature"}
        return self.rows


def test_last_pushed_clock():
    assert last_pushed_clock(api=_FakeAPI([{"itemid": "1", "lastclock": "1790800000"}])) == 1790800000.0
    assert last_pushed_clock(api=_FakeAPI([{"itemid": "1", "lastclock": "0"}])) is None
    assert last_pushed_clock(api=_FakeAPI([])) is None
