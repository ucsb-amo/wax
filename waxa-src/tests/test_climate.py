"""waxa.climate against a canned Zabbix.

The fake transport replays the shapes the real server returned on
2026-09-24 (Zabbix 7.0.30, guest login, host "K").  No network, no data
files, nothing written.
"""
from __future__ import annotations

import datetime as _dt
import json

import numpy as np
import pytest

from waxa.climate import (
    ClimateClient,
    ClimateSeries,
    ZabbixAPI,
    ZabbixError,
    climate_for_run,
    run_start_time,
    shot_times,
    to_unix,
)

# ---------------------------------------------------------------------------
# Canned server
# ---------------------------------------------------------------------------

HOSTS = [{"hostid": "10539", "name": "K"}, {"hostid": "10534", "name": "Li"}]

K_ITEMS = [
    {"itemid": "44037", "hostid": "10539", "name": "Machine Table", "key_": "MachineTableTemp",
     "value_type": "0", "units": "F", "delay": "1m", "lastvalue": "78.8", "lastclock": "1790261357"},
    {"itemid": "44041", "hostid": "10539", "name": "Vent Airflow", "key_": "Airflow1",
     "value_type": "3", "units": "", "delay": "1m", "lastvalue": "51", "lastclock": "1790261357"},
    {"itemid": "44045", "hostid": "10539", "name": "K Watchdog Humidity", "key_": "Humidity2.WatchdogK",
     "value_type": "0", "units": "%", "delay": "1m", "lastvalue": "34", "lastclock": "1790261357"},
    {"itemid": "99999", "hostid": "10539", "name": "Contact", "key_": "system.contact",
     "value_type": "1", "units": "", "delay": "15m", "lastvalue": "someone", "lastclock": "0"},
]

T0 = 1790260000  # a fixed epoch


def _history_rows(itemid, t_from, t_till):
    """One sample a minute on the minute, temp 78.0 + 0.1/min, airflow 50 + minute%5."""
    rows = []
    for k in range(-5, 200):
        t = T0 + 60 * k
        if t < t_from or t > t_till:
            continue
        if itemid == "44037":
            v = f"{78.0 + 0.1 * k:.1f}"
        elif itemid == "44041":
            v = str(50 + k % 5)
        else:
            v = "34"
        rows.append({"itemid": itemid, "clock": str(t), "value": v, "ns": "0"})
    return rows


class FakeServer:
    def __init__(self):
        self.calls = []
        self.tokens_issued = 0
        self.expire_next = False

    def handle(self, payload, token):
        m, p = payload["method"], payload["params"]
        self.calls.append((m, json.loads(json.dumps(p)), token))
        if m == "apiinfo.version":
            return {"result": "7.0.30"}
        if m == "user.login":
            assert p == {"username": "guest", "password": ""}
            self.tokens_issued += 1
            return {"result": f"tok{self.tokens_issued}"}
        if token is None or token != f"tok{self.tokens_issued}":
            return {"error": {"code": -32602, "message": "Invalid params.",
                              "data": "Not authorised."}}
        if self.expire_next:
            self.expire_next = False
            return {"error": {"code": -32602, "message": "Invalid params.",
                              "data": "Session terminated, re-login, please."}}
        if m == "user.logout":
            return {"result": True}
        if m == "host.get":
            return {"result": HOSTS}
        if m == "item.get":
            assert p["hostids"] == ["10539"]
            return {"result": K_ITEMS}
        if m == "history.get":
            (itemid,) = p["itemids"]
            it = next(i for i in K_ITEMS if i["itemid"] == itemid)
            # The real server returns nothing if the history table is wrong.
            if int(it["value_type"]) != p["history"]:
                return {"result": []}
            return {"result": _history_rows(itemid, p["time_from"], p["time_till"])}
        if m == "trend.get":
            (itemid,) = p["itemids"]
            return {"result": [
                {"itemid": itemid, "clock": str(T0 + 3600), "num": "60",
                 "value_min": "77.0", "value_avg": "78.0", "value_max": "79.0"},
                {"itemid": itemid, "clock": str(T0), "num": "58",
                 "value_min": "76.0", "value_avg": "77.5", "value_max": "78.5"},
            ]}
        return {"error": {"code": -32601, "message": "Method not found.", "data": m}}


@pytest.fixture
def server(monkeypatch):
    srv = FakeServer()

    def fake_post(self, payload, token):
        env = {"jsonrpc": "2.0", "id": payload["id"]}
        env.update(srv.handle(payload, token))
        return env

    monkeypatch.setattr(ZabbixAPI, "_post", fake_post)
    return srv


@pytest.fixture
def cc(server):
    return ClimateClient()


# ---------------------------------------------------------------------------
# ZabbixAPI
# ---------------------------------------------------------------------------

def test_version_needs_no_login(server):
    api = ZabbixAPI()
    assert api.version() == "7.0.30"
    assert server.tokens_issued == 0


def test_call_logs_in_lazily_and_sends_bearer_token(server):
    api = ZabbixAPI()
    api.call("host.get", {"output": ["hostid", "name"]})
    methods = [c[0] for c in server.calls]
    assert methods == ["user.login", "host.get"]
    assert server.calls[1][2] == "tok1"


def test_expired_session_is_renewed_once_and_call_retried(server):
    api = ZabbixAPI()
    api.call("host.get", {})
    server.expire_next = True
    out = api.call("host.get", {})
    assert out == HOSTS
    assert server.tokens_issued == 2
    assert api.token == "tok2"


def test_other_errors_raise_zabbix_error(server):
    api = ZabbixAPI()
    with pytest.raises(ZabbixError) as ei:
        api.call("nope.get", {})
    assert "nope.get" in str(ei.value)


# ---------------------------------------------------------------------------
# ClimateClient
# ---------------------------------------------------------------------------

def test_items_default_to_numeric_and_lookup_is_by_name_or_key(cc):
    names = [it.name for it in cc.items()]
    assert names == ["Machine Table", "Vent Airflow", "K Watchdog Humidity"]
    assert cc.item("machine table").itemid == "44037"
    assert cc.item("Airflow1").itemid == "44041"
    assert cc.item("humidity").itemid == "44045"      # unique substring
    with pytest.raises(KeyError):
        cc.item("does not exist")


def test_history_uses_the_items_value_type_and_converts_f_to_c(cc, server):
    s = cc.history("Machine Table", T0, T0 + 600)
    call = next(c for c in server.calls if c[0] == "history.get")
    assert call[1]["history"] == 0
    assert s.units == "C" and s.source_units == "F"
    assert len(s) == 11
    # first sample is 78.0 F
    assert s.v[0] == pytest.approx((78.0 - 32) * 5 / 9)
    assert s.to_fahrenheit().v[0] == pytest.approx(78.0)


def test_integer_items_read_the_uint_history_table(cc, server):
    s = cc.history("Vent Airflow", T0, T0 + 240)
    call = next(c for c in server.calls if c[0] == "history.get")
    assert call[1]["history"] == 3
    assert len(s) == 5
    assert s.units == "" and s.source_units == ""


def test_temperature_unit_f_leaves_values_untouched(server):
    cc = ClimateClient(temperature_unit="F")
    s = cc.history("Machine Table", T0, T0)
    assert s.units == "F"
    assert s.v[0] == pytest.approx(78.0)
    snap = cc.snapshot()
    assert snap["Machine Table"][0] == pytest.approx(78.8)


def test_snapshot_converts_temperatures_and_skips_text_items(cc):
    snap = cc.snapshot()
    assert set(snap) == {"Machine Table", "Vent Airflow", "K Watchdog Humidity"}
    assert snap["Machine Table"][0] == pytest.approx((78.8 - 32) * 5 / 9)
    assert snap["Vent Airflow"] == (51.0, 1790261357.0)


def test_long_windows_are_chunked_and_seams_deduplicated(cc, server):
    s = cc.history("Machine Table", T0, T0 + 3 * 3600, chunk_s=3600.0)
    n_calls = sum(1 for c in server.calls if c[0] == "history.get")
    assert n_calls == 3
    assert np.all(np.diff(s.t) > 0)
    assert len(s) == 3 * 60 + 1


def test_trends_are_sorted_with_min_max_in_extra(cc):
    s = cc.trends("Machine Table", T0 - 10, T0 + 7200)
    assert s.kind == "trend"
    assert list(s.t) == [T0, T0 + 3600]
    assert s.units == "C"
    assert s.extra["max"][1] == pytest.approx((79.0 - 32) * 5 / 9)
    assert list(s.extra["num"]) == [58, 60]


def test_room_fetches_every_numeric_sensor(cc):
    room = cc.room(T0, T0 + 60)
    assert set(room) == {"Machine Table", "Vent Airflow", "K Watchdog Humidity"}
    assert all(len(s) == 2 for s in room.values())


def test_to_unix_accepts_strings_datetimes_and_numbers():
    d = _dt.datetime(2026, 9, 24, 8, 30)
    assert to_unix(d) == d.timestamp()
    assert to_unix("2026-09-24 08:30") == d.timestamp()
    assert to_unix("2026-09-24") == _dt.datetime(2026, 9, 24).timestamp()
    assert to_unix(12.5) == 12.5
    with pytest.raises(ValueError):
        to_unix("yesterday")


# ---------------------------------------------------------------------------
# ClimateSeries.at
# ---------------------------------------------------------------------------

def _series(t, v, units="C"):
    from waxa.climate import ClimateItem
    it = ClimateItem("1", "1", "K", "x", "x", 0, units)
    return ClimateSeries(it, np.asarray(t, float), np.asarray(v, float), units, units)


def test_at_nearest_and_interp_with_gap_masking():
    s = _series([0, 60, 120, 600], [10, 20, 30, 40])
    q = np.array([[10, 50], [125, 310]])
    near = s.at(q, "nearest", max_gap_s=180)
    assert near[0, 0] == 10 and near[0, 1] == 20 and near[1, 0] == 30
    assert np.isnan(near[1, 1])                      # 190 s from nearest sample: outage
    assert s.at([300], "nearest", max_gap_s=180)[0] == 30   # exactly at the limit is kept
    lin = s.at([30], "interp", max_gap_s=None)
    assert lin[0] == pytest.approx(15.0)
    assert np.isnan(_series([], []).at([1.0])[0])   # empty series -> NaN
    with pytest.raises(ValueError):
        s.at([0], method="cubic")


def test_series_window_and_summary():
    s = _series([0, 60, 120], [1.0, 2.0, np.nan])
    w = s.window(30, 200)
    assert list(w.t) == [60, 120]
    st = s.summary()
    assert st["n"] == 2 and st["mean"] == 1.5 and st["min"] == 1.0


# ---------------------------------------------------------------------------
# Attaching to runs
# ---------------------------------------------------------------------------

class _RunInfo:
    def __init__(self, filepath):
        self.filepath = filepath
        self.run_datetime = _dt.datetime.now().timetuple()   # load time: must be ignored


class _AD:
    """Duck-typed atomdata: (*xvardims,) per-shot arrays."""

    def __init__(self, t_atoms=None, t_end=None, xvardims=(4,), filepath=None):
        self.xvardims = np.array(xvardims)
        if t_atoms is not None:
            self.img_timestamp_atoms = np.asarray(t_atoms)
        if t_end is not None:
            self.timestamp_shot_end = np.asarray(t_end)
        self.run_info = _RunInfo(filepath or [])


def test_shot_times_prefers_image_stamps_then_shot_end_then_filename():
    t = T0 + np.arange(4) * 7.0
    ad = _AD(t_atoms=t)
    got, src = shot_times(ad, return_source=True)
    assert src == "img_timestamp_atoms" and np.allclose(got, t)

    ad = _AD(t_atoms=np.zeros(4), t_end=t)     # zeros = no images: padding, not a clock
    got, src = shot_times(ad, return_source=True)
    assert src == "timestamp_shot_end" and np.allclose(got, t)

    fp = [r"B:\_K\PotassiumData\2026-08-21\0075862_2026-08-21_14-03-11_sigma_z.hdf5"]
    ad = _AD(xvardims=(2, 3), filepath=fp)
    got, src = shot_times(ad, return_source=True)
    assert src == "run_start" and got.shape == (2, 3)
    assert got[0, 0] == _dt.datetime(2026, 8, 21, 14, 3, 11).timestamp()
    assert run_start_time(ad) == got[0, 0]

    with pytest.raises(ValueError):
        shot_times(_AD())


def test_shot_times_masks_padded_shots_as_nan():
    t = np.array([T0, np.nan, 0.0, T0 + 60])
    got = shot_times(_AD(t_atoms=t))
    assert np.isnan(got[1]) and np.isnan(got[2]) and got[3] == T0 + 60


def test_climate_for_run_samples_each_sensor_per_shot(cc, server):
    t = T0 + np.array([[0, 30], [61, 125]], float)       # 2x2 scan
    ad = _AD(t_atoms=t, xvardims=(2, 2))
    out = climate_for_run(ad, client=cc, names=["Machine Table", "Vent Airflow"],
                          pad_s=120.0, attach=True)
    assert set(out) == {"_t_shot", "Machine Table", "Vent Airflow"}
    mt = out["Machine Table"]
    assert mt.shape == (2, 2)
    # shot at +30 s -> nearest sample is the minute mark at 0 s (78.0 F) or 60 s (78.1 F); tie -> earlier
    assert mt[0, 0] == pytest.approx((78.0 - 32) * 5 / 9)
    assert mt[1, 0] == pytest.approx((78.1 - 32) * 5 / 9)
    assert mt[1, 1] == pytest.approx((78.2 - 32) * 5 / 9)
    # fetched window is padded around the shots
    call = next(c for c in server.calls if c[0] == "history.get")
    assert call[1]["time_from"] == int(T0 - 120) and call[1]["time_till"] == int(T0 + 125 + 120)
    # attach=True puts it on the object in memory only
    assert ad.climate is out and set(ad.climate_series) == {"Machine Table", "Vent Airflow"}
