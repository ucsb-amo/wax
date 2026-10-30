"""FP lock server, client and protocol, against a fake Quarto (fakes/fake_fp_lock.py).

Nothing touches real hardware: the serial port is the fake (the server's
``serial_factory``), the TCP server binds 127.0.0.1 port 0, the client is
built at that address (no discovery), and the beacon is a no-op (conftest).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
import fake_fp_lock  # noqa: E402

from waxx.util.guis.fp_lock import fp_lock_protocol as proto  # noqa: E402
from waxx.util.guis.fp_lock import fp_lock_server as fserver  # noqa: E402
from waxx.util.guis.fp_lock.fp_lock_client import FPLockClient  # noqa: E402


def _until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def test_parse_row_all_firmware_widths():
    line16 = "6.410, 0.010, -0.0903, 2, 12.500, 9.10, 8.20, 3, 0, 0, 0, 1, 2, 0, -0.0900, 120.004"
    r = proto.parse_row(line16)
    assert r["sep"] == pytest.approx(6.41) and r["state"] == 2 and r["state_name"] == "LOCKED"
    assert r["t2"] == pytest.approx(18.91) and r["period"] == pytest.approx(120.004)
    assert r["bad_one"] == 2 and isinstance(r["bad_one"], int)
    r14 = proto.parse_row(", ".join(line16.split(", ")[:14]))
    assert "integ" not in r14 and r14["bad_win"] == 0
    assert proto.parse_row(", ".join(line16.split(", ")[:11]))["saturated"] == 0
    for not_a_row in ("sep is 6.4", "Unknown command: d1", "   6.410    +0.010  LOCKED",
                      "1, 2, 3", "## LOCKED -> HOLD", ""):
        assert proto.parse_row(not_a_row) is None, not_a_row


def test_parse_replies():
    assert proto.parse_reply("setp is 6.400000") == ("setp", "6.400000")
    assert proto.parse_reply("hold is 1") == ("hold", "1")
    assert proto.parse_reply("6.4, 0.1") is None
    assert proto.parse_unknown("Unknown command: hold") == "hold"
    assert proto.parse_unknown("hold is 1") is None


def test_format_value_validates():
    S = proto.SETTABLE
    assert proto.format_value(S["lock"], 1) == "1"
    assert proto.format_value(S["nmin"], "4") == "4"
    assert float(proto.format_value(S["p1"], "-0.02")) == -0.02
    for setting, bad in ((S["lock"], 2), (S["nmin"], 3.5), (S["setp"], "nan"), (S["setp"], "x")):
        with pytest.raises(ValueError):
            proto.format_value(setting, bad)


def _row(**kw):
    base = dict(sep=6.4, err=0.0, feedback=0.0, state=2, tfirst=40.0, h1=9.0, h2=8.0,
                saturated=0, period=120.0)
    base.update(kw)
    base["t2"] = base["tfirst"] + base["sep"]
    return base


def _codes(row, **kw):
    args = dict(edge=proto.EdgeConfig(120.0, 10.0), com_status="connected", data_age_s=1.0,
                stale_after_s=5.0)
    args.update(kw)
    return [w["code"] for w in proto.compute_warnings(row, **args)]


def test_warnings_peak_edges():
    assert _codes(_row()) == []
    assert _codes(_row(tfirst=9.9)) == ["first_near_start"]
    assert _codes(_row(tfirst=10.1)) == []
    assert _codes(_row(tfirst=104.0)) == ["second_near_end"]          # t2 = 110.4
    assert _codes(_row(tfirst=103.0)) == []                           # t2 = 109.4
    assert _codes(_row(tfirst=5.0, sep=0.0)) == []                    # no pair yet
    assert _codes(_row(tfirst=5.0, h1=0.0)) == []
    assert "period" in _codes(_row(period=90.0))


def test_warnings_state_link_and_order():
    assert _codes(_row(state=1)) == ["hold"]
    assert _codes(_row(state=0)) == ["off"]
    assert _codes(_row(saturated=1, state=1, tfirst=5.0)) == ["rail", "hold", "first_near_start"]
    assert _codes(None, com_status="error") == ["com"]
    assert _codes(None, data_age_s=None) == ["no_data"]
    assert _codes(_row(), data_age_s=9.0)[0] == "stale"


# ---------------------------------------------------------------------------
# Server with the fake board
# ---------------------------------------------------------------------------

@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setattr(fserver.FPLockLink, "BACKOFF_S", (0.05,))
    monkeypatch.setattr(fserver, "LOG_FLUSH_S", 0.0)
    made = []

    def _make(dev=None, **kw):
        dev = dev or fake_fp_lock.FakeQuarto()
        kw.setdefault("log_dir", str(tmp_path / "logs"))
        srv = fserver.FPLockServer("fp_lock_test", "COMFAKE35", host="127.0.0.1", port=0,
                                   serial_factory=dev.open, **kw)
        srv.start()
        made.append(srv)
        return srv, dev, FPLockClient.at("127.0.0.1", srv.port, "fp_lock_test")

    yield _make
    for srv in made:
        srv.stop()


def _connected(srv, dev):
    assert _until(lambda: srv.link.connected and dev.values.get("fmt") == 0
                  and srv.snapshot()["label"] is not None), srv.link.snapshot()


def test_connects_sets_csv_mode_reads_settings_and_label(make):
    srv, dev, client = make()
    _connected(srv, dev)
    snap = client.get_snapshot()
    assert snap["com"]["status"] == "connected" and snap["label"] == "raman fp lock"
    assert snap["settings"]["setp"] == pytest.approx(10.0)
    assert snap["settings"]["i1"] == pytest.approx(-0.01) and snap["settings"]["hold"] == 1
    assert snap["missing_settings"] == []
    assert [w["code"] for w in snap["warnings"]] == ["no_data"]


def test_rows_history_warnings_and_log(make, tmp_path):
    srv, dev, client = make()
    _connected(srv, dev)
    dev.emit_row(sep=6.4, state=2, tfirst=5.0, h1=9.0, h2=8.0)
    dev.emit_row(sep=6.4, state=2, tfirst=40.0, h1=9.0, h2=8.0)
    dev.emit_row(sep=6.4, state=1, tfirst=105.0, h1=9.0, h2=8.0)
    assert _until(lambda: client.get_snapshot()["seq"] == 3)
    snap = client.get_snapshot()
    assert snap["latest"]["t2"] == pytest.approx(111.4)
    assert [w["code"] for w in snap["warnings"]] == ["hold", "second_near_end"]
    hist = client.get_history(since=0)
    assert [r["tfirst"] for r in hist["rows"]] == [5.0, 40.0, 105.0]
    assert [r["seq"] for r in client.get_history(since=2)["rows"]] == [3]
    srv._flush_log(force=True)
    logs = list((tmp_path / "logs").glob("*.csv"))
    assert len(logs) == 1
    lines = logs[0].read_text().splitlines()
    assert lines[0].startswith("time_iso,unix_s,sep,err,feedback,state,tfirst")
    assert len(lines) == 4 and lines[-1].split(",")[2] == "6.4"


def test_set_lock_reset_setpoint_clear_and_raw(make):
    srv, dev, client = make()
    _connected(srv, dev)
    assert client.set_value("p1", -0.02)["value"] == pytest.approx(-0.02)
    assert dev.values["p1"] == pytest.approx(-0.02)
    assert client.set_value("nmin", 4)["value"] == 4 and dev.values["nmin"] == 4
    for bad in (("d1", 1), ("sep", 3), ("lock", 7), ("setp", "nan")):
        reply = client.set_value(*bad)
        assert reply["status"] == "error", bad
    assert dev.values["lock"] == 1 and dev.values["setp"] == 10.0
    assert client.lock(False)["value"] == 0 and dev.values["lock"] == 0
    assert client.lock(True)["value"] == 1 and dev.values["lock"] == 1
    dev.values["sep"] = 6.437
    reply = client.reset_setpoint()
    assert reply["status"] == "ok" and reply["old"] == pytest.approx(10.0)
    assert reply["new"] == pytest.approx(6.437) and dev.values["setp"] == pytest.approx(6.437)
    assert client.get_snapshot()["settings"]["setp"] == pytest.approx(6.437)
    dev.values["integ"] = 0.4
    assert client.clear()["status"] == "ok" and dev.values["integ"] == 0.0
    out = client.raw("help")
    assert out["status"] == "ok" and out["lines"][0].startswith("FP peak-separation lock")
    assert client.raw("a\nb")["status"] == "error"
    assert any(t["line"].startswith("FP peak") for t in client.get_text(0)["lines"])


def test_reset_setpoint_refuses_without_a_measurement(make):
    srv, dev, client = make()
    _connected(srv, dev)
    dev.values["sep"] = 0.0
    reply = client.reset_setpoint()
    assert reply["status"] == "error" and dev.values["setp"] == 10.0


def test_firmware_without_hold_reports_it_missing(make):
    srv, dev, client = make(fake_fp_lock.FakeQuarto(without=("hold",)))
    _connected(srv, dev)
    assert _until(lambda: client.get_snapshot()["missing_settings"] == ["hold"])
    assert client.set_value("hold", 0)["status"] == "error"


def test_port_missing_server_still_answers(make):
    dev = fake_fp_lock.FakeQuarto()
    dev.fail_open = True
    srv, dev, client = make(dev)
    assert _until(lambda: client.get_snapshot()["com"]["status"] == "error")
    snap = client.get_snapshot()
    assert snap["warnings"][0]["code"] == "com" and snap["latest"] is None
    assert client.lock(True)["status"] == "error"
    dev.fail_open = False
    _connected(srv, dev)


def test_read_error_reconnects_and_reasserts_csv(make):
    srv, dev, client = make()
    _connected(srv, dev)
    first = dev.port
    dev.values["fmt"] = 1            # as after a board reset
    first.fail_next_read = True
    assert _until(lambda: dev.opened == 2 and dev.port is not first and dev.values["fmt"] == 0)


def test_disconnect_holds_then_connect_reopens(make):
    srv, dev, client = make()
    _connected(srv, dev)
    assert client.com_disconnect() == (True, "disconnected and held")
    assert not dev.port.is_open and client.get_snapshot()["com"]["status"] == "held"
    time.sleep(0.2)
    assert dev.opened == 1
    ok, _ = client.com_connect()
    assert ok and _until(lambda: dev.opened == 2 and srv.link.connected)


def test_stop_puts_the_table_back(make):
    srv, dev, client = make()
    _connected(srv, dev)
    srv.stop()
    assert dev.values["fmt"] == 1 and not dev.port.is_open


def test_unknown_cmd_and_bare_word(make):
    srv, dev, client = make()
    assert client.request("bogus")["status"] == "error"
    import json  # noqa: PLC0415
    import socket  # noqa: PLC0415
    with socket.create_connection(("127.0.0.1", srv.port), timeout=2) as s:
        s.sendall(b"GET_SNAPSHOT\n")
        reply = json.loads(s.makefile().readline())
    assert reply["status"] == "ok" and "warnings" in reply
