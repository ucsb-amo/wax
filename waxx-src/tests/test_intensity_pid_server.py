"""Intensity PID protocol, server, client and CLI against a fake Quarto
(fakes/fake_intensity_pid.py).

Nothing touches real hardware: the serial port is the fake (the server's
``serial_factory``), the TCP server binds 127.0.0.1 port 0, the client is built
at that address (no discovery), and the beacon is a no-op (conftest).
"""
from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
import fake_intensity_pid  # noqa: E402

from waxx.util.guis.intensity_pid import intensity_pid_cli as cli  # noqa: E402
from waxx.util.guis.intensity_pid import intensity_pid_protocol as proto  # noqa: E402
from waxx.util.guis.intensity_pid import intensity_pid_server as iserver  # noqa: E402
from waxx.util.guis.intensity_pid.intensity_pid_client import (  # noqa: E402
    IntensityPIDClient, IntensityPIDError, require_ok,
)


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

def test_parse_row():
    r = proto.parse_row("0.5012,0.5000,1.0003,1.0000,0,3.2140,4.0000")
    assert r == {"meas1": 0.5012, "set1": 0.5, "meas2": 1.0003, "set2": 1.0, "hold": 0,
                 "out1": 3.214, "out2": 4.0}
    assert isinstance(r["hold"], int)
    assert proto.parse_row("0.5,0.5,1.0,1.0,1")["hold"] == 1           # without out1/out2
    for not_a_row in ("hold is 1", "integrators cleared", "p1 is -0.055000", "1,2,3",
                      "0.5,0.5,1,1,2,0,0", "0.5,0.5,1,1,0,0,0,9", "Unknown command: x", ""):
        assert proto.parse_row(not_a_row) is None, not_a_row


def test_replies_and_switches():
    assert proto.parse_reply("p1 is -0.055000") == ("p1", "-0.055000")
    assert proto.parse_reply("hold is true") == ("hold", "true")
    assert proto.parse_switch_reply("true") is True and proto.parse_switch_reply("false") is False
    assert proto.parse_switch_reply("1") is True and proto.parse_switch_reply("0") is False
    assert proto.parse_switch_reply("x") is None and proto.parse_switch_reply(None) is None
    assert proto.parse_unknown("Unknown command: hold") == "hold"
    assert float(proto.format_value(proto.SETTABLE["i1"], "-0.006")) == -0.006
    for bad in ("nan", "inf", "x"):
        with pytest.raises(ValueError):
            proto.format_value(proto.SETTABLE["p1"], bad)
    S = proto.SETTABLE
    assert proto.format_value(S["override1"], True) == "1"
    assert proto.format_value(S["override1"], "0") == "0"
    assert float(proto.format_value(S["ovolt2"], 2.5)) == 2.5
    for setting, bad in ((S["override1"], 2), (S["ovolt1"], -0.1), (S["ovolt1"], 10.0)):
        with pytest.raises(ValueError):
            proto.format_value(setting, bad)
    assert proto.parse_setting(S["override2"], "true") == 1
    assert proto.parse_setting(S["ovolt1"], "2.500000") == 2.5
    assert proto.GAIN_NAMES == ("p1", "i1", "p2", "i2")


def test_block_mean_and_binner():
    rows = [dict(t=float(i), meas1=0.5 + 0.001 * i, set1=0.5, meas2=1.0, set2=1.0,
                 hold=1 if i == 3 else 0, out1=2.0, out2=3.0) for i in range(4)]
    m = proto.block_mean(rows)
    assert m["n"] == 4 and m["t"] == 3.0
    assert m["meas1"] == pytest.approx(0.5015)
    assert m["meas1_sd"] == pytest.approx(0.0012909944, rel=1e-6)
    assert m["meas2_sd"] == 0.0 and m["hold_frac"] == 0.25
    assert math.isnan(proto.block_mean(rows[:1])["meas1_sd"])
    assert proto.block_mean([]) is None

    b = proto.Binner(1.0)
    assert b.add(dict(rows[0], t=0.0)) is None
    assert b.add(dict(rows[1], t=0.9)) is None
    closed = b.add(dict(rows[2], t=1.0))            # first row past the block closes it
    assert closed["n"] == 2 and closed["t"] == 0.9


def _codes(row, **kw):
    args = dict(limits=proto.OutputLimits(0.0, 9.9), com_status="connected", data_age_s=0.1,
                stale_after_s=2.0)
    args.update(kw)
    return [w["code"] for w in proto.compute_warnings(row, **args)]


def test_warnings():
    ok = dict(meas1=.5, set1=.5, meas2=1., set2=1., hold=0, out1=3., out2=4.)
    assert _codes(ok) == []
    assert _codes(dict(ok, hold=1)) == ["hold"]
    assert _codes(dict(ok, out1=9.9)) == ["rail1"]
    assert _codes(dict(ok, out2=0.0)) == ["rail2"]
    assert _codes(dict(ok, hold=1, out1=9.9)) == ["hold"]          # held: no rail warning
    assert _codes(ok, com_status="disconnected") == ["com"]
    assert _codes(None, data_age_s=None) == ["no_data"]
    assert _codes(ok, data_age_s=5.0) == ["stale"]
    over = {"override1": 1, "ovolt1": 2.5, "override2": 0}
    assert _codes(dict(ok, out1=9.9), settings=over) == ["override1"]   # no rail warning
    extra = [{"level": "warn", "code": "gains_reapplied", "text": "x"}]
    assert _codes(dict(ok, out1=9.9), com_status="disconnected", extra=extra) == \
        ["com", "gains_reapplied", "rail1"]


# ---------------------------------------------------------------------------
# Server + client
# ---------------------------------------------------------------------------

@pytest.fixture
def rig(monkeypatch, tmp_path):
    monkeypatch.setattr(iserver.IntensityPIDLink, "BACKOFF_S", (0.05,))
    monkeypatch.setattr(iserver, "PLOT_BLOCK_S", 0.0)    # every row closes a block
    monkeypatch.setattr(iserver, "LOG_BLOCK_S", 0.0)
    made = []

    def make(dev=None, **kw):
        dev = dev or fake_intensity_pid.FakeIntensityQuarto()
        args = dict(host="127.0.0.1", port=0, serial_factory=dev.open,
                    log_dir=str(tmp_path / "logs"), gains_path=str(tmp_path / "gains.json"),
                    loop_names=("80+ AOM", "150+ AOM"))
        args.update(kw)
        srv = iserver.IntensityPIDServer("ipid_test", "COMFAKE16", **args)
        srv.start()
        made.append(srv)
        assert _until(lambda: srv.link.connected and srv.snapshot()["label"] is not None)
        client = IntensityPIDClient.at("127.0.0.1", srv.port, "ipid_test")
        return dev, srv, client

    yield make, tmp_path
    for srv in made:
        srv.stop()


def test_connect_streams_reads_gains_and_saves_them(rig):
    make, tmp = rig
    dev, srv, client = make()
    snap = client.get_snapshot()
    assert snap["label"] == "raman intensity"
    assert {k: snap["settings"][k] for k in proto.GAIN_NAMES} == \
        pytest.approx(fake_intensity_pid.DEFAULT_GAINS)
    assert snap["settings"]["override1"] == 0 and snap["settings"]["ovolt2"] == 0.0
    saved = json.loads((tmp / "gains.json").read_text())
    assert saved["gains"] == pytest.approx(fake_intensity_pid.DEFAULT_GAINS)
    assert [w["code"] for w in snap["warnings"]] == ["no_data"]


def test_rows_history_and_snapshot(rig):
    make, _ = rig
    dev, srv, client = make()
    for i in range(5):
        dev.emit_row(meas1=0.5 + 0.001 * i, set1=0.5, out1=3.0)
    assert _until(lambda: (client.get_snapshot().get("latest") or {}).get("meas1") == 0.504)
    snap = client.get_snapshot()
    assert snap["held"] is False and snap["latest"]["out1"] == 3.0
    assert snap["warnings"] == []
    hist = client.get_history(since=0)
    assert hist["status"] == "ok" and len(hist["rows"]) == snap["seq"] >= 3
    st = client.status()
    assert st["meas1"] == 0.504 and st["held"] is False and st["p1"] == pytest.approx(-0.055)


def test_hold_release_toggle_bumpless(rig):
    make, _ = rig
    dev, srv, client = make()
    dev.emit_row(out1=4.2, out2=5.1)
    r = require_ok(client.hold(True))
    assert r["on"] is True and dev.hold == 1 and dev.received[-1] == "hold 1"
    assert client.get_snapshot()["held"] is True
    dev.emit_row(out1=0.1, out2=0.1)                  # held: the fake keeps the outputs
    assert _until(lambda: (client.get_snapshot().get("latest") or {}).get("out1") == 4.2)
    assert "hold" in [w["code"] for w in client.get_snapshot()["warnings"]]
    r = require_ok(client.hold(False))
    assert r["on"] is False and dev.integ == [4.2, 5.1]          # release: integrators = outputs
    r = require_ok(client.hold())                                # toggle: read, then set
    assert r["on"] is True and dev.received[-2:] == ["hold", "hold 1"]
    assert require_ok(client.release())["on"] is False


def test_override(rig):
    make, tmp = rig
    dev, srv, client = make()
    saved_before = (tmp / "gains.json").read_text()
    r = require_ok(client.override(1, True, 2.5))
    assert r["on"] is True and r["volt"] == 2.5
    assert dev.override["override1"] == 1 and dev.ovolt["ovolt1"] == 2.5
    assert dev.override["override2"] == 0
    dev.emit_row(out1=7.0)
    assert _until(lambda: (client.get_snapshot().get("latest") or {}).get("out1") == 2.5)
    snap = client.get_snapshot()
    assert snap["settings"]["override1"] == 1
    w = [w for w in snap["warnings"] if w["code"] == "override1"]
    assert len(w) == 1 and "2.500 V" in w[0]["text"]
    # out of range: refused, nothing sent
    r = client.override(2, True, 12.0)
    assert r["status"] == "error" and dev.override["override2"] == 0
    r = require_ok(client.override(1, False))
    assert r["on"] is False and dev.override["override1"] == 0
    assert (tmp / "gains.json").read_text() == saved_before        # overrides are not saved
    assert client.override(3, True)["status"] == "error"


def test_override_not_reapplied_after_board_reset(rig):
    make, _ = rig
    dev, srv, client = make()
    require_ok(client.override(2, True, 1.0))
    dev.power_cycle()
    dev.port.fail_next_read = True
    assert _until(lambda: dev.opened >= 2 and client.get_snapshot()["settings"]["override2"] == 0,
                  timeout=5.0)
    assert dev.override["override2"] == 0


def test_clear(rig):
    make, _ = rig
    dev, srv, client = make()
    r = require_ok(client.clear())
    assert _until(lambda: dev.integ == [0.0, 0.0]) and dev.received[-1] == "c"
    assert "integrators cleared" in r["message"]


def test_set_gain_is_read_back_and_remembered(rig):
    make, tmp = rig
    dev, srv, client = make()
    r = require_ok(client.set_value("i2", "-0.004"))
    assert r["value"] == pytest.approx(-0.004) and dev.gains["i2"] == -0.004
    assert json.loads((tmp / "gains.json").read_text())["gains"]["i2"] == pytest.approx(-0.004)
    r = require_ok(client.set_value("p1", 2.5e-7))                # below the 6-decimal readback
    assert r["value"] == 0.0 and "note" in r and dev.gains["p1"] == 2.5e-7
    # the store keeps what was sent, not the rounded readback
    assert json.loads((tmp / "gains.json").read_text())["gains"]["p1"] == 2.5e-7
    for name, value in (("set1", 0.3), ("hold", 1), ("p1", "nan")):
        assert client.set_value(name, value)["status"] == "error"


def test_gains_reapplied_after_board_reset_with_warning(rig):
    make, tmp = rig
    (tmp / "gains.json").write_text(json.dumps(
        {"gains": {"p1": -0.05, "i1": -0.005, "p2": -0.055, "i2": -0.006}}))
    dev, srv, client = make()                      # the board has the sketch defaults
    assert _until(lambda: dev.gains["p1"] == -0.05 and dev.gains["i1"] == -0.005)
    assert dev.gains["p2"] == -0.055
    warn = [w for w in client.get_snapshot()["warnings"] if w["code"] == "gains_reapplied"]
    assert len(warn) == 1 and "p1 -0.055 -> -0.05" in warn[0]["text"]
    assert "i1 -0.006 -> -0.005" in warn[0]["text"] and "p2" not in warn[0]["text"]

    # the board is power-cycled: the port drops, the link reconnects, gains go back again
    dev.power_cycle()
    dev.port.fail_next_read = True
    assert _until(lambda: dev.opened >= 2 and dev.gains["p1"] == -0.05, timeout=5.0)

    # a person sets a gain: the re-apply warning is gone, the new gain is the reference
    require_ok(client.set_value("p2", -0.06))
    assert not [w for w in client.get_snapshot()["warnings"] if w["code"] == "gains_reapplied"]
    assert json.loads((tmp / "gains.json").read_text())["gains"]["p2"] == pytest.approx(-0.06)


def test_unreadable_gain_file_warns_and_applies_nothing(rig):
    make, tmp = rig
    (tmp / "gains.json").write_text("{not json")
    dev, srv, client = make()
    assert _until(lambda: any(w["code"] == "gains_file" for w in client.get_snapshot()["warnings"]))
    assert dev.gains == fake_intensity_pid.DEFAULT_GAINS
    assert not any(line.split()[0] in ("p1", "i1", "p2", "i2") and len(line.split()) > 1
                   for line in dev.received)


def test_old_firmware_without_hold(rig):
    make, _ = rig
    dev, srv, client = make(fake_intensity_pid.FakeIntensityQuarto(without=("hold",)))
    r = client.hold(True)
    assert r["status"] == "error" and "does not know 'hold'" in r["message"]
    with pytest.raises(IntensityPIDError):
        require_ok(r)


def test_log_has_header_and_means(rig):
    make, tmp = rig
    dev, srv, client = make()
    for i in range(4):
        dev.emit_row(meas1=0.5, set1=0.5)
    assert _until(lambda: srv.snapshot()["seq"] >= 3)
    srv._flush_log(force=True)
    files = list((tmp / "logs").glob("*.csv"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert lines[0].startswith("# 1 s means")
    assert lines[1] == "time_iso,unix_s," + ",".join(proto.LOG_FIELDS)
    assert lines[1].endswith("override1,override2")
    assert len(lines) >= 5 and lines[2].split(",")[3] == "0.5"     # n, then meas1
    assert lines[2].split(",")[-2:] == ["0", "0"]


def test_com_disconnect_holds_the_port(rig):
    make, _ = rig
    dev, srv, client = make()
    ok, _msg = client.com_disconnect()
    assert ok and not srv.link.connected
    assert client.hold(True)["status"] == "error"
    ok, _msg = client.com_connect()
    assert ok and _until(lambda: srv.link.connected)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(client, *argv):
    out = io.StringIO()
    code = cli.main(list(argv), server_id="ipid_test", client_factory=lambda: client, out=out)
    return code, out.getvalue()


def test_cli_commands(rig):
    make, _ = rig
    dev, srv, client = make()
    dev.emit_row(meas1=0.5012, set1=0.5, out1=3.2)
    assert _until(lambda: client.get_snapshot().get("latest") is not None)
    code, text = _cli(client, "status")
    assert code == 0 and "80+ AOM: meas 0.5012 set 0.5000 out 3.2000 V" in text and "hold 0" in text
    code, text = _cli(client, "status", "--json")
    assert code == 0 and json.loads(text)["server_id"] == "ipid_test"
    assert _cli(client, "hold", "on") == (0, "hold on: both outputs fixed\n") and dev.hold == 1
    assert _cli(client, "release") == (0, "hold off: loops running\n") and dev.hold == 0
    assert _cli(client, "hold")[0] == 0 and dev.hold == 1
    _cli(client, "hold", "off")
    code, text = _cli(client, "clear")
    assert code == 0 and "integrators cleared" in text
    code, text = _cli(client, "set", "p1", "-0.05")
    assert code == 0 and text.startswith("p1 = -0.05")
    code, text = _cli(client, "gains")
    assert code == 0 and text.split()[:2] == ["p1", "-0.05"]
    assert _cli(client, "override", "1", "on", "2.5") == (0, "override1 on: output 2.500 V\n")
    assert dev.override["override1"] == 1 and dev.ovolt["ovolt1"] == 2.5
    code, text = _cli(client, "status")
    assert "override1 2.500 V" in text
    assert _cli(client, "override", "1", "off") == (0, "override1 off: the PID drives it\n")
    assert dev.override["override1"] == 0
    assert _cli(client, "override", "1", "off", "2")[0] == 2
    assert _cli(client, "override", "2", "on", "11")[0] == 1
    assert _cli(client, "hold", "maybe")[0] == 2
    assert _cli(client, "set", "set1", "0.3")[0] == 2


def test_cli_exit_codes_without_server(rig):
    make, _ = rig
    dev, srv, client = make(fake_intensity_pid.FakeIntensityQuarto(without=("hold",)))
    code, text = _cli(client, "hold", "on")
    assert code == 1 and text.startswith("refused:")

    def no_server():
        raise RuntimeError("no beacon (stub)")

    out = io.StringIO()
    assert cli.main(["status"], server_id="ipid_test", client_factory=no_server, out=out) == 3
    dead = IntensityPIDClient.at("127.0.0.1", 1, "ipid_test", timeout=0.3)
    code, text = _cli(dead, "status")
    assert code == 3 and "did not answer" in text
