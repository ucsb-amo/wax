"""An aborted run reports the device state its kernel had at the abort.

scan()'s exception handler snapshots every channel in the kernel and hands it
to Expt._report_abort_state (the kernel side is covered by the offline compile
check in k-exp/tests/test_abort_state_compiles.py).  Here: the server's
``abort_state`` request, the monitor turning a snapshot into the state end()
would send, the trust decision on the host, and the Scribe abort paths no
longer sending the host frames (which hold their pre-run values while a
kernel runs).

Nothing touches the network or the real state file: the server's broadcaster
is a recorder and every state file lives in a pytest temp dir.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from waxa.base.scribe import Scribe
from waxx.base import monitor as wmon
from waxx.base.expt import Expt
from waxx.util.device_state.op_journal import describe_entry
from waxx.util.device_state.state_file_io import read_state
from waxx.util.guis import monitor_server_gui as msg


STATE = {
    "dds": {"imaging": {"frequency": 350e6, "amplitude": 1.0, "v_pd": 0.3, "sw_state": 1,
                        "urukul_idx": 4, "ch": 1, "dac_ch_key": "imaging_pid"},
            "raman_switch": {"frequency": 150e6, "amplitude": 0.46, "v_pd": 0.0,
                             "sw_state": 0, "urukul_idx": 5, "ch": 1, "dac_ch_key": ""}},
    "dac": {"imaging_pid": {"ch": 20, "voltage": 0.3}, "coil": {"ch": 9, "voltage": 0.0}},
    "ttl": {"shutter": {"ch": 22, "ttl_state": 0}},
}


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class Recorder:
    def __init__(self, *a, **k):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


@pytest.fixture
def server(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    path = tmp_path / "state.json"
    path.write_text(json.dumps(STATE))
    s = msg.MonitorUDPServer(config_file_path=str(path), journal_dir=str(tmp_path / "journal"))
    s.path = path
    yield s
    s.sock.close()


def _ask(server, obj):
    return json.loads(server.generate_reply(json.dumps(obj)))


def _config(coil_v=4.5, shutter=1):
    cfg = json.loads(json.dumps({k: STATE[k] for k in ("dds", "ttl", "dac")}))
    cfg["dac"]["coil"]["voltage"] = coil_v
    cfg["ttl"]["shutter"]["ttl_state"] = shutter
    return cfg


# --- server ----------------------------------------------------------------------

def test_a_clean_abort_writes_the_state_and_trusts_it(server):
    server._set_trust(False, "run 83100 (hf_bec) took the core")
    _ask(server, {"type": "run_pending", "run_id": 83100, "expt": "hf_bec", "token": "t"})
    v0 = server._version
    reply = _ask(server, {"type": "abort_state", "config": _config(), "run_id": 83100,
                          "expt": "hf_bec", "cause": "the liveOD Abort button",
                          "trusted": True, "caveat": ""})
    assert reply == {"status": "ok", "version": v0 + 1}
    cfg = read_state(server.path)
    assert cfg["dac"]["coil"]["voltage"] == 4.5 and cfg["ttl"]["shutter"]["ttl_state"] == 1
    assert cfg["metadata"]["updated_from"] == "abort of run"
    assert server._trust["trusted"] is True
    assert server._trust["reason"] == ("state of run 83100 (hf_bec) at its abort "
                                       "(the liveOD Abort button)")
    assert cfg["metadata"]["state_trust"] == server._trust    # survives a restart
    assert server._run_pending is None
    kinds = [p["type"] for p in server._broadcaster.sent]
    assert kinds[-2:] == ["state_reset", "trust"]
    entry = server.journal.tail(1)[0]
    assert entry["kind"] == "run_end" and entry["aborted"] == "the liveOD Abort button"
    assert entry["trusted"] is True


def test_an_abort_on_a_possible_failed_write_stays_untrusted(server):
    reply = _ask(server, {"type": "abort_state", "config": _config(), "run_id": 7,
                          "expt": "x", "cause": "RTIOUnderflow", "trusted": False,
                          "caveat": "a channel write that raised RTIOUnderflow may not "
                                    "have reached the hardware"})
    assert reply["status"] == "ok"
    assert read_state(server.path)["dac"]["coil"]["voltage"] == 4.5   # still written
    assert server._trust["trusted"] is False
    assert server._trust["reason"].startswith("run 7 (x) aborted (RTIOUnderflow); the file "
                                              "holds its last commanded state, but a channel")
    assert server.journal.tail(1)[0]["trusted"] is False


def test_trust_must_be_asked_for_explicitly(server):
    _ask(server, {"type": "abort_state", "config": _config(), "run_id": 8})
    assert server._trust["trusted"] is False
    assert server._trust["reason"] == ("run 8 (experiment) aborted (an exception); the file "
                                       "holds its last commanded state")


def test_a_malformed_abort_state_changes_nothing(server):
    before = read_state(server.path)
    reply = _ask(server, {"type": "abort_state", "config": {"dds": {}}, "trusted": True})
    assert reply["status"] == "error" and "dds/ttl/dac" in reply["msg"]
    assert read_state(server.path) == before


def test_journal_line_for_an_aborted_run():
    line = describe_entry({"kind": "run_end", "t": "2026-09-26T12:00:00", "run_id": 5,
                           "expt": "x", "aborted": "RTIOUnderflow", "trusted": False})
    assert line.endswith("run 5 (x) aborted (RTIOUnderflow): state at the abort received, "
                         "UNTRUSTED")
    line = describe_entry({"kind": "run_end", "t": "2026-09-26T12:00:00", "run_id": 5,
                           "expt": "x"})
    assert line.endswith("run 5 (x) end state received")


# --- monitor: snapshot -> state ---------------------------------------------------------

class FakeClient:
    def __init__(self):
        self.sent = []
        self.reply = {"status": "ok", "version": 2}

    def abort_state(self, config, **kw):
        self.sent.append((config, kw))
        return self.reply


class Gen:
    """The Generator's output from host frames still at their pre-run values."""
    config_data = None

    def generate_device_config(self):
        self.config_data = json.loads(json.dumps(
            {k: STATE[k] for k in ("dds", "ttl", "dac")}))
        self.config_data["metadata"] = {}


@pytest.fixture
def monitor(monkeypatch, tmp_path):
    fake = FakeClient()
    monkeypatch.setattr(wmon, "MonitorClient", lambda *a, **k: fake)
    m = wmon.Monitor(SimpleNamespace(), device_state_json_path=str(tmp_path / "s.json"))
    m._snap_dds_keys = ["imaging", "raman_switch"]
    m._snap_dac_keys = ["imaging_pid", "coil"]
    m._snap_ttl_keys = ["shutter"]
    m.generator = Gen()
    m.fake = fake
    return m


def _snapshot():
    # imaging: new freq/amp, switched off, v_pd 0.9 in the DDS but its linked
    # DAC was last set to 0.7 directly; coil at 4.5 V; shutter open.
    return (np.array([351e6, 150e6]), np.array([0.5, 0.46]), np.array([0.9, 0.0]),
            np.array([0, 1], dtype=np.int32), np.array([0.7, 4.5]),
            np.array([1], dtype=np.int32))


def test_the_snapshot_replaces_every_value_and_keeps_the_static_fields(monitor):
    cfg = monitor._config_from_snapshot(*_snapshot())
    assert set(cfg) == {"dds", "ttl", "dac"}
    img = cfg["dds"]["imaging"]
    assert (img["frequency"], img["amplitude"], img["sw_state"]) == (351e6, 0.5, 0)
    assert img["v_pd"] == 0.7                    # from the linked DAC, as the Generator
    assert img["urukul_idx"] == 4 and img["dac_ch_key"] == "imaging_pid"
    assert cfg["dds"]["raman_switch"]["sw_state"] == 1
    assert cfg["dds"]["raman_switch"]["v_pd"] == 0.0     # unlinked: its own v_pd
    assert cfg["dac"]["coil"] == {"ch": 9, "voltage": 4.5}
    assert cfg["ttl"]["shutter"]["ttl_state"] == 1
    assert type(cfg["ttl"]["shutter"]["ttl_state"]) is int


def test_a_snapshot_that_does_not_cover_the_frames_is_refused(monitor, capsys):
    short = list(_snapshot())
    short[4] = np.zeros(1)                                   # the no-op default
    assert monitor.report_abort_state(*short, run_id=1) is False
    assert monitor.fake.sent == []
    assert "could not build" in capsys.readouterr().out


def test_report_sends_abort_state_and_clears_the_exit_withdraw(monitor):
    monitor._announced = ("tok", 9)
    assert monitor.report_abort_state(*_snapshot(), run_id=9, expt="x", cause="c",
                                      trusted=False, caveat="why") is True
    config, kw = monitor.fake.sent[-1]
    assert kw == {"run_id": 9, "expt": "x", "cause": "c", "trusted": False, "caveat": "why"}
    assert config["dac"]["coil"]["voltage"] == 4.5
    assert json.dumps(config)                                 # plain JSON, no numpy
    assert monitor._announced is None


def test_an_old_server_is_named_and_the_state_stays_untrusted(monitor, capsys):
    monitor._announced = ("tok", 9)
    monitor.fake.reply = {"status": "error", "msg": "unknown type abort_state"}
    assert monitor.report_abort_state(*_snapshot(), run_id=9) is False
    out = capsys.readouterr().out
    assert "runs older code; restart it" in out and "stays untrusted" in out
    assert monitor._announced == ("tok", 9)                   # still withdrawn at exit


# --- host: trust decision ----------------------------------------------------------------

class Run:
    """Just what Expt._report_abort_state and the Scribe abort paths use."""
    _report_abort_state = Expt._report_abort_state
    _restart_monitor_once = Scribe._restart_monitor_once
    _check_for_abort_signal = Scribe._check_for_abort_signal
    _abort_for_reset = Scribe._abort_for_reset
    _send_abort_to_server = Scribe._send_abort_to_server

    def __init__(self, accepted=True):
        self.reports = []
        self.ends = 0
        self.stale_sends = 0
        self.accepted = accepted
        run = self

        class Mon:
            def report_abort_state(self, *snap, **kw):
                if isinstance(run.accepted, Exception):
                    raise run.accepted
                run.reports.append(kw)
                return run.accepted

            def signal_end(self):
                run.ends += 1

            def update_device_states(self, *a, **k):   # the pre-run host frames
                run.stale_sends += 1

        self.monitor = Mon()
        self.run_info = SimpleNamespace(run_id=83100, save_on_underflow=0, save_data=True)
        self.live_od_client = SimpleNamespace(last_reset_requested=True,
                                              abort_run=lambda: None)
        self._shot_complete_count = 3
        self._abort_cause = ""
        self._shot_abort = ""

    def _expt_file_stem(self):
        return "hf_bec"


SNAP = _snapshot()


def test_the_liveod_abort_button_reports_a_trusted_state_and_restarts_once(capsys):
    run = Run()
    with pytest.raises(RuntimeError):
        run._check_for_abort_signal()          # the scan loop's reset check
    run._report_abort_state("", *SNAP)          # scan()'s handler: a RuntimeError
    assert run.reports == [{"run_id": 83100, "expt": "hf_bec",
                            "cause": "the liveOD Abort button", "trusted": True,
                            "caveat": ""}]
    assert run.ends == 1 and run.stale_sends == 0
    run._report_abort_state("", *SNAP)          # a second pass restarts nothing
    assert run.ends == 1
    assert "went to the monitor server (trusted)" in capsys.readouterr().out


@pytest.mark.parametrize("what", ["RTIOUnderflow", "RTIODestinationUnreachable", "ValueError"])
def test_an_exception_a_write_can_raise_is_reported_untrusted(what, capsys):
    run = Run()
    run._report_abort_state(what, *SNAP)
    kw = run.reports[0]
    assert kw["trusted"] is False and kw["cause"] == what
    assert kw["caveat"].startswith(f"a channel write that raised {what}")
    assert "marked UNTRUSTED" in capsys.readouterr().out


@pytest.mark.parametrize("what", ["RTIOOverflow", ""])
def test_other_exceptions_are_trusted(what):
    run = Run()
    run._report_abort_state(what, *SNAP)
    assert run.reports[0]["trusted"] is True
    assert run.reports[0]["cause"] == (what or "an exception (see the traceback)")


def test_a_shot_abandoned_on_an_underflow_taints_the_report():
    run = Run()
    with pytest.raises(RuntimeError, match="RTIOUnderflow: run 83100 aborted"):
        run._send_abort_to_server("RTIOUnderflow")
    assert run.stale_sends == 0 and run.ends == 0     # both left to the handler
    run._report_abort_state("", *SNAP)
    kw = run.reports[0]
    assert kw["trusted"] is False and kw["cause"] == "RTIOUnderflow in a shot"
    assert run.ends == 1


def test_a_shot_abandoned_on_a_trigger_timeout_does_not():
    run = Run()
    with pytest.raises(RuntimeError, match="TriggerTimeout"):
        run._send_abort_to_server("TriggerTimeout")
    run._report_abort_state("", *SNAP)
    assert run.reports[0]["trusted"] is True
    assert run.reports[0]["cause"] == "TriggerTimeout in a shot"


def test_save_on_underflow_still_returns_to_finish_normally():
    run = Run()
    run.run_info.save_on_underflow = 1
    assert run._send_abort_to_server("RTIOUnderflow") is None
    assert run.stale_sends == 0 and run.ends == 0 and run.reports == []


def test_a_reset_during_the_camera_wait_sends_no_state():
    """Before the scan the host frames are pre-run values and the kernel has
    not been snapshotted: nothing is reported, the monitor is restarted."""
    run = Run()
    with pytest.raises(RuntimeError):
        run._abort_for_reset('while waiting for the camera')
    assert run.stale_sends == 0 and run.reports == [] and run.ends == 1
    assert run._abort_cause == "the liveOD Abort button while waiting for the camera"


def test_a_failed_report_never_raises_and_still_restarts(capsys):
    run = Run(accepted=ConnectionError("down"))
    run._report_abort_state("", *SNAP)
    assert "stays untrusted" in capsys.readouterr().out
    assert run.ends == 1


def test_the_monitor_experiment_and_a_run_without_one_report_nothing():
    run = Run()
    run._is_monitor = True
    run._report_abort_state("", *SNAP)
    assert run.reports == [] and run.ends == 0
    bare = Run()
    del bare.monitor
    bare._report_abort_state("", *SNAP)                  # no AttributeError


# --- a run the run queue launched (review S1) ------------------------------------------------

@pytest.mark.parametrize("launcher, ends", [("kq", 0), ("run_loop", 1), (None, 1)])
def test_an_aborted_queue_job_leaves_the_monitor_to_the_server(monkeypatch, capsys,
                                                               launcher, ends):
    if launcher is None:
        monkeypatch.delenv("WAXX_LAUNCHER", raising=False)
    else:
        monkeypatch.setenv("WAXX_LAUNCHER", launcher)
    monkeypatch.setenv("WAXX_QUEUE_JOB", "12")
    run = Run()
    with pytest.raises(RuntimeError):
        run._abort_for_reset("during the camera wait")
    run._report_abort_state("", *SNAP)
    assert run.ends == ends
    out = capsys.readouterr().out
    assert ("not restarted here" in out) == (launcher == "kq")
    if launcher == "kq":
        assert out.count("not restarted here") == 1 and "job 12" in out
