"""Telemetry hub, the pre-run device-state stamp and the journal's one-line
descriptions.  No network, no Qt."""
import threading
import time

import pytest

from waxx.util.device_state import composite as cmp
from waxx.util.device_state.op_journal import OpJournal, describe_entry
from waxx.util.device_state.run_stamp import MAX_JOURNAL, pre_run_report, report_warnings
from waxx.util.device_state.telemetry import Sample, TelemetryHub, TelemetryProvider


class Clock:
    t = 100.

    def __call__(self):
        return self.t


class Fake(TelemetryProvider):
    name = "fake"
    interval_s = 0.1

    def __init__(self):
        self.calls = 0
        self.fail = None
        self.value = 1.

    def poll(self):
        self.calls += 1
        if self.fail:
            raise self.fail
        return {"x": Sample(self.value, age_s=0.5), "junk": "not a sample"}


def test_samples_age_while_they_sit_in_the_hub():
    clock = Clock()
    p = Fake()
    hub = TelemetryHub([p], clock=clock)
    hub.poll_once("fake")
    assert hub.samples()["fake/x"].age_s == pytest.approx(0.5)
    assert "fake/junk" not in hub.samples()
    clock.t += 4.
    assert hub.samples()["fake/x"].age_s == pytest.approx(4.5)
    assert hub.status()["fake"] == {"ok": True, "error": "", "age_s": pytest.approx(4.)}


def test_a_failing_provider_keeps_its_last_samples_and_says_why():
    clock = Clock()
    p = Fake()
    hub = TelemetryHub([p], clock=clock)
    hub.poll_once("fake")
    p.fail = ConnectionError("server gone")
    clock.t += 10.
    hub.poll_once("fake")
    s = hub.samples()["fake/x"]
    assert s.value == 1. and s.age_s == pytest.approx(10.5)       # stale, not refreshed
    status = hub.status()["fake"]
    assert not status["ok"] and "server gone" in status["error"]
    ctx = cmp.Context(telemetry=hub.samples())
    assert ctx.measured("fake/x", 5.) is None                     # too old to use


def test_threads_poll_only_while_active():
    p = Fake()
    hub = TelemetryHub([p])
    hub.start()
    try:
        time.sleep(0.3)
        assert p.calls == 0
        hub.set_active(True)
        deadline = time.monotonic() + 3.
        while p.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert p.calls >= 2
    finally:
        hub.stop()
    assert not any(t.name.startswith("telemetry-") and t.is_alive()
                   for t in threading.enumerate())


# --- run stamp -------------------------------------------------------------------

COIL = cmp.CompositeDevice(
    key="coil", title="Coil",
    ops=(cmp.Op("off", "Off", code="expt.coil.off()"),),
    hazard=lambda ctx: "ON 20.0 A" if ctx.is_on("ttl", "igbt") else None)


class FakeMonitor:
    def __init__(self, state=None, journal=None):
        self.state, self.journal = state, journal

    def server_state(self):
        return self.state

    def journal_since_last_run(self):
        return self.journal


def test_report_records_state_trust_hazards_and_journal():
    state = {"status": "ok", "version": 12,
             "config": {"ttl": {"igbt": {"ttl_state": 1}},
                        "dac": {"coil": {"voltage": 0.4, "ch": 9}},
                        "dds": {"a": {"frequency": 1e8, "amplitude": 0.5, "v_pd": 0.,
                                      "sw_state": 1, "urukul_idx": 0}}},
             "trust": {"trusted": False, "reason": "run 7 never reported"}}
    journal = [{"kind": "update", "i": i} for i in range(MAX_JOURNAL + 3)]
    report = pre_run_report(FakeMonitor(state, journal), [COIL])
    assert report["server"] and report["version"] == 12
    assert report["hazards"] == [{"device": "coil", "title": "Coil", "text": "ON 20.0 A"}]
    assert report["device_state"] == {"dds": {"a": [1e8, 0.5, 0., 1]}, "dac": {"coil": 0.4},
                                      "ttl": {"igbt": 1}}
    assert len(report["journal_since_last_run"]) == MAX_JOURNAL
    assert report["journal_truncated"] and report["journal_since_last_run"][-1]["i"] == \
        MAX_JOURNAL + 2
    lines = report_warnings(report)
    assert "UNTRUSTED" in lines[0] and "Coil is ON 20.0 A" in lines[1]


def test_report_without_a_server_says_nothing_was_checked():
    report = pre_run_report(FakeMonitor(None), [COIL])
    assert not report["server"]
    assert "nothing was checked" in report_warnings(report)[0]


# --- journal ---------------------------------------------------------------------

def test_journal_lines_for_people(tmp_path):
    j = OpJournal(str(tmp_path), clock=lambda: 1_790_000_000.)
    e = j.record("op_submit", seq=4, op="outer_coil.off", args={"t_ramp": 0.05, "i_meas": -1.},
                 operator="ada", client="pc2", origin="watchdog")
    line = describe_entry(e)
    assert "op_submit" in line and "outer_coil.off(t_ramp=0.05, i_meas=-1) queued #4" in line
    assert "by ada@pc2" in line and "[watchdog]" in line
    assert "UNTRUSTED: run 3" in describe_entry(j.record("trust", trusted=False, reason="run 3"))
    assert "run 9 (x) end state received" in describe_entry(j.record("run_end", run_id=9,
                                                                     expt="x"))
    assert "odd" in describe_entry(j.record("something_new", odd=1))
