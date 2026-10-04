"""Offline tests for waxx.control.artiq.ttl_pulser.TTLPulser.

No core device, no network, no monitor server: the core and the monitor
client are fakes handed in through the test hooks.  What is checked is the
sequence the pulser promises -- announce, connect, compile once, pulse N
times, probe, close, carry the trust flag through only when honest, restart
the monitor only when the machine is free -- and what it does when the core
is taken from under it.
"""
import pytest

from waxx.control.artiq.ttl_pulser import (CoreTaken, PulserNotHeld, TTLPulser,
                                           _PulseKernel)


class FakeComm:
    def __init__(self, core):
        self.core = core
        self.opened = False
        self.closed = 0
        self.info_calls = 0
        self.socket = None

    def open(self):
        self.opened = True

    def check_system_info(self):
        self.info_calls += 1
        if self.core.lost:
            raise ConnectionResetError("Core device connection closed unexpectedly")

    def close(self):
        self.closed += 1
        self.opened = False


class FakeCore:
    """Stands in for artiq.coredevice.core.Core: precompile() returns a
    callable that records the runs; `lost` makes every request fail as a
    reset connection (another program took the core)."""

    def __init__(self):
        self.comm = FakeComm(self)
        self.compiles = []
        self.runs = 0
        self.lost = False
        self.closed = 0

    def precompile(self, function, *args, **kwargs):
        assert getattr(function, "artiq_embedded", None) is not None, "not a kernel"
        self.compiles.append((function.__name__, args))

        def run():
            if self.lost:
                raise ConnectionResetError("Core device connection closed unexpectedly")
            self.runs += 1
        return run

    def close(self):
        self.closed += 1
        self.comm.close()


class FakeDeviceManager:
    def __init__(self, core):
        self.core = core
        self.closed = 0

    def close_devices(self):
        self.closed += 1
        self.core.close()


class FakeMonitor:
    """The monitor server as the pulser sees it: announce / status / state /
    replace_state / send_end, with knobs for what it reports."""

    def __init__(self, trusted=True, version=7, run_pending=None, config=None):
        self.trusted = trusted
        self.version = version
        self.run_pending = run_pending
        self.config = config if config is not None else {
            "dds": {"imaging": {"frequency": 1.0}}, "ttl": {"andor": {"ttl_state": 1},
                                                            "other": {"ttl_state": 1}},
            "dac": {"z": {"voltage": 0.1}}, "metadata": {}}
        self.announced = []
        self.withdrawn = []
        self.replaced = []
        self.ends = 0
        self.closed = 0
        self.refuse_announce = False
        self.fail_state = False

    def announce_run(self, run_id=None, expt="", client="", token="", timeout=8.0):
        self.announced.append({"run_id": run_id, "expt": expt, "client": client,
                               "token": token})
        if self.refuse_announce:
            return {"status": "error", "msg": "no"}
        return {"status": "ok"}

    def withdraw_run(self, token, run_id=None, timeout=2.0):
        self.withdrawn.append(token)
        return {"status": "ok"}

    def get_state(self):
        if self.fail_state:
            raise OSError("no reply")
        return {"status": "ok", "version": self.version, "config": self.config,
                "trust": {"trusted": self.trusted, "reason": "" if self.trusted else "x"},
                "run_pending": self.run_pending}

    def get_status(self):
        return {"state": 0, "run_pending": self.run_pending,
                "trust": {"trusted": self.trusted}}

    def replace_state(self, config, run_id=None, expt=""):
        self.replaced.append({"config": config, "run_id": run_id, "expt": expt})
        self.trusted = True
        self.version += 1
        return {"status": "ok", "version": self.version}

    def send_end(self):
        self.ends += 1

    def close(self):
        self.closed += 1


def make(monitor=None, run_check=None, **kw):
    core = FakeCore()
    dmgr = FakeDeviceManager(core)
    p = TTLPulser("ttl7", device_db_path="unused.py", t_pulse=200e-9, label="test pulser",
                  ttl_state_name="andor", run_check=run_check,
                  core_factory=lambda: (dmgr, core, object()),
                  monitor_factory=(lambda: monitor) if monitor is not None else
                  (lambda: (_ for _ in ()).throw(RuntimeError("no monitor server"))), **kw)
    return p, core, dmgr


def test_kernel_object_holds_only_the_two_devices():
    k = _PulseKernel("core", "ttl")
    assert vars(k) == {"core": "core", "ttl": "ttl"}
    assert getattr(_PulseKernel.pulse, "artiq_embedded", None) is not None


def test_acquire_pulse_release_sequence_compiles_once_and_restarts_the_monitor():
    mon = FakeMonitor(trusted=True)
    p, core, dmgr = make(mon)
    assert not p.held
    with pytest.raises(PulserNotHeld):
        p.pulse()

    notes = p.acquire()
    assert p.held and core.comm.opened and core.comm.info_calls == 1
    assert len(core.compiles) == 1 and core.compiles[0][1] == (200e-9, p.t_lead)
    assert len(mon.announced) == 1 and mon.announced[0]["expt"] == "test pulser"
    assert mon.announced[0]["token"] and mon.announced[0]["run_id"] is None
    assert any("announced" in n for n in notes) and any("precompiled" in n for n in notes)

    t = [p.pulse() for _ in range(5)]
    assert core.runs == 5 and p.n_pulses == 5 and len(core.compiles) == 1
    assert t == sorted(t)

    notes = p.release()
    assert not p.held
    assert core.comm.info_calls == 2                 # the probe at release
    assert dmgr.closed == 1 and core.closed == 1
    # trusted before, version unchanged, no run: the state goes back, this TTL low
    assert len(mon.replaced) == 1
    sent = mon.replaced[0]
    assert sent["expt"] == "test pulser" and sent["run_id"] is None
    assert sent["config"]["ttl"]["andor"]["ttl_state"] == 0
    assert sent["config"]["ttl"]["other"]["ttl_state"] == 1
    assert sent["config"]["dds"] == mon.config["dds"] and sent["config"]["dac"] == mon.config["dac"]
    assert "metadata" not in sent["config"]
    assert mon.ends == 1
    assert any("trusted again" in n for n in notes) and any("restarts the monitor" in n
                                                             for n in notes)
    # the next hold compiles again (a new session)
    p.acquire()
    assert len(core.compiles) == 2 and p.n_holds == 2
    p.release()
    assert mon.ends == 2


def test_untrusted_state_is_not_made_trusted_but_the_monitor_restarts():
    mon = FakeMonitor(trusted=False)
    p, core, _ = make(mon)
    notes = p.acquire()
    assert any("already UNTRUSTED" in n for n in notes)
    p.pulse()
    notes = p.release()
    assert mon.replaced == [] and mon.ends == 1
    assert any("stays UNTRUSTED" in n for n in notes)


def test_state_changed_during_the_hold_is_not_overwritten():
    mon = FakeMonitor(trusted=True, version=3)
    p, _, _ = make(mon)
    p.acquire()
    mon.version = 4                     # someone edited the state file meanwhile
    notes = p.release()
    assert mon.replaced == [] and mon.ends == 1
    assert any("changed during the hold" in n for n in notes)


def test_a_run_announced_to_the_server_keeps_the_monitor_down():
    mon = FakeMonitor(trusted=True)
    p, _, _ = make(mon)
    p.acquire()
    mon.run_pending = {"run_id": 84999, "expt": "hf_bec", "token": "other"}
    notes = p.release()
    assert mon.replaced == [] and mon.ends == 0
    assert any("NOT restarted" in n and "84999" in n for n in notes)


def test_our_own_fence_still_pending_does_not_count_as_a_run():
    mon = FakeMonitor(trusted=True)
    p, _, _ = make(mon)
    p.acquire()
    mon.run_pending = {"run_id": None, "expt": "test pulser", "token": p._token}
    p.release()
    assert mon.ends == 1 and len(mon.replaced) == 1


def test_the_callers_run_check_keeps_the_monitor_down_and_fails_closed():
    mon = FakeMonitor(trusted=True)
    p, _, _ = make(mon, run_check=lambda: (False, "run 85000 in progress"))
    p.acquire()
    notes = p.release()
    assert mon.ends == 0 and mon.replaced == []
    assert any("run 85000 in progress" in n for n in notes)

    def broken():
        raise RuntimeError("liveOD gone")
    mon2 = FakeMonitor(trusted=True)
    p2, _, _ = make(mon2, run_check=broken)
    p2.acquire()
    notes = p2.release()
    assert mon2.ends == 0 and any("run check failed" in n for n in notes)


def test_core_taken_during_the_hold_raises_and_sends_nothing_at_release():
    mon = FakeMonitor(trusted=True)
    p, core, dmgr = make(mon)
    p.acquire()
    p.pulse()
    core.lost = True                                   # a run connected to the core
    with pytest.raises(CoreTaken) as e:
        p.pulse()
    assert "another program has the core" in str(e.value)
    with pytest.raises(CoreTaken):                     # and stays so
        p.pulse()
    notes = p.release()
    assert dmgr.closed == 1
    assert mon.ends == 0 and mon.replaced == []
    assert any("NOT restarted" in n for n in notes)
    assert p.dropped


def test_core_lost_silently_is_noticed_by_the_probe_at_release():
    mon = FakeMonitor(trusted=True)
    p, core, _ = make(mon)
    p.acquire()
    core.lost = True                                   # nothing pulsed since
    notes = p.release()
    assert mon.ends == 0 and mon.replaced == []
    assert any("did not answer at release" in n for n in notes)


def test_no_monitor_server_still_pulses_and_says_so():
    p, core, _ = make(monitor=None)
    notes = p.acquire()
    assert p.held and any("no monitor server found" in n for n in notes)
    p.pulse()
    notes = p.release()
    assert core.closed == 1 and any("no monitor server to tell" in n for n in notes)


def test_announcement_refused_is_noted_and_nothing_is_carried():
    mon = FakeMonitor(trusted=True)
    mon.refuse_announce = True
    p, _, _ = make(mon)
    notes = p.acquire()
    assert any("refused the announcement" in n for n in notes)
    notes = p.release()
    # not announced: no state was read, so trust cannot be carried; the
    # monitor is still restarted (it was interrupted all the same)
    assert mon.replaced == [] and mon.ends == 1
    assert any("no state was read" in n for n in notes)


def test_a_failed_connect_withdraws_the_announcement_and_holds_nothing():
    mon = FakeMonitor(trusted=True)
    core = FakeCore()

    def bad_factory():
        raise OSError("connection refused")
    p = TTLPulser("ttl7", device_db_path="x.py", label="test pulser",
                  core_factory=bad_factory, monitor_factory=lambda: mon)
    with pytest.raises(OSError):
        p.acquire()
    assert not p.held
    assert len(mon.announced) == 1 and mon.withdrawn == [mon.announced[0]["token"]]
    assert core.runs == 0 and mon.ends == 0


def test_release_without_a_hold_is_a_no_op_and_close_releases():
    mon = FakeMonitor()
    p, core, _ = make(mon)
    assert p.release() == []
    p.acquire()
    p.close()
    assert not p.held and core.closed == 1 and mon.closed == 1 and mon.ends == 1


def test_default_core_needs_a_device_db():
    import os
    saved = os.environ.pop("db", None)
    try:
        p = TTLPulser("ttl7", device_db_path=None, monitor_factory=lambda: FakeMonitor())
        with pytest.raises(Exception) as e:
            p.acquire()
        assert "device database" in str(e.value)
        assert not p.held
    finally:
        if saved is not None:
            os.environ["db"] = saved
