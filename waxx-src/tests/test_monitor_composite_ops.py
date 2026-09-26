"""Composite ops through the monitor server's request handler and the monitor
experiment's host side.

Nothing touches the network: the server's UDP broadcaster is a recorder (its
socket is never bound and its beacon never started), and the monitor's server
client is a fake that plays the server against a device-state file in a
pytest temp dir.  The kernel side (apply_ops) is covered by the offline
compile check, not here.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from waxx.base import monitor as wmon
from waxx.util.comms_server.comm_server import STATES
from waxx.util.device_state import composite as cmp
from waxx.util.device_state.composite import Arg, CompositeDevice, Op, OpTable, Table
from waxx.util.device_state.op_queue import OpQueue
from waxx.util.device_state.state_file_io import read_state
from waxx.util.guis import monitor_server_gui as msg


def _merge(path, dtype, name, changes):
    """The fake server's own merge into its temp state file (the real server
    path is exercised through the handler tests above)."""
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault(dtype, {}).setdefault(name, {}).update(changes)
    with open(path, "w") as f:
        json.dump(cfg, f)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


STATE = {
    "dds": {"imaging": {"frequency": 350e6, "amplitude": 1.0, "v_pd": 0.3, "sw_state": 1,
                        "urukul_idx": 4, "ch": 1, "dac_ch_key": "imaging_pid"},
            "raman_switch": {"frequency": 150e6, "amplitude": 0.46, "v_pd": 0.0,
                             "sw_state": 0, "urukul_idx": 5, "ch": 1, "dac_ch_key": ""}},
    "dac": {"imaging_pid": {"ch": 20, "voltage": 0.3}, "coil": {"ch": 9, "voltage": 0.0}},
    "ttl": {"shutter": {"ch": 22, "ttl_state": 0}},
}


def _host_step(expt, args, payload):
    if payload["rows"] and payload["rows"][0][0] < 0:
        raise RuntimeError("card said no")
    expt.loaded = payload["rows"]


DEVICE = CompositeDevice(
    key="beam", title="Beam",
    fields=(Arg("v", minimum=0., maximum=5.),),
    tables=(Table("rows", "Rows", columns=(Arg("x", minimum=-1., maximum=1.),)),),
    ops=(Op("on", "On", args=("v",), code="expt.beam.on({v})"),
         Op("load", "Load", payload=("rows",), host=_host_step, code="expt.beam.trig()")),
    host_state=lambda expt: {"loaded": list(getattr(expt, "loaded", []))},
)


# --- server ----------------------------------------------------------------------

class Recorder:
    def __init__(self, *a, **k):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


class Clock:
    t = 50.

    def __call__(self):
        return self.t


@pytest.fixture
def server(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    path = tmp_path / "state.json"
    path.write_text(json.dumps(STATE))
    s = msg.MonitorUDPServer(config_file_path=str(path))
    s.clock = Clock()
    s.ops = OpQueue(clock=s.clock)
    s.path = path
    yield s
    s.sock.close()


def _ask(server, obj):
    return json.loads(server.generate_reply(json.dumps(obj)))


def _registered_ready(server):
    table = OpTable([DEVICE])
    assert _ask(server, table.registration(session="t"))["status"] == "ok"
    server.status.set_state(STATES.READY, "running")
    return table


def test_op_refused_unless_monitor_ready(server):
    table = OpTable([DEVICE])
    _ask(server, table.registration())
    e = table.get("beam.on")
    reply = _ask(server, {"type": "op", "op": e.name, "sig": e.signature, "args": {"v": 1}})
    assert reply["status"] == "error"
    assert reply["msg"].startswith("the monitor is not ready (never started)")
    assert server.ops.info()["queued"] == 0


def test_op_round_trip_through_the_handler(server):
    table = _registered_ready(server)
    e = table.get("beam.on")
    reply = _ask(server, {"type": "op", "op": e.name, "sig": e.signature, "args": {"v": 2.},
                          "client": "gui-pc"})
    assert reply["status"] == "ok"
    seq = reply["seq"]
    poll = _ask(server, {"type": "poll"})
    assert poll["version"] == server._version
    assert [o["seq"] for o in poll["ops"]] == [seq]
    assert poll["ops"][0]["args"][:1] == [2.]
    assert _ask(server, {"type": "op_done", "results": [
        {"seq": seq, "status": cmp.OP_OK, "message": ""}]})["status"] == "ok"
    results = [p for p in server._broadcaster.sent if p.get("type") == "op_result"]
    assert results and results[-1]["seq"] == seq and results[-1]["ok"]
    assert _ask(server, {"type": "op_status", "seq": seq})["state"] == "done"


def test_status_json_reports_registration(server):
    _registered_ready(server)
    detail = json.loads(server.generate_reply("status_json"))
    assert detail["composite_ops"]["registered"] is True
    assert detail["composite_ops"]["count"] == len(OpTable([DEVICE]))


def test_not_ready_retires_but_loading_does_not(server):
    table = _registered_ready(server)
    e = table.get("beam.on")
    seq = _ask(server, {"type": "op", "op": e.name, "sig": e.signature, "args": {"v": 1.}})["seq"]
    # A starting monitor registers while the owner still says LOADING.
    server.on_monitor_state(STATES.LOADING, "starting")
    assert server.ops.registered
    server.on_monitor_state(STATES.NOT_READY, "interrupted_by_run")
    assert not server.ops.registered
    expired = [p for p in server._broadcaster.sent if p.get("type") == "op_result"]
    assert expired[-1]["seq"] == seq and expired[-1]["status"] == cmp.OP_EXPIRED


def test_any_request_sweeps_expired_ops(server):
    table = _registered_ready(server)
    e = table.get("beam.on")
    seq = _ask(server, {"type": "op", "op": e.name, "sig": e.signature, "args": {"v": 1.}})["seq"]
    server.clock.t += 10.
    server.generate_reply("status")             # a GUI's status poll
    sent = [p for p in server._broadcaster.sent if p.get("type") == "op_result"]
    assert sent and sent[-1]["seq"] == seq and sent[-1]["status"] == cmp.OP_EXPIRED


def test_update_batch_bumps_version_per_device_and_propagates_links(server):
    v0 = server._version
    reply = _ask(server, {"type": "update_batch", "origin": "monitor write-back", "updates": [
        {"device_type": "dds", "device_name": "imaging", "changes": {"v_pd": 0.5}},
        {"device_type": "ttl", "device_name": "shutter", "changes": {"ttl_state": 1}}]})
    assert reply["status"] == "ok"
    cfg = read_state(server.path)
    assert cfg["dds"]["imaging"]["v_pd"] == 0.5
    assert cfg["dac"]["imaging_pid"]["voltage"] == 0.5        # linked DAC mirrored
    assert cfg["ttl"]["shutter"]["ttl_state"] == 1
    updates = [p for p in server._broadcaster.sent if p.get("type") == "state_update"]
    assert [u["version"] for u in updates] == list(range(v0 + 1, v0 + 1 + len(updates)))
    assert reply["version"] == server._version


def test_a_run_fence_is_lifted_only_by_its_own_token(server):
    table = _registered_ready(server)
    e = table.get("beam.on")
    assert _ask(server, {"type": "run_pending", "run_id": 81000, "expt": "x",
                         "token": "tok-a"})["status"] == "ok"
    refused = _ask(server, {"type": "op", "op": e.name, "sig": e.signature, "args": {"v": 1.}})
    assert refused["status"] == "error" and "81000" in refused["msg"]
    # a stale or foreign token changes nothing
    assert _ask(server, {"type": "run_withdrawn", "token": "tok-b"})["status"] == "error"
    assert _ask(server, {"type": "run_withdrawn"})["status"] == "error"
    assert server._run_pending is not None
    # the run withdraws itself at exit
    assert _ask(server, {"type": "run_withdrawn", "token": "tok-a",
                         "run_id": 81000})["status"] == "ok"
    assert server._run_pending is None
    assert server._broadcaster.sent[-1] == {"type": "run_pending", "run_pending": None}
    assert "exited without taking the core" in server.journal.tail(1)[0]["why"]
    assert _ask(server, {"type": "op", "op": e.name, "sig": e.signature,
                         "args": {"v": 1.}})["status"] == "ok"
    # withdrawn twice (or after the run took the core): harmless
    assert _ask(server, {"type": "run_withdrawn", "token": "tok-a"})["status"] == "error"


def test_an_operator_can_clear_a_dead_runs_fence(server):
    _registered_ready(server)
    _ask(server, {"type": "run_pending", "run_id": 7, "token": "tok"})
    public = [p for p in server._broadcaster.sent if p.get("type") == "run_pending"][-1]
    assert public["run_pending"]["token"] == "tok" and "t0" not in public["run_pending"]
    reply = _ask(server, {"type": "clear_run_pending", "token": "tok", "operator": "ada",
                          "client": "pc2"})
    assert reply["status"] == "ok" and server._run_pending is None
    assert "cleared on the Device Control GUI by ada" in server.journal.tail(1)[0]["why"]


def test_get_state_carries_device_host_state(server):
    table = _registered_ready(server)
    e = table.get("beam.load")
    seq = _ask(server, {"type": "op", "op": e.name, "sig": e.signature, "args": {},
                        "payload": {"rows": [[0.5]]}})["seq"]
    _ask(server, {"type": "poll"})
    _ask(server, {"type": "op_done", "results": [
        {"seq": seq, "status": cmp.OP_OK, "device": "beam", "state": {"loaded": [[0.5]]}}]})
    state = _ask(server, {"type": "get_state"})
    assert state["composite_state"] == {"beam": {"loaded": [[0.5]]}}


# --- monitor host side ---------------------------------------------------------------

class FakeServer:
    """Plays the monitor server for a Monitor: a real state file in tmp_path
    and a real OpQueue, driven in-process."""

    def __init__(self, path):
        self.path = path
        self.version = 1
        self.ops = OpQueue()
        self.reports = []
        self.batches = []
        self.refuse_batches = False
        self.requests = []
        self.replaced = []
        self.registrations = 0
        self.withdrawn = []
        self.announced = None

    # MonitorClient API used by Monitor
    def poll(self):
        return {"status": "ok", "version": self.version, "ops": self.ops.pop(),
                "registered": self.ops.registered}

    def send_message(self, message):
        return json.dumps({"status": "ok", "version": self.version})

    def register_ops(self, reg):
        self.registrations += 1
        return self.ops.register(reg)

    def report_ops(self, results):
        self.reports.extend(results)
        self.ops.done(results)
        return {"status": "ok"}

    def send_update_batch(self, updates, origin=""):
        self.batches.append(updates)
        if self.refuse_batches:
            return None                         # unreachable
        first = self.version
        for dtype, name, changes in updates:
            _merge(self.path, dtype, name, changes)
            self.version += 1
        return {"status": "ok", "version": self.version, "first_version": first}

    def request(self, obj):
        self.requests.append(obj)
        return {"status": "ok"}

    def replace_state(self, config, run_id=None, expt=""):
        self.replaced.append((config, run_id, expt))
        return {"status": "ok", "version": self.version + 1}

    def announce_run(self, run_id=None, expt="", client="", token=""):
        self.announced = token
        return {"status": "ok"}

    def withdraw_run(self, token, run_id=None, timeout=2.0):
        self.withdrawn.append((token, run_id))
        return {"status": "ok"}

    def send_ready(self):
        pass

    def gui_edit(self, dtype, name, changes):
        _merge(self.path, dtype, name, changes)
        self.version += 1


@pytest.fixture
def monitor(monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(STATE))
    fake = FakeServer(str(path))
    monkeypatch.setattr(wmon, "MonitorClient", lambda *a, **k: fake)
    expt = SimpleNamespace()
    m = wmon.Monitor(expt, device_state_json_path=str(path))
    m.dds_dict = {"imaging": 0, "raman_switch": 1}
    m.dac_dict = {"imaging_pid": 0, "coil": 1}
    m.ttl_dict = {"shutter": 0}
    m._snap_dds_keys = ["imaging", "raman_switch"]
    m._snap_dac_keys = ["imaging_pid", "coil"]
    m._snap_ttl_keys = ["shutter"]
    m._op_table = OpTable([DEVICE])
    m._composites_enabled = True
    m.fake = fake
    m.expt = expt
    return m


def _updates_all_default(lists):
    dds_fa, dds_v, dds_sw, ttl, dac = lists
    return (all(u == wmon.DEFAULT_UPDATE_2FLOAT for u in dds_fa)
            and all(u == wmon.DEFAULT_UPDATE_FLOAT for u in dds_v)
            and all(u == wmon.DEFAULT_UPDATE_INT for u in dds_sw)
            and all(u == wmon.DEFAULT_UPDATE_INT for u in ttl)
            and all(u == wmon.DEFAULT_UPDATE_FLOAT for u in dac))


def _snapshot(dds_f=(350e6, 150e6), dds_a=(1.0, 0.46), dds_v=(0.3, 0.0), dds_sw=(1, 0),
              dac_v=(0.3, 0.0), ttl_s=(0,)):
    return (np.array(dds_f), np.array(dds_a), np.array(dds_v),
            np.array(dds_sw, dtype=np.int32), np.array(dac_v), np.array(ttl_s, dtype=np.int32))


def test_write_back_is_not_applied_a_second_time(monitor):
    fake = monitor.fake
    assert _updates_all_default(monitor.detect_changes(verbose=False))   # initial load
    # An op switched raman_switch on, opened the shutter and moved the coil DAC;
    # the imaging frequency differs only by FTW quantization.
    snap = _snapshot(dds_f=(350e6 + 0.1, 150e6), dds_sw=(1, 1), dac_v=(0.3, 2.0), ttl_s=(1,))
    monitor._write_back_channels(*snap)
    sent = {(t, n): c for t, n, c in fake.batches[-1]}
    assert sent == {("dds", "raman_switch"): {"sw_state": 1},
                    ("dac", "coil"): {"voltage": 2.0},
                    ("ttl", "shutter"): {"ttl_state": 1}}
    # The server applied them and moved the version; the monitor re-reads the
    # file and finds nothing to do -- no second write of the same values.
    assert _updates_all_default(monitor.detect_changes(verbose=False))
    # ...but a real edit after that is still picked up.
    fake.gui_edit("dac", "coil", {"voltage": 3.0})
    dac = monitor.detect_changes(verbose=False)[4]
    assert dac[0] == (monitor.dac_dict["coil"], 3.0)


def test_accepted_write_back_skips_the_re_read(monitor, monkeypatch):
    fake = monitor.fake
    monitor.detect_changes(verbose=False)
    monitor._write_back_channels(*_snapshot(ttl_s=(1,)))
    assert monitor._last_seen_version == fake.version       # nothing else landed
    reads = []
    monkeypatch.setattr(monitor, "load_config_file", lambda: reads.append(1))
    assert _updates_all_default(monitor.detect_changes(verbose=False))
    assert reads == []


def test_failed_write_back_never_reverts_the_hardware(monitor):
    """The old code folded the write-back into last_config_data before the
    server accepted it; when it was refused, the next file read saw the old
    values as a change and put them back on the hardware."""
    fake = monitor.fake
    monitor.detect_changes(verbose=False)
    fake.refuse_batches = True
    monitor._write_back_channels(*_snapshot(dac_v=(0.3, 2.0)))    # coil ramped to 2 V
    assert monitor._writeback_pending == {("dac", "coil"): {"voltage": 2.0}}
    # An unrelated GUI edit moves the version: the file is re-read...
    fake.gui_edit("ttl", "shutter", {"ttl_state": 1})
    lists = monitor.detect_changes(verbose=False)
    # ...and only that edit is applied -- the coil is NOT set back to 0 V.
    assert lists[4][0] == wmon.DEFAULT_UPDATE_FLOAT
    assert lists[3][0] == (monitor.ttl_dict["shutter"], 1)
    # The server comes back: the retry lands and is folded.
    fake.refuse_batches = False
    monitor._writeback_retry_after = 0.
    monitor.poll_changes()
    assert monitor._writeback_pending == {}
    assert read_state(fake.path)["dac"]["coil"]["voltage"] == 2.0
    assert _updates_all_default(monitor.detect_changes(verbose=False))


def test_pending_write_back_yields_to_a_later_edit(monitor):
    fake = monitor.fake
    monitor.detect_changes(verbose=False)
    fake.refuse_batches = True
    monitor._write_back_channels(*_snapshot(dac_v=(0.3, 2.0)))
    fake.gui_edit("dac", "coil", {"voltage": 1.0})             # someone set it since
    lists = monitor.detect_changes(verbose=False)
    assert lists[4][0] == (monitor.dac_dict["coil"], 1.0)
    assert monitor._writeback_pending == {}                    # their value wins
    fake.refuse_batches = False
    monitor._writeback_retry_after = 0.
    n = len(fake.batches)
    monitor.poll_changes()
    assert len(fake.batches) == n                              # nothing stale re-sent


def test_poll_flags(monitor):
    fake = monitor.fake
    monitor._register_ops()
    assert monitor.poll_changes() == 0                         # initial load
    assert monitor.poll_changes() == 0                         # idle
    fake.gui_edit("dac", "coil", {"voltage": 1.0})
    assert monitor.poll_changes() == wmon.FLAG_CHANNELS
    assert monitor.channel_updates()[4][0] == (monitor.dac_dict["coil"], 1.0)
    e = monitor._op_table.get("beam.on")
    fake.ops.submit({"op": e.name, "sig": e.signature, "args": {"v": 1.}})
    assert monitor.poll_changes() == wmon.FLAG_OPS
    monitor.fetch_ops()
    assert monitor.poll_changes() == 0


def test_update_lists_are_sized_to_the_frames_and_cleared_on_read_failure(monitor, monkeypatch):
    fake = monitor.fake
    monitor.detect_changes(verbose=False)
    assert len(monitor.dds_vpd_updates) == len(monitor.dds_dict) + 1
    assert len(monitor.ttl_updates) == len(monitor.ttl_dict) + 1
    assert len(monitor.dac_updates) == len(monitor.dac_dict) + 1
    fake.gui_edit("dac", "coil", {"voltage": 1.0})
    assert monitor.detect_changes(verbose=False)[4][0][0] != -1
    fake.version += 1
    monkeypatch.setattr(monitor, "load_config_file", lambda: None)
    assert _updates_all_default(monitor.detect_changes(verbose=False))


def test_monitor_re_registers_with_a_restarted_server(monitor):
    fake = monitor.fake
    monitor._register_ops()
    assert fake.registrations == 1
    fake.ops = OpQueue()                                       # server restarted
    monitor.detect_changes(verbose=False)
    assert fake.registrations == 2 and fake.ops.registered
    fake.ops = OpQueue()
    monitor.detect_changes(verbose=False)                      # backoff: not yet
    assert fake.registrations == 2


def test_end_state_goes_through_the_server_not_the_file(monitor, tmp_path):
    before = read_state(monitor.fake.path)

    class Gen:
        config_data = None

        def generate_device_config(self):
            self.config_data = {"dds": {"imaging": {"frequency": np.float64(1.0)}},
                                "ttl": {}, "dac": {}, "metadata": {}}

    monitor.generator = Gen()
    assert monitor.update_device_states(run_id=123, expt="x") is True
    config, run_id, expt = monitor.fake.replaced[-1]
    assert run_id == 123 and expt == "x"
    assert set(config) == {"dds", "ttl", "dac"}
    assert type(config["dds"]["imaging"]["frequency"]) is float
    assert read_state(monitor.fake.path) == before             # untouched here
    monitor.fake.replace_state = lambda *a, **k: None
    assert monitor.update_device_states(run_id=124) is False
    assert read_state(monitor.fake.path) == before


def test_polled_ops_are_rechecked_and_packed_by_the_monitor(monitor):
    fake = monitor.fake
    monitor._register_ops()
    e_on, e_load = monitor._op_table.get("beam.on"), monitor._op_table.get("beam.load")
    fake.ops.submit({"op": e_on.name, "sig": e_on.signature, "args": {"v": 2.}})
    # A request the server accepted but outside the monitor's hard limits
    # (defense in depth: the server knows only names and signatures).
    bad = fake.ops.submit({"op": e_on.name, "sig": e_on.signature, "args": {"v": 9.}})["seq"]
    monitor.detect_changes(verbose=False)
    rejected = [r for r in fake.reports if r["seq"] == bad]
    assert rejected and rejected[0]["status"] == cmp.OP_REJECTED
    assert "above the limit" in rejected[0]["message"]
    slots = monitor.fetch_ops()
    assert slots[0][0] == e_on.index and slots[0][2] == 2.
    assert slots[1] == wmon.DEFAULT_OP
    assert len(slots) == wmon.N_OP_SLOTS
    assert e_load  # (used below)


def test_host_step_success_failure_and_report(monitor):
    fake = monitor.fake
    monitor._register_ops()
    e = monitor._op_table.get("beam.load")
    ok = fake.ops.submit({"op": e.name, "sig": e.signature, "args": {},
                          "payload": {"rows": [[0.5]]}})["seq"]
    bad = fake.ops.submit({"op": e.name, "sig": e.signature, "args": {},
                           "payload": {"rows": [[-0.5]]}})["seq"]
    monitor.detect_changes(verbose=False)
    monitor.fetch_ops()
    assert monitor.run_op_host_step(ok) == cmp.OP_OK
    assert monitor.expt.loaded == [[0.5]]
    assert monitor.run_op_host_step(bad) == cmp.OP_HOST_ERROR
    monitor.last_config_data = read_state(fake.path)
    snap = _snapshot()
    monitor.report_ops(2, np.array([ok, bad], dtype=np.int32),
                       np.array([cmp.OP_OK, cmp.OP_HOST_ERROR], dtype=np.int32), *snap)
    by_seq = {r["seq"]: r for r in fake.reports}
    assert by_seq[ok]["status"] == cmp.OP_OK
    assert by_seq[ok]["state"] == {"loaded": [[0.5]]} and by_seq[ok]["device"] == "beam"
    assert by_seq[bad]["status"] == cmp.OP_HOST_ERROR
    assert "card said no" in by_seq[bad]["message"]
    assert fake.ops.device_state["beam"] == {"loaded": [[0.5]]}


def test_sync_cache_seeds_frames_from_the_state_file(monitor):
    class Dev:
        # AD9910-like conversions: 0.25 Hz FTW step, 1/16383 ASF step
        def frequency_to_ftw(self, f):
            return int(round(f * 4))

        def ftw_to_frequency(self, ftw):
            return ftw / 4

        def amplitude_to_asf(self, a):
            return int(round(a * 16383))

        def asf_to_amplitude(self, asf):
            return asf / 16383

    def dds():
        return SimpleNamespace(frequency=1.0, amplitude=0.1, v_pd=9.0, sw_state=0,
                               _ftw=0, _asf=0, dds_device=Dev())

    monitor.dds = SimpleNamespace(imaging=dds(), raman_switch=dds())
    monitor.dac = SimpleNamespace(imaging_pid=SimpleNamespace(v=0.), coil=SimpleNamespace(v=0.))
    monitor.ttl = SimpleNamespace(shutter=SimpleNamespace(state=0))
    seen = []
    monitor._op_table = OpTable([CompositeDevice(
        key="hooked", title="h", ops=(Op("x", "x", code="pass"),),
        sync_cache=lambda expt, cfg: seen.append(sorted(cfg)))])
    monitor.sync_cache_from_state_file()
    img = monitor.dds.imaging
    assert img.frequency == 350e6 and img._ftw == 1400000000
    assert img.amplitude == 1.0 and img._asf == 16383
    assert img.v_pd == 0.3 and img.sw_state == 1
    assert monitor.dac.imaging_pid.v == 0.3
    assert monitor.ttl.shutter.state == 0
    assert seen == [["dac", "dds", "ttl"]]


def test_snapshot_kernel_source_covers_every_channel(monitor):
    monitor._build_snapshot_kernel()
    src = monitor._snapshot_kernels[0].artiq_embedded.function
    for i, name in enumerate(["imaging", "raman_switch"]):
        assert f"s[{i}] = expt.dds.{name}.sw_state" in src
        assert f"f[{i}] = expt.dds.{name}.frequency" in src
    assert "dv[1] = expt.dac.coil.v" in src
    assert "ts[0] = expt.ttl.shutter.state" in src
    assert monitor._snap_ttl_s.dtype == np.int32


def test_op_kernels_share_one_signature(monitor):
    monitor._build_op_kernels()
    assert len(monitor.op_kernels) == len(monitor._op_table)
    for fn, entry in zip(monitor.op_kernels, monitor._op_table):
        decl = fn.artiq_embedded.function
        assert decl.startswith("def kernel_from_string_fn(expt,a0,a1,a2,a3,a4,a5,a6,a7,):")
        assert decl.splitlines()[1].strip() == cmp.SLACK_PREFIX
    assert monitor._op_host_before == [False, False, True]
    assert monitor._op_host_after == [False, False, False]


def test_a_run_that_dies_before_its_end_withdraws_its_fence_at_exit(monitor, monkeypatch):
    registered = []
    import atexit
    monkeypatch.setattr(atexit, "register", registered.append)
    monitor.announce_run(run_id=81000, expt="x")
    token = monitor.fake.announced
    assert token and registered == [monitor.withdraw_run_at_exit]
    monitor.announce_run(run_id=81001, expt="x")             # registered once only
    assert len(registered) == 1
    token = monitor.fake.announced
    registered[0]()                                          # interpreter exit
    assert monitor.fake.withdrawn == [(token, 81001)]
    registered[0]()                                          # nothing twice
    assert len(monitor.fake.withdrawn) == 1


def test_a_run_whose_end_state_landed_withdraws_nothing(monitor, monkeypatch):
    registered = []
    import atexit
    monkeypatch.setattr(atexit, "register", registered.append)

    class Gen:
        config_data = None

        def generate_device_config(self):
            self.config_data = {"dds": {}, "ttl": {}, "dac": {}}

    monitor.generator = Gen()
    monitor.announce_run(run_id=5)
    assert monitor.update_device_states(run_id=5) is True
    registered[0]()
    assert monitor.fake.withdrawn == []
    # an end state the server refused leaves the withdraw in place
    monitor.announce_run(run_id=6)
    monitor.fake.replace_state = lambda *a, **k: None
    assert monitor.update_device_states(run_id=6) is False
    registered[0]()
    assert [r for _, r in monitor.fake.withdrawn] == [6]
