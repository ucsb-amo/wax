"""Composite-device definitions (waxx.util.device_state.composite) and the
server's op queue (waxx.util.device_state.op_queue).  Pure Python: no Qt, no
ARTIQ, no sockets."""
import math

import pytest

from waxx.util.device_state import composite as cmp
from waxx.util.device_state.composite import (
    Arg, Buttons, Check, CompositeDevice, Context, DefinitionError, FieldRow, Menu, Op,
    OpTable, Table, TableRow,
)
from waxx.util.device_state.op_queue import OpQueue


def _host_step(expt, args, payload):
    expt.calls.append((args, payload))


def _device(code="expt.thing.set({v})", **op_kw):
    return CompositeDevice(
        key="thing", title="Thing",
        fields=(Arg("v", "Volts", unit="V", minimum=0., maximum=5., warn_above=4.),
                Arg("n", "Count", kind=cmp.KIND_INT, minimum=0, maximum=10),
                Arg("on", "On", kind=cmp.KIND_BOOL),
                Arg("mode", "Mode", kind=cmp.KIND_CHOICE, choices=(("a", 0), ("b", 1)))),
        tables=(Table("rows", "Rows", columns=(Arg("x", "x", minimum=0., maximum=1.),)),),
        ops=(Op("set", "Set", code=code, args=("v",), **op_kw),
             Op("multi", "Multi", args=("n", "on", "mode", "v"),
                code="expt.thing.go({n}, {on}, {mode}, {v})"),
             Op("load", "Load", payload=("rows",), host=_host_step, code="expt.thing.trig()")),
        layout=(Buttons(("set",)), FieldRow(("v",), ("set",)), TableRow("rows", ("load",)),
                Menu(("multi",))),
    )


# --- definitions -----------------------------------------------------------------

def test_kernel_body_prefix_and_placeholders():
    table = OpTable([_device()])
    body = table.get("thing.multi").body
    lines = body.splitlines()
    # every op starts with the attribute-read slack delay (indeterminate I/O delay)
    assert lines[0] == cmp.SLACK_PREFIX
    assert lines[1] == "expt.thing.go(int(a0), (a1 > 0.5), int(a2), a3)"


def test_ping_is_index_zero_and_indices_follow_definition_order():
    table = OpTable([_device()])
    assert table.by_index(0).name == cmp.PING_OP
    assert table.by_index(0).body == cmp.SLACK_PREFIX
    assert [e.name for e in table][1:] == ["thing.set", "thing.multi", "thing.load"]


def test_signature_changes_with_code_and_hash_follows():
    a, b = OpTable([_device()]), OpTable([_device(code="expt.thing.set({v})\ndelay(1.e-3)")])
    assert a.get("thing.set").signature != b.get("thing.set").signature
    assert a.get("thing.multi").signature == b.get("thing.multi").signature
    assert a.hash != b.hash
    assert OpTable([_device()]).hash == a.hash      # deterministic


def test_pack_uses_op_argument_order():
    e = OpTable([_device()]).get("thing.multi")
    assert e.pack({"v": 2.5, "n": 3, "on": 1, "mode": 1}) == [3., 1., 1., 2.5, 0., 0., 0., 0.]


@pytest.mark.parametrize("bad, match", [
    (dict(code="expt.thing.set({w})"), "does not send"),
    (dict(code=""), "does nothing"),
])
def test_validate_rejects(bad, match):
    with pytest.raises(DefinitionError, match=match):
        OpTable([_device(**bad)])


def test_validate_rejects_payload_without_host_and_unknown_layout():
    d = _device()
    with pytest.raises(DefinitionError, match="no host step"):
        CompositeDevice(key="x", title="x", tables=d.tables,
                        ops=(Op("o", "o", payload=("rows",), code="pass"),)).validate()
    with pytest.raises(DefinitionError, match="unknown ops"):
        CompositeDevice(key="x", title="x", ops=(Op("o", "o", code="pass"),),
                        layout=(Buttons(("nope",)),)).validate()
    with pytest.raises(DefinitionError, match="max"):
        fields = tuple(Arg(f"f{i}") for i in range(cmp.N_OP_ARGS + 1))
        CompositeDevice(key="x", title="x", fields=fields,
                        ops=(Op("o", "o", code="pass", args=tuple(f.name for f in fields)),)
                        ).validate()
    with pytest.raises(DefinitionError):
        OpTable([_device(), _device()])          # duplicate device key
    with pytest.raises(DefinitionError):
        OpTable([CompositeDevice(key="monitor", title="m")])


def test_arg_checks_hard_soft_and_custom():
    v = Arg("v", "Volts", unit="V", minimum=0., maximum=5., warn_above=4.,
            check=lambda value, ctx: Check.warn("custom") if value == 3. else None)
    assert v.checks(2.) == []
    assert [c.level for c in v.checks(4.5)] == ["warn"]
    assert [c.message for c in v.checks(3.)] == ["custom"]
    assert v.checks(6.)[0].is_error and "above the limit" in v.checks(6.)[0].message
    assert v.checks(float("nan"))[0].is_error
    assert v.checks("x")[0].is_error
    choice = Arg("m", kind=cmp.KIND_CHOICE, choices=(("a", 0), ("b", 1)))
    assert choice.hard_problems(2.)[0].is_error
    assert choice.hard_problems(0.5)[0].is_error
    assert choice.hard_problems(1.) == []


def test_arg_display_scale_and_format():
    f = Arg("f", "Freq", unit="MHz", scale=1e-6, decimals=2)
    assert f.to_display(110e6) == pytest.approx(110.)
    assert f.from_display(110.) == pytest.approx(110e6)
    assert f.format(110.123e6) == "110.12 MHz"
    assert f.format(None) == "--"


def test_default_and_readback_resolve_callables_and_swallow_errors():
    a = Arg("a", default=lambda ctx: ctx.params.x, readback=lambda ctx: 1 / 0)
    ctx = Context(params=type("P", (), {"x": 2.0})())
    assert a.default_value(ctx) == 2.0
    assert a.readback_value(ctx) is None       # a broken readback reads as unknown


def test_entry_hard_problems_rechecks_limits_and_payload():
    table = OpTable([_device()])
    assert table.get("thing.set").hard_problems({"v": 1.}) == []
    assert "above the limit" in table.get("thing.set").hard_problems({"v": 9.})[0]
    assert "missing argument" in table.get("thing.set").hard_problems({})[0]
    load = table.get("thing.load")
    assert load.hard_problems({}, {"rows": [[0.5]]}) == []
    assert load.hard_problems({}, {}) == ["missing table 'rows'"]
    assert load.hard_problems({}, {"rows": [[2.0]]})


def test_table_checks():
    t = Table("rows", "Rows", columns=(Arg("x", "x", minimum=0., maximum=1.),),
              max_rows=2, check=lambda rows, ctx: Check.warn("sum") if len(rows) == 2 else None)
    assert t.checks([[0.1]]) == []
    assert [c.message for c in t.checks([[0.1], [0.2]])] == ["sum"]
    assert t.checks([[0.1]] * 3)[0].is_error
    assert t.checks([[5.]])[0].is_error


def test_context_helpers():
    cfg = {"dds": {"a": {"frequency": 1e6, "sw_state": 1, "amplitude": 0.5, "v_pd": 0.2}},
           "ttl": {"t": {"ttl_state": 0}}, "dac": {"d": {"voltage": -1.5}}}
    ctx = Context(cfg)
    assert ctx.is_on("dds", "a") is True
    assert ctx.is_on("ttl", "t") is False
    assert ctx.is_on("ttl", "missing") is None
    assert ctx.dac_voltage("d") == -1.5
    assert ctx.dds_frequency("a") == 1e6
    assert ctx.dds_v_pd("a") == 0.2


# --- op queue ----------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 100.

    def __call__(self):
        return self.t


@pytest.fixture
def queue():
    clock = Clock()
    q = OpQueue(ttl_s=3.0, clock=clock, wall_clock=lambda: 1_700_000_000.)
    q.clock = clock
    q.table = OpTable([_device()])
    return q


def _register(q):
    return q.register(q.table.registration(session="s"))


def _req(q, name, args, payload=None, sig=None):
    e = q.table.get(name)
    return {"op": name, "sig": sig or e.signature, "args": args, "payload": payload or {}}


def test_submit_refused_before_registration(queue):
    reply = queue.submit(_req(queue, "thing.set", {"v": 1.}))
    assert reply["status"] == "error" and "not registered" in reply["msg"]


def test_submit_pop_done_round_trip(queue):
    assert _register(queue)["status"] == "ok"
    reply = queue.submit(_req(queue, "thing.multi", {"v": 2.5, "n": 3, "on": 1, "mode": 1}),
                         client="gui")
    assert reply["status"] == "ok"
    seq = reply["seq"]
    assert queue.status(seq)["state"] == "queued"
    popped = queue.pop()
    assert len(popped) == 1
    assert popped[0]["args"] == [3., 1., 1., 2.5, 0., 0., 0., 0.]
    assert popped[0]["index"] == queue.table.get("thing.multi").index
    assert queue.status(seq)["state"] == "running"
    assert queue.pop() == []
    queue.clock.t += 0.4
    results = queue.done([{"seq": seq, "status": cmp.OP_OK, "message": ""}])
    assert results[0]["ok"] and results[0]["op"] == "thing.multi"
    assert results[0]["elapsed"] == pytest.approx(0.4)
    assert results[0]["client"] == "gui"
    assert queue.status(seq)["state"] == "done"


def test_signature_mismatch_and_unknown_op_refused(queue):
    _register(queue)
    r = queue.submit(_req(queue, "thing.set", {"v": 1.}, sig="0" * 16))
    assert r["status"] == "error" and "different" in r["msg"]
    r = queue.submit({"op": "thing.nope", "sig": "x", "args": {}})
    assert r["status"] == "error" and "no op" in r["msg"]


def test_missing_or_nonfinite_args_refused(queue):
    _register(queue)
    assert "missing" in queue.submit(_req(queue, "thing.set", {}))["msg"]
    assert "finite" in queue.submit(_req(queue, "thing.set", {"v": math.inf}))["msg"]
    assert "payload" in queue.submit(_req(queue, "thing.load", {}))["msg"]


def test_expiry_fails_ops_the_monitor_never_took(queue):
    _register(queue)
    seq = queue.submit(_req(queue, "thing.set", {"v": 1.}))["seq"]
    queue.clock.t += 2.9
    assert queue.expire() == []
    queue.clock.t += 0.2
    results = queue.expire()
    assert [r["seq"] for r in results] == [seq]
    assert results[0]["status"] == cmp.OP_EXPIRED and not results[0]["ok"]
    assert queue.pop() == []                     # an expired op never reaches the monitor


def test_unregister_expires_queued_and_loses_running(queue):
    _register(queue)
    s1 = queue.submit(_req(queue, "thing.set", {"v": 1.}))["seq"]
    queue.pop()
    s2 = queue.submit(_req(queue, "thing.set", {"v": 2.}))["seq"]
    results = {r["seq"]: r for r in queue.unregister("monitor restarting")}
    assert results[s1]["status"] == cmp.OP_LOST
    assert "may or may not" in results[s1]["status_text"]
    assert results[s2]["status"] == cmp.OP_EXPIRED
    assert not queue.registered
    assert queue.submit(_req(queue, "thing.set", {"v": 1.}))["status"] == "error"


def test_done_records_device_state_and_unregister_clears_it(queue):
    _register(queue)
    seq = queue.submit(_req(queue, "thing.load", {}, payload={"rows": [[0.1]]}))["seq"]
    popped = queue.pop()
    assert popped[0]["payload"] == {"rows": [[0.1]]}
    queue.done([{"seq": seq, "status": cmp.OP_OK, "device": "thing",
                 "state": {"connected": True}}])
    assert queue.device_state == {"thing": {"connected": True}}
    queue.unregister("stopped")
    assert queue.device_state == {}


def test_monitor_side_rejection_without_dispatch_is_recorded(queue):
    _register(queue)
    results = queue.done([{"seq": 999, "status": cmp.OP_REJECTED, "op": "thing.set",
                           "message": "above the limit"}])
    assert results[0]["text"].startswith("rejected by the monitor: above the limit")


def test_sequence_numbers_increase_and_fit_int32(queue):
    _register(queue)
    seqs = [queue.submit(_req(queue, "thing.set", {"v": 1.}))["seq"] for _ in range(3)]
    assert seqs == sorted(seqs) and len(set(seqs)) == 3
    assert max(seqs) < 2 ** 31
