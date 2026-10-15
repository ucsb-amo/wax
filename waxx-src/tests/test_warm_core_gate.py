"""waxx.util.artiq.core: the warm-start gate around the first kernel.

queue_gate is faked (its real version belongs to the queue); compile is stubbed and the
comm object records what it is asked to do. No connection is ever made: the host is
TEST-NET and ``create_connection`` is patched to raise everywhere.
"""

import pytest

from artiq.coredevice import comm_kernel
from artiq.coredevice.comm_kernel import CommKernel, CommKernelDummy
from artiq.coredevice.core import Core as ArtiqCore

from waxx.util.artiq import core as waxx_core
from waxx.util.artiq.core import Core, GateNotPassed, GatedCommKernel

TEST_HOST = "192.0.2.1"          # TEST-NET-1: never routed


class FakeGate:
    """Stands in for waxx.util.device_state.queue_gate (BUILD_SPEC §3 surface)."""
    EXIT_WARM_ABORTED = 78

    def __init__(self, log, warm=True, abort=False):
        self.log, self.warm, self.abort = log, warm, abort
        self.go = False
        self.claim = None
        self.claimed = False

    def is_warm(self):
        return self.warm

    def go_passed(self):
        return (not self.warm) or self.go

    def ensure_go(self):
        self.log.append("ensure_go")
        if self.warm and not self.go:
            if self.abort:
                raise SystemExit(self.EXIT_WARM_ABORTED)
            self.go = True

    def register_claim(self, fn):
        self.claim = fn

    def run_claim(self):
        if self.claim is not None and not self.claimed:
            self.claimed = True
            self.claim()


class RecordingComm:
    def __init__(self, log, result=None):
        self.log, self.result = log, result
        self.set_result = None

    def check_system_info(self):
        self.log.append("check_system_info")

    def load(self, kernel_library):
        self.log.append(("load", kernel_library))

    def upload_subkernel(self, kernel_library, sid, destination):
        self.log.append("upload_subkernel")

    def run(self):
        self.log.append("run")

    def serve(self, embedding_map, symbolizer, demangler):
        self.log.append("serve")
        if self.set_result is not None and self.result is not None:
            self.set_result(self.result)       # what the kernel's return RPC does

    def close(self):
        pass


class NoSubkernels:
    def subkernels(self):
        return {}

    def subkernel_messages_unpaired(self):
        return []


@pytest.fixture(autouse=True)
def no_connection(monkeypatch):
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise ConnectionRefusedError(f"test: connection to {args} refused")

    monkeypatch.setattr(comm_kernel, "create_connection", refuse)
    return attempts


def _gate(monkeypatch, log, **kw):
    gate = FakeGate(log, **kw)
    monkeypatch.setattr(waxx_core, "queue_gate", gate)
    return gate


def _wire(core, log, result=None):
    """Recording comm + stubbed compile (a fresh library name per compile)."""
    comm = RecordingComm(log, result)
    core.comm = comm
    n = [0]

    def compile(function, args, kwargs, set_result=None, attribute_writeback=True, **rest):
        n[0] += 1
        log.append(("compile", attribute_writeback))
        comm.set_result = set_result
        return NoSubkernels(), f"elf{n[0]}", None, None, {}

    core.compile = compile
    return comm


def _kernel(*args):
    pass


# ---- unarmed: upstream ---------------------------------------------------------------

def test_unarmed_core_is_upstream(monkeypatch):
    log = []
    gate = _gate(monkeypatch, log, warm=False)
    core = Core(None, TEST_HOST, 1e-9)
    assert core.queue_gate_armed is False
    assert type(core.comm) is CommKernel

    _wire(core, log, result=7)
    assert core.run(_kernel, (), {}) == 7

    ref_log = []
    ref = ArtiqCore(None, TEST_HOST, 1e-9)
    _wire(ref, ref_log, result=7)
    assert ref.run(_kernel, (), {}) == 7
    assert log == ref_log == [("compile", True), "check_system_info", ("load", "elf1"),
                              "run", "serve"]
    assert gate.claimed is False


# ---- armed ---------------------------------------------------------------------------

def test_armed_order_compile_gate_claim_then_core_device(monkeypatch):
    log = []
    gate = _gate(monkeypatch, log)
    gate.register_claim(lambda: log.append("claim"))
    core = Core(None, TEST_HOST, 1e-9)
    assert core.queue_gate_armed is True
    assert type(core.comm) is GatedCommKernel and core.comm.core is core
    assert (core.comm.host, core.comm.port) == (TEST_HOST, 1381)

    _wire(core, log, result=42)
    assert core.run(_kernel, (), {}) == 42
    assert log == [("compile", True), "ensure_go", "claim", "check_system_info",
                   ("load", "elf1"), "run", "serve"]


def test_gate_is_one_shot(monkeypatch):
    log = []
    gate = _gate(monkeypatch, log)
    gate.register_claim(lambda: log.append("claim"))
    core = Core(None, TEST_HOST, 1e-9)
    _wire(core, log)
    core.run(_kernel, (), {})
    log.clear()
    core.run(_kernel, (), {})
    assert log == [("compile", True), ("load", "elf2"), "run", "serve"]
    assert log.count("claim") == 0


def test_claim_registered_after_go_runs_before_the_next_kernel(monkeypatch):
    """A kernel that ran before finish_prepare took the gate with no claim registered;
    the claim registered afterwards must still run before the next kernel."""
    log = []
    gate = _gate(monkeypatch, log)
    core = Core(None, TEST_HOST, 1e-9)
    _wire(core, log)
    core.run(_kernel, (), {})
    gate.register_claim(lambda: log.append("claim"))
    log.clear()
    core.run(_kernel, (), {})
    assert log == ["claim", ("compile", True), ("load", "elf2"), "run", "serve"]


def test_abort_before_go_exits_with_nothing_sent(monkeypatch):
    log = []
    gate = _gate(monkeypatch, log, abort=True)
    gate.register_claim(lambda: log.append("claim"))
    core = Core(None, TEST_HOST, 1e-9)
    _wire(core, log)
    with pytest.raises(SystemExit) as exc:
        core.run(_kernel, (), {})
    assert exc.value.code == 78
    assert log == [("compile", True), "ensure_go"]


def test_failed_claim_sends_nothing(monkeypatch):
    log = []
    gate = _gate(monkeypatch, log)

    def claim():
        raise RuntimeError("liveOD refused the start")

    gate.register_claim(claim)
    core = Core(None, TEST_HOST, 1e-9)
    _wire(core, log)
    with pytest.raises(RuntimeError, match="refused the start"):
        core.run(_kernel, (), {})
    assert log == [("compile", True), "ensure_go"]


# ---- the connection guard ------------------------------------------------------------

def test_armed_comm_refuses_to_connect_before_go(monkeypatch, no_connection):
    log = []
    gate = _gate(monkeypatch, log)
    core = Core(None, TEST_HOST, 1e-9)
    with pytest.raises(GateNotPassed):
        core.comm.open()
    for exchange in (core.comm.check_system_info, lambda: core.comm.load(b"x"),
                     core.comm.run):
        with pytest.raises(GateNotPassed):
            exchange()
    # _run_compiled (what a precompiled kernel calls) is refused too, and never waits
    with pytest.raises(GateNotPassed):
        core._run_compiled(b"x", None, None, None)
    assert no_connection == []
    assert "ensure_go" not in log
    core.close()                        # nothing open: harmless


def test_armed_comm_connects_normally_after_go(monkeypatch, no_connection):
    log = []
    gate = _gate(monkeypatch, log)
    core = Core(None, TEST_HOST, 1e-9)
    gate.ensure_go()
    with pytest.raises(ConnectionRefusedError):       # reached upstream's open()
        core.comm.open()
    assert no_connection == [(TEST_HOST, 1381)]


def test_second_core_in_a_warm_process(monkeypatch, no_connection):
    """TTLPulser builds its own Core: before GO it must not connect either; after GO it
    behaves as upstream, and it never waits or claims a second time."""
    log = []
    gate = _gate(monkeypatch, log)
    gate.register_claim(lambda: log.append("claim"))
    first = Core(None, TEST_HOST, 1e-9)
    second = Core(None, TEST_HOST, 1e-9)
    with pytest.raises(GateNotPassed):
        second.comm.open()

    _wire(first, log)
    first.run(_kernel, (), {})
    assert log.count("claim") == 1
    log.clear()
    _wire(second, log)
    second.run(_kernel, (), {})
    assert log == [("compile", True), "check_system_info", ("load", "elf1"), "run", "serve"]
    assert no_connection == []


def test_dummy_host_keeps_the_dummy_comm(monkeypatch):
    _gate(monkeypatch, [])
    core = Core(None, None, 1e-9)
    assert core.queue_gate_armed is True
    assert type(core.comm) is CommKernelDummy


_CHILD = r"""
import os, sys
from artiq.coredevice import comm_kernel
def refuse(*a, **k):
    raise AssertionError("connected")
comm_kernel.create_connection = refuse
from waxx.util.artiq.core import Core, GateNotPassed
core = Core(None, "192.0.2.1", 1e-9)
second = Core(None, "192.0.2.1", 1e-9)
outcome = []
for c in (core, second):
    try:
        c.comm.open()
        outcome.append("opened")
    except Exception as exc:        # GateNotPassed armed; the patched refusal unarmed
        outcome.append(type(exc).__name__)
print(core.queue_gate_armed, second.queue_gate_armed, type(core.comm).__name__,
      "WAXX_QUEUE_WARM" in os.environ, *outcome)
"""


@pytest.mark.parametrize("warm", [False, True])
def test_real_queue_gate_in_a_fresh_process(warm):
    """With the real queue_gate (no fake): the env var decides, is popped at first read,
    and a second Core built after the pop is armed too. Only the parts of queue_gate that
    need no server are reached (is_warm, go_passed)."""
    import os
    import subprocess
    import sys
    env = dict(os.environ)
    env.pop("WAXX_QUEUE_WARM", None)
    if warm:
        env["WAXX_QUEUE_WARM"] = "test-token"
    out = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    if warm:
        assert out.stdout.split() == ["True", "True", "GatedCommKernel", "False",
                                      "GateNotPassed", "GateNotPassed"]
    else:
        # unarmed: open() is upstream's and reaches the (patched) create_connection
        assert out.stdout.split() == ["False", "False", "CommKernel", "False",
                                      "AssertionError", "AssertionError"]
