"""The startup profiler times the warm-start gate and reports "GO -> kernel running".

Runs in a child process: the profiler's hooks replace ARTIQ methods process-wide. The
child is warm-launched (WAXX_QUEUE_WARM set; the real queue_gate's is_warm reads it) with
ensure_go / go_passed replaced by a local GO after a short sleep -- never the queue's
server conversation. It compiles nothing (compile stubbed on the instance) and talks to
no device: the comm object is a CommKernel whose socket is a sink, so upstream's
CommKernel.run -- which the profiler marks as "kernel started" -- still executes.
"""
import os
import subprocess
import sys

_CHILD = r"""
import io, struct, sys, time
from types import SimpleNamespace
from artiq.coredevice import comm_kernel
def refuse(*a, **k):
    raise AssertionError("connected")
comm_kernel.create_connection = refuse

from waxx.util.device_state import queue_gate
# GO is local here: the real ensure_go would talk to the queue (the monitor server)
state = {"go": False}
def fake_ensure_go():
    if queue_gate.is_warm() and not state["go"]:
        time.sleep(0.05)
        state["go"] = True
queue_gate.ensure_go = fake_ensure_go
queue_gate.go_passed = lambda: (not queue_gate.is_warm()) or state["go"]

from waxx.util.profiling import startup_timer as st
ok, _ = st._install_hooks()
assert ok["Core._wait_for_go"] and ok["queue_gate.ensure_go"], ok

from artiq.coredevice.comm_kernel import CommKernel
class SinkComm(CommKernel):
    def open(self):
        if not queue_gate.go_passed():
            raise AssertionError("opened before GO")
        self.pack_header = struct.Struct("<lB").pack
        self.socket = SimpleNamespace(sendall=lambda data: None)
    def check_system_info(self): pass
    def load(self, kernel_library): pass
    def serve(self, embedding_map, symbolizer, demangler): pass

class NoSub:
    def subkernels(self): return {}
    def subkernel_messages_unpaired(self): return []

from waxx.util.artiq.core import Core
core = Core(None, "192.0.2.1", 1e-9)
assert core.queue_gate_armed
core.comm = SinkComm("192.0.2.1"); core.comm.core = core
core.compile = lambda f, a, k, set_result=None, **kw: (NoSub(), b"elf", None, None, {})
core.run(lambda: None, (), {})

buf = io.StringIO(); sys.stderr = buf
st._report(None, [])
sys.stderr = sys.__stderr__
print(buf.getvalue())
"""


def test_report_shows_the_wait_and_go_to_kernel():
    env = dict(os.environ)
    env["WAXX_QUEUE_WARM"] = "test-token"
    out = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    report = out.stdout
    assert "* GO received" in report
    assert "wait for GO + claim" in report
    assert "wait for GO (queue)" in report
    assert ">> submission -> kernel running" in report
    assert ">> GO -> kernel running" in report


def test_cold_report_has_no_go_line():
    env = dict(os.environ)
    env.pop("WAXX_QUEUE_WARM", None)
    child = _CHILD.replace("assert core.queue_gate_armed", "assert not core.queue_gate_armed")
    child = child.replace('raise AssertionError("opened before GO")', "pass")
    out = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert ">> submission -> kernel running" in out.stdout
    assert "GO" not in out.stdout
