"""A real (offline) compile through waxx's Core: armed and unarmed.

host=None gives upstream's CommKernelDummy, so nothing can connect; the comm object is
swapped for a recorder that keeps the uploaded library. Proves the subclass compiles
kernels that use the core's own kernel methods, and that the armed path keeps attribute
writeback (the ELF carries the ``typeinfo`` table the runtime writes attributes back
from; ``precompile`` -- writeback off -- does not).
"""
from types import SimpleNamespace

import pytest

from artiq.experiment import delay, kernel

from waxx.util.artiq import core as waxx_core
from waxx.util.artiq.core import Core


class Gate:
    def __init__(self, warm):
        self.warm, self.go, self.log = warm, False, []

    def is_warm(self):
        return self.warm

    def go_passed(self):
        return (not self.warm) or self.go

    def ensure_go(self):
        self.log.append("ensure_go")
        self.go = True

    def run_claim(self):
        self.log.append("claim")


class Recorder:
    def __init__(self):
        self.libraries = []

    def check_system_info(self):
        pass

    def load(self, kernel_library):
        self.libraries.append(bytes(kernel_library))

    def run(self):
        pass

    def serve(self, embedding_map, symbolizer, demangler):
        pass


class Tiny:
    def __init__(self):
        dmgr = SimpleNamespace()
        self.core = Core(dmgr, None, 1e-9, target="cortexa9")
        dmgr.get = lambda name: self.core
        self.core.comm = Recorder()
        self.t_wait = 1e-6
        self.counter = 0

    @kernel
    def run(self):
        self.core.reset()
        delay(self.t_wait)
        self.counter += 1
        self.core.break_realtime()


@pytest.mark.parametrize("warm", [True, False])
def test_kernel_compiles_and_keeps_writeback(monkeypatch, warm):
    gate = Gate(warm)
    monkeypatch.setattr(waxx_core, "queue_gate", gate)
    t = Tiny()
    assert t.core.queue_gate_armed is warm
    t.run()                                   # Core.run: compile, (gate), dummy upload
    assert gate.log == (["ensure_go", "claim"] if warm else [])
    (elf,) = t.core.comm.libraries
    assert elf[:4] == b"\x7fELF"
    assert b"typeinfo" in elf                 # attribute writeback compiled in

    # negative control: the same kernel without writeback has no typeinfo table
    _, no_wb, *_ = t.core.compile(Tiny.run, [t], {},
                                  attribute_writeback=False)
    assert b"typeinfo" not in bytes(no_wb)
