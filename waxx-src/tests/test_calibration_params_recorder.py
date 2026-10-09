"""The params override recorder (waxx.config.expt_params.ExptParams.__setattr__)
and Expt.start/_stop_param_override_recording, bound onto stand-ins.

Also compiles a small kernel over a recording params object with ARTIQ's
compiler, offline: a Core with no host has a dummy comm, so nothing connects."""
from types import SimpleNamespace

import numpy as np
import pytest

from waxx.base.expt import Expt
from waxx.config.expt_params import ExptParams


class P(ExptParams):
    def __init__(self):
        super().__init__()
        self.t_x = 1.0
        self.amp = 0.5

    def compute_double(self):
        self.t_double = 2 * self.t_x


def expt_with(params):
    e = SimpleNamespace(params=params, _param_overrides=None)
    return e


def start(e):
    Expt.start_param_override_recording(e)


def stop(e):
    Expt._stop_param_override_recording(e)


def test_records_only_what_prepare_assigns():
    p = P()
    p.compute_derived()                    # before recording: as before
    assert p.t_double == 2.0
    e = expt_with(p)
    start(e)
    p.t_x = 3.0                            # an override
    p.N_repeats = 5                        # an override (a waxa base param)
    vars(p)["t_scan"] = 0.                 # how xvar / the scan write: not recorded
    p._private = 1                         # underscore names: not recorded
    p.compute_derived()                    # derived assignments: not recorded
    assert p.t_double == 6.0               # ...but still computed
    stop(e)
    assert e._param_overrides == frozenset({"t_x", "N_repeats"})
    p.amp = 0.7                            # after finish_prepare: not recorded
    assert e._param_overrides == frozenset({"t_x", "N_repeats"})
    assert p.amp == 0.7


def test_compute_derived_restores_recording_even_when_it_raises():
    class Bad(P):
        def compute_bad(self):
            raise ValueError("boom")
    p = Bad()
    e = expt_with(p)
    start(e)
    with pytest.raises(ValueError):
        p.compute_derived()
    p.amp = 0.9
    stop(e)
    assert e._param_overrides == frozenset({"amp"})


def test_params_without_the_recorder_say_unknown():
    from waxa.config.expt_params import ExptParams as Plain
    p = Plain()
    e = expt_with(p)
    start(e)
    p.t_x = 1.0
    stop(e)
    assert e._param_overrides is None                 # unknown, not "none assigned"
    assert "_record_assignments" not in vars(p)


def test_no_new_public_or_kernel_visible_state():
    """Only underscore attributes are added; their types get no kernel writer
    (generate_assignment_kernels keys on int / float / ndarray type names)."""
    p = P()
    before = set(vars(p))
    e = expt_with(p)
    start(e)
    p.t_x = 2.0
    stop(e)
    added = set(vars(p)) - before
    assert added == {"_assigned_keys", "_record_assignments"}
    for k in added:
        t = str(type(vars(p)[k]))
        assert not any(s in t for s in ("int", "float", "ndarray")), t
    # compute_derived on a params object that never recorded adds nothing
    q = P()
    keys = set(vars(q))
    q.compute_derived()
    assert set(vars(q)) - keys == {"t_double"}


def test_a_recording_params_object_compiles_in_a_kernel():
    pytest.importorskip("artiq.coredevice.core")
    from artiq.coredevice.core import Core
    from artiq.experiment import delay, kernel
    from artiq.language.core import kernel_from_string

    class K:
        def __init__(self):
            dmgr = SimpleNamespace()
            self.core = Core(dmgr, None, 1e-9)          # no host: CommKernelDummy
            dmgr.get = lambda name: self.core
            self.params = P()
            self.p = self.params
            start(self)
            self.params.t_x = 2.0
            self.params.N_shots = np.int32(3)
            stop(self)
            self.writer = kernel_from_string(["self", "value"], "self.params.t_x = value")

        @kernel
        def run(self):
            self.writer(self, 4.0)
            delay(self.p.t_x * 1.e-6)
            self.params.amp = self.params.amp + 0.1
            for _ in range(self.p.N_shots):
                delay(1.e-6)

    k = K()
    k._param_overrides = None
    k.core.compile(K.run, [k], {}, attribute_writeback=True, print_as_rpc=False)

    # negative control: the compile above did type-check the params access
    from artiq.coredevice.core import CompileError

    class Broken(K):
        @kernel
        def run(self):
            delay(self.params.no_such_param_xyz)

    b = Broken()
    with pytest.raises(CompileError, match="no_such_param_xyz"):
        b.core.compile(Broken.run, [b], {}, attribute_writeback=True, print_as_rpc=False)
