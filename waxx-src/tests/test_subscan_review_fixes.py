"""Regression tests for the 2026-10-10 review of the interleaved sub-scan (waxx side).

1. A point that raises a non-RTIO error puts the machine in its safe state
   (cleanup_abort_kernel) BEFORE the liveOD push in _subscan_end, and the
   point's own exception propagates even with a liveOD Abort pending.
2. A point value is coerced to the existing param's type (a float param given
   integer points ships floats; an integer param refuses fractional points).
3. end_point advances the counter even when its bookkeeping raises.
Uses the host harness of test_scan_subscan_slot (the real _scan, a fake core).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from artiq.language.core import set_time_manager
from artiq.sim.time import Manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_scan_subscan_slot as H  # noqa: E402
from waxx.base.subscan import SubScanRefused  # noqa: E402


@pytest.fixture(autouse=True)
def sim_time():
    set_time_manager(Manager())


class ErrExpt(H.LoopExpt):
    """calibration_kernel raises a non-RTIO error at point ``err_point``."""
    err_point = 0

    def calibration_kernel(self):
        sub = self._subscan
        if sub.counter == self.err_point:
            self.calls.append(("cal_raises",))
            raise ValueError("DAC value out of range")
        H.LoopExpt.calibration_kernel(self)


def _prep(cls, n_data=5, every=2, **kw):
    e = cls(n_data=n_data, **kw)
    sub = e.add_sub_scan("cal", every_n_shots=every, at_start=True, require_push=False)
    sub.xvar("t_pulse", [0.0, 2.0])
    sub.xvar("mode", [0.0, 1.0])
    sub.data.apd = sub.add_data_container(2)
    e.init_xvars(shuffle=False)
    e.data.init()
    H._give_core(e.data, e.core)
    e._subscan.finish(n_data_shots=n_data, main_xvarnames=e.xvarnames, setup_camera=False)
    H._give_core(e._subscan.data, e.core)
    e._compile_subscan_switch()
    e._decide_subscan()
    return e


def test_error_in_a_point_with_an_abort_pending_still_cleans_up_and_keeps_the_error():
    e = _prep(ErrExpt, reset_at_point=0)
    with pytest.raises(ValueError):
        e.scan()
    kinds = [c[0] for c in e.calls]
    assert "cleanup_abort" in kinds


def test_error_in_a_point_cleans_up_before_the_liveod_push():
    e = _prep(ErrExpt)
    e.live_od_client = SimpleNamespace(
        supports=lambda f: True, last_reset_requested=False, poll_reset=lambda: False,
        put_data=lambda specs: (e.calls.append(("PUT_DATA",)) or {"ok": True}),
        abort_run=lambda: None, shot_complete=lambda *a, **k: False)
    with pytest.raises(ValueError):
        e.scan()
    kinds = [c[0] for c in e.calls]
    assert "cleanup_abort" in kinds
    if "PUT_DATA" in kinds:
        assert kinds.index("cleanup_abort") < kinds.index("PUT_DATA")


def test_point_values_take_the_existing_params_type():
    e = H.LoopExpt(n_data=5)
    e.params.amp = 0.1
    e.params.n_reads = 5
    sub = e.add_sub_scan("cal", every_n_shots=2, at_start=True, require_push=False)
    sub.xvar("amp", [0, 1])               # integers for a float param -> floats
    sub.xvar("n_reads", [5.0, 10.0])      # whole floats for an int param -> ints
    assert sub._point_values["amp"].dtype == np.float64
    assert np.issubdtype(sub._point_values["n_reads"].dtype, np.integer)
    e2 = H.LoopExpt(n_data=5)
    e2.params.n_reads = 5
    sub2 = e2.add_sub_scan("cal", every_n_shots=2, at_start=True, require_push=False)
    with pytest.raises(SubScanRefused):
        sub2.xvar("n_reads", [5.0, 7.5])


def test_end_point_advances_the_counter_even_when_its_bookkeeping_raises():
    e = _prep(H.LoopExpt)
    sub = e._subscan
    sub.begin_point()

    def boom(*a, **k):
        raise RuntimeError("push exploded")
    e.push_shot_data = boom
    before = sub.counter
    with pytest.raises(RuntimeError):
        sub.end_point(True, "")
    assert sub.counter == before + 1
