"""The scan loop puts the machine in its safe state (cleanup_abort_kernel)
before re-raising an exception out of cleanup_scan_kernel or out of the steps
between two shots (abort check, parameter update, init_scan_kernel) --
2026-10-23. Before, those ended the run with no cleanup at all (kexp: an
exception in the cleanup before reset_coils left the outer coil at 182 A;
a MOT pre-loaded at the end of a shot would stay on).

Runs the real Scanner.scan / _scan on the host with the fake core of
test_scan_subscan_slot. Nothing touches a device or the network."""
import pytest
from artiq.experiment import RTIOUnderflow
from artiq.language.core import TerminationRequested

from test_scan_subscan_slot import _prepared, _kinds, sim_time  # noqa: F401 (fixture)


def test_a_normal_run_runs_no_safe_state_kernel():
    e = _prepared(with_sub=False)
    e.scan()
    assert _kinds(e) == ["data"] * 5


def test_an_exception_out_of_the_cleanup_runs_the_safe_state_first():
    e = _prepared(with_sub=False)
    real = type(e).cleanup_scan_kernel

    def cleanup(self=e):
        if len([c for c in e.calls if c[0] == "data"]) == 2:
            raise ValueError("Incorrect number of PWA acquired during the shot.")
        real(e)
    e.cleanup_scan_kernel = cleanup
    with pytest.raises(ValueError, match="Incorrect number"):
        e.scan()
    assert _kinds(e) == ["data", "data", "cleanup_abort"]


def test_the_liveod_abort_between_shots_runs_the_safe_state_first():
    e = _prepared(with_sub=False)

    def abort_check(raise_error=True):
        if e._shot_complete_count == 3:
            raise TerminationRequested
        return False
    e._check_for_abort_signal = abort_check
    with pytest.raises(TerminationRequested):
        e.scan()
    assert _kinds(e) == ["data"] * 3 + ["cleanup_abort"]


def test_an_exception_in_init_scan_kernel_runs_the_safe_state_first():
    e = _prepared(with_sub=False)

    def init():
        if e._shot_complete_count == 1:
            raise RuntimeError("an RPC in init_scan_kernel failed")
    e.init_scan_kernel = init
    with pytest.raises(RuntimeError, match="init_scan_kernel"):
        e.scan()
    assert _kinds(e) == ["data", "cleanup_abort"]


def test_an_underflow_in_a_shot_is_cleaned_once_as_before():
    """The shot's own handler runs cleanup_scan_kernel; the new wrapper
    around the normal-path cleanup does not add a second cleanup."""
    e = _prepared(with_sub=False)
    real_kernel = type(e).scan_kernel

    def shot(self=e):
        real_kernel(e)
        if len(e.calls) == 2:
            raise RTIOUnderflow("test")
    e.scan_kernel = shot
    with pytest.raises(RTIOUnderflow):
        e.scan()
    assert _kinds(e) == ["data", "data"] and e.aborts == ["RTIOUnderflow"]
