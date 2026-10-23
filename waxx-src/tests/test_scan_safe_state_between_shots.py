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


# ---- cleanups inside the shot's error handlers (organizer / reviewer B, 2026-10-23) ----
# kexp: an underflow before the first frame leaves the image counters at 0, so
# cleanup_scan_kernel's first step (cleanup_image_count) raises ValueError
# before reset_coils. The handler's cleanup is now wrapped too: the safe state
# (which includes the full coil reset in kexp) runs before anything re-raises.

from artiq.coredevice.exceptions import RTIOOverflow  # noqa: E402
from waxx.control.exceptions import TriggerTimeout  # noqa: E402


def _cleanup_raising_when_no_image(e):
    real = type(e).cleanup_scan_kernel

    def cleanup(self=e):
        if not e.shot_completed:
            raise ValueError("Incorrect number of PWA acquired during the shot.")
        real(e)
    e.cleanup_scan_kernel = cleanup


@pytest.mark.parametrize("exc", [RTIOUnderflow, TriggerTimeout, RTIOOverflow])
def test_a_cleanup_failing_inside_a_shot_handler_reaches_the_safe_state(exc):
    e = _prepared(with_sub=False)
    _cleanup_raising_when_no_image(e)
    real_kernel = type(e).scan_kernel

    def shot(self=e):
        if len(e.calls) == 1:               # the 2nd shot fails before imaging
            raise exc("test")
        real_kernel(e)
    e.scan_kernel = shot
    with pytest.raises(ValueError, match="Incorrect number"):
        e.scan()
    assert _kinds(e) == ["data", "cleanup_abort"]


def test_a_failing_sub_scan_cleanup_reaches_the_safe_state():
    e = _prepared(n_data=3, every=2)
    e.fail_point = 0                          # the first point underflows

    def cal_cleanup():
        e.calls.append(("cleanup_cal",))
        raise ValueError("cleanup of the point failed")
    e.cleanup_calibration_kernel = cal_cleanup
    with pytest.raises(ValueError, match="cleanup of the point failed"):
        e.scan()
    assert _kinds(e)[-2:] == ["cleanup_cal", "cleanup_abort"]


def test_the_feature_flag_is_set():
    """kexp's MOT pre-load checks this flag before it runs (Base._check_mot_preload)."""
    from waxx.base import scanner
    assert scanner.SAFE_STATE_ON_EVERY_CLEANUP_FAILURE is True
    assert hasattr(scanner.Scanner, "_cleanup_or_safe_state")
