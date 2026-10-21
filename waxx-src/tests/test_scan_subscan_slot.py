"""The scan loop's interleaved-sub-scan slot, run on the host.

A Scanner subclass with a fake core (its ``run`` calls the kernel's function)
and ARTIQ's simulation time manager (``delay`` works) runs the real
``Scanner.scan`` / ``_scan`` / ``_run_subscan_point`` with the real
``SubScan`` and the real Expt RPCs bound in. Hooks only record their calls.
Nothing touches a device or the network."""
from types import SimpleNamespace

import numpy as np
import pytest
from artiq.experiment import RTIOUnderflow
from artiq.language.core import TerminationRequested, set_time_manager
from artiq.sim.time import Manager

from waxa.base.dealer import Dealer
from waxx.base.expt import Expt
from waxx.base.scanner import Scanner
from waxx.config.data_vault import DataVault


class FakeCore:
    def run(self, fn, args, kwargs):
        f = fn.artiq_embedded.function
        if not callable(f):                 # kernel_from_string keeps the source there
            f = fn.__wrapped__
        return f(*args, **kwargs)

    def break_realtime(self):
        pass

    def wait_until_mu(self, t):
        pass

    def reset(self):
        pass


class LoopExpt(Scanner, Dealer):
    """Scanner (+ Dealer's xvar plumbing, as Expt has it) + the Expt pieces
    the slot needs; Scribe's abort hooks stubbed."""

    push_shot_data = Expt.push_shot_data
    _push = Expt._push
    _note_push_failure = Expt._note_push_failure
    final_shot_index = Expt.final_shot_index
    current_shot_index = Expt.current_shot_index
    push_data_enabled = Expt.push_data_enabled
    add_sub_scan = Expt.add_sub_scan
    _decide_subscan = Expt._decide_subscan
    _subscan_note = Expt._subscan_note
    _subscan_due = Expt._subscan_due
    _subscan_begin = Expt._subscan_begin
    _subscan_end = Expt._subscan_end
    _shot_conditions = Expt._shot_conditions
    _notify_shot_complete = Expt._notify_shot_complete
    _progress_line = Expt._progress_line
    _progress_stride = Expt._progress_stride

    def __init__(self, n_data=5, save_on_underflow=False, reset_at_point=None):
        Scanner.__init__(self)
        self.core = FakeCore()
        self.setup_camera = False
        self.p = self.params
        self.params.t_fixed = 1.0
        self.run_info.save_data = False
        self.run_info.save_on_underflow = save_on_underflow
        self.run_info.run_id = 1
        self.data = DataVault(expt=self)
        self.sort_idx, self.sort_N = [], []
        self.live_od_client = None
        self._push_failed, self._n_pushed, self._n_push_failed = set(), 0, 0
        self._extra_file_texts = {}
        self._shot_complete_count = 0
        self._N_shots_total = n_data
        self._t_first_shot_done = None
        self._last_data_shot = (-1, -1)
        self._subscan = None
        self._abort_cause = ""
        self._shot_abort = ""
        self.calls = []
        self.aborts = []
        self.reset_at_point = reset_at_point
        self.fail_point = None
        self.xvar("t_main", np.arange(float(n_data)))

    # ---- what Scribe / Expt / kexp.Base would provide
    def _check_for_abort_signal(self, raise_error=True):
        sub = self._subscan
        if self.reset_at_point is not None and sub is not None and sub.counter == self.reset_at_point + 1:
            return True
        return False

    def _abort_shot(self, what="RTIOUnderflow"):
        self.aborts.append(what)
        return bool(self.run_info.save_on_underflow)

    def _send_abort_to_server(self, what=""):
        self.calls.append(("abort_sent", what))

    def _report_abort_state(self, *a, **k):
        pass

    def compute_new_derived(self):
        pass

    def scan_kernel(self):
        self.calls.append(("data", float(self.params.t_main), getattr(self.params, "t_pulse", 0.0)))

    def cleanup_scan_kernel(self):
        self.data.put_shot_data()
        self._notify_shot_complete()

    def init_calibration_kernel(self):
        self.calls.append(("init_cal",))

    def calibration_kernel(self):
        sub = self._subscan
        if self.fail_point is not None and sub.counter == self.fail_point:
            raise RTIOUnderflow("test")
        self.calls.append(("cal", sub.counter, self.params.t_pulse, self.params.mode))
        sub.data.apd.shot_data[:] = sub.counter
        sub.put_shot_data()

    def cleanup_calibration_kernel(self):
        self.calls.append(("cleanup_cal",))

    def cleanup_abort_kernel(self):
        self.calls.append(("cleanup_abort",))


@pytest.fixture(autouse=True)
def sim_time():
    set_time_manager(Manager())


def _give_core(vault, core):
    """On the host a @kernel method looks up ``self.core`` on the object it is
    called on; the compiler would inline it. Give the vault and every
    container (sentinels too) the fake core."""
    vault.core = core
    for lst in (vault._list_1d_f64, vault._list_2d_f64, vault._list_1d_i32,
                vault._list_2d_i32, vault._list_1d_i64, vault._list_2d_i64):
        for dc in lst:
            dc.core = core


def _prepared(n_data=5, with_sub=True, every=2, **kw):
    e = LoopExpt(n_data=n_data, **kw)
    if with_sub:
        sub = e.add_sub_scan("cal", every_n_shots=every, at_start=True, require_push=False)
        sub.xvar("t_pulse", [0.0, 2.0])
        sub.xvar("mode", [0.0, 1.0])
        sub.data.apd = sub.add_data_container(2)
    e.init_xvars(shuffle=False)
    e.data.init()
    _give_core(e.data, e.core)
    if with_sub:
        e._subscan.finish(n_data_shots=n_data, main_xvarnames=e.xvarnames, setup_camera=False)
        _give_core(e._subscan.data, e.core)
    e._decide_subscan()
    return e


def _kinds(e):
    return [c[0] for c in e.calls]


def test_without_a_sub_scan_the_loop_is_unchanged():
    e = _prepared(with_sub=False)
    e.scan()
    assert _kinds(e) == ["data"] * 5 and not e._subscan_enabled


def test_points_come_before_shot_0_and_before_every_nth_data_shot():
    e = _prepared(n_data=5, every=2)
    e.scan()
    assert _kinds(e) == (["init_cal", "cal", "cleanup_cal"] * 2 + ["data"] * 2) * 2 \
        + ["init_cal", "cal", "cleanup_cal"] * 2 + ["data"]
    cal = [c for c in e.calls if c[0] == "cal"]
    # the point's parameters reached the "kernel" (the host params the scan loop writes),
    # alternating pair order, and every data shot ran on the main value again
    assert [(c[2], c[3]) for c in cal] == [(0., 0.), (2., 1.), (2., 1.), (0., 0.), (0., 0.), (2., 1.)]
    assert all(c[2] == 0.0 for c in e.calls if c[0] == "data")     # t_pulse plugged value
    sub = e._subscan
    assert sub.counter == 6 and sub.n_ok == 6
    assert np.array_equal(sub.data.apd._run_data[:, 0], np.arange(6))
    assert np.array_equal(sub.data.status._run_data, np.ones(6))
    # the preceding data shot of each point: -1 before shot 0, then the acquisition index
    assert sub.data.main_index._run_data[:, 0].tolist() == [-1, -1, 1, 1, 3, 3]
    assert sub.data.main_index._run_data[:, 1].tolist() == [-1, -1, 1, 1, 3, 3]
    assert e._shot_complete_count == 5


def test_n_divisible_by_every_takes_no_point_after_the_last_shot():
    e = _prepared(n_data=4, every=2)
    e.scan()
    assert _kinds(e).count("cal") == 4 and _kinds(e)[-1] == "data"


def test_an_underflow_in_a_point_ends_the_run_with_the_shots_taken_under_save_on_underflow():
    e = _prepared(n_data=5, every=2, save_on_underflow=True)
    e.fail_point = 2          # the first point of the second event, after data shots 0 and 1
    e.scan()
    kinds = _kinds(e)
    assert kinds.count("data") == 2 and kinds.count("cal") == 2
    assert e.aborts == ["RTIOUnderflow"] and ("abort_sent", "RTIOUnderflow") in e.calls
    sub = e._subscan
    assert sub.data.status._run_data.tolist() == [1, 1, 2, 0, 0, 0]
    assert sub.log["points"][2]["what"] == "RTIOUnderflow"


def test_an_underflow_in_a_point_re_raises_without_save_on_underflow():
    e = _prepared(n_data=5, every=2, save_on_underflow=False)
    e.fail_point = 0
    with pytest.raises(RTIOUnderflow):
        e.scan()
    assert _kinds(e).count("data") == 0 and ("cleanup_cal",) in e.calls
    assert e._subscan.data.status._run_data[0] == 2


def test_the_liveod_abort_button_after_a_point_ends_the_run_like_after_a_data_shot():
    e = _prepared(n_data=5, every=2, reset_at_point=0)
    with pytest.raises(TerminationRequested):
        e.scan()
    assert _kinds(e).count("data") == 0 and _kinds(e).count("cal") == 1
    assert e._abort_cause == "the liveOD Abort button"


def test_a_host_error_in_a_point_does_not_end_the_run():
    e = _prepared(n_data=5, every=2)
    sub = e._subscan
    real = sub.begin_point
    sub.begin_point = lambda: (_ for _ in ()).throw(RuntimeError("estimator broke"))
    e.scan()
    assert _kinds(e).count("data") == 5 and _kinds(e).count("cal") == 0
    assert sub.counter == 6 and sub.n_failed == 6          # every point was skipped and logged
    assert any("begin_point" in n for n in sub.log["notes"])
    sub.begin_point = real
