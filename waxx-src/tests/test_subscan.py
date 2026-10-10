"""The interleaved sub-scan (waxx.base.subscan) on the host: its schedule,
point arrays, containers and registration, the params save/restore around a
point, the pushes and the END_RUN payload, the refusals, and the three RPCs
that never raise. No socket, no server: the push transport is a stub."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from waxx.base.expt import Expt
from waxx.base.subscan import SubScan, SubScanRefused
from waxx.config.data_vault import DataVault
from waxx.config.expt_params import ExptParams


class FakeCore:
    def run(self, fn, args, kwargs):
        return fn.artiq_embedded.function(*args, **kwargs)

    def break_realtime(self):
        pass

    def wait_until_mu(self, t):
        pass


def _expt(n_data=5, xvardims=None, sort_idx=(), push=True, client=True, setup_camera=False):
    """A stand-in with the Expt methods the sub-scan uses bound to it."""
    e = SimpleNamespace()
    e.core = FakeCore()
    e.params = ExptParams()
    e.params.t_fixed = 1.0
    e.p = e.params
    e.data = DataVault(expt=e)
    e.xvardims = list(xvardims or [n_data])
    e.xvarnames = ["t_main"]
    e.scan_xvars = [SimpleNamespace(key="t_main", counter=0, values=np.arange(e.xvardims[0]))]
    e.sort_idx = [np.array(s) for s in sort_idx]
    e.sort_N = [len(s) for s in sort_idx]
    e.setup_camera = setup_camera
    e.run_info = SimpleNamespace(save_data=True, run_id=7)
    e.sent = []
    if client:
        e.live_od_client = SimpleNamespace(
            supports=lambda f: push, last_reset_requested=False, poll_reset=lambda: False,
            put_data=lambda specs: (e.sent.append(specs) or {"ok": True}),
            abort_run=lambda: None)
    else:
        e.live_od_client = None
    e._push_failed, e._n_pushed, e._n_push_failed = set(), 0, 0
    e._extra_file_texts = {}
    e._shot_complete_count = 0
    e._last_data_shot = (-1, -1)
    e._subscan = None
    e._subscan_enabled = False
    e._abort_cause = ""
    e.compute_new_derived = lambda: None
    e.param_police = lambda xv: None
    e._check_for_abort_signal = lambda raise_error=True: False
    for name in ("current_shot_index", "final_shot_index", "push_shot_data", "_push",
                 "_note_push_failure", "add_sub_scan", "_decide_subscan", "_subscan_note",
                 "_subscan_due", "_subscan_begin", "_subscan_end", "_shot_conditions"):
        setattr(e, name, getattr(Expt, name).__get__(e))
    e.push_data_enabled = Expt.push_data_enabled.fget(e)
    e.data.init()
    return e


def _declare(e, every=2, at_start=True, alternate=True, n_reads=3, **kw):
    sub = e.add_sub_scan("cal", every_n_shots=every, at_start=at_start, alternate=alternate, **kw)
    sub.xvar("t_pulse", [0.0, 2.5e-6])
    sub.xvar("mode", [0.0, 1.0])
    sub.data.apd = sub.add_data_container(n_reads)
    return sub


def _finish(e, sub):
    sub.finish(n_data_shots=int(np.prod(e.xvardims)), main_xvarnames=e.xvarnames,
               setup_camera=e.setup_camera)
    e._decide_subscan()
    return sub


# ----------------------------------------------------------------- schedule

def test_events_before_shot_0_and_every_n_shots_and_none_after_the_last():
    e = _expt(n_data=5)
    sub = _finish(e, _declare(e, every=2))
    assert sub.event_starts == [0, 2, 4] and sub.n_events == 3 and sub.N == 6
    e2 = _expt(n_data=4)
    sub2 = _finish(e2, _declare(e2, every=2))
    assert sub2.event_starts == [0, 2]          # N % every == 0: no event after the last shot
    e3 = _expt(n_data=3)
    sub3 = _finish(e3, _declare(e3, every=20, at_start=False))
    assert not sub3.enabled and "no schedule event" in sub3.disabled_reason
    assert not e3._subscan_enabled


def test_points_are_tiled_per_event_with_the_order_reversed_on_odd_events():
    e = _expt(n_data=5)
    sub = _finish(e, _declare(e, every=2))
    assert np.allclose(sub.scan_xvars[0].values, [0, 2.5e-6, 2.5e-6, 0, 0, 2.5e-6])
    assert sub.xvarnames == ["t_pulse"] and len(sub.scan_xvars) == 1
    assert np.allclose(sub.point_xvars[1].values, [0, 1, 1, 0, 0, 1])
    # the second zipped key rides as a per-point container, filled in advance
    assert np.allclose(sub.data.mode._run_data, [0, 1, 1, 0, 0, 1])
    assert sub.xvardims == [6] and sub.data.apd._run_data.shape == (6, 3)
    e2 = _expt(n_data=5)
    sub2 = _finish(e2, _declare(e2, every=2, alternate=False))
    assert np.allclose(sub2.point_xvars[1].values, [0, 1, 0, 1, 0, 1])


def test_due_follows_the_data_shot_count():
    e = _expt(n_data=5)
    sub = _finish(e, _declare(e, every=2))
    assert sub.due(0)
    sub.counter = 2
    assert not sub.due(1) and sub.due(2)
    sub.counter = 6
    assert not sub.due(99)


# --------------------------------------------------------------- containers

def test_containers_are_registered_in_the_main_vault_but_not_in_its_kernel_lists():
    e = _expt(n_data=5)
    sub = _finish(e, _declare(e))
    assert set(sub.keys) == {"cal_apd", "cal_mode", "cal_time", "cal_main_index", "cal_status"}
    assert all(k in e.data.keys for k in sub.keys)
    assert vars(e.data)["cal_apd"] is sub.data.apd and sub.data.apd.key == "cal_apd"
    kernel_lists = (e.data._list_1d_f64 + e.data._list_2d_f64 + e.data._list_1d_i32
                    + e.data._list_2d_i32 + e.data._list_1d_i64 + e.data._list_2d_i64)
    assert sub.data.apd not in kernel_lists
    assert sub.data.apd in sub.data._list_1d_f64
    assert sub.data.time._run_data.shape == (6,) and np.all(np.isnan(sub.data.time._run_data))
    assert sub.data.main_index._run_data.shape == (6, 2) and np.all(sub.data.main_index._run_data == -1)
    assert sub.data.status._run_data.shape == (6,) and np.all(sub.data.status._run_data == 0)
    # the live viewer's per-shot conditions never see them
    e.data.i_outer = e.data.add_data_container(1)
    e.data.i_outer.key = "i_outer"
    e.data.keys.append("i_outer")
    assert "cal_status" not in e._shot_conditions()


def test_params_are_restored_bit_for_bit_after_a_point_derived_ones_too():
    e = _expt(n_data=5)
    p = e.params
    p.t_pulse = 1.0
    p.mode = 0.0
    calls = []
    p.compute_derived = lambda: (calls.append(1), setattr(p, "t_pulse_ideal", p.t_pulse - 1e-7))
    p.compute_derived()
    sub = _finish(e, _declare(e))
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(p).items()}
    sub.counter = 1
    sub.begin_point()
    assert p.t_pulse == 2.5e-6 and p.mode == 1.0 and p.t_pulse_ideal == pytest.approx(2.5e-6 - 1e-7)
    sub.data.apd.shot_data[:] = [1., 2., 3.]
    sub.data.apd.update_from_kernel(sub.data.apd.shot_data)
    sub.data.apd._put_shot_data_to_run_data()
    sub.end_point(True)
    after = vars(p)
    assert set(after) == set(before)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(after[k], v), k
        else:
            assert after[k] == v, k
    assert np.allclose(sub.data.apd._run_data[1], [1., 2., 3.])


def test_a_point_records_time_status_and_the_preceding_data_shot_and_pushes_its_slot():
    e = _expt(n_data=5, sort_idx=([4, 2, 0, 1, 3],))
    sub = _finish(e, _declare(e))
    e._last_data_shot = (3, 1)
    sub.begin_point()
    sub.data.apd._run_data[0] = [0.1, 0.2, 0.3]
    entry = sub.end_point(True)
    assert entry["slot"] == 0 and entry["ok"] and entry["main_acq_index"] == 3 and entry["main_final_slot"] == 1
    assert sub.data.status._run_data[0] == 1 and np.isfinite(sub.data.time._run_data[0])
    assert np.array_equal(sub.data.main_index._run_data[0], [3, 1])
    (specs,) = e.sent
    assert sorted(s["key"] for s in specs) == ["cal_apd", "cal_main_index", "cal_mode", "cal_status", "cal_time"]
    # the slot is the point's own counter, never remapped by the main scan's shuffle
    assert all(s["index"] == (0,) for s in specs)
    assert all(vars(e.data)[s["key"]]._pushed for s in specs)
    assert sub.counter == 1 and sub.n_ok == 1


def test_a_failed_point_keeps_its_fill_marks_status_2_and_advances():
    e = _expt(n_data=5)
    sub = _finish(e, _declare(e))
    sub.begin_point()
    entry = sub.end_point(False, "RTIOUnderflow")
    assert entry["what"] == "RTIOUnderflow" and sub.data.status._run_data[0] == 2
    assert np.isnan(sub.data.time._run_data[0]) and np.all(sub.data.apd._run_data[0] == 0)
    assert sub.counter == 1 and sub.n_failed == 1
    (specs,) = e.sent
    assert [s["key"] for s in specs] == ["cal_status"]


def test_a_failed_slot_push_is_pushed_whole_before_end_run_and_the_log_goes_to_the_attrs():
    e = _expt(n_data=5)
    sub = _finish(e, _declare(e))
    e.live_od_client.put_data = lambda specs: {"ok": False, "error": "busy"}
    sub.begin_point()
    sub.end_point(True)
    assert "cal_apd" in e._push_failed
    e.live_od_client.put_data = lambda specs: (e.sent.append(specs) or {"ok": True})
    sub.finish_run()
    whole = [s for specs in e.sent for s in specs if s["index"] is None]
    assert sorted(s["key"] for s in whole) == ["cal_apd", "cal_main_index", "cal_mode", "cal_status", "cal_time"]
    assert all(s["offset"] == 0 and s["array"].shape[0] == 6 for s in whole)
    assert not e._push_failed and sub.pushes_failed == []
    d = json.loads(e._extra_file_texts["cal_scan"])
    assert d["format"] == 1 and d["xvarnames"] == ["t_pulse"] and d["point_keys"] == ["t_pulse", "mode"]
    assert d["keys"] == ["apd", "mode", "time", "main_index", "status"]
    assert d["xvars"]["t_pulse"][:3] == [0.0, 2.5e-6, 2.5e-6] and d["point_xvars"]["mode"][:3] == [0.0, 1.0, 1.0]
    assert d["log"]["summary"] == {"n_points": 6, "n_taken": 1, "n_ok": 1, "n_failed": 0, "enabled": True,
                                   "disabled_reason": "", "pushes_failed_keys": []}
    assert d["log"]["points"][0]["values"] == {"t_pulse": 0.0, "mode": 0.0}


def test_end_run_payload_declares_pushed_containers_final_order():
    e = _expt(n_data=5)
    sub = _finish(e, _declare(e))
    sub.begin_point()
    sub.end_point(True)
    for key in sub.keys:
        dc = vars(e.data)[key]
        pushed = bool(getattr(dc, "_pushed", False)) and key not in e._push_failed
        assert pushed, key


# ---------------------------------------------------------------- refusals

def test_refused_on_a_run_that_captures_frames():
    e = _expt(n_data=5, setup_camera=True)
    sub = _declare(e)
    with pytest.raises(SubScanRefused, match="captures camera frames"):
        _finish(e, sub)


def test_refused_when_a_key_or_a_read_parameter_is_a_main_xvar():
    e = _expt(n_data=5)
    sub = e.add_sub_scan("cal")
    sub.xvar("t_main", [0., 1.])
    sub.data.apd = sub.add_data_container(2)
    with pytest.raises(SubScanRefused, match="main-scan xvars"):
        _finish(e, sub)
    e2 = _expt(n_data=5)
    sub2 = _declare(e2, reads_params=("amp_x", "t_main"))
    with pytest.raises(SubScanRefused, match="reads \\['t_main'\\]"):
        _finish(e2, sub2)
    e3 = _expt(n_data=5)
    sub3 = e3.add_sub_scan("cal")
    sub3.xvar("a", [0., 1.])
    sub3.xvar("b", [0., 1., 2.])
    with pytest.raises(SubScanRefused, match="equal lengths"):
        _finish(e3, sub3)
    with pytest.raises(ValueError, match="one sub-scan per run"):
        e3.add_sub_scan("other")


def test_disabled_loudly_without_a_liveod_that_takes_data(capsys):
    e = _expt(n_data=5, push=False)
    sub = _finish(e, _declare(e))
    assert not sub.enabled and not e._subscan_enabled
    assert "DISABLED" in capsys.readouterr().out
    assert "PUT_DATA" in e._extra_file_texts["cal_scan_disabled"]
    e2 = _expt(n_data=5, client=None)
    sub2 = _finish(e2, _declare(e2, require_push=False))
    assert sub2.enabled and e2._subscan_enabled          # an unshuffled run may carry them
    e3 = _expt(n_data=5, client=None, sort_idx=([1, 0, 2, 3, 4],))
    sub3 = _finish(e3, _declare(e3, require_push=False))
    assert not sub3.enabled and "unshuffled" in sub3.disabled_reason


# -------------------------------------------------------------------- RPCs

def test_the_rpcs_never_raise_and_log_the_host_error(capsys):
    e = _expt(n_data=5)
    sub = _finish(e, _declare(e))
    assert e._subscan_due() is True
    sub.due = lambda n: 1 / 0
    assert e._subscan_due() is False
    assert "due" in sub.log["notes"][0]
    sub.begin_point = lambda: 1 / 0
    assert e._subscan_begin() is False
    sub.end_point = lambda ok, what="": 1 / 0
    assert e._subscan_end(True, "") == 0
    assert len(sub.log["notes"]) == 3
    assert capsys.readouterr().out.count("host error") == 3


def test_the_end_rpc_ends_the_run_on_the_liveod_abort_button():
    from artiq.language.core import TerminationRequested
    e = _expt(n_data=5)
    sub = _finish(e, _declare(e))
    e._check_for_abort_signal = lambda raise_error=True: True
    sub.begin_point()
    with pytest.raises(TerminationRequested):
        e._subscan_end(True, "")
    assert sub.counter == 1 and e._abort_cause == "the liveOD Abort button"
