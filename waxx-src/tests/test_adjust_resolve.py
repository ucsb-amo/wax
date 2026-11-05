"""adjust(): settled at finish_prepare, never moves a param by itself.

Host side only: a stand-in experiment carrying the real Scanner/Expt methods,
fake params, and the liveOD server's adjust handlers called on a stand-in. No
liveOD, no Qt, no socket, no hardware.
"""
import threading
import types

import numpy as np
import pytest

from waxx.base.scanner import AdjustSpec, Scanner
from waxx.base.expt import Expt
from waxx.util.live_od.adjust_rules import adjust_range_error


class FakeParams:
    def __init__(self):
        self.t_tof = 20.e-6           # s
        self.i_mot = 60.
        self.n_shots = 3
        self.t_a = 1.e-3
        self.t_b = 2.e-3
        self.zero = 0.
        self.compute_derived()

    def compute_derived(self):
        self.t_total = self.t_a + self.t_b


class Stub:
    """The adjust machinery of an experiment, nothing else."""
    adjust = Scanner.adjust
    _resolve_adjust_specs = Scanner._resolve_adjust_specs
    _adjust_key_is_derived = Scanner._adjust_key_is_derived
    _detect_adjust_unit = Scanner._detect_adjust_unit
    compute_new_derived = Scanner.compute_new_derived
    update_params_from_xvars = Scanner.update_params_from_xvars
    apply_pending_adjust_values = Expt.apply_pending_adjust_values

    def __init__(self):
        self.params = FakeParams()
        self.p = self.params
        self.xvarnames = []
        self.scan_xvars = []

    def finish(self, save_data=False):
        self._resolve_adjust_specs(save_data=save_data)
        return {s.key: s for s in self._adjust_specs}

    def shot(self, panel_values=None):
        """One shot's host update, then the SHOT_COMPLETE reply's values."""
        self.update_params_from_xvars()
        used = dict(vars(self.params))
        if panel_values is not None:
            self._pending_adjust_values = dict(panel_values)
        return used


# ------------------------------------------------------------ start value

def test_an_override_after_the_adjust_line_is_what_every_shot_uses():
    """The original bug: shot 0 ran the override, shot 1 on the stale value."""
    e = Stub()
    e.adjust('t_tof', min_val=10.e-6, max_val=1.e-3)
    e.p.t_tof = 500.e-6                                  # later in prepare()
    specs = e.finish()
    assert specs['t_tof'].current_val == 500.e-6
    server_values = {k: s.current_val for k, s in specs.items()}   # INIT_RUN
    assert e.shot(server_values)['t_tof'] == 500.e-6
    assert e.shot(server_values)['t_tof'] == 500.e-6
    assert e.shot(server_values)['t_tof'] == 500.e-6


def test_a_param_outside_the_limits_widens_them_and_is_not_moved(capsys):
    e = Stub()
    e.p.t_tof = 30.e-3
    e.p.i_mot = 5.
    e.adjust('t_tof', min_val=20.e-6, max_val=20.e-3)
    e.adjust('i_mot', min_val=10., max_val=80.)
    specs = e.finish()
    assert (specs['t_tof'].min_val, specs['t_tof'].max_val) == (20.e-6, 30.e-3)
    assert (specs['i_mot'].min_val, specs['i_mot'].max_val) == (5., 80.)
    assert e.p.t_tof == 30.e-3 and e.p.i_mot == 5.
    out = capsys.readouterr().out
    assert "max raised to 0.03" in out and "min lowered to 5.0" in out
    vals = {k: s.current_val for k, s in specs.items()}
    assert e.shot(vals)['t_tof'] == 30.e-3 and e.shot(vals)['t_tof'] == 30.e-3


def test_default_val_loses_to_an_existing_param(capsys):
    e = Stub()
    e.adjust('i_mot', min_val=0., max_val=80., default_val=40.)
    assert e.finish()['i_mot'].current_val == 60.
    assert e.p.i_mot == 60.
    assert "default_val=40.0 ignored" in capsys.readouterr().out


def test_default_val_creates_a_missing_param():
    e = Stub()
    e.adjust('t_new', min_val=0., max_val=1., default_val=0.25)
    assert e.finish()['t_new'].current_val == 0.25 and e.p.t_new == 0.25
    e = Stub()
    e.adjust('n_new', min_val=1, max_val=9)
    assert e.finish()['n_new'].current_val == 1 and type(e.p.n_new) is int


def test_a_missing_param_with_no_start_value_raises():
    e = Stub()
    e.adjust('t_new', max_val=1., dtype=float)
    with pytest.raises(ValueError, match="does not exist"):
        e.finish()


def test_the_default_range_is_fifty_percent_and_zero_needs_limits():
    e = Stub()
    e.adjust('i_mot')
    e.adjust('n_shots')
    specs = e.finish()
    assert (specs['i_mot'].min_val, specs['i_mot'].max_val) == (30., 90.)
    assert (specs['n_shots'].min_val, specs['n_shots'].max_val) == (1., 5.)
    e = Stub()
    e.adjust('zero')
    with pytest.raises(ValueError, match="starts at 0"):
        e.finish()
    e = Stub()
    e.adjust('zero', min_val=-1., max_val=1.)
    assert e.finish()['zero'].current_val == 0.


def test_duplicate_adjust_raises_at_once():
    e = Stub()
    e.adjust('i_mot', min_val=0., max_val=80.)
    with pytest.raises(ValueError, match="already registered"):
        e.adjust('i_mot', min_val=0., max_val=80.)


# ------------------------------------------------------------ xvars

@pytest.mark.parametrize("adjust_first", [True, False])
def test_an_xvar_wins_whichever_line_comes_first(adjust_first, capsys):
    e = Stub()
    if adjust_first:
        e.adjust('i_mot', min_val=0., max_val=80.)
        e.xvarnames.append('i_mot')
    else:
        e.xvarnames.append('i_mot')
        e.adjust('i_mot', min_val=0., max_val=80.)
    assert e.finish() == {}
    assert "the xvar wins" in capsys.readouterr().out


def test_the_xvar_check_no_longer_refuses_an_adjusted_key():
    import inspect
    assert "already registered as an adjust param" not in inspect.getsource(Scanner.xvar)


# ------------------------------------------------------------ applying

def test_an_untouched_row_never_writes_its_param():
    """compute_new_derived may set an adjust param per shot; an untouched row
    must not fight it."""
    e = Stub()
    e.adjust('i_mot', min_val=0., max_val=80.)
    specs = e.finish()
    vals = {'i_mot': specs['i_mot'].current_val}
    e.shot(vals)
    e.p.i_mot = 70.               # something else in the run changes it
    assert e.shot(vals)['i_mot'] == 70.


def test_a_changed_value_applies_and_reverting_restores_the_start():
    e = Stub()
    e.adjust('i_mot', min_val=0., max_val=80.)
    e.adjust('n_shots', min_val=1, max_val=10)
    e.finish()
    e.shot({'i_mot': 60., 'n_shots': 3.})
    used = e.shot({'i_mot': 72.5, 'n_shots': 7.})
    assert used['i_mot'] == 60.                          # the reply lands for the next shot
    used = e.shot({'i_mot': 60., 'n_shots': 3.})
    assert used['i_mot'] == 72.5 and used['n_shots'] == 7 and type(used['n_shots']) is int
    assert e.shot()['i_mot'] == 60. and e.p.n_shots == 3
    assert e._adjust_applied == set()


def test_save_data_locks_every_row_and_applies_nothing(capsys):
    e = Stub()
    e.adjust('i_mot', min_val=0., max_val=80.)
    specs = e.finish(save_data=True)
    assert specs['i_mot'].locked == AdjustSpec.LOCK_SAVE_DATA
    assert specs['i_mot'].to_dict()['locked'] == 'save_data'
    e.shot({'i_mot': 10.})
    assert e.shot({'i_mot': 10.})['i_mot'] == 60.
    assert "read-only" in capsys.readouterr().out


def test_values_for_unknown_keys_are_ignored():
    e = Stub()
    e.adjust('i_mot', min_val=0., max_val=80.)
    e.finish()
    e.shot({'t_tof': 1.})
    assert e.shot()['t_tof'] == 20.e-6


# ------------------------------------------------------------ derived

def test_a_derived_param_is_locked_with_a_warning(capsys):
    e = Stub()
    e.adjust('t_total', min_val=0., max_val=1.e-2)
    e.adjust('t_a', min_val=0., max_val=1.e-2)
    specs = e.finish()
    assert specs['t_total'].locked == AdjustSpec.LOCK_DERIVED
    assert specs['t_a'].locked == ''
    assert "p.t_total is recomputed every shot" in capsys.readouterr().out
    # and the check left every param as it was
    assert e.p.t_total == pytest.approx(3.e-3) and e.p.t_a == 1.e-3


def test_compute_new_derived_counts_as_derived():
    class WithNewDerived(Stub):
        def compute_new_derived(self):
            self.p.i_mot = 2 * self.p.t_a * 1.e4
    e = WithNewDerived()
    e.adjust('i_mot', min_val=0., max_val=80.)
    assert e.finish()['i_mot'].locked == AdjustSpec.LOCK_DERIVED


def test_the_derived_check_restores_arrays_in_place():
    e = Stub()
    arr = np.array([1., 2., 3.])
    e.p.arr = arr
    e.p.amp = 0.5

    def compute_derived():
        e.p.arr[:] = e.p.amp
    e.p.compute_derived = compute_derived
    e.adjust('amp', min_val=0., max_val=1.)
    assert e.finish()['amp'].locked == ''
    assert e.p.arr is arr and list(arr) == [1., 2., 3.]


def test_a_failing_derived_check_leaves_the_row_live(capsys):
    class Broken(Stub):
        def compute_new_derived(self):
            raise RuntimeError("needs self.data")
    e = Broken()
    e.adjust('i_mot', min_val=0., max_val=80.)
    assert e.finish()['i_mot'].locked == ''
    assert "could not check" in capsys.readouterr().out
    assert e.p.i_mot == 60.


# ------------------------------------------------------------ server rules

def _server_stub(specs, values):
    from waxx.util.live_od.live_od_server import LiveODServer
    s = types.SimpleNamespace(_adjust_lock=threading.Lock(),
                              _adjust_specs=specs, _adjust_values=values)
    for name in ("_handle_set_adjust_spec", "_handle_set_adjust_value",
                 "update_adjust_value", "update_adjust_spec"):
        setattr(s, name, types.MethodType(getattr(LiveODServer, name), s))
    return s


def _wire(key='i_mot', lo=0., hi=80., cur=60., locked=''):
    return AdjustSpec(key, lo, hi, 1., float, cur, 'A', locked).to_dict()


def test_the_server_refuses_a_narrowing_that_excludes_the_value():
    s = _server_stub([_wire()], {'i_mot': 60.})
    r = s._handle_set_adjust_spec({'key': 'i_mot', 'min_val': 0., 'max_val': 50.})
    assert r['ok'] is False and "excludes the current value" in r['error']
    assert s._adjust_specs[0]['max_val'] == 80. and s._adjust_values['i_mot'] == 60.
    assert s._handle_set_adjust_spec({'key': 'i_mot', 'min_val': 50., 'max_val': 70.})['ok']
    assert s._adjust_specs[0]['min_val'] == 50. and s._adjust_values['i_mot'] == 60.


def test_the_server_refuses_changes_to_a_locked_spec():
    s = _server_stub([_wire(locked='save_data')], {'i_mot': 60.})
    assert not s._handle_set_adjust_value({'key': 'i_mot', 'value': 10.})['ok']
    assert not s._handle_set_adjust_spec({'key': 'i_mot', 'max_val': 100.})['ok']
    s.update_adjust_value('i_mot', 10.)
    assert s._adjust_values['i_mot'] == 60.


def test_adjust_range_error():
    assert adjust_range_error('k', 5., 0., 10.) is None
    assert adjust_range_error('k', None, 0., 10.) is None
    assert "min" in adjust_range_error('k', 5., 10., 0.)
    assert "excludes" in adjust_range_error('k', 11., 0., 10.)
