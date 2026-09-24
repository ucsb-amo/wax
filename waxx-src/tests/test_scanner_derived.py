"""compute_new_derived: both override forms run every shot.

Scanner.__init__ used to assign self.compute_new_derived = nothing, an
instance attribute that shadowed any method override defined in an
experiment class -- so per-experiment derived params silently never
recomputed. These tests pin the fix (method overrides work) and the old
workaround (instance assignment) staying valid.
"""

import numpy as np

from waxx.base.scanner import Scanner


class MethodOverrideScanner(Scanner):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def compute_new_derived(self):
        self.calls += 1
        self.params.t_derived = self.params.t_test * 2.


def test_method_override_runs_per_step():
    sc = MethodOverrideScanner()
    sc.params.t_test = 0.
    sc.xvar('t_test', np.array([1., 2., 3.]))

    sc.update_params_from_xvars()
    assert sc.calls == 1
    assert sc.params.t_derived == 2.

    sc.scan_xvars[0].counter = 2
    sc.update_params_from_xvars()
    assert sc.calls == 2
    assert sc.params.t_derived == 6.


def test_instance_assignment_still_works():
    sc = Scanner()
    sc.params.t_test = 0.
    sc.xvar('t_test', np.array([1., 2.]))
    calls = []
    sc.compute_new_derived = lambda: calls.append(1)
    sc.update_params_from_xvars()
    assert calls == [1]
