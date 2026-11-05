"""A host-side harness for the scan loop (imported by tests, not collected).

A Scanner subclass with a fake core (its ``run`` calls the kernel's function)
and ARTIQ's simulation time manager (``delay`` works) runs the real
``Scanner.scan`` / ``_scan`` with the real Expt RPCs bound in. Hooks only
record their calls. Nothing touches a device or the network."""
from types import SimpleNamespace

import numpy as np
import pytest
from artiq.language.core import set_time_manager
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
    the loop needs; Scribe's abort hooks stubbed."""

    push_shot_data = Expt.push_shot_data
    _push = Expt._push
    _note_push_failure = Expt._note_push_failure
    final_shot_index = Expt.final_shot_index
    current_shot_index = Expt.current_shot_index
    push_data_enabled = Expt.push_data_enabled
    _shot_conditions = Expt._shot_conditions
    _notify_shot_complete = Expt._notify_shot_complete
    _progress_line = Expt._progress_line
    _progress_stride = Expt._progress_stride

    def __init__(self, n_data=5, save_on_underflow=False):
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
        self._abort_cause = ""
        self._shot_abort = ""
        self.calls = []
        self.aborts = []
        self.xvar("t_main", np.arange(float(n_data)))

    # ---- what Scribe / Expt / kexp.Base would provide
    def _check_for_abort_signal(self, raise_error=True):
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


def _prepared(n_data=5, **kw):
    e = LoopExpt(n_data=n_data, **kw)
    e.init_xvars(shuffle=False)
    e.data.init()
    _give_core(e.data, e.core)
    return e


def _kinds(e):
    return [c[0] for c in e.calls]
