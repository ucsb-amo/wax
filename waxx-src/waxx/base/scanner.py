"""The scan: run one shot for every value of the scanned parameters.

What an experiment does with it
-------------------------------
In ``prepare()``::

    self.xvar('t_tof', np.linspace(1.e-3, 20.e-3, 10))    # scan a param
    self.adjust('i_mot', min_val=0., max_val=80.)         # optional: live knob in liveOD
    self.finish_prepare(shuffle=True)                     # builds the scan (init_xvars)

and in the kernel, ``self.scan()`` runs the shots. The experiment writes
``scan_kernel`` (one shot); the machine's base class (kexp.Base) fills in the
other hooks. In order::

    scan()
      pre_scan()                    once: warm-up shots
      for each point of the scan:
          <next point's params copied into the kernel>
          init_scan_kernel()        per-shot setup
          scan_kernel()             THE SHOT
          cleanup_scan_kernel()     per-shot cleanup, data write
      post_scan()                   once

Several xvars scan like nested for loops, the last one declared innermost.

When something goes wrong
-------------------------
==================================  ====================  ============================
what happened                       what runs             then
==================================  ====================  ============================
scan_kernel returned                cleanup_scan_kernel   next shot
RTIOUnderflow, RTIOOverflow or      cleanup_scan_kernel   the run ends [1]
TriggerTimeout in scan_kernel
any other exception in scan_kernel  cleanup_abort_kernel  the run ends, exception
                                                          re-raised; the shot is not
                                                          recorded
an exception in pre_scan, or        cleanup_abort_kernel  the run ends, exception
between two shots (liveOD Abort,                          re-raised
the param update, init_scan_kernel)
an exception in                     cleanup_abort_kernel  the run ends, exception
cleanup_scan_kernel itself                                re-raised
==================================  ====================  ============================

[1] The original exception is re-raised, so the host prints its traceback
(for an underflow: channel, timestamp, kernel line). With
``run_info.save_on_underflow`` (and save_data) the run instead ends normally
and saves the shots taken.

``cleanup_abort_kernel`` is the machine's safe state (kexp: coils off, high
power beams off) with no data write. Between two shots the machine may be
left running on purpose (kexp's MOT pre-load), which is why every exit above
reaches one of the two cleanups.

Whenever an exception leaves ``scan()``, the kernel first tells the host what
it left every channel at (scanner_parts/abort_state.py).

Where the rest is
-----------------
The plumbing is in ``waxx/base/scanner_parts/`` (see its ``__init__``): the
Adjust panel, the host-to-kernel param copy, shot counting, the abort-state
report and the camera exposure delays. ``Scanner`` inherits all of it.
"""
from artiq.experiment import *
import numpy as np

from waxa.base import xvar
from waxa.data import RunInfo
from waxa.dummy.camera_params import CameraParams

from artiq.experiment import RTIOUnderflow
from artiq.coredevice.exceptions import RTIOOverflow, RTIODestinationUnreachable

from waxx.control.exceptions import TriggerTimeout
from waxx.util.artiq.async_print import aprint

from waxx.base.scanner_parts import (AdjustPanel, ParamSync, ShotGrid, AbortState,
                                     ExposureDelays)
# these names are imported from here by other modules
from waxx.base.scanner_parts import AdjustSpec, WRITE_FAILURES, SNAPSHOT_PARAMS

# Feature flag: every exit of the scan loop reaches cleanup_scan_kernel or
# cleanup_abort_kernel (the table above). kexp's MOT pre-load, which keeps the
# MOT on between shots, refuses to run on a scanner without it.
SAFE_STATE_ON_EVERY_CLEANUP_FAILURE = True


class Scanner(AdjustPanel, ParamSync, ShotGrid, AbortState, ExposureDelays):

    def __init__(self):

        from waxx.config.expt_params import ExptParams
        self.params = ExptParams()
        self.run_info = RunInfo()
        self.camera_params = CameraParams()

        self.xvarnames = []
        self.scan_xvars = []
        self.Nvars = 0
        self.update_nvars()

        from waxx.control.artiq.dummy_core import DummyCore
        self.core = DummyCore()

        # True while cleanup_scan_kernel runs after a scan_kernel that
        # returned normally; False after one that raised. After an exception
        # (an underflow drops the events not yet played) the kernel's cached
        # device state -- a coil's current, say -- may not be the hardware's,
        # so a cleanup may skip work only when this is True.
        self.shot_completed = False

        # the plumbing's state (scanner_parts)
        self._init_adjust()
        self._init_param_sync()
        self._init_abort_state()
        self._init_exposure_delays()

    # =========================================================================
    # Declaring the scan (in prepare)
    # =========================================================================

    def xvar(self,key,values):
        """Adds an xvar to the experiment.

        Args:
            key (str): The key of the ExptParams attribute to scan.
            values (ndarray): Values to scan over. Can be n-dimensional, scan will step over first index.
        """
        # (an adjust() of the same key is dropped at finish_prepare: the xvar wins)
        this_xvar = xvar(key,values,position=len(self.scan_xvars))
        if this_xvar.key in [x.key for x in self.scan_xvars]:
            raise ValueError(f"xvar of key {this_xvar.key} is assigned more than once.")
        self.scan_xvars.append(this_xvar)
        self.xvarnames.append(this_xvar.key)
        # check if params has this xvar key already -- if not, add it
        self.new_param_check(this_xvar)
        self.update_nvars()

    def new_param_check(self,xvar):
        self.param_police(xvar)

        params_keylist = list(self.params.__dict__.keys())
        if xvar.key not in params_keylist:
            # set value to a single value (vs list), it will be overwritten per shot in scan
            vars(self.params)[xvar.key] = xvar.values[0]

    def param_police(self,xvar):
        forbidden_chars = [":",",","."," ","-","+","(",")","@","#","$","%","^","&","*","=","!","[","]",";","/","\\","`","~"]
        if any([fc in xvar.key for fc in forbidden_chars]):
            raise ValueError("Key contains forbidden characters.")

    def update_nvars(self):
        """Updates the number of xvars to be scanned.
        """
        self.Nvars = len(self.scan_xvars)

    def logspace(self,start,end,n):
        return np.logspace(np.log10(start),np.log10(end),int(n))

    def powspace(self,start,end,n,power=3):
        u = np.linspace(0, 1, n)
        return (start + (end - start) * (1 - (1 - u) ** power))

    # self.adjust(...) is in scanner_parts/adjust.py

    def init_xvars(self, shuffle=True, N_repeats=[]):
        """finish_prepare: builds the scan from the xvars declared in prepare.

        Adds the repeats, shuffles the order of the points, counts the shots
        and images, computes the derived params, and builds the kernels that
        copy the params into the kernel before each shot. A run with no xvar
        gets a one-point scan of a param named "dummy".
        """
        from waxa import img_types
        if self.run_info.imaging_type == img_types.ABSORPTION:
            if self.params.N_pwa_per_shot > 1:
                print("You indicated more than one PWA per shot, but the analysis is set to absorption imaging. Setting # PWA to 1.")
            self.params.N_pwa_per_shot = 1

        if not self.xvarnames:
            self.xvar("dummy",[0])
        if self.xvarnames and not self.scan_xvars:
            for key in self.xvarnames:
                self.xvar(key,vars(self.params)[key])
        self.plug_in_xvars()

        self.repeat_xvars(N_repeats=N_repeats)

        if shuffle:
            self.shuffle_xvars()

        self.params.N_img = self.get_N_img()
        self.prepare_image_array()

        self.params.compute_derived()
        self.compute_new_derived()

        self.xvardims = [len(xvar.values) for xvar in self.scan_xvars]
        if hasattr(self,'scope_data'):
            self.scope_data.xvardims = self.xvardims

        self.generate_assignment_kernels()

    # =========================================================================
    # The hooks: what an experiment or a machine's base class fills in
    # =========================================================================

    @kernel
    def scan_kernel(self):
        """One shot. Written per experiment.

        It should correspond to a single "shot" (single set of images to
        generate one OD) and accept no arguments. A scanned parameter is read
        as ``self.p.<key>``: it holds this shot's value.
        """
        pass

    @kernel
    def pre_scan(self):
        """Run once, before the first shot (kexp.Base: the warm-up shots)."""
        pass

    @kernel
    def init_scan_kernel(self):
        """Run before every shot, after its params are in the kernel
        (kexp.Base)."""
        pass

    @kernel
    def cleanup_scan_kernel(self):
        """Run after every shot that returned, and after one that ended on
        RTIOUnderflow, RTIOOverflow or TriggerTimeout (kexp.Base: hardware
        off, the shot's data write). ``self.shot_completed`` says which."""
        pass

    @kernel
    def cleanup_abort_kernel(self):
        """The machine's safe state, run before any other exception ends the
        run (the table at the top of this file). No data write, no shot
        notification. Overload per machine (kexp.Base)."""
        pass

    @kernel
    def post_scan(self):
        """Run once, after the last shot (kexp.Base)."""

    def compute_new_derived(self):
        """Per-experiment derived params, recomputed on the host before every
        shot, right after params.compute_derived().

        Override it as a method in the experiment class (or assign
        ``self.compute_new_derived = self.my_func``) to recompute the
        experiment's own derived quantities from this shot's xvar values.
        Host only: no @kernel.
        """
        pass

    # =========================================================================
    # The scan
    # =========================================================================

    @kernel
    def scan(self, raise_underflow=False):
        """Runs scan_kernel once for every point of the scan. See the top of
        this file for the order of the hooks and what each kind of failure
        does.

        raise_underflow: ignored (the traceback it asked for is now always
        printed).
        """
        self.load_exposure_delays()
        # An exception that ends the scan first hands the kernel's channel
        # state to the host. The handlers bind no names (the compiler gives a
        # local one type), hence one clause per exception the report names.
        try:
            self._scan(raise_underflow)
        except RTIOUnderflow:
            self._hand_over_abort_state("RTIOUnderflow")
            raise
        except RTIOOverflow:
            self._hand_over_abort_state("RTIOOverflow")
            raise
        except RTIODestinationUnreachable:
            self._hand_over_abort_state("RTIODestinationUnreachable")
            raise
        except ValueError:
            self._hand_over_abort_state("ValueError")
            raise
        except:
            self._hand_over_abort_state("")
            raise

    @kernel
    def _scan(self, raise_underflow):
        """The shot loop (see scan)."""

        if raise_underflow:
            self._note_raise_underflow_ignored()

        # The warm-up shots are full sequences and can fail like any shot.
        try:
            self.pre_scan()
        except:
            self._safe_state()
            raise

        scanning = True
        stop_after_shot = False     # this shot is the run's last
        failed_on = ""              # the exception that made it the last

        while scanning:

            # ---- between two shots: this shot's params into the kernel ------
            # No delay and no sync here: the RPCs below run at kernel-CPU time
            # whatever the timeline cursor says.
            try:
                stop_after_shot = self._check_for_abort_signal()
                self.update_params_from_xvars()
                self.write_host_params_to_kernel()
                self.core.break_realtime()
                self.init_scan_kernel()
            except:
                self._safe_state()
                raise

            self.core.break_realtime()

            # ---- the shot ----------------------------------------------------
            # A shot that raises is cleaned up INSIDE its handler and then
            # re-raised with a bare `raise`: an exception cannot be passed to
            # an RPC and is freed when its except block ends, so this is the
            # only way to keep its traceback.
            self.shot_completed = False
            try:
                self.scan_kernel()
                self.shot_completed = True
            except RTIOUnderflow:
                if not self._clean_up_failed_shot("RTIOUnderflow"):
                    raise
                stop_after_shot = True
                failed_on = "RTIOUnderflow"
            except TriggerTimeout:
                # A gated wait (line trigger, OPX hand-back) closed with no
                # edge, so the shot cannot continue.
                aprint("[scan] shot aborted: a triggered wait saw no edge "
                       "(TriggerTimeout, see the line above). Cleaning up and "
                       "ending the run.")
                if not self._clean_up_failed_shot("TriggerTimeout"):
                    raise
                stop_after_shot = True
                failed_on = "TriggerTimeout"
            except RTIOOverflow:
                # An input FIFO overflowed inside a gated wait (a bouncing or
                # free-running line on a TTLInOut).
                aprint("[scan] shot aborted: an RTIO input FIFO overflowed "
                       "(RTIOOverflow) -- a TTL input is toggling far faster "
                       "than expected. Cleaning up and ending the run.")
                if not self._clean_up_failed_shot("RTIOOverflow"):
                    raise
                stop_after_shot = True
                failed_on = "RTIOOverflow"
            except:
                # Anything else (a ValueError from a DAC/DDS value, a lost
                # DRTIO link, an exception from an RPC). The failed shot is
                # not recorded.
                self._safe_state()
                raise

            # ---- cleanup of a shot that returned (a failed one had its
            # cleanup in its handler) -----------------------------------------
            if self.shot_completed:
                self._cleanup_or_safe_state()

            self.core.break_realtime()

            # ---- step to the next point -------------------------------------
            scanning = self.step_scan()
            self.update_scan_xvar_counters()

            self.core.break_realtime()

            if stop_after_shot:
                scanning = False

        if stop_after_shot:
            self._send_abort_to_server(failed_on)

        self.post_scan()

    @kernel
    def _safe_state(self):
        """The machine's safe state, from wherever the timeline was left.
        Called in an except block, just before its bare ``raise``."""
        self.core.break_realtime()
        self.cleanup_abort_kernel()

    @kernel
    def _cleanup_or_safe_state(self):
        """cleanup_scan_kernel, and if IT raises (kexp: a wrong image count,
        the liveOD Abort in the shot-complete RPC), the machine's safe state
        before the exception goes on: a cleanup that raised before its coil
        reset must not leave a coil on."""
        try:
            self.cleanup_scan_kernel()
        except:
            self._safe_state()
            raise

    @kernel
    def _clean_up_failed_shot(self, what) -> TBool:
        """A shot ended on ``what`` (RTIOUnderflow, RTIOOverflow or
        TriggerTimeout): run its cleanup, then ask the host what to do with
        the run (_abort_shot). False: the run is aborted and the handler
        re-raises the exception. True (save_on_underflow): the run ends
        normally after this shot and saves the shots taken."""
        self.core.break_realtime()
        self._cleanup_or_safe_state()
        return self._abort_shot(what)

    def update_params_from_xvars(self):
        """Before each shot, on the host: this shot's value of every xvar goes
        into the params, then the Adjust panel's values, then the derived
        params are recomputed.

        Does not update the values of the kernel ExptParams -- to do so, use
        write_host_params_to_kernel().
        """

        # update each xvar parameter in the host params
        for xvar in self.scan_xvars:
            vars(self.params)[xvar.key] = xvar.values[xvar.counter]
        # apply any live GUI adjustments before recomputing derived params
        self.apply_pending_adjust_values()
        # update derived params in the host params
        self.params.compute_derived()
        self.compute_new_derived()

    def _note_raise_underflow_ignored(self):
        print("[scan] scan(raise_underflow=True) is no longer needed and is ignored: "
              "an aborted shot is always cleaned up, and the original exception is "
              "re-raised with its traceback (2026-09-27).")

    # _check_for_abort_signal, _abort_shot(what) -> bool and
    # _send_abort_to_server, the RPCs the loop calls, are defined in waxa
    # Scribe (no stubs here: Scanner precedes Scribe in Expt's MRO and would
    # shadow them).

    def cleanup_image_count(self):
        # dummy, overloaded by kexp.image.cleanup_image_count
        pass
