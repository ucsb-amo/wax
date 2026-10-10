import numpy as np
from pathlib import Path
import json
import os
import socket
import threading
import time

from artiq.experiment import *
from artiq.experiment import delay, delay_mu

from waxa.config.expt_params import ExptParams
from waxa.data import DataSaver, RunInfo, counter, server_talk
from waxa.base.dealer import Dealer
from waxa.base.scribe import Scribe
from waxa.dummy.camera_params import CameraParams
from waxa import img_types

from artiq.language.core import kernel_from_string, now_mu, TerminationRequested

from waxx.config.data_vault import DataVault
from waxx.base.scanner import Scanner, WRITE_FAILURES
from waxx.base.shot_data_queue import ShotDataQueue, NO_PUT_DATA
from waxx.control.misc.oscilloscopes import ScopeData
from waxx.util.artiq.async_print import aprint
from waxx.util import console

RPC_DELAY = 10.e-3

# An aborted run's process exits within a second or two. If it is still alive
# this long after the abort, every thread's stack is printed (see
# _arm_exit_hang_dump).
T_ABORT_EXIT_HANG_DUMP = 30.
# The same after a normal end(). Exit may wait up to 20 s for the run-done
# mail (waxx.util.notifications._EXIT_WAIT_S), so this is longer.
T_END_EXIT_HANG_DUMP = 60.

# makes Expt.shot_data_queue once when stream workers ask for it together
_SHOT_QUEUE_LOCK = threading.Lock()


def _arm_exit_hang_dump(seconds=T_ABORT_EXIT_HANG_DUMP, announce=True):
    """If this process is still alive ``seconds`` from now, print every
    thread's Python stack to stderr (once). The run is over by then, so those
    stacks show what is holding the exit up. Nothing prints on a clean exit:
    the process is gone first. ``announce`` says so in the terminal first.
    Never raises."""
    try:
        import faulthandler
        import sys
        if announce:
            print(f"[abort] if this process has not exited in {seconds:g} s, its thread "
                  f"stacks will be printed here to show what is holding it up.",
                  file=sys.stderr)
        faulthandler.dump_traceback_later(seconds, exit=False)
    except Exception:
        pass


def _fmt_duration(seconds):
    """'45s', '3m07s', '1h02m' (ASCII, for terminal progress lines)."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _process_started():
    """This process's creation time for INIT_RUN's ``client_started`` (None
    when it cannot be read: liveOD then goes by the pid alone)."""
    try:
        from waxx.util.device_state.detached import process_started
        return process_started()
    except Exception:
        return None


#: ``WAXX_LAUNCHER`` of a run the monitor server's run queue launched
#: (waxx.util.device_state.run_queue.LAUNCHER).
QUEUE_LAUNCHER = "kq"


def _queue_restart_monitor(restart_monitor):
    """A run the run queue launched never restarts the monitor at its end: the
    queue's next job follows at once, and the monitor server starts the
    monitor itself when the queue runs out (or starts again the loop it
    stopped).  Says so in one line; any other run keeps ``restart_monitor``."""
    if os.environ.get("WAXX_LAUNCHER") != QUEUE_LAUNCHER or not restart_monitor:
        return restart_monitor
    print(f"[Monitor] launched by the run queue (job {os.environ.get('WAXX_QUEUE_JOB') or '?'}"
          "): the monitor is not restarted at the end of this run -- the monitor server "
          "starts it when the queue runs out.")
    return False


class Expt(Scanner, Dealer, Scribe):
    def __init__(self,
                 setup_camera=True,
                 save_data=True,
                 absorption_image=None,
                 server_talk=None,
                 verbosity=None):

        # Process-wide terminal chattiness (see waxx.util.console):
        # 0 = warnings only, 1 = milestones (default), 2 = everything.
        # The kwarg beats the WAX_VERBOSITY env var. _verbosity is the int
        # copy kernels gate their prints on.
        if verbosity is not None:
            console.set_level(verbosity)
        self._verbosity = int(console.get_level())

        if absorption_image != None:
            print("Warning: The argument 'absorption_image' is depreciated -- change it out for 'imaging_type'")
            print("Defaulting to absorption imaging.")

        # Scanner.__init__(self)
        super().__init__()

        self.setup_camera = setup_camera

        # NOTE: self.live_od_client is NOT assigned here.
        # Base.__init__ assigns a LiveODClient instance after calling super().
        # Assigning None here causes "cannot unify NoneType with LiveODClient"
        # during ARTIQ kernel compilation. All Expt methods guard with getattr.

        # defer_run_id=True: run_id is set to 0 here and updated after
        # INIT_RUN returns the server-assigned run_id.
        self.run_info = RunInfo(self, save_data, server_talk=server_talk,
                                defer_run_id=True)
        self.scope_data = ScopeData()
        self.scope_data.attach_expt(self)      # per-shot traces into the run's file
        # data pushed into the run's file during the run (push_shot_data)
        self._push_failed = set()               # keys END_RUN must carry after all
        self._n_pushed = 0
        self._n_push_failed = 0
        # stream-only containers' shots on their way to liveOD (queue_shot_data);
        # made on first use
        self._shot_queue = None
        # Per-shot auxiliary camera clients (waxx.control.cameras
        # .camera_stream_client); each registers itself here and is drained /
        # closed in end_wax. Host-only: never touched by kernel code.
        self.camera_streams = []
        self._ridstr = "Run ID: " + str(self.run_info.run_id)
        self._counter = counter()

        self.camera_params = CameraParams()

        self.params = ExptParams()
        self.p = self.params

        self.images = []
        self.image_timestamps = []

        self.xvarnames = []
        self.sort_idx = []
        self.sort_N = []

        self._setup_awg = False

        self.data = DataVault(expt=self)
        self.ds = DataSaver()

        # Shot-notification bookkeeping (populated in finish_prepare_wax)
        self._shot_complete_count = 0
        self._N_shots_total = 1
        self._t_first_shot_done = None   # time.monotonic() at shot 1's end

        # Extra provenance texts saved as attrs of the run's HDF5 file
        # ({attr_name: text}), next to expt_file / params_file. Machine-
        # agnostic: anything (e.g. a generated OPX program) can stash the
        # exact source it ran from here before end() is called.
        self._extra_file_texts = {}

        # Why the run is being stopped, set where host code raises to stop it
        # (the liveOD Abort button, a shot abandoned on an RTIO error), for
        # the device-state report of scan()'s exception handler.
        self._abort_cause = ""
        self._shot_abort = ""
        self._monitor_restart_sent = False

    def finish_prepare_wax(self,shuffle=True,N_repeats=[]):
        """
        To be called at the end of prepare. 
        
        Automatically adds repeats either if specified in N_repeats argument or
        if previously specified in self.params.N_repeats. 
        
        Shuffles xvars if specified (defaults to True). Computes the number of
        images to be taken from the imaging method and the length of the xvar
        arrays.

        Computes derived parameters within ExptParams.

        Accepts an additional compute_derived method that is user defined in the
        experiment file. This is to allow for recomputation of derived
        parameters that the user created in the experiment file at each step in
        a scan. This must be an RPC -- no kernel decorator.
        """

        if hasattr(self,'monitor'):
            self.monitor.init_monitor()
            self._adopt_monitor_snapshot()

        self.init_xvars(shuffle,N_repeats)

        self.data.init()

        # Reset per-run shot counter
        self._shot_complete_count = 0
        self._t_first_shot_done = None
        try:
            self._N_shots_total = int(np.prod(self.xvardims)) if self.xvardims else 1
        except Exception:
            self._N_shots_total = 1

        # overloaded per machine (kexp.Base: a due SLM reinit, between runs)
        self.pre_init_run()

        _client = getattr(self, 'live_od_client', None)
        if _client is not None:
            payload = self._serialize_init_payload()
            response = _client.init_run(payload)
            self.run_info.run_id = response['run_id']
            self.run_info.filepath = response['filepath']
            self._ridstr = "Run ID: " + str(self.run_info.run_id)
            if response['run_id']:
                if os.environ.get('WAXX_LAUNCHER'):
                    # a launcher (the run queue, the run loop, run_lock) reads
                    # this line from the output: never hidden by WAX_VERBOSITY
                    print(f"Run ID: {self.run_info.run_id}", flush=True)
                else:
                    console.info(f"Run ID: {self.run_info.run_id}")
        else:
            if self.run_info.save_data and self.setup_camera:
                raise RuntimeError(
                    "No liveOD server connection found. "
                    "Start the liveOD GUI before running experiments."
                )
            elif self.run_info.save_data:
                print(
                    "[LiveOD] WARNING: No liveOD server connection — "
                    "data will not be saved (setup_camera=False)."
                )

        if self._adjust_specs and self.run_info.save_data:
            print(
                "[adjust] WARNING: adjustable params detected with save_data=True. "
                "Values changed in the Adjust panel between shots will NOT be reflected "
                "in saved data."
            )

        # Fence composite ops from here until this run's end state arrives
        # (the monitor experiment is the one run that must not fence itself).
        if hasattr(self, 'monitor') and not getattr(self, '_is_monitor', False):
            try:
                self.monitor.announce_run(run_id=self.run_info.run_id,
                                          expt=self._expt_file_stem())
            except Exception as e:
                print(f"[Monitor] note: could not announce this run to the monitor "
                      f"server ({e!r}); composite ops are not fenced for it.")

    def pre_init_run(self):
        """Host, in finish_prepare_wax just before INIT_RUN: the last moment
        before the run starts -- liveOD arms the camera and hands out the run
        id at INIT_RUN, so a wait here costs neither the first frame's timeout
        nor a run id. Raising here stops the run before it has one. Overload
        per machine (kexp.Base)."""
        pass

    @kernel
    def cleanup_scan_kernel_wax(self):
        # self.ttl is provided by the machine layer's Devices mixin (a
        # subclass of waxx.config.ttl_id.ttl_frame). Stale input events left
        # in a TTLInOut FIFO cause an immediate-return gate and an underflow
        # on the next shot's trigger wait, so drain them all every shot.
        self.core.break_realtime()
        self.ttl.clear_input_events()
        self.data.put_shot_data()
        self._notify_shot_complete()

    def _notify_shot_complete(self):
        """RPC: notify the liveOD server that one shot has completed."""
        n = self._shot_complete_count + 1
        N = self._N_shots_total
        now = time.monotonic()
        if n == 1:
            self._t_first_shot_done = now
        try:
            xvar_values = {
                xv.key: float(xv.values[xv.counter])
                for xv in self.scan_xvars
            }
        except Exception:
            xvar_values = {}
        _client = getattr(self, 'live_od_client', None)
        if _client is None:
            # the only progress source without liveOD: every shot
            self._shot_complete_count += 1
            print(self._progress_line(n, N, now, xvar_values, stride=1))
            return
        try:
            reset_requested = _client.shot_complete(
                self._shot_complete_count,
                self._N_shots_total,
                xvar_values,
                shot_conditions=self._shot_conditions(),
            )
        except TypeError:
            # a client from before shot_conditions existed
            reset_requested = _client.shot_complete(
                self._shot_complete_count,
                self._N_shots_total,
                xvar_values,
            )
        self._pending_adjust_values = getattr(_client, 'last_adjust_values', {})
        self._shot_complete_count += 1
        stride = self._progress_stride(N)
        if stride and (n == 1 or n % stride == 0 or n == N):
            print(self._progress_line(n, N, now, xvar_values, stride))
        if reset_requested:
            self._abort_cause = "the liveOD Abort button"
            _client.abort_run()
            _arm_exit_hang_dump()
            raise TerminationRequested

    @staticmethod
    def _progress_stride(N):
        """Every how many shots the terminal reports progress (the first
        and last shot always): every shot at VERBOSE, a quarter of the run
        at NORMAL, never (0) at QUIET -- liveOD already shows live
        progress."""
        level = console.get_level()
        if level >= console.VERBOSE:
            return 1
        if level < console.NORMAL:
            return 0
        return max(1, -(-int(N) // 4))

    def _progress_line(self, n, N, now, xvar_values, stride):
        """One terminal progress line for shot n of N. The first says how
        often the rest are printed; later ones give the mean time per shot
        since shot 1 completed (compile and init excluded) and the ETA at
        that rate. ASCII only: ExptBuilder pipes stdout through cp1252."""
        if n == 1:
            line = f"shot 1/{N} done"
            if stride > 1:
                last = "" if N % stride == 0 else " and the last"
                line += f" -- printing every {stride} shots{last}"
        else:
            line = f"shot {n}/{N} ({100 * n // N}%)"
            t_first = getattr(self, '_t_first_shot_done', None)
            if t_first is not None:
                per_shot = (now - t_first) / (n - 1)
                line += f" | {per_shot:.1f} s/shot"
                if n < N:
                    left = per_shot * (N - n)
                    eta = time.strftime('%H:%M:%S',
                                        time.localtime(time.time() + left))
                    line += f" | {_fmt_duration(left)} left, ETA {eta}"
        if xvar_values and console.get_level() >= console.VERBOSE:
            line += " | " + ", ".join(f"{k}={v:.6g}"
                                      for k, v in xvar_values.items())
        return line

    def _shot_conditions(self) -> dict:
        """What this shot recorded about itself, for the live viewer: every
        single-valued DataVault container, as ``{key: float}``.

        Called right after ``data.put_shot_data()``, which has just synced each
        container's ``shot_data`` to the host. liveOD uses it to pick the
        absorption cross section from the field at imaging (kexp records
        ``i_outer_imaging``); waxx does not need to know which keys matter.
        Never raises: a run must not die for the viewer's sake.
        """
        conditions = {}
        try:
            for key in list(self.data.keys):
                value = np.asarray(getattr(self.data, key).shot_data)
                if value.size == 1 and np.issubdtype(value.dtype, np.number):
                    conditions[key] = float(value.ravel()[0])
        except Exception:
            pass
        return conditions

    def _adopt_monitor_snapshot(self):
        """Give scan()'s exception handler the monitor's channel-snapshot
        kernel and arrays (built from the device frames by init_monitor)."""
        m = self.monitor
        self._abort_snapshot_kernels = m._snapshot_kernels
        self._abort_snap_dds_f = m._snap_dds_f
        self._abort_snap_dds_a = m._snap_dds_a
        self._abort_snap_dds_v = m._snap_dds_v
        self._abort_snap_dds_sw = m._snap_dds_sw
        self._abort_snap_dac_v = m._snap_dac_v
        self._abort_snap_ttl_s = m._snap_ttl_s

    def _report_abort_state(self, what, dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s):
        """RPC from scan()'s exception handler, with every channel's state in
        the kernel at the abort (``what``: the exception's name when the
        handler could tell, else '').  Sends it to the monitor server as this
        run's end state and restarts the monitor, as end() does for a run
        that finishes.

        The state is trusted unless the run ended on an exception a channel
        write can raise itself (WRITE_FAILURES) -- then the write that raised
        may not have reached the hardware, and the state goes in marked
        untrusted.  Never raises: the exception on its way out of the kernel
        is the one the terminal must show."""
        if getattr(self, '_is_monitor', False) or not hasattr(self, 'monitor'):
            return
        rid = self.run_info.run_id
        try:
            cause = self._abort_cause or what or "an exception (see the traceback)"
            failed = next((w for w in (what, self._shot_abort) if w in WRITE_FAILURES), "")
            caveat = (f"a channel write that raised {failed} may not have reached the "
                      f"hardware, so that channel can differ" if failed else "")
            accepted = self.monitor.report_abort_state(
                dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s, run_id=rid,
                expt=self._expt_file_stem(), cause=cause, trusted=not failed, caveat=caveat)
            if accepted and failed:
                print(f"[Monitor] run {rid} aborted ({cause}): its last commanded device "
                      f"state went to the monitor server, marked UNTRUSTED -- {caveat}. "
                      f"Check that channel, then Trust state on the Device Control GUI.")
            elif accepted:
                print(f"[Monitor] run {rid} aborted ({cause}): its device state at the "
                      f"abort went to the monitor server (trusted).")
        except Exception as e:
            print(f"[Monitor] WARNING: could not report run {rid}'s device state at the "
                  f"abort ({e!r}); the device state stays untrusted.")
        try:
            self._restart_monitor_once()
        except Exception as e:
            print(f"[Monitor] WARNING: could not ask for a monitor restart ({e!r}).")

    def apply_pending_adjust_values(self):
        """Apply any adjust-panel values received from the last SHOT_COMPLETE reply."""
        specs = {s.key: s for s in self._adjust_specs}
        for key, val in self._pending_adjust_values.items():
            spec = specs.get(key)
            if spec is not None:
                # values arrive from the GUI as floats -- restore the registered dtype
                val = spec.coerce(val, like=getattr(self.params, key, None))
            setattr(self.params, key, val)

    def end_wax(self, expt_filepath,
                notify=True,
                restart_monitor=True):

        restart_monitor = _queue_restart_monitor(restart_monitor)
        _t0 = time.monotonic()
        try:
            self.scope_data.close()
        except Exception as _e:
            print(f"[end_wax] WARNING: scope_data.close() raised: {_e} — continuing.")

        # Drain the per-shot auxiliary camera clients BEFORE the END_RUN
        # payload is serialized: finish() waits for in-flight snaps, writes
        # provenance into _extra_file_texts, restores camera settings and
        # closes the stream. finish() never raises by contract; the guard is
        # belt and braces so a broken client cannot cost the run's data.
        self._finish_camera_streams()
        # what the streams queued and liveOD never took is counted here, and
        # recorded (aux_frames_dropped) before END_RUN is built
        self._close_shot_data_queue()
        _t_streams = time.monotonic() - _t0

        self.cleanup_scanned()

        _t_end_run = _t_save = 0.
        _client = getattr(self, 'live_od_client', None)
        if _client is not None:
            payload = self._serialize_end_payload(expt_filepath)
            _t1 = time.monotonic()
            _client.end_run(payload)
            _t_end_run = time.monotonic() - _t1
            _t_save = float(_client.last_end_run_reply.get("save_s") or 0.)
        else:
            # Legacy fallback
            if self.setup_camera:
                if self.run_info.save_data:
                    self.write_data(expt_filepath)
                else:
                    self.remove_incomplete_data()

        # Data is saved at this point. The Gmail round trip takes seconds, so
        # send it in the background while the monitor state is written and the
        # process winds down (non-daemon thread: exit waits for it).
        if notify:
            from waxx.util.notifications import send_run_done_email_async
            send_run_done_email_async(self.run_info.run_id, expt_filepath)

        _t2 = time.monotonic()
        if hasattr(self,'monitor'):
            # The end state goes through the monitor server (the only writer
            # of the state file); it marks the state trusted again.
            self.monitor.update_device_states(run_id=self.run_info.run_id,
                                              expt=self._expt_name_from_filepath(expt_filepath))
            if restart_monitor:
                self.monitor.signal_end()
        _t_monitor = time.monotonic() - _t2

        self._run_done_printout(expt_filepath)
        # where the end of the run went (the save is liveOD's own figure)
        pushed = f", {self._n_pushed} arrays pushed during the run" if self._n_pushed else ""
        failed = f" ({self._n_push_failed} pushes failed)" if self._n_push_failed else ""
        console.info(f"[end] streams closed {_t_streams:.1f} s | liveOD end_run "
                     f"{_t_end_run:.1f} s (save {_t_save:.1f} s) | monitor {_t_monitor:.1f} s"
                     f"{pushed}{failed}")

        # Runs have hung after this point (2026-09-26, run 83102) with nothing
        # left to show why: if the process outlives this by a minute, its
        # thread stacks print here. Silent when the process exits normally.
        _arm_exit_hang_dump(T_END_EXIT_HANG_DUMP, announce=False)

    def _run_done_printout(self, expt_filepath):
        rid = self.run_info.run_id
        import datetime
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        expt_name = self._expt_name_from_filepath(expt_filepath)
        name_str = f"  ({expt_name})" if expt_name else ""
        n, N = self._shot_complete_count, self._N_shots_total
        if 0 < n < N:
            # a run that stopped early (save_on_underflow) is never hidden
            # by the verbosity level
            print(f'run id {rid} ended at {dt} after {n} of {N} shots{name_str}')
        else:
            shots = f', {n} shots' if n else ''
            console.info(f'run id {rid} complete at {dt}{shots}{name_str}')

    @staticmethod
    def _expt_name_from_filepath(expt_filepath):
        """Return the stem (filename without extension) of expt_filepath."""
        if not expt_filepath:
            return ""
        return Path(expt_filepath).stem

    # ------------------------------------------------------------------
    # Payload serialisation helpers
    # ------------------------------------------------------------------

    def _expt_file_stem(self) -> str:
        """Name (no extension) of the file that defines this experiment's class,
        or '' if it cannot be found."""
        import inspect
        cls = type(self)
        try:
            return Path(inspect.getsourcefile(cls)).stem
        except Exception:
            pass
        # ARTIQ's file_import never registers the module in sys.modules, so
        # inspect cannot find the class's file.  The methods defined in the class
        # body still carry it (unwrap @kernel's functools.wraps wrapper first).
        for attr in vars(cls).values():
            try:
                code = inspect.unwrap(attr).__code__
            except Exception:
                continue
            if code.co_filename.endswith('.py'):
                return Path(code.co_filename).stem
        # file_import names the module prefix + file stem.
        prefix = "file_import_"
        if cls.__module__.startswith(prefix):
            return cls.__module__[len(prefix):]
        return ""

    def _xvar_ranges(self) -> dict:
        """{name: [min, max]} of each numeric xvar's scan: liveOD picks the unit it
        shows that xvar in from this, once for the run."""
        ranges = {}
        for xv in self.scan_xvars:
            try:
                values = np.asarray(xv.values, dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    ranges[str(xv.key)] = [float(values.min()), float(values.max())]
            except (TypeError, ValueError):
                pass
        return ranges

    # ---- data pushed into the run's file during the run -----------------
    #
    # liveOD pre-allocates every DataVault dataset at INIT_RUN and, with
    # PUT_DATA, takes a shot's value as soon as the host has it, written at
    # the slot it has once the run is unshuffled. END_RUN then carries no copy
    # of that container. A container that was never pushed and is big goes in
    # PUT_DATA slices just before END_RUN; what is left in END_RUN is small.
    #
    # Stream-only containers (camera streams' frames; HostDataContainer with
    # keep_run_data=False) have no copy in this process at all: each shot
    # goes through one bounded queue (queue_shot_data, waxx.base
    # .shot_data_queue) whose sender thread pushes it. What the queue could
    # not deliver is counted, never resent at the end, and recorded in the
    # file's aux_frames_dropped attribute; those slots keep liveOD's fill.

    # a container bigger than this that was not pushed during the run is
    # pushed in slices before END_RUN rather than carried inside it
    BULK_PUSH_BYTES = 8 * 1024 * 1024
    PUT_DATA_SLICE_BYTES = 48 * 1024 * 1024
    # every auxiliary camera stream's finish() runs at once; this is the wait
    T_STREAM_FINISH_S = 90.
    # the stream-only shot queue holds at most this much waiting for liveOD;
    # an item that would go over it is dropped (and counted)
    PUSH_QUEUE_MAX_BYTES = 50 * 1024 * 1024
    # a queued item liveOD refuses is tried this many more times
    PUSH_QUEUE_RETRIES = 2
    # after the streams finished (each drained the queue), the queue gets
    # this long more before what is left is recorded as not sent
    T_PUSH_QUEUE_CLOSE_S = 5.

    def current_shot_index(self):
        """The scan counters of the shot in progress, one per xvar."""
        return tuple(int(x.counter) for x in self.scan_xvars)

    def final_shot_index(self, idx):
        """Where the shot at scan-counter index ``idx`` lands once the run is
        unshuffled: the slot its data has in the file (the inverse of what
        the saver's unshuffle does, axis by axis)."""
        idx = tuple(int(i) for i in idx)
        if not self.sort_idx:
            return idx
        sort_N = [int(n) for n in self.sort_N]
        out = []
        for i, N in zip(idx, self.xvardims):
            N = int(N)
            if N in sort_N:
                out.append(int(np.asarray(self.sort_idx[sort_N.index(N)])[i]))
            else:
                out.append(i)
        return tuple(out)

    @property
    def push_data_enabled(self) -> bool:
        """liveOD takes arrays during this run: a run liveOD knows (a client
        exists) against a server with PUT_DATA. A run that saves nothing
        pushes too: liveOD keeps the latest of each and broadcasts them, and
        writes nothing."""
        client = getattr(self, "live_od_client", None)
        return bool(client is not None and client.supports("put_data"))

    def push_shot_data(self, items, idx=None):
        """Push one shot's value of each container in ``items`` (``[(dc,
        value), ...]``) into the run's file at the shot's final slot. ``idx``
        is the shot's scan-counter index (the shot in progress when None).
        True when liveOD took them. On a refusal or a failure the container
        falls back to END_RUN, which has the value anyway. Never raises; any
        host thread may call it."""
        if not self.push_data_enabled:
            return False
        if idx is None:
            idx = self.current_shot_index()
        final = self.final_shot_index(idx)
        specs = [{"key": dc.key, "index": final, "array": value,
                  "full_shape": dc._run_data.shape,
                  "fill": getattr(dc, "_fill_value", None)}
                 for dc, value in items]
        ok = self._push(specs)
        for dc, _ in items:
            if ok:
                dc._pushed = True
            else:
                self._push_failed.add(dc.key)
        return ok

    def push_raw(self, specs) -> bool:
        """Push arrays by key (``[{key, index | offset, array, full_shape,
        fill}]``): data outside any container (scope traces). True when
        liveOD took them. Never raises."""
        if not self.push_data_enabled:
            return False
        return self._push(specs)

    def _push(self, specs) -> bool:
        try:
            reply = self.live_od_client.put_data(specs)
        except Exception as e:
            self._note_push_failure(f"{type(e).__name__}: {e}")
            return False
        if not reply.get("ok"):
            self._note_push_failure(str(reply.get("error", "refused")))
            return False
        self._n_pushed += len(specs)
        return True

    def _note_push_failure(self, why):
        self._n_push_failed += 1
        if self._n_push_failed <= 3:
            print(f"[push] liveOD did not take pushed data ({why}); it goes with "
                  f"END_RUN instead.")

    # ---- the stream-only shot queue ----------------------------------------

    @property
    def shot_data_queue(self) -> ShotDataQueue:
        """The run's one queue for stream-only containers (made on first use)."""
        q = getattr(self, "_shot_queue", None)
        if q is None:
            with _SHOT_QUEUE_LOCK:
                q = getattr(self, "_shot_queue", None)
                if q is None:
                    q = ShotDataQueue(self._push_queued, self.PUSH_QUEUE_MAX_BYTES,
                                      retries=self.PUSH_QUEUE_RETRIES)
                    self._shot_queue = q
        return q

    def queue_shot_data(self, items, idx=None, on_done=None) -> bool:
        """Queue one shot's value of each stream-only container in ``items``
        (``[(dc, value), ...]``, one PUT_DATA message) for the run's file, at
        the shot's final slot; ``idx`` is the shot's scan-counter index (the
        shot in progress when None). Returns at once: True when queued.
        ``on_done(ok, reason)`` gets the outcome (waxx.base.shot_data_queue)
        -- ok only once liveOD has taken it. Never raises; any host thread."""
        return self._queue_shot_specs(items, idx, on_done, clear=False)

    def queue_shot_clear(self, containers, idx, on_done=None) -> bool:
        """Queue a reset of the shot's slot in each container to its fill
        value (a warm-up / duplicate shot starting the slot over: the frame
        the earlier request sent must not stay there). Behind everything
        queued before it; never dropped for room."""
        try:
            items = [(dc, np.full(dc._cell_shape, dc._fill_value, dtype=dc._run_data.dtype))
                     for dc in containers]
        except Exception as e:
            print(f"[push] !! could not build a slot clear for "
                  f"{[getattr(dc, 'key', '?') for dc in containers]} ({e!r})")
            if on_done is not None:
                on_done(False, "clear_error")
            return False
        return self._queue_shot_specs(items, idx, on_done, clear=True)

    def _queue_shot_specs(self, items, idx, on_done, clear):
        try:
            if idx is None:
                idx = self.current_shot_index()
            final = self.final_shot_index(idx)
            specs = [{"key": dc.key, "index": final, "array": value,
                      "full_shape": dc._run_data.shape,
                      "fill": getattr(dc, "_fill_value", None)}
                     for dc, value in items]
            q = self.shot_data_queue
            if not self.push_data_enabled:
                self._warn_no_put_data()
                q.drop(specs, NO_PUT_DATA, on_done=on_done, clear=clear)
                return False
            return q.put(specs, on_done=on_done, clear=clear)
        except Exception as e:
            print(f"[push] !! could not queue shot data "
                  f"{[getattr(dc, 'key', '?') for dc, _ in items]} ({e!r})")
            if on_done is not None:
                try:
                    on_done(False, "queue_error")
                except Exception:
                    pass
            return False

    def _warn_no_put_data(self):
        if getattr(self, "_no_put_data_warned", False):
            return
        self._no_put_data_warned = True
        why = ("this run has no liveOD client" if getattr(self, "live_od_client", None) is None
               else "this liveOD takes no PUT_DATA (an older liveOD: restart it from "
                    "the current code)")
        print(f"[push] !! {why}: stream-only per-shot data (auxiliary / diagnostic "
              f"camera frames) is NOT saved this run; every such frame is counted as "
              f"dropped ({NO_PUT_DATA}) in the stream's record and in aux_frames_dropped.")

    def _push_queued(self, specs) -> bool:
        """The shot queue's push: True, or raises saying why (the queue
        retries, counts and reports; nothing falls back to END_RUN)."""
        try:
            reply = self.live_od_client.put_data(specs)
        except Exception:
            self._n_push_failed += 1
            raise
        if not reply.get("ok"):
            self._n_push_failed += 1
            raise RuntimeError(f"liveOD refused it: {reply.get('error', 'refused')}")
        self._n_pushed += len(specs)
        return True

    def drain_shot_data(self, timeout) -> bool:
        """Wait up to ``timeout`` s for every queued shot to have its outcome.
        True when the queue is empty (or was never used)."""
        q = getattr(self, "_shot_queue", None)
        return True if q is None else q.drain(timeout)

    def _stream_only_keys(self):
        try:
            return [k for k in self.data.keys
                    if getattr(vars(self.data)[k], "stream_only", False)]
        except Exception:
            return []

    def _close_shot_data_queue(self):
        """After the streams finished: close the queue (what is still in it
        is recorded as not sent), let the streams correct their records for
        any outcome that came after they wrote them, and write
        ``aux_frames_dropped`` -- whenever the run has stream-only data, with
        zeros too, so its absence means "no queue", never "no drops"."""
        keys = self._stream_only_keys()
        if getattr(self, "_shot_queue", None) is None and not keys:
            return
        try:
            q = self.shot_data_queue
            q.ensure_keys(keys)
            left = q.close(self.T_PUSH_QUEUE_CLOSE_S)
            if any(left.values()):
                print(f"[push] !! the shot-data queue still held data at the end: "
                      f"{left} -- not sent; those slots keep their fill value")
            for cs in list(getattr(self, "camera_streams", ())):
                refresh = getattr(cs, "refresh_after_queue_close", None)
                if refresh is not None:
                    try:
                        refresh()
                    except Exception as e:
                        print(f"[push] WARNING: could not update stream "
                              f"'{getattr(cs, 'key', '?')}''s record ({e!r})")
            report = q.report()
            self._extra_file_texts["aux_frames_dropped"] = json.dumps(report)
            lost = {k: t for k, t in report["keys"].items()
                    if t["dropped"] or t["clears_failed"]}
            if lost:
                parts = []
                for k, t in sorted(lost.items()):
                    s = f"{k}: {t['dropped']} dropped {t['reasons']}"
                    if t["clears_failed"]:
                        s += (f", {t['clears_failed']} slot clears NOT sent (an earlier "
                              f"frame may remain at {t['first_clear_failed_slots']})")
                    parts.append(s)
                print(f"[push] !! per-shot data that never reached the run's file "
                      f"(slots keep their fill value; see the aux_frames_dropped "
                      f"attribute): " + "; ".join(parts))
        except Exception as e:
            print(f"[push] !! WARNING: closing the shot-data queue failed ({e!r}); "
                  f"aux_frames_dropped may be missing or incomplete")

    def _push_whole_container(self, dc) -> bool:
        """A big container END_RUN would otherwise carry: unshuffled here and
        pushed in slices along its first axis. True when it all went. Its
        in-memory copy is complete, so this also covers a container a per-shot
        push failed for. Never for a stream-only container (no copy here)."""
        key = getattr(dc, "key", "")
        arr = dc._run_data
        if (not self.push_data_enabled or not getattr(self.run_info, "save_data", False)
                or getattr(dc, "stream_only", False)
                or not isinstance(arr, np.ndarray) or arr.ndim == 0
                or arr.nbytes <= self.BULK_PUSH_BYTES
                or not getattr(dc, "_data_gotten", False)
                or getattr(dc, "_external_data_bool", False)):
            return False
        if self.sort_idx:
            n_per_shot = max(0, arr.ndim - len(self.xvardims))
            arr = DataSaver._unshuffle_single_array(
                arr, [np.array(s).tolist() for s in self.sort_idx],
                [int(n) for n in self.sort_N], exclude_dims=n_per_shot)
        rows = max(1, self.PUT_DATA_SLICE_BYTES // max(1, arr[0].nbytes))
        fill = getattr(dc, "_fill_value", None)
        for start in range(0, arr.shape[0], rows):
            spec = {"key": key, "index": None, "offset": start,
                    "array": arr[start:start + rows], "full_shape": arr.shape, "fill": fill}
            if not self._push([spec]):
                self._push_failed.add(key)
                return False
        dc._pushed = True
        # every slot was just (re)written from the complete copy
        self._push_failed.discard(key)
        return True

    def _finish_camera_streams(self):
        """finish() every auxiliary camera stream, all at once (each has its
        own connection and takes a few round trips). One still closing after
        T_STREAM_FINISH_S is left to its own atexit cleanup."""
        streams = list(getattr(self, 'camera_streams', ()))
        if not streams:
            return

        def one(cs):
            try:
                cs.finish()
            except Exception as e:
                print(f"[end_wax] WARNING: camera stream "
                      f"'{getattr(cs, 'key', '?')}' finish() raised: {e} — continuing.")

        threads = [threading.Thread(target=one, args=(cs,), daemon=True,
                                    name=f"finish:{getattr(cs, 'key', '?')}")
                   for cs in streams]
        for t in threads:
            t.start()
        deadline = time.monotonic() + self.T_STREAM_FINISH_S
        for t in threads:
            t.join(max(0., deadline - time.monotonic()))
        late = [t.name for t in threads if t.is_alive()]
        if late:
            print(f"[end_wax] WARNING: still closing after {self.T_STREAM_FINISH_S:.0f} s, "
                  f"left to their exit cleanup: {late}")

    def _serialize_init_payload(self) -> dict:
        """Build the INIT_RUN payload from current experiment state."""
        cam_params_dict = {
            k: v for k, v in vars(self.camera_params).items()
            if not k.startswith('_')
        }

        dv_shapes = {}
        for key in self.data.keys:
            dc = vars(self.data)[key]
            dv_shapes[key] = {
                'shape': dc._run_data.shape,
                'dtype': str(dc._run_data.dtype),
                'external': bool(dc._external_data_bool),
                # the slot nothing writes reads back as this, not as zero
                'fill': getattr(dc, '_fill_value', None),
            }

        if self.setup_camera:
            if isinstance(self.images, np.ndarray) and self.images.ndim > 1:
                images_shape = tuple(self.images.shape)
                images_dtype = str(self.images.dtype)
            else:
                images_shape = (0,)
                images_dtype = 'uint16'

            if isinstance(self.image_timestamps, np.ndarray) and self.image_timestamps.ndim > 0:
                ts_shape = tuple(self.image_timestamps.shape)
            else:
                ts_shape = (0,)
        else:
            images_shape = (0,)
            images_dtype = 'uint16'
            ts_shape = (0,)

        return {
            'save_data': bool(self.run_info.save_data),
            'capture_images': bool(self.setup_camera),
            'camera_key': str(getattr(self.camera_params, 'key', '')),
            'camera_params': cam_params_dict,
            'run_date_str': str(self.run_info.run_date_str),
            'run_datetime_str': str(self.run_info.run_datetime_str),
            'expt_class': str(self.run_info.expt_class),
            # what liveOD calls the run; the file is not passed in until end_wax
            'expt_file': self._expt_file_stem(),
            'imaging_type': int(self.run_info.imaging_type),
            'save_data_flag': int(self.run_info.save_data),
            'xvarnames': list(self.xvarnames),
            'xvardims': list(self.xvardims),
            'xvar_ranges': self._xvar_ranges(),
            'sort_idx': [np.array(s).tolist() for s in self.sort_idx] if self.sort_idx else [],
            'sort_N': [int(n) for n in self.sort_N] if self.sort_N else [],
            'images_shape': images_shape,
            'images_dtype': images_dtype,
            'image_timestamps_shape': ts_shape,
            'datavault_shapes': dv_shapes,
            'params': {
                k: v for k, v in vars(self.params).items()
                if not k.startswith('_') and not callable(v)
            },
            'N_shots_with_repeats': int(getattr(self.params, 'N_shots_with_repeats', 1)),
            'N_pwa_per_shot': int(getattr(self.params, 'N_pwa_per_shot', 1)),
            'save_on_underflow': int(getattr(self.run_info, 'save_on_underflow', 0)),
            'adjust_specs': [s.to_dict() for s in self._adjust_specs],
            # who runs this: liveOD shows them in POLL, and a launcher's gate
            # (waxx.util.device_state.run_gate) can tell a run whose process
            # is gone from a live one. WAXX_LAUNCHER is set by the launcher
            # ("run_lock", "run_loop", ...); "" for a person's `ar`.
            'client_pid': os.getpid(),
            'client_host': socket.gethostname(),
            # this process's creation time (epoch s; None off Windows): with
            # the pid it names this process, so a pid reused after it ends is
            # never taken for it
            'client_started': _process_started(),
            'launcher': str(os.environ.get('WAXX_LAUNCHER') or ''),
            # the run queue's job this run is (WAXX_QUEUE_JOB, set by the
            # queue): liveOD reports it in POLL and last_outcome, so the
            # queue knows its job's run without reading the output
            'queue_job': str(os.environ.get('WAXX_QUEUE_JOB') or ''),
        }

    def _serialize_end_payload(self, expt_filepath: str) -> dict:
        """Build the END_RUN payload from the final experiment state."""
        # Scope data
        scope_data_list = []
        if self.scope_data._scope_trace_taken:
            for scope in self.scope_data.scopes:
                if getattr(scope, "pushed_ok", False):
                    # every shot's traces went into the file during the run
                    scope_data_list.append({'label': str(scope.label), 'data': None,
                                            'pushed': True})
                    continue
                try:
                    reshaped = scope.reshape_data()
                except Exception as _e:
                    print(f"[_serialize_end_payload] WARNING: scope '{scope.label}' reshape_data() raised: {_e} — scope data will be empty for this run.")
                    reshaped = None
                if reshaped is None or not isinstance(reshaped, np.ndarray) or reshaped.ndim < 3 or reshaped.size == 0:
                    print(f"[_serialize_end_payload] WARNING: scope '{scope.label}' produced no usable data (shape={getattr(reshaped, 'shape', None)}) — omitting from payload.")
                else:
                    scope_data_list.append({
                        'label': str(scope.label),
                        'data': reshaped,
                    })

        # DataVault. A container that went into the file during the run, or
        # goes now in slices (big ones), is in there at its final slots
        # already: END_RUN carries no copy of it.
        dv = {}
        for key in self.data.keys:
            dc = vars(self.data)[key]
            if getattr(dc, 'stream_only', False):
                # Never carried: there is no copy here (its _run_data is a
                # zero-memory view of the fill). Its shots went in during the
                # run; a slot that did not get one keeps liveOD's pre-allocated
                # fill, and aux_frames_dropped says which and why.
                dv[key] = {'data': None, 'data_gotten': True,
                           'external': True, 'final_order': True}
                continue
            pushed = bool(getattr(dc, '_pushed', False)) and key not in self._push_failed
            if not pushed:
                pushed = self._push_whole_container(dc)
            if pushed:
                dv[key] = {'data': None, 'data_gotten': True,
                           'external': True, 'final_order': True}
                continue
            dv[key] = {
                'data': dc._run_data,
                'data_gotten': bool(dc._data_gotten),
                'external': bool(dc._external_data_bool),
            }

        # Source file texts (read from client filesystem via ds)
        expt_text = self.ds._read_text_file_safe(expt_filepath, "experiment") if expt_filepath else ""
        params_text = self.ds._read_text_file_safe(self.ds._expt_params_path, "params")

        base_class_texts = {}
        if self.ds._base_class_dir and os.path.isdir(self.ds._base_class_dir):
            try:
                filenames = sorted(os.listdir(self.ds._base_class_dir))
            except Exception:
                filenames = []
            for filename in filenames:
                if filename.endswith('.py') and not filename.startswith('__'):
                    fp = os.path.join(self.ds._base_class_dir, filename)
                    if os.path.isfile(fp):
                        key = f"base_class_{filename[:-3]}"
                        base_class_texts[key] = self.ds._read_text_file_safe(fp, filename)

        return {
            'params': {
                k: v for k, v in vars(self.params).items()
                if not k.startswith('_') and not callable(v)
            },
            'datavault': dv,
            'sort_idx': [np.array(s).tolist() for s in self.sort_idx] if self.sort_idx else [],
            'sort_N': [int(n) for n in self.sort_N] if self.sort_N else [],
            'xvardims': list(self.xvardims),
            'N_shots_with_repeats': int(getattr(self.params, 'N_shots_with_repeats', 1)),
            'N_pwa_per_shot': int(getattr(self.params, 'N_pwa_per_shot', 1)),
            'capture_images': bool(self.setup_camera),
            'scope_data_taken': bool(self.scope_data._scope_trace_taken),
            'scope_data': scope_data_list,
            'expt_filepath': str(expt_filepath),
            'expt_file_text': expt_text,
            'params_file_text': params_text,
            'base_class_texts': base_class_texts,
            'extra_file_texts': {str(k): str(v)
                                 for k, v in self._extra_file_texts.items()},
        }