import numpy as np
from pathlib import Path
import os
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
from waxx.base.scanner import Scanner
from waxx.control.misc.oscilloscopes import ScopeData
from waxx.util.artiq.async_print import aprint
from waxx.util import console

RPC_DELAY = 10.e-3


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

        self.init_xvars(shuffle,N_repeats)

        self.data.init()

        # Reset per-run shot counter
        self._shot_complete_count = 0
        self._t_first_shot_done = None
        try:
            self._N_shots_total = int(np.prod(self.xvardims)) if self.xvardims else 1
        except Exception:
            self._N_shots_total = 1

        _client = getattr(self, 'live_od_client', None)
        if _client is not None:
            payload = self._serialize_init_payload()
            response = _client.init_run(payload)
            self.run_info.run_id = response['run_id']
            self.run_info.filepath = response['filepath']
            self._ridstr = "Run ID: " + str(self.run_info.run_id)
            if response['run_id']:
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
            _client.abort_run()
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
                line += (f" -- printing every {stride} shots{last}"
                         f" (Base(verbosity=2) or WAX_VERBOSITY=2: every shot)")
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

        try:
            self.scope_data.close()
        except Exception as _e:
            print(f"[end_wax] WARNING: scope_data.close() raised: {_e} — continuing.")

        self.cleanup_scanned()

        _client = getattr(self, 'live_od_client', None)
        if _client is not None:
            payload = self._serialize_end_payload(expt_filepath)
            # print(payload)
            _client.end_run(payload)
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

        if hasattr(self,'monitor'):
            # The end state goes through the monitor server (the only writer
            # of the state file); it marks the state trusted again.
            self.monitor.update_device_states(run_id=self.run_info.run_id,
                                              expt=self._expt_name_from_filepath(expt_filepath))
            if restart_monitor:
                self.monitor.signal_end()

        self._run_done_printout(expt_filepath)

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
        }

    def _serialize_end_payload(self, expt_filepath: str) -> dict:
        """Build the END_RUN payload from the final experiment state."""
        # Scope data
        scope_data_list = []
        if self.scope_data._scope_trace_taken:
            for scope in self.scope_data.scopes:
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

        # DataVault
        dv = {}
        for key in self.data.keys:
            dc = vars(self.data)[key]
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