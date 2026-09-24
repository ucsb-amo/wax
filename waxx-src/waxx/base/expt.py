import numpy as np
from pathlib import Path
import os

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

    @kernel
    def cleanup_scan_kernel_wax(self):
        self.data.put_shot_data()
        self._notify_shot_complete()

    def _notify_shot_complete(self):
        """RPC: notify the liveOD server that one shot has completed."""
        n = self._shot_complete_count + 1
        N = self._N_shots_total
        _client = getattr(self, 'live_od_client', None)
        if _client is None:
            print(f"shot {n}/{N}")
            self._shot_complete_count += 1
            return
        try:
            xvar_values = {
                xv.key: float(xv.values[xv.counter])
                for xv in self.scan_xvars
            }
        except Exception:
            xvar_values = {}
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
        if self._progress_worth_printing(n, N):
            print(f"shot {n}/{N} done")
        if reset_requested:
            _client.abort_run()
            raise TerminationRequested

    @staticmethod
    def _progress_worth_printing(n, N):
        """Shot progress on the terminal: every shot at VERBOSE, quarter
        milestones (and the last shot) at NORMAL, nothing at QUIET -- liveOD
        already shows live progress."""
        level = console.get_level()
        if level >= console.VERBOSE:
            return True
        if level < console.NORMAL:
            return False
        # True when n crosses a quarter boundary of N (~4 lines per run).
        return n * 4 // N > (n - 1) * 4 // N

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
            self.monitor.update_device_states()
            if restart_monitor:
                self.monitor.signal_end()

        self._run_done_printout(expt_filepath)

    def _run_done_printout(self, expt_filepath):
        rid = self.run_info.run_id
        import datetime
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        expt_name = self._expt_name_from_filepath(expt_filepath)
        name_str = f"  ({expt_name})" if expt_name else ""
        console.info(f'run id {rid} complete at {dt}{name_str}')

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
        }