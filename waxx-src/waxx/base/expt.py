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

from artiq.language import kernel_from_string, now_mu, TBool

from waxx.config.data_vault import DataVault
from waxx.base.scanner import Scanner
from waxx.control.misc.oscilloscopes import ScopeData
from waxx.util.artiq.async_print import aprint

from waxx.util.live_od.camera_client import CameraClient

RPC_DELAY = 10.e-3

class Expt(Dealer, Scanner, Scribe):
    def __init__(self,
                 setup_camera=True,
                 save_data=True,
                 absorption_image=None,
                 server_talk=None):
        
        if absorption_image != None:
            print("Warning: The argument 'absorption_image' is depreciated -- change it out for 'imaging_type'")
            print("Defaulting to absorption imaging.")

        Scanner.__init__(self)
        super().__init__()

        self.setup_camera = setup_camera
        self.run_info = RunInfo(self,save_data,server_talk=server_talk)
        self.scope_data = ScopeData()
        self._ridstr = " Run ID: "+ str(self.run_info.run_id)
        self._counter = counter()

        self.camera_params = CameraParams()

        self.live_od_client = CameraClient(None, None)

        self.params = ExptParams()
        self.p = self.params

        # self.images = []
        # self.image_timestamps = []

        self.xvarnames = []
        self.sort_idx = []
        self.sort_N = []

        self._setup_awg = False

        self.data = DataVault(expt=self)
        self.ds = DataSaver()

    def send_new_run(self):
        self.live_od_client.connect()
        data_spec = {}
        run_info_attrs = {}
        params_attrs = {}
        camera_params_attrs = {}
        if self.run_info.save_data:
            data_spec          = self._build_data_spec()
            run_info_attrs     = self._obj_to_pickle_dict(self.run_info)
            params_attrs       = self._obj_to_pickle_dict(self.params)
            camera_params_attrs = self._obj_to_pickle_dict(self.camera_params)
        self.live_od_client.send_new_run(
            camera_params=self.camera_params,
            data_filepath=getattr(self.run_info, 'filepath', ''),
            save_data=self.run_info.save_data,
            setup_camera=self.setup_camera,
            N_img=self.params.N_img,
            N_shots=self.params.N_shots,
            N_pwa_per_shot=self.p.N_pwa_per_shot,
            imaging_type=self.run_info.imaging_type,
            run_id=self.run_info.run_id,
            data_spec=data_spec,
            run_info_attrs=run_info_attrs,
            params_attrs=params_attrs,
            camera_params_attrs=camera_params_attrs,
        )
    
    @kernel
    def init_kernel_wax(self):
        # if self.run_info.save_data:
        #     self.ds.create_data_file(self)
        self.send_new_run()
        # pass

    def finish_prepare_wax(self,N_repeats=[],shuffle=True):
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

        self.set_up_imaging_containers()
        self.data.init()

        if self.run_info.save_data:
            # Compute the data filepath; the camera server creates the HDF5
            # file on the server side (via the data_spec sent in new_run), so
            # the experiment process never opens the data directory directly.
            fpath, _ = self.ds._data_path(self.run_info)
            self.run_info.filepath = fpath
            self.data_filepath     = fpath
            self.run_info.xvarnames = self.xvarnames

    def set_up_imaging_containers(self):
        # Image arrays are represented as data-vault containers and populated
        # by the camera server. Size the per-shot payload from camera params.
        if self.setup_camera:
            N_img_per_shot = self.p.N_pwa_per_shot+2
            image_shape = self.camera_params.resolution
            if self.camera_params.camera_type == 'andor':
                image_dtype = np.uint16
            elif self.camera_params.camera_type == 'basler':
                image_dtype = np.uint8
            else:
                image_dtype = np.uint8

            # the old code for image processing relied on the image data being 3*N
            # by px by py -- a totally linear array of images of length 3*number of
            # shots. the data container code would add an extra axis for those 3
            # images per shot, but for backward compatability with old analysis code
            # I've added the "flat" argument and allowed the user to specify the
            # number of values per shot -- this flattens the image data and
            # timestamp containers to be the same shape as the old ones.
            self.data.images = self.data.add_data_container(image_shape,dtype=image_dtype,flat=True,
                                                            flat_points_per_shot=N_img_per_shot)
            self.data.image_timestamps = self.data.add_data_container(1,dtype=float,flat=True,
                                                                    flat_points_per_shot=N_img_per_shot)

    @kernel
    def cleanup_scan_kernel_wax(self):
        self.data.put_shot_data()
        # print(self.data._expt)
    
    def compute_new_derived(self):
        pass
    
    def end_wax(self, expt_filepath):

        self.scope_data.close()

        if self.run_info.save_data:
            self.cleanup_scanned()
            self._send_write_data_to_server(expt_filepath)
        # else:
        #     self.remove_incomplete_data()

        if hasattr(self,'monitor'):
            self.monitor.update_device_states()
            self.monitor.signal_end()

        try:
            self.live_od_client.send_run_complete()
        except Exception as e:
            print(e)
                
        # server_talk.play_random_sound()

    # ------------------------------------------------------------------
    #  Server-side data file helpers
    # ------------------------------------------------------------------

    def _build_data_spec(self):
        """Build the file-creation spec dict to send to the camera server."""
        containers = []
        for key in self.data.keys:
            dc = vars(self.data)[key]
            containers.append({
                "key":   key,
                "shape": list(dc._run_data.shape),
                "dtype": dc._run_data.dtype.str,   # e.g. "<f8"
                "external": dc._external_data_bool,
            })
        sort_idx = [s.tolist() for s in self.sort_idx] if self.sort_idx else []
        sort_N   = [int(n) for n in self.sort_N]       if self.sort_N   else []
        return {
            "xvarnames": list(self.xvarnames),
            "xvardims": list(self.xvardims),
            "containers": containers,
            "sort_idx": sort_idx,
            "sort_N":   sort_N,
        }

    def _obj_to_pickle_dict(self, obj):
        """Serialize non-private object attributes to a pickle-safe dict."""
        import pickle
        result = {}
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            try:
                pickle.dumps(v)
                result[k] = v
            except Exception:
                try:
                    result[k] = str(v)
                except Exception:
                    pass
        return result

    def _read_text_file(self, path):
        """Return the text content of *path*; empty string on any failure."""
        if path and os.path.isfile(path):
            try:
                with open(path) as f:
                    return f.read()
            except Exception:
                pass
        return ""

    def _send_write_data_to_server(self, expt_filepath):
        """Collect all run data and send it to the camera server for writing.

        The camera server writes the data to the HDF5 file it created at
        the start of the run, keeping all NAS I/O on the server machine.
        This must be called BEFORE send_run_complete so the persistent
        connection is still open.
        """
        # Non-external DataContainer arrays (raw acquisition order).
        # Server applies unshuffle using Dealer context initialized at new_run.
        data = {}
        for key in self.data.keys:
            dc = vars(self.data)[key]
            if dc._external_data_bool:
                continue  # images already on server from grab loop
            if dc._data_gotten:
                data[key] = dc._run_data

        # Sort context so the server can unshuffle external data in-place
        sort_info = {
            "has_sort":           bool(self.sort_idx),
            "sort_idx":          [s.tolist() for s in self.sort_idx] if self.sort_idx else [],
            "sort_N":            [int(n) for n in self.sort_N]       if self.sort_N   else [],
            "xvardims":          list(getattr(self, 'xvardims', [])),
            "N_pwa_per_shot":     int(self.params.N_pwa_per_shot),
            "N_shots_with_repeats": int(getattr(self.params, 'N_shots_with_repeats',
                                               int(np.prod(self.xvardims) if self.xvardims else 1))),
        }

        # Scope data (raw acquisition order); server unshuffles.
        scope_data_list = []
        if self.scope_data._scope_trace_taken:
            for scope in self.scope_data.scopes:
                try:
                    data_arr = scope.reshape_data()
                    data_arr = np.asarray(data_arr).astype(np.float32)
                    t = np.take(np.take(data_arr, 0, axis=-2), 0, axis=-2)
                    v = np.take(data_arr, 1, -2)
                    scope_data_list.append({"label": scope.label, "t": t, "v": v})
                except Exception as e:
                    print(f"Warning: could not serialize scope {getattr(scope, 'label', '?')}: {e}")

        # Source-file texts (all local to the experiment machine)
        texts = {
            "expt":    self._read_text_file(expt_filepath),
            "params":  self._read_text_file(getattr(self.ds, '_expt_params_path', '')),
            "cooling": self._read_text_file(getattr(self.ds, '_cooling_path', '')),
            "imaging": self._read_text_file(getattr(self.ds, '_imaging_path', '')),
            "control": self._read_text_file(getattr(self.ds, '_control_path', '')),
        }

        try:
            self.live_od_client.send_write_data(
                data=data,
                params_attrs=self._obj_to_pickle_dict(self.params),
                sort_info=sort_info,
                scope_data_list=scope_data_list,
                texts=texts,
            )
            print("Data write acknowledged by server.")
        except Exception as e:
            print(f"Error sending write_data to server: {e}")

    def _check_data_file_exists(self, raise_error=True) -> bool:
        """Override Scribe._check_data_file_exists to query the server.

        With server-managed files the experiment process can't check the
        HDF5 file's existence on disk (it may not be mounted locally).
        Instead, ask the server whether a reset has been triggered.
        """
        if not self.run_info.save_data:
            return True
        try:
            interrupted = self.live_od_client.check_interrupted()
            if interrupted:
                if raise_error:
                    if hasattr(self, 'monitor'):
                        self.monitor.update_device_states()
                        self.monitor.signal_end()
                    raise RuntimeError(
                        f"Data file for run ID {self.run_info.run_id} not found."
                    )
                return False
            return True
        except RuntimeError:
            raise
        except Exception:
            # Server unreachable — don't interrupt the scan
            return True