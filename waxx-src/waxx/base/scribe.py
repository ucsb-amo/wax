import os
import numpy as np

from waxa.config.expt_params import ExptParams
from waxa.data import DataSaver, RunInfo
from waxx.config.data_vault import DataVault, DataContainer
from waxx.control.misc.oscilloscopes import ScopeData
from waxx.util.live_od.camera_client import CameraClient

class Scribe:
    """Server-side run-data transport helpers for Expt.

    The camera server owns file creation and final HDF5 writes. This mixin
    keeps experiment-side code focused on scan/control flow while still
    providing compatibility shims for older experiment `analyze()` methods.
    """

    def __init__(self):
        # placeholders
        self.ds = DataSaver()
        self.run_info = RunInfo()
        self.data = DataVault(expt=self)
        self.scope_data = ScopeData()
        self.live_od_client = CameraClient(None, None)

        self.xvarnames = []
        self.params = ExptParams()
        self.p = self.params

        self.xvarnames = []
        self.sort_idx = []
        self.sort_N = []

    def _setup_fpath(self):
        if self.run_info.save_data:
            # Compute the data filepath; the camera server creates the HDF5
            # file on the server side (via the data_spec sent in new_run), so
            # the experiment process never opens the data directory directly.
            fpath, _ = self.ds._data_path(self.run_info)
            self.run_info.filepath = fpath
            self.data_filepath     = fpath
            self.run_info.xvarnames = self.xvarnames


    def _build_data_spec(self):
        """Build the file-creation spec dict to send to the camera server."""
        containers = []
        for key in self.data.keys:
            dc = vars(self.data)[key]
            dc: DataContainer
            containers.append({
                "key": key,
                "shape": list(dc._run_data.shape),
                "dtype": dc._run_data.dtype.str,
                "external": dc._external_data_bool,
            })
        sort_idx = [s.tolist() for s in self.sort_idx] if self.sort_idx else []
        sort_N = [int(n) for n in self.sort_N] if self.sort_N else []
        return {
            "xvarnames": list(self.xvarnames),
            "xvardims": list(self.xvardims),
            "containers": containers,
            "sort_idx": sort_idx,
            "sort_N": sort_N,
        }

    def _available_data_field_names(self):
        names = []
        for key in self.data.keys:
            dc = vars(self.data)[key]
            dc: DataContainer
            if dc._external_data_bool:
                continue
            names.append(str(key))
        return names

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
        """Return the text content of path; empty string on any failure."""
        if path and os.path.isfile(path):
            try:
                with open(path) as f:
                    return f.read()
            except Exception:
                pass
        return ""

    def _send_write_data_to_server(self, expt_filepath):
        """Collect all run data and send it to the camera server for writing."""
        data = {}
        for key in self.data.keys:
            dc = vars(self.data)[key] 
            dc: DataContainer
            if dc._external_data_bool:
                continue
            if dc._data_gotten:
                data[key] = dc._run_data

        sort_info = {
            "has_sort": bool(self.sort_idx),
            "sort_idx": [s.tolist() for s in self.sort_idx] if self.sort_idx else [],
            "sort_N": [int(n) for n in self.sort_N] if self.sort_N else [],
            "xvardims": list(getattr(self, "xvardims", [])),
            "N_pwa_per_shot": int(self.params.N_pwa_per_shot),
            "N_shots_with_repeats": int(
                getattr(
                    self.params,
                    "N_shots_with_repeats",
                    int(np.prod(self.xvardims) if self.xvardims else 1),
                )
            ),
        }

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

        texts = {
            "expt": self._read_text_file(expt_filepath),
            "params": self._read_text_file(getattr(self.ds, "_expt_params_path", "")),
            "cooling": self._read_text_file(getattr(self.ds, "_cooling_path", "")),
            "imaging": self._read_text_file(getattr(self.ds, "_imaging_path", "")),
            "control": self._read_text_file(getattr(self.ds, "_control_path", "")),
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

    def write_data(self, expt_filepath):
        """Compatibility wrapper for older experiments.

        Old code called `self.write_data(...)` from `analyze()`. That now means
        sending final run data to the camera server for HDF5 writing.
        """
        self._send_write_data_to_server(expt_filepath)

    def remove_incomplete_data(self, delete_data_bool=True):
        """Compatibility no-op for legacy experiments.

        In the server-owned data path there is no experiment-side file to remove.
        Reset/deletion decisions are handled by the camera server.
        """
        return None

    def _check_if_interrupted(self, raise_error=True) -> bool:
        """Check run liveness by polling the camera server for interruption."""
        try:
            interrupted = self.live_od_client.check_interrupted()
            if interrupted:
                if raise_error:
                    if hasattr(self, "monitor"):
                        self.monitor.update_device_states()
                        self.monitor.signal_end()
                    raise RuntimeError(
                        f"Server reports run {self.run_info.run_id} interrupted."
                    )
                return False
            return True
        except RuntimeError:
            raise
        except Exception as e:
            print(e)
            return True

    def send_new_run(self):
        self.live_od_client.connect()
        data_spec = {}
        run_info_attrs = {}
        params_attrs = {}
        camera_params_attrs = {}
        if self.run_info.save_data:
            self._setup_fpath()
            data_spec          = self._build_data_spec()
            run_info_attrs     = self._obj_to_pickle_dict(self.run_info)
            params_attrs       = self._obj_to_pickle_dict(self.params)
            camera_params_attrs = self._obj_to_pickle_dict(self.camera_params)
        try:
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
                available_data_fields=self._available_data_field_names(),
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "duplicate run_id" in msg or "duplicate of a previous run" in msg:
                print("run ID is a duplicate of a previous run")
                raise RuntimeError("run ID is a duplicate of a previous run") from None
            raise