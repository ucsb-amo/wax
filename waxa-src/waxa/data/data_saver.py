import numpy as np
import os
import h5py

from waxa.data.server_talk import server_talk as st

# __DEFAULT_KEY = "no_one_will_ever_use_this_key000111"

from waxa.dummy.expt import Expt as DummyExpt

class DataSaver():
    def __init__(self,
                 data_dir="",
                 expt_repo_src_directory="",
                 expt_params_relative_filepath="",
                 base_class_relative_dirpath="",
                 server_talk=None):
        
        self._data_dir = data_dir
        self._expt_repo_path = expt_repo_src_directory
        self._expt_params_path = os.path.join(expt_repo_src_directory,
                                              expt_params_relative_filepath)
        self._base_class_dir = os.path.join(expt_repo_src_directory,
                                            base_class_relative_dirpath)

        if server_talk == None:
            server_talk = st(data_dir=data_dir)
        else:
            server_talk = server_talk
        self.server_talk = server_talk

    def save_data(self,expt:DummyExpt,expt_filepath="",data_object=None):

        # from wax.base.sub.dealer import Dealer
        # expt: Dealer

        if expt.setup_camera:
            
            pwd = os.getcwd()
            os.chdir(self._data_dir)
            
            fpath, _ = self._data_path(expt.run_info)

            if data_object:
                f = data_object
            else:
                f = h5py.File(fpath,'r+')
            
                    
            if expt.sort_idx:
                # these were read in by liveOD, so we replace the expt empty arrays
                expt.images = np.array(f['data']['images'])
                expt.image_timestamps = np.array(f['data']['image_timestamps'])

                # I think these two lines are redundant, should already happen in prepare
                expt.xvardims = [len(xvar.values) for xvar in expt.scan_xvars]
                expt.N_xvars = len(expt.xvardims)

                expt._unshuffle_struct(expt) # this usually does nothing
                # now replace the data from the h5 with the unscrambled data
                f['data']['images'][...] = expt.unscramble_images()
                f['data']['image_timestamps'][...] = expt._unscramble_timestamps()
                expt._unshuffle_struct(expt.params)

            self._save_data_vault(f,expt)
            self._save_scope_data(f,expt)

            del f['params']
            params_dset = f.create_group('params')
            self._class_attr_to_dataset(params_dset,expt.params)

            if expt_filepath:
                f['run_info']['experiment_filepath'][...] = expt_filepath
                f.attrs['experiment_filepath'] = expt_filepath

            self._save_expt_files_text(f,expt_filepath)

            f.close()
            print("Parameters saved, data closed.")
            os.chdir(pwd)

    def get_xvardims(self,expt:DummyExpt):
        return [len(xvar.values) for xvar in expt.scan_xvars]
    
    def pad_sort_idx(self,expt:DummyExpt):
        maxN = np.max(expt.sort_N)
        for i in range(len(expt.sort_idx)):
            N_to_pad = maxN - len(expt.sort_idx[i])
            expt.sort_idx[i] = np.append(expt.sort_idx[i], [-1]*N_to_pad).astype(int)

    def create_data_file(self,expt:DummyExpt):

        pwd = os.getcwd()

        self.server_talk.check_for_mapped_data_dir()
        os.chdir(self._data_dir)

        fpath, folder = self._data_path(expt.run_info)

        if not os.path.exists(folder):
            os.mkdir(folder)

        expt.run_info.filepath = fpath
        expt.run_info.xvarnames = expt.xvarnames

        f = h5py.File(fpath,'w')
        data = f.create_group('data')

        f.attrs['camera_ready'] = 0
        f.attrs['camera_ready_ack'] = 0
        
        f.attrs['xvarnames'] = expt.xvarnames
        data.create_dataset('images',data=expt.images)
        data.create_dataset('image_timestamps',data=expt.image_timestamps)
        for key in expt.data.keys:
            this_data = vars(expt.data)[key]._run_data
            data.create_dataset(key, data=this_data)

        if expt.sort_idx:
            # pad with [-1]s to allow saving in hdf5 (avoid staggered array)
            self.pad_sort_idx(expt)
            data.create_dataset('sort_idx',data=expt.sort_idx)
            data.create_dataset('sort_N',data=expt.sort_N)
        
        # store run info as attrs
        self._class_attr_to_attr(f,expt.run_info)
        # also store run info as dataset
        runinfo_dset = f.create_group('run_info')
        self._class_attr_to_dataset(runinfo_dset,expt.run_info)
        params_dset = f.create_group('params')
        self._class_attr_to_dataset(params_dset,expt.params)
        cam_dset = f.create_group('camera_params')
        self._class_attr_to_dataset(cam_dset,expt.camera_params)
        
        f.close()

        os.chdir(pwd)

        return fpath
        
    def _save_data_vault(self,
                         h5File:h5py.File,
                         expt:DummyExpt):
        f = h5File
        for key in expt.data.keys:
            this_dc = vars(expt.data)[key]
            if this_dc._external_data_bool:
                # overwrite with data from hdf5 in case populated by a process outside expt
                this_data = f['data'][key][...]
            else:
                # otherwise, take the data that was stuck into the array during the expt
                this_data = this_dc._run_data
            if this_dc._external_data_bool or this_dc._data_gotten:
                if expt.sort_idx:
                    # unshuffle if shuffled
                    ndims_per_shot = len(this_data.shape) - len(expt.scan_xvars)
                    this_data = expt._unshuffle_ndarray(this_data,exclude_dims=ndims_per_shot)
                f['data'][key][...] = this_data

    def _save_scope_data(self,
                         h5File:h5py.File,
                         expt:DummyExpt):
        f = h5File
        if expt.scope_data._scope_trace_taken:
            scope_data = f['data'].create_group('scope_data')
            for scope in expt.scope_data.scopes:
                data = scope.reshape_data()
                # data comes out as shape (n0,...,nN,Nch,2,Npts)
                # ni = values for ith xvar
                # Nch = # scope channels the user captured from
                # 2 = axis for picking time or voltage axis
                # Npts = points per scan
                if expt.sort_idx:
                    data = expt._unshuffle_ndarray(data,exclude_dims=3)
                data = data.astype(np.float32)
                this_scope_data = scope_data.create_group(scope.label)
                # time/voltage axis always -2, take the first one for each capture
                # only take one time axis for all the channels on a given shot
                # resulting shape: (n0,...,nN,Npts)
                t = np.take(np.take(data,0,axis=-2),0,axis=-2)
                # take the voltage values
                # resulting shape: (n0,...,nN,Nch,Npts)
                v = np.take(data,1,-2)
                this_scope_data.create_dataset('t', data=t, compression='gzip', compression_opts=4)
                this_scope_data.create_dataset('v', data=v, compression='gzip', compression_opts=4)

    def _save_expt_files_text(self,
                              h5File:h5py.File,
                              expt_filepath):
        
        self._check_for_expt_files()

        f = h5File
        f.attrs["expt_file"] = self._read_text_file_safe(expt_filepath, "experiment") if expt_filepath else ""
        f.attrs["params_file"] = self._read_text_file_safe(self._expt_params_path, "params")
        
        # Save all .py files from the base class directory
        if self._base_class_dir and os.path.isdir(self._base_class_dir):
            try:
                filenames = sorted(os.listdir(self._base_class_dir))
            except Exception as e:
                print(f"Failed to list base class directory {self._base_class_dir}: {e}")
                filenames = []

            for filename in filenames:
                if filename.endswith('.py') and not filename.startswith('__'):
                    filepath = os.path.join(self._base_class_dir, filename)
                    if os.path.isfile(filepath):
                        key = f"base_class_{filename[:-3]}"  # remove .py extension
                        f.attrs[key] = self._read_text_file_safe(filepath, filename)

    def _check_for_expt_files(self):
        if not os.path.isfile(self._expt_params_path):
            print(f'expt_params file not found at {self._expt_params_path}, saving contents skipped')
            self._expt_params_path = ""
        if not os.path.isdir(self._base_class_dir):
            print(f'base class directory not found at {self._base_class_dir}, saving base class files skipped')
            self._base_class_dir = ""

    def _read_text_file_safe(self, filepath, label="file"):
        if not filepath:
            return ""
        if not os.path.isfile(filepath):
            print(f'{label} file not found at {filepath}, saving contents skipped')
            return ""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            print(f'Unable to read {label} file at {filepath}: {e}')
            return ""

    def _class_attr_to_dataset(self,dset,obj):
        try:
            keys = list(vars(obj)) 
            for key in keys:
                if not key.startswith("_"):
                    value = vars(obj)[key]
                    try:
                        dset.create_dataset(key, data=value)
                    except Exception as e:
                        print(f"Failed to save attribute \"{key}\" of {obj}.")
                        print(e)
        except Exception as e:
            print(e)

    def _class_attr_to_attr(self,dset,obj):
        try:
            keys = list(vars(obj))  
            for key in keys:
                value = vars(obj)[key]
                dset.attrs[key] = value
        except Exception as e:
            print(e)

    def _data_path(self,run_info,lite=False):
        this_data_dir = self._data_dir
        run_id_str = f"{str(run_info.run_id).zfill(7)}"
        expt_class = self._bytes_to_str(run_info.expt_class)
        datetime_str = self._bytes_to_str(run_info.run_datetime_str)
        if lite:
            run_id_str += "_lite"
            this_data_dir = os.path.join(self._data_dir,"_lite")
        filename = run_id_str + "_" + datetime_str + "_" + expt_class + ".hdf5"
        filepath_folder = os.path.join(this_data_dir,
                                       self._bytes_to_str(run_info.run_date_str))
        filepath = os.path.join(filepath_folder,filename)
        return filepath, filepath_folder

    def _update_run_id(self,run_info):
        self.server_talk.update_run_id(run_info)

    def _get_rid(self):
        return self.server_talk.get_run_id()
    
    def _bytes_to_str(self,attr):
        if isinstance(attr,bytes):
            attr = attr.decode("utf-8")
        return attr

    # ------------------------------------------------------------------
    # Server-side methods (called by LiveODServer, not the experiment)
    # ------------------------------------------------------------------

    def create_data_file_from_payload(self, payload: dict, run_id: int) -> str:
        """Create an HDF5 data file from an INIT_RUN payload.

        This is the server-side counterpart of ``create_data_file``.  It
        runs on the liveOD machine (which has the data drive mounted) and
        receives all necessary metadata from the experiment client.

        When ``capture_images=False`` the ``images`` / ``image_timestamps``
        datasets are omitted and ``f.attrs['has_images']`` is set to
        ``False`` so that ``atomdata`` can skip image analysis on load.
        """
        # ------ minimal run_info proxy --------------------------------
        class _RunInfoProxy:
            pass

        ri = _RunInfoProxy()
        ri.run_id = run_id
        ri.run_date_str = str(payload.get("run_date_str", ""))
        ri.run_datetime_str = str(payload.get("run_datetime_str", ""))
        ri.expt_class = str(payload.get("expt_class", "expt"))
        ri.imaging_type = int(payload.get("imaging_type", 0))
        ri.save_data = int(payload.get("save_data_flag", 1))
        ri.filepath = []
        ri.xvarnames = list(payload.get("xvarnames", []))
        ri.experiment_filepath = ""

        # ------ path computation & folder creation --------------------
        pwd = os.getcwd()
        self.server_talk.check_for_mapped_data_dir()
        os.chdir(self._data_dir)

        fpath, folder = self._data_path(ri)
        if not os.path.exists(folder):
            os.mkdir(folder)

        capture_images = bool(payload.get("capture_images", False))

        # ------ write HDF5 -------------------------------------------
        f = h5py.File(fpath, "w")
        data_grp = f.create_group("data")

        f.attrs["has_images"] = capture_images
        f.attrs["xvarnames"] = list(payload.get("xvarnames", []))

        # Images pre-allocation
        if capture_images:
            images_shape = tuple(payload.get("images_shape", (0,)))
            images_dtype = str(payload.get("images_dtype", "uint16"))
            ts_shape = tuple(payload.get("image_timestamps_shape", (0,)))
            if images_shape and images_shape[0] > 0:
                data_grp.create_dataset(
                    "images", shape=images_shape, dtype=images_dtype
                )
                data_grp.create_dataset(
                    "image_timestamps", shape=ts_shape, dtype=np.float64
                )

        # DataVault pre-allocation
        for key, info in payload.get("datavault_shapes", {}).items():
            shape = tuple(info["shape"])
            dtype = np.dtype(info["dtype"])
            if shape:
                try:
                    data_grp.create_dataset(key, shape=shape, dtype=dtype)
                except Exception as exc:
                    print(f"[DataSaver] Could not pre-allocate DataVault '{key}': {exc}")

        # Sort index / sort N
        sort_idx_raw = payload.get("sort_idx", [])
        sort_N_raw = payload.get("sort_N", [])
        if sort_idx_raw:
            sort_idx_arrays = [np.array(s) for s in sort_idx_raw]
            maxN = max(len(s) for s in sort_idx_arrays)
            padded = np.full((len(sort_idx_arrays), maxN), -1, dtype=int)
            for i, s in enumerate(sort_idx_arrays):
                padded[i, : len(s)] = s
            data_grp.create_dataset("sort_idx", data=padded)
            data_grp.create_dataset("sort_N", data=np.array(sort_N_raw))

        # run_info group + attrs
        self._run_info_proxy_to_h5(f, ri)

        # params initial snapshot
        params_grp = f.create_group("params")
        for key, val in payload.get("params", {}).items():
            try:
                params_grp.create_dataset(key, data=val)
            except Exception:
                pass

        # camera_params group
        cam_grp = f.create_group("camera_params")
        for key, val in payload.get("camera_params", {}).items():
            try:
                cam_grp.create_dataset(key, data=val)
            except Exception:
                pass

        f.close()
        os.chdir(pwd)
        return fpath

    def save_data_from_payload(self, payload: dict, filepath: str):
        """Write final experiment data to an existing HDF5 file.

        This is the server-side counterpart of ``save_data``.  It is
        called by ``LiveODServer`` after receiving the END_RUN message.
        """
        sort_idx_raw = payload.get("sort_idx", [])
        sort_N_raw = payload.get("sort_N", [])
        capture_images = bool(payload.get("capture_images", False))
        expt_filepath = str(payload.get("expt_filepath", ""))

        pwd = os.getcwd()
        os.chdir(self._data_dir)

        with h5py.File(filepath, "r+") as f:
            # --- unshuffle images if the run was shuffled ---
            if capture_images and sort_idx_raw:
                if "images" in f["data"] and f["data"]["images"].size > 0:
                    images = f["data"]["images"][()]
                    timestamps = f["data"]["image_timestamps"][()]
                    images_ush, timestamps_ush = self._unshuffle_images_from_payload(
                        images, timestamps, payload
                    )
                    f["data"]["images"][...] = images_ush
                    f["data"]["image_timestamps"][...] = timestamps_ush

            # --- DataVault ---
            n_xvars = len(payload.get("xvardims", []))
            for key, dc_info in payload.get("datavault", {}).items():
                this_data = np.asarray(dc_info["data"])
                data_gotten = bool(dc_info["data_gotten"])
                is_external = bool(dc_info["external"])

                if is_external:
                    # Data was written directly to HDF5 by DataHandler
                    if key in f["data"]:
                        this_data = f["data"][key][()]
                    else:
                        continue

                if is_external or data_gotten:
                    if sort_idx_raw:
                        ndims_per_shot = max(0, len(this_data.shape) - n_xvars)
                        this_data = self._unshuffle_single_array(
                            this_data, sort_idx_raw, sort_N_raw,
                            exclude_dims=ndims_per_shot,
                        )
                    if key in f["data"]:
                        f["data"][key][...] = this_data

            # --- scope data ---
            self._save_scope_data_from_payload(f, payload, sort_idx_raw, sort_N_raw)

            # --- final params (overwrite initial snapshot) ---
            # Unshuffle all array-valued params before writing, mirroring
            # what the old save_data() path did via _unshuffle_struct(params).
            _protected_param_keys = {
                'xvarnames', 'sort_idx', 'sort_N', 'images', 'image_timestamps',
                'xvars', 'N_repeats', 'N_shots', 'N_shots_with_repeats',
                'scan_xvars', 'xvardims', 'data',
            }
            del f["params"]
            params_grp = f.create_group("params")
            for key, val in payload.get("params", {}).items():
                try:
                    if sort_idx_raw and key not in _protected_param_keys:
                        try:
                            arr = np.asarray(val)
                            if arr.dtype.kind in ('f', 'i', 'u', 'c') and arr.ndim >= 1:
                                val = self._unshuffle_single_array(
                                    arr, sort_idx_raw, sort_N_raw, exclude_dims=0
                                )
                        except Exception:
                            pass  # leave val unchanged if array conversion fails
                    params_grp.create_dataset(key, data=val)
                except Exception as exc:
                    print(f"[DataSaver] Failed to save param '{key}': {exc}")

            # --- experiment filepath ---
            if expt_filepath:
                f.attrs["experiment_filepath"] = expt_filepath
                if "experiment_filepath" in f["run_info"]:
                    try:
                        f["run_info"]["experiment_filepath"][()] = expt_filepath
                    except Exception:
                        pass

            # --- source file texts ---
            f.attrs["expt_file"] = payload.get("expt_file_text", "")
            f.attrs["params_file"] = payload.get("params_file_text", "")
            for key, text in payload.get("base_class_texts", {}).items():
                f.attrs[key] = text

        print("[DataSaver] Parameters saved, data closed.")
        os.chdir(pwd)

    # ------------------------------------------------------------------
    # Private helpers shared by server-side methods
    # ------------------------------------------------------------------

    def _run_info_proxy_to_h5(self, f: "h5py.File", ri) -> None:
        """Write run_info fields to HDF5 attrs and run_info group."""
        for key in vars(ri):
            try:
                f.attrs[key] = getattr(ri, key)
            except Exception:
                pass
        runinfo_grp = f.create_group("run_info")
        for key in vars(ri):
            try:
                runinfo_grp.create_dataset(key, data=getattr(ri, key))
            except Exception:
                pass

    @staticmethod
    def _unshuffle_single_array(
        arr: np.ndarray,
        sort_idx_raw: list,
        sort_N_raw: list,
        exclude_dims: int = 0,
    ) -> np.ndarray:
        """Unshuffle a single ndarray using sort metadata lists.

        Replicates the core logic of ``Dealer._unshuffle_ndarray`` as a
        standalone function so the server does not need a live ``Dealer``
        instance.
        """
        if not isinstance(arr, np.ndarray) or not sort_idx_raw:
            return arr
        sort_idx = [np.array(s) for s in sort_idx_raw]
        sort_N = [int(n) for n in sort_N_raw]
        ndims = arr.ndim
        last_dim = max(0, ndims - exclude_dims)
        for dim in range(last_dim):
            N = arr.shape[dim]
            if N in sort_N:
                i = sort_N.index(N)
                shuf = sort_idx[i].copy()
                shuf = shuf[shuf >= 0].astype(int)
                unshuf = np.zeros_like(shuf)
                unshuf[shuf] = np.arange(len(shuf))
                arr = arr.take(unshuf, axis=dim)
        return arr

    @staticmethod
    def _unshuffle_images_from_payload(
        images: np.ndarray,
        image_timestamps: np.ndarray,
        payload: dict,
    ):
        """Unshuffle images and timestamps acquired in shuffled xvar order.

        Mirrors the logic of ``Dealer.unscramble_images`` /
        ``_unscramble_timestamps`` using only the metadata in *payload*.
        """
        sort_idx_raw = payload.get("sort_idx", [])
        sort_N_raw = payload.get("sort_N", [])
        if not sort_idx_raw:
            return images, image_timestamps

        sort_idx = [np.array(s) for s in sort_idx_raw]
        sort_N = [int(n) for n in sort_N_raw]
        N_shots = int(payload["N_shots_with_repeats"])
        Nps = int(payload["N_pwa_per_shot"])
        xvardims = list(payload.get("xvardims", [N_shots]))

        def _unshuffle(arr, exclude_dims=0):
            ndims = arr.ndim
            last_dim = max(0, ndims - exclude_dims)
            for dim in range(last_dim):
                N = arr.shape[dim]
                if N in sort_N:
                    i = sort_N.index(N)
                    shuf = sort_idx[i].copy()
                    shuf = shuf[shuf >= 0].astype(int)
                    unshuf = np.zeros_like(shuf)
                    unshuf[shuf] = np.arange(len(shuf))
                    arr = arr.take(unshuf, axis=dim)
            return arr

        img_shape = images.shape[1:]   # (H, W)
        N_img = images.shape[0]        # = N_shots * (Nps + 2)

        # Split into pwa / pwoa / dark  →  shape (N_shots, Nps+2, H, W)
        imgs_4d = images.reshape(N_shots, Nps + 2, *img_shape)
        pwa  = imgs_4d[:, 0:Nps]
        pwoa = np.expand_dims(imgs_4d[:, Nps],     1).repeat(Nps, axis=1)
        dark = np.expand_dims(imgs_4d[:, Nps + 1], 1).repeat(Nps, axis=1)

        # Reshape to (*xvardims, Nps, H, W) then unshuffle xvar axes
        pwa  = _unshuffle(pwa.reshape(*xvardims,  Nps, *img_shape), exclude_dims=3)
        pwoa = _unshuffle(pwoa.reshape(*xvardims, Nps, *img_shape), exclude_dims=3)
        dark = _unshuffle(dark.reshape(*xvardims, Nps, *img_shape), exclude_dims=3)

        # Stack back to (N_img, H, W)
        pwa  = pwa.reshape(N_shots,  Nps, *img_shape)
        pwoa = pwoa.reshape(N_shots, Nps, *img_shape)
        dark = dark.reshape(N_shots, Nps, *img_shape)
        out  = np.empty((N_img,) + img_shape, dtype=images.dtype)
        for shot_i in range(N_shots):
            base = shot_i * (Nps + 2)
            out[base : base + Nps] = pwa[shot_i]
            out[base + Nps]        = pwoa[shot_i, 0]
            out[base + Nps + 1]    = dark[shot_i, 0]

        # Timestamps  (shape: N_img → scalars per image)
        ts_4d  = image_timestamps.reshape(N_shots, Nps + 2)
        ts_pwa  = _unshuffle(ts_4d[:, 0:Nps].reshape(*xvardims, Nps))
        ts_pwoa = _unshuffle(
            np.expand_dims(ts_4d[:, Nps], 1).repeat(Nps, axis=1).reshape(*xvardims, Nps)
        )
        ts_dark = _unshuffle(
            np.expand_dims(ts_4d[:, Nps + 1], 1).repeat(Nps, axis=1).reshape(*xvardims, Nps)
        )

        ts_pwa  = ts_pwa.reshape(N_shots,  Nps)
        ts_pwoa = ts_pwoa.reshape(N_shots, Nps)
        ts_dark = ts_dark.reshape(N_shots, Nps)
        ts_out  = np.empty(N_img, dtype=image_timestamps.dtype)
        for shot_i in range(N_shots):
            base = shot_i * (Nps + 2)
            ts_out[base : base + Nps] = ts_pwa[shot_i]
            ts_out[base + Nps]        = ts_pwoa[shot_i, 0]
            ts_out[base + Nps + 1]    = ts_dark[shot_i, 0]

        return out, ts_out

    def _save_scope_data_from_payload(
        self,
        f: "h5py.File",
        payload: dict,
        sort_idx_raw: list,
        sort_N_raw: list,
    ) -> None:
        """Save scope data from END_RUN payload into an open HDF5 file."""
        if not payload.get("scope_data_taken", False):
            return
        scope_data_grp = f["data"].create_group("scope_data")
        for scope_info in payload.get("scope_data", []):
            label = str(scope_info["label"])
            data = np.asarray(scope_info["data"])
            if sort_idx_raw:
                data = self._unshuffle_single_array(
                    data, sort_idx_raw, sort_N_raw, exclude_dims=3
                )
            data = data.astype(np.float32)
            this_scope = scope_data_grp.create_group(label)
            t = np.take(np.take(data, 0, axis=-2), 0, axis=-2)
            v = np.take(data, 1, axis=-2)
            this_scope.create_dataset("t", data=t, compression='gzip', compression_opts=4)
            this_scope.create_dataset("v", data=v, compression='gzip', compression_opts=4)