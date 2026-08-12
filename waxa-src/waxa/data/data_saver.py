import numpy as np
import os
import pickle
import time
import h5py

from waxa.data.server_talk import server_talk as st

# __DEFAULT_KEY = "no_one_will_ever_use_this_key000111"

from waxa.dummy.expt import Expt as DummyExpt

# --- END_RUN save resilience ------------------------------------------
# The data drive is a network share.  A transient SMB drop surfaces from
# h5py as OSError (errno 22, "Invalid argument" — Windows' catch-all for
# ERROR_NETNAME_DELETED) or as RuntimeError from a failed H5Fclose, and is
# usually over within seconds.  Delays (s) between successive attempts:
SAVE_RETRY_DELAYS_S = (2.0, 5.0, 15.0)

# Exceptions worth retrying an end-of-run save for.  h5py reports I/O
# failures as OSError and id/close failures as RuntimeError.
_RETRYABLE_SAVE_EXC = (OSError, RuntimeError)

# Where the END_RUN payload is stashed while the save is in flight.  Local
# disk on purpose: the failure mode being guarded against is the data drive
# going away, so the stash must not live on the data drive.
PENDING_SAVE_DIRNAME = "pending_saves"

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

    def reserve_run_id_and_path(self, payload: dict):
        """Atomically reserve a unique run_id and return ``(run_id, filepath)``.

        The claim is made by creating the HDF5 file in exclusive mode (``'x'``
        — create, fail if exists).  Because exclusive creation is atomic on the
        (shared) filesystem, two liveOD servers driving different hardware but
        writing to the same data drive can never obtain the same run_id: only
        one ``'x'`` create wins, and the loser retries with the next id.

        The starting candidate is seeded from ``max(counter, on-disk max)``, so
        the run_id counter file acts only as a fast monotonic floor (preventing
        id reuse after a reset deletes an in-progress file) while the filesystem
        is the true source of truth.

        The file is populated *completely* (``data`` / ``run_info`` / ``params``
        / ``camera_params``) inside the same exclusive open that claims the id.
        This matters: an earlier design left a bare stub here and re-opened the
        path with a truncating ``'w'`` on a background thread.  That second open
        could lose the HDF5 file lock to any concurrent reader of the newest
        data file (``server_talk._is_completed_run``), and the failure was
        invisible, so the ``data`` group was silently never created and every
        image of the run was dropped.  One open, no stub window, no background
        thread.
        """
        st = self.server_talk
        st.check_for_mapped_data_dir()

        try:
            counter = int(st.get_run_id())
        except Exception:
            counter = 0
        fs_max = st.get_latest_run_id_any()
        candidate = max(counter, (fs_max + 1) if fs_max is not None else 0, 1)

        class _RunInfoProxy:
            pass
        ri = _RunInfoProxy()
        ri.run_date_str = str(payload.get("run_date_str", ""))
        ri.run_datetime_str = str(payload.get("run_datetime_str", ""))
        ri.expt_class = str(payload.get("expt_class", "expt"))

        while True:
            ri.run_id = candidate
            fpath, folder = self._data_path(ri)
            os.makedirs(folder, exist_ok=True)
            try:
                f = h5py.File(fpath, "x")
            except (FileExistsError, OSError):
                if os.path.exists(fpath):
                    # Another run already claimed this id — try the next one.
                    candidate += 1
                    continue
                raise
            # The id is ours.  Fill in the full structure before releasing the
            # handle so no other process can ever observe a file without a
            # 'data' group.  If that fails, delete the file so the id is not
            # left claimed by an unusable corpse, and let the caller refuse to
            # start the run.
            try:
                self._populate_data_file(f, payload, candidate)
            except Exception:
                try:
                    f.close()
                except Exception:
                    pass
                try:
                    os.remove(fpath)
                except Exception:
                    pass
                raise
            else:
                f.close()
            break

        # Advance the monotonic floor to the next id.
        st.set_run_id(candidate + 1)
        return candidate, fpath

    def compute_data_filepath_from_payload(self, payload: dict, run_id: int) -> str:
        """Return the HDF5 file path for a run without creating the file.

        Pure string computation — no I/O, no network calls.  Safe to call
        on any thread including the ZMQ REP handler thread.
        """
        class _RunInfoProxy:
            pass
        ri = _RunInfoProxy()
        ri.run_id = run_id
        ri.run_date_str = str(payload.get("run_date_str", ""))
        ri.run_datetime_str = str(payload.get("run_datetime_str", ""))
        ri.expt_class = str(payload.get("expt_class", "expt"))
        fpath, _ = self._data_path(ri)
        return fpath

    def create_data_file_from_payload(self, payload: dict, run_id: int) -> str:
        """Create an HDF5 data file from an INIT_RUN payload.

        Legacy entry point, kept for callers that need to (re)create a data file
        at a known run_id without going through ``reserve_run_id_and_path``.
        The liveOD server no longer uses it: file creation now happens inside
        the exclusive open that reserves the run_id, so there is no window in
        which the file exists without its ``data`` group.
        """
        self.server_talk.check_for_mapped_data_dir()
        fpath = self.compute_data_filepath_from_payload(payload, run_id)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        f = h5py.File(fpath, "w")
        try:
            self._populate_data_file(f, payload, run_id)
        finally:
            f.close()
        return fpath

    def _populate_data_file(self, f: "h5py.File", payload: dict, run_id: int) -> None:
        """Write the full run structure into an already-open HDF5 handle.

        Creates the ``data``, ``run_info``, ``params`` and ``camera_params``
        groups plus the top-level attrs.  The caller owns *f* and is
        responsible for closing it.

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
        ri.save_on_underflow = int(payload.get("save_on_underflow", 0))
        ri.filepath = []
        ri.xvarnames = list(payload.get("xvarnames", []))
        ri.experiment_filepath = ""

        capture_images = bool(payload.get("capture_images", False))

        # ------ write HDF5 -------------------------------------------
        data_grp = f.create_group("data")

        f.attrs["has_images"] = capture_images
        f.attrs["xvarnames"] = list(payload.get("xvarnames", []))
        f.attrs["run_complete"] = False

        # Images pre-allocation is intentionally deferred to SaveWorker.
        # Pre-allocating a large dataset here (e.g. 300 × 1024 × 1024 × 2 B ≈ 600 MB on a NAS)
        # would hold the file open — and now block the INIT_RUN reply — for many seconds.
        # SaveWorker creates the images/image_timestamps datasets lazily on its first write,
        # after the file is already usable.
        # Store shape/dtype as HDF5 attributes so the lazy path can use them if needed.
        if capture_images:
            images_shape = tuple(payload.get("images_shape", (0,)))
            images_dtype = str(payload.get("images_dtype", "uint16"))
            ts_shape = tuple(payload.get("image_timestamps_shape", (0,)))
            f.attrs["images_shape"] = list(images_shape)
            f.attrs["images_dtype"] = images_dtype

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

    # Params that must never be unshuffled — they describe the scan itself
    # rather than per-shot results.
    _PROTECTED_PARAM_KEYS = {
        'xvarnames', 'sort_idx', 'sort_N', 'images', 'image_timestamps',
        'xvars', 'N_repeats', 'N_shots', 'N_shots_with_repeats',
        'scan_xvars', 'xvardims', 'data',
    }

    def save_data_from_payload(self, payload: dict, filepath: str, shot_timestamps=None):
        """Write final experiment data to an existing HDF5 file.

        This is the server-side counterpart of ``save_data``.  It is
        called by ``LiveODServer`` after receiving the END_RUN message.

        The work is split into three phases so a transient failure of the
        (network) data drive can be retried without corrupting the file:

        1. **read** — pull the values that have to come *from* the file
           (images, timestamps, externally-written DataVault arrays).
           Retriable because nothing has been written yet.
        2. **compute** — unshuffle everything in memory.  No I/O.
        3. **write** — write the results back.  Retriable because every
           write is derived from phase 2's in-memory values and never from
           a re-read of a possibly half-written file.

        The 100s-of-MB in-place image rewrite is deliberately the last thing
        written before ``run_complete``, so a drop during that long window
        still leaves a file whose params, source texts and DataVault are
        already final.

        Parameters
        ----------
        shot_timestamps:
            Optional list/array of Unix timestamps (one per shot) recorded
            server-side.  When provided they are reshaped, unshuffled, and
            written as ``data/timestamp_shot_end`` — before ``run_complete``
            is set to ``True``.
        """
        inputs = self._retry_io(
            lambda: self._read_end_run_inputs(filepath, payload),
            filepath, "end-of-run read",
        )
        outputs = self._compute_end_run_outputs(payload, inputs, shot_timestamps)
        self._retry_io(
            lambda: self._write_end_run_outputs(filepath, payload, outputs),
            filepath, "end-of-run write",
        )
        print("[DataSaver] Parameters saved, data closed.")

    # ------------------------------------------------------------------
    # End-of-run save: read / compute / write phases
    # ------------------------------------------------------------------

    def _read_end_run_inputs(self, filepath: str, payload: dict) -> dict:
        """Phase 1: read everything the end-of-run save needs *from* the file.

        Kept separate from the write phase so that a retry never re-reads
        data it may itself have partially overwritten.
        """
        sort_idx_raw = payload.get("sort_idx", [])
        capture_images = bool(payload.get("capture_images", False))

        images = None
        image_timestamps = None
        external = {}

        # filepath is absolute; do not os.chdir (process-global — it would race
        # with any other thread in this process).
        with h5py.File(filepath, "r") as f:
            # A file without a 'data' group means creation never completed —
            # fail with a diagnostic rather than a bare KeyError from every
            # f["data"] access below.
            if "data" not in f:
                raise ValueError(
                    f"Data file {filepath} has no 'data' group — file creation "
                    "never completed, so no run data can be saved."
                )

            applied = bool(f.attrs.get("unshuffle_applied", False))
            torn = bool(f.attrs.get("unshuffle_in_progress", False))

            # Only the shuffled case needs any of this: without sort_idx the
            # file-resident arrays are already in final order, and reading them
            # back just to write them again would be pure risk.
            if sort_idx_raw and not applied and not torn:
                if capture_images and "images" in f["data"] and f["data"]["images"].size > 0:
                    images = f["data"]["images"][()]
                    image_timestamps = f["data"]["image_timestamps"][()]

                for key, dc_info in payload.get("datavault", {}).items():
                    # Data written directly to HDF5 by DataHandler.
                    if bool(dc_info.get("external")) and key in f["data"]:
                        external[key] = f["data"][key][()]

        if torn:
            print(
                f"[DataSaver] WARNING: 'unshuffle_in_progress' is set on {filepath} "
                "— a previous save died partway through the in-place rewrite, so shot "
                "ordering of data/images and of externally-written DataVault arrays is "
                "UNRELIABLE.  Leaving them untouched and keeping the flag set."
            )

        return {
            "images": images,
            "image_timestamps": image_timestamps,
            "external": external,
            "torn": torn,
        }

    def _compute_end_run_outputs(self, payload: dict, inputs: dict, shot_timestamps) -> dict:
        """Phase 2: unshuffle everything in memory.  No I/O, so nothing here
        can fail because of the network."""
        sort_idx_raw = payload.get("sort_idx", [])
        sort_N_raw = payload.get("sort_N", [])
        n_xvars = len(payload.get("xvardims", []))

        # --- images ---
        images = inputs["images"]
        image_timestamps = inputs["image_timestamps"]
        if images is not None:
            images, image_timestamps = self._unshuffle_images_from_payload(
                images, image_timestamps, payload
            )

        # --- DataVault ---
        # Split by provenance: payload-derived arrays can be rewritten any
        # number of times, whereas externally-written ones are read back out
        # of the file and so must only ever be unshuffled once (they go in the
        # bracketed section of the write phase alongside the images).
        datavault = {}
        datavault_external = {}
        for key, dc_info in payload.get("datavault", {}).items():
            if bool(dc_info.get("external")):
                if key not in inputs["external"]:
                    continue
                this_data = inputs["external"][key]
                sink = datavault_external
            elif bool(dc_info.get("data_gotten")):
                this_data = np.asarray(dc_info["data"])
                sink = datavault
            else:
                continue

            if sort_idx_raw:
                ndims_per_shot = max(0, len(this_data.shape) - n_xvars)
                this_data = self._unshuffle_single_array(
                    this_data, sort_idx_raw, sort_N_raw,
                    exclude_dims=ndims_per_shot,
                )
            sink[key] = this_data

        # --- final params (overwrite initial snapshot) ---
        # Unshuffle all array-valued params, mirroring what the old
        # save_data() path did via _unshuffle_struct(params).
        params = {}
        for key, val in payload.get("params", {}).items():
            if sort_idx_raw and key not in self._PROTECTED_PARAM_KEYS:
                try:
                    arr = np.asarray(val)
                    if arr.dtype.kind in ('f', 'i', 'u', 'c') and arr.ndim >= 1:
                        val = self._unshuffle_single_array(
                            arr, sort_idx_raw, sort_N_raw, exclude_dims=0
                        )
                except Exception:
                    pass  # leave val unchanged if array conversion fails
            params[key] = val

        # --- shot timestamps (server-side, one per shot) ---
        ts_shot_end = None
        if shot_timestamps:
            ts_shot_end = np.array(shot_timestamps, dtype=np.float64)
            xvardims = list(payload.get("xvardims", []))
            if xvardims and int(np.prod(xvardims)) == len(ts_shot_end):
                ts_shot_end = ts_shot_end.reshape(xvardims)
            if sort_idx_raw:
                ts_shot_end = self._unshuffle_single_array(
                    ts_shot_end, sort_idx_raw, sort_N_raw, exclude_dims=0
                )

        return {
            "images": images,
            "image_timestamps": image_timestamps,
            "torn": inputs["torn"],
            "datavault": datavault,
            "datavault_external": datavault_external,
            "params": params,
            "scope": self._compute_scope_data_from_payload(payload, sort_idx_raw, sort_N_raw),
            "timestamp_shot_end": ts_shot_end,
        }

    def _write_end_run_outputs(self, filepath: str, payload: dict, out: dict) -> None:
        """Phase 3: write phase-2 results.  Safe to re-run after a failure —
        every value written comes from *out*, never from the file itself."""
        expt_filepath = str(payload.get("expt_filepath", ""))

        with h5py.File(filepath, "r+") as f:
            # --- small, cheap metadata first ---
            # All of this is kilobytes and lands in well under a second, so a
            # drop during the big image write below cannot take it down too.

            # final params (overwrite initial snapshot)
            if "params" in f:
                del f["params"]
            params_grp = f.create_group("params")
            for key, val in out["params"].items():
                try:
                    params_grp.create_dataset(key, data=val)
                except Exception as exc:
                    print(f"[DataSaver] Failed to save param '{key}': {exc}")

            # experiment filepath
            if expt_filepath:
                f.attrs["experiment_filepath"] = expt_filepath
                if "experiment_filepath" in f["run_info"]:
                    try:
                        f["run_info"]["experiment_filepath"][()] = expt_filepath
                    except Exception:
                        pass

            # source file texts
            f.attrs["expt_file"] = payload.get("expt_file_text", "")
            f.attrs["params_file"] = payload.get("params_file_text", "")
            for key, text in payload.get("base_class_texts", {}).items():
                f.attrs[key] = text

            # DataVault (payload-derived — safe to rewrite any number of times)
            for key, arr in out["datavault"].items():
                if key in f["data"]:
                    f["data"][key][...] = arr

            # scope data
            self._write_scope_data(f, out["scope"])

            # shot timestamps
            if out["timestamp_shot_end"] is not None:
                grp = f["data"]
                if "timestamp_shot_end" in grp:
                    del grp["timestamp_shot_end"]
                grp.create_dataset("timestamp_shot_end", data=out["timestamp_shot_end"])

            # --- in-place rewrite of file-resident arrays, last ---
            # These are the only writes that cannot be reconstructed from the
            # payload: their source is the file itself, so applying them twice
            # would shuffle the data rather than unshuffle it.  They are done
            # together, after everything else, bracketed by a flushed marker —
            # so a process death mid-write leaves the file explicitly flagged
            # as having an unknown shot order rather than silently mixing
            # shuffled and unshuffled shots.  The images dominate this window
            # (100s of MB); the DataVault arrays are along for the ride.
            if out["images"] is not None or out["datavault_external"]:
                f.attrs["unshuffle_in_progress"] = True
                f.flush()
                for key, arr in out["datavault_external"].items():
                    if key in f["data"]:
                        f["data"][key][...] = arr
                if out["images"] is not None:
                    f["data"]["images"][...] = out["images"]
                    f["data"]["image_timestamps"][...] = out["image_timestamps"]
                f.attrs["unshuffle_in_progress"] = False

            # True ⇒ every file-resident array is in final (unshuffled) order.
            # Also set for runs that were never shuffled, where it holds
            # trivially — so the attr always answers the question.
            if not out["torn"]:
                f.attrs["unshuffle_applied"] = True

            # --- mark file as fully written ---
            f.attrs["run_complete"] = True

    # ------------------------------------------------------------------
    # End-of-run save: retry plumbing
    # ------------------------------------------------------------------

    def _retry_io(self, fn, filepath: str, what: str):
        """Call *fn*, retrying on network-flavoured HDF5 failures.

        Between attempts the data dir is re-checked (which re-runs the
        drive-mapping batch file on Windows — the right remediation for a
        dropped share) and any HDF5 file id left behind by the failed close
        is dropped.
        """
        attempts = 1 + len(SAVE_RETRY_DELAYS_S)
        for attempt in range(attempts):
            if attempt:
                time.sleep(SAVE_RETRY_DELAYS_S[attempt - 1])
            try:
                return fn()
            except _RETRYABLE_SAVE_EXC as exc:
                print(f"[DataSaver] {what} failed (attempt {attempt + 1}/{attempts}): {exc}")
                if attempt == attempts - 1:
                    raise
                # A dropped SMB handle cannot be reused — drop the dead id
                # before reopening, then try to bring the share back.
                self._force_close_stale_handles(filepath)
                try:
                    self.server_talk.check_for_mapped_data_dir()
                except Exception as map_exc:
                    print(f"[DataSaver] ... could not re-check the data dir: {map_exc}")

    @staticmethod
    def _force_close_stale_handles(filepath: str) -> None:
        """Drop HDF5 file ids left open for *filepath* by a failed close.

        When a share disappears mid-write ``H5Fclose`` fails and h5py can keep
        the (now dead) file id registered, so the next open of the same path
        fails with a lock error caused by this process itself.  Safe to call
        here because the server only saves after DataHandler is done with the
        file.  Best-effort: never raises.
        """
        try:
            from h5py import h5f
            target = os.path.normcase(os.path.abspath(filepath))
            for fid in h5f.get_obj_ids(h5f.OBJ_ALL, h5f.OBJ_FILE):
                try:
                    name = h5f.get_name(fid)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="replace")
                    if os.path.normcase(os.path.abspath(name)) != target:
                        continue
                    while fid.valid:
                        fid.close()
                except Exception:
                    continue
        except Exception:
            pass

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

    def _compute_scope_data_from_payload(
        self,
        payload: dict,
        sort_idx_raw: list,
        sort_N_raw: list,
    ) -> list:
        """Unshuffle and downcast scope traces from the END_RUN payload.

        Pure computation — returns ``[(label, t, v), ...]`` for
        :meth:`_write_scope_data` to write.
        """
        if not payload.get("scope_data_taken", False):
            return []
        traces = []
        for scope_info in payload.get("scope_data", []):
            label = str(scope_info["label"])
            data = np.asarray(scope_info["data"])
            if data.ndim < 3 or data.size == 0:
                print(f"[DataSaver] WARNING: skipping scope '{label}' — data has unexpected shape {data.shape}")
                continue
            if sort_idx_raw:
                data = self._unshuffle_single_array(
                    data, sort_idx_raw, sort_N_raw, exclude_dims=3
                )
            data = data.astype(np.float32)
            t = np.take(np.take(data, 0, axis=-2), 0, axis=-2)
            v = np.take(data, 1, axis=-2)
            traces.append((label, t, v))
        return traces

    @staticmethod
    def _write_scope_data(f: "h5py.File", scope_traces: list) -> None:
        """Write pre-computed scope traces into an open HDF5 file.

        Re-creates the group, so a retry after a partially written attempt
        starts from a clean slate instead of failing on an existing name.
        """
        if not scope_traces:
            return
        if "scope_data" in f["data"]:
            del f["data"]["scope_data"]
        scope_data_grp = f["data"].create_group("scope_data")
        for label, t, v in scope_traces:
            this_scope = scope_data_grp.create_group(label)
            this_scope.create_dataset("t", data=t, compression='gzip', compression_opts=4)
            this_scope.create_dataset("v", data=v, compression='gzip', compression_opts=4)


# ----------------------------------------------------------------------
# Pending-save stash
# ----------------------------------------------------------------------
# The END_RUN payload is the only copy of the run's final params: the
# experiment sends it once and drops it.  If the end-of-run save fails (the
# data drive is a network share and does occasionally vanish mid-write) the
# payload would be lost and the run permanently stuck with its INIT_RUN
# params snapshot.  Stashing it locally first turns that unrecoverable loss
# into a deferred retry via retry_pending_save().


def pending_save_dir() -> str:
    """Return (creating if needed) the local directory holding stashed payloads."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "waxx", PENDING_SAVE_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def stash_end_run_payload(payload: dict, filepath: str, run_id, shot_timestamps=None) -> str:
    """Pickle an END_RUN payload to local disk before the save is attempted.

    Returns the stash path, or ``""`` if stashing failed — a stash failure
    must never prevent the save itself from being attempted.
    """
    try:
        stash_path = os.path.join(pending_save_dir(), f"{int(run_id):07d}_endrun.pkl")
        with open(stash_path, "wb") as fh:
            pickle.dump(
                {
                    "run_id": int(run_id),
                    "filepath": str(filepath),
                    "payload": payload,
                    "shot_timestamps": list(shot_timestamps or []),
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return stash_path
    except Exception as exc:
        print(f"[DataSaver] WARNING: could not stash END_RUN payload: {exc}")
        return ""


def clear_end_run_payload(stash_path: str) -> None:
    """Delete a stashed payload after its save succeeded.  Never raises."""
    if not stash_path:
        return
    try:
        os.remove(stash_path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[DataSaver] WARNING: could not remove stashed payload {stash_path}: {exc}")


def list_pending_saves() -> list:
    """Return the paths of all stashed END_RUN payloads, oldest run first."""
    try:
        d = pending_save_dir()
        names = [n for n in os.listdir(d) if n.endswith("_endrun.pkl")]
    except Exception:
        return []
    return [os.path.join(d, n) for n in sorted(names)]


def retry_pending_save(stash_path: str, data_saver=None) -> bool:
    """Re-run the end-of-run save for a stashed payload.

    Use after a save failed (e.g. the data drive dropped out) once the drive
    is back.  Removes the stash on success.  Returns True if the save
    completed.
    """
    with open(stash_path, "rb") as fh:
        stashed = pickle.load(fh)

    filepath = stashed["filepath"]
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file for run {stashed['run_id']} is gone: {filepath}"
        )

    if data_saver is None:
        data_saver = DataSaver(data_dir=os.getenv("data") or "")

    data_saver.save_data_from_payload(
        stashed["payload"], filepath,
        shot_timestamps=stashed.get("shot_timestamps") or None,
    )
    clear_end_run_payload(stash_path)
    print(f"[DataSaver] Pending save for run {stashed['run_id']} completed.")
    return True