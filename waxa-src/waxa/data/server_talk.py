import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
import glob
import numpy as np
import h5py

MAP_BAT_PATH = "\"G:\\Shared drives\\Weld Lab Shared Drive\\Infrastructure\\map_network_drives.bat\""
RECENT_COMPLETED_TRUST_WINDOW = 0
SERVER_TALK_TIMING_ENABLED = False

class server_talk():
    def __init__(self,
                 data_dir=os.getenv("data"),
                 run_id_relpath="run_id.py",
                 roi_spreadsheet_replath="roi.xlsx",
                 first_data_folder_date="",
                 on_data_dir_disconnected_bat_path=""):
        
        self.data_dir = data_dir
        self.run_id_path = os.path.join(data_dir, run_id_relpath) if data_dir is not None else None
        self.roi_csv_path = os.path.join(data_dir, roi_spreadsheet_replath) if data_dir is not None else None

        if first_data_folder_date == "":
            first_data_folder_date = datetime(2023,6,22)
        if on_data_dir_disconnected_bat_path == "":
            on_data_dir_disconnected_bat_path = MAP_BAT_PATH

        self._first_data_folder_date = first_data_folder_date
        self._bat_on_data_dir_disconnected = on_data_dir_disconnected_bat_path

        self._lite = False
        self._recent_completed_trust_window = RECENT_COMPLETED_TRUST_WINDOW
        self._timing_enabled = SERVER_TALK_TIMING_ENABLED
        self._run_id_lock = threading.Lock()  # serialises get_run_id / update_run_id

        self.set_data_dir()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_run_id_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._run_id_lock = threading.Lock()

    def set_data_dir(self, lite=False):
        if self._lite == lite:
            pass
        elif not lite:
            self.data_dir = os.path.dirname(self.data_dir)
        else:
            self.data_dir = os.path.join(self.data_dir, "_lite")
        self._lite = lite

    def get_data_file(self, idx=0, path="", lite=False):
        '''
        Returns the data file path corresponding to index idx. For idx > 0, idx is
        intepreted as a run ID. For idx = 0, gets the latest data file path. For idx
        < 0, increments chronologically backward from the latest file.

        If path is instead specified, loads the file at the specified path.

        Parameters
        ----------
        idx: int
            If a positive value is specified, it is interpreted as a run_id (as
            stored in run_info.run_id), and that data is found and loaded. If zero
            or a negative number are given, data is loaded relative to the most
            recent dataset (idx=0).
        path: str
            The full path to the file to be loaded. If not specified, loads the file
            as dictated by `idx`.

        Returns
        -------
        str: the full path to the specified data file
        int: the run ID for the specified data file
        '''
        t0 = time.perf_counter()
        if path == "":
            self.check_for_mapped_data_dir()
            if idx <= 0:
                relative_idx = abs(int(idx))
                if lite:
                    run_id = self.get_completed_run_id_by_relative_index(
                        relative_idx,
                        lite=False,
                        use_fresh_scan=True,
                        skip_check=True,
                    )
                    if run_id is None:
                        raise ValueError("No completed data files were found.")
                    file = self.find_data_file_by_run_id(run_id, lite=True, raise_on_missing=False, skip_check=True)
                    if file is None:
                        regular_file = self.find_data_file_by_run_id(run_id, lite=False, raise_on_missing=False, skip_check=True)
                        if regular_file is not None:
                            raise ValueError(
                                f"A lite copy does not exist for run ID {run_id}. Load the regular data or create a lite copy first."
                            )
                        raise ValueError(f"Data file with run ID {run_id:1.0f} was not found.")
                else:
                    file = self.get_completed_data_file_by_relative_index(
                        relative_idx,
                        lite=False,
                        use_fresh_scan=True,
                        skip_check=True,
                    )
                if file is None:
                    raise ValueError("No completed data files were found.")
            if idx > 0:
                file = self.find_data_file_by_run_id(idx, lite=lite, skip_check=True)
        else:
            if path.endswith('.hdf5'):
                file = path
            else:
                raise ValueError("The provided path is not a hdf5 file.")
            
        rid = self.run_id_from_filepath(file,lite)
        self._log_timing(f"get_data_file(idx={idx}, lite={lite})", t0)
        return file, rid

    def _log_timing(self, label, start_time):
        if self._timing_enabled:
            dt_ms = (time.perf_counter() - start_time) * 1e3
            print(f"[server_talk timing] {label}: {dt_ms:.2f} ms")

    def check_for_mapped_data_dir(self):
        self.set_data_dir()
        # If the dashboard configured the shared guard, defer to it so the
        # same throttling / logging / bat-missing handling applies to every
        # data-dir access (servers, GUIs, notebooks running under the
        # dashboard).  When the guard is not configured (e.g. a standalone
        # analysis script imported atomdata directly), fall back to the
        # original inline behavior so no consumer regresses.
        try:
            from waxx.util.dashboard import data_dir_guard  # noqa: PLC0415
            if data_dir_guard.is_configured():
                status = data_dir_guard.ensure_data_dir(self.data_dir)
                return bool(status.ok)
        except Exception:
            pass
        if not os.path.exists(self.data_dir):
            if sys.platform == "win32":
                print(f"Data dir ({self.data_dir}) not found. Attempting to re-map network drives.")
                cmd = self._bat_on_data_dir_disconnected
                subprocess.run(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
                if not os.path.exists(self.data_dir):
                    print(f"Data dir still not found. Are you connected to the physics network?")
                    return False
                else:
                    print("Network drives successfully mapped.")
                    return True
            else:
                print(f"Data dir ({self.data_dir}) not found. Are you connected to the network?")
                return False
        else:
            return True

    def get_latest_date_folder(self,lite=False,days_ago=0):
        self.set_data_dir(lite)
        date = datetime.today() - timedelta(days=days_ago)
        date_str = date.strftime('%Y-%m-%d')
        folderpath=os.path.join(self.data_dir,date_str)
        if not os.path.exists(folderpath) or next(os.scandir(folderpath), None) is None:
            folderpath = self.get_latest_date_folder(lite,days_ago+1)
        return folderpath
        
    def get_latest_data_file(self,lite=False):
        self.check_for_mapped_data_dir()
        return self.get_completed_data_file_by_relative_index(0, lite=lite, use_fresh_scan=True, skip_check=True)

    def _iter_date_dirs_desc(self, lite=False):
        self.set_data_dir(lite)
        root = self.data_dir
        if not root or not os.path.isdir(root):
            return

        # One directory listing instead of an isdir() probe per calendar day
        # back to the first data folder (well over a thousand round trips on
        # the network share, ~1 s per lookup of an old run). Same result:
        # existing date folders, newest first, within the same date window.
        today = datetime.today()
        first = self._first_data_folder_date
        dated = []
        try:
            with os.scandir(root) as it:
                for entry in it:
                    try:
                        if not entry.is_dir():
                            continue
                        date = datetime.strptime(entry.name, '%Y-%m-%d')
                    except (ValueError, OSError):
                        continue
                    if date < first or date > today:
                        continue
                    dated.append((date, entry.path))
        except OSError:
            return
        dated.sort(reverse=True)
        for _, path in dated:
            yield path

    def _iter_hdf5_files_desc(self, date_dir_path):
        files = []
        try:
            for file_entry in os.scandir(date_dir_path):
                if not file_entry.is_file() or not file_entry.name.lower().endswith('.hdf5'):
                    continue
                try:
                    run_id = int(file_entry.name.split('_')[0])
                except Exception:
                    continue
                files.append((run_id, file_entry.path))
        except OSError:
            return

        files.sort(key=lambda item: item[0], reverse=True)
        for _, path in files:
            yield path

    def _iter_completed_data_files_desc_fresh(self, lite=False, skip_check=False):
        if not skip_check:
            self.check_for_mapped_data_dir()
        yielded = 0
        for date_dir in self._iter_date_dirs_desc(lite=lite):
            for path in self._iter_hdf5_files_desc(date_dir):
                # Favor speed for newest files; keep strict completion checks for older files.
                if yielded < self._recent_completed_trust_window or self._is_completed_run(path):
                    yielded += 1
                    yield path

    def _is_completed_run(self, filepath):
        # locking=False: this probe runs newest-file-first over every candidate
        # (RECENT_COMPLETED_TRUST_WINDOW = 0), so it routinely opens the data
        # file of a run that is still being written — possibly from another
        # machine, over a mapped network drive.  Taking a read lock there can
        # make the writer's own open fail.  Everything below only reads attrs
        # and small datasets under a blanket except, so a torn read degrades to
        # "not complete", which is the right answer for an in-flight run.
        try:
            with h5py.File(filepath, 'r', locking=False) as f:
                # Fast path: explicit completion marker (present in new files).
                # True  → fully written; False → still being written by server.
                # None  → old file without the attr; fall through to xvar check.
                rc = f.attrs.get('run_complete', None)
                if rc is True:
                    return True
                if rc is False:
                    return False

                raw_xvarnames = f.attrs.get('xvarnames')
                if raw_xvarnames is None:
                    return True

                if isinstance(raw_xvarnames, np.ndarray):
                    xvarnames = [str(value.decode('utf-8', errors='replace') if isinstance(value, (bytes, np.bytes_)) else value) for value in raw_xvarnames.tolist()]
                elif isinstance(raw_xvarnames, (list, tuple)):
                    xvarnames = [str(value.decode('utf-8', errors='replace') if isinstance(value, (bytes, np.bytes_)) else value) for value in raw_xvarnames]
                else:
                    xvarnames = [str(raw_xvarnames.decode('utf-8', errors='replace') if isinstance(raw_xvarnames, (bytes, np.bytes_)) else raw_xvarnames)]

                xvarnames = [name for name in xvarnames if str(name).strip()]
                if not xvarnames:
                    return True

                if 'params' not in f:
                    return False

                for name in xvarnames:
                    if name not in f['params']:
                        return False
                    values = np.asarray(f['params'][name][()])
                    if values.ndim == 0:
                        return False
                return True
        except Exception:
            return False

    def _iter_completed_data_files_desc(self, lite=False, skip_check=False):
        yield from self._iter_completed_data_files_desc_fresh(lite=lite, skip_check=skip_check)

    def get_completed_data_file_by_relative_index(self, relative_idx=0, lite=False, use_fresh_scan=True, skip_check=False):
        t0 = time.perf_counter()
        iterator = self._iter_completed_data_files_desc_fresh(lite=lite, skip_check=skip_check) if use_fresh_scan else self._iter_completed_data_files_desc(lite=lite, skip_check=skip_check)
        for idx, path in enumerate(iterator):
            if idx == int(relative_idx):
                self._log_timing(
                    f"get_completed_data_file_by_relative_index(relative_idx={relative_idx}, lite={lite}, fresh={use_fresh_scan})",
                    t0,
                )
                return path
        self._log_timing(
            f"get_completed_data_file_by_relative_index(relative_idx={relative_idx}, lite={lite}, fresh={use_fresh_scan})",
            t0,
        )
        return None

    def get_completed_run_id_by_relative_index(self, relative_idx=0, lite=False, use_fresh_scan=True, skip_check=False):
        path = self.get_completed_data_file_by_relative_index(relative_idx=relative_idx,
                                                              lite=lite,
                                                              use_fresh_scan=use_fresh_scan,
                                                              skip_check=skip_check)
        if path is None:
            return None
        return self.run_id_from_filepath(path, lite=lite)

    def get_completed_data_files_window(self, start_relative_idx=0, count=1, lite=False, use_fresh_scan=True):
        start_relative_idx = int(start_relative_idx)
        count = int(count)
        if start_relative_idx < 0:
            raise ValueError('start_relative_idx must be >= 0')
        if count <= 0:
            return []

        iterator = self._iter_completed_data_files_desc_fresh(lite=lite) if use_fresh_scan else self._iter_completed_data_files_desc(lite=lite)
        out = []
        stop_idx = start_relative_idx + count
        for idx, path in enumerate(iterator):
            if idx < start_relative_idx:
                continue
            if idx >= stop_idx:
                break
            out.append(path)
        return out

    def recurse_find_data_file(self, r_id, lite=False, days_ago=0):
        # Superseded by find_data_file_by_run_id; kept as alias for backward compatibility.
        return self.find_data_file_by_run_id(r_id, lite=lite)

    def all_glob_find_data_file(self,run_id,lite=False):
        return self.find_data_file_by_run_id(run_id, lite=lite)

    def _find_data_file_by_run_id_fresh(self, run_id, lite=False, skip_check=False):
        return self._scan_for_run_id(run_id, lite=lite, skip_check=skip_check)[0]

    def _scan_for_run_id(self, run_id, lite=False, skip_check=False):
        """Walk the date folders for ``run_id``.

        Returns ``(path_or_None, scanned_any)`` where ``scanned_any`` says
        whether at least one date folder was actually listed -- the way to
        tell "the run is not there" from "the drive was not reachable".
        """
        if not skip_check:
            self.check_for_mapped_data_dir()
        run_id = int(run_id)
        prefix = f"{run_id:07d}_"
        scanned_any = False

        for date_dir in self._iter_date_dirs_desc(lite=lite):
            scanned_any = True
            max_seen = -1
            matches = []
            try:
                for file_entry in os.scandir(date_dir):
                    if not file_entry.is_file() or not file_entry.name.lower().endswith('.hdf5'):
                        continue
                    try:
                        file_run_id = int(file_entry.name.split('_')[0])
                    except Exception:
                        continue
                    if file_run_id > max_seen:
                        max_seen = file_run_id
                    if file_entry.name.startswith(prefix):
                        matches.append(file_entry.path)
            except OSError:
                continue

            if matches:
                if len(matches) > 1:
                    print(
                        f"[server_talk] WARNING: run ID {run_id} maps to "
                        f"{len(matches)} data files: {matches}. Loading "
                        f"{matches[0]}. This indicates a run_id collision."
                    )
                return matches[0], True
            # All run IDs in this folder are older than the target — stop searching.
            if max_seen >= 0 and max_seen < run_id:
                break

        return None, scanned_any

    def find_data_file_by_run_id(self, run_id, lite=False, raise_on_missing=True, refresh=False, skip_check=False):
        t0 = time.perf_counter()
        path, scanned_any = self._scan_for_run_id(run_id, lite=lite, skip_check=skip_check)

        # Retry once only when the first pass could not list any date folder
        # (a dropped network drive). A pass that walked the folders and found
        # nothing is a genuine miss -- e.g. probing for a lite copy that does
        # not exist -- and repeating it just doubled the cost.
        if path is None and not refresh and not scanned_any:
            path, _ = self._scan_for_run_id(run_id, lite=lite, skip_check=skip_check)
        
        if path is None and raise_on_missing:
            raise ValueError(f"Data file with run ID {run_id:1.0f} was not found.")
        self._log_timing(f"find_data_file_by_run_id(run_id={run_id}, lite={lite}, refresh={refresh})", t0)
        return path

    def find_nearest_run_date_and_id(self, requested_run_id, lite=False, refresh=False):
        self.check_for_mapped_data_dir()
        requested_run_id = int(requested_run_id)
        nearest_run_id = None
        nearest_run_date = None
        nearest_key = None

        for date_dir in self._iter_date_dirs_desc(lite=lite):
            date_folder = os.path.basename(date_dir)
            try:
                run_date = datetime.strptime(date_folder, '%Y-%m-%d').date()
            except ValueError:
                continue

            try:
                for file_entry in os.scandir(date_dir):
                    if not file_entry.is_file() or not file_entry.name.lower().endswith('.hdf5'):
                        continue
                    try:
                        run_id = int(file_entry.name.split('_')[0])
                    except Exception:
                        continue

                    key = (abs(run_id - requested_run_id), run_id)
                    if nearest_key is None or key < nearest_key:
                        nearest_key = key
                        nearest_run_id = run_id
                        nearest_run_date = run_date
            except OSError:
                continue

        if nearest_run_id is None:
            return None, None
        return nearest_run_id, nearest_run_date

    def run_id_from_filepath(self,filepath,lite=False):
        self.set_data_dir(lite)
        run_id = int(os.path.normpath(filepath).split(os.path.sep)[-1].split("_")[0])
        return run_id

    def get_run_id(self):
        self.set_data_dir()
        with self._run_id_lock:
            with open(self.run_id_path, 'r') as f:
                rid = f.read()
        return int(rid)

    def update_run_id(self, run_info=None):
        self.set_data_dir()
        with self._run_id_lock:
            if run_info is not None:
                rid = run_info.run_id
            else:
                with open(self.run_id_path, 'r') as f:
                    try:
                        rid = int(f.read())
                    except:
                        print(f'run id file at {self.run_id_path} is empty -- extracting from latest data file')
                        rid = self.run_id_from_filepath(self.get_latest_data_file())
            rid += 1
            with open(self.run_id_path, 'w') as f:
                f.write(f"{rid}")

    def get_latest_run_id_any(self):
        """Return the highest run_id across ALL hdf5 data files, including
        runs still in progress (``run_complete=False``).

        Unlike ``get_latest_data_file`` (which only considers completed runs),
        this scans every data file so a run_id reservation can avoid colliding
        with an in-progress run started by another server.  Returns ``None``
        when no data files exist.
        """
        self.check_for_mapped_data_dir()
        for date_dir in self._iter_date_dirs_desc():
            for path in self._iter_hdf5_files_desc(date_dir):
                # _iter_hdf5_files_desc yields the highest run_id first and
                # date dirs are newest-first, so the first hit is the global
                # maximum.
                return self.run_id_from_filepath(path)
        return None

    def set_run_id(self, value):
        """Overwrite the run_id counter file with ``value`` (the next run_id to
        be used).  Used by the reservation path to advance the monotonic floor
        after atomically claiming an id."""
        self.set_data_dir()
        with self._run_id_lock:
            with open(self.run_id_path, 'w') as f:
                f.write(f"{int(value)}")

    def create_lite_copy(self,run_idx,roi_id=None,use_saved_roi=True,
                         roix=None,roiy=None,path="",should_cancel=None):
        """Writes a lite copy of a run: non-image data copied, scope traces
        downcast to float32, images cropped to an ROI.

        Pass ``roix``/``roiy`` to crop headlessly: nothing here then touches
        Qt, so it is safe on a worker thread (the data browser does this). The
        ``roi_id``/``use_saved_roi`` form builds an ROI and may open the ROI
        dialog, so only call it that way from the main thread.

        The file is written to ``<lite>.tmp`` and renamed into place only when
        complete, so a failure or cancel never leaves a partial lite file.

        Args:
            run_idx (int): run id (see get_data_file).
            roi_id, use_saved_roi: as for ROI; ignored when roix/roiy are given.
            roix, roiy (sequence of 2 ints): explicit crop bounds in raw-image
                pixels, [x0, x1] and [y0, y1].
            path (str): the raw file, if already known (skips the run lookup).
            should_cancel (callable): polled between frames; returning True
                aborts, removes the temporary file and returns None.

        Returns:
            str or None: the lite file path, or None if cancelled.
        """
        from waxa.data import RunInfo, DataSaver
        from waxa.atomdata import unpack_group
        import h5py

        if (roix is None) != (roiy is None):
            raise ValueError("Pass both roix and roiy, or neither.")

        original_data_filepath, rid = self.get_data_file(run_idx, path=path)

        ri = RunInfo()
        with h5py.File(original_data_filepath,'r') as file:
            unpack_group(file,'run_info',ri)

        if roix is None:
            from waxa import ROI
            roi = ROI(rid,roi_id=roi_id,use_saved_roi=use_saved_roi,server_talk=self)
            roix, roiy = roi.roix, roi.roiy
        x0, x1 = (int(v) for v in roix)
        y0, y1 = (int(v) for v in roiy)

        ds = DataSaver(data_dir=self.data_dir, server_talk=self)
        lite_data_path, lite_data_folder = ds._data_path(ri,lite=True)
        os.makedirs(lite_data_folder, exist_ok=True)
        tmp_path = lite_data_path + '.tmp'

        cancelled = False
        try:
            cancelled = self._write_lite_file(
                original_data_filepath, tmp_path, x0, x1, y0, y1, should_cancel)
            if not cancelled:
                os.replace(tmp_path, lite_data_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        if cancelled:
            print(f'Lite creation for run {rid} cancelled.')
            return None
        print(f'Lite version of run {rid} saved at {lite_data_path}.')
        return lite_data_path

    def _write_lite_file(self, src_path, out_path, x0, x1, y0, y1, should_cancel=None):
        """Body of create_lite_copy. Returns True if cancelled."""
        import h5py

        with h5py.File(out_path,'w') as f_lite:
            with h5py.File(src_path,'r') as f_src:
                # copy over other datasets (not data)
                keys = f_src.keys()
                for key in keys:
                    if key != 'data':
                        f_src.copy(f_src[key],f_lite,key)

                # copy over non-image data
                dkeys = f_src['data'].keys()
                f_lite.create_group('data')
                for key in dkeys:
                    if key == 'images':
                        continue
                    if key == 'scope_data':
                        # Downcast float64 → float32 and apply compression.
                        scope_grp = f_lite['data'].create_group('scope_data')
                        for scope_label, scope_item in f_src['data']['scope_data'].items():
                            this_scope = scope_grp.create_group(scope_label)
                            for ch_key in scope_item.keys():
                                arr = scope_item[ch_key][()]
                                if arr.dtype == np.float64:
                                    arr = arr.astype(np.float32)
                                this_scope.create_dataset(
                                    ch_key, data=arr, compression='gzip', compression_opts=4
                                )
                    else:
                        f_src.copy(f_src['data'][key],f_lite['data'],key)

                # copy over attributes
                akeys = f_src.attrs.keys()
                for key in akeys:
                    f_lite.attrs[key] = f_src.attrs[key]

                if 'images' not in f_src['data']:
                    f_lite.attrs['has_images'] = False
                    for attr in ('roix', 'roiy'):
                        if attr in f_lite.attrs:
                            del f_lite.attrs[attr]
                    return False

                src_images = f_src['data']['images']
                N_img, H, W = src_images.shape
                if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                    raise ValueError(
                        f"ROI roix={[x0, x1]}, roiy={[y0, y1]} does not fit "
                        f"images of shape ({H}, {W}).")
                px, py = x1 - x0, y1 - y0

                # Lite images are pre-cropped, so the lite ROI is the whole
                # frame; the crop taken from the raw file is kept alongside.
                f_lite.attrs['roix'] = [0,px]
                f_lite.attrs['roiy'] = [0,py]
                f_lite.attrs['lite_source_roix'] = [x0,x1]
                f_lite.attrs['lite_source_roiy'] = [y0,y1]

                # Hyperslab reads pull only the ROI rows off disk, and a
                # frame-by-frame write keeps memory flat and makes cancel
                # responsive.
                out = f_lite['data'].create_dataset(
                    'images', shape=(N_img,py,px), dtype=src_images.dtype)
                for idx in range(N_img):
                    if should_cancel is not None and should_cancel():
                        return True
                    out[idx] = src_images[idx, y0:y1, x0:x1]
        return False