import os
import subprocess
import bisect
import time
import json
from datetime import datetime, timedelta
import glob
import numpy as np
import h5py

MAP_BAT_PATH = "\"G:\\Shared drives\\Weld Lab Shared Drive\\Infrastructure\\map_network_drives_PeterRecommended.bat\""
RUN_INDEX_TTL_S = 300.0
RECENT_COMPLETED_TRUST_WINDOW = 16
RELATIVE_INDEX_FRESH_FIRST_MAX = 8
SERVER_TALK_TIMING_ENABLED = False
RUN_INDEX_DISK_CACHE_ENABLED = True
RUN_INDEX_DISK_TTL_S = 3600.0
RUN_INDEX_DISK_CACHE_FILENAME = ".waxa_run_index_cache_v1.json"

class server_talk():
    def __init__(self,
                 data_dir=os.getenv("data"),
                 run_id_relpath="run_id.py",
                 roi_spreadsheet_replath="roi.xlsx",
                 first_data_folder_date="",
                 on_data_dir_disconnected_bat_path=""):
        
        self.data_dir = data_dir
        self.run_id_path = os.path.join(data_dir,run_id_relpath)
        self.roi_csv_path = os.path.join(data_dir,roi_spreadsheet_replath)

        if first_data_folder_date == "":
            first_data_folder_date = datetime(2023,6,22)
        if on_data_dir_disconnected_bat_path == "":
            on_data_dir_disconnected_bat_path = MAP_BAT_PATH

        self._first_data_folder_date = first_data_folder_date
        self._bat_on_data_dir_disconnected = on_data_dir_disconnected_bat_path

        self._lite = False
        self._run_index_cache = {
            False: None,
            True: None,
        }
        self._run_index_cache_time = {
            False: 0.0,
            True: 0.0,
        }
        self._completion_cache = {}
        self._run_index_ttl_s = RUN_INDEX_TTL_S
        self._recent_completed_trust_window = RECENT_COMPLETED_TRUST_WINDOW
        self._timing_enabled = SERVER_TALK_TIMING_ENABLED
        self._run_index_disk_cache_enabled = RUN_INDEX_DISK_CACHE_ENABLED
        self._run_index_disk_ttl_s = RUN_INDEX_DISK_TTL_S

        self.set_data_dir()

    def set_data_dir(self, lite=False):

        old_data_dir = self.data_dir
        old_lite = self._lite

        if self._lite == lite:
            pass
        elif not lite:
            self.data_dir = os.path.dirname(self.data_dir)
        else:
            self.data_dir = os.path.join(self.data_dir, "_lite")
        self._lite = lite

        # Data-root changes invalidate cached index metadata.
        if old_data_dir != self.data_dir or old_lite != self._lite:
            self._run_index_cache = {
                False: None,
                True: None,
            }
            self._run_index_cache_time = {
                False: 0.0,
                True: 0.0,
            }
            self._completion_cache = {}

    def _run_index_cache_path(self, lite=False):
        self.set_data_dir(lite)
        root = self.data_dir
        if not root:
            return None
        return os.path.join(root, RUN_INDEX_DISK_CACHE_FILENAME)

    def _serialize_run_index(self, index):
        out = {
            'run_ids': [int(v) for v in index.get('run_ids', [])],
            'paths_by_run_id': {},
            'dates_by_run_id': {},
        }
        for rid, path in index.get('paths_by_run_id', {}).items():
            out['paths_by_run_id'][str(int(rid))] = path
        for rid, run_date in index.get('dates_by_run_id', {}).items():
            if isinstance(run_date, datetime):
                run_date_str = run_date.date().isoformat()
            elif hasattr(run_date, 'isoformat'):
                run_date_str = run_date.isoformat()
            elif run_date is None:
                run_date_str = None
            else:
                run_date_str = str(run_date)
            out['dates_by_run_id'][str(int(rid))] = run_date_str
        return out

    def _deserialize_run_index(self, payload):
        run_ids = sorted([int(v) for v in payload.get('run_ids', [])])
        paths_by_run_id = {
            int(rid): path
            for rid, path in payload.get('paths_by_run_id', {}).items()
        }
        dates_by_run_id = {}
        for rid, run_date in payload.get('dates_by_run_id', {}).items():
            rid_int = int(rid)
            if run_date is None:
                continue
            try:
                dates_by_run_id[rid_int] = datetime.strptime(run_date, '%Y-%m-%d').date()
            except Exception:
                # Keep permissive parsing behavior for forward compatibility.
                try:
                    dates_by_run_id[rid_int] = datetime.fromisoformat(run_date).date()
                except Exception:
                    continue
        return {
            'run_ids': run_ids,
            'paths_by_run_id': paths_by_run_id,
            'dates_by_run_id': dates_by_run_id,
        }

    def _load_run_index_from_disk(self, lite=False):
        if not self._run_index_disk_cache_enabled:
            return None

        cache_path = self._run_index_cache_path(lite=lite)
        if cache_path is None or not os.path.isfile(cache_path):
            return None

        try:
            age_s = time.time() - os.path.getmtime(cache_path)
            if age_s > self._run_index_disk_ttl_s:
                return None
        except OSError:
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if payload.get('version') != 1:
                return None
            if bool(payload.get('lite')) != bool(lite):
                return None
            if payload.get('data_dir') != self.data_dir:
                return None
            return self._deserialize_run_index(payload.get('index', {}))
        except Exception:
            return None

    def _save_run_index_to_disk(self, index, lite=False):
        if not self._run_index_disk_cache_enabled:
            return

        cache_path = self._run_index_cache_path(lite=lite)
        if cache_path is None:
            return

        payload = {
            'version': 1,
            'lite': bool(lite),
            'data_dir': self.data_dir,
            'created_unix_s': time.time(),
            'index': self._serialize_run_index(index),
        }
        tmp_path = f"{cache_path}.tmp"
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            os.replace(tmp_path, cache_path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

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
            if idx <= 0:
                relative_idx = abs(int(idx))
                # If an index is already available in memory or on disk, prefer it
                # even for very recent relative lookups to avoid fresh directory scans.
                cached_index_available = self._get_run_index(lite=False, allow_build=False) is not None
                fresh_first = (relative_idx <= RELATIVE_INDEX_FRESH_FIRST_MAX) and (not cached_index_available)
                if lite:
                    run_id = self.get_completed_run_id_by_relative_index(
                        relative_idx,
                        lite=False,
                        use_fresh_scan=fresh_first,
                    )
                    if run_id is None:
                        run_id = self.get_completed_run_id_by_relative_index(
                            relative_idx,
                            lite=False,
                            use_fresh_scan=not fresh_first,
                        )
                    if run_id is None:
                        raise ValueError("No completed data files were found.")
                    file = self.find_data_file_by_run_id(run_id, lite=True, raise_on_missing=False)
                    if file is None:
                        regular_file = self.find_data_file_by_run_id(run_id, lite=False, raise_on_missing=False)
                        if regular_file is not None:
                            raise ValueError(
                                f"A lite copy does not exist for run ID {run_id}. Load the regular data or create a lite copy first."
                            )
                        raise ValueError(f"Data file with run ID {run_id:1.0f} was not found.")
                else:
                    file = self.get_completed_data_file_by_relative_index(
                        relative_idx,
                        lite=False,
                        use_fresh_scan=fresh_first,
                    )
                    if file is None:
                        file = self.get_completed_data_file_by_relative_index(
                            relative_idx,
                            lite=False,
                            use_fresh_scan=not fresh_first,
                        )
                if file is None:
                    raise ValueError("No completed data files were found.")
            if idx > 0:
                file = self.find_data_file_by_run_id(idx, lite=lite)
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
        if not os.path.exists(self.data_dir):
            print(f"Data dir ({self.data_dir}) not found. Attempting to re-map network drives.")
            cmd = self._bat_on_data_dir_disconnected         
            result = subprocess.run(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
            if not os.path.exists(self.data_dir):
                print(f"Data dir still not found. Are you connected to the physics network?") 
                return False
            else:
                print("Network drives successfully mapped.")
                return True
        else:
            return True

    def get_latest_date_folder(self,lite=False,days_ago=0):
        self.set_data_dir(lite)
        date = datetime.today() - timedelta(days=days_ago)
        date_str = date.strftime('%Y-%m-%d')
        folderpath=os.path.join(self.data_dir,date_str)
        if not os.path.exists(folderpath) or not os.listdir(folderpath):
            folderpath = self.get_latest_date_folder(lite,days_ago+1)
        return folderpath
        
    def get_latest_data_file(self,lite=False):
        self.check_for_mapped_data_dir()
        return self.get_completed_data_file_by_relative_index(0, lite=lite, use_fresh_scan=True)

    def _iter_date_dirs_desc(self, lite=False):
        self.set_data_dir(lite)
        root = self.data_dir
        if not root or not os.path.isdir(root):
            return

        date_dirs = []
        for date_entry in os.scandir(root):
            if not date_entry.is_dir():
                continue
            try:
                run_date = datetime.strptime(date_entry.name, '%Y-%m-%d').date()
            except ValueError:
                continue
            date_dirs.append((run_date, date_entry.path))

        date_dirs.sort(key=lambda item: item[0], reverse=True)
        for _, path in date_dirs:
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
                try:
                    mtime = file_entry.stat().st_mtime
                except OSError:
                    mtime = 0.0
                files.append((run_id, mtime, file_entry.path))
        except OSError:
            return

        files.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, path in files:
            yield path

    def _iter_completed_data_files_desc_fresh(self, lite=False):
        self.check_for_mapped_data_dir()
        yielded = 0
        for date_dir in self._iter_date_dirs_desc(lite=lite):
            for path in self._iter_hdf5_files_desc(date_dir):
                # Favor speed for newest files; keep strict completion checks for older files.
                if yielded < self._recent_completed_trust_window or self._is_completed_run(path):
                    yielded += 1
                    yield path

    def _is_completed_run(self, filepath):
        try:
            mtime = os.path.getmtime(filepath)
        except OSError:
            mtime = None

        cached = self._completion_cache.get(filepath)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        try:
            with h5py.File(filepath, 'r') as f:
                raw_xvarnames = f.attrs.get('xvarnames')
                if raw_xvarnames is None:
                    is_completed = True
                    self._completion_cache[filepath] = (mtime, is_completed)
                    return is_completed

                if isinstance(raw_xvarnames, np.ndarray):
                    xvarnames = [str(value.decode('utf-8', errors='replace') if isinstance(value, (bytes, np.bytes_)) else value) for value in raw_xvarnames.tolist()]
                elif isinstance(raw_xvarnames, (list, tuple)):
                    xvarnames = [str(value.decode('utf-8', errors='replace') if isinstance(value, (bytes, np.bytes_)) else value) for value in raw_xvarnames]
                else:
                    xvarnames = [str(raw_xvarnames.decode('utf-8', errors='replace') if isinstance(raw_xvarnames, (bytes, np.bytes_)) else raw_xvarnames)]

                xvarnames = [name for name in xvarnames if str(name).strip()]
                if not xvarnames:
                    is_completed = True
                    self._completion_cache[filepath] = (mtime, is_completed)
                    return is_completed

                if 'params' not in f:
                    is_completed = False
                    self._completion_cache[filepath] = (mtime, is_completed)
                    return is_completed

                for name in xvarnames:
                    if name not in f['params']:
                        is_completed = False
                        self._completion_cache[filepath] = (mtime, is_completed)
                        return is_completed
                    values = np.asarray(f['params'][name][()])
                    if values.ndim == 0:
                        is_completed = False
                        self._completion_cache[filepath] = (mtime, is_completed)
                        return is_completed
                is_completed = True
                self._completion_cache[filepath] = (mtime, is_completed)
                return is_completed
        except Exception:
            self._completion_cache[filepath] = (mtime, False)
            return False

    def _iter_completed_data_files_desc(self, lite=False):
        index = self._get_run_index(lite=lite)
        for idx, run_id in enumerate(reversed(index['run_ids'])):
            path = index['paths_by_run_id'].get(run_id)
            if not path:
                continue

            # Newest files dominate relative lookups. Trust a short recent window
            # and avoid expensive HDF5 completion checks there.
            if idx < self._recent_completed_trust_window:
                yield path
                continue

            if self._is_completed_run(path):
                yield path

    def get_completed_data_file_by_relative_index(self, relative_idx=0, lite=False, use_fresh_scan=True):
        t0 = time.perf_counter()
        iterator = self._iter_completed_data_files_desc_fresh(lite=lite) if use_fresh_scan else self._iter_completed_data_files_desc(lite=lite)
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

    def get_completed_run_id_by_relative_index(self, relative_idx=0, lite=False, use_fresh_scan=True):
        path = self.get_completed_data_file_by_relative_index(relative_idx=relative_idx,
                                                              lite=lite,
                                                              use_fresh_scan=use_fresh_scan)
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

    def recurse_find_data_file(self,r_id,lite=False,days_ago=0):
        indexed_file = self.find_data_file_by_run_id(r_id, lite=lite, raise_on_missing=False)
        if indexed_file is not None:
            return indexed_file

        date = datetime.today() - timedelta(days=days_ago)

        if date < self._first_data_folder_date:
            raise ValueError(f"Data file with run ID {r_id:1.0f} was not found.")
        
        date_str = date.strftime('%Y-%m-%d')

        self.set_data_dir(lite)
        folderpath=os.path.join(self.data_dir,date_str)

        if os.path.exists(folderpath):
            pattern = os.path.join(folderpath,'*.hdf5')
            files = np.array(list(glob.iglob(pattern)))
            r_ids = np.array([self.run_id_from_filepath(file,lite) for file in files])

            if len(r_ids) > 0 and int(np.max(r_ids)) < int(r_id):
                raise ValueError(f"Data file with run ID {r_id:1.0f} was not found.")

            files_mask = (r_id == r_ids)
            file_with_rid = files[files_mask]

            if len(file_with_rid) > 1:
                print(f"There are two data files with run ID {r_id:1.0f}. Choosing the more recent one.")
                file_with_rid = max(file_with_rid,key=os.path.getmtime)
            elif len(file_with_rid) == 1:
                file_with_rid = file_with_rid[0]
            
            if not file_with_rid:
                file_with_rid = self.recurse_find_data_file(r_id,lite,days_ago+1)
        else:
            file_with_rid = self.recurse_find_data_file(r_id,lite,days_ago+1)
        return file_with_rid

    def all_glob_find_data_file(self,run_id,lite=False):
        return self.find_data_file_by_run_id(run_id, lite=lite)

    def _build_run_index(self, lite=False):
        self.set_data_dir(lite)
        root = self.data_dir
        paths_by_run_id = {}
        dates_by_run_id = {}

        if not root or not os.path.isdir(root):
            return {
                'run_ids': [],
                'paths_by_run_id': {},
                'dates_by_run_id': {},
            }

        for date_entry in os.scandir(root):
            if not date_entry.is_dir():
                continue
            try:
                run_date = datetime.strptime(date_entry.name, '%Y-%m-%d').date()
            except ValueError:
                continue

            for file_entry in os.scandir(date_entry.path):
                if not file_entry.is_file() or not file_entry.name.lower().endswith('.hdf5'):
                    continue
                try:
                    run_id = int(file_entry.name.split('_')[0])
                except Exception:
                    continue

                existing = paths_by_run_id.get(run_id)
                if existing is None:
                    paths_by_run_id[run_id] = file_entry.path
                    dates_by_run_id[run_id] = run_date
                else:
                    try:
                        if os.path.getmtime(file_entry.path) > os.path.getmtime(existing):
                            paths_by_run_id[run_id] = file_entry.path
                            dates_by_run_id[run_id] = run_date
                    except OSError:
                        pass

        run_ids = sorted(paths_by_run_id.keys())
        return {
            'run_ids': run_ids,
            'paths_by_run_id': paths_by_run_id,
            'dates_by_run_id': dates_by_run_id,
        }

    def _get_run_index(self, lite=False, refresh=False, allow_build=True):
        t0 = time.perf_counter()
        lite_key = bool(lite)
        cache_age_s = time.time() - self._run_index_cache_time[lite_key]
        cache_stale = cache_age_s > self._run_index_ttl_s
        in_memory = self._run_index_cache.get(lite_key)

        if not refresh and in_memory is not None and not cache_stale:
            return in_memory

        if not refresh:
            disk_index = self._load_run_index_from_disk(lite=lite)
            if disk_index is not None:
                self._run_index_cache[lite_key] = disk_index
                self._run_index_cache_time[lite_key] = time.time()
                self._log_timing(f"_load_run_index_from_disk(lite={lite})", t0)
                return disk_index

        if not allow_build:
            return None

        self._run_index_cache[lite_key] = self._build_run_index(lite=lite)
        self._save_run_index_to_disk(self._run_index_cache[lite_key], lite=lite)
        self._run_index_cache_time[lite_key] = time.time()
        self._log_timing(f"_build_run_index(lite={lite})", t0)
        return self._run_index_cache[lite_key]

    def _find_data_file_by_run_id_fresh(self, run_id, lite=False):
        self.check_for_mapped_data_dir()
        run_id = int(run_id)
        prefix = f"{run_id}_"
        candidate_paths = []

        for date_dir in self._iter_date_dirs_desc(lite=lite):
            try:
                for file_entry in os.scandir(date_dir):
                    if not file_entry.is_file() or not file_entry.name.lower().endswith('.hdf5'):
                        continue
                    if file_entry.name.startswith(prefix):
                        try:
                            mtime = file_entry.stat().st_mtime
                        except OSError:
                            mtime = 0.0
                        candidate_paths.append((mtime, file_entry.path))
            except OSError:
                continue

        if not candidate_paths:
            return None
        candidate_paths.sort(key=lambda item: item[0], reverse=True)
        return candidate_paths[0][1]

    def _update_run_index_cache_with_path(self, run_id, path, lite=False):
        cache = self._run_index_cache.get(bool(lite))
        if cache is None:
            return
        run_id = int(run_id)
        cache['paths_by_run_id'][run_id] = path
        if run_id not in cache['run_ids']:
            cache['run_ids'].append(run_id)
            cache['run_ids'].sort()

        date_folder = os.path.basename(os.path.dirname(path))
        try:
            cache['dates_by_run_id'][run_id] = datetime.strptime(date_folder, '%Y-%m-%d').date()
        except Exception:
            pass
        self._run_index_cache_time[bool(lite)] = time.time()
        self._save_run_index_to_disk(cache, lite=lite)

    def find_data_file_by_run_id(self, run_id, lite=False, raise_on_missing=True, refresh=False):
        t0 = time.perf_counter()
        index = self._get_run_index(lite=lite, refresh=refresh)
        path = index['paths_by_run_id'].get(int(run_id))
        if path is None and not refresh:
            # Strict-fresh targeted lookup before paying full index rebuild cost.
            fresh_path = self._find_data_file_by_run_id_fresh(run_id, lite=lite)
            if fresh_path is not None:
                self._update_run_index_cache_with_path(run_id, fresh_path, lite=lite)
                self._log_timing(f"find_data_file_by_run_id(run_id={run_id}, lite={lite}, refresh={refresh})", t0)
                return fresh_path

            # Fallback to full index rebuild once.
            return self.find_data_file_by_run_id(run_id,
                                                 lite=lite,
                                                 raise_on_missing=raise_on_missing,
                                                 refresh=True)
        if path is None and raise_on_missing:
            raise ValueError(f"Data file with run ID {run_id:1.0f} was not found.")
        self._log_timing(f"find_data_file_by_run_id(run_id={run_id}, lite={lite}, refresh={refresh})", t0)
        return path

    def find_nearest_run_date_and_id(self, requested_run_id, lite=False, refresh=False):
        index = self._get_run_index(lite=lite, refresh=refresh)
        run_ids = index['run_ids']
        if not run_ids and not refresh:
            return self.find_nearest_run_date_and_id(requested_run_id, lite=lite, refresh=True)
        if not run_ids:
            return None, None

        requested_run_id = int(requested_run_id)
        dates_by_run_id = index['dates_by_run_id']

        if requested_run_id in dates_by_run_id:
            return requested_run_id, dates_by_run_id[requested_run_id]

        insert_idx = bisect.bisect_left(run_ids, requested_run_id)
        candidates = []
        if insert_idx > 0:
            candidates.append(run_ids[insert_idx - 1])
        if insert_idx < len(run_ids):
            candidates.append(run_ids[insert_idx])

        nearest_run_id = min(candidates, key=lambda rid: (abs(rid - requested_run_id), rid))
        return nearest_run_id, dates_by_run_id[nearest_run_id]

    def run_id_from_filepath(self,filepath,lite=False):
        self.set_data_dir(lite)
        run_id = int(os.path.normpath(filepath).split(os.path.sep)[-1].split("_")[0])
        return run_id

    def get_run_id(self):
        self.set_data_dir()

        pwd = os.getcwd()
        os.chdir(self.data_dir)
        with open(self.run_id_path,'r') as f:
            rid = f.read()
        os.chdir(pwd)
        return int(rid)

    def update_run_id(self,run_info=None):
        self.set_data_dir()

        pwd = os.getcwd()
        os.chdir(self.data_dir)

        if run_info is not None:
            rid = run_info.run_id
        else:
            with open(self.run_id_path,'r') as f:
                try: 
                    rid = int(f.read())
                except:
                    print(f'run id file at {self.run_id_path} is empty -- extracting from latest data file')
                    rid = self.run_id_from_filepath(self.get_latest_data_file())
        rid += 1
        with open(self.run_id_path,'w') as f:
            f.write(f"{rid}")

        os.chdir(pwd)

    def create_lite_copy(self,run_idx,roi_id=None,use_saved_roi=True):
        from waxa.data import RunInfo, DataSaver
        from waxa.atomdata import unpack_group
        import h5py
        from waxa import ROI

        original_data_filepath, rid = self.get_data_file(run_idx)

        ri = RunInfo()
        with h5py.File(original_data_filepath) as file:
            unpack_group(file,'run_info',ri)

        ds = DataSaver(data_dir=self.data_dir, server_talk=self)
        lite_data_path, lite_data_folder = ds._data_path(ri,lite=True)

        os.makedirs(lite_data_folder, exist_ok=True)

        with h5py.File(lite_data_path,'w') as f_lite:
            with h5py.File(original_data_filepath,'r') as f_src:
                # copy over other datasets (not data)
                keys = f_src.keys()
                for key in keys:
                    if key != 'data':
                        f_src.copy(f_src[key],f_lite,key)

                # copy over non-image data
                dkeys = f_src['data'].keys()
                f_lite.create_group('data')
                for key in dkeys:
                    if key != 'images':
                        f_src.copy(f_src['data'][key],f_lite['data'],key)

                # copy over attributes
                akeys = f_src.attrs.keys()
                for key in akeys:
                    f_lite.attrs[key] = f_src.attrs[key]

                roi = ROI(rid,roi_id=roi_id,use_saved_roi=use_saved_roi,server_talk=self)
                
                N_img = f_src['data']['images'].shape[0]
                px = np.diff(roi.roix)[0]
                py = np.diff(roi.roiy)[0]

                f_lite.attrs['roix'] = [0,px]
                f_lite.attrs['roiy'] = [0,py]

                dtype = f_src['data']['images'][0].dtype
                images = np.zeros((N_img,py,px),dtype=dtype)

                for idx in range(N_img):
                    images[idx] = roi.crop(f_src['data']['images'][idx])
                f_lite['data']['images'] = images
        print(f'Lite version of run {rid} saved at {lite_data_path}.')