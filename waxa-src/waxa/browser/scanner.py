import glob
import json
import os
from datetime import date, datetime

import h5py
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .cache import MetadataCache
from ..plotting.plotting_1d import detect_unit
from .run_summary import RunSummary

EXCLUDED_DATA_KEYS = {
    "images",
    "image_timestamps",
    "sort_N",
    "sort_idx",
    "scope_data",
}


def _decode_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _attr_to_str_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [_decode_str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_decode_str(v) for v in value]
    return [_decode_str(value)]


def _path_basename_no_ext(path_value):
    path_str = _decode_str(path_value).replace("\\", "/")
    basename = path_str.split("/")[-1]
    if basename.endswith(".py"):
        basename = basename[:-3]
    return basename, path_str


def _is_completed_run(h5file):
    xvarnames = _attr_to_str_list(h5file.attrs.get("xvarnames"))
    xvarnames = [name for name in xvarnames if str(name).strip()]
    if not xvarnames:
        return True

    if "params" not in h5file:
        return False

    for name in xvarnames:
        if name not in h5file["params"]:
            return False
        values = np.asarray(h5file["params"][name][()])
        if values.ndim == 0:
            return False
    return True


class RunScanner:
    def __init__(self, data_dir: str, date_from: date, date_to: date):
        self.data_dir = data_dir
        self.date_from = date_from
        self.date_to = date_to
        self._lite_runs_by_date = {}
        self._cache = MetadataCache(data_dir)

    def scan(self):
        if not self.data_dir or not os.path.isdir(self.data_dir):
            return

        try:
            for folder in self._iter_date_folders():
                for filepath in self._iter_hdf5_files(folder):
                    try:
                        stat_result = os.stat(filepath)
                    except OSError:
                        continue

                    cached_summary = self._cache.get(filepath, stat_result)
                    if cached_summary is not None:
                        cached_summary.has_lite = self._has_lite_copy(
                            cached_summary.run_id,
                            cached_summary.run_date_str,
                        )
                        yield cached_summary
                        continue

                    summary = self._read_summary(filepath)
                    if summary is not None:
                        self._cache.put(summary, stat_result)
                        yield summary
        finally:
            self._cache.save()

    def _iter_date_folders(self):
        folders = []
        for name in os.listdir(self.data_dir):
            full = os.path.join(self.data_dir, name)
            if not os.path.isdir(full) or name == "_lite":
                continue
            try:
                folder_date = datetime.strptime(name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if self.date_from <= folder_date <= self.date_to:
                folders.append((folder_date, full))

        folders.sort(key=lambda item: item[0], reverse=True)
        for _, full in folders:
            yield full

    def _iter_hdf5_files(self, folder):
        files = glob.glob(os.path.join(folder, "*.hdf5"))

        def rid_key(path):
            try:
                return int(os.path.basename(path).split("_")[0])
            except Exception:
                return -1

        files.sort(key=rid_key, reverse=True)
        return files

    def _read_summary(self, filepath):
        try:
            with h5py.File(filepath, "r") as f:
                if not _is_completed_run(f):
                    return None

                run_id = self._read_run_id(f, filepath)
                run_date_str = _decode_str(
                    f.attrs.get("run_date_str", os.path.basename(os.path.dirname(filepath)))
                )
                run_datetime_str = _decode_str(f.attrs.get("run_datetime_str", ""))

                xvarnames = _attr_to_str_list(f.attrs.get("xvarnames"))
                experiment_name, experiment_filepath = self._read_experiment_info(f)
                xvardims = self._read_xvardims(f, xvarnames)

                if "data" in f:
                    data_keys = list(f["data"].keys())
                else:
                    data_keys = []

                data_container_keys = [
                    key for key in data_keys if key not in EXCLUDED_DATA_KEYS
                ]
                has_scope_data = "scope_data" in data_keys
                has_lite = self._has_lite_copy(run_id, run_date_str)

                # User-set tags and comment (stored as browser_tags / browser_comment attrs)
                tags = []
                raw_tags = f.attrs.get("browser_tags", "")
                raw_tags_str = raw_tags.decode("utf-8", errors="replace") if isinstance(raw_tags, bytes) else str(raw_tags)
                if raw_tags_str:
                    try:
                        tags = json.loads(raw_tags_str)
                    except Exception:
                        tags = [t.strip() for t in raw_tags_str.split(",") if t.strip()]

                raw_comment = f.attrs.get("browser_comment", "")
                comment = raw_comment.decode("utf-8", errors="replace") if isinstance(raw_comment, bytes) else str(raw_comment)

                return RunSummary(
                    run_id=run_id,
                    experiment_name=experiment_name,
                    experiment_filepath=experiment_filepath,
                    run_date_str=run_date_str,
                    run_datetime_str=run_datetime_str,
                    filepath=filepath,
                    xvarnames=xvarnames,
                    xvardims=xvardims,
                    data_container_keys=data_container_keys,
                    has_scope_data=has_scope_data,
                    has_lite=has_lite,
                    tags=tags,
                    comment=comment,
                )
        except Exception:
            return None

    def _read_experiment_info(self, h5file):
        if "run_info" in h5file and "experiment_filepath" in h5file["run_info"]:
            experiment_filepath = h5file["run_info"]["experiment_filepath"][()]
            return _path_basename_no_ext(experiment_filepath)

        fallback = h5file.attrs.get("expt_class", "")
        fallback_str = _decode_str(fallback)
        return fallback_str, fallback_str

    def _read_xvardims(self, h5file, xvarnames):
        if "data" in h5file and "sort_N" in h5file["data"]:
            sort_n = np.asarray(h5file["data"]["sort_N"][()]).reshape(-1)
            return tuple(int(value) for value in sort_n.tolist())

        if "params" in h5file:
            dims = []
            for name in xvarnames:
                if name in h5file["params"]:
                    values = np.asarray(h5file["params"][name][()]).reshape(-1)
                    dims.append(int(values.size))
            return tuple(dims)

        return tuple()

    def _read_run_id(self, h5file, filepath):
        run_id_attr = h5file.attrs.get("run_id", None)
        if run_id_attr is None:
            return int(os.path.basename(filepath).split("_")[0])
        if isinstance(run_id_attr, np.ndarray):
            return int(run_id_attr[()])
        return int(run_id_attr)

    def _has_lite_copy(self, run_id: int, run_date_str: str):
        if run_date_str not in self._lite_runs_by_date:
            self._lite_runs_by_date[run_date_str] = self._index_lite_runs_for_date(run_date_str)
        return run_id in self._lite_runs_by_date[run_date_str]

    def _index_lite_runs_for_date(self, run_date_str: str):
        lite_day_dir = os.path.join(self.data_dir, "_lite", run_date_str)
        if not os.path.isdir(lite_day_dir):
            return set()

        pattern = os.path.join(lite_day_dir, "*_lite_*.hdf5")
        run_ids = set()
        for path in glob.iglob(pattern):
            try:
                run_ids.add(int(os.path.basename(path).split("_")[0]))
            except Exception:
                continue
        return run_ids


class ScanWorker(QThread):
    run_found = pyqtSignal(object)
    run_batch_found = pyqtSignal(list)
    scan_done = pyqtSignal(int)
    scan_error = pyqtSignal(str)

    def __init__(self, scanner: RunScanner, batch_size: int = 64):
        super().__init__()
        self.scanner = scanner
        self._stop_requested = False
        self.batch_size = max(1, int(batch_size))

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        count = 0
        batch = []
        try:
            for run_summary in self.scanner.scan():
                if self._stop_requested:
                    break
                batch.append(run_summary)
                if len(batch) >= self.batch_size:
                    self.run_batch_found.emit(batch)
                    batch = []
                count += 1
            if batch:
                self.run_batch_found.emit(batch)
            self.scan_done.emit(count)
        except Exception as exc:
            self.scan_error.emit(str(exc))


class XvarDetailLoader(QThread):
    details_ready = pyqtSignal(list)
    details_error = pyqtSignal(str)

    def __init__(self, filepath: str, xvarnames: list[str]):
        super().__init__()
        self.filepath = filepath
        self.xvarnames = xvarnames

    def run(self):
        details = []
        try:
            with h5py.File(self.filepath, "r") as f:
                if "params" not in f:
                    self.details_ready.emit(details)
                    return

                params_group = f["params"]
                for name in self.xvarnames:
                    if name not in params_group:
                        details.append({"name": name, "unit": "", "n": 0, "min": "NA", "max": "NA"})
                        continue

                    values = params_group[name][()]
                    details.append(self._summarize_values(name, values))

            self.details_ready.emit(details)
        except Exception as exc:
            self.details_error.emit(str(exc))

    def _summarize_values(self, name: str, values):
        array = np.asarray(values)
        if array.size == 0:
            unit, _, _ = detect_unit(xvarnames=[name], xvar_idx=0, xvar_values=array)
            return {"name": name, "unit": unit or "", "n": 0, "min": "NA", "max": "NA"}

        flat = array.reshape(-1)
        n = int(flat.size)
        unit, multiplier, _ = detect_unit(xvarnames=[name], xvar_idx=0, xvar_values=flat)
        unit = unit or ""

        def decimals_from_spacing(scaled_values):
            finite = np.asarray(scaled_values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if finite.size < 2:
                return 6

            unique_sorted = np.unique(np.sort(finite))
            if unique_sorted.size < 2:
                return 6

            diffs = np.diff(unique_sorted)
            positive = diffs[diffs > 0]
            if positive.size == 0:
                return 6

            step = float(np.min(positive))
            decimals = int(np.ceil(-np.log10(step))) if step < 1.0 else 0
            # Keep one guard digit so adjacent values remain distinguishable
            # after floating-point conversion and formatting.
            return max(0, min(10, decimals + 1))

        def format_numeric_value(value, decimals):
            scaled = float(value) * multiplier
            if not np.isfinite(scaled):
                return "NA"
            if scaled != 0.0 and (abs(scaled) >= 1e7 or abs(scaled) < 1e-4):
                return f"{scaled:.6g}"
            return f"{scaled:.{decimals}f}"

        if np.issubdtype(flat.dtype, np.number):
            min_val = float(np.nanmin(flat))
            max_val = float(np.nanmax(flat))
            scaled_vals = np.asarray(flat, dtype=np.float64) * multiplier
            decimals = decimals_from_spacing(scaled_vals)
            return {
                "name": name,
                "unit": unit,
                "n": n,
                "min": format_numeric_value(min_val, decimals),
                "max": format_numeric_value(max_val, decimals),
            }

        as_text = [_decode_str(item) for item in flat]
        return {
            "name": name,
            "unit": unit,
            "n": n,
            "min": min(as_text),
            "max": max(as_text),
        }


class LiteCreateWorker(QThread):
    created = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, data_dir: str, run_id: int):
        super().__init__()
        self.data_dir = data_dir
        self.run_id = run_id

    def run(self):
        try:
            from waxa.data.server_talk import server_talk

            talk = server_talk(data_dir=self.data_dir)
            talk.create_lite_copy(self.run_id)
            lite_path, _ = talk.get_data_file(self.run_id, lite=True)
            self.created.emit(self.run_id, lite_path)
        except Exception as exc:
            self.error.emit(str(exc))


class BatchLiteCreateWorker(QThread):
    created = pyqtSignal(int, str)
    completed = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(self, data_dir: str, run_ids: list[int]):
        super().__init__()
        self.data_dir = data_dir
        self.run_ids = [int(rid) for rid in run_ids]

    def run(self):
        try:
            from waxa.data.server_talk import server_talk

            if not self.run_ids:
                raise ValueError("No runs were selected for lite creation.")

            talk = server_talk(data_dir=self.data_dir)
            total = len(self.run_ids)
            created_count = 0

            if total > 1:
                from waxa.roi import ROI

                oldest_run_id = min(self.run_ids)
                # Select ROI once on the oldest run, then reuse it for all selected runs.
                roi = ROI(run_id=oldest_run_id, use_saved_roi=False, printouts=False, server_talk=talk)
                roi.save_roi_h5(printouts=False)

                for run_id in self.run_ids:
                    talk.create_lite_copy(run_id, roi_id=oldest_run_id, use_saved_roi=True)
                    lite_path, _ = talk.get_data_file(run_id, lite=True)
                    self.created.emit(run_id, lite_path)
                    created_count += 1
            else:
                run_id = self.run_ids[0]
                talk.create_lite_copy(run_id)
                lite_path, _ = talk.get_data_file(run_id, lite=True)
                self.created.emit(run_id, lite_path)
                created_count += 1

            self.completed.emit(created_count, total)
        except Exception as exc:
            self.error.emit(str(exc))
