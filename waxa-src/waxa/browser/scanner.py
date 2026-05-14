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

PARAM_SEARCH_MODES = ("params", "camera_params", "data")


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


def _preview_dataset_value(dataset, max_items: int = 8):
    try:
        if dataset.shape == ():
            return dataset[()]
        if dataset.size == 0:
            return np.asarray(dataset[()])
        if dataset.size <= max_items:
            return np.asarray(dataset[()])

        first_dim = dataset.shape[0] if dataset.shape else 0
        slice_len = max(1, min(max_items, first_dim))
        sample = np.asarray(dataset[tuple([slice(0, slice_len)] + [slice(None)] * (dataset.ndim - 1))])
        return sample
    except Exception:
        return None


def _stringify_value(value, max_chars: int = 240):
    if value is None:
        return "-"

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, np.bytes_):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U", "O"}:
            text = np.array2string(
                np.asarray([_decode_str(item) for item in value.reshape(-1)]).reshape(value.shape),
                threshold=8,
                edgeitems=3,
            )
        else:
            text = np.array2string(value, threshold=8, edgeitems=3)
    elif isinstance(value, np.generic):
        text = str(value.item())
    elif isinstance(value, (list, tuple)):
        text = repr(value)
    else:
        text = str(value)

    text = text.strip() or "-"
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def _decimals_from_spacing(scaled_values):
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
    return max(0, min(10, decimals + 1))


def _format_numeric_value(value: float, multiplier: float, decimals: int):
    scaled = float(value) * float(multiplier)
    if not np.isfinite(scaled):
        return "NA"
    if scaled != 0.0 and (abs(scaled) >= 1e7 or abs(scaled) < 1e-4):
        return f"{scaled:.6g}"
    return f"{scaled:.{decimals}f}"


def _axis_all_same(value_array: np.ndarray, axis: int):
    if value_array.shape[axis] <= 1:
        return True

    reference = np.take(value_array, 0, axis=axis)
    reference = np.expand_dims(reference, axis=axis)
    try:
        equal_mask = value_array == reference
    except Exception:
        return False

    if np.issubdtype(value_array.dtype, np.floating):
        equal_mask = equal_mask | (np.isnan(value_array) & np.isnan(reference))
    elif np.issubdtype(value_array.dtype, np.complexfloating):
        equal_mask = equal_mask | (
            np.isnan(value_array.real)
            & np.isnan(reference.real)
            & np.isnan(value_array.imag)
            & np.isnan(reference.imag)
        )

    try:
        return bool(np.all(equal_mask))
    except Exception:
        return False


def _all_same_summary(values):
    array = np.asarray(values)

    if array.size == 0:
        return "all_same: -"

    if array.ndim == 0:
        return "all_same: True"

    same_flags = tuple(_axis_all_same(array, axis) for axis in range(array.ndim))
    if all(same_flags):
        return "all_same: True"
    return f"all_same: {same_flags}"


def _value_summary(name: str, values):
    array = np.asarray(values)
    if array.size == 0:
        return None

    all_same_text = _all_same_summary(array)

    if array.ndim == 0:
        scalar = array.item()
        scalar_text = _decode_str(scalar) if isinstance(scalar, (bytes, np.bytes_)) else str(scalar)
        return f"min: {scalar_text} | max: {scalar_text} | {all_same_text}"

    flat = array.reshape(-1)
    if np.issubdtype(flat.dtype, np.number):
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return f"min: NA | max: NA | {all_same_text}"

        unit, multiplier, _ = detect_unit(xvarnames=[name], xvar_idx=0, xvar_values=finite)
        unit = unit or ""
        scaled_vals = np.asarray(finite, dtype=np.float64) * float(multiplier)
        decimals = _decimals_from_spacing(scaled_vals)
        min_text = _format_numeric_value(np.nanmin(finite), multiplier, decimals)
        max_text = _format_numeric_value(np.nanmax(finite), multiplier, decimals)
        unit_suffix = f" {unit}" if unit else ""
        return f"min: {min_text}{unit_suffix} | max: {max_text}{unit_suffix} | {all_same_text}"

    as_text = [_decode_str(item) for item in flat]
    return f"min: {min(as_text)} | max: {max(as_text)} | {all_same_text}"


def _read_n_repeats_value(h5file):
    if "params" not in h5file or "N_repeats" not in h5file["params"]:
        return 1

    raw = np.asarray(h5file["params"]["N_repeats"][()]).reshape(-1)
    if raw.size == 0:
        return 1

    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return 1

    try:
        return max(1, int(finite[0]))
    except Exception:
        return 1


def _build_value_record(mode: str, name: str, value, dataset=None):
    array_value = None
    shape = "()"
    dtype_name = type(value).__name__

    if dataset is not None:
        shape = str(tuple(int(dim) for dim in dataset.shape)) if dataset.shape else "()"
        dtype_name = str(dataset.dtype)
        array_value = _preview_dataset_value(dataset)
    else:
        array_value = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray, np.generic)) else value
        if isinstance(array_value, np.ndarray):
            shape = str(tuple(int(dim) for dim in array_value.shape)) if array_value.shape else "()"
            dtype_name = str(array_value.dtype)

    preview_source = array_value if array_value is not None else value
    preview = _stringify_value(preview_source, max_chars=160)
    detail = _stringify_value(preview_source, max_chars=8000)

    if dataset is not None or mode in {"params", "camera_params"}:
        values_for_summary = value
        if values_for_summary is None and dataset is not None:
            try:
                values_for_summary = dataset[()]
            except Exception:
                values_for_summary = None
        stats_text = _value_summary(str(name), values_for_summary) if values_for_summary is not None else None
        if stats_text:
            preview = _stringify_value(f"{stats_text} | {preview}", max_chars=160)
            detail = f"{stats_text}\n\n{detail}"

    if dataset is not None and dataset.size > 8:
        detail += f"\n\nPreview truncated from dataset with shape {shape} and dtype {dtype_name}."

    return {
        "mode": mode,
        "name": str(name),
        "dtype": dtype_name,
        "shape": shape,
        "preview": preview,
        "detail": detail,
        "size": int(dataset.size) if dataset is not None else int(np.asarray(array_value).size) if isinstance(array_value, np.ndarray) else 1,
    }


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
                n_repeats = _read_n_repeats_value(f)

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
                    n_repeats=n_repeats,
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
            return {"name": name, "unit": unit or "", "n": 0, "min": "NA", "max": "NA", "preview": "-"}

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

        def unique_preserve_order(items):
            unique_items = []
            for item in items:
                if any(item == existing for existing in unique_items):
                    continue
                unique_items.append(item)
            return unique_items

        def format_preview_text(items, formatter=str):
            if not items:
                return "-"

            formatted = [formatter(item) for item in items]
            if len(formatted) <= 6:
                return ", ".join(formatted)
            return ", ".join([*formatted[:3], "...", *formatted[-3:]])

        if np.issubdtype(flat.dtype, np.number):
            min_val = float(np.nanmin(flat))
            max_val = float(np.nanmax(flat))
            scaled_vals = np.asarray(flat, dtype=np.float64) * multiplier
            decimals = decimals_from_spacing(scaled_vals)
            preview_values = unique_preserve_order(np.asarray(flat).tolist())
            return {
                "name": name,
                "unit": unit,
                "n": n,
                "min": format_numeric_value(min_val, decimals),
                "max": format_numeric_value(max_val, decimals),
                "preview": format_preview_text(preview_values, lambda value: format_numeric_value(value, decimals)),
            }

        as_text = [_decode_str(item) for item in flat]
        preview_values = unique_preserve_order(as_text)
        return {
            "name": name,
            "unit": unit,
            "n": n,
            "min": min(as_text),
            "max": max(as_text),
            "preview": format_preview_text(preview_values),
        }


class ParamSearchLoader(QThread):
    records_ready = pyqtSignal(dict)
    load_error = pyqtSignal(str)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        records = {mode: [] for mode in PARAM_SEARCH_MODES}
        try:
            with h5py.File(self.filepath, "r") as f:
                records["params"] = self._load_group_records(f, "params")
                records["camera_params"] = self._load_group_records(f, "camera_params")
                records["data"] = self._load_data_records(f)
            self.records_ready.emit(records)
        except Exception as exc:
            self.load_error.emit(str(exc))

    def _load_group_records(self, h5file, group_name: str):
        if group_name not in h5file:
            return []

        records = []
        group = h5file[group_name]
        for key in sorted(group.keys()):
            dataset = group[key]
            try:
                value = dataset[()]
            except Exception:
                value = None
            records.append(_build_value_record(group_name, key, value, dataset=dataset))
        return records

    def _load_data_records(self, h5file):
        if "data" not in h5file:
            return []

        records = []
        group = h5file["data"]
        for key in sorted(group.keys()):
            item = group[key]
            if isinstance(item, h5py.Group):
                records.append(
                    {
                        "mode": "data",
                        "name": str(key),
                        "dtype": "group",
                        "shape": "-",
                        "preview": f"group with {len(item.keys())} entries",
                        "detail": f"HDF5 group '{key}' with children: {', '.join(sorted(item.keys())) or '(none)'}",
                        "size": len(item.keys()),
                    }
                )
                continue

            records.append(_build_value_record("data", key, None, dataset=item))
        return records


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
