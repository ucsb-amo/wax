import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

import h5py
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

_SCAN_WORKERS = 8  # parallel HDF5 reader threads
LOGGER = logging.getLogger(__name__)

from .cache import MetadataCache
from ..plotting.units import detect_unit
from .run_summary import RunSummary

EXCLUDED_DATA_KEYS = {
    "images",
    "image_timestamps",
    "sort_N",
    "sort_idx",
    "scope_data",
}

PARAM_SEARCH_MODES = ("params", "camera_params", "data")

# Datasets larger than this (in elements) are never read in full by the param
# search loader: only a small leading slice is fetched for the preview and the
# min/max/all_same summary is skipped.  Keeps `data/images` (hundreds of MB)
# from being pulled over the network every time the selected run changes.
PARAM_FULL_READ_MAX_ELEMENTS = 200_000

# Datasets that are never worth previewing element-wise in the param search.
PARAM_SEARCH_NO_READ_KEYS = {"images", "scope_data"}

# Date folders older than this many days are treated as immutable for the
# purposes of the per-folder run-id index (a run that starts before midnight
# still lands in that day's folder, so "yesterday" can gain files).
FOLDER_INDEX_MUTABLE_DAYS = 2

# The root directory listing (one entry per date folder) changes at most once
# a day; re-reading it from the network on every lookup is wasted latency.
ROOT_LISTING_TTL_S = 30.0
_root_listing_cache: dict = {}  # data_dir -> (monotonic_time, [(date, path)])
_root_listing_lock = threading.Lock()


def _format_worker_exception(context: str, exc: Exception):
    return f"{context}: {exc}\n{traceback.format_exc()}"


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


def _run_id_from_filename(name: str):
    try:
        return int(name.split("_")[0])
    except Exception:
        return None


def _parse_date_folder_name(name: str):
    try:
        return datetime.strptime(name, "%Y-%m-%d").date()
    except ValueError:
        return None


def _list_date_folders(data_dir: str):
    """Return [(date, full_path)] for every YYYY-MM-DD folder under data_dir,
    sorted ascending by date.  One directory read; no per-entry stat calls."""
    folders = []
    try:
        with os.scandir(data_dir) as entries:
            for entry in entries:
                if entry.name == "_lite":
                    continue
                folder_date = _parse_date_folder_name(entry.name)
                if folder_date is None:
                    continue
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                folders.append((folder_date, entry.path))
    except OSError:
        return []
    folders.sort(key=lambda item: item[0])
    return folders


def _list_date_folders_cached(data_dir: str, max_age_s: float = ROOT_LISTING_TTL_S):
    """_list_date_folders with a short-lived process-wide cache.  Pass
    max_age_s=0 to force a fresh listing (and refresh the cache)."""
    now = time.monotonic()
    with _root_listing_lock:
        hit = _root_listing_cache.get(data_dir)
        if hit is not None and now - hit[0] <= max_age_s:
            return list(hit[1])
    folders = _list_date_folders(data_dir)
    with _root_listing_lock:
        _root_listing_cache[data_dir] = (now, folders)
    return list(folders)


def _scan_hdf5_files(folder: str, with_stat: bool = True):
    """Return [(run_id, path, stat_or_None)] for *.hdf5 files in folder.

    Uses os.scandir so the run id, path and (on Windows) the stat result all
    come out of a single directory listing instead of one glob plus one
    os.stat round-trip per file -- a big win over a mapped network drive.
    """
    files = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                name = entry.name
                if not name.lower().endswith(".hdf5"):
                    continue
                run_id = _run_id_from_filename(name)
                if run_id is None:
                    continue
                try:
                    if not entry.is_file():
                        continue
                    stat_result = entry.stat() if with_stat else None
                except OSError:
                    continue
                files.append((run_id, entry.path, stat_result))
    except OSError:
        return []
    return files


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

    unique_sorted = np.unique(finite)
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


def _build_value_record(mode: str, name: str, dataset, max_full_read: int = PARAM_FULL_READ_MAX_ELEMENTS):
    """Build a param-search record for one HDF5 dataset.

    Small datasets are read once in full (value + stats).  Large ones only
    contribute a leading slice for the preview so that switching runs never
    drags an image stack over the network.
    """
    shape = str(tuple(int(dim) for dim in dataset.shape)) if dataset.shape else "()"
    dtype_name = str(dataset.dtype)
    size = int(dataset.size)

    full_value = None
    if size <= max_full_read:
        try:
            full_value = dataset[()]
        except Exception:
            full_value = None

    # h5py returns an ndarray, a numpy scalar, or bytes here; _stringify_value
    # handles all three directly.
    preview_source = full_value if full_value is not None else _preview_dataset_value(dataset)

    preview = _stringify_value(preview_source, max_chars=160)
    detail = _stringify_value(preview_source, max_chars=8000)

    if full_value is not None:
        stats_text = _value_summary(str(name), full_value)
        if stats_text:
            preview = _stringify_value(f"{stats_text} | {preview}", max_chars=160)
            detail = f"{stats_text}\n\n{detail}"
        if size > 8:
            detail += f"\n\nPreview truncated from dataset with shape {shape} and dtype {dtype_name}."
    else:
        detail += (
            f"\n\nDataset with shape {shape} and dtype {dtype_name} ({size:,} elements) "
            f"exceeds the {max_full_read:,}-element preview limit; only a leading slice was read."
        )

    return {
        "mode": mode,
        "name": str(name),
        "dtype": dtype_name,
        "shape": shape,
        "preview": preview,
        "detail": detail,
        "size": size,
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


# ---------------------------------------------------------------------------
# Fast directory-level lookups (no HDF5 opens)
# ---------------------------------------------------------------------------


def find_nearest_run_date_and_id(data_dir: str, requested_run_id: int, cache: MetadataCache | None = None,
                                 stats: dict | None = None):
    """Locate the run id closest to ``requested_run_id`` and its date folder.

    Run ids increase monotonically with date, so the date folders form a sorted
    sequence of run-id ranges and the search is a binary search over folders.
    With a populated folder index (see MetadataCache.update_folder_index and
    FolderIndexWorker) every probe of a folder older than
    FOLDER_INDEX_MUTABLE_DAYS is answered from memory, so a lookup costs the
    (cached) root listing plus at most one folder listing -- the one that has
    to be enumerated to pick the exact nearest id.  Without an index it
    degrades to ~log2(n_folders) listings.  Nothing here opens an HDF5 file,
    and no validity filtering is applied.

    ``stats`` (optional dict) receives {"listings": n} for diagnostics.
    """
    requested_run_id = int(requested_run_id)
    folders = _list_date_folders_cached(data_dir)
    if not folders:
        return None, None

    today = date.today()
    index = cache.folder_index() if cache is not None else {}
    listed = {}  # folder position -> sorted run ids
    listings = 0

    def trusted_entry(pos):
        folder_date = folders[pos][0]
        if (today - folder_date).days <= FOLDER_INDEX_MUTABLE_DAYS:
            return None
        return index.get(folder_date.isoformat())

    def folder_ids(pos):
        nonlocal listings
        if pos not in listed:
            folder_date, path = folders[pos]
            ids = sorted(run_id for run_id, _, _ in _scan_hdf5_files(path, with_stat=False))
            listings += 1
            listed[pos] = ids
            if cache is not None:
                cache.update_folder_index(folder_date.isoformat(), ids)
        return listed[pos]

    def bounds(pos):
        """(min_id, max_id) for the folder, or None if empty.  Served from the
        index when possible; lists the folder otherwise."""
        if pos in listed:
            ids = listed[pos]
            return (ids[0], ids[-1]) if ids else None
        entry = trusted_entry(pos)
        if entry is not None:
            lo, hi, n = entry
            return None if n == 0 or lo is None else (lo, hi)
        ids = folder_ids(pos)
        return (ids[0], ids[-1]) if ids else None

    def nearest_nonempty(mid, lo, hi):
        """Position of the non-empty folder closest to mid within [lo, hi], or None."""
        for step in range(0, hi - lo + 1):
            for probe in ((mid - step, mid + step) if step else (mid,)):
                if lo <= probe <= hi and bounds(probe) is not None:
                    return probe
        return None

    # Binary search for the last folder whose smallest id is <= requested.
    lo, hi = 0, len(folders) - 1
    candidate = None
    while lo <= hi:
        mid = nearest_nonempty((lo + hi) // 2, lo, hi)
        if mid is None:
            break
        if bounds(mid)[0] <= requested_run_id:
            candidate = mid
            lo = mid + 1
        else:
            hi = mid - 1

    def next_nonempty(pos):
        nxt = pos + 1
        while nxt < len(folders) and bounds(nxt) is None:
            nxt += 1
        return nxt if nxt < len(folders) else None

    def key(rid):
        return (abs(rid - requested_run_id), rid)

    best_id = None
    best_pos = None
    if candidate is None:
        # Requested id predates everything; the earliest non-empty folder wins.
        first = next_nonempty(-1)
        if first is not None:
            best_id = bounds(first)[0]
            best_pos = first
    else:
        cand_lo, cand_hi = bounds(candidate)
        if requested_run_id <= cand_hi:
            # Inside the candidate folder's range: list it to resolve gaps.
            ids = folder_ids(candidate)
            best_id = min(ids, key=key)
            best_pos = candidate
        else:
            # Between candidate.max and the next folder's min: both ends are
            # known from bounds, so no listing is needed.
            best_id, best_pos = cand_hi, candidate
            nxt = next_nonempty(candidate)
            if nxt is not None:
                nxt_lo = bounds(nxt)[0]
                if key(nxt_lo) < key(best_id):
                    best_id, best_pos = nxt_lo, nxt

    if stats is not None:
        stats["listings"] = listings
    if best_id is None:
        return None, None
    return best_id, folders[best_pos][0]


class FolderIndexWorker(QThread):
    """Background builder for the per-date-folder run-id index.

    Lists every date folder that is not yet indexed (newest first, one
    directory listing each, no HDF5 opens) and records (min_id, max_id, n) in
    the metadata cache.  Runs once per data dir after the first scan and is
    stopped while a scan is active so it never competes with HDF5 reads.
    """

    progress = pyqtSignal(int, int)  # indexed_so_far, total_to_index
    done = pyqtSignal(int, bool)  # folders_indexed, completed

    def __init__(self, data_dir: str, cache: MetadataCache):
        super().__init__()
        self.data_dir = data_dir
        self.cache = cache
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        indexed = 0
        completed = False
        try:
            folders = _list_date_folders_cached(self.data_dir, max_age_s=0)
            index = self.cache.folder_index()
            today = date.today()
            todo = [
                (folder_date, path)
                for folder_date, path in folders
                if folder_date.isoformat() not in index
                or (today - folder_date).days <= FOLDER_INDEX_MUTABLE_DAYS
            ]
            todo.sort(key=lambda item: item[0], reverse=True)
            total = len(todo)
            for folder_date, path in todo:
                if self._stop_requested:
                    break
                ids = [run_id for run_id, _, _ in _scan_hdf5_files(path, with_stat=False)]
                self.cache.update_folder_index(folder_date.isoformat(), ids)
                indexed += 1
                if indexed % 25 == 0:
                    self.progress.emit(indexed, total)
                    self.cache.save_if_dirty()
            completed = not self._stop_requested
            self.cache.save()
        except Exception as exc:
            LOGGER.error(_format_worker_exception("FolderIndexWorker failed", exc))
        self.done.emit(indexed, completed)


def find_newer_completed_run_id(data_dir: str, current_latest: int, is_completed=None):
    """Return the newest completed run id greater than ``current_latest``.

    Only files whose id is above ``current_latest`` are ever opened, so in the
    steady state (no new files) this costs one directory listing of the newest
    date folder and zero HDF5 opens.  ``is_completed(path) -> bool`` defaults to
    the browser's own attr/xvar completion check.
    """
    current_latest = int(current_latest)
    # Fresh listing: a new day's folder must be seen as soon as it appears.
    folders = _list_date_folders_cached(data_dir, max_age_s=0)
    if not folders:
        return None

    if is_completed is None:
        def is_completed(path):
            try:
                with h5py.File(path, "r", locking=False) as f:
                    rc = f.attrs.get("run_complete", None)
                    if rc is True:
                        return True
                    if rc is False:
                        return False
                    return _is_completed_run(f)
            except Exception:
                return False

    for _, folder in reversed(folders):
        files = _scan_hdf5_files(folder, with_stat=False)
        if not files:
            continue
        files.sort(key=lambda item: item[0], reverse=True)
        newest_here = files[0][0]
        if newest_here <= current_latest:
            return None
        for run_id, path, _ in files:
            if run_id <= current_latest:
                break
            if is_completed(path):
                return run_id
        if current_latest < 0:
            # Nothing loaded yet: only the newest populated folder matters.
            return None
    return None


class RunScanner:
    def __init__(self, data_dir: str, date_from: date, date_to: date, cache: MetadataCache | None = None):
        self.data_dir = data_dir
        self.date_from = date_from
        self.date_to = date_to
        self._lite_runs_by_date = {}
        self._cache = cache if cache is not None else MetadataCache(data_dir)
        self._lite_lock = threading.Lock()

    def scan(self, stop_requested=None):
        if not self.data_dir or not os.path.isdir(self.data_dir):
            return

        def should_stop():
            return bool(stop_requested is not None and stop_requested())

        uncached = []  # (filepath, stat_result) pairs not yet in cache
        try:
            # --- Fast pass: yield cached summaries immediately (preserves order) ---
            for folder in self._iter_date_folders():
                if should_stop():
                    return
                for _, filepath, stat_result in self._iter_hdf5_files(folder):
                    if stat_result is None:
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

                    uncached.append((filepath, stat_result))

            if not uncached:
                return

            # --- Parallel pass: read uncached files with a thread pool ---
            with ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as executor:
                future_to_stat = {
                    executor.submit(self._read_summary, fp): (fp, sr)
                    for fp, sr in uncached
                }
                for future in as_completed(future_to_stat):
                    if should_stop():
                        for pending in future_to_stat:
                            pending.cancel()
                        break
                    fp, sr = future_to_stat[future]
                    try:
                        summary = future.result()
                    except Exception:
                        continue
                    if summary is not None:
                        self._cache.put(summary, sr)
                        self._cache.save_if_dirty()
                        yield summary
        finally:
            self._cache.save()

    def _iter_date_folders(self):
        folders = [
            (folder_date, path)
            for folder_date, path in _list_date_folders_cached(self.data_dir, max_age_s=0)
            if self.date_from <= folder_date <= self.date_to
        ]
        folders.sort(key=lambda item: item[0], reverse=True)
        for _, full in folders:
            yield full

    def _iter_hdf5_files(self, folder):
        files = _scan_hdf5_files(folder, with_stat=True)
        files.sort(key=lambda item: item[0], reverse=True)
        # Free by-product of the scan: keep the run-id folder index current.
        self._cache.update_folder_index(os.path.basename(folder), [run_id for run_id, _, _ in files])
        return files

    def _read_summary(self, filepath):
        try:
            with h5py.File(filepath, "r") as f:
                xvarnames = _attr_to_str_list(f.attrs.get("xvarnames"))

                # Single-pass: validate completion and collect dims+details together.
                xvar_result = self._read_and_validate_xvars(f, xvarnames)
                if xvar_result is None:
                    return None
                xvardims, xvar_details = xvar_result

                run_id = self._read_run_id(f, filepath)
                run_date_str = _decode_str(
                    f.attrs.get("run_date_str", os.path.basename(os.path.dirname(filepath)))
                )
                run_datetime_str = _decode_str(f.attrs.get("run_datetime_str", ""))

                experiment_name, experiment_filepath = self._read_experiment_info(f)
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
                    xvar_details=xvar_details,
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

    def _read_and_validate_xvars(self, h5file, xvarnames):
        """Single-pass: validate run completion, return (xvardims, xvar_details) or None if incomplete."""
        filtered = [name for name in xvarnames if str(name).strip()]
        if not filtered:
            return (), []
        if "params" not in h5file:
            return None
        dims = []
        details = []
        for name in filtered:
            if name not in h5file["params"]:
                return None
            values = np.asarray(h5file["params"][name][()])
            if values.ndim == 0:
                return None
            flat = values.reshape(-1)
            dims.append(int(flat.size))
            details.append(_summarize_xvar_values(name, flat))
        return tuple(dims), details

    def _read_xvardims_and_details(self, h5file, xvarnames):
        """Kept for external callers; delegates to _read_and_validate_xvars."""
        result = self._read_and_validate_xvars(h5file, xvarnames)
        if result is None:
            return (), []
        return result

    def _read_xvardims(self, h5file, xvarnames):
        dims, _ = self._read_xvardims_and_details(h5file, xvarnames)
        return dims

    def _read_run_id(self, h5file, filepath):
        run_id_attr = h5file.attrs.get("run_id", None)
        if run_id_attr is None:
            return int(os.path.basename(filepath).split("_")[0])
        if isinstance(run_id_attr, np.ndarray):
            return int(run_id_attr[()])
        return int(run_id_attr)

    def _has_lite_copy(self, run_id: int, run_date_str: str):
        with self._lite_lock:
            if run_date_str not in self._lite_runs_by_date:
                self._lite_runs_by_date[run_date_str] = self._index_lite_runs_for_date(run_date_str)
            return run_id in self._lite_runs_by_date[run_date_str]

    def _index_lite_runs_for_date(self, run_date_str: str):
        lite_day_dir = os.path.join(self.data_dir, "_lite", run_date_str)
        run_ids = set()
        try:
            with os.scandir(lite_day_dir) as entries:
                for entry in entries:
                    name = entry.name
                    if "_lite_" not in name or not name.lower().endswith(".hdf5"):
                        continue
                    run_id = _run_id_from_filename(name)
                    if run_id is not None:
                        run_ids.add(run_id)
        except OSError:
            return set()
        return run_ids


class ScanWorker(QThread):
    run_found = pyqtSignal(object)
    run_batch_found = pyqtSignal(list)
    scan_done = pyqtSignal(int)
    scan_error = pyqtSignal(str)

    def __init__(self, scanner: RunScanner, batch_size: int = 256):
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
            for run_summary in self.scanner.scan(stop_requested=lambda: self._stop_requested):
                if self._stop_requested:
                    break
                batch.append(run_summary)
                count += 1
                if len(batch) >= self.batch_size:
                    self.run_batch_found.emit(batch)
                    batch = []
            if self._stop_requested:
                return
            if batch:
                self.run_batch_found.emit(batch)
            self.scan_done.emit(count)
        except Exception as exc:
            message = _format_worker_exception("ScanWorker failed", exc)
            LOGGER.error(message)
            self.scan_error.emit(message)


def _unique_preserve_order(items):
    """O(n) order-preserving unique, replacing the previous O(n^2) scan."""
    seen = set()
    unique_items = []
    for item in items:
        key = item
        if isinstance(item, float) and item != item:  # NaN never equals itself
            key = "__nan__"
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    return unique_items


def _summarize_xvar_values(name: str, values) -> dict:
    """Summarize a single xvar array into a detail dict.

    This is a module-level function so that both the scan path
    (RunScanner._read_summary) and the async fallback
    (XvarDetailLoader) can call it without duplicating logic.
    """
    array = np.asarray(values)
    if array.size == 0:
        unit, _, _ = detect_unit(xvarnames=[name], xvar_idx=0, xvar_values=array)
        return {"name": name, "unit": unit or "", "n": 0, "min": "NA", "max": "NA", "preview": "-"}

    flat = array.reshape(-1)
    n = int(flat.size)
    unit, multiplier, _ = detect_unit(xvarnames=[name], xvar_idx=0, xvar_values=flat)
    unit = unit or ""

    def format_numeric_value(value, decimals):
        return _format_numeric_value(value, multiplier, decimals)

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
        decimals = _decimals_from_spacing(scaled_vals)
        preview_values = _unique_preserve_order(flat.tolist())
        return {
            "name": name,
            "unit": unit,
            "n": n,
            "min": format_numeric_value(min_val, decimals),
            "max": format_numeric_value(max_val, decimals),
            "preview": format_preview_text(preview_values, lambda value: format_numeric_value(value, decimals)),
        }

    as_text = [_decode_str(item) for item in flat]
    preview_values = _unique_preserve_order(as_text)
    return {
        "name": name,
        "unit": unit,
        "n": n,
        "min": min(as_text),
        "max": max(as_text),
        "preview": format_preview_text(preview_values),
    }


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
                    if self.isInterruptionRequested():
                        return

                    if name not in params_group:
                        details.append({"name": name, "unit": "", "n": 0, "min": "NA", "max": "NA"})
                        continue

                    values = params_group[name][()]
                    details.append(_summarize_xvar_values(name, values))

            if not self.isInterruptionRequested():
                self.details_ready.emit(details)
        except Exception as exc:
            if not self.isInterruptionRequested():
                message = _format_worker_exception(f"XvarDetailLoader failed for {self.filepath}", exc)
                LOGGER.error(message)
                self.details_error.emit(message)


class ParamSearchLoader(QThread):
    records_ready = pyqtSignal(dict)
    partial_records_ready = pyqtSignal(str, list)  # (mode, records) — emitted per group as soon as ready
    load_error = pyqtSignal(str)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        records = {mode: [] for mode in PARAM_SEARCH_MODES}
        try:
            with h5py.File(self.filepath, "r") as f:
                for mode in PARAM_SEARCH_MODES:
                    if self.isInterruptionRequested():
                        return
                    if mode == "data":
                        loaded = self._load_data_records(f)
                    else:
                        loaded = self._load_group_records(f, mode)
                    if loaded is None or self.isInterruptionRequested():
                        return
                    records[mode] = loaded
                    self.partial_records_ready.emit(mode, loaded)
            self.records_ready.emit(records)
        except Exception as exc:
            if not self.isInterruptionRequested():
                message = _format_worker_exception(f"ParamSearchLoader failed for {self.filepath}", exc)
                LOGGER.error(message)
                self.load_error.emit(message)

    def _load_group_records(self, h5file, group_name: str):
        if group_name not in h5file:
            return []

        records = []
        group = h5file[group_name]
        for key in sorted(group.keys()):
            if self.isInterruptionRequested():
                return None
            item = group[key]
            if not isinstance(item, h5py.Dataset):
                continue
            records.append(_build_value_record(group_name, key, item))
        return records

    def _load_data_records(self, h5file):
        if "data" not in h5file:
            return []

        records = []
        group = h5file["data"]
        for key in sorted(group.keys()):
            if self.isInterruptionRequested():
                return None
            item = group[key]
            if isinstance(item, h5py.Group):
                child_keys = sorted(item.keys())
                records.append(
                    {
                        "mode": "data",
                        "name": str(key),
                        "dtype": "group",
                        "shape": "-",
                        "preview": f"group with {len(child_keys)} entries",
                        "detail": f"HDF5 group '{key}' with children: {', '.join(child_keys) or '(none)'}",
                        "size": len(child_keys),
                    }
                )
                continue

            if key in PARAM_SEARCH_NO_READ_KEYS:
                shape = str(tuple(int(dim) for dim in item.shape)) if item.shape else "()"
                records.append(
                    {
                        "mode": "data",
                        "name": str(key),
                        "dtype": str(item.dtype),
                        "shape": shape,
                        "preview": f"shape {shape} (not read)",
                        "detail": f"Dataset '{key}' with shape {shape} and dtype {item.dtype} is not previewed by the param search.",
                        "size": int(item.size),
                    }
                )
                continue

            records.append(_build_value_record("data", key, item))
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
            message = _format_worker_exception(f"LiteCreateWorker failed for run {self.run_id}", exc)
            LOGGER.error(message)
            self.error.emit(message)


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
            message = _format_worker_exception(f"BatchLiteCreateWorker failed for runs {self.run_ids}", exc)
            LOGGER.error(message)
            self.error.emit(message)


class AnnotationWriteWorker(QThread):
    """Write browser_tags / browser_comment attrs to one or more HDF5 files
    off the GUI thread (opening a file in append mode over the network can
    stall for a noticeable fraction of a second)."""

    written = pyqtSignal(int)  # run_id
    error = pyqtSignal(int, str)  # run_id, message
    completed = pyqtSignal(int, int)  # ok_count, total

    def __init__(self, jobs: list[tuple[int, str, object, object]]):
        """jobs: [(run_id, filepath, tags_or_None, comment_or_None)]"""
        super().__init__()
        self.jobs = list(jobs)

    def run(self):
        ok = 0
        for run_id, filepath, tags, comment in self.jobs:
            try:
                with h5py.File(filepath, "a") as f:
                    if tags is not None:
                        f.attrs["browser_tags"] = json.dumps(list(tags))
                    if comment is not None:
                        f.attrs["browser_comment"] = str(comment)
                ok += 1
                self.written.emit(int(run_id))
            except Exception as exc:
                self.error.emit(int(run_id), str(exc))
        self.completed.emit(ok, len(self.jobs))
