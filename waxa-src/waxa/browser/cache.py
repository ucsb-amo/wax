import json
import os
import threading
import time
from typing import Optional

from .run_summary import RunSummary


class MetadataCache:
    """On-disk cache of RunSummary objects keyed by file path.

    Thread-safe: a single instance is shared across successive scans (and can be
    hit by several scanner worker threads at once), so every public method takes
    the internal lock.  Writes are atomic (temp file + replace) so a crash or a
    second browser instance never leaves a truncated JSON file behind.
    """

    VERSION = 2
    FILENAME = ".waxa_browser_cache.json"
    MIN_SAVE_INTERVAL_S = 5.0

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.cache_path = os.path.join(data_dir, self.FILENAME) if data_dir else ""
        self._entries = {}
        self._dirty = False
        self._last_save_time = 0.0
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not self.cache_path or not os.path.isfile(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return

        if payload.get("version") != self.VERSION:
            return

        entries = payload.get("entries", {})
        if isinstance(entries, dict):
            self._entries = entries

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def get(self, filepath: str, stat_result: os.stat_result) -> Optional[RunSummary]:
        with self._lock:
            entry = self._entries.get(filepath)
        if not entry:
            return None
        if entry.get("mtime_ns") != stat_result.st_mtime_ns:
            return None
        if entry.get("size") != stat_result.st_size:
            return None
        summary_payload = entry.get("summary")
        if not isinstance(summary_payload, dict):
            return None
        try:
            return RunSummary.from_cache_dict(summary_payload)
        except Exception:
            return None

    def put(self, summary: RunSummary, stat_result: os.stat_result):
        if not self.cache_path:
            return
        entry = {
            "mtime_ns": stat_result.st_mtime_ns,
            "size": stat_result.st_size,
            "summary": summary.to_cache_dict(),
        }
        with self._lock:
            self._entries[summary.filepath] = entry
            self._dirty = True

    def update_summary(self, summary: RunSummary):
        """Refresh the cached payload for a file whose stat did not change
        (e.g. tags/comments edited in place by the browser itself)."""
        if not self.cache_path:
            return
        with self._lock:
            entry = self._entries.get(summary.filepath)
            if entry is None:
                return
            entry["summary"] = summary.to_cache_dict()
            self._dirty = True

    def invalidate(self, filepath: str):
        with self._lock:
            if filepath in self._entries:
                del self._entries[filepath]
                self._dirty = True

    def save(self):
        with self._lock:
            if not self._dirty or not self.cache_path:
                return
            payload = {
                "version": self.VERSION,
                "entries": self._entries,
            }
            # Serialise under the lock (entries may be mutated concurrently),
            # but the JSON encode itself is the expensive bit; keep it here so
            # a concurrent put() cannot corrupt the dict mid-dump.
            try:
                text = json.dumps(payload)
            except Exception:
                return
            self._dirty = False
            self._last_save_time = time.monotonic()

        tmp_path = f"{self.cache_path}.{os.getpid()}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp_path, self.cache_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            with self._lock:
                self._dirty = True

    def save_if_dirty(self, min_interval_s: float | None = None):
        """Save when dirty, but no more often than ``min_interval_s``.

        Called periodically during long scans so partial results survive a
        crash without re-writing a multi-megabyte JSON file every few entries.
        """
        interval = self.MIN_SAVE_INTERVAL_S if min_interval_s is None else float(min_interval_s)
        with self._lock:
            if not self._dirty:
                return
            if time.monotonic() - self._last_save_time < interval:
                return
        self.save()
