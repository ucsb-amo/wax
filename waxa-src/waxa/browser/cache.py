import json
import os
from typing import Optional

from .run_summary import RunSummary


class MetadataCache:
    VERSION = 1
    FILENAME = ".waxa_browser_cache.json"

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.cache_path = os.path.join(data_dir, self.FILENAME) if data_dir else ""
        self._entries = {}
        self._dirty = False
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

    def get(self, filepath: str, stat_result: os.stat_result) -> Optional[RunSummary]:
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
        self._entries[summary.filepath] = {
            "mtime_ns": stat_result.st_mtime_ns,
            "size": stat_result.st_size,
            "summary": summary.to_cache_dict(),
        }
        self._dirty = True

    def save(self):
        if not self._dirty or not self.cache_path:
            return
        payload = {
            "version": self.VERSION,
            "entries": self._entries,
        }
        try:
            with open(self.cache_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception:
            return
        self._dirty = False
