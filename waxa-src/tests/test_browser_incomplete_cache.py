"""The data browser remembers unfinished runs so a scan does not re-open them.

All files are fakes in a temp dir; nothing touches the data drive.
"""

import json
import os
import time
from datetime import date

import h5py
import numpy as np

from waxa.browser import scanner as sc
from waxa.browser.cache import MetadataCache

DAY = date(2026, 8, 25)
HOUR = 3600.0


def _write_run(root, run_id, complete=True, age_s=2 * HOUR):
    folder = os.path.join(root, DAY.isoformat())
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{run_id:07d}_2026-08-25_12-00-00_test.hdf5")
    with h5py.File(path, "w") as f:
        f.attrs["xvarnames"] = ["t_tof"]
        f.attrs["run_id"] = run_id
        f.attrs["run_complete"] = complete
        if complete:
            f.create_dataset("params/t_tof", data=np.linspace(1e-3, 5e-3, 5))
        else:
            f.create_dataset("params/t_tof", data=1e-3)  # scalar until END_RUN
    mtime = time.time() - age_s
    os.utime(path, (mtime, mtime))
    return path


def _scan(root, cache, monkeypatch):
    """Run a scan; return (run ids found, paths opened by the scanner)."""
    opened = []
    real = sc.RunScanner._read_summary_status

    def counting(self, filepath):
        opened.append(filepath)
        return real(self, filepath)

    monkeypatch.setattr(sc.RunScanner, "_read_summary_status", counting)
    found = [s.run_id for s in sc.RunScanner(root, DAY, DAY, cache=cache).scan()]
    return sorted(found), opened


def test_old_unfinished_run_is_read_once(tmp_path, monkeypatch):
    d = str(tmp_path)
    done = _write_run(d, 100, complete=True)
    unfinished = _write_run(d, 101, complete=False)
    cache = MetadataCache(d)

    found, opened = _scan(d, cache, monkeypatch)
    assert found == [100]
    assert sorted(opened) == sorted([done, unfinished])

    found, opened = _scan(d, cache, monkeypatch)
    assert found == [100]
    assert opened == []

    # Survives a reload of the cache file (a new browser session).
    found, opened = _scan(d, MetadataCache(d), monkeypatch)
    assert found == [100]
    assert opened == []


def test_recent_unfinished_run_is_read_every_time(tmp_path, monkeypatch):
    d = str(tmp_path)
    path = _write_run(d, 200, complete=False, age_s=60.0)
    cache = MetadataCache(d)
    for _ in range(2):
        found, opened = _scan(d, cache, monkeypatch)
        assert found == []
        assert opened == [path]


def test_run_that_finishes_later_appears(tmp_path, monkeypatch):
    d = str(tmp_path)
    _write_run(d, 300, complete=False)
    cache = MetadataCache(d)
    assert _scan(d, cache, monkeypatch)[0] == []

    path = _write_run(d, 300, complete=True)  # rewritten -> new mtime/size
    found, opened = _scan(d, cache, monkeypatch)
    assert found == [300]
    assert opened == [path]


def test_unreadable_file_is_not_remembered(tmp_path, monkeypatch):
    d = str(tmp_path)
    folder = os.path.join(d, DAY.isoformat())
    os.makedirs(folder)
    path = os.path.join(folder, "0000400_2026-08-25_12-00-00_broken.hdf5")
    with open(path, "wb") as fh:
        fh.write(b"not an hdf5 file")
    old = time.time() - 2 * HOUR
    os.utime(path, (old, old))
    cache = MetadataCache(d)
    for _ in range(2):
        found, opened = _scan(d, cache, monkeypatch)
        assert found == []
        assert opened == [path]


def test_unfinished_entry_is_invisible_to_get(tmp_path, monkeypatch):
    """An older browser reading the shared cache file only uses get() and the
    'summary' key; an unfinished-run entry must look like a miss to it."""
    d = str(tmp_path)
    path = _write_run(d, 500, complete=False)
    cache = MetadataCache(d)
    _scan(d, cache, monkeypatch)

    st = os.stat(path)
    assert cache.get(path, st) is None
    assert cache.is_known_incomplete(path, st)
    with open(cache.cache_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["version"] == 2
    assert "summary" not in payload["entries"][path]
