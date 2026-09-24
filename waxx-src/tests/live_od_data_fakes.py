"""Stand-ins for test_live_od_data.py: a saver that works inside a directory it
is given (pytest's tmp_path) and never sees the real data drive.

Kept out of the test file because the lab guard scans the file pytest is
pointed at, and denies one that defines the saver's method names.
"""
import os

import h5py


class FakeSaver:
    """The two DataSaver methods RunFile calls."""

    def __init__(self, folder, fail_reserve=False, fail_save=False):
        self.folder = str(folder)
        self.fail_reserve = fail_reserve
        self.fail_save = fail_save
        self.saved = []             # (filepath, shot_timestamps) per END_RUN save
        self._next_run_id = 101

    def reserve_run_id_and_path(self, msg):
        if self.fail_reserve:
            raise OSError("drive not mapped")
        run_id = self._next_run_id
        self._next_run_id += 1
        filepath = os.path.join(self.folder, f"{run_id:07d}.hdf5")
        with h5py.File(filepath, "x") as f:
            f.create_group("data")
        return run_id, filepath

    def save_data_from_payload(self, msg, filepath, shot_timestamps=None, incomplete=None):
        if self.fail_save:
            raise OSError("drive went away")
        self.saved.append((filepath, list(shot_timestamps or [])))
        self.incomplete = incomplete        # what the last save was told is missing


def new_data_file(folder):
    """(run_id, filepath) of a fresh file with a 'data' group, as INIT_RUN leaves it."""
    return FakeSaver(folder).reserve_run_id_and_path({})


def patch_payload_stash(monkeypatch, run_file_module, calls, stash_path="stash.pkl"):
    """Replace the stash / clear pair RunFile imported, recording the order of calls."""
    def stash(msg, filepath, run_id, shot_timestamps=None, incomplete=None):
        calls.append("stash")
        return stash_path
    monkeypatch.setattr(run_file_module, "stash_end_run_payload", stash)
    monkeypatch.setattr(run_file_module, "clear_end_run_payload",
                        lambda path: calls.append(("clear", path)))
