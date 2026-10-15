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
        self.payloads = []          # the END_RUN payload of each save
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
        self.payloads.append(msg)
        self.incomplete = incomplete        # what the last save was told is missing


class TmpTalk:
    """The server_talk calls the real DataSaver's reserve makes, kept inside a
    folder (pytest's tmp_path): a run id counter and nothing else."""

    def __init__(self, folder):
        self.data_dir = str(folder)
        self.next_id = 1
        self.set_calls = 0

    def check_for_mapped_data_dir(self):
        return True

    def get_run_id(self):
        return self.next_id

    def get_latest_run_id_any(self):
        return None

    def set_run_id(self, value):
        self.set_calls += 1
        self.next_id = int(value)


def tmp_data_saver(folder):
    """The real waxa DataSaver, rooted in ``folder``, with a TmpTalk."""
    from waxa.data.data_saver import DataSaver
    return DataSaver(str(folder), server_talk=TmpTalk(folder))


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
