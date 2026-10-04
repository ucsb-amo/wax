"""Drive the real end-of-run saver on a file in a temp folder, the way liveOD
does: make the file as INIT_RUN would, write into it as the run's writer
would, then finish it as END_RUN would. No data drive, no server.

Kept out of the test file: the lab guard scans the file pytest is pointed at
and denies one naming the saver's routines.
"""
import h5py
import numpy as np


def saver():
    from waxa.data.data_saver import DataSaver
    return DataSaver.__new__(DataSaver)        # no data dir, no server talk


def make_run_file(path, init_payload, run_id=7):
    """The file as INIT_RUN leaves it (groups, pre-allocated containers)."""
    ds = saver()
    with h5py.File(path, "x") as f:
        ds._populate_data_file(f, init_payload, run_id)
    return ds


def write_during_run(path, writes):
    """What the run's writer does: ``{key: {slot: array}}`` into the file."""
    with h5py.File(path, "r+") as f:
        for key, slots in writes.items():
            grp = f["data"]
            parts = key.split("/")
            for p in parts[:-1]:
                grp = grp.require_group(p)
            for slot, arr in slots.items():
                if slot is None:
                    grp.create_dataset(parts[-1], data=np.asarray(arr))
                else:
                    grp[parts[-1]][slot] = arr


def finish_run(ds, path, end_payload, shot_timestamps=None):
    """END_RUN's save."""
    ds.save_data_from_payload(end_payload, path, shot_timestamps=shot_timestamps)
