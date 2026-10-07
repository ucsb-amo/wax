"""A run file atomdata can load, made, filled and finished the way liveOD
does it (see end_run_helpers): no camera images, the data containers given.
Kept out of the test files so each of them stays small."""
import numpy as np
from end_run_helpers import make_run_file, write_during_run, finish_run


def make_loadable_run(folder, containers, xvar=(1., 2., 3., 4.), params=None,
                      run_id=85001, date="2026-10-05", xvarname="t"):
    """``containers`` maps key -> array shaped (n_shots, ...); ``xvar`` is the
    scanned value per shot (repeats are equal consecutive values); ``params``
    are extra entries for the params group. Returns the file path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = str(folder / f"{run_id:07d}_{date}_12-00-00_FakeExpt.hdf5")
    n = len(xvar)
    p = {xvarname: [float(v) for v in xvar]}
    p.update(params or {})
    shapes = {k: {"shape": tuple(v.shape), "dtype": str(v.dtype), "external": True, "fill": 0}
              for k, v in containers.items()}
    init = {"capture_images": False, "xvarnames": [xvarname], "sort_idx": [], "sort_N": [],
            "datavault_shapes": shapes, "params": p, "camera_params": {},
            "run_date_str": date, "run_datetime_str": f"{date}_12-00-00",
            "expt_class": "FakeExpt"}
    ds = make_run_file(path, init, run_id=run_id)
    write_during_run(path, {k: {i: v[i] for i in range(n)} for k, v in containers.items()})
    end = {"params": p,
           "datavault": {k: {"data": None, "data_gotten": True, "external": True,
                             "final_order": True} for k in containers},
           "sort_idx": [], "sort_N": [], "xvardims": [n], "N_shots_with_repeats": n,
           "N_pwa_per_shot": 1, "capture_images": False, "scope_data_taken": False,
           "scope_data": [], "expt_filepath": "", "expt_file_text": "",
           "params_file_text": "", "base_class_texts": {}, "extra_file_texts": {}}
    finish_run(ds, path, end, shot_timestamps=[float(i + 1) for i in range(n)])
    return path
