import numpy as np
import os
from joblib import Parallel, delayed
from waxa.fitting import GaussianFit

# Minimum number of fits to justify spawning a process pool.
# Below this threshold the pool startup cost exceeds the parallelism gain.
N_PROC_THRESHOLD = 64

# Module-level worker function (must be importable by loky workers).
def _fit_one_worker(this_sum_dist, xaxis):
    try:
        fit = GaussianFit(xaxis, this_sum_dist, print_errors=False)
        return fit
    except Exception:
        return None

def fit_gaussian_sum_dist(sum_dist: np.ndarray, camera_params) -> list[GaussianFit]:
    '''
    Performs a guassian fit on each summedOD in the input list.

    Returns a GaussianFit object which contains the fit parameters.

    Length fit parameters are returned in units of meters. Amplitude and offset
    are in raw summedOD units.

    Parameters
    ----------
    summedODs: ArrayLike
        A list of summedODs.

    Returns
    -------
    fits: ArrayLike
        An array of GaussianFit objects.
    '''

    xaxis = camera_params.pixel_size_m / camera_params.magnification * np.arange(sum_dist.shape[-1])
    
    # reshape to effectively a list of sumODs:
    sum_dist_list = sum_dist.reshape(-1, sum_dist.shape[-1])
    # prep a fit array of len == # sum ODs
    fits = np.empty(sum_dist_list.shape[:-1], dtype=GaussianFit)

    error_count = 0
    total = sum_dist_list.shape[0]

    # For large batches use joblib's loky backend (true parallelism, and unlike
    # ProcessPoolExecutor it works correctly in Jupyter/IPython on Windows).
    # For small batches the process-pool startup cost exceeds the gain.
    if total >= N_PROC_THRESHOLD:
        n_workers = min(os.cpu_count() or 1, total)
        results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_fit_one_worker)(arr, xaxis) for arr in sum_dist_list
        )
        for i, fit in enumerate(results):
            if fit is None:
                error_count += 1
            fits[i] = fit
    else:
        for i in range(total):
            fit = _fit_one_worker(sum_dist_list[i], xaxis)
            if fit is None:
                error_count += 1
            fits[i] = fit

    # reshape back to the n-dim'nal sumOD shape
    fits = fits.reshape(sum_dist.shape[:-1])

    if error_count > 0:
        print(f"{error_count}/{total} fits failed")

    return fits