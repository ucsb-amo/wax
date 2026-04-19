import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from waxa.fitting import GaussianFit

# Module-level worker function for thread pool mapping.
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
    
    # Create a worker function with xaxis bound via partial
    worker = partial(_fit_one_worker, xaxis=xaxis)

    # Fit each summed OD independently. For larger batches, use ThreadPoolExecutor;
    # for smaller batches, use serial to avoid scheduling overhead.
    use_parallel = total >= 8

    if use_parallel:
        max_workers = min(max(os.cpu_count() or 1, 1), total)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(worker, sum_dist_list))
        for i, fit in enumerate(results):
            if fit is None:
                error_count += 1
            fits[i] = fit
    else:
        for i in range(total):
            fit = worker(sum_dist_list[i])
            if fit is None:
                error_count += 1
            fits[i] = fit

    # reshape back to the n-dim'nal sumOD shape
    fits = fits.reshape(sum_dist.shape[:-1])

    if error_count > 0:
        print(f"{error_count}/{total} fits failed")

    return fits