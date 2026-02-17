import numpy as np

def remove_infnan(*arrays):
    """
    Accepts any number of numpy arrays, finds indices with NaN or Inf,
    constructs a mask to remove those elements from all arrays,
    and returns the masked arrays.
    """
    if not arrays:
        raise ValueError("At least one array must be provided")
    # Stack arrays to find invalid values across all arrays
    stacked = np.vstack(arrays)
    # Identify indices where any array contains NaN or Inf
    invalid_mask = np.any(np.isnan(stacked) | np.isinf(stacked), axis=0)
    # Filter out invalid elements
    masked_arrays = tuple(arr[~invalid_mask] for arr in arrays)
    return masked_arrays

def normalize(array,
              map_minimum_to_zero=False,
              override_normalize_maximum=None,
              override_normalize_minimum=None):
    x = np.asarray(array)

    if override_normalize_maximum != None:
        x_max = override_normalize_maximum
    else:
        x_max = np.max(x)

    if override_normalize_minimum != None:
        x_min = override_normalize_minimum
    else:
        x_min = np.min(x)

    if map_minimum_to_zero:
        x = (x-x_min)/(x_max-x_min)
    else:
        x = x/x_max
    return x

def rm_outliers(array,
                outlier_method='mean',
                outlier_threshold=0.3,
                return_outlier_mask = True,
                return_outlier_idx = False,
                return_good_data = False,
                return_good_data_idx = False):
    
    x = array
    
    if outlier_method == 'mean':
        mask = np.abs(x/np.mean(x) - 1) < outlier_threshold
    elif outlier_method == 'std':
        mask = np.abs(x - np.mean(x)) < (np.std(x) * outlier_threshold)
    else:
        raise ValueError("`outlier_method` must be either 'mean' or 'std'")
    
    out = ()
    if return_outlier_mask:
        out += (mask,)
    if return_outlier_idx:
        outlier_idx = np.arange(len())[~mask].astype(int)
        out += (outlier_idx,)
    if return_good_data:
        out += (x[mask],)
    if return_good_data_idx:
        good_idx = np.arange(len())[mask].astype(int)
        out += (good_idx,)

    if len(out) == 1:
        out = out[0]

    return out

def rms(x):
    return np.sqrt(np.sum(x**2)/len(x))

def crop_array_by_index(array, include_idx=[0, -1], exclude_idx=None):
    """
    Crops a numpy array to include elements between the indices in `include_idx`,
    and excludes elements at indices specified in `exclude_idx` (relative to the original array).

    Args:
        array (array-like): The array to be cropped.
        include_idx (tuple or list, optional): Start and end indices (inclusive start, exclusive end).
            Defaults to (0, -1). -1 as end includes the last element.
        exclude_idx (list or None, optional): Indices to exclude from the result, relative to the original array.

    Returns:
        np.ndarray: Cropped array with specified elements removed.
    """
    array = np.asarray(array)
    n = len(array)
    start = include_idx[0]
    end = include_idx[1]
    if end == -1:
        end = n
    elif end < 0:
        end = n + end + 1
    else:
        end = int(end)
    # Get indices to keep
    indices = np.arange(start, end)
    if exclude_idx:
        exclude_idx = np.array(exclude_idx)
        # Only exclude indices that are within the selected range
        mask = ~np.isin(indices + start, exclude_idx)
        indices = indices[mask]
    return array[indices]

def find_n_max_indices(arr, N):
    """Find the indices of the N maximum values in a numpy ndarray."""
    if N > arr.size:
        raise ValueError("N cannot be greater than the number of elements in the array.")
    
    # Get the indices of the top N values
    indices = np.argpartition(arr.flatten(), -N)[-N:]  # Unsorted top N indices
    sorted_indices = indices[np.argsort(-arr.flatten()[indices])]  # Sort indices by value
    
    # Convert back to multi-dimensional indices
    return [tuple(idx) for idx in np.array(np.unravel_index(sorted_indices, arr.shape)).T]

def get_repeat_std_error(array,N_repeats):
    if isinstance(N_repeats,np.ndarray):
        N_repeats = N_repeats[0]
        
    Nr = N_repeats
    means = np.mean(np.reshape(array,(-1,Nr)),axis=1)
    std_error = np.std(np.reshape(array,(-1,Nr)),axis=1)/np.sqrt(Nr)

    return means, std_error

def ensure_ndarray(var, enforce_1d=True):
    """Ensures that the input is a numpy ndarray. If the input is a float,
    int, or list, it converts it to a 1D numpy array. If the input is already
    a numpy ndarray, it returns it as is. If enforce_1d is True, it raises an error
    if the input is more than 1-dimensional.
    """
    if isinstance(var, (float, int)):
        arr = np.array([var])
    elif isinstance(var, list) or isinstance(var,range):
        arr = np.array(var)
    elif isinstance(var, np.ndarray):
        arr = var
    else:
        raise TypeError("Input must be float, int, list, or ndarray")
    if arr.ndim > 1 and enforce_1d:
        raise ValueError("Input array must be at most 1-dimensional")
    return arr

def remove_element_by_index(data, index):
    """Removes the element at the specified index from data, which can
    be a list or a numpy array."""
    if isinstance(data, list):
        del data[index]
    elif isinstance(data, np.ndarray):
        data = np.delete(data, index)
    return data