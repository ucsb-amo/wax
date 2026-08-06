import numpy as np

def key_from_attribute(obj, value, first_match_only=True, check_duplicates=False):
    """
    Returns the attribute name (key) of `obj` whose value exactly matches `value`.
    
    Handles numpy arrays, scalars, and other types. Returns None if no match found.
    
    Args:
        obj: Object to search through
        value: The value to match (any data type)
        first_match_only: If True, return on first match (faster). 
                         If False, check for duplicates.
        check_duplicates: If True, raise ValueError if multiple attributes match.
                         Ignored if first_match_only=True.
        
    Returns:
        str: The attribute name, or None if no exact match found
        
    Raises:
        ValueError: If check_duplicates=True and multiple attributes match
    """
    matches = [] if check_duplicates else None
    
    for key, val in vars(obj).items():
        if key.startswith('_'):  # Skip private attributes
            continue
            
        try:
            # Handle numpy arrays
            if isinstance(val, np.ndarray) and isinstance(value, np.ndarray):
                if np.array_equal(val, value):
                    if first_match_only:
                        return key
                    elif check_duplicates:
                        matches.append(key)
                    else:
                        return key
            # Handle scalars and other types
            elif val == value:
                if first_match_only:
                    return key
                elif check_duplicates:
                    matches.append(key)
                else:
                    return key
        except (ValueError, TypeError):
            pass
    
    if check_duplicates and matches and len(matches) > 1:
        raise ValueError(f"Multiple attributes match: {matches}")
    
    return matches[0] if (check_duplicates and matches) else None

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
              override_normalize_minimum=None,
              axis=None,
              return_map=False):
    """
    Normalizes an array to a maximum of 1, optionally mapping its minimum to 0.

    Args:
        array (array-like): 1D or 2D array to normalize.
        map_minimum_to_zero (bool, optional): If True, maps the minimum to 0 and
            the maximum to 1, i.e. (x - x_min) / (x_max - x_min). If False, only
            divides by the maximum. Defaults to False.
        override_normalize_maximum (scalar or array-like, optional): Value(s) to
            use in place of the computed maximum. See `axis` for shape rules.
        override_normalize_minimum (scalar or array-like, optional): Value(s) to
            use in place of the computed minimum. See `axis` for shape rules.
        axis (int or None, optional): If None, the whole array is normalized by a
            single max/min, and any overrides must be scalar. If 0, each column is
            normalized individually; if 1, each row is. Overrides may then be
            either a scalar (same value used for every vector) or a vector with
            one entry per normalized vector (length = array.shape[1] for axis=0,
            array.shape[0] for axis=1). Ignored for 1D input. Defaults to None.
        return_map (bool, optional): If True, also returns the pair of functions
            (to_normalized, to_raw) implementing this normalization and its
            inverse, so other data can be mapped with the same scaling.
            Defaults to False.

    Returns:
        np.ndarray: Normalized array, same shape as the input. If `return_map`
            is True, returns (normalized_array, (to_normalized, to_raw)).

    Raises:
        ValueError: If the input has more than 2 dimensions, if `axis` is not
            None, 0, or 1, or if an override has a shape incompatible with `axis`.
    """
    x = np.asarray(array)

    if x.ndim > 2:
        raise ValueError(f"`array` must be at most 2-dimensional (got {x.ndim} dimensions)")
    if axis not in (None, 0, 1):
        raise ValueError("`axis` must be None, 0, or 1")

    # axis is meaningless for a 1D array -- normalize the whole thing
    if x.ndim < 2:
        axis = None

    def _resolve(override, reduce_func):
        if override is None:
            value = reduce_func(x) if axis is None else reduce_func(x, axis=axis)
        else:
            value = np.asarray(override)
            if axis is None:
                if value.ndim != 0:
                    raise ValueError("Overrides must be scalar when `axis` is None")
            elif value.ndim == 0:
                pass  # scalar broadcasts to every vector
            elif value.ndim == 1:
                # one entry per normalized vector
                n_vectors = x.shape[1 - axis]
                if value.size != n_vectors:
                    raise ValueError(
                        f"Override must be scalar or have length {n_vectors} "
                        f"for axis={axis} (got length {value.size})")
            else:
                raise ValueError("Overrides must be scalar or 1-dimensional")
        # reduce over axis=1 gives one value per row -- make it a column vector
        # so it broadcasts back against the original array
        if axis == 1:
            value = np.reshape(value, (-1, 1))
        return value

    x_max = _resolve(override_normalize_maximum, np.max)
    x_min = _resolve(override_normalize_minimum, np.min)

    if map_minimum_to_zero:
        to_normalized = lambda raw: (np.asarray(raw)-x_min)/(x_max-x_min)
        to_raw = lambda norm: np.asarray(norm)*(x_max-x_min) + x_min
    else:
        to_normalized = lambda raw: np.asarray(raw)/x_max
        to_raw = lambda norm: np.asarray(norm)*x_max

    x = to_normalized(x)

    if return_map:
        return x, (to_normalized, to_raw)
    return x

def sort(x, y):
    """
    Sorts the first array and reorders the second array using the same sorting order.
    
    Args:
        x (array-like): The array to sort by
        y (array-like): The array to reorder using x's sort indices
        
    Returns:
        tuple: (sorted_x, sorted_y) - both arrays sorted/reordered by x's sort order
        
    Raises:
        ValueError: If x and y have different lengths
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError("Arrays x and y must have the same length")
    
    # Get the indices that would sort x
    sort_indices = np.argsort(x)
    
    # Apply the same sorting to both arrays
    sorted_x = x[sort_indices]
    sorted_y = y[sort_indices]
    
    return sorted_x, sorted_y

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