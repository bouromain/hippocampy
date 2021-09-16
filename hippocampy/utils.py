import bottleneck as bn
import numpy as np


def value_cross(x, threshold=0):
    """
    Function finding the crossing point between a vector and a value.
    Particularly useful when you want to detect crossing of a threshold
    were you are not likely to find exactly a value in your input vector.

    Parameters
    ----------
    x:
        vector of data
    threshold:
        threshold to cross

    Returns
    -------
    up:
        logical vector with True value for up crossing
    down:
        logical vector with True value for down crossing
    """
    before = np.array(x[:-1])
    after = np.array(x[1:])

    up = np.logical_and(before < threshold, after > threshold)
    down = np.logical_and(before > threshold, after < threshold)

    up = np.append(up, False)
    down = np.append(down, False)

    return up, down


def localExtrema(x, method="max"):
    """
    Find local extrema and return their index

    Parameters
    ----------
    x:
        vector
    method:
        type of extrema to consider [max, min, all] (default: max)

    Returns
    -------
    index of the local extrema as defined by the parameter 'method'

    """
    assert method in ["max", "min", "all"], "Invalid Method in localExtrema"

    x = np.asarray(x, dtype=float)
    D = np.diff(x)
    E = np.diff(D / abs(D))

    if method == "max":
        return np.nonzero(E == -2)[0] + 1
    elif method == "min":
        return np.nonzero(E == 2)[0] + 1
    else:
        return np.nonzero(np.logical_or(E == 2, E == -2))[0] + 1


def _remove_nan(x, x_mask=None, axis=0):
    """
    Remove NaN in a 1D array.
    adapted from Pinguin _remove_na_single
    """
    if x_mask is None:
        x_mask = _nan_mask(x, axis=axis)

    # Check if missing values are present
    if ~x_mask.all():
        ax = 0 if axis == 0 else 1
        ax = 0 if x.ndim == 1 else ax
        x = x.compress(x_mask, axis=ax)

    return x


def _nan_mask(x, axis=0):
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
    else:
        # 2D arrays
        ax = 1 if axis == 0 else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)

    return x_mask


def remove_nan(x, y=None, paired=False, axis=0):
    """
    Helper function to remove nan from 1D or 2D

    Parameters
    ----------
    x:
        vector or matrix with nans
    y:
        vector or matrix with nans
    paired:
        should we pair the removal of nan values in x and y
    axis:
        axis to operate on

    Returns
    -------
            - x (and y) without nan values
    """
    x = np.asarray(x)
    if y is None:
        return _remove_nan(x, axis=axis)
    else:
        y = np.asarray(y)

        if not paired:
            x = _remove_nan(x, axis=axis)
            y = _remove_nan(y, axis=axis)

            return x, y

        else:
            x_mask = _nan_mask(x, axis=axis)
            y_mask = _nan_mask(y, axis=axis)

            xy_mask = np.logical_and(x_mask, y_mask)

            if ~np.all(xy_mask):
                x = _remove_nan(x, x_mask=xy_mask, axis=axis)
                y = _remove_nan(y, x_mask=xy_mask, axis=axis)

            return x, y


def nearest_idx(array, values, method="sorted"):
    """
    Find nearest index of value(s) in a given array
    By default this function will assume that array is sorted.
    Use the "unsorted method" otherwise
    Note that:
            - idx(values < min(array) ) = 0
            - idx(values > max(array) ) = max(array)

    Parameters
    ----------
    array:
        array where you wnt to index values (np.array)
    values:
        array or float of values of which you want the index

    Returns
    -------
    idx:
        closest indices of values in array

    """
    array = np.asarray(array)
    values = np.asarray(values)

    if method == "sorted":
        idx = np.array(np.searchsorted(array, values, side="left"), dtype=int)

    elif method == "unsorted":
        idx = np.array([(np.abs(array - val)).argmin() for val in values], dtype=int)

    return idx


def calc_dt(t):
    """
    Calculate dt from a time vector
    """
    return bn.nanmedian(np.diff(t))


def float_to_int(array) -> np.array:
    """
    float_to_int 
    Cast float to int but check value are then equal 

    Parameters
    ----------
    array : [type]
        input array

    Returns
    -------
    array:
        output array converted to int 

    Raises
    ------
    TypeError
        [description]

    Reference
    ---------
    https://stackoverflow.com/questions/36505879/convert-float64-to-int64-using-safe    
    """
    if array.dtype.kind == "i":
        return array

    int_array = array.astype(int, casting="unsafe", copy=True)
    if not np.equal(array, int_array).all():
        raise TypeError(
            "Cannot safely convert float array to int dtype. "
            "Array must only contain whole numbers."
        )
    return int_array
