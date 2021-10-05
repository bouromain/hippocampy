import bottleneck as bn
import numpy as np
from typing import Union


def value_cross(M, threshold=0, axis=-1):
    """
    Function finding the crossing point between a vector and a value.
    Particularly useful when you want to detect crossing of a threshold
    were you are not likely to find exactly a value in your input vector.

    Parameters
    ----------
    M:
        data array
        
    threshold:
        threshold to cross

    Returns
    -------
    up:
        logical vector with True value for up crossing
    down:
        logical vector with True value for down crossing
    """
    M_b = np.array(M, ndmin=2)
    return start_stop(M_b > threshold, axis=axis)


def start_stop(B: np.ndarray, axis=-1):
    """
    start_stop find start and stop in a boolean array

    Parameters
    ----------
    B : np.ndarray
        input boolean array 
    axis : int, optional
        axis to work on, by default -1

    Returns
    -------
    start: np.ndarray
        boolean array with True values for starts 

    stops: np.ndarray
        boolean array with True values for stops     
    """
    B = np.array(B, dtype=bool, ndmin=2)
    start = np.empty_like(B, dtype=bool)
    stop = np.empty_like(B, dtype=bool)

    if axis == 1 or axis == -1:
        # take into account that fist value can be a start or end value a stop
        start[:, 0] = B[:, 0]
        stop[:, -1] = B[:, -1]

        start[:, 1:] = ~B[:, :-1] & B[:, 1:]
        stop[:, :-1] = B[:, :-1] & ~B[:, 1:]

    elif axis == 0:
        # take into account that fist value can be a start or end value a stop
        start[0, :] = B[0, :]
        stop[-1, :] = B[-1, :]

        start[1:, :] = ~B[:-1, :] & B[1:, :]
        stop[:-1, :] = B[:-1, :] & ~B[1:, :]

    return np.squeeze(start), np.squeeze(stop)


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
        return np.array(np.nonzero(E == -2)[0] + 1)
    elif method == "min":
        return np.array(np.nonzero(E == 2)[0] + 1)
    elif method == "all":
        return np.array(np.nonzero(np.logical_or(E == 2, E == -2))[0] + 1)


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
        idx = np.array(np.searchsorted(values, array, side="left"), dtype=int) - 1

    elif method == "unsorted":
        idx = np.array(bn.nanargmin(np.abs(array - values[:, None]), axis=0), dtype=int)

    return idx


def calc_dt(t):
    """
    Calculate dt from a time vector
    """
    return bn.nanmedian(np.diff(t))


def nearest_odd(x: Union[float, np.ndarray]):
    """
    nearest_odd round number or array to nearest odd number

    Parameters
    ----------
    x : Union[float, np.ndarray]
        values or array to round

    Returns
    -------
    out
        values or array rounded to the nearest odd number
    """
    x = np.asarray(x)
    idx = (x % 2) < 1
    x = np.floor(x)
    if isinstance(x, np.ndarray):
        x[idx] = x[idx] + 1
    elif idx:
        x += 1
    return x


def nearest_even(x: Union[float, np.ndarray]):
    """
    nearest_even round number or array to nearest even number

    Parameters
    ----------
    x : Union[float, np.ndarray]
        values or array to round

    Returns
    -------
    out
        values or array rounded to the nearest even number
    """
    x = np.asarray(x)
    return np.round(x / 2) * 2
