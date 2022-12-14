import numpy as np


def remove_nan(x, y=None, paired=False, axis=-1):
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


def interp_nan(var: np.ndarray):
    """
    interp_nan interpolater nanvalues
    this function could be made more funky with scipy interp notably for the
    kind of interpolation used

    Parameters
    ----------
    var : _type_
        interpolated vector
    """
    m_valid = ~np.isnan(var)
    return np.interp(np.arange(len(var)), np.nonzero(m_valid)[0], var[m_valid])

