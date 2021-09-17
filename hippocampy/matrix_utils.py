import warnings

import bottleneck as bn
import numpy as np
from astropy.convolution import convolve
from skimage import measure, morphology
import pandas as pd
import tqdm as tqdm


#%% SMOOTH
def smooth1D(
    data,
    kernel_half_width=3,
    kernel_type="gauss",
    padtype="reflect",
    preserve_nan_opt=True,
):
    """
    One dimensional Smoothing of data. It can deal with vector or matrix of vector.
    In case of 2D inputs, it will smooth along dim=1

    padtypes are given to numpy.pad. Availlable option are:

    - ‘symmetric’ Pads with the reflection of the vector mirrored along the edge
      of the array.
    - 'reflect’ Pads with the reflection of the vector mirrored on the first and
      last values of the vector along each axis.

      NB: The difference between symmetric and reflect is that reflect mirror the
          the data without duplicating the end value:
          a = np.array([1 , 2 , 3 , 4])
          a_p = np.pad(a , (0,2) , 'reflect')
          [1 2 3 4 3 2]

          a_p = np.pad(a , (0,2) , 'symmetric')
          [1 2 3 4 4 3]

    - ‘wrap’ : Pads with the wrap of the vector along the axis. The first values are
      used to pad the end and the end values are used to pad the beginning.
    """
    # Check Input
    if kernel_half_width % 2 != 1:
        kernel_half_width += 1
        # kernel size has to be odd

    if len(data.shape) == 1:
        data = data[np.newaxis, :]

    acceptedPad = ["reflect", "symmetric", "wrap"]
    assert padtype in acceptedPad, "Not Implemented pad type"

    acceptedType = ["gauss", "box", "ramp"]
    assert kernel_type in acceptedType, "Not Implemented smoothing kernel type"

    # pad the data
    data_p = np.pad(data, ((0, 0), (kernel_half_width, kernel_half_width)), padtype)

    if kernel_type == "box":
        # here bn.movemean seem to be much faster (~10x) than using a convolution as
        # for the gaussian or ramp kernel. It affect the value of the moving mean to
        # the last index in the moving window, that why the output 'un pading' is
        # peculiar

        data_c = bn.move_mean(
            data_p, kernel_half_width * 2 + 1, min_count=kernel_half_width, axis=1
        )
        data_c = data_c[:, kernel_half_width * 2 :]

        if preserve_nan_opt:
            data_c[:, np.isnan(data)] = np.nan

        return data_c

    else:
        # Make convolution kernel
        kernel = np.zeros(kernel_half_width * 2 + 1)

        if kernel_type == "gauss":
            kernel = np.arange(0, kernel_half_width) - (kernel_half_width - 1.0) / 2.0
            kernel = np.exp(
                -(kernel ** 2) / (2 * kernel_half_width * kernel_half_width)
            )

        elif kernel_type == "ramp":
            kernel = np.linspace(1, kernel_half_width + 1, kernel_half_width + 1)
            kernel = np.hstack((kernel, kernel[-2::-1]))

        # Normalize kernel to one
        kernel = kernel / bn.nansum(kernel)
        # Convolve. Astropy  seems to deal really well with nan values
        data_c = np.apply_along_axis(
            convolve, axis=1, arr=data_p, kernel=kernel, preserve_nan=preserve_nan_opt
        )

        return data_c[:, kernel_half_width:-kernel_half_width]


def smooth2D(
    data,
    kernel_half_width=3,
    kernel_type="gauss",
    padtype="reflect",
    preserve_nan_opt=True,
):
    """
    function to smooth 2 dimensional data.
    Parameters
    ----------
    - data: matrix with your 2D data
    - kernel_half_width: half width of the smoothing kernel
    - kernel_type: way to smooth the data ('gauss': gaussian, 'box': boxcar smoothing)
    - padtype: the matrix will be padded in order to remove border artefact
        so we will pad the matrix.
        Available option:   - symmetric: reflect the vector on the edge 1 2 3 4 [3 2 1]
                            - reflect: reflect the vector on the edge 1 2 3 4 [4 3 2]
                            - wrap: circularly wrap opposing edges
     - preserve_nan_opt = do we smooth NaN or put them back att the end (default: True)

    Returns
    -------
    - data_c smoothed version of data

    """

    # Check Input
    if kernel_half_width % 2 != 1:
        # kernel size has to be odd
        kernel_half_width += 1

    acceptedPad = ["reflect", "symmetric", "wrap"]
    assert padtype in acceptedPad, "Not Implemented pad type"

    acceptedType = ["gauss", "box"]
    assert kernel_type in acceptedType, "Not Implemented smoothing kernel type"

    # pad the data
    data_p = np.pad(data, ((kernel_half_width, kernel_half_width)), padtype)

    # Initialize convolution kernel
    kernel = np.zeros((kernel_half_width * 2 + 1, kernel_half_width * 2 + 1))

    if kernel_type == "box":
        kernel = np.ones((kernel_half_width * 2 + 1, kernel_half_width * 2 + 1))
    elif kernel_type == "gauss":
        kernel_1D = np.arange(0, kernel_half_width) - (kernel_half_width - 1.0) / 2.0
        kernel_1D = np.exp(
            -(kernel_1D ** 2) / (2 * kernel_half_width * kernel_half_width)
        )
        kernel = np.outer(kernel_1D, kernel_1D)

    # Normalize kernel to one
    kernel = kernel / bn.nansum(kernel)

    # Convolve. Astropy seems to deal really well with nan values
    data_c = convolve(data_p, kernel=kernel, preserve_nan=preserve_nan_opt)

    return data_c[
        kernel_half_width:-kernel_half_width, kernel_half_width:-kernel_half_width
    ]


#%% STATISTICS
def corr_mat(a, axis=1):
    """
    Compute correlation between all the rows (or column) of a given matrix

    Parameters
    ----------
    - Matrix for example [unit, samples]
    - axis on which we want to work on

    Returns
    -------
    - correlation matrix

    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    a = np.asarray(a)

    a_z = zscore(a, axis)
    n = a.shape[axis]

    if axis == 0:
        return (1 / (n - 1)) * (a_z.T @ a_z)
    else:
        return (1 / (n - 1)) * (a_z @ a_z.T)


def zscore(matrix, axis=1):
    """
    Compute zscores along one axis.

    Parameters
    ----------
    matrix: np.array

    ax: int
        axis to work along

    Returns
    -------
    z: np.array()
        zscore matrix
    """
    if axis == 1:
        z = (matrix - bn.nanmean(matrix, axis=axis)[:, None]) / bn.nanstd(
            matrix, axis=axis, ddof=1
        )[:, None]
    else:
        z = (matrix - bn.nanmean(matrix, axis=axis)[None, :]) / bn.nanstd(
            matrix, axis=axis, ddof=1
        )[None, :]
    return z


#%% OTHER
def label_continuous_1d(v) -> np.array:
    """
    label_continuous_1d 
    label continuous values in either a bolean or int/float vector


    Parameters
    ----------
    v : [bool, int, float]
        input vector

    Returns
    -------
    np.array
        labeled vector

    Raises
    ------
    ValueError
        If the vector is not 1d
    
    Reference
    ---------
    https://github.com/ml31415/numpy-groupies/blob/master/numpy_groupies/utils_numpy.py


    Example
    -------
    for bool inputs
    >>> label_1d(np.array([1,1,1,0,0,0,1,0,0,1],dtype=bool))
    >>> array([1, 1, 1, 0, 0, 0, 2, 0, 0, 3])

    or for other types 
    >>> label_1d(np.array([1,1,1,2,2,3,4,5,5,6]))
    >>> array([1, 1, 1, 2, 2, 3, 4, 5, 5, 6])

    """

    if v.ndim != 1:
        raise ValueError("Input should be 1 dimensional")

    # initialize output vector
    st = np.empty_like(v, dtype=bool)
    st[0] = v[0]

    if v.dtype.kind == "b":
        st[1:] = ~v[:-1] & v[1:]
        C = np.cumsum(st)
        C[~v] = False
        return C
    else:
        mask = v.astype(bool)
        st[1:] = v[:-1] != v[1:]
        st[~mask] = False
        C = np.cumsum(st)
        C[~mask] = 0
        return C


def label(M):
    """
    label input vector or matrix

    Parameters
    ----------
    M: np.array,
        matrix or array that we want to label

    Return
    ------
    np.array of labeled data
    """

    if M.ndim == 1:
        return label_continuous_1d(M)
    else:
        # the following line ensure we feed a boolean data to the label
        # function. Its helps with
        M_new = np.array(M, dtype=bool)
        return measure.label(M_new)


def remove_small_objects(M, min_sz=3):
    """
    remove small non-zero "objects" from a vector

    Parameters
    ----------
    M: np.array
        boolean or zero and non-zero values vector
    min_sz: int
        minimum size of the object to keep

    Returns
    -------
    np.array only with connected components bigger than min_sz

    """
    M_l = label(M.astype(bool))
    M_l = morphology.remove_small_objects(M_l, min_size=min_sz)
    return np.array(M_l, dtype=bool)


def mean_at(idx, vals, fillvalue=np.nan, dtype=np.dtype(np.float64)) -> np.array:
    """
    mean_at [summary]
    It will mean value in vector vals at
    index in vector idx. This is usefull for resampling
    according to the 2p frame index for example

    Parameters
    ----------
    idx : [type]
        index vector, 
    vals : [type]
        [description]
    fillvalue : [type], optional
        [description], by default np.nan
    dtype : [type], optional
        [description], by default None

    Returns
    -------
    np.array
        [description]

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    
    Reference
    ---------
    https://gist.github.com/d1manson/5f78561c0f52d3073fe8
    """

    if idx.ndim == 0 or vals.ndim == 0:
        raise ValueError("Inputs should not be scalar")
    if idx.ndim > 1 or vals.ndim > 1:
        raise ValueError("Inputs should be have only one dimension")
    if len(idx) != len(vals):
        raise ValueError("Inputs should have the same length")

    # to rectify non-zero based indexes
    m = bn.nanmin(idx)
    if m != 0:
        idx = idx - m

    minlen = len(np.unique(idx))
    count_idx = np.bincount(idx, minlength=minlen)
    sum_vals = np.bincount(idx, weights=vals, minlength=minlen)

    with np.errstate(divide="ignore"):
        ret = sum_vals.astype(dtype) / count_idx

    if not np.isnan(fillvalue):
        ret[count_idx == 0] = fillvalue
    return ret


def moving_win(a, length, overlap=0, axis=None, end="cut", endvalue=0):
    """Generate a new array that chops the given array along the given axis into overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').

    See:
    IPython Interactive Computing and Visualization Cookbook, Second Edition (2018), by Cyrille Rossant:
    https://ipython-books.github.io/46-using-stride-tricks-with-numpy/

    This function is a slightly adapted version of:
    segment_axis.py from:
    https://scipy-cookbook.readthedocs.io/items/SegmentAxis.html
    https://scipy-cookbook.readthedocs.io/_static/items/attachments/SegmentAxis/segmentaxis.py

    Note:
    We could implement symmetric padding
    """

    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise ValueError("overlap must be nonnegative and length must be positive")

    if l < length or (l - length) % (length - overlap):
        if l > length:
            roundup = length + (1 + (l - length) // (length - overlap)) * (
                length - overlap
            )
            rounddown = length + ((l - length) // (length - overlap)) * (
                length - overlap
            )
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length - overlap) or (
            roundup == length and rounddown == 0
        )
        a = a.swapaxes(-1, axis)

        if end == "cut":
            a = a[..., :rounddown]
        elif end in ["pad", "wrap"]:  # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if end == "pad":
                b[..., l:] = endvalue
            elif end == "wrap":
                b[..., l:] = a[..., : roundup - l]
            a = b

        a = a.swapaxes(-1, axis)

    l = a.shape[axis]
    if l == 0:
        raise ValueError(
            "Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'"
        )
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1 :]
    newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1 :]

    try:
        return np.ndarray.__new__(
            np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype
        )
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = (
            a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1 :]
        )
        return np.ndarray.__new__(
            np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype
        )


def rolling_quantile(data, window_len, quantile):

    out = np.empty_like(data)
    for i, x in tqdm.tqdm(enumerate(data), total=data.shape[0]):
        out[i, :] = (
            pd.DataFrame(x)
            .rolling(window_len, center=True, min_periods=1)
            .quantile(quantile=quantile)
            .squeeze()
        )
    return out


def row_closest_idx_to_val(v, zero_val=None):
    """
    This function identify the closest index
    of non-zero element to a given value.

    Parameters
    ----------
    v
        vector, preferably logical
    zero_val
        value of the "zero"

    Returns
    -------

    index of the closest index
    """
    if zero_val is None:
        zero_val = v.size / 2

    tmp = np.squeeze(np.nonzero(v))

    if tmp.size > 0:
        return tmp[np.argmin(abs(tmp - zero_val))]
    else:
        return np.nan


def find_peak_row(M, zero_idx=None):
    """
    find_peak_row will find the peak per row that is closest to a
    particular value.

    Input:
            - M: Matrix of value, this function will by applied
            over rows
            - zero_val: value of the "zero"
      Returns:
            - index of the closest index

    Example:
    a = np.array([ [1, 2, 3, 2, 0] [ 4, 8, 9, 12, 1] ])
    p =

    """
    if zero_idx is None:
        zero_idx = M.shape[1] / 2

    bef = np.hstack((np.atleast_2d(M[:, 0]).T, M[:, :-1]))
    aft = np.hstack((M[:, 1:], np.atleast_2d(M[:, -1]).T))
    peaks = np.logical_and(M - bef >= 0, M - aft >= 0)

    return np.apply_along_axis(
        row_closest_idx_to_val, axis=1, arr=peaks, zero_val=zero_idx
    )


def find_peaks(M, min_amplitude=None):
    """
    find peaks over in each rows in a matrix
    Example:
    M = np.array([ [1, 2, 3, 2, 0], [ 4, 8, 9, 12, 1] ])
    [ P, P_idx ] = find_peaks(M)
    P = array([[False, False,  True, False, False],
         [False, False, False,  True, False]])
    P_idx = [array(2), array(3)]
    """

    bef = np.hstack((np.atleast_2d(M[:, 0]).T, M[:, :-1]))
    aft = np.hstack((M[:, 1:], np.atleast_2d(M[:, -1]).T))

    if min_amplitude is None:
        peaks = np.logical_and(M - bef >= 0, M - aft >= 0)
    else:
        peaks = np.logical_and.reduce([M - bef >= 0, M - aft >= 0, M >= min_amplitude])

    peaks_idx = [np.squeeze(np.nonzero(valP)) for itP, valP in enumerate(peaks)]

    return peaks, peaks_idx
