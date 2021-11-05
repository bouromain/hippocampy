import bottleneck as bn
import numpy as np
from astropy.convolution import convolve
from skimage import measure
import pandas as pd
import tqdm as tqdm

#%% SMOOTH
def smooth_1d(
    data: np.ndarray,
    kernel_half_width: int = 2,
    axis: int = -1,
    *,
    kernel_type: str = "gauss",
    padtype: str = "reflect",
    preserve_nan_opt: bool = True,
):
    """
    One dimensional Smoothing of data. It can deal with vector or matrix of vector

    Parameters
    ----------
    data : np.ndarray
        input values
    kernel_half_width : int, optional
        half width of the smoothing kernel, by default 2
    axis : int, optional
        axis along which the function is performed, by default -1
    kernel_type : str, optional
        type of smoothing, by default "gauss"
        option: - gauss: gaussian
                - ramp: ramp kernel
                - box: median box kernel
    padtype : str, optional
        type of padding, by default "reflect"
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

    preserve_nan_opt : bool, optional
        Define if nan value from the input data ae perserved in the output, by default True

    Returns
    -------
    out np.ndarray
        smoothed input values

    """
    # check input
    if kernel_type not in ["gauss", "box", "ramp"]:
        raise ValueError(f"Kernel type value {kernel_type} not recognized")
    if padtype not in ["reflect", "symmetric", "wrap"]:
        raise ValueError(f"Pad type value {padtype} not recognized")

    if kernel_half_width % 2 != 1:
        kernel_half_width += 1

    # pad the data along the correct dimension
    n_pad = [(0, 0)] * data.ndim
    n_pad[axis] = (kernel_half_width, kernel_half_width)
    data_p = np.pad(data, n_pad, padtype)

    # smooth !
    if kernel_type == "box":
        # here bn.movemean seem to be much faster (~10x) than using a convolution as
        # for the gaussian or ramp kernel. It affect the value of the moving mean to
        # the last index in the moving window, that why the output 'un pading' is
        # peculiar

        data_c = bn.move_mean(
            data_p, kernel_half_width * 2 + 1, min_count=kernel_half_width, axis=axis
        )
        idx = np.arange(0, data_p.shape[axis] - (kernel_half_width * 2))
        data_c = np.take(data_c, idx, axis=axis)

        if preserve_nan_opt:
            data_c[np.isnan(data)] = np.nan
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
            convolve,
            axis=axis,
            arr=data_p,
            kernel=kernel,
            preserve_nan=preserve_nan_opt,
        )

        idx = np.arange(kernel_half_width, data_p.shape[axis] - kernel_half_width)
        data_c = np.take(data_c, idx, axis=axis)
    return data_c


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
def corr_mat(a: np.ndarray, axis=-1) -> np.ndarray:
    """
    Compute correlation between all the rows (or column) of a given matrix

    Parameters
    ----------
    a : np.ndarray
        input matrix for example [unit, samples]
    axis : int, optional
        axis along which the function is performed, by default -1

    Returns
    -------
    np.ndarray
        correlation matrix
    Reference
    ---------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    a = np.asarray(a)

    a_z = zscore(a, axis=axis)
    n = a.shape[axis]

    if axis == 0:
        return (1 / (n - 1)) * (a_z.T @ a_z)
    else:
        return (1 / (n - 1)) * (a_z @ a_z.T)


def zscore(matrix, axis=-1):
    """
    Compute zscores along one axis.

    Parameters
    ----------
    matrix: np.array

    ax: int
        axis along which the function is performed, by default -1

    Returns
    -------
    z: np.array()
        zscore matrix
    """
    mu = bn.nanmean(matrix, axis=axis)
    sigma = bn.nanstd(matrix, axis=axis, ddof=1)

    if isinstance(mu, np.ndarray):
        mu = np.expand_dims(mu, axis=axis)
        sigma = np.expand_dims(sigma, axis=axis)
    return (matrix - mu) / sigma


def norm_axis(matrix: np.ndarray, method="max", axis=-1) -> np.ndarray:
    """
    normalize a matrix collumn or rawwise with a given method

    Parameters
    ----------
    matrix : np.ndarray
        matrix or vector to normalize
    method : str, optional
        normalisation method, by default "max"
            max: normalize such as the new max per row/column is 1
            one: normalize such as the sum of the values per row/column 
                sums to one (eg: become a probability distribution)
            zscore: compute the row/column wise zscore 
    axis : int, optional
        axis along which the function is performed, by default -1

    Returns
    -------
    np.ndarray
        normalized matrix

    """
    if method.lower() not in ["max", "one", "zscore"]:
        raise ValueError(f"Method {method} not recognized")

    if method == "zscore":
        return zscore(matrix, axis=axis)
    elif method == "one":
        divisor = bn.nansum(matrix, axis=axis)
    elif method == "max":
        divisor = bn.nanmax(matrix, axis=axis)

    return matrix / np.expand_dims(divisor, axis=axis)


def circ_shift(M: np.ndarray, max_shift: int = None, axis: int = -1):
    """
    circ_shift circularly shift a matrix along a given dimension 


    Parameters
    ----------
    M : np.ndarray
        input matrix
    max_shift : int, optional
        max shift, by default size matrix along the given dimension
    axis : int, optional
        axis along which the function is performed, by default -1

    Returns
    -------
    out: np.ndarray
        shifted input matrix
    """
    sz = M.shape
    if max_shift is None:
        max_shift = sz[axis]
    else:
        assert max_shift < sz[axis]

    new_x, new_y = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]))

    if axis == 0:
        shifts = np.random.randint(max_shift, size=(sz[1], 1))
        new_x = np.mod(new_x + shifts, sz[axis])
        return M[new_x, new_y].T

    elif axis == 1 or axis == -1:
        shifts = np.random.randint(max_shift, size=(1, sz[0]))
        new_y = np.mod(new_y + shifts, sz[axis])
        return M[new_x, new_y].T


#%% OTHER
def label(v, axis=-1, *, unique_label=False) -> np.array:
    """
    label continuous values in either a boolean or int/float vector


    Parameters
    ----------
    v : [bool, int, float]
        input vector
    axis: [int]
        axis along which the function is performed, by default -1
    unique_label: [bool]
        if we need unique labels or if labels can be repeated for each row/column

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
    >>> label(np.array([1,1,1,0,0,0,1,0,0,1],dtype=bool))
    >>> array([1, 1, 1, 0, 0, 0, 2, 0, 0, 3])

    or for other types 
    >>> label(np.array([1,1,1,2,2,3,4,5,5,6]))
    >>> array([1, 1, 1, 2, 2, 3, 4, 5, 5, 6])

    """
    v = np.array(v, ndmin=2)

    if not 1 <= v.ndim <= 2:
        raise ValueError("Input should not be scalar")

    # initialize output vector
    st = np.empty_like(v, dtype=bool)

    if axis == 1 or axis == -1:
        st[:, 0] = v[:, 0]

        if v.dtype.kind == "b":
            st[:, 1:] = ~v[:, :-1] & v[:, 1:]
            C = np.cumsum(st, axis=axis, dtype=np.int64)
            C[~v] = False
        else:
            mask = v.astype(bool)
            st[:, 1:] = v[:, :-1] != v[:, 1:]
            st[~mask] = False
            C = np.cumsum(st, axis=axis, dtype=np.int64)
            C[~mask] = 0
    elif axis == 0:
        st[0, :] = v[0, :]

        if v.dtype.kind == "b":
            st[1:, :] = ~v[:-1, :] & v[1:, :]
            C = np.cumsum(st, axis=axis, dtype=np.int64)
            C[~v] = False
        else:
            mask = v.astype(bool)
            st[1:, :] = v[:-1, :] != v[1:, :]
            st[~mask] = False
            C = np.cumsum(st, axis=axis, dtype=np.int64)
            C[~mask] = 0
    C = np.squeeze(C)

    if unique_label:
        # give a unique label for the total stack and not per row
        C_max = bn.nanmax(C, axis=axis)
        if isinstance(C_max, np.ndarray):
            mask = C.astype(bool)
            C_max = np.roll(C_max, 1)
            C_max[0] = 0
            C = C + np.expand_dims(C_max, axis=axis)
            C[~mask] = 0
    return C


def label2D(M):
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
    # the following line ensure we feed a boolean data to the label
    # function.
    M_new = np.array(M, dtype=bool)
    return measure.label(M_new)


def first_true(v: np.ndarray, axis=-1):
    """
    first_true Return first true value of contiguous sequence of True
    per row or colum

    Example:
    first_true([0,0,1,1,0,1,1,1])
    array([[False,  True, False, False,  True, False, False]])

    Parameters
    ----------
    v : np.ndarray
        input array/vector
    axis : int, optional
        axis along which the function is performed, by default -1

    Returns
    -------
    out
        boolean array only containing first true value occuring 
        along one axis
    """
    v = np.array(v, ndmin=2)
    v = v.astype(bool)

    st = np.empty_like(v, dtype=bool)
    if axis == 1 or axis == -1:
        st[:, 0] = v[:, 0]
        st[:, 1:] = ~v[:, :-1] & v[:, 1:]
    elif axis == 0:
        st[0, :] = v[0, :]
        st[1:, :] = ~v[:-1, :] & v[1:, :]
    return np.squeeze(st)


def last_true(v: np.ndarray, axis=-1):
    """
    Return last true value of contiguous sequence of True
    per row or colum

    Example:
    last_true([0,0,1,1,0,1,1,1])
    array([[False,  False, False, True,  False, False, True]])

    Parameters
    ----------
    v : np.ndarray
        input array/vector
    axis : int, optional
        axis along which the function is performed, by default -1

    Returns
    -------
    out
        boolean array only containing last true value occuring 
        along one axis
    """
    v = np.array(v, ndmin=2)
    v = v.astype(bool)

    st = np.empty_like(v, dtype=bool)
    if axis == 1 or axis == -1:
        st[:, -1] = v[:, -1]
        st[:, :-1] = v[:, :-1] & ~v[:, 1:]
    elif axis == 0:
        st[-1, :] = v[-1, :]
        st[:-1, :] = v[:-1, :] & ~v[1:, :]
    return np.squeeze(st)


def remove_small_objects(M, min_size=3, axis=-1):
    """
    remove small non-zero "objects" from a vector
    not that this function will not preserve nans

    Parameters
    ----------
    M: np.array
        boolean or zero and non-zero values vector
    min_size: int
        minimum size of the object to keep
    axis: int
        axis along which the function is performed, by default -1

    Returns
    -------
    np.array only with connected components bigger than min_size

    TODO: 
    -make this function work for 2D. it should be doable by using 
    ravel and reshaping at the end
    """
    M[np.isnan(M)] = 0
    # label our matrix per row/column
    M_l = label(M.astype(bool), axis=axis, unique_label=True)

    # find all the connected components and check their size
    comp_size = np.bincount(M_l.ravel())
    comp_id = np.unique(M_l.ravel())

    for id, sz in zip(comp_id, comp_size):
        if sz < min_size:
            M_l[M_l == id] = False

    return M_l.astype(bool)


def remove_holes(M, min_size=3, axis=-1):
    """
    remove holes smaller than minsize

    Parameters
    ----------
    M: np.array
        boolean or zero and non-zero values vector
    min_sz: int
        minimum size of the object to keep
    axis: int
        axis along which the function is performed, by default -1

    Returns
    -------
    np.array only with connected components bigger than min_sz
    """
    M = np.array(M, dtype=bool)
    M_b = np.logical_not(M)
    M_b = remove_small_objects(M_b, min_size=min_size, axis=axis)

    return np.logical_not(M_b)


def mean_at(idx, vals, fillvalue=np.nan, dtype=np.dtype(np.float64)) -> np.array:
    """
    mean_at [summary]
    It will mean value in vector vals at
    index in vector idx. This is usefull for resampling
    according to the 2p frame index for example

    Parameters
    ----------
    idx : [type]
        input, 
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

    minlen = len(np.unique(idx))
    count_idx = np.bincount(idx, minlength=minlen)
    sum_vals = np.bincount(idx, weights=vals, minlength=minlen)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = sum_vals.astype(dtype) / count_idx

    if not np.isnan(fillvalue):
        ret[count_idx == 0] = fillvalue
    return ret


def moving_win(
    a: np.ndarray,
    win_length: int,
    overlap: int = 0,
    axis: int = None,
    padding: str = "cut",
    endvalue: float = 0,
) -> np.ndarray:
    """
    moving_win Generate a view of the input array (or a copy if needed) with the 
    specified window length and overlap.
    It is particularly useful to perform some rolling window like operation.

    However it worth checking other types of function that are more optimized such
    as moving window function of bottleneck (mean, std, median,...) or pandas/dask 
    dataframe.rolling functions

    Parameters
    ----------
    a : np.ndarray
        input vector or array
    win_length : int
        length of the moving window (in samples)
    overlap : int, optional
        number of overlaping samples between consecutive windows, by default 0
    axis : int, optional
        axis along which the function is performed, by default None
    padding : str, optional
        method to deal with borders, by default "cut"
    endvalue : float, optional
        value to use to fill out of array values, by default 0

    Returns
    -------
    np.ndarray
        view of the input array (or a copy if needed) with the 
        specified window length and overlap

    Example
    -------
    >>> moving_win(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    Reference
    ---------
    IPython Interactive Computing and Visualization Cookbook, Second Edition (2018), by Cyrille Rossant:
    https://ipython-books.github.io/46-using-stride-tricks-with-numpy/

    This function is a slightly adapted version of:
    segment_axis.py from:
    https://scipy-cookbook.readthedocs.io/items/SegmentAxis.html
    https://scipy-cookbook.readthedocs.io/_static/items/attachments/SegmentAxis/segmentaxis.py

    TODO
    ----
    make a symmetric padding
    check if we could avoid the try, except by checking if the stride is correct 
    before initializing the array
    """

    if overlap >= win_length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or win_length <= 0:
        raise ValueError("overlap must be nonnegative and length must be positive")
    pad_methods = ["cut", "wrap", "pad"]
    if padding not in pad_methods:
        raise ValueError(f"Mehtod should be: {pad_methods}")

    # if no axis are specified operate on a flattened version of the array
    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

    l = a.shape[axis]

    if l < win_length or (l - win_length) % (win_length - overlap):
        if l > win_length:
            roundup = win_length + (1 + (l - win_length) // (win_length - overlap)) * (
                win_length - overlap
            )
            rounddown = win_length + ((l - win_length) // (win_length - overlap)) * (
                win_length - overlap
            )
        else:
            roundup = win_length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (win_length - overlap) or (
            roundup == win_length and rounddown == 0
        )
        a = a.swapaxes(-1, axis)

        if padding == "cut":
            a = a[..., :rounddown]
        elif padding in ["pad", "wrap"]:  # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if padding == "pad":
                b[..., l:] = endvalue
            elif padding == "wrap":
                b[..., l:] = a[..., : roundup - l]
            a = b

        a = a.swapaxes(-1, axis)

    # we then update the length after cutting/paddding...
    l = a.shape[axis]
    if l == 0:
        raise ValueError(
            "Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'"
        )
    assert l >= win_length
    assert (l - win_length) % (win_length - overlap) == 0

    n = 1 + (l - win_length) // (win_length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, win_length) + a.shape[axis + 1 :]
    newstrides = (
        a.strides[:axis] + ((win_length - overlap) * s, s) + a.strides[axis + 1 :]
    )

    try:
        return np.ndarray.__new__(
            np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype
        )
    except (TypeError, ValueError):
        # warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = (
            a.strides[:axis] + ((win_length - overlap) * s, s) + a.strides[axis + 1 :]
        )
        return np.ndarray.__new__(
            np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype
        )


def rolling_quantile(data, window_len, quantile):
    """
    rolling_quantile Calculate a rolling quantile a a window of size window len

    Note: For now, the pandas method seem convenient and fast. 
    It seem to be faster than numpy strides
    To Do:
    this function should take an axis input

    Parameters
    ----------
    data : [type]
        [description]
    window_len : [type]
        [description]
    quantile : [type]
        [description]
    Returns
    -------
    [type]
        [description]
    """

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
