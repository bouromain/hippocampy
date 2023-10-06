import bottleneck as bn
import numpy as np
from numba import jit

from hippocampy.matrix_utils import zscore, smooth_1d, circ_shift_idx, smooth_2d
from hippocampy.utils.type_utils import float_to_int


def rate_map(
    var: np.ndarray,
    samples: np.ndarray,
    *,
    bins=10,
    fs: int = 1,
    min_occ: float = 0,
    smooth_half_win: int = 0,
    smooth_axis: int = 0,
    smooth_pad_type: str = "reflect",
    method: str = "point_process",
    preserve_nan_opt: bool = True,
):
    """
    rate_map [summary]

    Parameters
    ----------

    var : np.ndarray
        for multidimensional data this function expect the input to be var[n_samples, n_dim]
    samples : np.ndarray
        can be either a continuous variable of size nsamples or indexes
    bins : int, optional
        [description], by default 10
    fs : int
        sampling rate
    smooth_half_win : np.ndarray
        half window size (in samples) of the smoothing kernel
    smooth_axis : int
        axis along which to apply the smoothing
    smooth_pad_type : str
        type of padding before the smoothing. ex: reflect, circular,...
    method : str, optional
        "point_process" or "continuous", by default "point_process"
    preserve_nan_opt : bool
        specify if nan value should be preserved in the output

    Returns
    -------
    rate : np.ndarray
        [description]
    act : np.ndarray
        [description]
    occ : np.ndarray
        [description]
    Raises
    ------
    ValueError
        in case of error in provided method
    """
    # check inputs
    var = np.asarray(var)
    samples = np.asarray(samples)

    if method not in ["point_process", "continuous"]:
        raise ValueError("Method should be either continuous or point_process")
    if method == "point_process" and samples.dtype.kind != "i":
        samples = float_to_int(samples)

    if type(smooth_axis) == int:
        smooth_axis = [smooth_axis]  # to make it iterable

    if type(smooth_axis) not in [int, np.ndarray, list]:
        raise ValueError("Smooth axis should be int, np.array or list")

    # first take care of the bins
    if isinstance(bins, (list, np.ndarray)):
        bins = [bins]
    elif isinstance(bins, tuple):
        bins = [*bins]

    # calculate occupancy
    occ, _ = np.histogramdd(var, bins)

    # affect no occupancy to nan. This avoid zero division later
    # and is necessary to take into account non explored areas
    no_occ = occ <= min_occ
    occ[no_occ] = np.nan

    if method == "continuous":
        act, _ = np.histogramdd(var, bins, weights=samples)
        act /= occ
    elif method == "point_process":
        occ = occ / fs  # convert in Hz
        act, _ = np.histogramdd(var[samples], bins)

    act_s, occ_s = act, occ
    if smooth_half_win > 0:
        for curr_axis in np.array(smooth_axis):
            act_s = smooth_1d(
                act_s,
                smooth_half_win,
                axis=curr_axis,
                padtype=smooth_pad_type,
                preserve_nan_opt=preserve_nan_opt,
            )
            occ_s = smooth_1d(
                occ_s,
                smooth_half_win,
                axis=curr_axis,
                padtype=smooth_pad_type,
                preserve_nan_opt=preserve_nan_opt,
            )

    if method == "point_process":
        rate_s = act_s / occ_s
    elif method == "continuous":
        rate_s = act_s

    return rate_s, act_s, occ_s


def ccg(
    spikes1: np.ndarray,
    spikes2: np.ndarray,
    binsize: float = 1e-3,
    max_lag: float = 1000e-3,
    normalization: str = "count",
    safe: bool = False,
):
    """
    Fast crosscorrelation code:
    Assume that the spike trains are sorted

    Parameters
    ----------
    spikes1 : np.ndarray
        [description]
    spikes2 : np.ndarray
        [description]
    binsize : float, optional
        [description], by default 1e-3
    max_lag : float, optional
        [description], by default 1000e-3
    normalization : str, optional
        - count [default]: no normalisation
        - conditional: probability of spk2 knowing spk1
        - probability: sum to one
    safe : bool, optional
        assume sorted if unsafe, otherwise it will sort the
        inputs all the time, by default False

    Returns
    -------
    C : np.ndarray
        cross-correlogram
    E : np.ndarray
    edges of the corss-correlogram
    """

    assert normalization in [
        "count",
        "conditional",
        "probability",
    ], "Method not recognized"

    if safe:
        spikes1 = np.asarray(spikes1).sort()
        spikes2 = np.asarray(spikes2).sort()

    C, E = ccg_heart(spikes1, spikes2, binsize=binsize, max_lag=max_lag)

    if normalization == "conditional":
        sz1 = len(spikes1)
        C = C / sz1
    elif normalization == "probability":
        C = C / bn.nansum(C)

    return C, E


@jit(nopython=True)
def ccg_heart(spikes1: np.ndarray, spikes2: np.ndarray, binsize=1, max_lag=100):
    """
    Fast cross-correlation code:
    Assume that the spike trains are sorted

    Parameters
    ----------
    spikes1:
        first time series of spikes/indexes
    spikes2:
        second time series of spikes/indexes
    binsize:
        size of one bin
    max_lag:
        size of the half window

    Returns
    -------
    C:
        Cross-correlogram
    E:
        Edges of the Cross-correlogram

    TO DO
    -----
    this code could be slightly faster by storing the
    high bound in the last loop

    Adapted from:
    G. Viejo crossCorr function
    M. Zugaro CCGEngine.c
    """
    # create edges and ensure that they are odd
    winsize_bins = int((2 * max_lag) // binsize)

    if winsize_bins % 2 != 1:
        winsize_bins += 1
        max_lag += binsize / 2

    halfbin = int(winsize_bins / 2) + 1

    # Make edges (seem faster than np.linspace)
    E = np.zeros(winsize_bins)
    for i in range(winsize_bins):
        E[i] = -max_lag + i * binsize

    # initialise CCG
    C = np.zeros(winsize_bins)

    # loop over spikes
    idx2 = 0
    sz1 = len(spikes1)
    sz2 = len(spikes2)

    # if the first spike train is smaller we iterate over it
    for idx1 in range(sz1):
        # define the window around the spike of interest
        l_bound = spikes1[idx1] - max_lag
        H_bound = spikes1[idx1] + max_lag

        # search for the max index in spike 2 in window:
        while idx2 < sz2 and spikes2[idx2] < l_bound:
            idx2 += 1
        while idx2 > 0 and spikes2[idx2 - 1] > l_bound:
            idx2 -= 1
        # now we have this index we can accumulate value
        # in the ccg as long as we are in the window
        idx2_H = idx2
        while idx2_H < sz2 and spikes2[idx2_H] < H_bound:
            idx_C = halfbin + int((spikes2[idx2_H] - spikes1[idx1]) / binsize)
            idx_C = idx_C - 1  # to make it zero indexed
            C[idx_C] += 1
            idx2_H += 1

    return C, E


def psth(
    mat: np.ndarray,
    events_idx: np.ndarray,
    *,
    n_bins_bef: int = 20,
    n_bins_aft: int = 20,
    method: str = "mean",
    kernel_half_width: int = 0,
    norm_rows_method: str = None,
    norm_row_len: int = 10,
    return_temp: bool = False,
    axis=1,
):
    """
    psth _summary_

    Parameters
    ----------
    mat : np.ndarray
        input array (1 or 2d)
    events_idx : np.ndarray
        list of indexes where to perform the psth in a given axis
    n_bins_bef : int, optional
        number of bins to take before, by default 20
    n_bins_aft : int, optional
        number of bins to take after, by default 20
    method : str, optional
        method to perform in the psth ["mean", "median", "sum"], by default "mean"
    kernel_half_width : int, optional
        half width of the smoothing, by default 0
    norm_rows_method : str, optional
        normalization to perform before averaging the psth
    norm_row_len : int, optional
        number of sample to consider to calculate the normalization
    return_temp : bool, optional
        specify if we return the intermediate matrix [n_bins_bef+n_bins_aft, n_events]
    axis : int, optional
        axis along which the function is performed, by default 1

    Returns
    -------
    out: np.ndarray
        output psth
    temp_mat: optional, np.ndarray
    """
    # check inputs
    if method not in ["mean", "median", "sum"]:
        raise NotImplementedError(f"Method {method} not implemented")

    if mat.ndim > 2:
        raise ValueError("Input should have a maximum of two dimensions")

    if norm_rows_method not in ["mean", "median", "none", None]:
        raise NotImplementedError(f"Normalization method {method} not implemented")
    if norm_row_len > n_bins_bef + n_bins_aft:
        raise ValueError(
            "norm_row_len should be smaller than the psth window (n_bins_bef + n_bins_aft)"
        )

    # to be sure we have floats here, this can create problems with nan later
    mat = np.array(mat, dtype=float, ndmin=2)
    events_idx = float_to_int(events_idx)

    # initialise values
    sz = mat.shape
    idx_to_take = [np.arange(e - n_bins_bef, e + n_bins_aft, 1) for e in events_idx]
    idx_to_take = np.concatenate(idx_to_take)

    # fix problems in case we have negative or out of shape indices
    neg_idx = idx_to_take < 0
    out_idx = idx_to_take > sz[axis] - 1
    pad_before = bn.nansum(neg_idx)
    pad_after = bn.nansum(out_idx)
    idx_to_take = idx_to_take[~np.logical_or(neg_idx, out_idx)]

    temp_mat = np.take(mat, idx_to_take[:, None], axis=axis).squeeze()

    if temp_mat.ndim == 1:
        temp_mat = np.pad(
            temp_mat,
            pad_width=[pad_before, pad_after],
            mode="constant",
            constant_values=np.nan,
        )
    else:
        npad = [[0, 0], [0, 0]]
        npad[axis] = [pad_before, pad_after]
        temp_mat = np.pad(
            temp_mat, pad_width=npad, mode="constant", constant_values=np.nan
        )

    # make a 3d matrix (n_bins,-1,n_events)
    # this axis change  is not super elegant
    if axis == 1 or axis == -1:
        new_shape = [
            -1,
            len(events_idx),
            n_bins_bef + n_bins_aft,
        ]
    elif axis == 0:
        new_shape = [
            len(events_idx),
            n_bins_bef + n_bins_aft,
            -1,
        ]

    temp_mat = np.reshape(temp_mat, new_shape)

    if norm_rows_method == "mean":
        norm_rows = np.take(temp_mat, np.arange(0, norm_row_len), axis=axis).squeeze()
        temp_mat = temp_mat - bn.nanmean(norm_rows, axis=0)
    elif norm_rows_method == "median":
        norm_rows = np.take(temp_mat, np.arange(0, norm_row_len), axis=axis).squeeze()
        temp_mat = temp_mat - bn.nanmedian(norm_rows, axis=0)

    # now we perform the average/median along the correct dimension
    if method == "mean":
        out = bn.nanmean(temp_mat, axis=axis)
    elif method == "median":
        out = bn.nanmedian(temp_mat, axis=axis)
    elif method == "sum":
        out = bn.nansum(temp_mat, axis=axis)

    if kernel_half_width > 0:
        out = smooth_1d(out, kernel_half_width=kernel_half_width)

    if return_temp:
        return out.squeeze(), temp_mat.squeeze()
    else:
        return out.squeeze()


def mua(
    mat: np.ndarray,
    *,
    axis: int = -1,
    smooth_first: bool = True,
    kernel_half_width: int = 10,
):
    """
    Compute "multi-unit" activity from a Transient matrix.
    The input matrix can be binary matrix of transient or spikes (True), or
    rate vectors
    Individual traces are first smoothed and then summed
    to finally have the multi-unit activity.

    Parameters
    ----------
    mat : np.nd_array
        input matrix, can be binary matrix of transient or spikes (True), or
        rate vectors
    axis : int, optional
        axis along which the function is performed, by default -1
    smooth_first : bool, optional
        if we fist sooth individual neuron activities or only at the end,
        by default True
    kernel_half_width : int, optional
        size of the smoothing half window (in samples), by default 10

    Returns
    -------
    mua_act
        _description_
    """

    mua_act = np.array(mat)

    if smooth_first:
        mua_act = smooth_1d(mua_act, kernel_half_width=kernel_half_width, axis=axis)

    mua_act = zscore(mua_act, axis=axis)
    mua_act = bn.nansum(mua_act, axis=axis)

    if not smooth_first:
        mua_act = smooth_1d(mua_act, kernel_half_width=kernel_half_width, axis=axis)

    return mua_act


# def continuous_ccg(spikes1, spikes2, tau=10e-3, max_lag=100e-3):
#     """

#     for now this function iis s simple translation of the following code:
#     http://www.memming.com/codes/ccc.m.html
#     http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.8458&rep=rep1&type=pdf

#     """
#     raise NotImplementedError("This should be tested")

#     spikes1 = np.random.uniform(0, 1, 100)
#     spikes2 = np.random.uniform(0, 1, 100)

#     spikes1 = np.asarray(spikes1)
#     spikes2 = np.asarray(spikes2)

#     assert spikes1.size is not 0, "at least one spike is required for first spike train"
#     assert (
#         spikes2.size is not 0
#     ), "at least one spike is required for second spike train"

#     # calculate all the lags
#     delta_t = spikes1[None, :] - spikes2[:, None]

#     # Only keep the ones that are smaller than max_lag
#     delta_t = np.ravel(delta_t)
#     m = np.abs(delta_t) <= max_lag
#     delta_t = delta_t[m]

#     # sort the lags
#     delta_t = np.sort(delta_t)

#     # Now do the little stuff I am not sure to understand
#     Q_plus = np.zeros_like(delta_t)
#     Q_minus = np.zeros_like(delta_t)
#     Q_minus[0] = 1

#     exp_delta = np.exp(-np.diff(delta_t) / tau)
#     N = delta_t.size

#     for k in range(0, N - 1):
#         Q_minus[k] = 1 + Q_minus[k - 1] * exp_delta[k - 1]
#         Q_plus[-k + 2] = (Q_plus[-k + 1] + 1) * exp_delta[-k + 2]

#     return Q_minus + Q_plus, delta_t
