import bottleneck as bn
import numpy as np
from numba import jit

from hippocampy.matrix_utils import circ_shift, smooth_1d, circ_shift_idx
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
    preserve_nan_opt: bool = True
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

    if not method in ["point_process", "continuous"]:
        raise ValueError("Method should be either continuous or point_process")
    if method == "point_process" and samples.dtype.kind != "i":
        samples = float_to_int(samples)

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
    occ = occ / fs  # convert in Hz

    if method == "continuous":
        act, _ = np.histogramdd(var, bins, weights=samples)
        act /= occ
    elif method == "point_process":
        act, _ = np.histogramdd(var[samples], bins)

    if smooth_half_win > 0:
        act_s = smooth_1d(
            act,
            smooth_half_win,
            axis=smooth_axis,
            padtype=smooth_pad_type,
            preserve_nan_opt=preserve_nan_opt,
        )
        occ_s = smooth_1d(
            occ,
            smooth_half_win,
            axis=smooth_axis,
            padtype=smooth_pad_type,
            preserve_nan_opt=preserve_nan_opt,
        )
    else:
        act_s, occ_s = act, occ

    if method == "point_process":
        rate_s = act_s / occ_s
    elif method == "continuous":
        rate_s = act_s

    return rate_s, act_s, occ_s


def boostrap_1d(
    var: np.ndarray,
    samples: np.ndarray,
    boot_var: np.ndarray,
    bins: np.ndarray,
    *,
    n_rep: int = 1000,
    fs: int = 1,
    min_occ: float = 0,
    smooth_half_win: int = 0,
    smooth_axis: int = 0,
    smooth_pad_type: str = "reflect",
    method: str = "point_process",
    preserve_nan_opt: bool = True
):
    # check inputs
    var = np.asarray(var)
    samples = np.asarray(samples)
    boot_var = np.asarray(boot_var)

    if not method in ["point_process", "continuous"]:
        raise ValueError("Method should be either continuous or point_process")
    if method == "point_process" and samples.dtype.kind != "i":
        samples = float_to_int(samples)

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
    occ = occ / fs  # convert in Hz

    # calculate normal rate maps
    if method == "continuous":
        act, _ = np.histogramdd(var, bins, weights=samples)
    elif method == "point_process":
        act, _ = np.histogramdd(var[samples], bins)

    if smooth_half_win > 0:
        act_s = smooth_1d(
            act,
            smooth_half_win,
            axis=smooth_axis,
            padtype=smooth_pad_type,
            preserve_nan_opt=preserve_nan_opt,
        )
        occ_s = smooth_1d(
            occ,
            smooth_half_win,
            axis=smooth_axis,
            padtype=smooth_pad_type,
            preserve_nan_opt=preserve_nan_opt,
        )
    else:
        act_s, occ_s = act, occ

    if method == "point_process":
        rate_s = act_s / occ_s
    elif method == "continuous":
        act_s /= occ
        rate_s = act_s

    # calculate the bootstrap
    boot_mat = np.zeros((act.shape[0], act.shape[1], n_rep))
    for it_rep in np.arange(n_rep):
        var_shifted = circ_shift_idx(var, boot_var)
        if method == "continuous":
            boot_mat[:, :, it_rep], _ = np.histogramdd(
                var_shifted, bins, weights=samples
            )
        elif method == "point_process":
            boot_mat[:, :, it_rep], _ = np.histogramdd(var_shifted[samples], bins)

    return rate_s, act_s, occ_s, boot_mat


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
def ccg_heart(spikes1, spikes2, binsize=1e-3, max_lag=1000e-3):
    """
    Fast crosscorrelation code:
    Assume that the spike trains are sorted

    Parameters
    ----------
    spikes1:
        first time serie of spikes
    spikes2:
        second time serie of spikes
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
    winsize_bins = 2 * int(max_lag / binsize)

    if winsize_bins % 2 != 1:
        winsize_bins += 1
        max_lag += binsize / 2

    halfbin = int(winsize_bins / 2) + 1

    # Make edges (seem faster than np.linspace)
    E = np.zeros(winsize_bins + 1)
    for i in range(winsize_bins + 1):
        E[i] = -max_lag + i * binsize

    # initialise CCG
    C = np.zeros(winsize_bins)

    # loop over spikes
    idx2 = 0
    sz1 = len(spikes1)
    sz2 = len(spikes2)

    if sz1 <= sz2:
        # if the first spike train is smaller we iterate over it
        for idx1 in range(sz1):
            # define the window around the spike of interest
            l_bound = spikes1[idx1] - max_lag
            H_bound = spikes1[idx1] + max_lag

            # search for the max index in spike 2 in window:
            while idx2 < sz2 and spikes2[idx2] < l_bound:
                idx2 += 1
            while idx2 > 1 and spikes2[idx2 - 1] > l_bound:
                idx2 -= 1
            # now we have this index we can accumulate value
            # in the ccg as long as we are in the window
            idx2_H = idx2
            while idx2_H < sz2 and spikes2[idx2_H] < H_bound:
                idx_C = halfbin + int(0.5 + (spikes2[idx2_H] - spikes1[idx1]) / binsize)
                idx_C = idx_C - 1  # to make it zero indexed
                C[idx_C] += 1
                idx2_H += 1
    else:
        # if the second spike train is smaller we iterate over it but CCG
        # is flipped at the end to be consistent with the input
        for idx1 in range(sz2):
            # define the window around the spike of interest
            l_bound = spikes2[idx1] - max_lag
            H_bound = spikes2[idx1] + max_lag

            # search for the max index in spike 2 in window:
            while idx2 < sz1 and spikes1[idx2] < l_bound:
                idx2 += 1
            while idx2 > 1 and spikes1[idx2 - 1] > l_bound:
                idx2 -= 1
            # now we have this index we can accumulate value
            # in the ccg as long as we are in the window
            idx2_H = idx2
            while idx2_H < sz1 and spikes1[idx2_H] < H_bound:
                idx_C = halfbin + int(0.5 + (spikes1[idx2_H] - spikes2[idx1]) / binsize)
                # instead of flipping the CCG take the "flipped" index here
                idx_C = winsize_bins - (idx_C - 1) + 1
                C[idx_C] += 1
                idx2_H += 1
    return C, E


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
