import os.path as op

import bottleneck as bn
import numpy as np
import pandas as pd
import tqdm as tqdm

# from oasis.functions import deconvolve as _deconvolve
from sklearn.linear_model import RANSACRegressor

from hippocampy.io.matlab import loadmat
from hippocampy.matrix_utils import (
    first_true,
    remove_small_objects,
    rolling_quantile,
    zscore,
)
from hippocampy.stats.stats import mad


def subtract_neuropil(Froi, Fneu, *, method="fixed", downsample_ratio=10):
    """
    This function will perform neuropil substraction from a given fluorescence
    trace F = Froi - (c * Fneu). The constant c can be defined as a fixed value
    (classically 0.7) or calculated by a bounded robust regression between Froi
    and Fneu (see Ref 1)

    Parameters
    ----------
    - Froi: ROI fluorescence vector or matrix given as [cells, samples]
    - Fneu: Neuropil fluorescence vector or matrix given as [cells, samples]
    - method:
            - fixed (default) where c = 0.7
            - robust: calculation with robust regression
    - downsample_ratio: downsampling ratio of the data for "robust" method

    Returns
    -------
    - F: Fluorescence vector or matrix given as [cells, samples]
        with Fneu substracted

    Reference
    ---------
    [1] Ultrasensitive fluorescent proteins for imaging neuronal activity,
        TW Chen TJ Wardill Y Sun SR Pulver SL Renninger A Baohan ER Schreiter
        RA Kerr MB Orger V Jayaraman LL Looger K Svoboda DS Kim  (2013b)
    """
    Froi = np.asarray(Froi)
    Fneu = np.asarray(Fneu)

    if method not in ["fixed", "robust"]:
        raise NotImplementedError("Method not implemented")

    # this reshape is particularly important for the robust method but I put it
    # here for homogeneity
    if Froi.shape[1] < Froi.shape[0] or Fneu.shape[1] < Fneu.shape[0]:
        raise SyntaxError("Data should be given as [cells, samples]")

    if method == "fixed":
        F = Froi - (0.7 * Fneu)

    elif method == "robust":
        # Robustly fit linear model with RANSAC algorithm
        ransac = RANSACRegressor()
        c = np.empty(Froi.shape[0])

        for itF in tqdm.tqdm(range(Froi.shape[0])):
            x = np.atleast_2d(Froi[itF, ::downsample_ratio])
            y = np.atleast_2d(Fneu[itF, ::downsample_ratio])
            ransac.fit(x.T, y.T)
            c[itF] = ransac.estimator_.coef_

        # Values outside of 0.5 and 1 are considered as
        # outliers and will be assigned to the median
        # of the coefficients in range [0.5 < c < 1]
        c_valid = np.logical_and(c > 0.5, c < 1)

        if bn.nansum(c_valid) > 3:
            c[np.logical_not(c_valid)] = bn.nanmedian(c[c_valid])
        else:
            # this is quite rare but need to be taken care of. if we have less
            # than 3 valid value, go back to 0.7
            c[:] = 0.7

        # Calculate F
        F = Froi - c[:, None] * Fneu

    return F


# def deconvolve(F: np.ndarray, fs: int = 30, tau: float = 0.7, verbose: bool = True):
#     """
#     deconvolve calcium traces using oasis algorithm

#     Parameters
#     ----------
#     F : np.ndarray
#         calcium traces [n_traces, n_samples]
#     fs : int, optional
#         sampling frequency, by default 30
#     tau : float, optional
#         decay time constant in sec, by default 0.7
#     verbose : bool, optional
#         make this function chatty, by default True

#     Returns
#     -------
#     c : np.ndarray
#         denoised calcium traces [n_traces, n_samples]
#     s : np.ndarray
#         deconvolved calcium traces [n_traces, n_samples]
#     b : np.ndarray
#         deconvolved calcium traces baseline
#     """

#     g = np.exp(-(1 / (fs * tau)))

#     c = np.empty_like(F)
#     s = np.empty_like(F)
#     b = np.empty((F.shape[0], 1))
#     if verbose:
#         for itf, f in tqdm.tqdm(enumerate(F), total=F.shape[0]):
#             c[itf, :], s[itf, :], b[itf], _, _ = _deconvolve(f, g=[g])
#     else:
#         for itf, f in enumerate(F):
#             c[itf, :], s[itf, :], b[itf], _, _ = _deconvolve(f, g=[g])

#     return c, s, b


def calc_dF(
    F: np.ndarray, window: int, *, type_win="median", quantile: float = 0.08, axis=-1
):
    """
    calc_dF calculate dF/F as F - F_0 / F0 with F_0 defined as
    the mean, median in a sliding time window

    Parameters
    ----------
    F : [np.ndarray]
        input traces
    window : [int]
        window length in samples
    quantile : [float]
        value of the quantile, ignored for mean or median
    axis : int, optional
        axis to work on, by default -1

    Returns
    -------
    dF/F [np.ndarray]
        deltaF over F
    """
    if type_win not in ["median", "mean", "quantile"]:
        raise ValueError(f"Window type {type_win} not recognized")

    if type_win.lower() == "median":
        F_0 = bn.move_median(F, window, axis=axis, min_count=1)
    elif type_win.lower() == "mean":
        F_0 = bn.move_mean(F, window, axis=axis, min_count=1)
    elif type_win.lower() == "quantile":
        F_0 = rolling_quantile(F, window, quantile=quantile)

    dF = (F - F_0) / F_0
    return dF.astype(np.float32)  # to make it like suite2p F and Fneu (also float 32)


def transient(
    F: np.ndarray,
    S: np.ndarray,
    threshold: float = 1.1,
    min_len_event: int = 3,
    spike_norm: str = "sliding",
    sliding_win_len: int = 18000,
):
    """
    Transient detection inspired from Grosmark 2020.
    It will take the traces, slightly denoise them to them deconvolve them
    Events are then defined as deconvolved signal higher than Threshold time
    the level of estimated noise. This noise is defined as the
    median absolute deviation between the initial trace and the traces
    reconstructed with the deconvolution algorithm OASIS

    Parameters
    ----------
    F : np.ndarray
        calcium traces [n_traces, n_samples]
    S : np.ndarray
        deconvolved traces. It is easier to provide the deconvolved traces here
        in order to be able to feed deconvolved traces from different deconvolution
        algo such as OASIS or CASCADE
    threshold : float, optional
        threshold to detect event. Defined as threshold  times the "mad"
        or "zscore" of the deconvolved trace, by default 1.1

        This can be given as a float or a vector if you need different
        threshold for different epoch (activity, inactivity). Grosmark 2021 set
        a threshold of 1.5 during active epoch and 1.25 during rest.
    min_len_event: int
        minimum number of frame crossing the threshold to be kept as a potential
        event
    spike_norm : str, optional
        normalisation of the deconvolved spikes: ["mad", "zscore","sliding",
        by default "sliding"
    sliding_win_len: int, optional (only for option sliding)
        define the size of the sliding window

    Returns
    -------
    S_b np.ndarray
        binarised spike matrix
    Ts list
        spike times

    Raises
    ------
    ValueError
        [description]
    """
    if spike_norm not in ["mad", "zscore", "sliding"]:
        raise ValueError(f"{spike_norm} not a valid spike normalization ")

    n_cells, _ = F.shape

    if spike_norm == "mad":
        F_reconvolved = F * S
        # estimate noise as the mad of the residuals of the difference between
        #  the initial traces and the denoised ones. Then normalize the spike estimate
        noise = mad(F_reconvolved - F, axis=1)
        S_b = S / noise[:, None]
    elif spike_norm == "zscore":
        S_b = zscore(S, axis=1)
    elif spike_norm == "sliding":
        # calculate a sliding mad in a given window
        F_reconvolved = F * S
        noise = F_reconvolved - F
        for i, x in tqdm.tqdm(enumerate(noise), total=n_cells):
            tmp = (
                pd.DataFrame(x)
                .rolling(sliding_win_len, center=True, min_periods=1)
                .quantile(quantile=0.5)
                .squeeze()
            )
            tmp = np.abs(tmp - noise[i, :])
            noise[i, :] = (
                pd.DataFrame(tmp)
                .rolling(sliding_win_len, center=True, min_periods=1)
                .quantile(quantile=0.5)
                .squeeze()
            )
        S_b = S / noise

    # threshold the spike estimate
    if isinstance(threshold, np.ndarray):
        S_b = np.greater_equal(S_b, threshold)
    else:
        S_b = S_b > threshold

    # remove short events
    S_b = remove_small_objects(S_b, min_size=min_len_event, axis=1)

    # in case multiple successive samples cross the threshold, only keep the first
    S_b = first_true(S_b)

    if S_b.squeeze().ndim == 1:
        S_b = S_b.squeeze()
        return S_b, np.nonzero(S_b)[0]
    else:
        return S_b, [np.nonzero(s)[0] for s in S_b]


def transient_simple(
    F: np.ndarray, threshold: float = 2.5, min_length: int = 5, axis: int = -1
) -> list:
    """
    transientRoy
    find transient as in Roy 2017
    Ca 2+ events were detected by applying a threshold (greater than 2 standard
    deviations from the dF/F signal) at the local maxima of the dF/F signal.
    Since we employed GCaMP6f, our analysis used a threshold of > = 5 frames (250 ms)


    Parameters
    ----------
    F : np.ndarray (n_cells, n_samples) or (n_samples,n_cells)
        Fluorescence matrix
    threshold : float, optional
        number of std above the mean to use as a threshold, by default 2.5
    min_length : int, optional
        minimum of frame (samples) that the event needs to last to be kept, by default 5
    axis : int, optional
        axis to perform the detection across, by default -1

    Returns
    -------
    list
        List containing the indexes of the detected events per cells
    """

    # define the threshold for candidate events
    F_mean = bn.nanmean(F, axis=axis)
    F_std = bn.nanstd(F, axis=axis)
    T = F_mean + threshold * F_std

    # Threshold the signal
    F_b = F > T[:, None]

    # remove transients that are shorter than a given threshold
    F_t = np.apply_along_axis(
        remove_small_objects, axis=axis, arr=F_b, min_sz=min_length
    )

    F_t = first_true(F_t)
    if axis == 1 or axis == -1:
        return [np.nonzero(f)[0] for f in F_t]
    elif axis == 0:
        return [np.nonzero(f)[0] for f in F_t.T]


def detrend_F(F, win_size, quantile=0.08):
    """
    Function to remove slow time scale changes in fluorescence traces. It
    does it as describes in Dombeck 2010. It calculates the 8th percentile in
    a window of size win_size around each sample time to define a baseline. This
    baseline can then be substracted from the raw signal to correct it.

    Parameters
    ----------
    -F: fluorescence trace [n_cells, n_samples]
    -winsize: size of the window in samples
    -quantile: quantile to subtract [0-1] (default 8th)

    Return
    ------
    np.array of Fluorescence with the baseline subtracted

    Reference
    ---------
    Dombeck 2010

    Slow time-scale changes in the fluorescence traces were removed by
    examining the distribution of fluorescence in a ~15-s interval around
    each sample time point and subtracting the 8% percentile value.
    """
    Q = rolling_quantile(F, win_size, quantile)
    return F - Q


def noise_level(F: np.ndarray, fs: int, axis=-1) -> np.ndarray:
    """
    compute noise level as defined in the cascade paper [1]

    Parameters
    ----------
    F : np.ndarray
        fluorescence traces array
    fs : int
        sampling rate
    axis : int, optional
        axis to work along, by default -1

    Returns
    -------
    np.ndarray
        array of noise level in percent
    Reference
    ---------
    [1] Rupprecht P, Carta S, Hoffmann A, Echizen M, Blot A, AC Kwan, Dan Y,
        Hofer SB, Kitamura K, Helmchen F*, Friedrich RW*,
        A database and deep learning toolbox for noise-optimized, generalized
        spike inference from calcium imaging, Nature Neuroscience (2021)
    """

    noise = mad(F, axis=axis) / np.sqrt(fs)
    return noise * 100


def load_Fall_mat(Fall_path: str):
    """
    Load the Fall.mat file from suite2p to numpy format

    Parameters
    ----------
    Fall_path : str
        path of the Fall file

    Returns
    -------
    F: np.array
        Fluorescence (n_roi, n_sample)
    Fneu: np.array
        Neuropile fluorescence (n_roi, n_sample)
    spks: np.array
        deconvolved spike estimate (n_roi, n_sample)
    iscell: np.array
        output of suite2p classifier (n_roi, 2) 
    ops: dict
        ops dict from suite2p containing various infor

    stat: list
        list of dict containing information about each roi  

    """

    if not op.exists(Fall_path):
        raise FileNotFoundError

    tmps2p = loadmat(Fall_path)

    # store suite2p data in npy format
    F = np.array(tmps2p["F"], dtype=np.float32)
    Fneu = np.array(tmps2p["Fneu"], dtype=np.float32)
    spks = np.array(tmps2p["spks"], dtype=np.float32)
    iscell = np.array(tmps2p["iscell"], dtype=np.float32)

    # for ops and stat data, the loading of the matlab data in python is a bit weird
    # it loads the keys of the values in a dtype fields and the values as items of an
    # 'empty' np.array. For the stat variable, it does the same but in a list of
    # "empty" np.arrays.
    ops = tmps2p["ops"]
    # reformat loaded data
    k = eval(str(ops.dtype))
    val = ops.item()
    ops = {kk[0]: vv for kk, vv in zip(k, val)}

    stat = tmps2p["stat"]
    # do the same for stat
    stat = [{kk[0]: vv for kk, vv in zip(eval(str(s.dtype)), s.item())} for s in stat]

    return F, Fneu, spks, iscell, stat, ops


def Fall_to_npy(Fall_path: str, overwrite: bool = False, return_data: bool = False):
    """
    Fall_to_npy read Fall.mat file, convert it to npy and save it

    Parameters
    ----------
    Fall_path : str
        path of the Fall data
    overwrite : bool, optional
        specify if it should overwrite existing data, by default False
    return_data : bool, optional
        specify if we want to output the data, by default False

    Returns
    -------
    F: np.array
        Fluorescence (n_roi, n_sample)
    Fneu: np.array
        Neuropile fluorescence (n_roi, n_sample)
    spks: np.array
        deconvolved spike estimate (n_roi, n_sample)
    iscell: np.array
        output of suite2p classifier (n_roi, 2) 
    ops: dict
        ops dict from suite2p containing various infor

    stat: list
        list of dict containing information about each roi  
    """
    F, Fneu, spks, iscell, stat, ops = load_Fall_mat(Fall_path)

    F_path = Fall_path.replace("Fall.mat", "F.npy")
    if not op.exists(F_path) or overwrite:
        with open(F_path, "wb") as fio:
            np.save(fio, F, allow_pickle=True)

    Fneu_path = Fall_path.replace("Fall.mat", "Fneu.npy")
    if not op.exists(Fneu_path) or overwrite:
        with open(Fneu_path, "wb") as fio:
            np.save(fio, Fneu, allow_pickle=True)

    spks_path = Fall_path.replace("Fall.mat", "spks.npy")
    if not op.exists(spks_path) or overwrite:
        with open(spks_path, "wb") as fio:
            np.save(fio, spks, allow_pickle=True)

    ops_path = Fall_path.replace("Fall.mat", "ops.npy")
    if not op.exists(ops_path) or overwrite:
        with open(ops_path, "wb") as fio:
            np.save(fio, ops, allow_pickle=True)

    iscell_path = Fall_path.replace("Fall.mat", "iscell.npy")
    if not op.exists(iscell_path) or overwrite:
        with open(iscell_path, "wb") as fio:
            np.save(fio, iscell, allow_pickle=True)

    stat_path = Fall_path.replace("Fall.mat", "stat.npy")
    if not op.exists(stat_path) or overwrite:
        with open(stat_path, "wb") as fio:
            np.save(fio, stat, allow_pickle=True)

    if return_data:
        return F, Fneu, spks, iscell, stat, ops

