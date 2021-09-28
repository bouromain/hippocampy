import bottleneck as bn
import numpy as np
import tqdm as tqdm
from sklearn.linear_model import RANSACRegressor
from oasis.functions import deconvolve as _deconvolve
from hippocampy.matrix_utils import remove_small_objects, rolling_quantile, first_true


def subtract_neuropil(Froi, Fneu, *, method="fixed", downsample_ratio=10):
    """
    This function will perform neuropil substraction from a given fluorescence
    trace F = Froi - (c * Fneu). The constant c can be defined as a fixed value
    (classically 0.7) or calculated by a bounded robust regression between Froi
    and Fneu (see Chen 2013b)

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
    - Ultrasensitive fluorescent proteins for imaging neuronal activity,
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
        c[np.logical_not(c_valid)] = np.median(c[c_valid])

        # Calculate F
        F = Froi - c[:, None] * Fneu

    return F


def deconvolve(F: np.ndarray, fs: int = 30, tau: float = 0.7, verbose: bool = True):
    """
    deconvolve calcium traces using oasis algorithm

    Parameters
    ----------
    F : np.ndarray
        calcium traces [n_traces, n_samples]
    fs : int, optional
        sampling frequency, by default 30
    tau : float, optional
        decay time constant in sec, by default 0.7
    verbose : bool, optional
        make this function chatty, by default True

    Returns
    -------
    c : np.ndarray
        denoised calcium traces [n_traces, n_samples]
    s : np.ndarray
        deconvolved calcium traces [n_traces, n_samples]
    """

    g = np.exp(-(1 / (fs * tau)))

    c = np.empty_like(F)
    s = np.empty_like(F)
    if verbose:
        for itf, f in tqdm.tqdm(enumerate(F), total=F.shape[0]):
            c[itf, :], s[itf, :], _, _, _ = _deconvolve(f, g=[g])
    else:
        for itf, f in enumerate(F):
            c[itf, :], s[itf, :], _, _, _ = _deconvolve(f, g=[g])

    return c, s


def transient(F: np.ndarray, threshold=2.5, fs: int = 30):
    ...


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
    -quantile: quantile to substract [0-1] (default 8th)

    Return
    ------
    np.array of Fluorescence with the baseline substracted

    Reference
    ---------
    Dombeck 2010

    Slow time-scale changes in the fluorescence traces were removed by 
    examining the distribution of fluorescence in a ~15-s interval around 
    each sample time point and subtracting the 8% percentile value.
    """
    Q = rolling_quantile(F, win_size, quantile)
    return F - Q
