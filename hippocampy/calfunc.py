import bottleneck as bn
import numpy as np
import tqdm as tqdm
from scipy.stats import siegelslopes
from sklearn.linear_model import RANSACRegressor

from hippocampy.matrix_utils import remove_small_objects, zscore, moving_win


def subtract_neuropil(Froi, Fneu, method="fixed", downsample_ratio=10):
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

    # this reshape is particularly important for the robust method but I put it
    # here for homogeneity
    if Froi.shape[1] < Froi.shape[0] or Fneu.shape[1] < Fneu.shape[0]:
        raise SyntaxError("Data should be given as [cells, samples]")

    if method is "fixed":
        F = Froi - (0.7 * Fneu)

    elif method is "robust":
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
    else:
        raise NotImplementedError("Method not implemented")

    return F


def transientSH(F, axis=1):
    """
    find transient as in Allegra, Posani, Schmidt-Hieber

    “Events” were identified as contiguous regions in the d​ F ​ / ​ F signal exceeding a
    threshold of mean +2.5 standard deviations of the overall d​ F ​ / ​ F signal, and exceeding an integral
    above threshold of 7,000 d​ F ​ / ​ F ​ .
    """
    F_mean = bn.nanmean(F, axis=axis)
    F_std = bn.nanstd(F, axis=axis)


def transientRoy(F, threshold=2.5, min_length=9):
    """
    find transient as in Roy 2017
    Ca 2+ events were detected by applying a threshold (greater than 2 standard
    deviations from the dF/F signal) at the local maxima of the dF/F signal.
    Since we employed GCaMP6f, our analysis used a threshold of > = 5 frames (250 ms)
    """

    # Zscore traces
    F_z = zscore(F, axis=1)

    # threshold trace above 2.5 std
    F_t = F_z > threshold
    # remove transients that are shorter than a given threshold
    F_t = np.apply_along_axis(remove_small_objects, axis=1, arr=F_t, min_sz=min_length)
    return F_t


# p = "/home/bouromain/Documents/tmpData/m4550/20210720/1/suite2p/plane0/"
# from hippocampy.io.suite2p import loadAllS2p

# F, Fneu, spks, stat, ops, iscell = loadAllS2p(p)

# dF = subtract_neuropil(F, Fneu, method="fixed", downsample_ratio=10)

# a = moving_win(dF[0,:] ,15*60, (15*60)-1, end="pad")
# aa = np.percentile(a , 8, axis=1)

# import matplotlib.pyplot as plt
# plt.plot(dF[0,:])
# plt.plot(aa)
# plt.xlim([391000,394000])
