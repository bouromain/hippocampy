import numpy as np
import bottleneck as bn
from numba import jit


def ccg(
    spikes1, spikes2, binsize=1e-3, max_lag=1000e-3, normalization="count", safe=False
):
    """
    Fast crosscorrelation code:
    Assume that the spike trains are sorted

    Parameters:
                - spikes1
                - spikes2
                - binsize
                - max_lag
                - normalisation:
                        - count: no normalisation
                        - conditional: probability of spk2 knowing spk1
                        - probability: sum to one
    """
    assert normalization in [
        "count",
        "conditional",
        "probability",
    ], "Method not recognized"

    if safe:
        spikes1 = np.asarray(spikes1).sort()
        spikes2 = np.asarray(spikes2).sort()

    C = ccg_heart(spikes1, spikes2, binsize=binsize, max_lag=max_lag)

    if normalization is "conditional":
        sz1 = len(spikes1)
        C = C / sz1
    elif normalization is "probability":
        C = C / bn.nansum(C)

    return C


@jit(nopython=True)
def ccg_heart(spikes1, spikes2, binsize=1e-3, max_lag=1000e-3):
    """
    Fast crosscorrelation code:
    Assume that the spike trains are sorted

    Parameters:
                - spikes1: first time serie of spikes
                - spikes2: second time serie of spikes
                - binsize: size of one bin
                - max_lag: size of the half window

    Return:
                - C: Cross-correlogram
                - E: Edges of the Cross-correlogram

    To Do:
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
        print("yo")
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
    return C


a = np.array([10, 20, 30, 40, 50])
b = np.array([12, *a])

c = ccg(b, a, 1, 5)

import matplotlib.pyplot as plt

plt.plot(np.linspace(-10, 10, 21), c)
plt.xlim(-4, 4)


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
