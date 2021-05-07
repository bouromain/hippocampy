import numpy as np
import bottleneck as bn
from numba import jit


def ccg(spikes1, spikes2, binsize=1e-3, max_lag=1000e-3, normalization="count"):
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
                        - brillinger: as defined in Brillinger 1976
    """
    C, E = ccg_heart(spikes1, spikes2, binsize=1e-3, max_lag=1000e-3)

    if normalization is "conditional":
        ...
    elif normalization is "probability":
        ...
    elif normalization is "brillinger":
        ...

    return C, E


@jit(nopython=True)
def ccg_heart(spikes1, spikes2, binsize=1e-3, max_lag=1000e-3):
    """
    Fast crosscorrelation code:
    Assume that the spike trains are sorted

    Parameters:
                - spikes1
                - spikes2
                - binsize
                - max_lag


    Adapted from:
    G. Viejo crossCorr function
    M. Zugaro CCGEngine.c
    """

    assert normalization in [
        "count",
        "conditional",
        "probability",
        "brillinger",
    ], "Method not recognized"

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

    if sz1 < sz2:
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
                idx_C = winsize_bins - idx_C
                C[idx_C] += 1
                idx2_H += 1

    return C, E


# def ccg_slow(spikes1, spikes2, binsize=1e-3, max_lag=1000e-3):
#     """
#     Compute cross-correlograms between two spike trains
#     BNot the fastest option right now, but I wil improve it after
#     Time should be given in sec
#     """

#     spikes1 = np.asarray(spikes1)
#     spikes2 = np.asarray(spikes2)

#     # calculate all lags
#     all_delta = spikes1[None, :] - spikes2[:, None]

#     # create edges and ensure that they are odd
#     winsize_bins = 2 * int(max_lag / binsize)
#     if winsize_bins % 2 != 1:
#         winsize_bins += 1
#         max_lag += binsize / 2

#     E = np.linspace(-max_lag, max_lag, winsize_bins + 1)

#     # make the ccg
#     c, _ = np.histogram(all_delta, E)

#     return c, E

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
