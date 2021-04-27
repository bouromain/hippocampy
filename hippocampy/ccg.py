import numpy as np
import bottleneck as bn


def ccg(spikes1, spikes2, binsize=1e-3, max_lag=1000e-3):
    """
    Compute cross-correlograms between two spike trains
    BNot the fastest option right now, but I wil improve it after
    Time should be given in sec
    """

    spikes1 = np.asarray(spikes1)
    spikes2 = np.asarray(spikes2)

    # calculate all lags
    all_delta = spikes1[None, :] - spikes2[:, None]

    # create edges and ensure that they are odd
    winsize_bins = 2 * int(max_lag / binsize)
    if winsize_bins % 2 != 1:
        winsize_bins += 1
        max_lag += binsize / 2

    E = np.linspace(-max_lag, max_lag, winsize_bins + 1)

    # make the ccg
    c, _ = np.histogram(all_delta, E)

    return c, E


# from numba import jit
# @jit(nopython=True)
# def ccg2(spikes1, spikes2, binsize=1e-3, max_lag=1000e-3):
#     """
#     Fast crossCorr
#     """
#     # create edges and ensure that they are odd
#     winsize_bins = 2 * int(max_lag / binsize)

#     if winsize_bins % 2 != 1:
#         winsize_bins += 1
#         max_lag += binsize / 2

#     # Make edges (seem faster than np.linspace)
#     E = np.zeros(winsize_bins + 1)
#     for i in range(winsize_bins + 1):
#         E[i] = -max_lag + i * binsize

#     # initialise CCG
#     C = np.zeros(winsize_bins - 1)
#     # loop over spikes
#     for i in spikes1:
#         for j in spikes2:
#             d = i - j
#             if -max_lag < d < max_lag:
#                 C[np.searchsorted(E, d, side="left") - 1] += 1
#     return C, E


def continuous_ccg(spikes1, spikes2, tau=10e-3, max_lag=100e-3):
    """

    for now this function iis s simple translation of the following code:
    http://www.memming.com/codes/ccc.m.html
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.8458&rep=rep1&type=pdf

    """
    raise NotImplementedError("This should be tested")

    spikes1 = np.random.uniform(0, 1, 100)
    spikes2 = np.random.uniform(0, 1, 100)

    spikes1 = np.asarray(spikes1)
    spikes2 = np.asarray(spikes2)

    assert spikes1.size is not 0, "at least one spike is required for first spike train"
    assert (
        spikes2.size is not 0
    ), "at least one spike is required for second spike train"

    # calculate all the lags
    delta_t = spikes1[None, :] - spikes2[:, None]

    # Only keep the ones that are smaller than max_lag
    delta_t = np.ravel(delta_t)
    m = np.abs(delta_t) <= max_lag
    delta_t = delta_t[m]

    # sort the lags
    delta_t = np.sort(delta_t)

    # Now do the little stuff I am not sure to understand
    Q_plus = np.zeros_like(delta_t)
    Q_minus = np.zeros_like(delta_t)
    Q_minus[0] = 1

    exp_delta = np.exp(-np.diff(delta_t) / tau)
    N = delta_t.size

    for k in range(0, N - 1):
        Q_minus[k] = 1 + Q_minus[k - 1] * exp_delta[k - 1]
        Q_plus[-k + 2] = (Q_plus[-k + 1] + 1) * exp_delta[-k + 2]

    return Q_minus + Q_plus, delta_t
