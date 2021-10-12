import numpy as np
import bottleneck as bn


def entr(p, axis=-1):
    """
    calculate the entropy of a vector
    """
    return -bn.nansum(p * np.log2(p), axis=axis)


def rel_entr(p, q, axis=-1):
    """
    TO DO:
            - potentially implement change of base with out /= out
            - potentially add a check to verify probability distributions
    """
    # iszero = np.logical_and(p == 0 , q == 0)
    # isneg = np.logical_or(p < 0 , q < 0 )

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    # z[iszero] = 0
    # z[isneg] = np.Inf
    return bn.nansum(p * np.log2(p / q), axis=axis)


def kl_div(p, q, axis=-1):
    """
    See  http://cvxr.com/cvx/ kldiv
    for a potentialadditional term is why the - p + q is added
    """
    return rel_entr(p, q, axis=axis)


def jensen_shannon(p, q, base=None, axis=-1):
    """
    Calculate Jensen-Shannon distance between two 1D probability
    distributions adapted from the scipy version.

    Return Jensen-Shannon distance which is the square root
    of the JS divergence

    """

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    m = (p + q) / 2
    js = rel_entr(p, m, axis=axis) + rel_entr(q, m, axis=axis)

    if base is not None:
        js /= np.log(base)

    return np.sqrt(js)
