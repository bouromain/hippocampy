import numpy as np
import bottleneck as bn

from hippocampy.utils.type_utils import float_to_int


def bayesian_1d(Q: np.ndarray, Tc: np.ndarray, prior=None, method="caim") -> np.ndarray:
    ...
    # this code can be applied in 2D by raveling the positional bins and reshaping at the end
    # Q [n_neurons, n_samples]
    # Tc [n_neurons, n_bins]

    # print("still in development")

    # Q = np.random.uniform(0, 1, (50, 15000)) > 0.95
    # Tc = np.random.uniform(0, 1, (50, 100))
    # prior = None

    if method not in ["caim", "classic"]:
        raise ValueError(f"Method {method} not implemented")

    _, n_samples = Q.shape
    n_bins = Tc.shape[1]

    if prior is None:
        # if not specified assume homogenous prior
        prior = np.ones((1, n_bins)) / n_bins
    else:
        # else check  the prior is correct
        prior = np.array(prior)
        assert prior.shape[1] == n_bins
        # make sure the prior sum to 1
        prior = prior / bn.nansum(prior)

    # initialize output probability matrix
    if method == "caim":
        # in caim decoding we use the probability that a cell is active
        # but also that it is inactive
        # turn Tuning curves into probabilities
        Q = Q.astype(bool)
        prob_active_knowing_bin = Tc / bn.nansum(Tc, axis=1)[:, None]
        prob_active = bn.nansum(Q, axis=1) / n_samples

        # calculate the bayes prob for each time steps
        pTc = (prob_active_knowing_bin * prior) / prob_active[:, None]
        n_pTc = ((1 - prob_active_knowing_bin) * prior) / (1 - prob_active[:, None])

        # here it is a bit tricky, we do a 3D matrix to replicate the previous
        # probability in time.
        # However, as we multiply by Q (1 active, 0 not active) the probability will
        # be replicated only when the cell is active (for step prod act, the reverse for
        # step_prod_nact). So then we can sum them to obtain the correct step probability.

        step_prod_act = pTc[:, :, None] * Q[:, None, :]
        step_prod_nact = n_pTc[:, :, None] * ~Q[:, None, :]
        step_prod = step_prod_act + step_prod_nact

    elif method == "classic":
        # turn Tuning curves into probabilities
        Q = Q.astype(bool)
        prob_active_knowing_bin = Tc / bn.nansum(Tc, axis=1)[:, None]
        prob_active = bn.nansum(Q, axis=1) / n_samples

        #
        pTc = (prob_active_knowing_bin * prior) / prob_active[:, None]
        step_prod = pTc[:, :, None] * Q[:, None, :]

    # to avoid numerical overflow we use exp( log( x + 1 ) -1)
    P = np.expm1(bn.nansum(np.log1p(step_prod), axis=0))
    # re-normalize to have a probability for each time step
    P = P / bn.nansum(P, axis=0)

    return P


def decoded_state(P: np.ndarray, method: str = "max") -> np.ndarray:
    """
    Returns the decoded state from a probability matrix

    Parameters
    ----------
    P : np.ndarray
        Probability matrix in the form [n_states, n_time]
    method : str, optional
        method to define the decoded state, by default "max"

    Returns
    -------
    np.ndarray
        vector of decoded states
    """

    if method == "max":
        return bn.nanargmax(P, axis=0)
    elif method == "com":
        return P * np.arange(P.shape[0])[:, None] / bn.nansum(P, axis=0)


def decoded_error(var_real: np.ndarray, var_decoded: np.ndarray) -> np.ndarray:
    """
    decoded_error [summary]

    Parameters
    ----------
    var_real : np.ndarray
        [description]
    var_decoded : np.ndarray
        [description]

    Returns
    -------
    np.ndarray
        [description]

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    ValueError
        [description]
    """

    if var_real.ndim > 1:
        raise ValueError("Real values should be 1D")
    if var_decoded.ndim > 1:
        raise ValueError("Decoded values should be 1D")

    if (var_decoded.shape == var_real.shape).all():
        raise ValueError("Real and decoded vector should have the same size")

    return np.abs(var_real - var_decoded)


def confusion_matrix(
    decoded: np.ndarray, true_vals: np.ndarray, full_posterior=None
) -> np.ndarray:
    """
    compute confusion_matrix or full posterior confusion matrix

    Parameters
    ----------
    decoded : np.ndarray
        either a posterior matrix or a vector of decoded values
    true_vals : np.ndarray
        vector of true values 
    full_posterior : str, optional
        method to compute the full posterior, by default None

    Returns
    -------
    np.ndarray
        confusion matrix
    """

    if full_posterior not in [None, "median", "mean"]:
        raise ValueError("Method should be either [None, median,mean] ")

    if full_posterior is None:
        bins = np.unique(np.hstack((decoded.ravel(), true_vals.ravel())))
        bin_edges = np.append(bins, bins[-1] + 1)
        conf_mat, E = np.histogramdd(
            np.array([decoded, true_vals]).T, (bin_edges, bin_edges)
        )

    else:
        n_bins, n_sample = decoded.shape
        true_vals = float_to_int(true_vals)
        full_conf_mat = np.empty((n_bins, n_bins, n_sample)) * np.nan
        full_conf_mat[:, true_vals, :] = decoded

        if full_posterior == "mean":
            conf_mat = bn.nanmean(full_conf_mat, axis=2)
        elif full_posterior == "median":
            conf_mat = bn.nanmedian(full_conf_mat, axis=2)

    conf_mat = conf_mat / bn.nansum(conf_mat, axis=0)
    return conf_mat
