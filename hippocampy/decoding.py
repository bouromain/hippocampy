import numpy as np
import bottleneck as bn
from hippocampy.matrix_utils import zscore
from hippocampy.binning import rate_map

from hippocampy.utils.type_utils import float_to_int


def cross_validate(
    var: np.ndarray,
    Q: np.ndarray,
    *,
    bins: np.ndarray,
    cross_validate_vec: np.ndarray,
    fs: int = 1,
    decode_method: str = "bayes",
    bin_method: str = "continuous",
    smooth_var_half_win: int = 5,
    smooth_pad_type: str = "reflect",
    verbose: bool = True,
):
    # check inputs
    var = np.array(var)
    Q = np.array(Q)

    if not decode_method in ["bayes", "frv"]:
        raise ValueError(f"Decoding method not recognized")

    if not bin_method in ["point_process", "continuous"]:
        raise ValueError("Method should be either continuous or point_process")

    if var.ndim != 1:
        raise ValueError("This function only work for 1D inputs")

    if Q.shape[1] != len(var) != len(cross_validate_vec):
        raise ValueError("Inputs should have the same size")

    # init inputs
    n_neurons, n_samples = Q.shape
    n_bins = len(bins)
    Z = np.empty((n_bins - 1, n_samples))
    Z.fill(np.nan)
    Tc = np.empty((n_neurons, n_bins - 1))
    Tc.fill(np.nan)

    cv_u = np.unique(cross_validate_vec)
    cv_u = cv_u[~np.isnan(cv_u)]

    for it_cv in cv_u:
        if verbose:
            print(f"Decoding for epoch {it_cv} of {len(cv_u)}")
        # compute tuning curves excluding the test epoch
        mask = cross_validate_vec == it_cv
        for it_q, tmp_q in enumerate(Q):
            Tc[it_q, :], _, _ = rate_map(
                var[~mask],
                tmp_q[~mask],
                bins=bins,
                fs=fs,
                smooth_half_win=smooth_var_half_win,
                smooth_pad_type=smooth_pad_type,
                method=bin_method,
            )
        # decode in one epoch for cross validation
        if decode_method == "bayes":
            Z[:, mask] = bayesian_1d(Q[:, mask], Tc)
        elif decode_method == "frv":
            Z[:, mask] = frv(Q[:, mask], Tc)

    return Z


def bayesian_1d(Q: np.ndarray, Tc: np.ndarray, prior=None, method="caim") -> np.ndarray:
    """
    bayesian_1d [summary]

    Parameters
    ----------
    Q : np.ndarray
        Boolean activity matrix, binned spikes/transient Q [n_neurons, n_samples]
    Tc : np.ndarray
        Tuning curves Tc [n_neurons, n_bins]
    prior : [type], optional
        either provide a prior or the code wll use a flat/ uniform one, by default None
    method : str, optional
        method to decode, either like in ref [1] "caim" or a naive "classic" baysian decode
        like in ref [2], ["caim","classic"] by default "caim"

    Returns
    -------
    np.ndarray
        [description]

    Reference
    ---------
    [1] Etter, G., Manseau, F. & Williams, S. A probabilistic framework 
        for decoding behavior from in vivo calcium imaging data. 
        Front. Neural Circuits 14, (2020).

    [2] Zhang, K., Ginzburg, I., McNaughton, B. L., and Sejnowski, T. J. (1998). 
        Interpreting neuronal population activity by reconstruction: unified 
        framework with application to hippocampal place cells. 
        J. Neurophysiol. 79, 1017–1044. doi: 10.1152/jn.1998.79.2.1017

    Note
    ----
    this code can be applied in 2D by raveling the positional bins and reshaping at the end

    """
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

    # ensure q is boolean
    Q = Q.astype(bool)
    # turn tuning curves into probabilities
    prob_active_knowing_bin = Tc / bn.nansum(Tc, axis=1)[:, None]
    prob_active = bn.nansum(Q, axis=1) / n_samples

    if method == "caim":
        # in caim decoding we use the probability that a cell is active
        # but also that it is inactive
        # turn tuning curves into probabilities
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
        # calculate the bayes prob for each time steps
        pTc = (prob_active_knowing_bin * prior) / prob_active[:, None]
        step_prod = pTc[:, :, None] * Q[:, None, :]

    # to avoid numerical overflow we use exp( log( x + 1 ) - 1)
    P = np.expm1(bn.nansum(np.log1p(step_prod), axis=0))
    # re-normalize to have a probability for each time step
    P = P / bn.nansum(P, axis=0)

    return P


def frv(Q: np.ndarray, Tc: np.ndarray) -> np.ndarray:
    """
    Decode by doing the correlation of the activity of each time steps 
    with a template activity. This decoding is similar to the one performed 
    in ref [1] or [2]

    To be efficient here we calculate the pearson correlation as:

    corr(x,y) = (1/n-1) sum( zscore(x) * zscore(y) )
    with n the number of sample in x and y

    Parameters
    ----------
    Q : np.ndarray
        activity matrix Q [n_neurons, n_samples]
    Tc : np.ndarray
        tunning curves matrix Tc [n_neurons, n_bins]

    Returns
    -------
    np.ndarray
        matrix of correlation of temporal bin activity with template activity 
        accross time
    
    Note
    ----
    this code can also be generalized in 2D by raveling input and reshaping at the end
    
    Reference
    ---------
    [1] Middleton SJ, McHugh TJ. Silencing CA3 disrupts temporal coding 
        in the CA1 ensemble.
        Nat Neurosci. 2016 Jul;19(7):945-51. doi: 10.1038/nn.4311. 
    [2] Wilson MA, McNaughton BL. Dynamics of the hippocampal ensemble 
        code for space. Science. 1993 Aug 20;261(5124):1055-8. 
        doi: 10.1126/science.8351520.
    """
    if Q.shape[0] != Tc.shape[0]:
        raise ValueError(
            "Q and tunning curve matrix should share their 0th dimension (number neurons)"
        )

    n_neurons = Q.shape[0]

    Tc_z = zscore(Tc, axis=0)
    Q_z = zscore(Q, axis=0)

    return (1 / (n_neurons - 1)) * (Tc_z.T @ Q_z)


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
    if not method in ["max", "com"]:
        raise ValueError("Method not recognized")

    tmp = np.empty((P.shape[1]))
    tmp.fill(np.nan)
    mask_nan = np.all(np.isnan(P), axis=0)

    if method == "max":
        tmp[~mask_nan] = bn.nanargmax(P[:, ~mask_nan], axis=0)
    elif method == "com":
        tmp[~mask_nan] = (
            P[~mask_nan]
            * np.arange(P.shape[0])[:, None]
            / bn.nansum(P[~mask_nan], axis=0)
        )

    return tmp


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
    true_vals: np.ndarray, decoded: np.ndarray, full_posterior=None
) -> np.ndarray:
    """
    compute confusion_matrix or full posterior confusion matrix

    Parameters
    ----------
    true_vals : np.ndarray
        vector of true values 
    decoded : np.ndarray
        either a posterior matrix or a vector of decoded values
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

    return conf_mat / bn.nansum(conf_mat, axis=0)
