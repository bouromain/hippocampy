import numpy as np
import bottleneck as bn
from hippocampy.matrix_utils import zscore
from hippocampy.binning import rate_map

from hippocampy.utils.type_utils import float_to_int
from hippocampy.stats.distance import cos_sim, pairwise_euclidian
from scipy.sparse import coo_matrix


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


def frv(Q: np.ndarray, Tc: np.ndarray, *, method="pearson") -> np.ndarray:
    """
    Decode by doing the similarity of the activity of each time steps
    with a template activity. This decoding is similar to the one performed
    in ref [1] or [2]

    Parameters
    ----------
    Q : np.ndarray
        activity matrix Q [n_neurons, n_samples]
    Tc : np.ndarray
        tunning curves matrix Tc [n_neurons, n_bins]
    method : str
        method to use to compute the firing rate vector similarity
        [pearson, cosine, euclidian]

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
    [3] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    if Q.shape[0] != Tc.shape[0]:
        raise ValueError(
            "Q and tunning curve matrix should share their 0th dimension (n samples)"
        )

    if method not in ["pearson", "cosine", "euclidian"]:
        raise ValueError(f"Method {method} not recognized")

    if method == "pearson":
        n = Q.shape[0]
        Tc_z = zscore(Tc, axis=0)
        Q_z = zscore(Q, axis=0)

        return (1 / (n - 1)) * (Tc_z.T @ Q_z)

    elif method == "cosine":
        return cos_sim(Q, Tc)

    elif method == "euclidian":
        return pairwise_euclidian(Tc.T, Q)


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
    mask_nan = np.all(np.isnan(P), axis=0)

    if method == "max":
        tmp[~mask_nan] = bn.nanargmax(P[:, ~mask_nan], axis=0)
    elif method == "com":
        tmp = ((np.arange(P.shape[0])[None, :] @ P) / bn.nansum(P, axis=0)).squeeze()

    tmp[mask_nan] = np.nan

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
    x_true: np.ndarray,
    x_predicted: np.ndarray,
    *,
    sample_weight=None,
    labels=None,
    normalize=None,
) -> np.ndarray:
    """
    confusion_matrix _summary_

    Parameters
    ----------
    x_true : np.ndarray
        true values
    x_predicted : np.ndarray
        predicted values
    labels : _type_, optional
        label corresponding to the values, by default None
    normalize : _type_, optional
        _description_, by default None

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_

    Reference
    ---------
    Mostly taken from:
    https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/metrics/_classification.py#L222
    """
    # check input
    x_true = np.array(x_true)
    x_predicted = np.array(x_predicted)

    # for now we will only accept integer input (eg index of real and predicted values).
    # categorical and or float value need stronger check, that it slightly more
    # complicated but I am not sure this is needed. I'll implement it later
    if x_true.dtype.kind != "i":
        x_true = float_to_int(x_true)
    if x_predicted.dtype.kind != "i":
        x_predicted = float_to_int(x_predicted)

    if labels is None:
        labels = np.unique(x_true)
    else:
        labels = np.array(labels)
        # if we did not input labels check the user provided ones
        if not all([True if l in x_true else False for l in labels]):
            raise ValueError("All label should be in x_true")
        if not all([True if l in x_predicted else False for l in labels]):
            raise ValueError("All label should be in x_predicted")

    if sample_weight is None:
        sample_weight = np.ones(x_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    if normalize not in ["true", "pred", "all", None]:
        raise ValueError("normalize should either be 'true', 'pred', 'all', None")

    n_labels = labels.size
    # If labels are not consecutive integers starting from zero, then
    # y_true and y_pred must be converted into index form
    need_index_conversion = not (
        labels.dtype.kind in {"i", "u", "b"}
        and np.all(labels == np.arange(n_labels))
        and x_true.min() >= 0
        and x_predicted.min() >= 0
    )
    if need_index_conversion:
        label_to_ind = {y: x for x, y in enumerate(labels)}
        x_predicted = np.array([label_to_ind.get(x, n_labels + 1) for x in x_predicted])
        x_true = np.array([label_to_ind.get(x, n_labels + 1) for x in x_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(x_predicted < n_labels, x_true < n_labels)
    if not np.all(ind):
        x_predicted = x_predicted[ind]
        x_true = x_true[ind]
        # also eliminate weights of eliminated items
        sample_weight = sample_weight[ind]

    # check if we need to bin or digitize x_true, x_predicted
    # bin them with the sparse matrix trick
    if sample_weight.dtype.kind in {"i", "u", "b"}:
        dtype = np.int64
    else:
        dtype = np.float64

    cm = coo_matrix(
        (sample_weight, (x_true, x_predicted)),
        shape=(n_labels, n_labels),
        dtype=dtype,
    ).toarray()

    with np.errstate(all="ignore"):
        # to avoid division errors display
        if normalize == "true":
            cm = cm / bn.nansum(cm, axis=1)
        elif normalize == "pred":
            cm = cm / bn.nansum(cm, axis=0)
        elif normalize == "all":
            cm = cm / bn.nansum(cm)
    return cm


def confusion_matrix_full(x_true: np.ndarray, P: np.ndarray, method: str = "mean"):
    """
    confusion_matrix_full average the posterior probability matrix for each true value.

    Parameters
    ----------
    x_true : np.ndarray
        vector of true values
    P : np.ndarray
        Posterior probability matrix for each true values
    method : str, optional
        method to average the posterior probability matrix [ "median", "mean"],
        by default "mean"

    Returns
    -------
    confusion_matrix: np.ndarray

    """

    if method not in ["median", "mean"]:
        raise ValueError("Method should be either [median,mean] ")

    x_true = np.array(x_true)
    P = np.array(P)

    n_bins, n_sample = P.shape

    if x_true.dtype.kind != "i":
        x_true = float_to_int(x_true)

    full_conf_mat = np.empty((n_bins, n_bins, n_sample)) * np.nan
    full_conf_mat[:, x_true, range(n_sample)] = P

    if method == "mean":
        cm = bn.nanmean(full_conf_mat, axis=2)
    elif method == "median":
        cm = bn.nanmedian(full_conf_mat, axis=2)

    return cm / bn.nansum(cm, axis=0)
