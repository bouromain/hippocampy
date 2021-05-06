import numpy as np
import bottleneck as bn
from sklearn.decomposition import FastICA
from hippocampy.matrix_utils import smooth1D


"""
For other methods see also:

https://elifesciences.org/articles/19428

"""


def calc_template(spike_count, method="ICA"):
    """
    method: PCA, ICA
    https://github.com/tortlab/Cell-Assembly-Detection/blob/master/assembly_patterns.m

    todo:


    """
    assert method in ["PCA", "ICA"], "Method not recognized"

    spike_count = np.asarray(spike_count)
    [n_cells, n_bins] = spike_count.shape

    # compute correlation matrix of binned spikes matrix
    spike_count_z = (
        spike_count - bn.nanmean(spike_count, axis=1)[:, None]
    ) / bn.nanstd(spike_count, ddof=1, axis=1)[:, None]

    correlation_matrix = (1 / (n_bins - 1)) * (spike_count_z @ spike_count_z.T)

    # compute eigenvalues/vectors and sort them in descending order
    eigvals, eigvecs = np.linalg.eig(correlation_matrix)
    i_sort = eigvals.argsort()
    eigvals = eigvals[i_sort][::-1]
    eigvecs = eigvecs[:, i_sort][::-1]

    # define significant assemblies as assemblie having a eigenvalue greater
    # than a threshold, lambda max, defined using Marchenko-Pastur law
    q = n_bins / n_cells
    assert q > 1, "Number or time bins should be greater than the number of neurons"
    lambda_max = (1 + np.sqrt(1 / q)) ** 2
    significant_vals = eigvals > lambda_max

    n_assemblies = np.sum(significant_vals)

    if method == "PCA":
        template = eigvecs[:, :n_assemblies]

    elif method == "ICA":
        ica = FastICA(n_components=n_assemblies)
        # ica take X with shape (n_samples, n_features)
        ica.fit_transform(spike_count_z.T)
        template = ica.mixing_
    else:
        raise NotImplementedError("Method not implented")

    return template, correlation_matrix


def calc_activity(spike_count, template, kernel_half_width=None):
    """
    see
    https://github.com/tortlab/Cell-Assembly-Detection/blob/master/assembly_activity.m
    see also fig S2 of van de ven 2016
    """

    spike_count = np.asarray(spike_count)
    [_, n_samples] = spike_count.shape

    template = np.asarray(template)
    [_, n_components] = template.shape

    # compute correlation matrix of binned spikes matrix
    spike_count_z = (
        spike_count - bn.nanmean(spike_count, axis=1)[:, None]
    ) / bn.nanstd(spike_count, ddof=1, axis=1)[:, None]

    activity = np.zeros((n_components, n_samples))
    for i, template_i in enumerate(template.T):
        # generate projector matrix
        projector = np.outer(template_i.T, template_i)
        # set diag to zeros so only co-activations of neurons
        # will contribute to the assembly pattern
        np.fill_diagonal(projector, 0)

        # calculate assembly pattern expression strength
        # as defined in Lopez-dos-Santos 2013 R(b)= Z(b).T * P * Z(b)
        activity[i, :] = bn.nansum(
            spike_count_z.T.dot(projector) * spike_count_z.T, axis=1
        )

    # here we could convolve the spike_count_z with a gaussian
    # cf Van de Ven 2016
    if kernel_half_width is not None:
        activity = smooth1D(
            activity, kernel_half_width=kernel_half_width, kernel_type="gauss"
        )

    return activity


def corr_mat(a, axis=1):
    """
    Compute correlation between all the rows (or collunm) of a given matrix

    Parameters:
            - Matrix for example [unit, samples]
            - axis on wich we want to work on
    Returns:
            - correlation matrix

    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    a = np.asarray(a)

    a_z = zscore(a, axis)
    n = a.shape[axis]

    if axis == 0:
        return (1 / (n - 1)) * (a_z.T @ a_z)
    else:
        return (1 / (n - 1)) * (a_z @ a_z.T)


def zscore(matrix, ax=1):
    """
    Compute zscores along one axis.
    """
    if ax == 1:
        z = (matrix - bn.nanmean(matrix, axis=ax)[:, None]) / bn.nanstd(
            matrix, axis=ax, ddof=1
        )[:, None]
    else:
        z = (matrix - bn.nanmean(matrix, axis=ax)[None, :]) / bn.nanstd(
            matrix, axis=ax, ddof=1
        )[None, :]
    return z


def sim_assemblies(
    n_neurons=20,
    n_bins=10000,
    neuron_assembly=[[1, 2, 3, 4], [3, 6, 7]],
    n_act=[300, 300],
    act_lambda=[3, 3],
):
    """
    Generate reactivation pattern to test other function form this module

    Parameters:
                - n_neurons: number of neurons

    """

    spikes_binned = np.random.poisson(1, (n_neurons, n_bins))

    for it, curr_neuron in enumerate(neuron_assembly):
        n_neu_ass = len(curr_neuron)
        r_rdx = np.random.random_integers(0, n_bins - 1, n_act[it])

        spikes_binned[np.array(curr_neuron)[:, None], r_rdx] = np.random.poisson(
            act_lambda[it], (n_neu_ass, n_act[it])
        )
    return spikes_binned