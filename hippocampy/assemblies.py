import numpy as np
import bottleneck as bn
from sklearn.decomposition import FastICA


"""
See also
https://elifesciences.org/articles/19428

"""


def calc_template(spike_count, method="ICA"):
    """
    method: PCA, ICA
    https://github.com/tortlab/Cell-Assembly-Detection/blob/master/assembly_patterns.m
    https://github.com/tortlab/Cell-Assembly-Detection/blob/master/assembly_activity.m

    todo:

    Check it and implement ica cf sklearn fast ica

    """
    assert method in ["PCA", "ICA"], "Method not recognized"

    spike_count = np.asarray(spike_count)
    [n_cells, n_bins] = spike_count.shape

    # compute correlation matrix of binned spikes matrix
    spike_count_z = (spike_count - bn.nanmean(spike_count)[:, None]) / bn.nanstd(
        spike_count, ddof=1
    )[:, None]

    correlation_matrix = (1 / (n_bins - 1)) * (spike_count_z @ spike_count_z.T)

    # compute eigenvalues/vectors and sort them in descending order
    eigvals, eigvecs = np.linalg.eig(correlation_matrix)
    i_sort = eigvals.argsort()
    eigvals = eigvals[i_sort][::-1]
    eigvecs = eigvecs[:, i_sort][::-1]

    # calculate lambda max using Marchenko-Pastur
    q = n_cells / n_bins
    assert q < 1, "Number or time bins should be greater than the number of neurons"
    lambda_max = (1 + np.sqrt(1 / q)) ** 2
    significant_vals = eigvals > lambda_max

    n_assemblies = np.sum(significant_vals)

    if method == "PCA":
        template = eigvecs[:, n_assemblies - 1]

    elif method == "ICA":
        ica = FastICA(n_components=n_assemblies)
        S = ica.fit_transform(spike_count_z)
        template = ica.mixing_
    else:
        raise NotImplementedError("Method not implented")

    return template, correlation_matrix


def calc_activity(spike_count, template,kernsize):
    """
    see
    https://github.com/tortlab/Cell-Assembly-Detection/blob/master/assembly_activity.m
    see also fig S2 of van de ven 2016
    """

    spike_count = np.asarray(spike_count)
    [n_cells, n_samples] = spike_count.shape

    template = np.asarray(template)
    [n_cells, n_components] = template.shape

    # compute correlation matrix of binned spikes matrix
    spike_count_z = (spike_count - bn.nanmean(spike_count)[:, None]) / bn.nanstd(
        spike_count, ddof=1
    )[:, None]

    
    activity = np.zeros(n_components,n_samples)
    for i, t in enumerate(eigvecs.T):

        # generate projector matrix
        projector = np.outer(template.T, template)
        # set diag to zeros so only co-activations of neurons 
        # will contribute to the assembly pattern
        np.fill_diagonal(projector, 0)

        for t_bin in spike_count_z:


    # calculate assembly pattern expression strength
    # as defined in Lopez-dos-Santos 2013 R(b)= Z(b).T * P * Z(b)

    # here we could convolve the spike_count_z with a gaussian
    # cf Van de Ven 2016


    return time_projection


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
