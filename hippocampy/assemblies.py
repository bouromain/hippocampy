import numpy as np
import bottleneck as bn


def calc_template(a):
    """

    https://github.com/tortlab/Cell-Assembly-Detection/blob/master/assembly_patterns.m
    https://github.com/tortlab/Cell-Assembly-Detection/blob/master/assembly_activity.m

    todo:

    Check it and inplement ica cf sklearn fast ica

    """

    a = np.asarray(a)
    [n_cells, n_bins] = a.shape

    # compute correlation matrix of binned spikes matrix a
    c = corr_mat(a)
    # compute eigenvalue/vectors and sort them in descending order
    eigvals, eigvecs = np.linalg.eig(c)
    i_sort = eigvals.argsort()
    eigvals = eigvals[i_sort][::-1]
    eigvecs = eigvecs[:, i_sort][::-1]

    # calculate lambda max using Marchenko-Pastur

    q = n_cells / n_bins
    assert q < 1, "Number or time bins should be higher than the number of neurons"

    lambda_max = (1 + np.sqrt(1 / q)) ** 2
    significant_vals = eigvals > lambda_max

    n_assemblies = 4  # np.sum(significant_vals)
    eigvecs = eigvecs[:, n_assemblies - 1]

    template = np.zeros((n_cells, n_cells, n_assemblies))
    for i, t in enumerate(eigvecs.T):
        template[:, :, i] = np.outer(t.T, t)
        np.fill_diagonal(template[:, :, i], 0)

    return (template,)


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
