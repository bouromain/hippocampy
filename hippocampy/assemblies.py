"""
Various method to extra neuronal assemblies in the activity of a population of neurons.


References:
Lopes-dos-Santos V, Ribeiro S, Tort ABL Detecting cell assemblies 
in large neuronal populations,(2013) Journal of Neuroscience Methods.


TO DO:
https://elifesciences.org/articles/19428

"""

import numpy as np
import bottleneck as bn
from sklearn.decomposition import FastICA
from hippocampy.matrix_utils import smooth1D, corr_mat, zscore


def calc_template(spike_count, method="ICA"):
    """
    Calculate assemblies pattern as described in Lopes-dos-Santos V,
    Ribeiro S, Tort ABL Detecting cell assemblies in large neuronal populations,
    (2013) Journal of Neuroscience Methods.

    Parameters:
                - spike_count = matrix of size (n_cells,n_samples)
                - method = Method to extract pattern (PCA, ICA) ICA is better
                as it will take into account neurons that can be in multiple
                assemblies


    Return:
                - template: number of significant templates (n_cells,n_patterns)
                - correlation_matrix (n_cells,n_cells)

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
    Calculate the activity in time of given assemblies (templates)

    Parameters:
                - spike_count = matrix of size (n_cells,n_samples)
                - template: number of significant templates (n_cells,n_patterns)
                - kernel_half_width = size of the half window size of
                the gaussian kernel used to smooth the activity profile

    Returns:
                - activity = matrix of size (n_template,n_samples)

    For a nice visual explanation see also fig S2 of van de Ven et al 2016
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