"""
Various method to extra neuronal assemblies in the activity of 
a population of neurons.


References
----------

[1] Peyrache A, Benchenane K, Khamassi M, Wiener SI, Battaglia FP. Principal
    component analysis of ensemble recordings reveals cell assemblies at high
    temporal resolution. J Comput Neurosci. 2010 Aug;29(1-2):309-325. 
    doi: 10.1007/s10827-009-0154-6

[2] Lopes-dos-Santos V, Conde-Ocazionez S, Nicolelis MA, Ribeiro ST, 
    Tort AB. Neuronal assembly detection and cell membership specification 
    by principal component analysis. PLoS One. 2011;6(6):e20996. 
    doi: 10.1371/journal.pone.0020996. 

[3] An integrated calcium imaging processing toolbox for the analysis of neuronal 
    population dynamics. Sebastián A. Romano, Verónica Pérez-Schuster, 
    Adrien Jouary, Jonathan Boulanger-Weill, Alessia Candeo, Thomas Pietri, 
    Germán Sumbre
    Plos Comp Biol 2017, https://doi.org/10.1371/journal.pcbi.1005526

[4] Tracy CA, Widom H. Level-Spacing Distributions and the Airy Kernel. 
    Commun Math Phys. 1992;159: 35. https://doi.org/10.1016/0370-2693(93)91114-3

[5] van de Ven GM, Trouche S, McNamara CG, Allen K, Dupret D. Hippocampal 
    Offline Reactivation Consolidates Recently Formed Cell Assembly Patterns 
    during Sharp Wave-Ripples. Neuron. 2016 Dec 7;92(5):968-974. 
    doi: 10.1016/j.neuron.2016.10.020. 

TODO
-----
see if we can speed up the cross correlation matrices by transforming binned 
spike/transient matrices to sparse ones

https://elifesciences.org/articles/19428 

Latent Ensemble Recruitment from Sparks, Liao, et al., NatComms (2020)
https://github.com/losonczylab/Sparks_Liao_NatComms2020

"""

import bottleneck as bn
import numpy as np
from sklearn.decomposition import FastICA
from hippocampy.matrix_utils import smooth_1d, zscore
from hippocampy.utils.nan import remove_nan


def calc_template(
    spike_count: np.ndarray, method: str = "ICA", correction: bool = False
):
    """
    Calculate assemblies pattern as described in ref [1-2]

    Parameters
    ----------
    spike_count
        matrix of size (n_cells,n_samples)
    method
        method to extract pattern (PCA, ICA) ICA is better
        as it will take into account neurons that can be in multiple
        assemblies
    correction: bool
        Tracy-Widom correction as described in Ref [3,4]

    Return
    ------
    template
        number of significant templates (n_cells,n_patterns)
    correlation_matrix
        (n_cells,n_cells)
    

    Nota bene
    ---------
    -   Some paper threshold the template vector to find the cells of an assembly
        for example van de Ven et al 2016 defines it as 2 std above mean (see fig S2)
    -   I should may be have a look at the advantage of the promax approach used 
        in some paper

    """
    assert method in ["PCA", "ICA"], "Method not recognized"

    spike_count = np.asarray(spike_count)
    [n_cells, n_bins] = spike_count.shape

    # compute correlation matrix of binned spikes matrix
    spike_count_z = zscore(spike_count, axis=1)

    # remove nan values to avoid the correlation matrix to be only nans
    spike_count_z = remove_nan(spike_count_z)

    correlation_matrix = (1 / (n_bins - 1)) * (spike_count_z @ spike_count_z.T)

    # compute eigenvalues/vectors and sort them in descending order
    eigvals, eigvecs = np.linalg.eig(correlation_matrix)
    i_sort = eigvals.argsort()
    eigvals = eigvals[i_sort][::-1]
    eigvecs = eigvecs[:, i_sort][::-1]

    # define significant assemblies as an assembly having an eigenvalue greater
    # than a threshold, lambda max, defined using Marchenko-Pastur law
    q = n_bins / n_cells
    assert q > 1, "Number or time bins should be greater than the number of neurons"

    if correction:
        # Tracy-Widom correction
        lambda_max = ((1 + np.sqrt(1 / q)) ** 2) + n_cells ** (2 / 3)
    else:
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
        raise NotImplementedError("Method not implemented")

    return template, correlation_matrix


def calc_activity(spike_count, template, kernel_half_width=None):
    """
    Calculate the activity in time of given assemblies (templates)

    Parameters
    ----------
    spike_count
        matrix of size (n_cells,n_samples)
    template
        number of significant templates (n_cells,n_patterns)
    kernel_half_width
        size of the half window size of the gaussian kernel
        used to smooth the activity profile

    Returns
    -------
    activity
        matrix of size (n_template,n_samples)

    Note
    ----
    For a nice visual explanation see also fig S2 of ref [5]
    """

    spike_count = np.asarray(spike_count)
    [_, n_samples] = spike_count.shape

    template = np.asarray(template)
    [_, n_components] = template.shape

    # here we could convolve the spike_count_z with a gaussian
    # cf ref [5]
    if kernel_half_width is not None:
        spike_count = smooth_1d(
            spike_count, kernel_half_width=kernel_half_width, kernel_type="gauss"
        )

    # compute correlation matrix of binned spikes matrix
    spike_count_z = zscore(spike_count, axis=1)

    activity = np.zeros((n_components, n_samples))
    for i, template_i in enumerate(template.T):
        # generate projector matrix
        projector = np.outer(template_i.T, template_i)
        # set diag to zeros so only co-activations of neurons
        # will contribute to the assembly pattern
        np.fill_diagonal(projector, 0)

        # calculate assembly pattern expression strength
        # as defined in ref [2]:
        #               R(b)= Z(b).T * P * Z(b)
        activity[i, :] = bn.nansum(
            spike_count_z.T.dot(projector) * spike_count_z.T, axis=1
        )

    return activity


def cells_in_template(template: np.ndarray, n_std: float = 2.0):
    """
    cells_in_template 
    Find cells that are belonging to each template

    It seem to be done is various papers. Here I used the threshold from 
    Van de Veen paper.

    Parameters
    ----------
    template : np.ndarray
        template assemblies vestors. Output of calc_template
    n_std : float, optional
        number of std above the mean need to be crossed to be part of the 
        template assembly, by default 2.0

    Returns
    -------
    output: np.ndarray
        logical matrix telling if a cell is part of a template
    """
    template_fixed = template.copy()

    max = bn.nanmax(template, axis=0)
    min = bn.nanmin(template, axis=0)

    to_flip = np.abs(min) > max

    template_fixed[:, to_flip] = -template[:, to_flip]

    m = bn.nanmean(template_fixed, axis=0)
    s = bn.nanstd(template_fixed, axis=0)

    return template_fixed - m[None, :] > n_std * s[None, :]


def sim_assemblies(
    n_neurons=20,
    n_bins=10000,
    neuron_assembly=[[1, 2, 3, 4], [3, 6, 7]],
    n_act=[300, 300],
    act_lambda=[3, 3],
):
    """
    Generate reactivation pattern to test other function
    form this module

    Parameters
    ----------
    n_neurons: int
        number of neurons
    n_bins: int
        number of sample bins
    neuron_assembly list of list of int [[list1] , [list2],...]
        definition of neurons in the different assemblies.
    n_act: list of int
        number of reactivation per assemblies
    act_lambda: list of int/float
        value of the lambda during the reactivation, lambda is
        hardcoded to one outside of the reactivation

    Returns
    -------
    spikes_binned: np.array
        simulated spike binned matrix
    """

    spikes_binned = np.random.poisson(1, (n_neurons, n_bins))

    for it, curr_neuron in enumerate(neuron_assembly):
        n_neu_ass = len(curr_neuron)
        r_rdx = np.random.randint(0, n_bins - 1, n_act[it])

        spikes_binned[np.array(curr_neuron)[:, None], r_rdx] = np.random.poisson(
            act_lambda[it], (n_neu_ass, n_act[it])
        )
    return spikes_binned
