import numpy as np
import bottleneck as bn

from hippocampy.binning import mua
from hippocampy.matrix_utils import remove_small_objects, zscore
from hippocampy.core.Iv import Iv


def mua_event(
    mat: np.ndarray,
    fs: int = 1,
    axis: int = -1,
    threshold_low: float = 1,
    threshold_high: float = 3.5,
    sample_gap: float = 0.2,
    min_len: int = 2,
    smooth_first: bool = True,
    kernel_half_width: float = 0.120,
):
    """
    mua_event identify putative Population Synchronous Events as defined in [1]
    (and other papers).
    A PSE is defined as moments during which the zscored mua activity exceeds
    3.5 SD with borders of the event edges extended to the location where the
    signal reaches 1SD. These PSE should  have a inter-event interval of at least
    0.2 s.

    Parameters
    ----------
    mat : np.nd_array
        input matrix, can be binary matrix of transient or spikes (True), or
        rate vectors
    fs : int
        sampling frequency of the input matrix
    axis : int, optional
        axis along which the function is performed, by default -1
    threshold_low : int, optional
        low threshold where the edges will be extended, by default 1
    threshold_high : int, optional
        high threshold should be exceeded during the SCE to be kept ,
        by default 3.5
    sample_gap : float
        minimum inter-event interval in second for the SCE to be kept
    smooth_first : bool, optional
        _description_, by default True
    kernel_half_width : int, optional
        time of the smoothing time window (in s), by default 10


    Reference
    ---------
    [1] Grosmark, A.D., Sparks, F.T., Davis, M.J. et al. Reactivation predicts
        the consolidation of unbiased long-term cognitive maps. Nat Neurosci 24,
        1574–1585 (2021). https://doi.org/10.1038/s41593-021-00920-7

    """

    smooth_kern = np.floor(fs * (kernel_half_width))
    mua_act = mua(
        mat, axis=axis, smooth_first=smooth_first, kernel_half_width=smooth_kern
    )

    # zscore mua activity
    mua_z = zscore(mua_act)

    # detect candidates above 1 std
    mua_mean = mua_z > threshold_low
    cand_SCE = Iv().from_bool(mua_mean)

    # merge event close in time
    n_sample_gap = sample_gap * fs
    cand_SCE.merge(gap=n_sample_gap, overlap=0.0)

    # remove very small event
    n_min_samples = min_len * fs
    mask_small = cand_SCE.stops - cand_SCE.starts > n_min_samples
    cand_SCE = cand_SCE[mask_small]

    # keep only event with peaks above 3.5
    mua_peaks = mua_z > threshold_high
    high_event = Iv().from_bool(mua_peaks)
    mask = cand_SCE.contain(high_event)

    cand_SCE = cand_SCE[mask]

    # make the boolean vector and store peak location
    peak_times = np.empty(len(cand_SCE))
    cand_SCE_mask = np.zeros_like(mua_z).astype(bool)

    for i, it_c in enumerate(cand_SCE):
        cand_SCE_mask[it_c.min : it_c.max] = True
        peak_times[i] = it_c.starts + bn.nanargmax(mua_z[it_c.min : it_c.max])

    return cand_SCE, cand_SCE_mask, peak_times


def sce(
    T: np.ndarray,
    fs: int = 1,
    restrict: np.ndarray = None,
    window_len: float = 0.2,
    min_n_cells: int = 5,
    intersample: float = 0.2,
    perc_threshold: int = 95,
    n_shuffle: int = 150,
    max_shuff: int = 5,
):
    """
    Detect Synchronous Calcium Events inspired form publication in ref [1-2]. 
    It will look at the number of cell active in a given sliding time widow and 
    detect synchronous event where more cells are active than in a shuffled 
    distribution.

    Parameters
    ----------
    T : np.ndarray
        Matrix of calcium events detected with the function (eg: detected with 
        calfunc.transient), [n_cells, n_samples]
    fs : int, optional
        sampling frequency in Hz, by default 1
    restrict : np.ndarray, optional
        vector to restrict the activity to particular epoch (eg: quiet moment),
        it is a convenience input and can be omitted if the data are masked 
        before by default None
    window_len : float, optional
        length of the sliding sum window (in sec) , by default 0.2
    min_n_cells : int, optional
        minimum number of cells that should be active in a SCE to be kept, 
        by default 5
    perc_threshold : int, optional
        percentile of the shuffled distribution that should be exceeded to be 
        considered as a candidate SCE, by default 95
    n_shuffle : int, optional
        number of shuffle  to perform, by default 100
    max_shuff : int, optional
        the max allowed (circular) shift allowed, in second,  when performing 
        the shuffling. Setting it to a seconds/ tens of second allow to adapt 
        the threshold in time to potential changes in basal activity. 
        If None the shift will be performed on all the data (max = n_samples)
    
    Return
    ------
    SCE: Iv,
        Interval array of detected SCE. It can be converted to a boolean vector 
        with IV().to_bool()

    Reference
    ---------
    [1] Malvache A, Reichinnek S, Villette V, Haimerl C, Cossart R. 
        Awake hippocampal reactivations project onto orthogonal neuronal 
        assemblies. Science. 2016 Sep 16 
        doi: 10.1126/science.aaf3319

    [2] Modol, L., Bollmann, Y., Tressard, T., Baude, A., Che, A., 
        Duan, Z., Babij, R., De Marco García, N. V., & Cossart, R. (2020). 
        Assemblies of Perisomatic GABAergic Neurons in the Developing Barrel 
        Cortex. Neuron, https://doi.org/10.1016/j.neuron.2019.10.007
    """

    # check inputs
    T = np.array(T)
    window_len_samples = int(window_len * fs)
    n_cells, n_samples = T.shape

    if max_shuff is None:
        max_shuff_sample = n_samples

    max_shuff_sample = int(max_shuff * fs)

    if max_shuff_sample > n_samples - 1:
        max_shuff_sample = n_samples - 1

    T_shuff = np.empty((n_cells, n_samples, n_shuffle))

    for it_shuff in range(n_shuffle):
        for it_cell in range(n_cells):
            T_shuff[it_cell, :, it_shuff] = np.roll(
                T[it_cell, :], np.random.randint(0, max_shuff)
            )

    # we should may be check the size of the restrict vector
    if restrict.dtype.kind != "b":
        raise ValueError("Restrict vector should be boolean")

    if restrict is None:
        restrict = np.ones((n_samples), dtype=bool)

    n_samples_restricted = bn.nansum(restrict)

    if max_shuff is None:
        max_shuff_sample = n_samples_restricted

    max_shuff_sample = int(max_shuff * fs)

    if max_shuff_sample > n_samples_restricted - 1:
        max_shuff_sample = n_samples_restricted - 1

    T_sum = bn.move_sum(T[:, restrict], window_len_samples, axis=1)
    # here move_sum creates a positive shift on the sum created. When it calculates
    # the value it considers the window starting at the current index i and
    # end at i+n while what we want is to make it start at i-(n/2) and end at i+(n/2)
    T_sum = np.roll(T_sum, int(-np.floor(window_len_samples / 2)), axis=1)
    # we can roll the array here because it will start with window_len_samples
    # element being nans. It will put them at the end

    avg = bn.nansum(T_sum, axis=0)
    avg_shuff = np.empty((n_shuffle, n_samples_restricted))

    for it_shuff in range(n_shuffle):
        tmp = np.zeros((n_cells, n_samples_restricted))
        for it_cell in range(n_cells):
            tmp[it_cell, :] = np.roll(
                T_sum[it_cell, :],
                np.random.randint(-max_shuff_sample, max_shuff_sample),
            )
        avg_shuff[it_shuff, :] = bn.nansum(tmp, axis=0)

    # calculate the threshold from the shuffled distribution
    T_thresh = np.ones(n_samples) * np.nan
    T_thresh[restrict] = np.nanpercentile(avg_shuff, perc_threshold, axis=0)

    # now SCE are defined as epoch of the summed activity (in a temporal window)
    # higher than chance and with more than a certain number of cells
    avg_full = np.zeros(n_samples)
    avg_full[restrict] = avg

    cand = avg_full > T_thresh
    cand_SCE = Iv().from_bool(cand)

    if len(cand_SCE) < 1:
        # in case there is no SCE detected
        return None
    cand_SCE.merge(gap=intersample * fs)

    # remove candidate that are too short
    m = cand_SCE.stops - cand_SCE.starts > 2
    cand_SCE = cand_SCE[m]

    # extract the number of cells active in the candidate SCE
    m = np.empty(len(cand_SCE))
    peak_times = np.empty(len(cand_SCE))

    for it_c, c in enumerate(cand_SCE):
        m[it_c] = bn.nanmax(avg_full[c.min : c.max])
        peak_times[it_c] = c.starts + bn.nanargmax(avg_full[c.min : c.max])

    # only keep event with a minimum of cells
    cand_SCE = cand_SCE[m > min_n_cells]
    peak_times = peak_times[m > min_n_cells]

    return cand_SCE, peak_times
