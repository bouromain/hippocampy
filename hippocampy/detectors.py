import numpy as np
import bottleneck as bn

from hippocampy.binning import mua
from hippocampy.matrix_utils import zscore
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
    perc_threshold: int = 99,
    n_shuffle=100,
):
    """
    sce _summary_

    Parameters
    ----------
    T : np.ndarray
        _description_
    fs : int, optional
        _description_, by default 1
    restrict : np.ndarray, optional
        _description_, by default None
    window_len : float, optional
        _description_, by default 0.2
    min_n_cells : int, optional
        _description_, by default 5
    perc_threshold : int, optional
        _description_, by default 3
    n_shuffle : int, optional
        _description_, by default 100


    Reference
    ---------
    [1] Malvache A, Reichinnek S, Villette V, Haimerl C, Cossart R. 
        Awake hippocampal reactivations project onto orthogonal neuronal 
        assemblies. Science. 2016 Sep 16;353(6305):1280-3. 
        doi: 10.1126/science.aaf3319
    """

    ## TO DO
    # check the IV from and to bool it seem wrong
    # incorporate the restrict to quiet period
    T = np.array(T, dtype=bool)
    window_len_samples = int(window_len * fs)
    n_cells, n_samples = T.shape

    T_shuff = np.zeros((n_cells, n_samples, n_shuffle), dtype=bool)
    for it_shuff in range(n_shuffle):
        for it_cell in range(n_cells):
            T_shuff[it_cell, :, it_shuff] = np.roll(
                T[it_cell, :], np.random.randint(0, n_samples)
            )

    T_sum = bn.move_sum(T, window_len_samples, axis=1)
    T_sum_shuff = bn.move_sum(T_shuff, window_len_samples, axis=1)  # , min_count=12

    # we could restrict the previous vectors here

    # now we could to a percentile of the shuffling
    avg = bn.nansum(T_sum, axis=0)
    avg_shuff = bn.nansum(T_sum_shuff, axis=0)

    # 3sd correspond to ~99 percentile
    T_thresh = np.nanpercentile(avg_shuff, perc_threshold, axis=1)
    # pure zeros can come from performing percentile
    # operation on nans only. Could be written in a better way
    T_thresh[T_thresh == 0] = np.nan

    # now SCE are defined as epoch of the summeed activity (in a temporal window)
    # higher than chance and with more than a certain number of cells
    cand_SCE = Iv().from_bool(avg > T_thresh)
    cand_SCE.merge(gap=intersample * fs)

    m = np.empty(len(cand_SCE))
    for it_c, c in cand_SCE:
        m[it_c] = bn.nanmax(avg[c.min : c.max])

    # only keep envent with a minimum of cells
    cand_SCE = cand_SCE[m > min_n_cells]

    return cand_SCE.to_bool()

