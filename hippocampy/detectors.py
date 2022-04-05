import numpy as np
import bottleneck as bn

from hippocampy.binning import mua
from hippocampy.matrix_utils import zscore
from hippocampy.core.Iv import Iv


def mua_event(
    mat: np.nd_array,
    fs: int = 1,
    axis: int = -1,
    threshold_low: float = 1,
    threshold_high: float = 3.5,
    sample_gap: float = 0.2,
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

    # keep only event with peaks above 3.5
    mua_peaks = mua_z > threshold_high
    high_event = Iv().from_bool(mua_peaks)
    mask = cand_SCE.contain(high_event)

    cand_SCE = cand_SCE[mask]

    cand_SCE_mask = np.zeros_like(mua_z).astype(bool)
    for it_c in cand_SCE:
        cand_SCE_mask[it_c.min : it_c.max] = True

    return cand_SCE

