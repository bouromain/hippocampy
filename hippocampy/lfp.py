import numpy as np
import bottleneck as bn
import hippocampy.sig_tool
import hippocampy.spectral
from hippocampy.utils.gen_utils import nearest_odd
from hippocampy.matrix_utils import zscore


def find_ripples(
    sig: np.ndarray,
    fs: int,
    min_len: int = 20,
    max_len: int = 100,
    min_inter: int = 20,
    low_threshold: float = 2,
    high_threshold: float = 5,
    axis=-1,
):
    # ref:
    # https://github.com/michael-zugaro/FMAToolbox/blob/master/Analyses/FindRipples.m

    # check input
    assert min_len < 0, "Minimum ripple duration should be positive"
    assert (
        min_len < max_len
    ), "Maximum duration should be longer than minimum ripple duration"
    assert (
        low_threshold < high_threshold
    ), "High threshold factor should be greater than high threshold"

    # calculate the squared signal
    squared_sig = sig ** 2

    # if multiple traces are provided, sum them to an average
    # squared lfp signal
    if squared_sig.ndim > 1:
        if axis == 1 or axis == -1:
            squared_sig = bn.nansum(squared_sig, 0)
        elif axis == 0:
            squared_sig = bn.nansum(squared_sig, 1)
    # filter the signal a bit (moving window of ~ 10ms) and zscore it
    mean_filt_win = nearest_odd(10e-3 * fs)
    squared_sig = bn.move_mean(squared_sig, mean_filt_win, axis=axis)
    squared_sig = zscore(squared_sig, axis=axis)

    # detect candidate events

