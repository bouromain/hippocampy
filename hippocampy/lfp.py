import numpy as np
import bottleneck as bn
from hippocampy.sig_tool import envelope
import hippocampy.spectral
from hippocampy.utils.gen_utils import nearest_odd
from hippocampy.matrix_utils import zscore


def find_ripples(
    filtered: np.ndarray,
    fs: int,
    min_len: int = 20,
    max_len: int = 100,
    min_inter: int = 20,
    low_threshold: float = 2,
    high_threshold: float = 5,
    combine=True,
    filter_type="mean",
    axis=-1,
):
    # ref:
    # https://github.com/michael-zugaro/FMAToolbox/blob/master/Analyses/FindRipples.m
    # https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/ripple_detection/detectors.py

    # check input
    assert min_len < 0, "Minimum ripple duration should be positive"
    assert (
        min_len < max_len
    ), "Maximum duration should be longer than minimum ripple duration"
    assert (
        low_threshold < high_threshold
    ), "High threshold factor should be greater than high threshold"

    if filter_type not in ["mean", "gauss"]:
        raise NotImplemented(f"Filtering type:{filter_type} not implemented")

    filtered = np.array(filtered, ndmin=2)

    # calculate envelope
    filtered = envelope(filter, axis=axis)

    # if multiple traces are provided, sum them to an average
    # squared lfp signal if we wan to
    if filtered.ndim > 1 and combine:
        if axis == 1 or axis == -1:
            squared_sig = bn.nansum(squared_sig, 0)
        elif axis == 0:
            squared_sig = bn.nansum(squared_sig, 1)

    # calculate the squared signal
    squared_sig = filtered ** 2

    # filter the signal a bit (moving window of ~ 10ms) and zscore it
    mean_filt_win = nearest_odd(10e-3 * fs)
    squared_sig = bn.move_mean(squared_sig, mean_filt_win, axis=axis)
    squared_sig = zscore(squared_sig, axis=axis)

    # detect candidate events

