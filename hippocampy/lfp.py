import numpy as np
import bottleneck as bn
from hippocampy.core.Iv import Iv
from hippocampy.sig_tool import envelope
from hippocampy.utils.gen_utils import nearest_odd, start_stop
from hippocampy.matrix_utils import smooth1D, zscore


def find_ripples(
    filtered: np.ndarray,
    fs: int,
    min_len: int = 20,
    max_len: int = 100,
    min_inter: int = 20,
    low_threshold: float = 2,
    high_threshold: float = 5,
    combine=True,
    smooth_type="mean",
    restrict: np.ndarray = None,
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

    if smooth_type not in ["mean", "gauss"]:
        raise NotImplemented(f"Smooth type:{smooth_type} not implemented")

    filtered = np.array(filtered, ndmin=2)

    # calculate envelope
    filtered = envelope(filtered, axis=axis)

    # if multiple traces are provided, sum them to an average
    # squared lfp signal if we want to
    if filtered.ndim > 1 and combine:
        if axis == 1 or axis == -1:
            squared_sig = bn.nansum(filtered, 0)
        elif axis == 0:
            squared_sig = bn.nansum(filtered, 1)

    # calculate the squared signal
    squared_sig = squared_sig ** 2

    # filter the signal a bit
    # (moving window of ~ 10ms) and zscore it
    filt_half_win = nearest_odd(5e-3 * fs)
    squared_sig = smooth1D(
        squared_sig, filt_half_win, kernel_type=smooth_type, axis=axis
    )

    squared_sig = zscore(squared_sig, axis=axis)

    # detect candidate events
    thresholded = squared_sig > low_threshold
    cand_event = Iv().from_bool(thresholded)

    n_sample_gap = min_inter * 10e-3 * fs
    n_sample_max_len = max_len * 10e-3 * fs

    cand_event.merge(gap=n_sample_gap, overlap=0.0, max_len=n_sample_max_len)

    # remove small intervals
    too_small = cand_event.stops - cand_event.starts > min_len * 10e-3 * fs
    cand_event = cand_event[~too_small]

    starts, stops = start_stop(thresholded)
    ripple_epoch = np.array([np.nonzero(starts)[0], np.nonzero(stops)[0]])

    # merge them if they are close but only if they do not create huge ripples
    inter_ripple_time = ripple_epoch[0, 1:] - ripple_epoch[1, :-1]
    iri_to_keep = inter_ripple_time > int(min_len * 10e-3 * fs)

