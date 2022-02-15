import numpy as np
import bottleneck as bn
from hippocampy.core.Iv import Iv
from hippocampy.sig_tool import envelope
from hippocampy.utils.gen_utils import nearest_odd, start_stop
from hippocampy.matrix_utils import smooth_1d, zscore


def find_ripples(
    filtered: np.ndarray,
    fs: int,
    *,
    min_len: int = 20,
    max_len: int = 200,
    min_inter: int = 20,
    low_threshold: float = 2,
    high_threshold: float = 5,
    combine=True,
    smooth_type="box",
    restrict: np.ndarray = None,
    axis=-1,
):
    """
    Ripple detection on a filtered signal. This code is inspired from existing 
    ones from Frank and Zugaro Lab (see Ref.)
    The detection is performed on the Normalized Squared Signal. A lower 
    threshold is used to define the start and end of the events. Then a cleaning
    of the candidate event is performed to remove too small, too big events. 
    Event containing a peak value smaller than a high threshold will also be 
    removed

    Parameters
    ----------
    filtered : np.ndarray
        Filtered local field potential [150,250]
        using signal.band_filter for example
    fs : int
        sampling frequency
    min_len : int, optional
        minimum length of a candidate ripple event (ms), by default 20
    max_len : int, optional
        maximum length of a candidate ripple event (ms), by default 200
    min_inter : int, optional
        minimum inter event time between two successive candidate ripple event (ms). 
        If two events are closer , they will be merged, unless this merge make 
        it exceed the max length specified above, by default 20
    low_threshold : float, optional
        Threshold for the detection of the begining and end of a candidate event
        (in multiple of std of the signal), by default 2 std
    high_threshold : float, optional
        Threshold for the detection of the peak inside an even (in multiple of 
        std of the signal), by default 5 std
    combine : bool, optional
        in case multiple channel are given as inputs, perform the mean lfp of the 
        signal, by default True
    smooth_type : str, optional
        type of smoothing to perform on the Normalised Squared Signal, by default "box"
    restrict : np.ndarray, optional
        Boolean vector of the same size than the filtered input. Mask moment where we 
        should mot detect anything (activity periods,...), by default None
    axis : int, optional
        axis along which to perform the computation, by default -1

    Returns
    -------
    cand_event: Iv
        Interval object containing the start index and stop index of the detected 
        events. 
    
    peak_times: np.array
        array containing the peak index of each ripple

    References
    ----------
    https://github.com/michael-zugaro/FMAToolbox/blob/master/Analyses/FindRipples.m
    https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/ripple_detection/detectors.py

    """
    # check input
    assert min_len > 0, "Minimum ripple duration should be positive"
    assert (
        min_len < max_len
    ), "Maximum duration should be longer than minimum ripple duration"
    assert (
        low_threshold < high_threshold
    ), "High threshold factor should be greater than high threshold"

    if smooth_type not in ["box", "gauss"]:
        raise NotImplemented(f"Smooth type:{smooth_type} not implemented")

    if type(fs) != int:
        raise ValueError(f"Sampling frequency Fs should be and scalar and not {fs}")

    filtered = np.array(filtered, ndmin=2)

    # calculate envelope
    filtered = envelope(filtered, axis=axis)

    # if multiple traces are provided, sum them to an average
    # squared lfp signal if we want to
    if filtered.ndim > 1 and combine:
        squared_sig = bn.nansum(filtered, axis=axis)
    else:
        squared_sig = filtered

    # calculate the squared signal
    squared_sig = squared_sig ** 2

    # filter the signal a bit
    # (moving window of ~ 10ms) and zscore it
    filt_half_win = nearest_odd(5e-3 * fs)

    squared_sig = smooth_1d(
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

    # remove event with values above the high-threshold
    thresholded_high = squared_sig > high_threshold
    high_event = Iv().from_bool(thresholded_high)
    mask = cand_event.contain(high_event)
    cand_event = cand_event[mask]

    # if we sk to restrict to some particular times
    if restrict is not None:
        Iv_to_mask = Iv().from_bool(restrict)
        mask = cand_event.contain(Iv_to_mask)
        cand_event = cand_event[mask]

    # now also search for peaks location
    peak_times = np.empty(len(cand_event))

    for i, it_event in enumerate(cand_event):
        peak_times[i] = it_event.starts + bn.nanargmax(
            squared_sig[it_event.min : it_event.max]
        )

    return cand_event, peak_times

