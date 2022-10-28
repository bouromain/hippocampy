import numpy as np
from scipy.fftpack import next_fast_len
from scipy.signal import butter, cheby2, filtfilt
from scipy.signal import decimate as _decimate, resample_poly
from scipy.signal.signaltools import hilbert
from hippocampy.utils.gen_utils import value_cross

########################################################################
## Down and resampling
########################################################################
def resample(
    sig: np.ndarray,
    fs: float,
    fs_up=None,
    fs_down=None,
    method: str = "decimate",
    axis=-1,
):
    """
    Resample signal avoiding aliasing. This method is particularly 
    usefull for noisy data. It should be preferred over taking every nth 
    amples of a signal as it can cause aliasing, artefact in the resulting
    downsampled signal

    Parameters
    ----------
    sig:
        signal to resample
    fs:
        sampling frequency or the input signal in Hz
    fs_up: 
        upsampling sampling frequency in Hz, only considered 
        for the poly method
    fs_down: 
        downsampling sampling frequency in Hz
    method: 
        decimate or poly
    axis:
        axis along which to downsample
    Returns
    -------
    sig_d
        downsampled signal

    
    Reference
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html#scipy.signal.decimate
    """
    # first check the downsampling factor, if it is an integer
    if not method in ["decimate", "poly"]:
        raise ValueError(f"method {method} not found, use 'poly\ or 'decimate' instead")

    if fs_up is None:
        fs_up = fs

    # check the down and up sampling factor are integers
    if not fs % fs_down == 0:
        raise ValueError(
            "Downsampling frequency should be a multiple of the original sampling rate"
        )

    if not fs % fs_up == 0:
        raise ValueError(
            "Upsampling frequency should be a multiple of the original sampling rate"
        )

    if method == "poly":
        q_up = fs_up // fs
        q_down = fs // fs_down
        sig_d = resample_poly(sig, q_up, q_down, axis=axis, padtype="line")

    elif method == "decimate":
        q = fs // fs_down
        # when decimating the downsampling factor should not be bigger than 13
        # otherwise the decimation should be done in steps
        if q > 12:
            q = _limit_q(q, max_mult=12)

        if isinstance(q, list):
            sig_d = sig
            for down_factor in q:
                sig_d = _decimate(sig_d, down_factor, axis=axis)
        else:
            sig_d = _decimate(sig, q, axis=axis)

    return sig_d


def _limit_q(q: int, max_mult: int = 12) -> list:
    """
    _limit_q decompose a number as a suite of multiple 
    smaller than a given number max_mult

    Parameters
    ----------
    q : int
        number to decompose
    max_mult : int, optional
        maximum divisor, by default 12

    Returns
    -------
    list
        a list of number smaller than max_mult decoposing q

    Raises
    ------
    ValueError
        if no divisor smaller than max_mult are found 

    Example
    -------
    >>> _limit_q(15000, 10)
    >>> [10, 10, 10, 5, 3]
    """
    bigger_mult = []
    bigger_L = []

    for l in np.arange(2, max_mult + 1)[::-1]:
        d, rem = np.divmod(q, l)
        if rem == 0:
            # if the remainder is zero and if the two numbers are smaller than
            # max_mult we can stop
            if d <= max_mult:
                return [l, d]
            if not bigger_mult:
                bigger_mult = d
                bigger_L = l
    # if we looped over all the number without going out of this function it
    # means that we either have a number too big or a prime number
    # print(f" mult {bigger_mult} and L {bigger_L}")
    if bigger_mult is []:
        raise ValueError(f"No Divisor of {q} smaller than {max_mult} found")
    #
    return [bigger_L] + _limit_q(bigger_mult, max_mult)


########################################################################
## Filtering and phase extraction
########################################################################


def band_filter(sig, fRange, fs, method="cheby2", order=4, axis=-1) -> np.ndarray:
    """
    Filter a signal in a certain frequency band and with a particular filter type
    
    Parameters
    ----------
    sig: 
        signal to filter
    fRange: 
        frequency band to filter with. 
        For example:
            Theta [5 12]
            Ripple []
    fs: 
        sampling frequency
    method: 
        filter method.
            - Chebyshev Type II (cheby2, default)
            - Butterworth (butter)
    order:
        order of the filter
    axis:
        axis along which the function is performed, by default -1

    Returns
    -------
    sig_f: array_like
        filtered signal

    TODO
    -----
    implement sosfilter
    implement remez method:
    https://github.com/Eden-Kramer-Lab/ripple_detection/blob/4c3ae1cdf421f38db1c4dcd67cdd967c63989d4a/ripple_detection/core.py#L95
    """

    allMethods = ["butter", "cheby2"]
    assert any(method == s for s in allMethods), "Invalid Method in bandpassSig"

    sig = np.asarray(sig)
    fRange = np.asarray(fRange)
    assert max(fRange.shape) == 2, "fRange should be given in the format [low, high]"

    nyquist = 0.5 * fs

    if method == "butter":
        b, a = butter(order, [fRange[0] / nyquist, fRange[1] / nyquist], btype="band")
    elif method == "cheby2":
        b, a = cheby2(
            order, 20, [fRange[0] / nyquist, fRange[1] / nyquist], btype="band"
        )
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    return filtfilt(b, a, sig, axis=axis)


def phase(sig, method="hilbert", axis=-1) -> np.ndarray:
    """
    Compute phase of a signal.

    Parameters
    ----------
    sig : [type]
        input values
    method : str, optional
        method to compute the phase, by default "hilbert"
            - hilbert: only use hilbert transform
            - peak: perform linear interpolation between peaks 
            - asymmetric: perform linear interpolation between peaks and through 
                    can be useful to preserve an asymmetric oscillation 
                    (eg: for theta oscillation )
    axis : int, optional
        axis along which the function is performed, by default -1

    Returns
    -------
    np.ndarray
        [description]
        

    TODO: implement shape preserving phase
    """

    if method not in ["hilbert", "peak", "asymmetric"]:
        raise ValueError("Method should be hilbert,peak or asymmetric")

    sig = np.array(sig, ndmin=2)

    n_samples = sig.shape[axis]
    bestLen = next_fast_len(n_samples)
    analytic_signal = hilbert(sig, bestLen)
    # remove padding
    analytic_signal = np.take(analytic_signal, np.arange(n_samples), axis=axis)
    sig_phase = np.mod(np.angle(analytic_signal), 2 * np.pi)
    sig_envelope = np.abs(analytic_signal)

    if method in ["peak", "asymmetric"]:

        up, down = value_cross(sig_phase, np.pi)
        phase_u = np.unwrap(sig_phase.squeeze())

        if method == "peak":
            up_down = down

        elif method == "asymmetric":
            up_down = np.logical_or(up, down)

        sig_phase = np.interp(
            np.arange(len(up_down)), np.nonzero(up_down)[0], phase_u[up_down]
        )
        sig_phase = np.mod(sig_phase, 2 * np.pi)

    return sig_phase.squeeze(), sig_envelope.squeeze()


def envelope(sig: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute phase and amplitude of a signal.

    Parameters
    ----------
    sig : np.ndarray
        input values
    axis : int, optional
        axis along which the function is performed, by default -1

    Returns
    -------
    np.ndarray
        [description]
    """
    sig = np.array(sig, ndmin=2)
    n_samples = sig.shape[axis]
    bestLen = next_fast_len(n_samples)

    analytic_signal = hilbert(sig, bestLen)
    # remove padding
    analytic_signal = np.take(analytic_signal, np.arange(n_samples), axis=axis)
    amplitude_envelope = np.abs(analytic_signal)

    return amplitude_envelope.squeeze()


def instantaneousFreq(sig_p: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute instantaneous frequency of a signal from its phase.
    This phase can be computed with hilbertPhase()

    Parameters
    ----------
    sig_p : np.ndarray
        input values
    fs : int
        sampling frequency of this signalS

    Returns
    -------
    np.ndarray
        vector of instantaneous frequency
    """

    sig_p_u = np.unwrap(sig_p)
    return np.diff(sig_p_u) / (2.0 * np.pi) * fs
