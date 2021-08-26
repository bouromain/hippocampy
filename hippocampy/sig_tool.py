import numpy as np
from scipy.fftpack import next_fast_len
from scipy.signal import butter, cheby2, filtfilt
from scipy.signal import decimate as _decimate, resample_poly
from scipy.signal.signaltools import hilbert

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
    # first check the down sampling factor, if it is an integer
    if not method in ["decimate", "poly"]:
        raise ValueError(f"method {method} not found, use 'poly\ or 'decimate' instead")

    if method == "poly" and fs_up is None:
        fs_up = fs

    # check the down and up sampling factor are integers
    if not fs % fs_down != 0:
        raise ValueError(
            "Downsampling frequency should be a multiple of the original sampling rate"
        )

    if not fs % fs_up != 0:
        raise ValueError(
            "Upsampling frequency should be a multiple of the original sampling rate"
        )

    if method == "poly":
        q_up = fs_up // fs
        q_down = fs_down // fs
        [sig_d] = resample_poly(sig, q_up, q_down, axis=axis, padtype="line")

    elif method == "decimate":
        q = fs_down // fs
        # when decimating the downsampling factor should not be bigger than 13
        # otherwise the decimation should be done in steps
        if q > 12:
            q = _limit_q(q, max_mult=12)

        sig_d = sig
        for down_factor in q:
            [sig_d] = _decimate(sig_d, down_factor, axis=axis)

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


def band_filter(sig, fRange, fs, method="cheby2", order=4) -> np.ndarray:
    """
    Filter a signal in a certain frequency band and with a particular filter type
    
    Parameters
    ----------
    sig: 
        signal to filter
    fRange: 
        frequency band to filter with eg [5 12]
    fs: 
        sampling frequency
    method: 
        filter method.
            - Chebyshev Type II (cheby2, default)
            - Butterworth (butter)
    order:
        order of the filter
    
    Returns
    -------
    sig_f: array_like
        filtered signal

    TO DO
    -----
    implement sosfilter
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
        raise NotImplementedError

    return filtfilt(b, a, sig)


def hilbertPhase(sig, method="hilbert") -> np.ndarray:
    """
    Compute hilbert phase and amplitude of a signal.

    TO DO: implement linear interpolation and shape preserving phase
    """

    if method == "hilbert":
        n_sig = len(sig)
        bestLen = next_fast_len(n_sig)
        analytic_signal = hilbert(sig, bestLen)
        # remove padding
        analytic_signal = np.mod(
            np.angle(analytic_signal[:n_sig]) + 2 * np.pi, 2 * np.pi
        )
        amplitude_envelope = np.abs(analytic_signal)
    else:
        raise NotImplementedError("phase method not implemented")

    return analytic_signal, amplitude_envelope


def instantaneousFreq(sig_p: np.ndarray, fs) -> np.ndarray:
    """
    Return instantaneous frequency of a signal from its phase.
    This phase can be computed with hilbertPhase()
    """

    sig_p_u = np.unwrap(sig_p)
    return np.diff(sig_p_u) / (2.0 * np.pi) * fs
