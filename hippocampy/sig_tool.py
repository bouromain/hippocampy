import numpy as np
import bottleneck as bn
from scipy.fftpack import next_fast_len
from scipy.signal import butter, cheby2, filtfilt, sosfiltfilt
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
    if method not in ["decimate", "poly"]:
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

    for L in np.arange(2, max_mult + 1)[::-1]:
        d, rem = np.divmod(q, L)
        if rem == 0:
            # if the remainder is zero and if the two numbers are smaller than
            # max_mult we can stop
            if d <= max_mult:
                return [L, d]
            if not bigger_mult:
                bigger_mult = d
                bigger_L = L
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


def band_filter(
    sig: np.ndarray,
    fRange: np.ndarray,
    fs: int,
    method: str = "cheby2",
    order: int = 4,
    output: str = "sos",
    axis=-1,
) -> np.ndarray:
    """
    Filter a signal in a certain frequency band and with a particular filter type

    Parameters
    ----------
    sig:
        signal to filter
    fRange:
        frequency band to filter with.
        For example:
            Theta [5, 12]
            Ripple [100,225]
    fs:
        sampling frequency
    method:
        filter method.
            - Chebyshev Type II (cheby2, default)
            - Butterworth (butter)
    order:
        order of the filter
    output{"ba", "sos"}, optional
        Type of output: numerator/denominator ("ba") or second-order sections ("sos").
        Default"sos" should be used for general-purpose filtering "ba" for backwards
        compatibility.
    axis:
        axis along which the function is performed, by default -1

    Returns
    -------
    sig_f: array_like
        filtered signal

    TODO
    -----
    implement remez method:
    https://github.com/Eden-Kramer-Lab/ripple_detection/blob/4c3ae1cdf421f38db1c4dcd67cdd967c63989d4a/ripple_detection/core.py#L95
    """

    allMethods = ["butter", "cheby2"]
    assert any(method == s for s in allMethods), f"Method {method} not recognized"

    all_output = ["ba", "sos"]
    if output not in all_output:
        raise ValueError(f"Output format should either be {all_output}")

    sig = np.asarray(sig)
    fRange = np.asarray(fRange)
    assert max(fRange.shape) == 2, "fRange should be given in the format [low, high]"

    nyquist = 0.5 * fs

    if method == "butter":
        filter = butter(
            order,
            [fRange[0] / nyquist, fRange[1] / nyquist],
            btype="band",
            output=output,
        )
    elif method == "cheby2":
        filter = cheby2(
            order,
            20,
            [fRange[0] / nyquist, fRange[1] / nyquist],
            btype="band",
            output=output,
        )
    if output == "ba":
        return filtfilt(filter[0], filter[1], sig, axis=axis)
    elif output == "sos":
        return sosfiltfilt(filter, sig, axis=axis)


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


def xcorr(x: np.ndarray, y: np.ndarray = None, scale: str = None, maxlag=None):
    """
    Calculate the corelation between two sequences. It is a wrapper of
    np.correlate and implement various corrections and a maximum lag.

    TODO
    properly deal with vector of different length
    for now this function will return an error if x.shape != y.shape

    Parameters
    ----------
    x : np.ndarray
        vector with the first input sequence
    y : np.ndarray, optional
        vector with the second input sequence, if None, y is set to be x and this
        function will thus return the autocorrelation of the signal.
    scale : str, optional
        'biased' - scales the raw cross-correlation by 1/M.
        'unbiased' - scales the raw correlation by 1/(M-abs(lags)).
        'normalized' or 'coeff' - normalizes the sequence so that the
                                auto-correlations at zero lag are
                                identically 1.0.
        'none' - no scaling (this is the default).

    maxlag : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    assert scale in ["biased", "unbiased", "coeff", None]

    if y is None:
        y = x
    else:
        assert np.equal(x.shape, y.shape), "Input x and y should have the same lenght"
    m = x.shape[0]
    c = np.correlate(x, y, mode="full")

    if maxlag is not None:
        if maxlag < 1 or maxlag > m:
            raise ValueError(f"maxlags must be None or strictly positive < {m}")

        half = int(np.floor(c.shape[0] / 2) + 1)
        c = c[half - (maxlag + 1) : half + maxlag]

    if scale == "biased":
        c = c / m
    elif scale == "unbiased":
        L = (c.shape[0] - 1) / 2
        scale_unbiased = m - np.abs(np.arange(-L, L + 1))
        scale_unbiased[scale_unbiased <= 0] = 1
        c = c / scale_unbiased
    elif scale == "coeff":
        cxx0 = bn.nansum(np.abs(x) ** 2)
        cyy0 = bn.nansum(np.abs(y) ** 2)
        c = c / np.sqrt(cxx0 * cyy0)

    return c
