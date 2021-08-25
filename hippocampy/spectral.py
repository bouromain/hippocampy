import numpy as np
from scipy.signal import spectrogram as sp_spectrogram
from scipy.fftpack import next_fast_len
import pywt

# import ghostipy as gsp
# alternatively see ghostipy for mtspectrogram, cwt and synchrosqueezed

########################################################################
## Spectrogram
########################################################################
def spectrogram(data, fs, *, nperseg=258, noverlap=16, axis=-1):
    """
    ...
    """
    l = data.shape[axis]
    coefs, frequencies = sp_spectrogram(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=axis,
        nfft=next_fast_len(l),
    )

    return coefs, frequencies


def periods2scales(periods, wavelet=None, dt=1.0):
    return (periods / dt) * pywt.central_frequency(wavelet)


def freqs2scales(freqs, wavelet=None, dt=1.0):
    return pywt.central_frequency(wavelet) / (freqs * dt)


def cwt_spectrogram(data, freqs, Fs, *, wavelet="cmorl3-7", method="fft", axis=-1):
    """
    Compute the wavelet transform of the input signal.

    Parameters
    ----------
    - data: array_like
        signal to be computed
    - freqs: array_like
        array of frequency of interest for the continuous wavelet transform
    - fs: float
        Sampling frequency in Hertz

    Returns
    -------

    - coefs: array_like
        Continuous wavelet transform of the input data for the given
        scales and wavelet. The first axis of coefs corresponds to the
        scales and the second is time.

    - frequencies : array_like
        frequencies corresponding to the coefficients

    References
    ----------
    https://github.com/alsauve/scaleogram
    """
    if any(freqs <= 0):
        freqs[freqs <= 0] = 1

    dt = 1 / Fs
    scales = freqs2scales(freqs, wavelet=wavelet, dt=dt)

    coefs, frequencies = pywt.cwt(
        data, scales, wavelet=wavelet, sampling_period=dt, method=method, axis=axis
    )

    return np.abs(coefs), frequencies
