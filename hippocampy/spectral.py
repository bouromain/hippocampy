import numpy as np
import ghostipy as gsp

########################################################################
## Spectrogram
########################################################################
def spectrogram(data, fs, *, bandwidth=15, nperseg=258, noverlap=16):
    """
    ...
    """
    coefs, frequencies, timestamps = gsp.mtm_spectrogram(
        data, fs=fs, bandwidth=bandwidth, nperseg=nperseg, noverlap=noverlap
    )

    return coefs, frequencies, timestamps


def cwt_spectrogram(data, fs, *, freqs=[2, 250], axis=-1):
    """
    Compute the wavelet transform of the input signal. 

    Parameters
    ----------
    - data: array_like
        signal to be computed
    - freqs: 

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


  """
    if any(freqs <= 0):
        freqs[freqs <= 0] = 1
    coefs, _, frequencies, timestamps, _ = gsp.cwt(data, fs=fs, freq_limits=freqs)
    return np.abs(coefs), frequencies, timestamps

