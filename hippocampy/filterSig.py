import numpy as np
from scipy.signal import filtfilt, butter, cheby2
from scipy.signal.signaltools import hilbert
from scipy.fftpack import next_fast_len


def bandpassSig(sig, fRange, fs, method="cheby2", order=4):
    """ 
    Filter a signal in a certain frequency band and with a particular filter type
    """

    allMethods = ['butter', 'cheby2']
    assert any(method == s for s in allMethods), "Invalid Method in bandpassSig"

    sig = np.asarray(sig)
    nyquist = 0.5 * fs

    if method == 'butter':
        b, a = butter(order, [fRange[0]/nyquist,
                              fRange[1]/nyquist], btype='band')
    elif method == 'cheby2':
        b, a = cheby2(order, 20, [fRange[0]/nyquist,
                                  fRange[1]/nyquist], btype='band')
    else:
        raise NotImplementedError
    sig_f = filtfilt(b, a, sig)

    return sig_f


def hilbertPhase(sig, method='hilbert'):
    """ 
    Compute hilbert phase and amplitude of a signal.

    TO DO: implement linear interpolation and shape preserving phase
    """

    if method == 'hilbert':
        n_sig = len(sig)
        bestLen = next_fast_len(n_sig)
        analytic_signal = hilbert(sig, bestLen)
        # remove padding
        analytic_signal = analytic_signal[:n_sig]
        amplitude_envelope = np.abs(analytic_signal)
    else: 
        raise NotImplementedError("phase method not implemented")

    return analytic_signal, amplitude_envelope


def instantaneousFreq(sig_p, fs):
    """
    Return instantaneous frequency of a signal from its phase. 
    This phase can be computed with hilbertPhase()
    """

    sig_p_u = np.unwrap(sig_p)
    return (np.diff(sig_p_u) / (2.0*np.pi) * fs)
