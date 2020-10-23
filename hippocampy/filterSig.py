import numpy as np
import scipy.signal
import scipy.signal.signaltools

def bandpassSig(sig,fRange,fs,method ="cheby2",order=4):
    """ 
    Filter a signal in a certain frequency band and with a particular filter type
    """

    allMethods = ['butter', 'cheby2']
    assert any(method == s for s in allMethods), "Invalid Method in bandpassSig"

    sig = np.squeeze(sig)
    nyquist = 0.5 * fs

    if method == 'butter':
        b, a = scipy.signal.butter(order, [fRange[0]/nyquist, fRange[1]/nyquist], btype='band')
    elif method == 'cheby2':
        b, a = scipy.signal.cheby2(order, 20, [fRange[0]/nyquist, fRange[1]/nyquist] , btype='band')
    else: 
        raise NotImplementedError
    sig_f = scipy.signal.filtfilt(b, a, sig)


    return sig_f


def hilbertPhase(sig):
    """ 
    Compute hilbert phase and amplitude of a signal.

    TO DO: implement linear interpolation and shape preserving phase
    """
    n_sig = len(sig)
    bestLen = scipy.fftpack.next_fast_len(n_sig)
    analytic_signal = scipy.signal.hilbert(sig,bestLen)
    # remove padding
    analytic_signal = analytic_signal[:n_sig]
    amplitude_envelope = np.abs(analytic_signal)

    return analytic_signal , amplitude_envelope

def instantaneousFreq(sig_p, fs):
    """
    Return instantaneous freqyuency of a signal from its phase. 
    This phase can be computed with hilbertPhase
    """

    sig_p_u = np.unwrap(sig_p)
    return (np.diff(sig_p_u) / (2.0*np.pi) * fs)
