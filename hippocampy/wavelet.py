import numpy as np
import pywt
import bottleneck as bn


########################################################################
## Spectrogram
########################################################################


def cwt_spectrogram(
    data, scales, Fs, *, window=1, wavelet="morl", method="fft", axis=-1
):
    """
    Compute the wavelet transform of the input signal. 

    Parameters
    ----------
    - data: array_like
        signal to be computed
    - scales: 

    - Fs: float
        Sampling frequency in Hertz
    - window: int
        size of the temporal window (in samples) to average data into
    - wavelet: str 
        type of wavelet to use. available wavelet can be found using:
            import pywt
            print(pywt.wavelist())
    - method: str ( fft / conv)
        method to compute the cwt, fft will be faster for long input 
        while conv will be better for small ones
    - axis: int
        axis to work along

    Returns
    -------

    - coeffs: array_like
        Continuous wavelet transform of the input data for the given 
        scales and wavelet. The first axis of coeffs corresponds to the 
        scales and the second is time.
    
    - frequencies : array_like
        frequencies corresponding to the coefficients


  """
    P = 1 / Fs
    coeffs, freq = pywt.cwt(
        data, scales, wavelet=wavelet, sampling_period=P, method=method, axis=axis
    )

    return coeffs, freq


# import matplotlib.pyplot as plt
# import pywt
# import numpy as np
# from scipy.signal import decimate

# fpath = "/home/bouromain/Documents/tmpData/m4540_2021-07-30_15-04-15/Record Node 107/experiment_1.nwb"

# data, timestamps = load_nwb_oe(fpath)

# D = decimate(data[:, 14], 10)
# DD = D[:100000]

# P = 1 / 2000
# scales = np.arange(2, 200)  ## scales are wrong here


# coeffs, freq = pywt.cwt(DD, scales, wavelet="morl", sampling_period=P, method="fft")
# # calculate power from coefficient
# pow_c = np.abs(coeffs[:1000]) ** 2
# # calculate 1/f
# pow_unbiased = pow_c / freq[:, None]

# plt.imshow(pow_unbiased, aspect="auto")
# # plt.yticks(pywt.scale2frequency("morl", scales) / P)


########################################################################
## Denoising
########################################################################


def _w_noise_est(dC, n_sample, noise_est_method, axis=-1):
    """
    Compute threshold for noise filtering using various methods
    """
    threshold = [None] * len(dC)

    for i, c in enumerate(dC):
        # define threshold
        if noise_est_method == "sqtwolog":
            threshold[i] = np.sqrt(2 * np.log(n_sample))

        elif noise_est_method == "minimaxi":
            if n_sample < 32:
                threshold = 0
            else:
                threshold[i] = 0.3936 + 0.1829 * (np.log(n_sample) / np.log(2))

        elif noise_est_method == "rigrsure":
            raise NotImplementedError(
                "Noise estimation method %s not implemented" % (noise_est_method)
            )

        elif noise_est_method == "heursure":
            raise NotImplementedError(
                "Noise estimation method %s not implemented" % (noise_est_method)
            )

        else:
            raise ValueError(
                "Invalid noise estimation method, noise_est_method = %s"
                % (noise_est_method)
            )

    return threshold


def _thresh_coeff(coeffs, threshold, threshold_type, axis):
    """
    threshold coefficients with a given threshold and type
    """
    coeffs_f = coeffs
    for i, (cD, T) in enumerate(zip(coeffs_f[1:], threshold)):
        if axis == 1 or axis == -1:
            n_sig = cD.shape[0]
            for j in range(n_sig):
                coeffs_f[i + 1][j, :] = pywt.threshold(
                    cD[j, :], T[j], mode=threshold_type, substitute=0
                )
        elif axis == 0:
            n_sig = cD.shape[1]
            for j in range(n_sig):
                coeffs_f[i + 1][:, j] = pywt.threshold(
                    cD[:, j], T[j], mode=threshold_type, substitute=0
                )
    return coeffs_f


def _calc_sigma(coeffs, scaling, axis):
    # find threshold rescaling coefficients
    if scaling.lower() == "one":
        sigma = [np.ones_like(detcoef) for detcoef in coeffs[1:]]

    elif scaling.lower() == "sln":
        # scale according to the first order coefficient
        sigma = [
            bn.nanmedian(np.abs(coeffs[1]), axis=axis) / 0.6745
            for detcoef in coeffs[1:]
        ]

    elif scaling.lower() == "mln":
        # for multi level scaling
        sigma = [
            bn.nanmedian(np.abs(detcoef), axis=axis) / 0.6745 for detcoef in coeffs[1:]
        ]
    return sigma


def wden2(
    data,
    *,
    wavelet_name="sym5",
    level=None,
    noise_est_method="sqtwolog",
    scaling="mln",
    threshold_type="soft",
    axis=-1
):
    """

    Parameters
    ----------
    - data: array
    - wavelet_name: str
      name of the wavelet to use 'db1' 'sym4'. See all available wavelet with:
        import pywt
        print(pywt.wavelist())
    - level: int
        Decomposition level (must be >= 0). If level is None (default) then
        it will be calculated using the dwt_max_level function.
    - noise_est_method
            'rigrsure', adaptive threshold selection using principle
              of Stein's Unbiased Risk Estimate.
            'heursure', heuristic variant of the first option.
            'sqtwolog', threshold is sqrt(2*log(length(X))).
            'minimaxi', minimax thresholding.
    - scaling:
            'one' for no rescaling
            'sln' for rescaling using a single estimation of level noise based
               on first-level coefficients
            'mln' for rescaling done using level-dependent estimation of level noise
    - threshold_type (soft / hard)
        threshold type to filter coefficent

    Reference
    ---------
    [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
             by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
             DOI: 10.1093/biomet/81.3.425

    https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
    https://github.com/matthewddavis/lombardi/blob/master/processing/NPS-1.3.2/WDen.py

    TODO
    ----application
    - adapt for application along one axis
    - implement rigsur and eursur from
      https://github.com/holgern/pyyawt/blob/master/pyyawt/denoising.py
    """

    data = np.array(data, ndmin=2)

    if axis == 1 or axis == -1:
        n_sample = data.shape[1]
        # n_cell = data.shape[0]
    elif axis == 0:
        n_sample = data.shape[0]
        # n_cell = data.shape[1]
    else:
        raise ValueError("Axis should be either [0,1,-1]")

    # decompose the signal
    coeffs = pywt.wavedec(
        data, wavelet_name, level=level, mode="symmetric", axis=axis
    )  # axis to add

    sigma = _calc_sigma(coeffs, scaling, axis=axis)

    threshold = _w_noise_est(coeffs[1:], n_sample, noise_est_method)

    # rescale threshold
    threshold = [s * t for (s, t) in zip(sigma, threshold)]

    # Estimates via the noise via method in [2]  and define
    coeffs_f = _thresh_coeff(coeffs, threshold, threshold_type, axis=axis)

    # reconstruct the signal
    data_rec = pywt.waverec(coeffs_f, wavelet_name, mode="symmetric", axis=axis)

    # check if they are the same size
    # get rid of the extended part for wavelet decomposition
    return data_rec


def swt_denoise(data, *, wavelet_name="sym5", level=None):
    """
    swt_denoise is made to mach matlab option 'modwtsqtwolog' in wden.
    Note that modwt is a synonym of stationay wavelet transform [1]

    Reference
    ---------
    [1] https://en.wikipedia.org/wiki/Stationary_wavelet_transform#Synonyms
    """
    data = np.array(data, ndmin=2)
    n_sample = data.shape[-1]

    coeffs = pywt.swt(data, wavelet_name, level=level, trim_approx=True)

    s = [np.sqrt(2) * bn.nanmedian(np.abs(c)) / 0.6745 for c in coeffs[1:]]
    s = np.array(s)

    n1 = 2 * s ** 2
    d1 = 2 ** (np.arange(level) + 1)
    n2 = np.log(n_sample)

    # find threshold
    threshold = np.sqrt(n1 / d1 * n2)

    # threshold coefficients
    coeffs_f = coeffs
    for i, (cD, T) in enumerate(zip(coeffs, threshold)):
        coeffs_f[i + 1] = pywt.threshold(cD, T, mode="soft", substitute=0)
    # reconstruct the signal
    data_rec = pywt.iswt(coeffs_f, wavelet_name)

    return data_rec
