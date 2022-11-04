import numpy as np
import pywt
import bottleneck as bn


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


def wden(
    data,
    *,
    wavelet_name="sym4",
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

    # enforce same size and return (we sometimes have problems for odd size)
    return data_rec[: data.shape[0], : data.shape[1]]


def swt_denoise(data, *, wavelet_name="sym4", level=5, axis=-1):
    """
    swt_denoise is made to mach matlab option 'modwtsqtwolog' in wden.
    Note that modwt is a synonym of stationary wavelet transform [1]

    Reference
    ---------
    [1] https://en.wikipedia.org/wiki/Stationary_wavelet_transform#Synonyms
    """

    if axis == 1 or axis == -1:
        n_sample = data.shape[1]
        n_sig = data.shape[0]
    elif axis == 0:
        n_sample = data.shape[0]
        n_sig = data.shape[1]
    else:
        raise ValueError("Axis should be either [0,1,-1]")

    data = np.array(data, ndmin=2)
    n_sample = data.shape[-1]
    coeffs = pywt.swt(data, wavelet_name, level=level, trim_approx=True, axis=axis)

    s = np.array([np.sqrt(2) * bn.nanmedian(np.abs(c)) / 0.6745 for c in coeffs[1:]])

    n1 = 2 * s ** 2
    d1 = 2 ** (np.arange(level) + 1)
    n2 = np.log(n_sample)

    # find threshold
    threshold = np.sqrt(n1 / d1 * n2)
    threshold = np.ones((len(threshold), n_sig)) * threshold[:, None]

    coeffs_f = _thresh_coeff(coeffs, threshold, threshold_type="soft", axis=axis)

    # reconstruct the signal
    data_rec = np.empty_like(data)
    for it in np.arange(n_sig):
        if axis == 1 or axis == -1:
            c = [tmp_c[it, :] for tmp_c in coeffs_f]
            data_rec[it, :] = pywt.iswt(c, wavelet_name)
        elif axis == 0:
            c = [tmp_c[:, it] for tmp_c in coeffs_f]
            data_rec[:, it] = pywt.iswt(c, wavelet_name)

    return data_rec
