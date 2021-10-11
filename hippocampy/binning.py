import bottleneck as bn
import numpy as np

from hippocampy.utils.type_utils import float_to_int
from hippocampy.matrix_utils import smooth_1d


def rate_map(
    var: np.ndarray,
    samples: np.ndarray,
    *,
    bins=10,
    fs: int = 30,
    smooth_half_win: int = 0,
    smooth_axis: int = 0,
    method="spk",
    preserve_nan_opt=True
):
    """
    rate_map [summary]

    Parameters
    ----------
    var : np.ndarray
        for multidimensional data this function expect the input to be var[n_samples, n_dim]
    samples : np.ndarray
        can be either a continuous variable of size 
    bins : int, optional
        [description], by default 10
    method : str, optional
        [description], by default "spk"

    Returns
    -------
    rate : np.ndarray
        [description]
    act : np.ndarray
        [description]
    occ : np.ndarray
        [description]
    Raises
    ------
    ValueError
        in case of error in provided method
    """
    # TODO take care of the no_occupancy case and set them to nan
    # take care of the bins
    # correctly smooth

    # check inputs
    var = np.asarray(var)
    samples = np.asarray(samples)

    if not method in ["mean", "spk"]:
        raise ValueError("Method should be either mean or spk")
    if method == "spk" and samples.dtype.kind != "i":
        samples = float_to_int(samples)

    # first take care of the bins
    if isinstance(bins, (list, np.ndarray)):
        bins = [bins]
    elif isinstance(bins, tuple):
        bins = [*bins]

    # calculate occupancy
    occ, _ = np.histogramdd(var, bins)

    # affect no occupancy to nan. This avoid zero division later
    # and is necessary to take into account non explored areas
    no_occ = occ == 0
    occ[no_occ] = np.nan
    occ = occ / fs  # convert in Hz

    if method == "mean":
        act, _ = np.histogramdd(var, bins, weights=samples)
        act /= occ
    elif method == "spk":
        act, _ = np.histogramdd(var[samples], bins)

    if smooth_half_win > 0:
        act_s = smooth_1d(
            act, smooth_half_win, axis=smooth_axis, preserve_nan_opt=preserve_nan_opt
        )
        occ_s = smooth_1d(
            occ, smooth_half_win, axis=smooth_axis, preserve_nan_opt=preserve_nan_opt
        )
    else:
        act_s, occ_s = act, occ

    if method == "spk":
        rate_s = act_s / occ_s
    elif method == "mean":
        rate_s = act_s

    return rate_s, act_s, occ_s
