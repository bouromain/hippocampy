import numpy as np
import bottleneck as bn
from itertools import combinations
from hippocampy.utils.type_utils import float_to_int
from hippocampy.utils.nan import remove_nan


def stability(map_laps: np.ndarray, method: str = "all-median") -> float:
    """
    compute the stability across laps of a given cell with a specified method

    Parameters
    ----------
    map_laps : np.ndarray
        matrix of activity per lap map_laps[n_laps,n_bins]
    method : str, optional
        method to compute the stability, by default "all-median"

    Returns
    -------
    float
        stability index

    """
    if method not in ["half", "odd-even", "all-mean", "all-median"]:
        raise ValueError(f"Method {method} not recognized")

    n_lap = map_laps.shape[0]

    if method in ["half", "odd-even"] and (n_lap % 2 != 0):
        # if the number of laps is odd make it even
        tmp_map_lap = map_laps[:-1, :]
    else:
        tmp_map_lap = map_laps

    if method == "half":
        half_idx = float_to_int(tmp_map_lap.shape[0] / 2)
        a = tmp_map_lap[:half_idx, :].ravel()
        b = tmp_map_lap[half_idx:, :].ravel()
        a, b = remove_nan(a, b, paired=True)
        SI = np.corrcoef(a, b)[0, 1]
    elif method == "odd-even":
        a = tmp_map_lap[::2, :].ravel()
        b = tmp_map_lap[1::2, :].ravel()
        a, b = remove_nan(a, b, paired=True)
        SI = np.corrcoef(a, b)[0, 1]
    elif method == "all-mean":
        all_coor = [np.corrcoef(a, b)[0, 1] for a, b in combinations(tmp_map_lap, 2)]
        SI = bn.nanmean(all_coor)
    elif method == "all-median":
        all_coor = [np.corrcoef(a, b)[0, 1] for a, b in combinations(tmp_map_lap, 2)]
        SI = bn.nanmedian(all_coor)

    return SI


def spatial_info(rate: np.ndarray, occ: np.ndarray, method="bit_sec"):
    # ref Skaggs, Markus 1996
    # Climer paper
    # Sousa paper
    # repo
    # https://github.com/DombeckLab/infoTheory/blob/master/smgmMI.m
    # https://github.com/kevin-allen/spatialInfoScore/blob/main/SpatialInfoScore.ipynb
    # https://github.com/tortlab/spatial-information-metrics/blob/master/info_metrics.m

    # I do not like it it needs to be re-written

    if method not in ["bit_sec", "bit_sec_hertz"]:
        raise ValueError(f"method {method} not recognized")

    no_occ = occ == 0
    tmp_rate = rate[~no_occ]
    tmp_occ = occ[~no_occ]

    # turn occ in a probability
    p_occ = tmp_occ / bn.nansum(tmp_occ)
    rate_m = bn.nanmean(tmp_rate)

    mask = tmp_rate > 0
    SI = bn.nansum(
        p_occ[mask] * (tmp_rate[mask] / rate_m) * np.log2(tmp_rate[mask] / rate_m)
    )

    if method == "bit_sec_hertz":
        SI = SI / rate_m
    return SI
