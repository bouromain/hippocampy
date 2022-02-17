from typing import Union

import bottleneck as bn
import numpy as np

from hippocampy.matrix_utils import (
    first_true,
    last_true,
    remove_holes,
    remove_small_objects,
    smooth_1d,
)


def find_lap_1d(
    pos: np.ndarray,
    len_maze: Union[float, None] = None,
    thresh_teleportation: int = 20,
    remove_start_end_pauses: bool = True,
    min_dist_ratio: float = 0.8,
    max_dist_ratio: float = 1.5,
) -> np.ndarray:
    """
    find_lap_1d find laps from a 1d position vector
    Note: non laps will be set to nan

    Parameters
    ----------
    pos : np.ndarray
        position vector
    len_maze : Union[float, None], optional
        [description], by default None
    thresh_teleportation : int, optional
        [description], by default 20
    remove_start_end_pauses : bool, optional
        [description], by default True
    min_dist_ratio : float, optional
        [description], by default 0.8
    max_dist_ratio : float, optional
        [description], by default 1.5

    Returns
    -------
    np.ndarray
        [description]
    """

    # check inputs
    if np.squeeze(pos).ndim > 1:
        raise ValueError("Positon vector should be 1 D")

    # to define a minimum distance per lap (remove small artefactual laps)
    if len_maze is None:
        min_dist = 0
        max_dist = np.Inf
    else:
        min_dist = len_maze * min_dist_ratio
        max_dist = len_maze * max_dist_ratio

    out = np.empty_like(pos)
    out[:] = np.nan
    # detect teleportation
    pos_d = np.abs(np.diff(pos, append=np.nan))
    idx_tel = first_true(pos_d > thresh_teleportation)
    idx_tel = np.nonzero(idx_tel)[0]
    idx_tel = np.hstack((0, idx_tel, len(pos)))

    # define starts and stops
    starts = idx_tel[:-1]
    stops = idx_tel[1:]

    lap_i = 0
    for i_start, i_stop in zip(starts, stops):
        # if we exceed a minimum distance we are in a candidate lap
        tmp_pos_d = pos_d[i_start:i_stop]
        # here I take a little margin {+-10} to be sure to remove the teleportation
        # an other approach would be to take into account double the distance
        # (length of the maze and the teleport to zero)

        if len(tmp_pos_d) > 20 and (min_dist < bn.nansum(tmp_pos_d[10:-10]) < max_dist):
            # remove the pauses of when the animal is
            # blocked at the beguining or start of a lap
            if remove_start_end_pauses:
                if tmp_pos_d[0] == 0:
                    offset_start = last_true(tmp_pos_d > 0)
                    offset_start = min(np.nonzero(offset_start)[0])
                else:
                    offset_start = 0

                if tmp_pos_d[-1] == 0:
                    offset_stop = first_true(tmp_pos_d == 0)
                    offset_stop = max(np.nonzero(offset_stop)[0])
                    offset_stop = len(tmp_pos_d) - offset_stop
                else:
                    offset_stop = 0

                out[i_start + offset_start : i_stop - offset_stop] = lap_i
            else:
                out[i_start:i_stop] = lap_i

            lap_i += 1
    return out


def find_active(
    speed: np.ndarray,
    speed_thresh: float = 2.0,
    min_min_n_samples: int = 0,
    min_inter_samples: int = 0,
    smooth_half_win: int = 10,
) -> np.ndarray:
    """
    find_active find active period from a vector of speed

    Parameters
    ----------
    speed : np.ndarray
        speed values
    speed_thresh : float, optional
        minimum speed threshold, by default 2.0
    min_min_n_samples : int, optional
        minimum samples in an activity period , by default 10
    min_inter_samples : int, optional
        minimum inter pauses distances, it will merge activity periods if 
        they are not spaced by at least this distance, 
        by default 10 [samples]
    smooth_half_win: int, optional
        width of the half window in case we want to perform a gaussian smooth on
        the data

    Returns
    -------
    np.ndarray
        [description]
    """
    if smooth_half_win > 0:
        speed_s = smooth_1d(speed, kernel_half_width=smooth_half_win)
    else:
        speed_s = speed

    speed_b = np.abs(speed_s) >= speed_thresh

    # merge very close activity periodes
    speed_b = remove_holes(speed_b, min_inter_samples)
    # remove periods that are too small
    return remove_small_objects(speed_b, min_min_n_samples)

