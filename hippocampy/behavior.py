from typing import Union

import bottleneck as bn
import numpy as np

from hippocampy.matrix_utils import first_true, last_true


def find_lap_1d(
    pos: np.ndarray,
    len_maze: Union[float, None] = None,
    thresh_teleportation: int = 20,
    remove_start_end_pauses: bool = True,
    min_dist_ratio: float = 0.8,
    max_dist_ratio: float = 1.5,
) -> np.ndarray:
    """
    find_lap_1d [summary]

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

    out = np.zeros_like(pos, dtype=int)
    # detect teleportation
    pos_d = np.abs(np.diff(pos, append=np.nan))
    idx_tel = first_true(pos_d > thresh_teleportation)
    idx_tel = np.nonzero(idx_tel)[0]
    idx_tel = np.hstack((0, idx_tel, len(pos)))

    # define starts and stops
    starts = idx_tel[:-1]
    stops = idx_tel[1:]

    lap_i = 1
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

