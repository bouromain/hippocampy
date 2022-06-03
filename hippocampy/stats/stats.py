import bottleneck as bn
import numpy as np
from scipy.optimize import curve_fit


def mad(x: np.ndarray, scale: float = None, axis: int = -1):
    """
    mad Calculate median absolute deviation

    Parameters
    ----------
    x : np.ndarray
        input vector or array
    scale : (float, str = "normal" ) optional
        scaling factor of the MAD, by default None
        if scale = "normal" scale will be affected to 1.48 (see Ref)
    axis : int, optional
        axis to work on, by default -1

    Returns
    -------
    mad [float, np.ndarray]
        median absolute deviation

    Raises
    ------
    ValueError
        [description]


    Reference
    ---------
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """

    if scale is None:
        scale = 1
    elif isinstance(scale, str):
        if scale.lower() == "normal":
            scale = 1.482602218505602
        else:
            raise ValueError(f"{scale} is not a valid scale value")

    med = bn.nanmedian(x, axis=axis)

    if isinstance(med, np.ndarray):
        med = np.expand_dims(med, axis=axis)

    return bn.nanmedian(np.abs(x - med), axis=axis) * scale
