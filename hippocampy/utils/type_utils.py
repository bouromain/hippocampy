import numpy as np
from typing import Union


def is_number(s: Union[str, int, float]):
    try:
        float(s)
        return True
    except ValueError:
        return False


def float_to_int(array) -> np.array:
    """
    float_to_int
    Cast float to int but check value are then equal

    Parameters
    ----------
    array : [type]
        input array

    Returns
    -------
    array:
        output array converted to int

    Raises
    ------
    TypeError
        [description]

    Reference
    ---------
    https://stackoverflow.com/questions/36505879/convert-float64-to-int64-using-safe
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if array.dtype.kind == "i":
        return array

    int_array = array.astype(int, casting="unsafe", copy=True)
    if not np.equal(array, int_array).all():
        raise TypeError(
            "Cannot safely convert float array to int dtype. "
            "Array must only contain whole numbers."
        )
    return int_array
