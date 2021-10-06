import numpy as np
import os


def save_npy(fpath: str, arr: np.ndarray, *, overwrite: bool = False):
    """
    wrapper to np.save that check if file exist and save arr in it 

    Parameters
    ----------
    fpath : str
        full path of the file to save
    arr : np.ndarray
        variable to save 
    overwrite : bool, optional
        if we should overwrite existing files, by default False
    Raises
    ------
    FileExistsError
        [description]
    ValueError
        [description]
    """

    if os.path.exist(fpath) and not overwrite:
        raise FileExistsError(f"File {fpath} already exist, use overwrtie=True")

    if os.path.isdir(fpath) and not overwrite:
        raise ValueError(f"File {fpath}should be a path not a directory")

    with open(fpath, "wb") as fio:
        np.save(fio, arr, allow_pickle=True)

