import numpy as np
import os


def load_dat(path, n_vars=None, dtype="float"):
    """
    Open file in .data format produced by ViRMen

    Parameters
    ----------
            - path: file path
            - n_vars: number of column needed to reshape the data
            - dtype: type of the data to read

    Return
    ------
            - data: data as a numpy array
    """

    # to avoid problems
    path = os.path.expanduser(path)

    # check file exist
    if not os.path.isfile(path):
        raise FileNotFoundError

    # now we can open the file safely
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)

    n_sample = len(data) / n_vars

    assert n_sample == np.floor(
        n_sample
    ), "number of sample in data is not divisible by n_vars "

    if n_vars is not None:
        data = np.reshape(data, (int(n_sample), n_vars))

    return data


# def export_data_to_npy(scr_folder_path, dest_folder_path=None):
#     """
#     Export and slightly clean the content of a folder containing .data file to npy
#     """

#     if dest_folder_path is None:
#         dest_folder_path = scr_folder_path

