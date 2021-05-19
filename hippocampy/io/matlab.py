from scipy.io import loadmat as loadmat_old
from mat73 import loadmat as loadmat73


def loadmat(path_file, squeeze_me=True, use_attrdict=True, verbose=False):
    """
    This funciton implement the loading of Matlab files .mat
    By default it will use scipy.io but as it does not implement
    loading Matlab 7.3 data it will then use mat73 if an error is raised

    Parameters
    ----------
    path_file: str
        path of the file to open
    squeeze_me: bool
        if we need to sueeze the data returned by loadmat
        only for non matlab 7.3 files
    use_attrdict: bool
        make it possible to access structs like in MATLAB
        only for matlab 7.3 files
    verbose:bool
        verbose execution

    Returns
    -------
    data:
        data extracted from the mat file

    """

    try:
        data = loadmat_old(path_file, squeeze_me=squeeze_me)
    except NotImplementedError as err:
        if (
            repr(err)
            == "NotImplementedError('Please use HDF reader for matlab v7.3 files')"
        ):
            if verbose:
                print("Loading Matlab 7.3 data")
            data = loadmat73(path_file, use_attrdict=True)
    return data
