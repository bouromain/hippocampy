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


def _setval(d, path_d, val):
    if len(path_d) == 1:
        d[path_d[0]] = val
    else:
        first_level = path_d[0]
        if first_level not in d:
            d[first_level] = {}

        _setval(d[first_level], path_d[1:], val)


def matlab_string2dict(in_string: str):
    """
    Parse matlab style string to make a nice dictionary from it 
    It will recognize numeric or boolean value and convert them
    accordingly to their type 

    Parameters
    ----------
    met_string: str
        raw matlab string
    Returns
    -------
    out: dict
        nice dict converted from the matlaby string

    Example
    -------
    Matlab format:

    SI.extTrigEnable = 1
    SI.hBeams.beamCalibratedStatus = false

    Converted output:

    out['extTrigEnable'] = 1.0
    out['SI']['hBeams']['beamCalibratedStatus'] = False

    """
    # initialise empty dict
    out = {}

    # split per lines
    lines = in_string.split("\n")
    for l in lines:
        # split path and values
        x = l.split("=")
        if len(x) == 2:
            head = x[0]
            tail = x[1].strip()
            seq = head.split(".")
            seq = [s.strip() for s in seq]

            # convert value if needed
            if _is_number(tail):
                tail = float(tail)
                if tail.is_integer():
                    tail = int(tail)
            elif tail.lower() == "true":
                tail = True
            elif tail.lower() == "false":
                tail = False

            # set the value in the dictionary
            # as it can be nested we need to use the following function
            _setval(out, seq, tail)

    return out


def _is_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False
