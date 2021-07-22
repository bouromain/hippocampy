import numpy as np
import os
import suite2p

## DOC: https://suite2p.readthedocs.io/en/latest/settings.html
## DOC: https://github.com/MouseLand/suite2p/blob/master/jupyter/run_pipeline_tiffs_or_batch.ipynb


#%% wrapper tho load data more conveniently
def loadS2p(pathfile, typeF="F.npy"):
    """
    Load a single suite2p file
    """
    data = np.load(os.path.join(pathfile, typeF), allow_pickle=True)
    return data


def loadOps(pathfile):
    ops = np.load(os.path.join(pathfile, "ops.npy"), allow_pickle=True)
    return ops.item()


def loadAllS2p(pathfile):
    """
    Load all suite2p files generated after preprocessing
    """
    F = loadS2p(pathfile, "F.npy")
    Fneu = loadS2p(pathfile, "Fneu.npy")
    spks = loadS2p(pathfile, "spks.npy")
    stat = loadS2p(pathfile, "stat.npy")
    ops = loadOps(pathfile)
    iscell = loadS2p(pathfile, "iscell.npy")

    return F, Fneu, spks, stat, ops, iscell


#%% preprocessing function utils
def make_ops(fs=None):
    """
    make 'custom' ops file for hippocampal 6f recording at 30hz
    """
    ops = suite2p.default_ops()
    ops["diameter"] = 10
    ops["tau"] = 0.7
    if fs is None:
        ops["fs"] = 30
    else:
        ops["fs"] = fs
    return ops


def run_s2p(pathfile):
    ops = make_ops()
    ops["data_path"] = [str(pathfile)]
    suite2p.run_s2p(ops)


#%% Miscellaneous helpers


def filterCell(iscell, data):
    """
    This function allow to easily filter data with the iscell variable from suite2p

    """

    if len(data.shape) == 1:
        return data[iscell[:, 0] == 1]
    else:
        return data[iscell[:, 0] == 1, :]
