import numpy as np
import os
import suite2p

## DOC: https://suite2p.readthedocs.io/en/latest/settings.html
## DOC: https://github.com/MouseLand/suite2p/blob/master/jupyter/run_pipeline_tiffs_or_batch.ipynb

#%% wrapper tho load data more conveniently
def load_s2p(pathfile, typeF="F.npy"):
    """
    Load a single suite2p file
    """
    data = np.load(os.path.join(pathfile, typeF), allow_pickle=True)
    return data


def load_ops(pathfile):
    ops = np.load(os.path.join(pathfile, "ops.npy"), allow_pickle=True)
    return ops.item()


def load_all_s2p(pathfile):
    """
    Load all suite2p files generated after preprocessing
    """
    F = load_s2p(pathfile, "F.npy")
    Fneu = load_s2p(pathfile, "Fneu.npy")
    spks = load_s2p(pathfile, "spks.npy")
    stat = load_s2p(pathfile, "stat.npy")
    ops = load_ops(pathfile)
    iscell = load_s2p(pathfile, "iscell.npy")

    return F, Fneu, spks, stat, ops, iscell

