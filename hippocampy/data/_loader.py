import numpy as np
import os.path as op


def load_calcium():
    fpath = op.join(op.dirname(__file__), "2p")
    data = np.load(op.join(fpath, "fluo.npz"), allow_pickle=True)
    return data["F"], data["Fneu"], data["iscell"], data["spks"], data["stat"]


def load_fluo():
    fpath = op.join(op.dirname(__file__), "2p")
    data = np.load(op.join(fpath, "fluo.npz"), allow_pickle=True)
    return data["F"], data["Fneu"]

