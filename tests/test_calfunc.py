from hippocampy.calfunc import detrend_F
import numpy as np
import hippocampy as hp
import matplotlib.pyplot as plt


F, Fneu, iscell, spk, stats = hp.data.load_calcium()
F_f = hp.calfunc.subtract_neuropil(F, Fneu)
F_d = hp.calfunc.detrend_F(F_f, 600)

