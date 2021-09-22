import unittest
import numpy as np
import os
from hippocampy.io.s2p import load_all_s2p
import hippocampy as hp
import matplotlib.pyplot as plt
from hippocampy.matrix_utils import remove_small_objects, first_true
import bottleneck as bn


# p = "/mnt/data_pool/DataToShare/DataDori/Data2Process/m4540/20210708/plane0"
# axis = 1
# min_length = 5

# F, Fneu, spks, stat, ops, iscell = load_all_s2p(p)
# F_f = hp.calfunc.subtract_neuropil(F, Fneu)
# F_d = hp.calfunc.detrend_F(F_f, 600)
# F_d = hp.wavelet.wden(F_d, level=4)


# # define the threshold for candidate events
# F_mean = bn.nanmean(F_d, axis=axis)
# F_std = bn.nanstd(F_d, axis=axis)
# T = F_mean + 2.5 * F_std
# # Threshold the signal
# F_b = F_d > T[:, None]

# # label the events
# F_b = hp.matrix_utils.label(F_b, axis=1)

# F_t = first_true(F_t)

# events = [np.nonzero(f)[0] for f in F_t]

# plt.plot(F_d[0,:])
# plt.plot(events[0],F_d[0,events[0]],'or')
# plt.xlim([35000,37000])
# plt.show()


# import matplotlib.pyplot as plt

# plt.plot(C[1, :10000])
# plt.plot(F_d[1, :10000] * 0.0010)
# plt.xlim([0, 10000])
# plt.show()
