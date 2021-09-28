from hippocampy.calfunc import detrend_F, deconvolve
from hippocampy.wavelet import wden
import numpy as np
import hippocampy as hp
import matplotlib.pyplot as plt

# F, Fneu, iscell, spk, stats = hp.data.load_calcium()
# # F_f = hp.calfunc.subtract_neuropil(F, Fneu)
# # F_d = hp.calfunc.detrend_F(F_f, 600)
# # F_d = wden(F_d,level=4)

# # c, s = deconvolve(F_d)

# S_d = S / noise[:, None]

# n = 5000
# idxc = 8

# # plt.plot(F_d[idxc,:n] - B[idxc])
# # plt.plot(F_c[idxc,:n])
# plt.plot(S[idxc,:n])
# plt.plot(S_d[idxc,:n])
# plt.show()
