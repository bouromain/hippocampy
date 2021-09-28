from hippocampy.calfunc import detrend_F, deconvolve
from hippocampy.wavelet import wden
import numpy as np
import hippocampy as hp
import matplotlib.pyplot as plt

# F, Fneu, iscell, spk, stats = hp.data.load_calcium()
# F_f = hp.calfunc.subtract_neuropil(F, Fneu)
# F_d = hp.calfunc.detrend_F(F_f, 600)
# F_d = wden(F_d,level=4)

# c, s = deconvolve(F_d)

# # [Q(it,:),S(it,:)] = deconvolveCa(F_dn(it,:), 'foopsi', ...
# #         'ar1', 'smin', -thresh,'optimize_pars', true, 'optimize_b', true);
# n = 5000
# idxc = 1

# plt.plot(F_d[idxc,:n])
# plt.plot(c[idxc,:n])
# plt.plot(s[idxc,:n])
# plt.plot(spk[idxc,:n],"--")
# plt.show()

