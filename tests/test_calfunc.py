from hippocampy import wavelet
import unittest
import numpy as np
import os
from hippocampy.io.s2p import load_all_s2p
import hippocampy as hp
import matplotlib.pyplot as plt

p = "/mnt/data_pool/DataToShare/DataDori/Data2Process/m4540/20210708/plane0"

F, Fneu, spks, stat, ops, iscell = load_all_s2p(p)
F_f = hp.calfunc.subtract_neuropil(F, Fneu)


F_d = hp.wavelet.swt_denoise(F_f[:2, :10000], level=4)

import pywt
import bottleneck as bn

data = F_f[:20, :]
wavelet_name = "sym4"
level = 2
axis = -1


data = np.array(data, ndmin=2)

if axis == 1 or axis == -1:
    n_sample = data.shape[1]
    n_sig = data.shape[0]
elif axis == 0:
    n_sample = data.shape[0]
    n_sig = data.shape[1]
else:
    raise ValueError("Axis should be either [0,1,-1]")

coeffs = pywt.swt(data, wavelet_name, level=level, trim_approx=True, axis=axis)

s = np.array([np.sqrt(2) * bn.nanmedian(np.abs(c)) / 0.6745 for c in coeffs[1:]])

n1 = 2 * s ** 2
d1 = 2 ** (np.arange(level) + 1)
n2 = np.log(n_sample)

# find threshold
threshold = np.sqrt(n1 / d1 * n2)
threshold = np.ones((len(threshold),n_sig)) * threshold[:,None]

coeffs_f = hp.wavelet._thresh_coeff(coeffs,threshold,threshold_type="soft",axis=axis)

# reconstruct the signal
data_rec = np.empty_like(data)
for it in np.range(n_sig):
    if axis == 1 or axis == -1:
        c = [tmp_c[it,:] for tmp_c in coeff_f]
        data_rec[:,it] = pywt.iswt(c, wavelet_name)
    elif axis = 0:
        c = [tmp_c[:,it] for tmp_c in coeff_f]
        data_rec[it,:] = pywt.iswt(c, wavelet_name)


data_rec = pywt.iswtn(coeffs_f, wavelet_name,axes=axis)

