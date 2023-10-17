# %% Import
import warnings
from hippocampy import sig_tool
import unittest
import numpy as np


# %%
class TestLabel(unittest.TestCase):
    def test_xcorr(self):
        # test boolean vector
        x = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1])
        out = sig_tool.xcorr(a)
        assert all(out == np.array([1, 0, 2, 2, 0, 3]))
