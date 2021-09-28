#%% Import
import unittest

import numpy as np
from hippocampy.stats import stats


#%%
class TestStats(unittest.TestCase):
    def test_mad(self):
        dat = np.array([1, 4, 4, 7, 12, 13, 16, 19, 22, 24])
        m = stats.mad(dat)
        assert m == 7.5

    def test_mad_norm(self):
        dat = np.array([1, 4, 4, 7, 12, 13, 16, 19, 22, 24])
        m = stats.mad(dat, scale="normal")
        self.assertAlmostEqual(m, 11.1195, places=4)

    def test_mad_nan(self):
        dat = np.array([1, 4, 4, 7, 12, 13, 16, 19, 22, np.nan])
        m = stats.mad(dat)
        self.assertAlmostEqual(m, 7.0, places=3)

    def test_mad_2D(self):
        dat = np.array(
            [[1, 4, 4, 7, 12, 13, 16, 19, 22, 24], [1, 4, 4, 7, 12, 13, 16, 19, 22, 24]]
        )
        m = stats.mad(dat)
        np.testing.assert_array_almost_equal(m, [7.5, 7.5]) == None

