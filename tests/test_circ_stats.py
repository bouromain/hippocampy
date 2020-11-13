from hippocampy.stats import circ_stats
import pytest
import unittest
import numpy as np
from numpy import pi

#%%
class TestHelper(unittest.TestCase):
    def test_isradians(self):
        a = np.array([-1,2,3])
        b = np.array([1,4,5])
        c = np.array([1,4,24])

        self.assertAlmostEqual(circ_stats.isradians(a), 1)
        self.assertEqual(circ_stats.isradians(b), 2)
        self.assertEqual(circ_stats.isradians(c), 0)

    def test_circmean(self):
        theta = [0.1, 2*pi + 0.2, 6*pi + 0.3]
        self.assertAlmostEqual(circ_stats.circ_mean(theta), 0.2)

    def test_circ_r(self):
        theta = [pi , 2*pi , 2*pi + 0.3 , pi + 0.4]
        self.assertAlmostEqual(circ_stats.circ_r(theta), 0.025, places=3)
