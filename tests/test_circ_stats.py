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

    def test_circ_cc(self):
        x = [0.785, 1.570, 3.141, 3.839, 5.934]
        y = [0.593, 1.291, 2.879, 3.892, 6.108]

        r , p = circ_stats.corr_cc(x,y)
        self.assertAlmostEqual(r, 0.942, places=3)
        self.assertAlmostEqual(p, 0.0658, places=3)



    
