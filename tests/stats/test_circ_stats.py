from hippocampy.stats import circular
import unittest
import numpy as np
from numpy import pi

#%%
class TestHelper(unittest.TestCase):
    def test_isradians(self):
        a = np.array([-1, 2, 3])
        b = np.array([1, 4, 5])
        c = np.array([1, 4, 24])

        self.assertAlmostEqual(circular.isradians(a), 1)
        self.assertEqual(circular.isradians(b), 2)
        self.assertEqual(circular.isradians(c), 0)

    def test_circmean(self):
        theta = [0.1, 2 * pi + 0.2, 6 * pi + 0.3]
        self.assertAlmostEqual(circular.circ_mean(theta), 0.2)

    def test_circ_r(self):
        theta = [pi, 2 * pi, 2 * pi + 0.3, pi + 0.4]
        self.assertAlmostEqual(circular.circ_r(theta), 0.025, places=3)

    def test_circ_cc(self):
        x = [0.785, 1.570, 3.141, 3.839, 5.934]
        y = [0.593, 1.291, 2.879, 3.892, 6.108]

        r, p = circular.corr_cc(x, y)
        self.assertAlmostEqual(r, 0.942, places=3)
        self.assertAlmostEqual(p, 0.0658, places=3)

    def test_circ_cl(self):
        # fmt: off
        x = np.asarray([107, 46, 33, 67, 122, 69, 43, 30, 12, 25, 37,\
        69, 5, 83, 68, 38, 21, 1, 71, 60, 71, 71, 57, 53, 38, 70, 7, 48, 7, 21, 27])
        phi = np.asarray([67, 66, 74, 61, 58, 60, 100, 89, 171, 166, 98,\
            60, 197, 98, 86, 123, 165, 133, 101, 105, 71, 84, 75, 98, 83, 71, 74, 91, 38, 200, 56])
        # fmt: on
        phi = np.deg2rad(phi)

        rho, slope, phi0, p_c = circular.lin_circ_regress(x, phi)
        self.assertAlmostEqual(rho, -0.499, places=3)
        self.assertAlmostEqual(slope, -0.012, places=3)
        self.assertAlmostEqual(phi0, 2.28, places=2)
        self.assertAlmostEqual(p_c, 0.018, places=2)
