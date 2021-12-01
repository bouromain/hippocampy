import unittest
import numpy as np
from hippocampy.stats import distance


class TestDistance(unittest.TestCase):
    def test_cos_sim(self):
        a = np.array([1, 2, 3])
        b = np.array([3, 2, 1])

        self.assertAlmostEqual(distance.cos_sim(a, b).squeeze(), 0.714285, places=5)

    def test_cos_sim_orthog(self):
        a = np.array([1, 0, 1])
        b = np.array([0, 2, 0])

        self.assertEqual(distance.cos_sim(a, b), 0)

    def test_cos_sim_orthog(self):
        a = np.array([1, 0, 1])
        b = np.array([0, 2, 0])

        self.assertEqual(distance.cos_sim(a, b), 0)

    def test_cos_sim_error_size(self):
        a = np.array([1, 0, 1])
        b = np.array([0, 2, 0, 0])

        with self.assertRaises(ValueError):
            distance.cos_sim(a, b)

