import unittest
import numpy as np
from hippocampy.stats import distance


class TestDistance(unittest.TestCase):
    def test_cos_sim_equal(self):
        a = np.array([-1, 2, 3])
        b = np.array([-1, 2, 3])

        self.assertEqual(distance.cos_sim(a, b), 1)

    def test_cos_sim_orthog(self):
        a = np.array([1, 0, 1])
        b = np.array([0, 2, 0])

        self.assertEqual(distance.cos_sim(a, b), 0)

    def test_cos_sim_error_size(self):
        a = np.array([1, 0, 1])
        b = np.array([0, 2, 0, 0])

        with self.assertRaises(ValueError):
            distance.cos_sim(a, b)

