import unittest
import numpy as np
from hippocampy.stats import distance


class TestCosSim(unittest.TestCase):
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


class TestEuclidian(unittest.TestCase):
    def test_euclidian(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])

        self.assertAlmostEqual(distance.pairwise_euclidian(a, b), 0, places=1)

    def test_euclidian_pairwise(self):
        a = np.array([[1, 2, 3], [1, 5, 3], [1, 2, 3]])
        b = np.array([[1, 2, 3], [2, 2, 2]])

        out = distance.pairwise_euclidian(a, b)

        result = np.array([[0.0, 1.41421356], [3.0, 3.31662479], [0.0, 1.41421356]])

        assert np.allclose(result, out, atol=1e-3)

