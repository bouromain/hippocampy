#%% THIS CODE DOES NOT WORK
from hippocampy.utils.gen_utils import (
    localExtrema,
    nearest_even,
    nearest_odd,
    start_stop,
    value_cross,
    nearest_idx,
    calc_dt,
)
import unittest
import numpy as np

#%%
class TestUtils(unittest.TestCase):
    def test_local_extrema(self):
        a = np.array([0, 0.1, 1.2, 2.5, 3.7, 2.1, 1.3, 0, 1.9])

        self.assertEqual(localExtrema(a), np.array([4]))
        self.assertEqual(localExtrema(a, method="min"), np.array([7]))
        self.assertTrue((localExtrema(a, method="all") == np.array([4, 7])).all())

    def test_start_stop(self):
        B = np.array([True, False, False, True, True, False, True])
        exp_start = np.array([[True, False, False, True, False, False, True]])
        exp_stop = np.array([[True, False, False, False, True, False, True]])
        start, stop = start_stop(B, axis=-1)

        assert (exp_start == start).all()
        assert (exp_stop == stop).all()

    def test_start_stop_2D_transposed(self):
        B = np.array(
            [
                [True, False, False, True, True, False, True],
                [True, False, False, True, True, False, True],
            ]
        )

        exp_start = np.array(
            [
                [True, False, False, True, False, False, True],
                [True, False, False, True, False, False, True],
            ]
        ).T
        exp_stop = np.array(
            [
                [True, False, False, False, True, False, True],
                [True, False, False, False, True, False, True],
            ]
        ).T
        start, stop = start_stop(B.T, axis=0)

        assert (exp_start == start).all()
        assert (exp_stop == stop).all()

    def test_value_cross(self):
        a = np.array([1, 2, 3, 6, 2, 4])
        up, down = value_cross(a, 2.5)
        exp_up = np.array([[False, False, True, False, False, True]])
        exp_down = np.array([[False, False, False, True, False, True]])
        assert (exp_up == up).all()
        assert (exp_down == down).all()

    def test_nearest_idx_sorted(self):
        a = nearest_idx([0.5, 1.6, 2.2, 3.4, 4.5], np.arange(5))

        exp_a = np.array([0, 1, 2, 3, 4])
        self.assertTrue((a == exp_a).all())

    def test_nearest_idx_unsorted(self):
        a = nearest_idx([0.5, 1.6, 2.2, 3.4, 4.5], np.arange(5), method="unsorted")

        exp_a = np.array([0, 2, 2, 3, 4])
        self.assertTrue((a == exp_a).all())

    def test_calc_dt(self):
        t = np.array([0.01, 0.02, 0.03, 0.05, 0.06, 0.14])
        dt = calc_dt(t)
        self.assertAlmostEqual(dt, 0.01, places=2)

    def test_nearest_odd(self):
        a = nearest_odd([1.1, 1.9, 2.9, 2, -1.9])
        exp_a = np.array([1.0, 1.0, 3.0, 3.0, -1.0])
        self.assertTrue((a == exp_a).all())

    def test_nearest_even(self):
        a = nearest_even([1, 0.9, 3.09, -0.5, -1.9])
        exp_a = np.array([0.0, 0.0, 4.0, -0.0, -2.0])
        self.assertTrue((a == exp_a).all())


# %%
