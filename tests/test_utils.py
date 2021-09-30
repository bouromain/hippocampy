#%% THIS CODE DOES NOT WORK
from hippocampy.utils.gen_utils import localExtrema, start_stop, value_cross
import unittest
import numpy as np

#%%
class TestUtils(unittest.TestCase):
    def test_local_extrema(self):
        a = np.array([0, 0.1, 1.2, 2.5, 3.7, 2.1, 1.3, 0, 1.9])

        self.assertEqual(localExtrema(a), np.array([4]))
        self.assertEqual(localExtrema(a, method="min"), np.array([7]))

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


# %%
