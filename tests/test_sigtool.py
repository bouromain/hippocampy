# %% Import
import warnings
from hippocampy.sig_tool import xcorr
import unittest
import numpy as np


# %%
# class TestLabel(unittest.TestCase):
#     def test_xcorr(self):
#         # test 1 vector
#         x = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1])
#         out = sig_tool.xcorr(x)
#         exp_out = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 5, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0])

#         assert all(out == exp_out)
#     def test_xcorr_xy(self):
#         # test two vectors
#         x = np.array([0,0,0,1,1,1])
#         y = np.array([0,1,1,0,1,0])
#         out = sig_tool.xcorr(x,y)
#         exp_out = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 1, 0])

#         assert all(out == exp_out)

#     def test_xcorr_xy_maxlag(self):
#         # test two vectors and maxlag
#         x = np.array([0,0,0,1,1,1])
#         y = np.array([0,1,1,0,1,0])
#         out = sig_tool.xcorr(x,y)
#         exp_out = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 1, 0])

#         assert all(out == exp_out)


class TestXCorrFunction(unittest.TestCase):
    def test_biased_scaling(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = xcorr(x, y, scale="biased")
        expected = np.array([0.2, 0.8, 2.0, 4.0, 7.0, 8.8, 9.2, 8.0, 5.0])
        self.assertTrue(np.allclose(result, expected))

    def test_unbiased_scaling(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = xcorr(x, y, scale="unbiased")
        expected = np.array([1.0, 2.0, 3.33, 5.0, 7.0, 11.0, 15.33, 20.0, 25.0])
        self.assertTrue(np.allclose(result, expected, atol=10e2))

    def test_none_scaling(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = xcorr(x, y, scale=None)
        expected = np.array([1, 4, 10, 20, 35, 44, 46, 40, 25])
        self.assertTrue(np.allclose(result, expected))

    def test_coeff_scaling(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = xcorr(x, y, scale="coeff")
        expected = np.array(
            [
                0.018,
                0.072,
                0.181,
                0.363,
                0.636,
                0.8,
                0.836,
                0.727,
                0.454,
            ]
        )
        self.assertTrue(np.allclose(result, expected, atol=10e2))

    def test_maxlag(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = xcorr(x, y, scale="biased", maxlag=2)
        expected = np.array([2.0, 4.0, 7.0, 8.8, 9.2])
        self.assertTrue(np.allclose(result, expected))

    def test_invalid_maxlag(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        with self.assertRaises(ValueError):
            xcorr(x, y, scale="biased", maxlag=10)
