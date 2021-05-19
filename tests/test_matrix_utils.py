#%% Import
from hippocampy import matrix_utils
import pytest
import unittest
import numpy as np

#%%
class TestLabel(unittest.TestCase):
    def test_label_bool(self):
        # test boolean vector
        a = np.array([True, False, True, True, False, True])
        out = matrix_utils.label(a)
        assert all(out == np.array([1, 0, 2, 2, 0, 3]))

    def test_label_int(self):
        # test int matrix
        a = np.array([1, 0, 1, 1, 0, 1])
        out = matrix_utils.label(a)
        assert all(out == np.array([1, 0, 2, 2, 0, 3]))

    def test_label_int2(self):
        # test random int matrix
        a = np.array([8, 0, 4, 3, 0, 2])
        out = matrix_utils.label(a)
        assert all(out == np.array([1, 0, 2, 2, 0, 3]))

    def test_labelled(self):
        # test labeled matrix
        a = np.array([1, 0, 2, 2, 0, 3])
        out = matrix_utils.label(a)
        assert all(out == np.array([1, 0, 2, 2, 0, 3]))

    def test_label_nan(self):
        # test matrix with nan
        a = np.array([1, 0, np.nan, 1, 0, 1])
        out = matrix_utils.label(a)
        assert all(out == np.array([1, 0, 2, 2, 0, 3]))


class TestRemoveSmallObject(unittest.TestCase):
    def test_remove_small_objects(self):
        # test matrix with nan
        a = np.array([1, 0, np.nan, 1, 0, 1])
        out = matrix_utils.remove_small_objects(a, 2)

        assert all(out == np.array([False, False, True, True, False, False]))
