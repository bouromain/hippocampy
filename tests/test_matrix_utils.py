#%% Import
from hippocampy import matrix_utils
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
        assert all(out == np.array([1, 0, 2, 3, 0, 4]))

    def test_labelled(self):
        # test labeled matrix
        a = np.array([1, 0, 2, 2, 0, 3])
        out = matrix_utils.label(a)
        assert all(out == np.array([1, 0, 2, 2, 0, 3]))

    def test_label_nan(self):
        # test matrix with nan
        a = np.array([1, 0, np.nan, 1, 0, 1])
        out = matrix_utils.label(a)
        assert all(out == np.array([1, 0, 2, 3, 0, 4]))


class TestRemoveSmallObject(unittest.TestCase):
    def test_remove_small_objects(self):
        # test matrix with nan
        a = np.array([1, 0, np.nan, 1, 1, 1])
        out = matrix_utils.remove_small_objects(a, 2)

        assert all(out == np.array([False, False, False, True, True, True]))

    def test_remove_small_objects_axis(self):
        # test matrix with nan
        a = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]])
        out = matrix_utils.remove_small_objects(a, 2)
        exp_out = np.array(
            [
                [False, False, False, False, False, False],
                [True, True, False, True, True, True],
            ]
        )

        assert (exp_out == out).all()


class TestFirstTrue(unittest.TestCase):
    def test_first_true_row(self):
        a = matrix_utils.first_true([0, 0, 1, 1, 0, 1, 1, 1])
        res = np.array([[False, False, True, False, False, True, False, False]])
        assert (a == res).all()

    def test_first_true_col(self):
        v = np.array([0, 0, 1, 1, 0, 1, 1, 1], ndmin=2).T
        a = matrix_utils.first_true(v, axis=0)
        res = np.array([[False, False, True, False, False, True, False, False]])
        assert (a.T == res).all()


class TestMeanAt(unittest.TestCase):
    def test_mean_at_non_zero(self):
        # test for non zero indexing
        i1 = np.array([1, 1, 2, 3, 3, 3, 4, 5, 5])
        v1 = np.array([1, 1, 0.5, 3, 6, 3, 1, 5, 50])
        out = matrix_utils.mean_at(i1, v1)
        assert (
            np.testing.assert_equal(out, np.array([np.nan, 1.0, 0.5, 4.0, 1.0, 27.5]))
            == None
        )

    def test_mean_at_missing_index(self):
        # test for non zero indexing
        i1 = np.array([1, 1, 2, 3, 3, 3, 4, 8, 8])
        v1 = np.array([1, 1, 0.5, 3, 6, 3, 1, 5, 50])
        out = matrix_utils.mean_at(i1, v1)
        assert (
            np.testing.assert_equal(
                out,
                np.array([np.nan, 1.0, 0.5, 4.0, 1.0, np.nan, np.nan, np.nan, 27.5]),
            )
            == None
        )


# %%
