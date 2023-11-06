# %% Import
import warnings
from hippocampy import matrix_utils
import unittest
import numpy as np


# %%
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


class TestSmooth1D(unittest.TestCase):
    # fmt: off
    def test_smooth(self):
        v = np.array([[0,0,0,0,1,0,0,0,0,1],[0,0,0,0,1,0,0,0,0,1]])
        v_s = matrix_utils.smooth_1d(v,2)

        exp_v = np.array([[0., 0., 0., 0.32710442, 0.34579116,0.32710442, 0., 0., 0.32710442, 0.34579116],
        [0., 0., 0., 0.32710442, 0.34579116,0.32710442, 0., 0., 0.32710442, 0.34579116]])
    # fmt: on
        assert (np.testing.assert_array_almost_equal(v_s,exp_v) == None)
    
    def test_smooth_float(self):
        v = np.array([[0,0,0,0,1,0,0,0,0,1],[0,0,0,0,1,0,0,0,0,1]])

        exp_v = np.array([[0., 0., 0., 0.32710442, 0.34579116,0.32710442, 0., 0., 0.32710442, 0.34579116],
        [0., 0., 0., 0.32710442, 0.34579116,0.32710442, 0., 0., 0.32710442, 0.34579116]])
    # fmt: on
        with self.assertWarns(Warning):
            v_s = matrix_utils.smooth_1d(v,2.1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v_s = matrix_utils.smooth_1d(v,2.1)
            assert (np.testing.assert_array_almost_equal(v_s,exp_v) == None)


class TestCorrMat(unittest.TestCase):
    def test_vect_corr(self):
        # this example is taken from numpy corrcoef function docstring
        rng = np.random.default_rng(seed=42)
        xarr = rng.random((3, 3))

        a = matrix_utils.corr_mat(xarr[0, :], xarr[1, :], axis=1)
        b = np.corrcoef(xarr[0, :], xarr[1, :])[0, 1]

        self.assertAlmostEqual(a, b, places=9)

    def test_vect_rows(self):
        # this example is taken from numpy corrcoef function docstring
        rng = np.random.default_rng(seed=42)
        xarr = rng.random((3, 3))

        a = matrix_utils.corr_mat(xarr, axis=1)
        b = np.corrcoef(xarr)

        np.testing.assert_array_almost_equal(a, b)

    def test_vect_col(self):
        # this example is taken from numpy corrcoef function docstring
        rng = np.random.default_rng(seed=42)
        xarr = rng.random((3, 3))

        a = matrix_utils.corr_mat(xarr, axis=0)
        b = np.corrcoef(xarr.T)

        np.testing.assert_array_almost_equal(a, b.T)


class Testdiagonality(unittest.TestCase):
    def test_mat(self):
        a = np.array([[6, 1, 0], [1, 5, 2], [1, 3, 6]])
        d = matrix_utils.diagonality(a)
        self.assertAlmostEqual(d, 0.674149, places=6)

    def test_nan(self):
        a_n = np.array([[6, 1, np.nan], [1, 5, 2], [1, 3, 6]])
        with self.assertRaises(ValueError):
            matrix_utils.diagonality(a_n)

    def test_tridiag(self):
        a = np.array(
            [
                [2, 1, 0, 0, 0],
                [1, 3, 2, 0, 0],
                [0, 2, 3, 4, 0],
                [0, 0, 1, 2, 3],
                [0, 0, 0, 1, 1],
            ]
        )
        d = matrix_utils.diagonality(a)
        self.assertAlmostEqual(d, 0.812383, places=6)

    def test_transpose(self):
        a = np.array([[6, 1, 0], [1, 5, 2], [1, 3, 6]])
        d = matrix_utils.diagonality(a)
        d_t = matrix_utils.diagonality(a.T)
        self.assertEqual(d, d_t)

    def test_opositdiag(self):
        a = np.array([[6, 1, 0], [1, 5, 2], [1, 3, 6]])
        d = matrix_utils.diagonality(a)
        d_op = matrix_utils.diagonality(np.rot90(a))
        self.assertEqual(d, -d_op)


# Define a test case for invert_indices function
class TestInvertIndices(unittest.TestCase):
    def test_basic_functionality(self):
        np.testing.assert_array_almost_equal(
            matrix_utils.invert_indices([1, 3, 5], 10), [0, 2, 4, 6, 7, 8, 9]
        )
        np.testing.assert_array_almost_equal(
            matrix_utils.invert_indices([0, 2, 4, 6, 8], 10), [1, 3, 5, 7, 9]
        )

    def test_empty_indices(self):
        self.assertTrue(
            np.array_equal(matrix_utils.invert_indices([], 10), np.arange(10))
        )

    def test_full_range(self):
        self.assertEqual(len(matrix_utils.invert_indices(np.arange(10), 10)), 0)

    def test_single_index(self):
        self.assertTrue(
            np.array_equal(
                matrix_utils.invert_indices([5], 10), [0, 1, 2, 3, 4, 6, 7, 8, 9]
            )
        )

    def test_index_out_of_bounds(self):
        with self.assertRaises(ValueError):
            matrix_utils.invert_indices([10], 10)
        with self.assertRaises(ValueError):
            matrix_utils.invert_indices([-1], 10)

    def test_duplicate_indices(self):
        # Assuming function should handle duplicates by ignoring them
        self.assertTrue(
            np.array_equal(
                matrix_utils.invert_indices([1, 1, 3, 5], 10), [0, 2, 4, 6, 7, 8, 9]
            )
        )

    def test_large_range(self):
        # This will also test the function's performance on a larger scale
        size = 10000
        idx = np.random.choice(size, size // 2, replace=False)
        result = matrix_utils.invert_indices(idx, size)
        # Check that there are no indices in result that are in idx
        self.assertTrue(np.intersect1d(result, idx).size == 0)
        # Check that the result and idx together make up the full range
        self.assertEqual(len(np.union1d(result, idx)), size)


# %%
