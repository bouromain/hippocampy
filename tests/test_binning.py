from hippocampy import binning
import unittest
import numpy as np


class TestBinning(unittest.TestCase):
    def test_1d_rate_map(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        spk = np.array([2, 2, 12, 13, 16])
        bins = np.arange(0, 11)

        rate, act, occ = binning.rate_map(x, spk, bins=bins, fs=1)

        good_occ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        good_act = np.array([0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        good_rate = np.array([0.0, 0.0, 1.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])

        assert all(occ == good_occ)
        assert all(act == good_act)
        assert all(rate == good_rate)

    def test_no_occ(self):
        # missing value for fifth bin
        x = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 0, 1, 2, 3, 4, 6, 7, 8, 9])
        spk = np.array([2, 2, 11, 12, 14])
        bins = np.arange(0, 11)

        rate, act, occ = binning.rate_map(x, spk, bins=bins, fs=1)

        good_occ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0])
        good_act = np.array([0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        good_rate = np.array([0.0, 0.0, 1.5, 0.5, 0.0, np.nan, 0.5, 0.0, 0.0, 0.0])

        assert np.testing.assert_equal(occ, good_occ) == None
        assert np.testing.assert_equal(act, good_act) == None
        assert np.testing.assert_equal(rate, good_rate) == None

    def test_no_occ(self):
        # missing value for fifth bin
        x = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 0, 1, 2, 3, 4, 6, 7, 8, 9, 15, 16])
        spk = np.array([2, 2, 11, 12, 14])
        bins = np.arange(0, 11)

        rate, act, occ = binning.rate_map(x, spk, bins=bins, fs=1)

        good_occ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0])
        good_act = np.array([0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        good_rate = np.array([0.0, 0.0, 1.5, 0.5, 0.0, np.nan, 0.5, 0.0, 0.0, 0.0])

        assert np.testing.assert_equal(occ, good_occ) == None
        assert np.testing.assert_equal(act, good_act) == None
        assert np.testing.assert_equal(rate, good_rate) == None

    def test_out_of_bins(self):
        # missing value for fifth bin
        x = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 0, 1, 2, 3, 4, 6, 7, 8, 9, 15, 16, -1])
        spk = np.array([2, 2, 11, 12, 14])
        bins = np.arange(0, 11)

        rate, act, occ = binning.rate_map(x, spk, bins=bins, fs=1)

        good_occ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0])
        good_act = np.array([0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        good_rate = np.array([0.0, 0.0, 1.5, 0.5, 0.0, np.nan, 0.5, 0.0, 0.0, 0.0])

        assert np.testing.assert_equal(occ, good_occ) == None
        assert np.testing.assert_equal(act, good_act) == None
        assert np.testing.assert_equal(rate, good_rate) == None

    def test_rate_map_mean(self):
        x = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 0, 1, 2, 3, 4, 6, 7, 8, 9])
        spk = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0])
        bins = np.arange(0, 11)
        rate, act, occ = binning.rate_map(x, spk, bins=bins, fs=1, method="continuous")

        good_occ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0])
        good_act = np.array([0.0, 0.5, 2.0, 0.5, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0])
        good_rate = np.array([0.0, 0.5, 2.0, 0.5, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0])

        assert np.testing.assert_equal(occ, good_occ) == None
        assert np.testing.assert_equal(act, good_act) == None
        assert np.testing.assert_equal(rate, good_rate) == None

    def test_2D_map(self):
        # test multidim bining
        x = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ]
        )

        lap = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            ]
        )
        X = np.vstack((x.ravel(), lap.ravel()))
        spk = np.array([2, 2, 12, 13, 16])
        bins = (np.arange(0, 11), [0, 1, 2, 3])

        rate, act, occ = binning.rate_map(X.T, spk, bins=bins, fs=1)

        good_rate = np.array(
            [
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        good_act = np.array(
            [
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        good_occ = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        assert np.testing.assert_equal(occ, good_occ.T) == None
        assert np.testing.assert_equal(act, good_act.T) == None
        assert np.testing.assert_equal(rate, good_rate.T) == None
