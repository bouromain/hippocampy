import hippocampy as hp
import pytest
import unittest
import numpy as np


class TestIv(unittest.TestCase):
    def test_in(self):
        a = hp.Interval([[1, 3], [6, 10]])
        b = [2, 3]
        c = [2.5, 3]
        c_iv = hp.Interval([2.5, 3])

        d = [0, 1]

        # check with int intervals
        self.assertIn(b, a)
        # check with float
        self.assertIn(c, a)
        # check Iv in Iv
        self.assertIn(c_iv, a)
        # check Iv out
        self.assertNotIn(d, a)
