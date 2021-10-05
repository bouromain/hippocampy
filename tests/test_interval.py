import hippocampy as hp
import unittest
import numpy as np

from hippocampy.core.Iv import Iv


class TestIv(unittest.TestCase):
    def test_in(self):
        a = Iv([[1, 3], [6, 10]])
        b = [2, 3]
        c = [2.5, 3]
        c_iv = Iv([2.5, 3])

        d = [0, 1]

        # check with int intervals
        self.assertIn(b, a)
        # check with float
        self.assertIn(c, a)
        # check Iv in Iv
        self.assertIn(c_iv, a)
        # check Iv out
        self.assertNotIn(d, a)

    def test_eq(self):
        f = Iv([[2, 4], [6, 8], [10, 12]])
        e = Iv([[2, 4], [6, 8], [10, 12]])

        self.assertTrue(f == e)

    def test_neq_domain(self):
        f = Iv([[2, 4], [6, 8], [10, 12]])
        e = Iv([[2, 4], [6, 8], [10, 12]], domain=[0, 10000])

        self.assertFalse(f == e)

    def test_merge(self):
        b = Iv([[10, 12], [6, 8], [2, 4]])
        b = b.merge()

        exp_b = Iv([[2, 4], [6, 8], [10, 12]])

        self.assertTrue(b == exp_b)

    def test_merge_overlap(self):
        b = Iv([[3, 12], [12, 17], [-5, 3]])
        b = b.merge()

        exp_b = Iv([[-5, 17]])

        self.assertTrue(b == exp_b)

    def test_union(self):
        a = Iv([[3, 12], [12, 17]])
        b = Iv([[-5, 3]])
        c = a | b
        c2 = a.union(b)

        exp_c = Iv([[-5, 17]])

        self.assertTrue(c == exp_c)
        self.assertTrue(c2 == exp_c)

