#%% THIS CODE DOES NOT WORK 
from hippocampy import helper
import pytest
import unittest
import numpy as np
#%%
class TestHelper(unittest.TestCase):
    def test_isradians(self):
        a = np.array([-1,2,3])
        b = np.array([1,4,5])
        c = np.array([1,4,24])

        self.assertAlmostEqual(helper.isradians(a), 1)
        self.assertEqual(helper.isradians(b), 2)
        self.assertEqual(helper.isradians(c), 0)


