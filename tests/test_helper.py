#%% THIS CODE DOES NOT WORK 
from hippocampy import helper
import pytest
import unittest
import numpy as np
#%%
class TestHelper(unittest.TestCase):
    def test_local_extrema(self):
        a = np.array([0,0,1,2,3,2,1,0,1])

        self.assertEqual(helper.localExtrema(a), np.array([3]))
        self.assertEqual(helper.localExtrema(a, method='min'), np.array([6]))