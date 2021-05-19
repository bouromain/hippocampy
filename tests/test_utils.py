#%% THIS CODE DOES NOT WORK
from hippocampy import utils
import pytest
import unittest
import numpy as np

#%%
class TestUtils(unittest.TestCase):
    def test_local_extrema(self):
        a = np.array([0, 0.1, 1.2, 2.5, 3.7, 2.1, 1.3, 0, 1.9])

        self.assertEqual(utils.localExtrema(a), np.array([4]))
        self.assertEqual(utils.localExtrema(a, method="min"), np.array([7]))
