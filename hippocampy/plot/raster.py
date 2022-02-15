import matplotlib.pyplot as plt
import bottleneck as bn
import numpy as np
from hippocampy.assemblies import calc_template


def sort_F(F: np.array):
    T, _ = calc_template(F)
