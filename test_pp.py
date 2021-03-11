import numpy as np
import matplotlib.pyplot as plt
import hippocampy.model.gen_pc as mdpc
import hippocampy as hp
import bottleneck as bn

x_pos, t = mdpc.gen_lin_path(500, Fs=100)

c1 = mdpc.gen_pc_gauss_oi(t, x_pos, 75, amplitude=12)
s1 = mdpc.inhomogeneous_poisson_process(c1, t)

c2 = mdpc.gen_pc_gauss_oi(t, x_pos, 70, amplitude=12)
s2 = mdpc.inhomogeneous_poisson_process(c2, t)

# generate theta
omega = 8  # theta frequency
phi = 0  # theta phase at zero
theta = np.cos((omega * 2 * np.pi) * t + phi) + 1 / 2
theta_p, _ = hp.filterSig.hilbertPhase(theta)

s1_idx = hp.utils.nearest_idx(t, s1)
s2_idx = hp.utils.nearest_idx(t, s2)

# plt.plot(
#     [x_pos[s1_idx], x_pos[s1_idx]], [theta_p[s1_idx], theta_p[s1_idx] - 2 * np.pi], "or"
# )
plt.plot(
    [x_pos[s1_idx], x_pos[s1_idx]], [theta_p[s1_idx], theta_p[s1_idx] + 2 * np.pi], ".r"
)
plt.plot(
    [x_pos[s2_idx], x_pos[s2_idx]], [theta_p[s2_idx], theta_p[s2_idx] + 2 * np.pi], ".k"
)
plt.xlim((60, 80))
