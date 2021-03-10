import numpy as np
import matplotlib.pyplot as plt
import hippocampy.model.gen_pc as mdpc
import hippocampy as hp
import bottleneck as bn

x_pos, t = mdpc.gen_lin_path(100)

c1 = mdpc.gen_pc_gauss_oi(t, x_pos, 75, omega_theta=11, amplitude=20)
s1 = mdpc.inhomogeneous_poisson_process(c1, t)

c2 = mdpc.gen_pc_gauss_oi(t, x_pos, 70, omega_theta=11, amplitude=20)
s2 = mdpc.inhomogeneous_poisson_process(c2, t)

# generate theta
omega = 8  # theta frequency
phi = 0  # theta phase at zero
theta = np.cos((omega * 2 * np.pi) * t + phi) + 1 / 2
theta_p, _ = hp.filterSig.hilbertPhase(theta)

s1_idx = hp.utils.nearest_idx(t, s1)
s2_idx = hp.utils.nearest_idx(t, s1)


plt.plot(x_pos[s1_idx], theta_p[s1_idx], ".")


plt.subplot(311)
plt.plot(t, x_pos)
plt.xlim((53, 54))

plt.subplot(312)
plt.plot(t, c1)
# plt.plot(t, c2, "g")
plt.xlim((53, 54))

plt.subplot(313)
plt.plot([s1, s1], [theta_p[s1_idx], theta_p[s1_idx] - 2 * np.pi], "or")
plt.xlim((53, 54))


plt.plot(
    [x_pos[s1_idx], x_pos[s1_idx]], [theta_p[s1_idx], theta_p[s1_idx] - 2 * np.pi], "or"
)


###########################################
import numpy as np
import matplotlib.pyplot as plt
import hippocampy.model.gen_pc as mdpc
import hippocampy as hp
import bottleneck as bn

pos, t = mdpc.gen_lin_path(100, Fs=100)
centers = 100
sigma = 15
omega_theta = 8

# first define the gaussian envelope of the place field
G = mdpc.gaussian_envelope(pos, centers, sigma=sigma)

theta = np.cos((omega_theta * 2 * np.pi) * t)
theta_p, _ = hp.filterSig.hilbertPhase(theta)

# calculate variables
R = sigma * np.sqrt(2 * np.log(10))  # distance from place field center where
# firing decrased to 10%
delta_t = bn.nanmedian(np.diff(t))
delta_x = bn.nanmedian(np.diff(pos))
speed = np.abs(delta_x / delta_t)
# omega_cell = omega_theta + (np.pi / R) * speed
omega_cell = 1 / (1 / 1 / omega_theta * (1 - 0.06 * (20 / sigma)))

# now find the indexes of entries in the fields
idx_entries = G >= np.max(G) * 0.1
idx_entries = np.hstack((0, np.diff(idx_entries.astype(int)) == 1))
entries_cum = np.cumsum(idx_entries)

L = np.cos((omega_cell * 2 * np.pi) * t + 0)
L_phase, _ = hp.filterSig.hilbertPhase(L)

delta_phase = L_phase[idx_entries.astype(bool)] - theta_p[idx_entries.astype(bool)]

phase_offset = np.ones_like(t)

for it, val in enumerate(delta_phase):
    phase_offset[entries_cum == it] = -val

# L = (np.cos((omega_cell * 2 * np.pi) * t + phase_offset) + 1 ) / 2

plt.plot(t, G * L)
plt.xlim((54, 56))
plt.plot(t, theta)

plt.plot(t, theta)
plt.plot(t, L)
plt.xlim((54, 55))
