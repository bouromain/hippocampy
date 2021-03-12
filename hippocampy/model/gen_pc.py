import numpy as np
import bottleneck as bn
import hippocampy as hp


def gen_pc_gauss_oi(t, pos, centers, sigma=10, amplitude=1, omega_theta=8):
    """
    generate a place cells using a gaussian envelope modulated
    by a oscillatory interference

    see methods of
    Monsalve-Mercado Roudi 2017
    Leibold 2017

    """

    # first define the gaussian envelope of the place field
    G = gaussian_envelope(pos, centers, sigma=sigma)

    theta = np.cos((omega_theta * 2 * np.pi) * t)
    theta_p, _ = hp.filterSig.hilbertPhase(theta)

    # calculate variables
    # R = sigma * np.sqrt(2 * np.log(10))  # distance from place field center where
    # omega_cell = omega_theta + (np.pi / R) * speed
    omega_cell = omega_theta / (1 - 0.06 * (20 / sigma))  # should be 0.06

    # now find the indexes of entries in the fields
    idx_entries = G >= np.max(G) * 0.1
    idx_entries = np.hstack((0, np.diff(idx_entries.astype(int)) == 1))
    entries_cum = np.cumsum(idx_entries)

    theta_p_u = np.unwrap(theta_p)
    delta_phase = (
        theta_p_u[idx_entries.astype(bool)] * omega_cell / omega_theta
    ) - theta_p_u[idx_entries.astype(bool)]
    delta_phase = np.mod(delta_phase, 2 * np.pi)

    phase_offset = np.ones_like(t)
    for it, val in enumerate(delta_phase):
        phase_offset[entries_cum == it + 1] = val

    L = (np.cos((omega_cell * 2 * np.pi) * t - phase_offset) + 1) / 2

    return amplitude * G * L


def gaussian_envelope(pos, centers, sigma=10, amplitude=1):
    """
    f(x,y) =
    A\exp(- (\frac{(x(t)-x_{0})^{2}}{2\sigma_{x}^{2}} + \frac{(y(t)-y_{0})^{2}}{2\sigma_{y}^{2}}) )
    """

    # prepare variables
    pos = np.asarray(pos)
    centers = np.atleast_2d(np.asarray(centers))
    sigma = np.atleast_2d(np.asarray(sigma))

    num = (pos - centers) ** 2
    denum = 2 * sigma

    expo = num / denum

    return amplitude * np.exp(-bn.nansum(expo, 0))


def gen_lin_path(time_max, v=20, range_x=200, Fs=50, up_down=True):
    """
    Generate synthetic linear path

    Parameters:
            - time_max: max time of the path
            - v: constant speed
            - range_x: length of the linear track
            - Fs: sampling rate of the path
            - up_down: does the animal goes only up or up and down

    Returns:
            -x: synthetic path
            -t: time vector for the synthetic path
    """

    t = np.linspace(0, time_max, time_max * Fs)
    d = v * t

    if up_down:
        d = np.mod(d, range_x * 2) - range_x
        x = np.abs(d)
    else:
        x = np.mod(d, range_x)

    return x, t


def _remove_refractory_period(spike_time, refractory_period):
    """
    Remove spikes with inter-spike interval smaller than a
    given refractory period
    """
    spike_time_diff = np.diff(spike_time)
    m = np.hstack((True, spike_time_diff > refractory_period))

    return spike_time[m]


def homogeneous_poisson_process(
    lam, t_0=0, t_end=10, delta_t=1e-3, refractory_period=None, method="uniform"
):
    """
    Reference:
    http://www.cns.nyu.edu/~david/handouts/poisson.pdf
    """

    if method == "uniform":
        time = np.arange(t_0, t_end, delta_t)
        uniform = np.random.uniform(size=time.size)
        idx_spk = uniform <= lam * delta_t

        spk_time = time[idx_spk]

    elif method == "exponential":
        # calculate the number of expected spikes
        expected_spikes = np.ceil((t_end - t_0) * lam)
        # draw inter spikes interval (isi) from a random exponential distribution
        spk_isi = [np.random.exponential(1 / lam) for i in np.arange(expected_spikes)]
        spk_time = np.cumsum(spk_isi)

        # correct spikes time with t_0 and  t_end
        spk_time = spk_time + t_0
        spk_time = spk_time[spk_time < t_end]

    else:
        raise NotImplementedError("Method should be uniform of exponential")

    if refractory_period is not None:
        spk_time = _remove_refractory_period(spk_time, refractory_period)

    return spk_time


def inhomogeneous_poisson_process(rate, time, refractory_period=None, method="uniform"):
    """
    This function will simulate an inhomogeneous Poisson Process with a
    time varying firing rate. This inhomogeneous Poisson process with intensity
    function rate(t) is simulated by rejection sampling from a homogeneous Poisson
    process with fixed rate λ = max(rate(t)).
    Event from this distribution are then kept for time t with probability p(t).

    Parameters:
                - rate: firing rate vector
                - time: time vector corresponding to the rate vector
                - refractory period: refractory period of spike generation.
                spike with isi < to refractory period will be removed

    Returns:
                - spk_time: time of spikes in the same unit than the input time vector


    Reference:
    http://www.cns.nyu.edu/~david/handouts/poisson.pdf
    inhomogeneous_poisson_process from:
    https://github.com/NeuralEnsemble/elephant/blob/master/elephant/spike_train_generation.py

    """

    rate_max = np.max(rate)
    t_0 = np.min(time)
    t_end = np.max(time)
    delta_t = bn.median(np.diff(time))

    hpp = homogeneous_poisson_process(
        rate_max, t_0=t_0, t_end=t_end, delta_t=delta_t, method=method
    )

    # find the rate for these values
    rate_i = np.interp(hpp, time, rate)

    # generate random distribution
    uniform = np.random.uniform(size=hpp.size) * rate_max

    # select spikes from this distribution according to their probability
    # Nb: we select according to their probability as uniform is multiplied by
    # rate_max. A rate close to rate_max will have a high probability to be
    # selected, the contrary will be true too
    spk_time = hpp[uniform < rate_i]

    if refractory_period is not None:
        spk_time = _remove_refractory_period(spk_time, refractory_period)

    return spk_time