import bottleneck as bn
import hippocampy as hp
import numpy as np


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
    generate intensity function with a gaussian envelope
    with given centers and sigma
    """

    # prepare variables
    pos = np.asarray(pos)
    centers = np.atleast_2d(centers)
    sigma = np.atleast_2d(sigma)

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

    for i, s in enumerate(spike_time):
        m = np.empty_like(s, dtype=bool)
        spike_time_diff = np.diff(s)
        m[0] = True
        m[1:] = spike_time_diff > refractory_period
        spike_time[i] = s[m]

    return spike_time


def homogeneous_poisson_process(
    lam: np.ndarray = None,
    t_0: float = 0,
    t_end: float = 10,
    delta_t: float = 1e-3,
    refractory_period: float = None,
    method: str = "uniform",
):
    """
    homogeneous_poisson_process [summary]

    Parameters
    ----------
    lam : np.ndarray, optional
        Average rate value or vector, by default None
    t_0 : float, optional
        start time in second, by default 0
    t_end : float, optional
        end time in seconds, by default 10
    delta_t : float, optional
        sampling period in second, by default 1e-3
    refractory_period : float, optional
        length of the refractory period in second, by default None
    method : str, optional
        method to generate the poisson process. Can be "uniform" or 
        "exponential", by default "uniform"

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]

    Reference
    ---------
    http://www.cns.nyu.edu/~david/handouts/poisson.pdf
    """

    if lam is None:
        lam = np.random.randint(1, 15, (10, 1))

    n_samples = int((t_end - t_0) / delta_t)
    n_lambda = len(lam)

    if method == "uniform":
        uniform = np.random.uniform(size=(n_lambda, n_samples))
        idx_spk = uniform <= lam[:, None] * delta_t

        spk_time = [np.nonzero(st)[0] * delta_t for st in idx_spk]

    elif method == "exponential":
        # calculate the number of expected spikes
        expected_spikes = np.ceil((t_end - t_0) * lam)
        # draw inter spikes interval (isi) from a random exponential distribution
        spk_time = [None] * n_lambda

        for it_l, (l, n_spk) in enumerate(zip(lam, expected_spikes)):
            spk_isi = np.random.exponential(1 / l, size=(1, int(n_spk)))
            spk_isi = np.cumsum(spk_isi)
            # correct spikes time with t_0 and  t_end
            spk_isi = spk_isi + t_0
            spk_isi = spk_isi[spk_isi < t_end]
            spk_time[it_l] = spk_isi
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

    Parameters
    ----------
    - rate: 
        firing rate vector in time 
    - time: 
        time vector corresponding to the rate vector
    - refractory period: 
        refractory period of spike generation.
        spike with isi < to refractory period will be removed

    Returns
    -------
    - spk_time: time of spikes in the same unit than the input time vector


    Reference:
    http://www.cns.nyu.edu/~david/handouts/poisson.pdf
    inhomogeneous_poisson_process from:
    https://github.com/NeuralEnsemble/elephant/blob/master/elephant/spike_train_generation.py
    https://gitlab.com/e.reifenstein/synaptic-learning-rules-for-sequence-learning/-/blob/master/Code_for_SynapticLearningRulesForSequenceLearning.py

    """
    rate = np.asarray(rate)
    rate_max = bn.nanmax(rate, axis=1)
    delta_t = bn.nanmedian(np.diff(time))

    uniform = np.random.uniform(size=rate.shape) * rate_max[:, None]
    # reifenstein do that insted, I will compare the two
    # avg_prob = 1 - np.exp(-avg_rate * delta)
    avg_prob = 1 - np.exp(-rate * delta_t)
    uniform[uniform < avg_prob] = 0

    return uniform
