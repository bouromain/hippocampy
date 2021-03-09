import numpy as np
import bottleneck as bn


def gen_pc_gauss_oi(t,pos, centers, sigma= 10, amp= 1,omega= 8.1, phi = 0):
    """
    generate a place cells using a gaussian envelope modulated
    by a oscilatory interference

    see methods of Monsalve-Mercado Roudi 2019
    """

    G = gaussian_envelope(pos, centers, sigma = [1,1])

    
    # make the oscilation 
    L = np.cos( (omega * 2 * np.pi) * t + phi) + 1 / 2

    return amp * G * L 


def gaussian_envelope(pos, centers, sigma = 10, A = 1):
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

    return A * np.exp( - bn.nansum(expo,0) ) 


def gen_lin_path(t_max, v=20, range_x=200,Fs=50,up_down=True):
    """
    generate 
    """
 
    t = np.linspace(0,t_max,t_max*Fs)
    d = v * t

    if up_down:
        d = np.mod(d,range_x * 2 ) - range_x
        x = np.abs(d)
    else:
        x = np.mod(d,range_x)

    return x,t


def homogeneous_poisson_process(lam, t_0=0, t_end=10, delta_t=1e-3, refractory_period=None, method='uniform'):
    '''
    Reference:
    http://www.cns.nyu.edu/~david/handouts/poisson.pdf
    '''

    if method == 'uniform':
        time = np.arange(t_0,t_end,delta_t)
        uniform = np.random.uniform(size=time.size)
        idx_spk = uniform <= lam * delta_t

        spk_time = time[idx_spk]

    elif method == 'exponential':
        # calculate the number of expected spikes 
        expected_spikes = np.ceil((t_end - t_0) * lam)
        # draw inter spikes interval (isi) from a random exponential distribution
        spk_isi = [np.random.exponential( 1/lam) for i in np.arange(expected_spikes) ]
        spk_time = np.cumsum(spk_isi)

        # correct spikes time with t_0 and  t_end
        spk_time = spk_time + t_0
        spk_time = spk_time[spk_time < t_end]

    else:
        raise NotImplementedError('Method should be uniform of exponential')


    if refractory_period is not None:
        raise NotImplementedError('I shoudl code that ')
    
    return spk_time


def inhomogeneous_poisson_process(rate, time, refractory_period=None):
    '''
    https://elephant.readthedocs.io/en/latest/_modules/elephant/spike_train_generation.html
    https://github.com/NeuralEnsemble/elephant/blob/master/elephant/spike_train_generation.py
    around line 500
    see function:
    inhomogeneous_poisson_process

    Reference:
    http://www.cns.nyu.edu/~david/handouts/poisson.pdf

    '''
    
    rate_max = np.max(rate)
    t_0 = np.min(time)
    t_end = np.max(time)
    delta_t = bn.median(diff(time))

    hpp = homogeneous_poisson_process(rate_max,t_0=t_0, t_end==t_end, delta_t=delta_t)

    uniform = np.random.uniform(size=hpp.size) * rate_max
    spk_time = time[uniform < hpp]
    
    return spk_time



