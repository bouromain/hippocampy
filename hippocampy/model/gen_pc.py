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

def spk_from_if(rate,refractory_period=None):
    '''
    https://elephant.readthedocs.io/en/latest/_modules/elephant/spike_train_generation.html
    see function:
    inhomogeneous_poisson_process
    '''
    avg_prob = 1 - np.exp(-rate)
    rand_var = np.random.uniform(size=rate.size)

    return avg_prob >= rand_var

