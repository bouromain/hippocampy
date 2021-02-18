import numpy as np
import bottleneck as bn


def gen_pc_gauss_oi(t,pos, centers, sigma = [1,1], amp =1,omega = 8.1, phi = 0):
    """
    generate a place cells using a gaussian envelope modulated
     by a oscilatory interference

    see methods of Monsalve-Mercado Roudi 2019
    """

    G = gaussian_envelope(pos, centers, sigma = [1,1])
    # make the oscilation 
    L = np.cos( (omega * 2 * np.pi) * t + phi) + 1 / 2

    return amp * G * L 


def gaussian_envelope(pos, centers, sigma = [1,1], A = 1):
    """
    f(x,y) = 
    A\exp(- (\frac{(x(t)-x_{0})^{2}}{2\sigma_{x}^{2}} + \frac{(y(t)-y_{0})^{2}}{2\sigma_{y}^{2}}) )
    """
    
    # prepare variables
    pos = np.asarray(pos)
    centers = np.asarray(centers)
    sigma = np.asarray(sigma)
   
    num = (pos - centers[:,None]) ** 2
    denum = 2 * sigma[:,None]

    expo = num / denum

    return A * np.exp( - bn.nansum(expo,0) ) 