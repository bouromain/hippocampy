# Code heavily inspired from matlab circ_stat toolbox and python package Pinguin
# by Raphael Vallat. I also addded further function and helpers to work with 
# circular variables 
#
# Provide helper function, descriptive statistics and statistical tests
# for circular data
# 
# References:
# Berens, Philipp. 2009. CircStat: A MATLAB Toolbox for Circular Statistics.
# Journal of Statistical Software, Articles 31 (10): 1–21.
# Vallat, R. (2018). Pingouin: statistics in Python. Journal of Open Source 
# Software, 3(31), 1026, https://doi.org/10.21105/joss.01026
# 
# Rabin, J., Delon, J. & Gousseau, Y. Transportation Distances on the Circle.
# J Math Imaging Vis 41, 147 (2011). https://doi.org/10.1007/s10851-011-0284-0
# https://arxiv.org/pdf/0906.5499v2.pdf
#
# J. Rabin, J. Delon and Y. Gousseau, "Circular Earth Mover’s Distance for the
# comparison of local features," 2008 19th International Conference on Pattern
# Recognition, Tampa, FL, 2008, pp. 1-4, doi: 10.1109/ICPR.2008.4761372.
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.214.4116&rep=rep1&type=pdf
#
# Author: RB 12/11/20

import numpy as np
import bottleneck as bn
from numpy import pi

########################################################################
## Helpers
########################################################################
def isradians(x):
  """
  verify if input vector is in radians and return warning otherwise
  """
  if all(x>=-pi) & all(x<=pi):
      return 1
  elif all(x>=0) & all(x<=2*pi):
      return 2
  else:
      # print("Warning, radians not in [-pi,pi] or [0,2pi] range")
      return 0  

########################################################################
## Descriptive statistics
########################################################################

def circ_mean(alpha, weight=None, dim=0):
    '''
    function that calculate the circular mean of a given vector
    '''
    alpha = np.asarray(alpha)
    if weight is None:
        weight = np.ones_like(alpha)
    
    # compute weighted sum of cos and sin of angles
    r = bn.nansum(np.multiply( weight , np.exp( 1j * alpha)), dim)
    mu = np.angle(r)

    return mu


def circ_r(alpha, weight=None , d=False , dim=0 ):
    '''
    
    '''
    alpha = np.asarray(alpha)
    if weight is None:
        weight = np.ones_like(alpha)

    # compute weighted sum of cos and sin of angles
    r = bn.nansum(np.multiply( weight , np.exp( 1j * alpha)), dim)

    r = abs(r)/bn.nansum(weight,dim)

    # from matlab circ_r function 
    # for data with known spacing, apply correction factor 
    # to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)

    if d:
        c = d/2./np.sin(d/2)
        r = c*r

    return r 


def circ_std(alpha, weight=None , d=False , dim=0 ):
    '''
    Compute circular Standard Deviation
    '''
    alpha = np.asarray(alpha)
    if weight is None:
        weight = np.ones_like(alpha)

    # compute the resultant vector of the data
    r = circ_r(alpha,weight,d,dim)

    # calculate angular deviation
    s = np.sqrt( 2 * (1-r))

    # calculate circular standard deviation 
    s0 = np.sqrt( -2 * np.log(r))

    return s , s0


########################################################################
## Test statistics
########################################################################

def corr_cl( x, theta, tail='two-sided'):
    '''
    circular linear correlation between a linear variable x and circular theta 
    '''
    from scipy.stats import pearsonr, chi2

    x = np.asarray(x)
    theta = np.asarray(theta)

    assert x.size == theta.size, 'x and theta should have the same length'
    assert tail in ['one-sided' , 'two-sided'] , 'Tail should be one-sided or two-sided '

    n = x.size

    # Compute pearson correlation forsin and cos of angular data 
    rxs = pearsonr(theta, np.sin(x))[0]
    rxc = pearsonr(theta, np.cos(x))[0]
    rcs = pearsonr(np.sin(x), np.cos(x))[0]


    r = np.sqrt((rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2))

    # Calculate p-value
    pval = chi2.sf(n * r**2, 2)
    pval = pval / 2 if tail == 'one-sided' else pval
    return r, pval



########################################################################
## other    
########################################################################

def cemd(f,g, period=[0 , 2*pi] ):
    ''' 
    Calculate circular earth mover distance between two histograms
    of circular distributions 

    Input: 
            - f and g two histograms of circular distributions 
            - period: is a variable defining on which domain we calculate the distance
            (eg: [0, 2*pi] for radian, [0, 24] for hours, ....)
    Output:
            - CEMD: circular Wassertein distance

    References:
    Rabin et al 2011 

    see for potential non discret adaptation (around page 150)
    https://pastel.archives-ouvertes.fr/file/index/docid/472442/filename/these.pdf
    '''
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)

    assert f.size == f.size, 'f and g should have the same length'

    #initialise variables
    n = len(f)
    D = np.empty_like(f)
    idx = np.arange(n)
    
    # normalize histograms just in case it was not done before
    # histograms should be probability distributions such as sum(f) = sum(g) = 1 
    f /= bn.nansum(f)
    g /= bn.nansum(g)

    # loop on al the possible starting positions
    for k in idx:
        # shift indexes 
        idx_k = np.roll(idx,k-1)

        # calculate cumulative histograms
        F_k = np.cumsum(f[idx_k])
        G_k = np.cumsum(g[idx_k])

        F_k /= bn.nansum(F_k)
        G_k /= bn.nansum(G_k)

        # calculate sum of distances
        D[k] = bn.nansum( abs(F_k-G_k) )
    
    # divide by n and multiply by "period" in order to set the max distance 
    # to this value. 
    D = D / n * np.diff(period)

    return bn.nanmin(D)
