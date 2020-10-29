#%% utils for various codes 
# author RB 06/20
import numpy as np
from numpy import pi

#%% 
def wrap(x, rangeVal = 1): 
  """
  wrap radian values between range [-pi,pi] (range = 1) or [0,2pi] (range = 2)
  """
  x = x % (2*pi)

  if rangeVal == 1:
    underPi = x > pi
    x[underPi] = x[underPi] - (2*pi)
  
  return x 

#%%
def valueCross(x,v=0):
  """
  function finding the crossing point between a vector and a value.
  Particularly useful when you want to detect phase inversion of crossing
  were you are not likely to find exactly a value in your vector strictly equal
  to a value
  """
  before = np.array(x[:-1]) 
  after = np.array(x[1:])

  up = np.logical_and(before<v, after>v)
  down = np.logical_and(before>v, after<v)

  up = np.append(up, False)
  down = np.append(down, False)

  return up, down

# %%
def localExtrema(x,method='max'):
  """
  
  Find local extrema and return their index

  Inputs: 
          - x: vector
          - method: type of extrema to consider [max, min, all] (default: max)

  """
  allMethods = ['max', 'min', 'all']
  assert any(method == s for s in allMethods), "Invalid Method in localExtrema"

  D = np.diff(x)
  E = np.diff( D /abs(D) )

  if method == 'max':
    return np.nonzero(E==2)
  elif method == 'min':
    return np.nonzero(E==-2)
  else:
    return np.nonzero( np.logical_or(E==2,E==-2) )
