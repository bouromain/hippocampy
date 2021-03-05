#%% utils for various codes 
# author RB 06/20
import numpy as np

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

def localExtrema(x,method='max'):
  """
  
  Find local extrema and return their index

  Inputs: 
          - x: vector
          - method: type of extrema to consider [max, min, all] (default: max)

  """
  assert method in ['max', 'min', 'all'], "Invalid Method in localExtrema"
  
  x = np.asarray(x, dtype=np.float)
  D = np.diff(x)
  E = np.diff( D /abs(D) )

  if method == 'max':
    return np.nonzero(E==-2)
  elif method == 'min':
    return np.nonzero(E==2)
  else:
    return np.nonzero( np.logical_or(E==2,E==-2) )


def _remove_nan(x, x_mask=None, axis=0):
    """
    Remove NaN in a 1D array.
    adapted from Pinguin _remove_na_single
    """
    if x_mask is None:
       x_mask = _nan_mask(x,axis=axis)

    # Check if missing values are present
    if ~x_mask.all():
      ax = 0 if axis == 0 else 1
      ax = 0 if x.ndim == 1 else ax
      x = x.compress(x_mask, axis=ax)
    
    return x

def _nan_mask(x,axis=0):
  if x.ndim == 1:
    # 1D arrays
    x_mask = ~np.isnan(x)
  else:
    # 2D arrays
    ax = 1 if axis == 0 else 0
    x_mask = ~np.any(np.isnan(x), axis=ax)

  return x_mask

def remove_nan(x, y=None, paired=False, axis=0):
  """
  Helper function to remove nan from 1D or 2D
  """
  x = np.asarray(x)
  if y is None:
    return _remove_nan(x,axis=axis)
  else:
    y = np.asarray(y)

    if not paired:
      x = _remove_nan(x,axis=axis)
      y = _remove_nan(y,axis=axis)

      return x, y

    else:
      x_mask = _nan_mask(x,axis=axis)
      y_mask = _nan_mask(y,axis=axis)

      xy_mask = np.logical_and(x_mask ,y_mask)

      if ~np.all(xy_mask):
        x = _remove_nan(x,x_mask=xy_mask , axis=axis)
        y = _remove_nan(y,x_mask=xy_mask , axis=axis)

      return x, y



  


# %%
