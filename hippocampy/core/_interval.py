import numpy as np
import bottleneck as bn

class Iv:
  def __init__(self, data):
    data = np.array(data,ndmin=2,dtype=float)
    data = np.reshape(data,[-1,2])

    self.starts = data[:,0]
    self.stops = data[:,1]
    ### Should make a better check of the size and type of the input data

  def __getitem__(self, idx):
      return Iv([self.starts[idx], self.stops[idx]])

  def __iter__(self):
    for start, stop in zip(self.starts, self.stops):
      yield Iv([start, stop])

  def __len__(self): 
      return len(self.starts)

  def __contains__(self ,other):
    """
    Contains is defined for Iv, arrays of size(1,2) and values
    """

    if isinstance(other, type(self)):
      is_in = np.logical_and(other.starts >= self.starts, other.stops <= self.stops)   
      return any(is_in)

    elif isinstance(other, (np.ndarray , list) ):
      new = np.array(other,ndmin=2)
      assert new.shape == (1,2)
      
      is_in = np.logical_and(other[0] >= self.starts, other[1] <= self.stops)   
      return any(is_in)  

    elif isinstance(other, (int, float, complex)) and not isinstance(other, bool):
      is_in = np.logical_and(other >= self.starts, other <= self.stops)   
      return any(is_in)

    else:
      raise TypeError

  def data(self):
    return np.array([self.starts, self.stops]).T

  def min(self):
    return bn.min(self.starts)

  def max(self):
    return bn.max(self.stops)
  
  def length(self):
    return self.stops-self.starts
  
  def __and__(self, other):
    """
    https://scicomp.stackexchange.com/questions/26258/the-easiest-way-to-find-intersection-of-two-intervals

    """
    if isintance(other, type(self)):

      raise NotImplemented
    else:
      raise NotImplemented

  def sort(self):
    """Sort interval by start time"""
    idx = np.argsort(self.starts)
    self.starts = self.starts[idx]
    self.stops = self.stops[idx]

  def append(self, other):

    if not self:
      return Iv(other)
    if not other:
      return self

    self_data = np.array([self.starts,self.stops], ndmin=2)

    if isinstance(other, Iv):
      other_data = np.array([other.starts,other.stops])
      newVal = np.hstack((self_data,other_data)).T
      return Iv(newVal)
    elif isinstance(other, (np.ndarray , list)):
      ### Should make a better check of the size and type of the input data
      other_data = np.array(other, ndmin=2) 
      assert other_data.shape[1] ==2 , "Dimension mismatch"

      newVal = np.hstack((self_data,other_data)).T
      return Iv(newVal)
    else: 
      raise NotImplementedError

  def clean(self):
    """
    Clean a set of interval. eg: merge overlapping intervals and remove intervals
    included in others
    """
    lBounds = self.starts
    uBounds = self.stops
    isMerged = np.zeros_like(self.starts)
    newdata= np.array([], dtype=np.int64).reshape(0,2)

    for it, _ in enumerate(self):
      if isMerged[it]==0:
        l = lBounds[it]
        u = uBounds[it]
        # does the currents interval overlap others
        ov = np.logical_and( l < uBounds , u > lBounds)
        # does the current interval contains others
        cont = np.logical_and( l < lBounds , u > uBounds )
        # make global mask
        mask = np.logical_or(ov, cont)
      
        if bn.nansum(mask)==1:
          # out interval do not overlap or contain anything so we keep it
          newdata = np.vstack([newdata, [l,u] ])
        else:
          # from this, determine the new interval
          l_new = bn.nanmin(lBounds[mask])
          u_new = bn.nanmax(uBounds[mask])
        
          newdata =  np.vstack( [newdata, [l_new,u_new] ])

        isMerged[mask] = 1
    return Iv(newdata)
    
  def merge(self, other):
    """
    merge two sets of interval. 
    Append the interval and clean it 
    """
    tmp = self.append(other)
    return tmp.clean()