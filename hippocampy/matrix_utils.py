import numpy as np
import bottleneck as bn
import warnings
from astropy.convolution import convolve
from skimage import measure , morphology

def smooth1D(data, kernelHalfWidth=3, kernelType='gauss', padtype='reflect', preserve_nan_opt=True):
  '''
  One dimensional Smoothing of data. It can deal with vector or matrix of vector.
  In case of 2D inputs, it will smooth along dim=1

  padtypes are given to numpy.pad. Availlable option are:

  - ‘symmetric’ Pads with the reflection of the vector mirrored along the edge 
    of the array.
  - 'reflect’ Pads with the reflection of the vector mirrored on the first and 
    last values of the vector along each axis.

    NB: The difference between symmetric and reflect is that reflect mirror the 
        the data without duplicating the end value:
        a = np.array([1 , 2 , 3 , 4])
        a_p = np.pad(a , (0,2) , 'reflect')
        [1 2 3 4 3 2]
         
        a_p = np.pad(a , (0,2) , 'symmetric')
        [1 2 3 4 4 3]

  - ‘wrap’ : Pads with the wrap of the vector along the axis. The first values are
    used to pad the end and the end values are used to pad the beginning.
  '''
  # Check Input

  if kernelHalfWidth%2 !=1:
    kernelHalfWidth += 1
    # kernel size has to be odd

  if len(data.shape) ==1:
    data = data[np.newaxis,:]
  
  acceptedPad = ['reflect', 'symmetric', 'wrap']
  assert any([i == padtype for i in acceptedPad]) , 'Not Implemented pad type'

  acceptedType = ['gauss', 'box', 'ramp']
  assert any([i == kernelType for i in acceptedType]) , 'Not Implemented smoothing kernel type'

  # pad the data
  data_p = np.pad(data, ((0,0) ,  (kernelHalfWidth , kernelHalfWidth)) ,padtype)

  if kernelType == 'box':
    # here bn.movemean seem to be much faster (~10x) than using a convolution as 
    # for the gaussian or ramp kernel. It affect the value of the moving mean to 
    # the last index in the moving window, that why the output 'un pading' is 
    # peculiar

    data_c = bn.move_mean(data_p,kernelHalfWidth*2+1, min_count=kernelHalfWidth, axis=1)
    data_c = data_c[:,kernelHalfWidth*2:]

    if preserve_nan_opt:
      data_c[:,np.isnan(data)] = np.nan

    return data_c

  else:
    # Make convolution kernel 
    kernel = np.zeros(kernelHalfWidth*2 +1)

    if kernelType == 'gauss':
      kernel = np.arange(0, kernelHalfWidth) - (kernelHalfWidth - 1.0) / 2.0
      kernel = np.exp(-kernel ** 2 / (2 * kernelHalfWidth * kernelHalfWidth) )
  
    elif kernelType == 'ramp':
      kernel = np.linspace(1,kernelHalfWidth+1,kernelHalfWidth+1)
      kernel = np.hstack( (kernel , kernel[-2::-1] ))

    # Normalize kernel to one 
    kernel = kernel/bn.nansum(kernel)
    # Convolve. Astropy  seems to deal realy well with nan values
    data_c = np.apply_along_axis(convolve , axis=1 , arr= data_p , kernel = kernel , preserve_nan=preserve_nan_opt)

    return data_c[:,kernelHalfWidth:-kernelHalfWidth]


def smooth2D(data, kernelHalfWidth=3, kernelType='gauss', padtype='reflect', preserve_nan_opt=True):
  '''
  function to smooth 2 dimensional data. 
  Take: 
          - data: matrix with your 2D data
          - kernelHalfWidth: half width of the smoothing kernel
          - kernelType: way to smooth the data ('gauss': gaussian, 'box': boxcar smoothing)
          - padtype: the matrix will be padded in order to remove border artefact
            so we will pad the matrix. 
            Available option: - symmetric: reflect the vector on the edge 1 2 3 4 [3 2 1]
                              - reflect: reflect the vector on the edge 1 2 3 4 [4 3 2]
                              - wrap: circularly wrap opposing edges 
          - preserve_nan_opt = do we smooth NaN or put them back att the end (default: True)

  '''
  
  # Check Input
  if kernelHalfWidth%2 !=1:
    # kernel size has to be odd
    kernelHalfWidth += 1
    
  acceptedPad = ['reflect', 'symmetric', 'wrap']
  assert any([i == padtype for i in acceptedPad]) , 'Not Implemented pad type'

  acceptedType = ['gauss', 'box']
  assert any([i == kernelType for i in acceptedType]) , 'Not Implemented smoothing kernel type'

  # pad the data
  data_p = np.pad(data, ((kernelHalfWidth , kernelHalfWidth)) , padtype)

  # Initialize convolution kernel 
  kernel = np.zeros((kernelHalfWidth*2 +1 , kernelHalfWidth*2 +1))

  if kernelType == 'box':
    kernel = np.ones( (kernelHalfWidth*2 +1 , kernelHalfWidth*2 +1) )
  elif kernelType == 'gauss':
    kernel_1D = np.arange(0, kernelHalfWidth) - (kernelHalfWidth - 1.0) / 2.0 
    kernel_1D = np.exp(-kernel_1D ** 2 / (2 * kernelHalfWidth * kernelHalfWidth) )
    kernel = np.outer(kernel_1D,kernel_1D)
  
  # Normalize kernel to one 
  kernel = kernel/bn.nansum(kernel)
  
  # Convolve. Astropy seems to deal really well with nan values
  data_c = convolve(data_p, kernel=kernel, preserve_nan=preserve_nan_opt)

  return data_c[kernelHalfWidth:-kernelHalfWidth ,kernelHalfWidth:-kernelHalfWidth]

def label(M):
    # the following line ensure we feed a boolean data to the label
    # function. Its helps with 
    M_new = np.array(M, dtype=bool)

    return measure.label(M_new)

def remove_small_objects(M,min_sz=3):
    M_l= label(M)
    M_l= morphology.remove_small_objects(M_l,min_size=min_sz)
    return np.array(M_l,dtype=bool)

def moving_win(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis into overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is 
    unevenly strided and being flattened or because end is set to 
    'pad' or 'wrap').

    See:
    IPython Interactive Computing and Visualization Cookbook, Second Edition (2018), by Cyrille Rossant:
    https://ipython-books.github.io/46-using-stride-tricks-with-numpy/

    This function is a slightly adapted version of:
    segment_axis.py from:
    https://scipy-cookbook.readthedocs.io/items/SegmentAxis.html
    https://scipy-cookbook.readthedocs.io/_static/items/attachments/SegmentAxis/segmentaxis.py

    Note: 
    We could implement symetric padding 

    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap<0 or length<=0:
        raise ValueError("overlap must be nonnegative and length must be positive")

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b
        
        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError("Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'")
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis]+(n,length)+a.shape[axis+1:]
    newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]

    try: 
        return np.ndarray.__new__(np.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)



def row_closest_min_to_val(v, zero_val=None):
  """
  This function identify the identify the closest index
  of non-zero element to a given value.

  Input:
        - v: vector, preferably logical
        - zero_val: value of the "zero"
  Returns:
        - index of the closest index
  """
  if zero_val is None:
      zero_val = v.size/2

  tmp = np.squeeze(np.nonzero(v))

  if tmp.size > 0:
    return tmp[np.argmin(abs(tmp-zero_val))]
  else:
    return np.nan

def find_peak_row(M,zero_idx=None):
  """
  find_peak_row will find the peak per row that is closest to a 
  particular value.

  Input:
          - M: Matrix of value, this function will by applied
          over rows
          - zero_val: value of the "zero"
    Returns:
          - index of the closest index

  Example:
  a = np.array([ [1, 2, 3, 2, 0] [ 4, 8, 9, 12, 1] ])
  p = 

  """
  if zero_idx is None:
    zero_idx = M.shape[1]/2

  bef = np.hstack((np.atleast_2d(M[:,0]).T , M[:,:-1]))
  aft = np.hstack((M[:,1:], np.atleast_2d(M[:,-1]).T ))
  peaks = np.logical_and(M-bef >=0 , M-aft >=0)

  return np.apply_along_axis(row_closest_min_to_val, axis=1, arr=peaks , zero_val = zero_idx)

def find_peaks(M, min_amplitude= None):
  """
  find peaks over in each rows in a matrix
  Example:
  M = np.array([ [1, 2, 3, 2, 0], [ 4, 8, 9, 12, 1] ])
  [ P, P_idx ] = find_peaks(M)
  P = array([[False, False,  True, False, False],
       [False, False, False,  True, False]])
  P_idx = [array(2), array(3)]
  """

  bef = np.hstack((np.atleast_2d(M[:,0]).T , M[:,:-1]))
  aft = np.hstack((M[:,1:], np.atleast_2d(M[:,-1]).T ))

  if min_amplitude is None:
    peaks = np.logical_and(M-bef >=0 , M-aft >=0)
  else:
    peaks = np.logical_and.reduce([M-bef >=0 , M-aft >=0 , M>=min_amplitude])

  peaks_idx = [np.squeeze(np.nonzero(valP)) for itP,valP in enumerate(p) ]

  return peaks,peaks_idx
