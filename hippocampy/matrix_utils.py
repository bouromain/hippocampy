import numpy as np
import bottleneck as bn
from astropy.convolution import convolve
from skimage import measure
from skimage import morphology

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
  # Check Input

  if kernelHalfWidth%2 !=1:
    kernelHalfWidth += 1
    # kernel size has to be odd
 
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
