import numpy as np
import bottleneck as bn
from hippocampy.matrix_utils import remove_small_objects


def calcDF(Froi,Fneu, method=None):
    """
    Here we do the classical neuropil extraction. A more refined version could be done by estimating c for
    for each cells as in:
     - Ultrasensitive fluorescent proteins for imaging neuronal activity, 
     TW Chen TJ Wardill Y Sun SR Pulver SL Renninger A Baohan ER Schreiter RA Kerr MB Orger V Jayaraman LL Looger K Svoboda DS Kim 
     (2013b)

     Note: 

     the last option is performed with rtobust linear regresion function robustfit.m in matlab

     in python we could use cikit learn with different option as descibed here.
     https://machinelearningmastery.com/robust-regression-for-machine-learning-in-python/

     matlab robustfir use bisquare option by default
    """

    if method is None:
        F = Froi - (0.7 * Fneu)

    else:
        raise NotImplementedError("Method not implemented")


    # here we should also male local detrend with Dombeck approach (substraction of 8th percentile of a sliding window)
    # or by using high pass filter, local detrend 
    return F

def transientSH(F):
    """
    find transient as in Allegra, Posani, Schmidt-Hieber

    “Events” were identified as contiguous regions in the d​ F ​ / ​ F signal exceeding a
    threshold of mean +2.5 standard deviations of the overall d​ F ​ / ​ F signal, and exceeding an integral
    above threshold of 7,000 d​ F ​ / ​ F ​ .
    """
    F_mean = bn.nanmean(F,axis=1)
    F_std = bn.nanstd(F,axis=1)

def transientRoy(F , threshold = 2.5 , minSize = 9):
    """
    find transient as in Roy 2017
    Ca 2+ events were detected by applying a threshold (greater than 2 standard
    deviations from the DF/F signal) at the local maxima of the DF/F signal.
    Since we employed GCaMP6f, our analysis used a threshold of > = 5 frames (250 ms)    
    """

    F_mean = bn.nanmean(F,axis=1)
    F_std = bn.nanstd(F,axis=1)

    # Zscore traces
    F_z = F - F_mean[:,np.newaxis]
    F_z = F_z / F_std[:,np.newaxis]

    # threshold trace above 2.5 std 
    F_t = F_z > threshold
    # remove small transients
    F_t = np.apply_along_axis(remove_small_objects, axis =1 ,arr=F_t, min_sz = minSize )





# %matplotlib widget

# t = 5

# i = list(range(t,19000))
# x = F_z[t,i]
# x_m = F_t[t,i]
# m = np.array(np.where(x_m))

# plt.figure()
# plt.plot(x)

# tmp = x[m]
# plt.plot(m.T, tmp.T ,marker='o', markerfacecolor=(1,0,0,1 ) )

