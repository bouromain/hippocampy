import numpy as np
import bottleneck as bn
from hippocampy.matrix_utils import remove_small_objects
from scipy.stats import siegelslopes
from sklearn.linear_model import RANSACRegressor
import tqdm as tqdm


def subtract_neuropil(Froi,Fneu, method="fixed", downsample_ratio= 10):
    """
    Here we do the classical neuropil extraction with two methods. Either by calculating:
                F = Froi - (0.7 * Fneu)
    or by using a robust fit between Froi and Fneu as perfromed in the following paper:

    - Ultrasensitive fluorescent proteins for imaging neuronal activity, 
    TW Chen TJ Wardill Y Sun SR Pulver SL Renninger A Baohan ER Schreiter
    RA Kerr MB Orger V Jayaraman LL Looger K Svoboda DS Kim  (2013b)
    """
    Froi = np.asarray(Froi)
    Fneu = np.asarray(Fneu)

    # this reshape is particularly important for the robust method but I put it 
    # here for homogeneity 
    if Froi.shape[1] < Froi.shape[0] or Fneu.shape[1] < Fneu.shape[0]:
        raise SyntaxError('Data should be given as [cells, samples]')
    
        return

    if method is "fixed":
        F = Froi - (0.7 * Fneu)

    elif method is "robust":
        # Robustly fit linear model with RANSAC algorithm
        ransac = RANSACRegressor()
        c = np.empty(Froi.shape[0])

        for itF in tqdm.tqdm(range(Froi.shape[0])):
            x = np.atleast_2d(Froi[itF , ::downsample_ratio]) 
            y = np.atleast_2d(Fneu[itF, ::downsample_ratio])
            ransac.fit(x.T, y.T)
            c[itF] = ransac.estimator_.coef_

        # now only select the correct values between 0.5 and 1
        c_valid = np.logical_and(c>0.5 , c<1) 
        c[ np.logical_not(c_valid) ] = np.median(c[ c_valid ])
        
        # Calculate F
        F = Froi - c[:,None] * Fneu
    else:
        raise NotImplementedError("Method not implemented")

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

