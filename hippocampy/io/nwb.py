import h5py as h5
import os.path as op


def load_nwb_oe(fpath):
    """
    Load open ephys nwb file

    Parameter
    ---------
    fpath: str
        - path of the file to open 


    Return
    ------
    data: array_like
        data extracted from the file 
    timestamps: array_like
        timestamps of this data

    Reference
    ---------
    https://github.com/open-ephys/open-ephys-python-tools/blob/main/open_ephys/analysis/formats/NwbRecording.py

    """

    # Check inputs
    if not op.exists(fpath):
        raise FileNotFoundError("File does not exist" % (fpath))

    if not fpath.endswith(".nwb"):
        raise ValueError("File does not seem to be a nwb file")

    # Open the file and extrat data
    with h5.File(fpath, "r") as fio:
        dataset = fio["acquisition"]["timeseries"]["recording1"]["continuous"]

        processors = dataset.keys()

        if len(processors) == 1:
            # here I use a for loop to access the processor more easily
            # however here there will be only one iteration
            for processor in dataset.keys():
                data = dataset[processor]["data"][()]
                timestamps = dataset[processor]["timestamps"][()]
        elif len(processors) > 1:
            data = [None]
            timestamps = [None]
            for i, processor in enumerate(dataset.keys()):
                data[i] = dataset[processor]["data"][()]
                timestamps[i] = dataset[processor]["timestamps"][()]

        else:
            raise ValueError("Invalid or empty processor for file = %s" % (fpath))

        return data, timestamps


# import matplotlib.pyplot as plt
# import pywt
# import numpy as np
# from scipy.signal import decimate

# fpath = "/home/bouromain/Documents/tmpData/m4540_2021-07-30_15-04-15/Record Node 107/experiment_1.nwb"

# data, timestamps = load_nwb_oe(fpath)

# D = decimate(data[:, 14], 10)
# DD = D[:100000]

# P = 1 / 2000
# scales = np.arange(2, 200)  ## scales are wrong here


# coeffs, freq = pywt.cwt(DD, scales, wavelet="morl", sampling_period=P, method="fft")
# # calculate power from coefficient
# pow_c = np.abs(coeffs[:1000]) ** 2
# # calculate 1/f
# pow_unbiased = pow_c / freq[:, None]

# plt.imshow(pow_unbiased, aspect="auto")
# # plt.yticks(pywt.scale2frequency("morl", scales) / P)

