from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import bottleneck as bn
import cv2

from hippocampy.data import load_calcium

# def mov_from_Q(Q, footprints,*, fps = 30):


# F, Fneu, iscell, spk, stats = load_calcium()

# Q = spk
# Q = Q / bn.nanmean(Q, axis=1)[:, None]
# n_cells = len(stats)
# footprints = np.zeros((n_cells, 512, 512))


# for i, s in enumerate(stats):
#     ypix = s["ypix"][~s["overlap"]]
#     xpix = s["xpix"][~s["overlap"]]
#     footprints[i, ypix, xpix] = 1

# HSV_footprint = np.repeat(footprints[None, ...], 3, axis=0)
# HSV_footprint[0, ...] = HSV_footprint[0, ...] * np.random.rand((n_cells))[:, None, None]
# HSV_footprint[1, ...] = HSV_footprint[1, ...] * np.ones((n_cells, 1, 1))


# import tqdm

# # initialize the FourCC and a video writer object
# fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
# output = cv2.VideoWriter("/home/bouromain/Documents/output.mp4", fourcc, 30, (512, 512))

# ## we need to check the range of the output image, it the act it too small we will see nothing
# for i, act_frame in tqdm.tqdm(enumerate(Q[:, :500].T), total=Q.shape[1]):
#     im = HSV_footprint
#     im[2, ...] = im[2, ...] * act_frame[:, None, None]

#     im = bn.nansum(im, axis=1)
#     im = hsv_to_rgb(im.T)
#     im = np.floor(im * 256).astype(np.uint8)
#     im[im > 256] = 256
#     output.write(im)
# output.release()
