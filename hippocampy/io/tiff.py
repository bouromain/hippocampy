from ScanImageTiffReader import ScanImageTiffReader
import h5py
import os
import json

from hippocampy.io.matlab import matlab_string2dict
import cv2
import bottleneck as bn
import numpy as np


def get_tiff_metadata(file_path: str):
    """
    Open tiff and return all the metadata
    input

    Parameters
    ----------
    file_path
        path of the file to process

    Returns
    -------
    data:
        metadata extracted from the tif file
    """
    # to avoid some bugs
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError

    with ScanImageTiffReader(file_path) as reader:
        try:
            out = json.loads(reader.metadata())
        except:
            out = matlab_string2dict(reader.metadata())
    return out


def get_tiff_fs(file_path: str) -> int:
    """
    get_tiff_fs deduce tiff sampling rate from the second
    frame metadata. It seem to be faster than searching the
    file metadata on some big files

    Parameters
    ----------
    file_path : str
        path of the file to process

    Returns
    -------
    int
        sampling frequency
    """

    # to avoid some bugs
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError

    with ScanImageTiffReader(file_path) as reader:
        # deduce the frame rate from the second frame only
        s = reader.description(1)
        s = s.split("frameTimestamps_sec =")[1]
        s = s.split("acqTriggerTimestamps_sec")[0]

        return round(1 / float(s.strip()))


def get_tiff_aspect_ratio(file_path: str) -> float:
    """
    get_tiff_aspect_ratio,

    Parameters
    ----------
    file_path : str
        path of the file to process

    Returns
    -------
    aspect_ratio:
        aspact ratio
    """
    _, y, x = get_tiff_shape(file_path)

    return x / y


def get_tiff_shape(file_path: str):
    """
    Open tiff and return its shape

    Parameters
    ----------
    file_path
        path of the file to process

    Returns
    -------
    data:
        shape of the data in the tif file
    """
    # to avoid some bugs
    file_path = os.path.expanduser(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError

    with ScanImageTiffReader(file_path) as reader:
        out = reader.shape()
    return out


def get_tiff_data(file_path: str):
    """
    Open tiff and return all its data

    Parameters
    ----------
    file_path
        path of the file to open

    Returns
    -------
    data:
        data extracted from the tif file
    """

    # to avoid some bugs
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError

    with ScanImageTiffReader(file_path) as reader:
        out = reader.data()
    return out


def tiff2h5(
    pfile: str,
    compression_type="gzip",
    compression_lvl=5,
    overwrite=False,
    n_frblk=2000,
):
    """
    convert a scanImage tiff file and convert is to a h5 file with compression

    Parameters
    ----------
        - pfile: path of the file to process
        - compression_type: type of compression ["gzip"(default), "lzf"]
        - compression_lvl: level of compression
        - overwrite: if should overwrite existing file
        - n_frblk: number of frames to read per blocks

    TO DO
    -----
        - better handling of existing file eg erase it
        - calculate nfr depending of available ram of the system
        - store metadata of the tiff file

    References
    ----------
    https://vidriotech.gitlab.io/scanimagetiffreader-python
    https://gitlab.com/vidriotech/scanimagetiffreader-python/-/blob/master/src/ScanImageTiffReader/__init__.py
    https://docs.h5py.org/en/stable/high/dataset.html
    """
    # Check inputs
    list_compression = ["gzip", "lzf"]
    assert compression_type in list_compression, "Available compressions are gzip, lzf "
    assert pfile.endswith(".tif"), "Input file should be a tiff file"

    # prepare some things
    h5_file_name = pfile.replace(".tif", ".h5")

    if os.path.isfile(h5_file_name) and not overwrite:
        raise FileExistsError("This file already exist, use overwrite=True")

    # get shape of the tiff data [n_frames, pixels, pixels ]
    tiff_shape = get_tiff_shape(pfile)

    # open files for the loop
    h5_f = h5py.File(h5_file_name)
    tiff_f = ScanImageTiffReader(pfile)

    # extract first frames and initiate a dataset with it
    tmp_tiff_data = tiff_f.data(beg=0, end=n_frblk)
    dset = h5_f.create_dataset(
        "data",
        data=tmp_tiff_data,
        compression=compression_type,
        compression_opts=compression_lvl,
        shuffle=True,
        chunks=True,
        maxshape=(None, tiff_shape[1], tiff_shape[2]),
    )

    # now we can loop to incrementally fill the new h5 file
    it_frame = n_frblk
    while it_frame < tiff_shape[0]:

        n_frblk = min(tiff_shape[0] - it_frame, n_frblk)

        dset.resize(it_frame + n_frblk, axis=0)
        tmp_tiff_data = tiff_f.data(beg=it_frame, end=it_frame + n_frblk)
        dset[-n_frblk:, :, :] = tmp_tiff_data
        it_frame += n_frblk

    tiff_f.close()
    h5_f.close()


def bin2mp4(
    bin_path: str,
    dest_path: str = None,
    fs_out: int = 30,
    Ly: int = 512,
    Lx: int = 512,
    time_smooth_sample: int = 30,
):

    ## init
    if format not in ["mp4", "avi"]:
        raise ValueError("Format should be either ['mp4', 'avi'] ")

    if format == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fourcc = cv2.VideoWriter_fourcc(*'h264')
    elif format == "avi":
        fourcc = cv2.VideoWriter_fourcc("F", "M", "P", "4")

    if dest_path is None:
        dest_path = bin_path.replace(".bin", f".{format}")

    halfwin_sample = int(np.floor(time_smooth_sample / 2))

    # we may have problem for very big tiff
    # data = get_tiff_data(tiff_path)
    # data = data.astype(np.uint8)

    #####
    with open(bin_path, mode="rb") as fio:
        data = np.fromfile(fio, np.int16).reshape((-1, Ly, Lx))
    data = cv2.convertScaleAbs(data)  # convert to uint8

    out = cv2.VideoWriter(dest_path, fourcc, fs_out, ((data.shape[1:])))
    for it_im in range(1000):  # data.shape[0]):
        # rotate the buffer
        idx = np.arange(it_im - halfwin_sample, it_im + halfwin_sample)
        idx = idx[idx >= 0]
        idx = idx[idx < data.shape[0] - 1]

        buffer = data[idx, :, :]
        im_m = bn.nanmean(buffer, axis=0).astype(np.uint8)
        im_m = cv2.equalizeHist(im_m)
        out.write(cv2.cvtColor(im_m, cv2.COLOR_GRAY2BGR))
    out.release()


### for dev
# tiff_path = "/home/bouromain/Documents/tmpData/tmp2p/m4368_20200210_00001.tif"
# bin_path = "/home/bouromain/Documents/tmpData/tmp2p/suite2p/plane0/data.bin"
# fs_out = 30
# time_smooth_sample = 30
# Ly:int = 512,
# Lx:int= 512,
# format = "mp4"
# dest_path = None


# bin2mp4(
#     bin_path,
#     dest_path= None,
#     fs_out  = 30,
#     Ly = 512,
#     Lx= 512,
#     time_smooth_sample = 30,
# )

### end dev
