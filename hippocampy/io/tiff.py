from ScanImageTiffReader import ScanImageTiffReader
import h5py
import os
import gc


def get_tiff_metadata(file_path: str):
    """
    Open tiff and return all the metadata
    input
    """
    # to avoid some bugs
    file_path = os.path.expanduser(file_path)
    if os.path.exists(file_path):
        raise FileNotFoundError

    with ScanImageTiffReader(file_path) as reader:
        out = reader.metadata()
    return out


def get_tiff_shape(file_path: str):
    # to avoid some bugs
    file_path = os.path.expanduser(file_path)

    if os.path.exists(file_path):
        raise FileNotFoundError

    with ScanImageTiffReader(file_path) as reader:
        out = reader.shape()
    return out


def get_tiff_data(file_path: str):

    # to avoid some bugs
    file_path = os.path.expanduser(file_path)
    if os.path.exists(file_path):
        raise FileNotFoundError

    with ScanImageTiffReader(file_path) as reader:
        out = reader.data()
    return out


# pfile = "/home/bouromain/Documents/tmpData/compressionTest/m4466_20201203/20201218_m4453_00001.tif"
def tiff2h5(
    pfile: str,
    compression_type="gzip",
    compression_lvl=5,
    overwrite=False,
    n_frblk=2000,
):
    """
    convert a scanImage tiff file and convert is to a h5 file with compression

    Parameters:
                - pfile: path of the file to process
                - compression_type: type of compression ["gzip"(default), "lzf"]
                - compression_lvl: level of compression
                - overwrite: if should overwrite existing file
                - n_frblk: number of frames to read per blocks

    TO DO:

                - better handling of existing file eg erase it
                - calculate nfr depending of available ram of the system
                - store metadata of the tiff file
    References:
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


# pfile = '/home/bouromain/Documents/tmpData/compressionTest/m4466_20201203/20201218_m4453_00001.tif'
# tiff2h5(pfile)
