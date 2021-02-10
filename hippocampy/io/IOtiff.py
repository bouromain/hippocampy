from ScanImageTiffReader import ScanImageTiffReader
import h5py

def get_tiff_metadata(pfile:str):
    '''
    Open tiff and return all the metadata
    input
    '''
    with ScanImageTiffReader(pfile) as reader:
        out = reader.metadata()
    return out

def get_tiff_shape(pfile:str):
    with ScanImageTiffReader(pfile) as reader:
        out = reader.shape()
    return out


def get_tiff_data(pfile:str):
    with ScanImageTiffReader(pfile) as reader:
        out = reader.data()
    return out

shapeTiff = get_tiff_shape(pfile)


# ## create a normal h5 file without compression
# hfile = "/home/bouromain/Documents/tmpData/2020-03-13_12-20-06/m4368_20200313_00001.hdf5"

# f = h5py.File(hfile)  # make an hdf5 file
# dataTiff = get_tiff_data(pfile)
# out = f.create_dataset('/x', data=dataTiff)
# f.close()

# ## create a normal h5 file without compression
# gzipfile = "/home/bouromain/Documents/tmpData/2020-03-13_12-20-06/m4368_20200313_00001-gzip.hdf5"

# f = h5py.File(gzipfile)  # make an hdf5 file
# dataTiff = get_tiff_data(pfile)
# out = f.create_dataset('/x', data=dataTiff, compression="gzip", shuffle=True, chunks= True)
# f.close()

# ## create a normal h5 file without compression
# lzffile = "/home/bouromain/Documents/tmpData/2020-03-13_12-20-06/m4368_20200313_00001-lzf.hdf5"

# f = h5py.File(lzffile)  # make an hdf5 file
# dataTiff = get_tiff_data(pfile)
# out = f.create_dataset('/x', data=dataTiff, compression="lzf", shuffle=True, chunks= True)
# f.close()


