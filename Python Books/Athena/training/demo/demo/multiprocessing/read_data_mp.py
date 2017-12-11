import numpy as np


def read_file(ncols, nrows, offset, dtype):
    with open(fname, 'rb') as fh:
        fh.seek(offset)
        arr = np.fromfile(fh, dtype, count=ncols * nrows)
    arr.shape = (nrows, ncols)
    return arr


if __name__ == '__main__':
    fname = "fake_data.dat"
