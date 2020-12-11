import numpy as np
from h5py import File
from sklearn import preprocessing


SCALERS = {"robust": preprocessing.RobustScaler, "standard": preprocessing.StandardScaler}


def get_h5_info(filename):

    with File(filename, "r") as h5:
        dataT = h5["trainD"]
        seqs_in_file = dataT.shape[0]

    return seqs_in_file


def load_h5_data(filename, seg_size):

    with File(filename, "r") as h5:
        # print(h5.keys())
        dataT = h5["trainD"][:].astype("float32")
        targetT = h5["trainL"][:].astype("float32")
        weights = h5["trainW"][:].astype("float32")
        # dataT = h5["trainD"]
        # print("hej")

    # Hack to make sure axis order is preserved
    if dataT.shape[-1] == 300:
        dataT = np.swapaxes(dataT, 0, 2)
        targetT = np.swapaxes(targetT, 0, 2)
        weights = weights.T

    # print(dataT.shape)
    # print(targetT.shape)
    # print(weights.shape)
    # print(f'{filename} loaded - Training')

    seq_in_file = dataT.shape[0]
    # n_segs = dataT.shape[1] // seg_size
    n_segs = dataT.shape[-1] // seg_size

    # return (
    #     np.reshape(dataT, [seq_in_file, n_segs, seg_size, -1]),
    #     np.reshape(targetT, [seq_in_file, n_segs, seg_size, -1]),
    #     np.reshape(weights, [seq_in_file, n_segs, seg_size]),
    #     seq_in_file,
    # )
    return (
        np.reshape(dataT, [seq_in_file, -1, n_segs, seg_size]),
        np.reshape(targetT, [seq_in_file, -1, n_segs, seg_size]),
        np.reshape(weights, [seq_in_file, n_segs, seg_size]),
        seq_in_file,
    )


def load_psg_h5_data(filename, scaling=None):
    scaler = None

    if scaling:
        scaler = SCALERS[scaling]()

    with File(filename, "r") as h5:
        N, C, T = h5["M"].shape
        sequences_in_file = N

        if scaling:
            scaler.fit(h5["M"][:].transpose(1, 0, 2).reshape((C, N * T)).T)

    return sequences_in_file, scaler
