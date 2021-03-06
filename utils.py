import numpy as np
import torch
import pickle


def savepkl(obj, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(obj, fh)


def loadpkl(filename):
    with open(filename, 'rb') as fh:
        obj = pickle.load(fh)
    return obj


def csr2torch(mat):
    coo = mat.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    torch_mat = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return torch_mat


def parse_csr(mat):
    coo = mat.tocoo()
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    shape = coo.shape
    return indices, values, shape[0]
