from scipy import sparse as sp
import torch
import numpy as np


def csr2torch(mat):
    coo = mat.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    torch_mat = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return torch_mat
