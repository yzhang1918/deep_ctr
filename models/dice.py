import torch
from torch import nn


class Dice(nn.Module):

    # https://github.com/DiligentPanda/Tencent_Ads_Algo_2018/blob/master/src/module/dice.py

    def __init__(self, dim, init=0):
        super().__init__()
        self.dim = dim
        # todo the momentum in paper is weird, we use the default choice here
        # todo maybe affine?
        self.bn = nn.BatchNorm1d(dim, momentum=0.01, affine=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.full((self.dim,), init))

    def forward(self, x):
        if x.dim() == 2:
            y = self.bn(x)
        elif x.dim() == 3:
            y = self.bn(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError
        p = self.sigmoid(y)
        a = self.alpha * (1 - p) * x + p * x
        return a
