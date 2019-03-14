import torch
from torch import nn
from torch_sparse import spmm


class SparseLinear(nn.Linear):

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            ind, val = x
        else:
            ind, val = x._indices(), x._values()
        y = spmm(ind, val, ind.size(0), self.weight.t())
        if self.bias is not None:
            y += self.bias
        return y


def trunc_normal_(x, mean=0., std=1.):
    # From Fast.ai
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def embedding(ni, nf, padding_idx=None):
    # From Fast.ai
    emb = nn.Embedding(ni, nf, padding_idx=padding_idx)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad():
        trunc_normal_(emb.weight, std=0.01)
    return emb


def pretrained_embedding(ni, nf, weights, bias=False, padding_idx=None, requires_grad=True):
    _, nh = weights.shape
    emb_layer = nn.Embedding(ni, nh, padding_idx=padding_idx)
    emb_layer.weight.data.copy_(weights)
    emb_layer.weight.requires_grad = requires_grad
    linear_layer = nn.Linear(nh, nf, bias=bias)
    nn.init.xavier_uniform_(linear_layer.weight.data)
    if bias:
        nn.init.zeros_(linear_layer.bias.data)
    layer = nn.Sequential(emb_layer, linear_layer)
    return layer


def pretrained_sparse_linear(ni, nf, weights, bias=False, requires_grad=True):
    _, nh = weights.shape
    layer_1 = SparseLinear(ni, nh, bias=False)
    layer_1.weight.data.copy_(weights)
    layer_1.weight.requires_grad = requires_grad
    layer_2 = nn.Linear(nh, nf, bias=bias)
    nn.init.xavier_uniform_(layer_2.weight.data)
    if bias:
        nn.init.zeros_(layer_2.bias.data)
    layer = nn.Sequential(layer_1, layer_2)
    return layer