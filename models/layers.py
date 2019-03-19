import torch
from torch import nn
from torch_sparse import spmm


class ColdStartEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.,
                 scale_grad_by_freq=False, sparse=False, _weight=None, use_mean=False):
        super().__init__(num_embeddings, embedding_dim, padding_idx,
                         max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        self.use_mean = use_mean if self.padding_idx is not None else True
        self.register_buffer('flags', torch.zeros(self.weight.size(0), dtype=torch.uint8))

    def forward(self, ids):
        if self.training:
            self.flags.scatter_(0, ids.flatten(), 1)
        return super().forward(ids)

    def eval(self):
        with torch.no_grad():
            seen = self.weight[self.flags]
            if self.use_mean:
                if seen.size(0) > 0:
                    mu = seen.mean(0)
                    self.weight.data[~self.flags] = mu
            else:
                mu = self.weight.data[self.padding_idx]
                self.weight.data[~self.flags] = mu
        super().eval()


class SparseLinear(nn.Linear):

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            ind, val, size = x
        else:
            ind, val, size = x._indices(), x._values(), x.size(0)
        y = spmm(ind, val, size, self.weight.t())
        if self.bias is not None:
            y += self.bias
        return y


def trunc_normal_(x, mean=0., std=1.):
    # From Fast.ai
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def embedding(ni, nf, padding_idx=None):
    # From Fast.ai
    emb = ColdStartEmbedding(ni, nf, padding_idx=padding_idx)
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
