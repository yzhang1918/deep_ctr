import torch
from torch import nn

from .field_encoder import FieldEncoder


class FM(nn.Module):

    def __init__(self, bias=False):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor([]))
            nn.init.zeros_(self.bias.data)
        else:
            self.bias = None

    def forward(self, linear_part, embs):
        # linear_part : [bs, 1, n_fields]
        # embs : [bs, emb_size, n_fields]
        order_1_score = linear_part.sum([1, 2])
        order_2_score = (embs.sum(-1) ** 2).sum(-1) - (embs ** 2).sum(dim=[1, 2])
        fm_score = order_1_score + 0.5 * order_2_score
        if self.bias is not None:
            fm_score += self.bias
        return fm_score


class LinearBlock(nn.Module):

    def __init__(self, in_size, out_size, residual=False,
                 use_bn=True, dropout=0.5, bias=True, act=None):
        super().__init__()
        self.residual = residual if in_size == out_size else False
        self.bn = nn.BatchNorm1d(in_size) if use_bn else None
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_size, out_size, bias=bias))
        if act is not None:
            layers.append(act)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.bn is not None:
            if x.dim() == 2:
                x = self.bn(x)
            elif x.dim() == 3:
                x = self.bn(x.transpose(1, 2)).transpose(1, 2)
            else:
                raise NotImplementedError
        z = self.layers(x)
        if self.residual:
            z += x
        return z


class PlainDNN(nn.Module):

    def __init__(self, input_size, net_dims, residual=False, use_bn=True, dropout=0.5, slope=0.01):
        super().__init__()
        net_layers = []
        prev_dim = input_size
        for i, d in enumerate(net_dims):
            dropout_p = 0 if i == 0 else dropout
            act = nn.LeakyReLU(slope) if i < len(net_dims) - 1 else None
            net_layers.append(LinearBlock(prev_dim, d, residual=residual, use_bn=use_bn,
                                          dropout=dropout_p, act=act))
            prev_dim = d
        self.net_layers = nn.Sequential(*net_layers)

    def forward(self, x):
        return self.net_layers(x)


class DeepFM(nn.Module):

    def __init__(self, emb_size, net_dims,
                 onehot_vocab_sizes=None, sparse_input_sizes=None, dense_input_sizes=None,
                 pretrained_onehot=None, pretrained_sparse=None,
                 pretrained_bias=False, pretrained_finetune=False, padding_idx=None,
                 residual=True, use_bn=True, slope=0.2, emb_dropout=0.5, dropout=0.5):
        super().__init__()
        self.encoder = FieldEncoder(emb_size, onehot_vocab_sizes, sparse_input_sizes, dense_input_sizes,
                                    pretrained_onehot, pretrained_sparse,
                                    pretrained_bias, pretrained_finetune,
                                    padding_idx, emb_dropout)
        self.linear_encoder = FieldEncoder(1, onehot_vocab_sizes, sparse_input_sizes, dense_input_sizes,
                                           padding_idx=padding_idx, emb_dropout=0.)

        self.n_fields = self.encoder.n_fields
        self.fm_layer = FM(False)
        self.deep_layer = PlainDNN(emb_size * self.n_fields, net_dims=net_dims,
                                   residual=residual, use_bn=use_bn,
                                   dropout=dropout, slope=slope)

    def forward(self, ids=None, sparse_xs=None, dense_xs=None):
        embs = self.encoder(ids, sparse_xs, dense_xs)
        # FM Part
        linear_part = self.linear_encoder(ids, sparse_xs, dense_xs)
        fm_score = self.fm_layer(linear_part, embs)
        # Deep Part
        bs = embs.size(0)
        z = self.deep_layer(embs.view(bs, -1))  # [bs, output_size]

        z = z + fm_score.unsqueeze(1)
        return z
