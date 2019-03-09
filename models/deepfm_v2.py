import torch
from torch import nn

from .field_encoder import FieldEncoder


class DeepFMv2(nn.Module):

    def __init__(self, emb_size, net_dims,
                 onehot_vocab_sizes=None, sparse_input_sizes=None, dense_input_sizes=None,
                 pretrained_onehot=None, pretrained_sparse=None,
                 pretrained_bias=False, pretrained_finetune=False, padding_idx=None,
                 use_bn=True, slope=0.2, emb_dropout=0.5, dropout=0.5):
        super().__init__()
        self.encoder = FieldEncoder(emb_size, onehot_vocab_sizes, sparse_input_sizes, dense_input_sizes,
                                    pretrained_onehot, pretrained_sparse,
                                    pretrained_bias, pretrained_finetune,
                                    padding_idx, emb_dropout)
        self.linear_encoder = FieldEncoder(1, onehot_vocab_sizes, sparse_input_sizes, dense_input_sizes,
                                           padding_idx=padding_idx, emb_dropout=0.)

        self.n_fields = self.encoder.n_fields
        # Deep
        net_layers = []
        prev_dim = emb_size * self.n_fields
        for i, d in enumerate(net_dims):
            if use_bn:
                net_layers.append(nn.BatchNorm1d(prev_dim))
            if i != 0:
                net_layers.append(nn.Dropout(dropout))
            net_layers.append(nn.Linear(prev_dim, d))
            prev_dim = d
            if i < len(net_dims) - 1:
                net_layers.append(nn.LeakyReLU(slope))
        self.net_layers = nn.Sequential(*net_layers)

        # Global Bias
        self.global_bias = nn.Parameter(torch.FloatTensor(net_dims[-1]))
        nn.init.zeros_(self.global_bias.data)

    def forward(self, ids, sparse_xs, dense_xs):
        embs = self.encoder(ids, sparse_xs, dense_xs)
        # FM Part
        linear_part = self.linear_encoder(ids, sparse_xs, dense_xs)
        fm_score = calc_fm_score(linear_part, embs)
        # Deep Part
        bs = embs.size(0)
        z = self.net_layers(embs.view(bs, -1))  # [bs, output_size]

        z = z + fm_score.unsqueeze(1) + self.global_bias
        return z


def calc_fm_score(linear_part, embs):
    # linear_part : [bs, 1, n_fields]
    # embs : [bs, emb_size, n_fields]
    order_1_score = linear_part.sum([1, 2])
    order_2_score = (embs.sum(-1) ** 2).sum(-1) - (embs ** 2).sum(dim=[1, 2])
    fm_score = order_1_score + 0.5 * order_2_score
    return fm_score
