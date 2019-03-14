import torch
from torch import nn

from .layers import (embedding, pretrained_embedding, pretrained_sparse_linear,
                     SparseLinear, trunc_normal_)


class FieldEncoder(nn.Module):

    def __init__(self, emb_size, onehot_vocab_sizes=None,
                 sparse_input_sizes=None, dense_input_sizes=None,
                 pretrained_onehot=None, pretrained_sparse=None,
                 pretrained_bias=False, pretrained_finetune=False,
                 padding_idx=None, emb_dropout=0.1):
        super().__init__()
        onehot_vocab_sizes = onehot_vocab_sizes if onehot_vocab_sizes else []
        sparse_input_sizes = sparse_input_sizes if sparse_input_sizes else []
        dense_input_sizes = dense_input_sizes if dense_input_sizes else []

        self.emb_size = emb_size
        self.n_onehot_fields = len(onehot_vocab_sizes)
        self.n_sparse_fields = len(sparse_input_sizes)
        self.n_dense_fields = len(dense_input_sizes)
        self.n_fields = self.n_onehot_fields + self.n_sparse_fields + self.n_dense_fields
        assert self.n_fields > 0

        if self.n_onehot_fields:
            self.onehot_encoder = OnehotEncoderList(emb_size, onehot_vocab_sizes,
                                                    padding_idx=padding_idx,
                                                    pretrained_weights=pretrained_onehot,
                                                    pretrained_finetune=pretrained_finetune,
                                                    pretrained_bias=pretrained_bias)
        if self.n_sparse_fields:
            self.sparse_encoder = SparseEncoderList(emb_size, sparse_input_sizes,
                                                    pretrained_weights=pretrained_sparse,
                                                    pretrained_finetune=pretrained_finetune,
                                                    pretrained_bias=pretrained_bias)
        if self.n_dense_fields:
            self.dense_encoder = DenseEncoderList(emb_size, dense_input_sizes)
        self.emb_dropout_layer = nn.Dropout(emb_dropout)

    def forward(self, ids=None, sparse_xs=None, dense_xs=None):
        embs = []
        if ids is not None:
            embs.append(self.onehot_encoder(ids))
        if sparse_xs is not None:
            embs.append(self.sparse_encoder(sparse_xs))
        if dense_xs is not None:
            embs.append(self.dense_encoder(dense_xs))
        embs = torch.cat(embs, dim=-1)  # [bs, emb_size, n_fields]
        embs = self.emb_dropout_layer(embs)
        return embs


class SparseEncoderList(nn.Module):

    def __init__(self, emb_size, input_sizes,
                 pretrained_weights=None, pretrained_finetune=False, pretrained_bias=False):
        super().__init__()
        pretrained_weights = pretrained_weights if pretrained_weights else {}
        layers = []
        for i, vsize in enumerate(input_sizes):
            if emb_size == 1:
                layer = SparseLinear(vsize, 1, bias=False)
                nn.init.zeros_(layer.weight.data)
            else:
                weight = pretrained_weights.get(i, None)
                if weight is not None:
                    layer = pretrained_sparse_linear(vsize, emb_size, weight,
                                                     bias=pretrained_bias,
                                                     requires_grad=pretrained_finetune)
                else:
                    layer = SparseLinear(vsize, emb_size, bias=False)
                    trunc_normal_(layer.weight.data, std=0.01)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, xs):
        zs = [l(x) for x, l in zip(xs, self.layers)]
        z = torch.stack(zs, dim=-1)
        return z


class DenseEncoderList(nn.Module):

    def __init__(self, emb_size, input_sizes):
        super().__init__()
        layers = []
        for i, vsize in enumerate(input_sizes):
            if emb_size == 1:
                layer = nn.Linear(vsize, emb_size, bias=False)
                nn.init.zeros_(layer.weight.data)
            else:
                layer = nn.Linear(vsize, emb_size)
                nn.init.xavier_uniform_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, xs):
        zs = [l(x) for x, l in zip(xs, self.layers)]
        z = torch.stack(zs, dim=-1)
        return z


class OnehotEncoderList(nn.Module):

    def __init__(self, emb_size, vocab_sizes, padding_idx=0,
                 pretrained_weights=None, pretrained_finetune=False, pretrained_bias=True):
        super().__init__()
        layers = []
        pretrained_weights = pretrained_weights if pretrained_weights else {}
        for i, vsize in enumerate(vocab_sizes):
            if emb_size == 1:
                emb = nn.Embedding(vsize, 1, padding_idx=padding_idx)
                nn.init.zeros_(emb.weight.data)
            else:
                weight = pretrained_weights.get(i, None)
                if weight is not None:
                    emb = pretrained_embedding(vsize, emb_size, weight,
                                               bias=pretrained_bias,
                                               padding_idx=padding_idx,
                                               requires_grad=pretrained_finetune)
                else:
                    emb = embedding(vsize, emb_size, padding_idx=padding_idx)
            layers.append(emb)
        self.layers = nn.ModuleList(layers)

    def forward(self, ids):
        ids = [ids[:, i] for i in range(ids.size(1))]
        zs = [l(x) for x, l in zip(ids, self.layers)]
        z = torch.stack(zs, dim=-1)
        return z
