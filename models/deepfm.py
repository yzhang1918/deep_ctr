import torch
from torch import nn

from models.layers import embedding, pretrained_embedding, pretrained_sparse_linear, SparseLinear, trunc_normal_


class DeepFM(nn.Module):

    def __init__(self, emb_size, net_dims,
                 onehot_vocab_sizes=None, sparse_input_sizes=None, dense_input_sizes=None,
                 pretrained_onehot=None, pretrained_sparse=None,
                 pretrained_bias=False, pretrained_finetune=False, padding_idx=None,
                 use_bn=True, slope=0.2, emb_dropout=0.5, dropout=0.5):
        super().__init__()
        onehot_vocab_sizes = onehot_vocab_sizes if onehot_vocab_sizes else []
        sparse_input_sizes = sparse_input_sizes if sparse_input_sizes else []
        dense_input_sizes = dense_input_sizes if dense_input_sizes else []
        pretrained_onehot = pretrained_onehot if pretrained_onehot else {}
        pretrained_sparse = pretrained_sparse if pretrained_sparse else {}
        self.emb_size = emb_size
        self.n_onehot_fields = len(onehot_vocab_sizes)
        self.n_sparse_fields = len(sparse_input_sizes)
        self.n_dense_fields = len(dense_input_sizes)
        self.n_fields = self.n_onehot_fields + self.n_sparse_fields + self.n_dense_fields
        assert self.n_fields > 0
        # FM
        kwargs = dict(pretrained_bias=pretrained_bias, pretrained_finetune=pretrained_finetune)
        onehot_v, onehot_b = self.create_emb_tables(emb_size=emb_size,
                                                    vocab_sizes=onehot_vocab_sizes,
                                                    weights_dict=pretrained_onehot,
                                                    padding_idx=padding_idx,
                                                    **kwargs)
        sparse_v, sparse_b = self.create_sparse_projs(emb_size=emb_size,
                                                      input_sizes=sparse_input_sizes,
                                                      weights_dict=pretrained_sparse,
                                                      **kwargs)
        dense_v, dense_b = self.create_dense_projs(emb_size=emb_size, input_sizes=dense_input_sizes)

        self.onehot_b_emb_layers = nn.ModuleList(onehot_b)
        self.onehot_v_emb_layers = nn.ModuleList(onehot_v)
        self.sparse_b_emb_layers = nn.ModuleList(sparse_b)
        self.sparse_v_emb_layers = nn.ModuleList(sparse_v)
        self.dense_b_emb_layers = nn.ModuleList(dense_b)
        self.dense_v_emb_layers = nn.ModuleList(dense_v)

        self.emb_dropout_layer = nn.Dropout(emb_dropout)

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

    @staticmethod
    def create_emb_tables(emb_size, vocab_sizes, weights_dict,
                          pretrained_bias, padding_idx, pretrained_finetune):
        bias_emb_layers = []
        vec_emb_layers = []
        for i, vsize in enumerate(vocab_sizes):
            # first order
            b_emb_layer = nn.Embedding(vsize, 1, padding_idx=padding_idx)
            nn.init.zeros_(b_emb_layer.weight)
            bias_emb_layers.append(b_emb_layer)
            # second order
            weights = weights_dict.get(i, None)
            if weights is not None:
                vec_emb_layers.append(pretrained_embedding(vsize, emb_size, weights,
                                                           bias=pretrained_bias,
                                                           padding_idx=padding_idx,
                                                           requires_grad=pretrained_finetune))
            else:
                vec_emb_layers.append(embedding(vsize, emb_size, padding_idx))
        return vec_emb_layers, bias_emb_layers

    @staticmethod
    def create_sparse_projs(emb_size, input_sizes, weights_dict, pretrained_bias, pretrained_finetune):
        bias_proj_layers = []
        vec_proj_layers = []
        for i, vsize in enumerate(input_sizes):
            # first order
            b_proj_layer = SparseLinear(vsize, 1, bias=False)
            nn.init.zeros_(b_proj_layer.weight.data)
            bias_proj_layers.append(b_proj_layer)
            # second layer
            weights = weights_dict.get(i, None)
            if weights is not None:
                v_proj_layer = pretrained_sparse_linear(vsize, emb_size, weights,
                                                        bias=pretrained_bias,
                                                        requires_grad=pretrained_finetune)
            else:
                v_proj_layer = SparseLinear(vsize, emb_size, bias=False)
                trunc_normal_(v_proj_layer.weight.data, std=0.01)
            vec_proj_layers.append(v_proj_layer)
        return vec_proj_layers, bias_proj_layers

    @staticmethod
    def create_dense_projs(emb_size, input_sizes):
        bias_proj_layers = []
        vec_proj_layers = []
        for i, vsize in enumerate(input_sizes):
            # first order
            b_proj_layer = nn.Linear(vsize, 1, bias=False)
            nn.init.zeros_(b_proj_layer.weight.data)
            bias_proj_layers.append(b_proj_layer)
            # second order
            v_proj_layer = nn.Linear(vsize, emb_size)
            nn.init.xavier_uniform_(v_proj_layer.weight.data)
            nn.init.zeros_(v_proj_layer.bias.data)
            vec_proj_layers.append(v_proj_layer)
        return vec_proj_layers, bias_proj_layers

    def get_embs(self, xs, order, input_type):
        if order == 1:
            layers = getattr(self, f'{input_type}_b_emb_layers')
        elif order == 2:
            layers = getattr(self, f'{input_type}_v_emb_layers')
        else:
            raise NotImplementedError
        embs = [layer(x) for x, layer in zip(xs, layers)]
        embs = torch.stack(embs, dim=-1)  # [bs, emb_size/1, n_*_fields
        if order == 2:
            embs = self.emb_dropout_layer(embs)
        return embs

    def forward(self, onehot_ids=None, sparse_xs=None, dense_xs=None):
        # ids : [bs, n_onehot_fields]
        # sparse_xs : list of [bs, *]
        # dense_xs : list of [bs, *]
        # get latent representations that are used in FM Part and Deep Part
        embs = []
        if onehot_ids is not None:
            embs.append(self.get_embs([onehot_ids[:, i] for i in range(onehot_ids.size(1))],
                                      order=2, input_type='onehot'))
        if sparse_xs is not None:
            embs.append(self.get_embs(sparse_xs, order=2, input_type='sparse'))
        if dense_xs is not None:
            embs.append(self.get_embs(dense_xs, order=2, input_type='dense'))

        embs = torch.cat(embs, dim=-1)  # [bs, emb_size, n_fields]

        # FM Part
        scores = []
        if onehot_ids is not None:
            scores.append(self.get_embs([onehot_ids[:, i] for i in range(onehot_ids.size(1))],
                                         order=1, input_type='onehot'))
        if sparse_xs is not None:
            scores.append(self.get_embs(sparse_xs, order=1, input_type='sparse'))
        if dense_xs is not None:
            scores.append(self.get_embs(dense_xs, order=1, input_type='dense'))
        scores = torch.cat(scores, dim=-1)  # [bs, 1, n_fields]

        order_2_score = (embs.sum(-1) ** 2).sum(-1) - (embs ** 2).sum(dim=[1, 2])
        fm_score = 0.5 * order_2_score + scores.sum(dim=[1, 2])  # [bs]

        # Deep Part
        bs = embs.size(0)
        z = self.net_layers(embs.view(bs, -1))  # [bs, output_size]

        z = z + fm_score.unsqueeze(1) + self.global_bias

        return z
