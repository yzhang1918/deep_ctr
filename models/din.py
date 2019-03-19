import numpy as np

import torch
from torch import nn

from .dice import Dice
from .field_encoder import FieldEncoder
from .deepfm import PlainDNN
from .xdeepfm import CIN


class HistEncoder(nn.Module):

    def __init__(self, n_fields, emb_size, act_fn='dice', slope=0.01):
        super().__init__()
        if act_fn == 'dice':
            self.act = Dice(emb_size)
        elif act_fn == 'prelu':
            self.act = nn.PReLU(emb_size)
        elif act_fn == 'lrelu':
            self.act = nn.LeakyReLU(slope)
        else:
            raise NotImplementedError

        self.score_layer = nn.Sequential(nn.Linear(n_fields * emb_size * 4, emb_size),
                                         self.act,
                                         nn.Linear(emb_size, 1))
        self.default_hist = nn.Parameter(torch.zeros([1, n_fields * emb_size]),
                                         requires_grad=True)

    def forward(self, i_embs, hist_embs, hist_len):
        assert np.all(np.equal(hist_len, hist_len[0]))
        # i_embs [bs, n, d]
        # hist_embs [bs * k, n, d]
        bs, n_fields, emb_size = i_embs.size()
        k = hist_len[0]
        h = hist_embs.view(bs, k, n_fields * emb_size)  # [bs, k, nd]
        x = i_embs.view(bs, 1, n_fields * emb_size)  # [bs, 1, nd]
        diff = x - h
        prod = x * h
        concat = torch.cat([x.expand(-1, k, -1), h, diff, prod], -1)  # [bs, k, 4nd]
        w = self.score_layer(concat)  # [bs, k, 1]
        z = (h * w).sum(dim=1)  # [bs, nd]
        z = z.view(bs, n_fields, emb_size)
        return z


class DINEncoder(nn.Module):

    def __init__(self, emb_size, user_config, item_config, context_config, act_fn='dice', slope=0.01, emb_dropout=0.1,
                 padding_idx=None):
        super().__init__()
        self.emb_size = emb_size
        self.user_encoder = FieldEncoder(emb_size, padding_idx=padding_idx, emb_dropout=emb_dropout, **user_config)
        self.item_encoder = FieldEncoder(emb_size, padding_idx=padding_idx, emb_dropout=emb_dropout, **item_config)
        self.cont_encoder = FieldEncoder(emb_size, padding_idx=padding_idx, emb_dropout=emb_dropout, **context_config)
        self.hist_encoder = HistEncoder(self.item_encoder.n_fields, emb_size, act_fn=act_fn, slope=slope)
        self.n_fields = self.user_encoder.n_fields + self.cont_encoder.n_fields + 2 * self.item_encoder.n_fields

    def forward(self, user_features, item_features, context_features, hist_features, hist_len):
        u_embs = self.user_encoder(*user_features)  # [bs, dim, *nf]
        i_embs = self.item_encoder(*item_features)  # [bs, dim, *nf]
        c_embs = self.cont_encoder(*context_features)  # [bs, dim, *nf]
        hist_embs = self.item_encoder(*hist_features)  # [*, dim, n_fields]
        hist_embs = self.hist_encoder(i_embs, hist_embs, hist_len)

        embs = torch.cat([u_embs, i_embs, c_embs, hist_embs], dim=-1)
        return embs


class DIN(nn.Module):

    def __init__(self, emb_size, net_dims, cin_fields, user_config, item_config, context_config, padding_idx=None,
                 cin_bias=False, cin_use_act=False, use_fm=False, use_weights=False, act_fn='dice',
                 residual=True, use_bn=True, slope=0.2, emb_dropout=0.1, dropout=0.5):
        super().__init__()
        self.encoder = DINEncoder(emb_size, user_config, item_config, context_config, act_fn=act_fn, slope=slope,
                                  emb_dropout=emb_dropout, padding_idx=padding_idx)
        self.n_fields = self.encoder.n_fields

        self.use_fm = use_fm
        if use_fm:
            raise NotImplementedError
        self.deep_layer = PlainDNN(emb_size * self.n_fields, net_dims=net_dims,
                                   residual=residual, use_bn=use_bn,
                                   dropout=dropout, slope=slope)
        self.cin_layer = CIN(self.n_fields, cin_fields, in_size=emb_size, out_size=net_dims[-1],
                             bias=cin_bias, use_act=cin_use_act, slope=slope, dropout=dropout, use_bn=use_bn)
        self.use_weights = use_weights
        if use_weights:
            self.logits = nn.Parameter(torch.FloatTensor(3 if use_fm else 2))
            nn.init.zeros_(self.logits)

    def forward(self, user_features, item_features, context_features, hist_featurs, hist_len):
        embs = self.encoder(user_features, item_features, context_features, hist_featurs, hist_len)
        # Deep Part
        bs = embs.size(0)
        z_deep = self.deep_layer(embs.view(bs, -1))  # [bs, output_size]

        # CIN
        z_cin = self.cin_layer(embs)

        if self.use_weights:
            weights = torch.softmax(self.logits, dim=0)
        else:
            weights = [1., 1., 1.]

        if self.use_fm:
            raise NotImplementedError
        else:
            z = z_deep * weights[0] + z_cin * weights[1]
        return z