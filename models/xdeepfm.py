import torch
from torch import nn

from .field_encoder import FieldEncoder
from .deepfm import LinearBlock, FM, PlainDNN


class CINLayer(nn.Module):

    def __init__(self, n_fields, m_fields, o_fields, dropout=0., use_bn=True, bias=False):
        super().__init__()
        self.layer = LinearBlock(n_fields * m_fields, o_fields, residual=False,
                                 use_bn=use_bn, dropout=dropout, act=None, bias=bias)

    def forward(self, x, y):
        # x : [bs, dim, n]
        # y : [bs, dim, m]
        bs, dim, _ = x.size()
        z = x.unsqueeze(-1) @ y.unsqueeze(-2)  # [bs, dim, n, m]
        z = z.view([bs, dim, -1])
        z = self.layer(z)  # [bs, dim, o]
        return z


class CIN(nn.Module):

    def __init__(self, n_fields, net_fields, in_size, out_size,
                 bias=False, use_act=False, slope=0.01, dropout=0.,
                 use_bn=True):
        super().__init__()
        net_fields = [n_fields] + net_fields
        layers = []
        for i in range(len(net_fields) - 1):
            m, o = net_fields[i], net_fields[i+1]
            layers.append(CINLayer(n_fields, m, o,
                                   dropout=dropout,
                                   use_bn=use_bn,
                                   bias=bias))
        self.layers = nn.ModuleList(layers)
        if use_act:
            self.act = nn.LeakyReLU(slope)
        else:
            self.act = None
        self.linear = LinearBlock(in_size * (len(net_fields)-1), out_size,
                                  residual=False, use_bn=use_bn, dropout=dropout)

    def forward(self, x):
        # x : [bs, dim, n]
        prev_x = x
        ps = []
        for l in self.layers:
            prev_x = l(x, prev_x)  # [bs, dim, ?]
            if self.act is not None:
                prev_x = self.act(prev_x)
            ps.append(prev_x.sum(dim=-1))
        ps = torch.cat(ps, dim=-1)  # [bs, dim * n_layers]
        z = self.linear(ps)
        return z


class xDeepFM(nn.Module):

    def __init__(self, emb_size, net_dims, cin_fields,
                 onehot_vocab_sizes=None, sparse_input_sizes=None, dense_input_sizes=None,
                 pretrained_onehot=None, pretrained_sparse=None,
                 pretrained_bias=False, pretrained_finetune=False, padding_idx=None,
                 cin_bias=False, cin_use_act=False, use_fm=False, use_weights=False,
                 residual=True, use_bn=True, slope=0.2, emb_dropout=0.5, dropout=0.5):
        super().__init__()
        self.encoder = FieldEncoder(emb_size, onehot_vocab_sizes, sparse_input_sizes, dense_input_sizes,
                                    pretrained_onehot, pretrained_sparse,
                                    pretrained_bias, pretrained_finetune,
                                    padding_idx, emb_dropout)
        self.n_fields = self.encoder.n_fields

        self.use_fm = use_fm
        if use_fm:
            self.linear_encoder = FieldEncoder(1, onehot_vocab_sizes, sparse_input_sizes, dense_input_sizes,
                                               padding_idx=padding_idx, emb_dropout=0.)
            self.fm_layer = FM(False)
        self.deep_layer = PlainDNN(emb_size * self.n_fields, net_dims=net_dims,
                                   residual=residual, use_bn=use_bn,
                                   dropout=dropout, slope=slope)
        self.cin_layer = CIN(self.n_fields, cin_fields, in_size=emb_size, out_size=net_dims[-1],
                             bias=cin_bias, use_act=cin_use_act, slope=slope, dropout=dropout, use_bn=use_bn)
        self.use_weights = use_weights
        if use_weights:
            self.logits = nn.Parameter(torch.FloatTensor(3 if use_fm else 2))
            nn.init.zeros_(self.logits)

    def forward(self, ids=None, sparse_xs=None, dense_xs=None):
        embs = self.encoder(ids, sparse_xs, dense_xs)
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
            # FM Part
            linear_part = self.linear_encoder(ids, sparse_xs, dense_xs)
            fm_score = self.fm_layer(linear_part, embs)
            z = z_deep * weights[0] + z_cin * weights[1] + fm_score.unsqueeze(-1) * weights[2]
        else:
            z = z_deep * weights[0] + z_cin * weights[1]
        return z
