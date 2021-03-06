from scipy import sparse as sp
import numpy as np
import pandas as pd

import pathlib
from numbers import Integral
from typing import Iterable

import torch
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader

from ..utils import loadpkl, parse_csr


def bucketize_by_percentile(value, n_buckets):
    ps = np.linspace(0, 100, n_buckets + 1)[1: -1]
    bins = [value.min()] + [np.percentile(value, p) for p in ps] + [value.max()]
    ret = pd.cut(value, bins=bins, include_lowest=True, duplicates='drop')
    return ret.cat.codes


def feat_to_tensor(feat):
    a, b, c = feat
    a = torch.from_numpy(a.astype(int, copy=False))
    b = [parse_csr(x) for x in b]
    c = [torch.from_numpy(x.astype(np.float32, copy=False)) for x in c]
    return a, b, c


class Features:

    def __init__(self, onehot_mat=None, sparse_mats=None, dense_mats=None,
                 onehot_names=None, sparse_names=None, dense_names=None,
                 infos=None):
        if infos is not None:
            self.n = infos['len']
        else:
            self.n = self.get_length(onehot_mat, sparse_mats, dense_mats)

        self.onehot_mat = onehot_mat if onehot_mat is not None else np.full([self.n, 0], 0, dtype=int)
        self.sparse_mats = sparse_mats if sparse_mats is not None else []
        self.dense_mats = dense_mats if dense_mats is not None else []

        if infos is not None:
            self.infos = infos
            self.onehot_sizes = infos['onehot_sizes']
            self.sparse_sizes = infos['sparse_sizes']
            self.dense_sizes = infos['dense_sizes']
            self.onehot_names = infos['onehot_names']
            self.sparse_names = infos['sparse_names']
            self.dense_names = infos['dense_names']

        else:
            # size
            if self.onehot_mat.shape[1] > 0:
                self.onehot_sizes = (self.onehot_mat.max(0) + 1).tolist()
            else:
                self.onehot_sizes = []
            self.sparse_sizes = [x.shape[1] for x in self.sparse_mats]
            self.dense_sizes = [x.shape[1] for x in self.dense_mats]

            self.onehot_names = onehot_names if onehot_names is not None else ['' for _ in self.onehot_sizes]
            self.sparse_names = sparse_names if sparse_names is not None else ['' for _ in self.sparse_sizes]
            self.dense_names = dense_names if dense_names is not None else ['' for _ in self.dense_sizes]

            self.infos = {'len': self.n,
                          'onehot_sizes': self.onehot_sizes,
                          'sparse_sizes': self.sparse_sizes,
                          'dense_sizes': self.dense_sizes,
                          'onehot_names': self.onehot_names,
                          'sparse_names': self.sparse_names,
                          'dense_names': self.dense_names,
                          }

    def __str__(self):
        s = self.__class__.__name__ + '['
        sizes = (f'Sizes(len={len(self)}, onehot={str(self.onehot_sizes)},'
                 f' sparse={str(self.sparse_sizes)}, dense={str(self.dense_sizes)})')
        s += sizes
        s += ']'
        return s

    __repr__ = __str__

    def __iter__(self):
        yield self.onehot_mat
        yield self.sparse_mats
        yield self.dense_mats

    @staticmethod
    def get_length(onehot_mat, sparse_mats, dense_mats):
        ns = []
        if onehot_mat is not None:
            ns.append(len(onehot_mat))
        if sparse_mats is not None and len(sparse_mats) > 0:
            for m in sparse_mats:
                ns.append(m.shape[0])
        if dense_mats is not None and len(dense_mats) > 0:
            for m in dense_mats:
                ns.append(m.shape[0])
        if len(ns) == 0:
            return 0
        if not np.all(np.array(ns) == ns[0]):
            raise ValueError('Inputs Lengths Mismatch!')
        return ns[0]

    @property
    def sizes(self):
        return self.onehot_sizes, self.sparse_sizes, self.dense_sizes

    @property
    def names(self):
        return self.onehot_names, self.sparse_names, self.dense_names

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if isinstance(i, Integral):
            if i >= len(self):
                raise IndexError('index out of range')
            if i < 0:
                raise IndexError('negative index is not supported')
            i = slice(i, i + 1)  # for consistence
        sub_onehot = self.onehot_mat[i]
        sub_sparse = [x[i] for x in self.sparse_mats]
        sub_dense = [x[i] for x in self.dense_mats]
        infos = self.infos.copy()
        infos['len'] = len(sub_onehot)
        return Features(sub_onehot, sub_sparse, sub_dense, infos=infos)

    @classmethod
    def concat_fields(cls, feat_list):
        # Update Infos
        lens = [len(f) for f in feat_list]
        assert np.all(np.array(lens) == lens[0])
        infos = {'len': lens[0]}
        for feat_type in ['onehot', 'sparse', 'dense']:
            for attr in ['sizes', 'names']:
                key = f'{feat_type}_{attr}'
                infos[key] = []
                for f in feat_list:
                    infos[key] += f.infos[key]
        new_onehot = np.concatenate([f.onehot_mat for f in feat_list], -1)
        new_sparse = []
        new_dense = []
        for f in feat_list:
            new_sparse.extend(f.sparse_mats)
            new_dense.extend(f.dense_mats)
        return cls(new_onehot, new_sparse, new_dense, infos=infos)

    @classmethod
    def empty(cls, n):
        return Features(np.zeros([n, 0], dtype=int), [], [])

    @classmethod
    def concat_records(cls, feat_list):
        infos = feat_list[0].infos.copy()
        infos['len'] = sum([len(f) for f in feat_list])
        new_onehot = np.concatenate([f.onehot_mat for f in feat_list], 0)
        # sparse
        new_sparse = [[] for _ in infos['sparse_sizes']]
        for f in feat_list:
            for l, x in zip(new_sparse, f.sparse_mats):
                l.append(x)
        new_sparse = [sp.vstack(x) for x in new_sparse]
        # dense
        new_dense = [[] for _ in infos['dense_sizes']]
        for f in feat_list:
            for l, x in zip(new_dense, f.dense_mats):
                l.append(x)
        new_dense = [np.concatenate(x, 0) for x in new_dense]
        return cls(new_onehot, new_sparse, new_dense, infos=infos)


class ItemFeatures(Features):
    n_items = 4122690

    def __init__(self, root='data/feature', use_normed=True):
        self.root = pathlib.Path(root)
        # Load Data
        rets = self.load_data(use_normed)
        super().__init__(*rets)

    def load_data(self, use_normed):
        root = self.root
        # Basic Features
        basic_item_fdf = pd.read_csv(root / 'basic_item_features.csv')
        basic_item_fdf.set_index('item_id', drop=True, inplace=True)
        basic_item_fdf = self.enhance_basic_item_features(basic_item_fdf)
        basic_item_fdf = basic_item_fdf.reindex(range(-1, self.n_items - 1), fill_value=-1)  # pad

        video_fmat = loadpkl(root / f'video_features_fill_zero{"_norm" if use_normed else ""}.pkl')
        audio_fmat = loadpkl(root / f'audio_features_fill_zero{"_norm" if use_normed else ""}.pkl')

        title_fspmat = loadpkl(root / 'title_csr_50.pkl')
        title_fspmat = normalize(title_fspmat, norm='l1', axis=1, copy=True).astype(np.float32)

        face_fdf = pd.read_csv(root / f'face_features_{"norm" if use_normed else "raw"}.csv')
        face_fdf = face_fdf.reindex(range(-1, self.n_items - 1), fill_value=0)

        onehot_mat = basic_item_fdf.values + 1
        sparse_mats = [title_fspmat]
        dense_mats = [video_fmat, audio_fmat, face_fdf.values.astype(np.float32)]

        onehot_names = basic_item_fdf.columns.tolist()
        sparse_names = ['title']
        dense_names = ['video', 'audio', 'face']

        return onehot_mat, sparse_mats, dense_mats, onehot_names, sparse_names, dense_names

    @staticmethod
    def enhance_basic_item_features(basic_item_fdf):
        # duration_time
        basic_item_fdf['duration_bucket'] = bucketize_by_percentile(basic_item_fdf.duration_time, 20)
        # time
        series_t = basic_item_fdf.time
        series_t //= 3600  # hour-level
        basic_item_fdf['hourinday'] = series_t % 24
        basic_item_fdf['dayinmonth'] = series_t // 24 % 30
        basic_item_fdf['dayinweek'] = series_t // 24 % 7
        basic_item_fdf['monthinyear'] = series_t // (24 * 30) % 12
        basic_item_fdf['time_bucket'] = bucketize_by_percentile(basic_item_fdf.time, 100)
        # Drop
        basic_item_fdf.drop(['time', 'duration_time'], axis=1, inplace=True)
        return basic_item_fdf


class UserFeatures(Features):
    n_users = 73975

    def __init__(self, use_uid, root):
        self.root = pathlib.Path(root)
        if use_uid:
            onehot_mat = np.arange(self.n_users)[:, None].astype(int)
            onehot_names = ['user_id']
        else:
            onehot_mat = np.zeros([self.n_users, 0], dtype=int)
            onehot_names = None
        clicked_video = loadpkl(root / 'user_clicked_video_norm.pkl')
        super().__init__(onehot_mat=onehot_mat, onehot_names=onehot_names,
                         dense_mats=[clicked_video], dense_names=['clicked_video'])


# The following three classes do not use history information.
class MainFeatures(Features):

    def __init__(self, df, dense_mats, dense_names, onehot_sizes, user_feats, item_feats):
        self.df = df
        self.user_feats = user_feats
        self.item_feats = item_feats
        onehot_mat = self.df[['user_city', 'channel']].values + 1
        super().__init__(onehot_mat=onehot_mat, onehot_names=['user_city', 'channel'],
                         dense_mats=dense_mats, dense_names=dense_names)
        # fix onehot sizes (IMPORTANT)
        self.onehot_sizes = onehot_sizes
        self.infos['onehot_sizes'] = onehot_sizes

        self.uids = self.df.uid.values + 1
        self.iids = self.df.item_id.values + 1
        # a little hack here
        tmp = self[0]
        self.onehot_sizes = tmp.onehot_sizes
        self.sparse_sizes = tmp.sparse_sizes
        self.dense_sizes = tmp.dense_sizes
        self.onehot_names = tmp.onehot_names
        self.sparse_names = tmp.sparse_names
        self.dense_names = tmp.dense_names
        # We do not update infos!

    def __getitem__(self, i):
        context_f = super().__getitem__(i)
        uids = self.uids[i]
        iids = self.iids[i]
        user_f = self.user_feats[uids]
        item_f = self.item_feats[iids]
        f = Features.concat_fields([context_f, user_f, item_f])
        if hasattr(user_f, 'hist_feats'):
            f.hist_feats = user_f.hist_feats
            f.hist_lens = user_f.hist_lens
        return f


class FeatDataset(Dataset):

    def __init__(self, feats, targets):
        self.feats = feats
        self.targets = targets

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i], self.targets[i]


class FeatureBatch:

    def __init__(self, batch):
        feats, y = zip(*batch)
        self.y = torch.from_numpy(np.stack(y, 0)).float()
        cat_feats = Features.concat_records(feats)
        self.onehot_ids, self.sparse_xs, self.dense_xs = feat_to_tensor(cat_feats)

    def __iter__(self):
        yield (self.onehot_ids, self.sparse_xs, self.dense_xs)
        yield self.y

    def cuda(self):
        self.onehot_ids = self.onehot_ids.cuda()
        self.sparse_xs = [[a.cuda(), b.cuda(), c] for a, b, c in self.sparse_xs]
        self.dense_xs = [x.cuda() for x in self.dense_xs]
        self.y = self.y.cuda()
        return self


# The following three classes are for DIN.
class UserHistoryFeatures(UserFeatures):
    n_users = 73975

    def __init__(self, item_feats, use_uid=False, max_len=None, sample_pad=False, root='data/feature'):
        self.max_len = max_len
        self.sample_pad = sample_pad
        self.root = pathlib.Path(root)
        self.item_feats = item_feats
        self.user_hist_dict = loadpkl(self.root / 'user_history_dict.pkl')
        # For now, we have no features.
        super().__init__(use_uid, root)

    def __len__(self):
        return self.n_users

    def __getitem__(self, i):
        if isinstance(i, Iterable):
            hist_info = self.get_batch_hist_data(i)
        else:
            hist_info = self.get_hist_data(i)
        f = super().__getitem__(i)
        f.hist_feats, f.hist_lens = hist_info
        return f

    def get_batch_hist_data(self, indices):
        lens = []
        data = []
        for i in indices:
            hist_feat, hist_l = self.get_hist_data(i)
            data.append(hist_feat)
            lens.append(hist_l)
        f = Features.concat_records(data)
        return f, lens

    def get_hist_data(self, i):
        hist = self.user_hist_dict.get(i - 1, [])  # padding idx = 0
        if self.max_len is None:
            hist = hist
        elif self.sample_pad:
            hist = np.random.choice(hist, size=self.max_len, replace=True)
        elif len(hist) > self.max_len:
            hist = np.random.choice(hist, size=self.max_len, replace=False)
        else:
            raise NotImplementedError
        hist_data = self.item_feats[np.add(hist, 1)]
        return hist_data, len(hist)


class ContextFeatures(Features):

    def __init__(self, df, dense_mats, dense_names, onehot_sizes):
        self.df = df
        onehot_mat = self.df[['user_city', 'channel']].values + 1
        super().__init__(onehot_mat=onehot_mat, onehot_names=['user_city', 'channel'],
                         dense_mats=dense_mats, dense_names=dense_names)
        # fix onehot sizes (IMPORTANT)
        self.onehot_sizes = onehot_sizes
        self.infos['onehot_sizes'] = onehot_sizes

        self.uids = self.df.uid.values + 1
        self.iids = self.df.item_id.values + 1

    def get_uid_iid(self, i):
        uids = self.uids[i]
        iids = self.iids[i]
        return uids, iids


class HistFeatDataset(Dataset):

    def __init__(self, user_feats, item_feats, context_feats, targets):
        self.user_feats = user_feats
        self.item_feats = item_feats
        self.context_feats = context_feats
        self.targets = targets

    def __len__(self):
        return len(self.context_feats)

    def __getitem__(self, i):
        c_fs = self.context_feats[i]
        uids, iids = self.context_feats.get_uid_iid(i)
        u_fs = self.user_feats[uids]
        i_fs = self.item_feats[iids]
        return (u_fs, i_fs, c_fs, u_fs.hist_feats, u_fs.hist_lens), self.targets[i]


class HistFeatureBatch:

    def __init__(self, batch):
        hete_feats, y = zip(*batch)
        self.y = torch.from_numpy(np.stack(y, 0)).float()

        u_fs, i_fs, c_fs, h_fs, self.h_lens = zip(*hete_feats)

        self.u_fs = feat_to_tensor(Features.concat_records(u_fs))
        self.i_fs = feat_to_tensor(Features.concat_records(i_fs))
        self.c_fs = feat_to_tensor(Features.concat_records(c_fs))
        self.h_fs = feat_to_tensor(Features.concat_records(h_fs))

    def __iter__(self):
        yield (self.u_fs, self.i_fs, self.c_fs, self.h_fs, self.h_lens)
        yield self.y

    def _to_cuda(self, feat):
        onehot_ids, sparse_xs, dense_xs = feat
        onehot_ids = onehot_ids.cuda()
        sparse_xs = [[a.cuda(), b.cuda(), c] for a, b, c in sparse_xs]
        dense_xs = [x.cuda() for x in dense_xs]
        return onehot_ids, sparse_xs, dense_xs

    def cuda(self):
        self.u_fs = self._to_cuda(self.u_fs)
        self.i_fs = self._to_cuda(self.i_fs)
        self.c_fs = self._to_cuda(self.c_fs)
        self.h_fs = self._to_cuda(self.h_fs)
        self.y = self.y.cuda()
        return self


def get_dataloader(bs=128, test_bs=None, use_uid=True, num_workers=8, root='data/feature'):
    test_bs = test_bs if test_bs else bs
    root = pathlib.Path(root)
    item_feats = ItemFeatures(root=root)
    user_feats = UserFeatures(use_uid, root=root)

    main_df = pd.read_csv(root / 'sample_split_slim.csv')
    train_idx = main_df.train == 1
    valid_idx = main_df.valid == 1
    test_idx = main_df.test == 1
    cols = ['uid', 'item_id', 'user_city', 'channel']
    onehot_sizes = (main_df[['user_city', 'channel']].values.max(0) + 2).tolist()

    # Extra Features provided by fhj
    dense_dfs = loadpkl(root / 'processed_features.pkl')
    train_dense = [df[train_idx].values for df in dense_dfs]
    valid_dense = [df[valid_idx].values for df in dense_dfs]
    test_dense = [df[test_idx].values for df in dense_dfs]
    dense_names = ['ex_author', 'ex_icity', 'ex_item', 'ex_music', 'ex_ucity', 'ex_user']

    train_feats = MainFeatures(main_df.loc[train_idx, cols], train_dense, dense_names,
                               onehot_sizes, user_feats, item_feats)
    valid_feats = MainFeatures(main_df.loc[valid_idx, cols], valid_dense, dense_names,
                               onehot_sizes, user_feats, item_feats)
    test_feats = MainFeatures(main_df.loc[test_idx, cols], test_dense, dense_names,
                              onehot_sizes, user_feats, item_feats)

    train_ds = FeatDataset(train_feats, main_df.loc[train_idx, ['finish', 'like']].values)
    valid_ds = FeatDataset(valid_feats, main_df.loc[valid_idx, ['finish', 'like']].values)
    test_ds = FeatDataset(test_feats, main_df.loc[test_idx, ['finish', 'like']].values)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
                          drop_last=False, collate_fn=FeatureBatch)
    valid_dl = DataLoader(valid_ds, batch_size=test_bs, num_workers=num_workers, collate_fn=FeatureBatch)
    test_dl = DataLoader(test_ds, batch_size=test_bs, num_workers=num_workers, collate_fn=FeatureBatch)

    sizes = train_feats.sizes

    return train_dl, valid_dl, test_dl, sizes


def get_hist_dataloader(bs=128, test_bs=None, use_uid=True, max_len=300, sample_pad=True, num_workers=8,
                        root='data/feature'):
    test_bs = test_bs if test_bs else bs
    root = pathlib.Path(root)
    item_feats = ItemFeatures(root=root)
    user_feats = UserHistoryFeatures(item_feats, use_uid=use_uid, max_len=max_len, sample_pad=sample_pad, root=root)

    main_df = pd.read_csv(root / 'sample_split_slim.csv')
    train_idx = main_df.train == 1
    valid_idx = main_df.valid == 1
    test_idx = main_df.test == 1
    cols = ['uid', 'item_id', 'user_city', 'channel']
    onehot_sizes = (main_df[['user_city', 'channel']].values.max(0) + 2).tolist()

    # Extra Features provided by fhj
    dense_dfs = loadpkl(root / 'processed_features.pkl')
    train_dense = [df[train_idx].values for df in dense_dfs]
    valid_dense = [df[valid_idx].values for df in dense_dfs]
    test_dense = [df[test_idx].values for df in dense_dfs]
    dense_names = ['ex_author', 'ex_icity', 'ex_item', 'ex_music', 'ex_ucity', 'ex_user']

    # Context Features
    train_cont_feats = ContextFeatures(main_df.loc[train_idx, cols], train_dense, dense_names, onehot_sizes)
    valid_cont_feats = ContextFeatures(main_df.loc[valid_idx, cols], valid_dense, dense_names, onehot_sizes)
    test_cont_feats = ContextFeatures(main_df.loc[test_idx, cols], test_dense, dense_names, onehot_sizes)

    # Dataset
    train_ds = HistFeatDataset(user_feats, item_feats, train_cont_feats,
                               main_df.loc[train_idx, ['finish', 'like']].values)
    valid_ds = HistFeatDataset(user_feats, item_feats, valid_cont_feats,
                               main_df.loc[valid_idx, ['finish', 'like']].values)
    test_ds = HistFeatDataset(user_feats, item_feats, test_cont_feats, main_df.loc[test_idx, ['finish', 'like']].values)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=False,
                          collate_fn=HistFeatureBatch)
    valid_dl = DataLoader(valid_ds, batch_size=test_bs, num_workers=num_workers, collate_fn=HistFeatureBatch)
    test_dl = DataLoader(test_ds, batch_size=test_bs, num_workers=num_workers, collate_fn=HistFeatureBatch)

    u_config = {k: v for k, v in
                zip(['onehot_vocab_sizes', 'sparse_input_sizes', 'dense_input_sizes'], user_feats.sizes)}
    i_config = {k: v for k, v in
                zip(['onehot_vocab_sizes', 'sparse_input_sizes', 'dense_input_sizes'], item_feats.sizes)}
    c_config = {k: v for k, v in
                zip(['onehot_vocab_sizes', 'sparse_input_sizes', 'dense_input_sizes'], valid_cont_feats.sizes)}

    return train_dl, valid_dl, test_dl, u_config, i_config, c_config
