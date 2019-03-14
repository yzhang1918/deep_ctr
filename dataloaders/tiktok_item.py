import numpy as np
import pandas as pd

import pathlib

from ..utils import loadpkl


class ItemFeatures:
    n_items = 4122690

    def __init__(self, root='data/feature', use_normed=True):
        self.root = pathlib.Path(root)
        # Load Data
        self.onehot_mat, self.sparse_mats, self.dense_mats = self.load_data(use_normed)
        # Get Size
        self.onehot_sizes = (self.onehot_mat.max(axis=0) + 1).tolist()
        self.sparse_sizes = [x.shape[1] for x in self.sparse_mats]
        self.dense_sizes = [x.shape[1] for x in self.dense_mats]

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

        face_fdf = pd.read_csv(root / f'face_features_{"norm" if use_normed else "raw"}.csv')
        face_fdf = face_fdf.reindex(range(-1, self.n_items - 1), fill_value=0)

        onehot_mat = basic_item_fdf.values + 1
        sparse_mats = [title_fspmat]
        dense_mats = [video_fmat, audio_fmat, face_fdf.values]
        return onehot_mat, sparse_mats, dense_mats

    @property
    def sizes(self):
        return self.onehot_sizes, self.sparse_sizes, self.dense_sizes

    def get(self, idx):
        return self.onehot_mat[idx], [x[idx] for x in self.sparse_mats], [x[idx] for x in self.dense_mats]

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self):
        return self.n_items

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


def bucketize_by_percentile(value, n_buckets):
    ps = np.linspace(0, 100, n_buckets + 1)[1: -1]
    bins = [value.min()] + [np.percentile(value, p) for p in ps] + [value.max()]
    ret = pd.cut(value, bins=bins, include_lowest=True, duplicates='drop')
    return ret.cat.codes
