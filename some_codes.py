
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                        PROCESS CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np


# ------------------------------------------------------------------------------
#  remove outliers from log2 transformed labels and features


INLIERS_LOG2 = {
    'label': {'2pow': -6, 'min': -3.0, 'max': 14.0},
    'S1': {
        0: {'2pow': None, 'min': -25.0, 'max': 30.0},
        1: {'2pow': None, 'min': -63.0, 'max': 29.0},
        2: {'2pow': None, 'min': -25.0, 'max': 32.0},
        3: {'2pow': None, 'min': -70.0, 'max': 23.0},
    },
    'S2': {
        0:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        1:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        2:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        3:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        4:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        5:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        6:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        7:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        8:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        9:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        10: {'2pow': None, 'min': 0.0, 'max': 100.0},
    }
}


def remove_outliers_by_log2(data, data_name, data_index=None):

    if data_name == 'label':
        inlier_dict = INLIERS_LOG2['label']
    else:
        inlier_dict = INLIERS_LOG2[data_name][data_index]

    if inlier_dict['2pow'] is not None:
        min_thresh = 2 ** inlier_dict['2pow']
        data = np.where(data < min_thresh, min_thresh, data)
        data = np.log2(data)

    min_ = inlier_dict['min']
    max_ = inlier_dict['max']

    if (data_name == 'S2') and (data_index == 10):
        data = np.where(data < min_, min_, data)
        data = np.where(data > max_, min_, data)
    else:
        data = np.where(data < min_, min_, data)
        data = np.where(data > max_, max_, data)

    return data


# ------------------------------------------------------------------------------
#  remove outliers from plain labels and features


INLIERS_PLAIN = {
    'S1': {
        0: {'min': -25.0, 'max': 30.0},
        1: {'min': -63.0, 'max': 29.0},
        2: {'min': -25.0, 'max': 32.0},
        3: {'min': -70.0, 'max': 23.0},
    },
    'S2': {
        10: {'min': 0.0, 'max': 100.0},
    }
}


def remove_outliers_by_plain(data, data_name, data_index=None):

    if data_name == 'label':
        data = np.where(data < 0.0, 0.0, data)

    elif data_name == 'S1':
        inlier_dict = INLIERS_PLAIN['S1'][data_index]
        min_ = inlier_dict['min']
        max_ = inlier_dict['max']
        data = np.where(data < min_, min_, data)
        data = np.where(data > max_, max_, data)

    elif data_name == 'S2':
        if data_index == 10:
            inlier_dict = INLIERS_PLAIN['S2'][10]
            min_ = inlier_dict['min']
            max_ = inlier_dict['max']
            data = np.where(data < min_, min_, data)
            data = np.where(data > max_, min_, data)

    return 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from .clean import INLIERS_LOG2


GT_SHAPE = (1,  256, 256)
S1_SHAPE = (4,  256, 256)
S2_SHAPE = (11, 256, 256)


def read_raster(data_path, return_zeros=False, data_shape=None):

    if os.path.isfile(data_path):
        raster = rasterio.open(data_path)
        data = raster.read()
    else:
        if return_zeros:
            assert data_shape is not None
            data = np.zeros(data_shape).astype(np.float32)
        else:
            data = None

    return data


def calculate_statistics(
    data, data_name, exclude_mins=False,
    p=None, hist=False, plot_dir=None
):

    if exclude_mins:
        min_ = np.min(data) 
        data = data[np.where(data > min_)]

    if p is not None:
        assert 0 <= p <= 100
        data_min = np.percentile(data, 100 - p)
        data_max = np.percentile(data, p)
        data = np.where(data < data_min, data_min, data)
        data = np.where(data > data_max, data_max, data)
    else:
        data_min = np.min(data)
        data_max = np.max(data)

    data_avg = np.mean(data)
    data_std = np.std(data)

    print(f'Statistics of {data_name} with percentile {p}:')
    print(f'- min: {data_min:.3f}')
    print(f'- max: {data_max:.3f}')
    print(f'- avg: {data_avg:.3f}')
    print(f'- std: {data_std:.3f}')

    if hist:
        assert plot_dir is not None
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = f'stats_{data_name}_p{p}.png'
        plot_path = os.path.join(plot_dir, plot_file)

        plt.figure()
        plt.title(f'{data_name} - P:{p}')
        plt.hist(data.reshape(-1), bins=100, log=True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return {
        'min': data_min,
        'max': data_max,
        'avg': data_avg,
        'std': data_std,
    }


def normalize(data, norm_stats, norm_method):
    assert norm_method in ['minmax', 'zscore']

    min_ = norm_stats['min']
    max_ = norm_stats['max']
    data = np.where(data < min_, min_, data)
    data = np.where(data > max_, max_, data)

    if norm_method == 'minmax':
        range_ = max_ - min_
        data = (data - min_) / range_
    elif norm_method == 'zscore':
        avg = norm_stats['avg']
        std = norm_stats['std']
        data = (data - avg) / std

    return data


def recover_label(data, norm_stats, recover_method, norm_method='minmax'):
    assert norm_method in ['minmax', 'zscore']
    assert recover_method in ['log2', 'plain']

    if norm_method == 'minmax':
        min_ = norm_stats['min']
        max_ = norm_stats['max']
        range_ = max_ - min_
        data = data * range_ + min_
    else:
        avg = norm_stats['avg']
        std = norm_stats['std']
        data = data * std + avg

    if recover_method == 'log2':
        data = 2 ** data
        min_thresh = 2 ** INLIERS_LOG2['label']['2pow']
        data = np.where(data < min_thresh, 0, data)

    return 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                       DATASET & DATALOADER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

__all__ = ['get_dataloader']


import os
import numpy as np
import volumentations as V

from os.path import join as opj
from torch.utils.data import Dataset, DataLoader

from ..utils import *
from ..process import *


class BMDataset(Dataset):

    def __init__(
        self, mode, data_list, norm_stats, augment=False,
        process_method='plain', s1_index_list='all',
        s2_index_list='all', months_list='all'
    ):
        super(BMDataset, self).__init__()
        assert norm_stats is not None
        assert process_method in ['log2', 'plain']

        self.mode           = mode
        self.augment        = augment
        self.transform      = None
        self.data_list      = data_list
        self.norm_stats     = norm_stats
        self.process_method = process_method

        if self.augment:
            self.transform = V.Compose([
                V.Flip(1, p=0.1),
                V.Flip(2, p=0.1),
                V.RandomRotate90((1, 2), p=0.1)
            ], p=1.0)

        if process_method == 'log2':
            self.remove_outliers_func = remove_outliers_by_log2
        elif process_method == 'plain':
            self.remove_outliers_func = remove_outliers_by_plain

        self.months_list = months_list
        if months_list == 'all':
            self.months_list = list(range(12))

        self.s1_index_list = s1_index_list
        if s1_index_list == 'all':
            self.s1_index_list = list(range(4))
        
        self.s2_index_list = s2_index_list
        if s2_index_list == 'all':
            self.s2_index_list = list(range(11))

    def __len__(self):
        return len(self.data_list)

    def _load_data(self, subject_path):

        subject = os.path.basename(subject_path)

        # loads label data
        label_path = opj(subject_path, f'{subject}_agbm.tif')
        assert os.path.isfile(label_path), f'label {label_path} is not exist'
        label = read_raster(label_path, True, GT_SHAPE)
        if self.mode == 'train':
            label = self.remove_outliers_func(label, 'label')
            label = normalize(label, self.norm_stats['label'], 'minmax')
        label = np.expand_dims(label, axis=-1)

        # loads S1 and S2 features
        feature_list = []
        for month in self.months_list:
            s1_path = opj(subject_path, 'S1', f'{subject}_S1_{month:02d}.tif')
            s2_path = opj(subject_path, 'S2', f'{subject}_S2_{month:02d}.tif')
            s1 = read_raster(s1_path, True, S1_SHAPE)
            s2 = read_raster(s2_path, True, S2_SHAPE)

            s1_list = []
            for index in self.s1_index_list:
                s1i = self.remove_outliers_func(s1[index], 'S1', index)
                s1i = normalize(s1i, self.norm_stats['S1'][index], 'zscore')
                s1_list.append(s1i)

            s2_list = []
            for index in self.s2_index_list:
                s2i = self.remove_outliers_func(s2[index], 'S2', index)
                s2i = normalize(s2i, self.norm_stats['S2'][index], 'zscore')
                s2_list.append(s2i)

            feature = np.stack(s1_list + s2_list, axis=-1)
            feature = np.expand_dims(feature, axis=0)
            feature_list.append(feature)
        feature = np.concatenate(feature_list, axis=0)

        return label, feature

    def __getitem__(self, index):

        subject_path = self.data_list[index]
        label, feature = self._load_data(subject_path)
        # label:   (1, 256, 256, 1)
        # feature: (M, 256, 256, F)

        if self.augment:
            data = {'image': feature, 'mask': label}
            aug_data = self.transform(**data)
            feature, label = aug_data['image'], aug_data['mask']
            if label.shape[0] > 1:
                label = label[:1]

        feature = feature.transpose(3, 0, 1, 2).astype(np.float32)
        # feature: (F, M, 256, 256)
        label = label[0].transpose(2, 0, 1).astype(np.float32)
        # label: (1, 256, 256)

        return feature, label


def get_dataloader(
    mode, data_list, configs, norm_stats=None,
    process_method='plain'
):
    assert mode in ['train', 'val']

    if mode == 'train':
        batch_size = configs.train_batch
        drop_last  = True
        shuffle    = True
        augment    = configs.apply_augment
    else:  # mode == 'val'
        batch_size = configs.val_batch
        drop_last  = False
        shuffle    = False
        augment    = False

    dataset = BMDataset(
        mode           = mode,
        data_list      = data_list,
        norm_stats     = norm_stats,
        augment        = augment,
        process_method = process_method,
        s1_index_list  = configs.s1_index_list,
        s2_index_list  = configs.s2_index_list,
        months_list    = configs.months_list,
    )

    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        num_workers = configs.num_workers,
        pin_memory  = configs.pin_memory,
        drop_last   = drop_last,
        shuffle     = shuffle,
    )

    return 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
