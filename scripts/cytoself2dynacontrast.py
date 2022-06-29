"""
Script to convert OpenCell patches in Cytoself format to dynacontrast format (zarr) for training
"""
from copy import copy
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import zarr
import os

from os.path import join, basename, dirname
import pandas as pd
from glob import glob
from utils.train_utils import zscore_patch


def get_file_df(basepath, suffix='label', extension='npy'):
    """
    Return a DataFrame of data paths.
    :param basepath: base path that contains all npy files
    :param suffix: only designated suffix will be loaded
    :param extension: file extension
    :return: a DataFrame of data paths.
    """
    df = pd.DataFrame()
    if isinstance(suffix, str):
        df[suffix] = glob(join(basepath, '*_' + suffix + '.' + extension))
    elif isinstance(suffix, list):
        filelist = glob(join(basepath, '*_' + suffix[0] + '.' + extension))
        df[suffix[0]] = filelist
        for sf in suffix[1:]:
            flist = []
            for p in filelist:
                flist.append(
                    join(dirname(p), '_'.join(basename(p).split('_')[:-1]) + '_' + sf + '.' + extension)
                )
            df[sf] = flist
    else:
        raise TypeError('Only str or list is accepted for suffix.')
    return df

def cumsum_split(counts, split_perc, arr=None):
    """
    Split by the cumulative sum.
    Use for splitting data on cell counts in each FOV.
    :param counts: an numpy array of cell counts in each FOV.
    :param split_perc: split percentage.
    :param arr: array to be split
    :return: arrays of index if arr is None, otherwise split arr.
    """
    if arr is not None:
        assert len(counts) == len(counts)
    if sum(split_perc) != 1:
        split_perc = [i / sum(split_perc) for i in split_perc]

    # Sort counts in descendent order
    ind0 = np.argsort(counts)[::-1]
    counts = counts[ind0]

    # Find split indices
    cumsum_counts = np.cumsum(counts)
    count_limits = np.cumsum(np.array(split_perc)[:-1] * cumsum_counts[-1])
    split_counter = 0
    split_indices = []
    for i, csum in enumerate(cumsum_counts):
        if csum >= count_limits[split_counter]:
            split_counter += 1
            split_indices.append(i)
        if split_counter == len(count_limits):
            break
    rng0 = [0] + split_indices
    rng1 = split_indices + [len(counts)]
    ind = [ind0[i0: i1] for i0, i1 in zip(rng0, rng1)]

    if arr is not None:
        return [arr[d] for d in ind]
    else:
        return ind

def single_proc(label, data_split, fovpath_idx=3):
    uniq, counts = np.unique(label[:, fovpath_idx], return_counts=True)
    fovpaths = cumsum_split(counts, data_split, uniq)
    return [np.isin(label[:, fovpath_idx], pths) for pths in fovpaths]


def splitdata_on_fov(
        label_all, split_perc, cellline_id_idx=0, fovpath_idx=3, num_cores=15, shuffle_seed=1,
):
    """
    Split train/val/test data based on FOV paths
    :param label_all: all labels
    :param split_perc: split percentage
    :param cellline_id_idx: the index of cell line id in the label
    :param fovpath_idx: the index of FOV path in the label
    :param num_cores: number of cores for multiprocessing
    :param shuffle_seed: random seed for shuffling
    :return: split indices
    """
    df_label_all = pd.DataFrame(label_all)
    df_label_all_gp = df_label_all.groupby(cellline_id_idx)
    cell_line_id = df_label_all_gp.count().index.to_numpy()

    if num_cores == 1:
        results = [
            single_proc(label_all[label_all[:, cellline_id_idx] == cid], split_perc, fovpath_idx)
            for cid in tqdm(cell_line_id)
        ]
    else:
        results = Parallel(n_jobs=num_cores)(
            delayed(single_proc)(label_all[label_all[:, cellline_id_idx] == cid], split_perc, fovpath_idx)
            for cid in tqdm(cell_line_id)
        )

    train_ind = []
    val_ind = []
    test_ind = []
    for d, cid in zip(results, cell_line_id):
        idx0 = np.where(label_all[:, cellline_id_idx] == cid)[0]
        for i, d0 in enumerate(d):
            if i == 0:
                train_ind.append(idx0[d0])
            elif i == 1:
                val_ind.append(idx0[d0])
            elif i == 2:
                test_ind.append(idx0[d0])
    if len(train_ind) > 0:
        train_ind = np.hstack(train_ind)
    if len(val_ind) > 0:
        val_ind = np.hstack(val_ind)
    if len(test_ind) > 0:
        test_ind = np.hstack(test_ind)

    if shuffle_seed:
        np.random.seed(shuffle_seed)
        np.random.shuffle(train_ind)
        np.random.shuffle(val_ind)
        np.random.shuffle(test_ind)

    return train_ind, val_ind, test_ind

class DataManager:
    """
    A class object to manage training, validation and test data.
    The original npy data must be single npy file per protein.
    """
    def __init__(self, basepath, intensity_balance=None):
        """
        :param basepath: the directory that contains all npy files
        :param intensity_balance: intensity balance adjustment among gfp, nuc and nucdist
        """
        self.basepath = basepath
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.train_label = []
        self.val_label = []
        self.test_label = []
        self.train_label_onehot = []
        self.val_label_onehot = []
        self.test_label_onehot = []
        self.file_df = None
        self.label_all = []

        self.uniq_label_all = []
        self.uniq_loc_all = []
        self.cellid_uniq_test = []  # for back compatibility with analysis; should change in the future.

        intensity_balance_default = {'gfp': 1, 'nuc': 1, 'nucdist': 0.01}
        # Check intensity_balance items
        if isinstance(intensity_balance, dict):
            self.intensity_balance = intensity_balance
        else:
            self.intensity_balance = {}
        for key in intensity_balance_default.keys():
            if key not in self.intensity_balance:
                self.intensity_balance.update({key: intensity_balance_default[key]})

    def load_data(
            self, num_species=None, channel_list=[], data_split=(0.82, 0.098, 0.082), num_cores=15, split_on_fov=True,
            shuffle_seed=1, lab_col=0, labels_toload=[], labels_tohold=[],
    ):
        """
        Load data
        :param num_species: Number of target proteins
        :param channel_list: a list of 'gfp', 'nuc' or 'nucdist'
        :param data_split: split train or test data into arbitrary ratio. (train data ratio, test data ratio)
        data_split must be a tuple of floats that sums up to 1 if split_on_fov is True.
        :param num_cores: number of cores for multi-core processing
        :param split_on_fov: split train, val & test data on FOV if True, otherwise on patch
        :param shuffle_seed: specify random seed for reproducibility
        :param lab_col: the column index to be used for generating onehot vectors.
        :param labels_toload: a list of label names to be loaded.
        :param labels_tohold: a list of label names not to be loaded.
        """
        if 'label' not in channel_list:
            channel_list = copy(channel_list)
            channel_list.append('label')
        self.file_df = get_file_df(self.basepath, channel_list)
        if labels_toload:
            ind0 = self.file_df.iloc[:, 0].str.split('/', expand=True).iloc[:, -1].str.split(
                '_', expand=True).iloc[:, 1].isin(labels_toload)
            self.file_df = self.file_df[ind0]
        if labels_tohold:
            ind0 = self.file_df.iloc[:, 0].str.split('/', expand=True).iloc[:, -1].str.split(
                '_', expand=True).iloc[:, 1].isin(labels_tohold)
            self.file_df = self.file_df[~ind0]
        df_toload = self.file_df.iloc[:num_species]

        # Load data
        imgdata = []
        for ch in channel_list:
            print(f'Loading {ch} data...')
            results = Parallel(n_jobs=num_cores)(
                delayed(np.load)(row[ch], allow_pickle=ch == 'label')
                for _, row in tqdm(df_toload.iterrows(), total=len(df_toload))
            )
            if ch == 'label':
                self.label_all = np.vstack(results)
            else:
                if results[0].ndim == 3:
                    d = np.vstack(results)[..., np.newaxis]
                else:
                    d = np.vstack(results)
                if ch in self.intensity_balance:
                    d *= self.intensity_balance[ch]
                else:
                    print('Channel not found in intensity balance.')
                imgdata.append(d)
        if len(imgdata) > 0:
            imgdata = np.concatenate(imgdata, axis=-1)
        else:
            print('No image data was loaded.')

        # Get unique labels
        self.uniq_label_all = np.unique(self.label_all[:, lab_col])

        # Split data
        print('Splitting data...')
        if split_on_fov:
            train_ind, val_ind, test_ind = splitdata_on_fov(
                self.label_all, split_perc=data_split, cellline_id_idx=0, fovpath_idx=7,
                num_cores=num_cores, shuffle_seed=shuffle_seed
            )
        else:
            np.random.seed(shuffle_seed)
            ind = np.random.choice(len(self.label_all), size=len(self.label_all), replace=False)
            split_ind = list(np.cumsum([int(len(self.label_all) * i) for i in data_split[:-1]]))
            train_ind =ind[0, split_ind[0]]
            val_ind = ind[split_ind[0], split_ind[1]]
            test_ind = ind[split_ind[1], len(self.label_all)]
        if len(train_ind) > 0:
            if len(imgdata) > 0:
                self.train_data = imgdata[train_ind]
            self.train_label = self.label_all[train_ind]
        if len(val_ind) > 0:
            if len(imgdata) > 0:
                self.val_data = imgdata[val_ind]
            self.val_label = self.label_all[val_ind]
        if len(test_ind) > 0:
            if len(imgdata) > 0:
                self.test_data = imgdata[test_ind]
            self.test_label = self.label_all[test_ind]

        self.cellid_uniq_test = np.unique(self.test_label[:, 0])  # for back compatibility; should be deprecated in the future
        print(f'Train: Validation : Test = {len(self.train_label)} : {len(self.val_label)} : {len(self.test_label)}')

    def get_cellid_uniq_test(self, col=0):
        if len(self.test_label) == 0:
            raise ValueError('test_label is not loaded.')
        self.cellid_uniq_test = np.unique(self.test_label[:, col])

    def get_label_onhot(self, col):
        n_class = len(self.uniq_label_all)
        self.train_label_onehot = np.zeros((len(self.train_label), n_class), dtype=np.float32)
        self.val_label_onehot = np.zeros((len(self.val_label), n_class), dtype=np.float32)
        self.test_label_onehot = np.zeros((len(self.test_label), n_class), dtype=np.float32)
        print('Converting labels to onehot vectors...')
        for i, c in enumerate(tqdm(self.uniq_label_all)):
            if self.train_label_onehot.size > 0:
                self.train_label_onehot[:, i] = self.train_label[:, col] == c
            if self.val_label_onehot.size > 0:
                self.val_label_onehot[:, i] = self.val_label[:, col] == c
            if self.test_label_onehot.size > 0:
                self.test_label_onehot[:, i] = self.test_label[:, col] == c

    def get_label_multihot(self, grade=3, label_starts_at=2):
        """
        Construct multi-hot vector for multi-label classificaiton
        :param grade: localization grade
        :param label_starts_at: the column index in the label data to construct multihot vectors
        """
        if isinstance(grade, int):
            grade = [3 - grade]
        else:
            grade = [3 - i for i in grade]

        # Find unique localization labels
        loc_df = pd.DataFrame(self.label_all[:, label_starts_at: label_starts_at + 3])
        loc_df_train = pd.DataFrame(self.train_label[:, label_starts_at: label_starts_at + 3])
        loc_df_val = pd.DataFrame(self.val_label[:, label_starts_at: label_starts_at + 3])
        loc_df_test = pd.DataFrame(self.test_label[:, label_starts_at: label_starts_at + 3])

        # Get unique label for all
        loc_lab_all = []
        for i in range(3):
            loc_lab_all.append(loc_df.iloc[:, i].str.split(';', expand=True).to_numpy())
        loc_lab_all = np.hstack(loc_lab_all).astype(str)
        loc_lab_all[loc_lab_all == 'nan'] = 'None'
        self.uniq_loc_all = np.unique(loc_lab_all)[1:]

        # Expand labels for train, val & test data
        loc_train = []
        loc_val = []
        loc_test = []
        for i in grade:
            loc_train.append(loc_df_train.iloc[:, i].str.split(';', expand=True).to_numpy())
            loc_val.append(loc_df_val.iloc[:, i].str.split(';', expand=True).to_numpy())
            loc_test.append(loc_df_test.iloc[:, i].str.split(';', expand=True).to_numpy())
        loc_train = np.hstack(loc_train)
        loc_val = np.hstack(loc_val)
        loc_test = np.hstack(loc_test)

        n_class = len(self.uniq_loc_all)
        self.train_label_onehot = np.zeros((len(self.train_label), n_class), dtype=np.float32)
        self.val_label_onehot = np.zeros((len(self.val_label), n_class), dtype=np.float32)
        self.test_label_onehot = np.zeros((len(self.test_label), n_class), dtype=np.float32)
        for i, p in enumerate(self.uniq_loc_all):
            if self.train_label_onehot.size > 0:
                ind = np.isin(loc_train, p).sum(axis=1)
                self.train_label_onehot[ind == 1, i] = True
            if self.val_label_onehot.size > 0:
                ind = np.isin(loc_val, p).sum(axis=1)
                self.val_label_onehot[ind == 1, i] = True
            if self.test_label_onehot.size > 0:
                ind = np.isin(loc_test, p).sum(axis=1)
                self.test_label_onehot[ind == 1, i] = True

    def minmax(self, type='train'):
        if type == 'train':
            data = self.train_data
        elif type == 'val':
            data = self.val_data
        elif type == 'test':
            data = self.test_data
        else:
            raise ValueError('Please select type from train, val and test.')
        return [[data[..., i].min(), data[..., i].max()] for i in range(data.shape[-1])]

    def clear_training_data(self):
        """
        Clear training & validation data to reduce memory usage.
        """
        self.train_data = []
        self.val_data = []
        self.train_label = []
        self.val_label = []
        self.train_label_onehot = []
        self.val_label_onehot = []

def main(input_dir, output_dir, gt_uniorg, gt_corum):
    datamgr = DataManager(input_dir)
    # datamgr.load_data(
    #     num_species=None, channel_list=['gfp', 'nuc'], data_split=(0.82, 0.098, 0.082),
    #     num_cores=25, split_on_fov=True,
    # )
    datamgr.load_data(
        num_species=None, channel_list=['label'], data_split=(0.82, 0.098, 0.082),
        num_cores=25, split_on_fov=True,
    )
    datasets = {'train': datamgr.train_data, 'val': datamgr.val_data, 'test': datamgr.test_data}
    labels = {'train': datamgr.train_label, 'val': datamgr.val_label, 'test': datamgr.test_label}
    gt_corum.rename(columns={'family':'protein-complex-level ground truth'}, inplace=True)
    gt_uniorg.rename(columns={'family':'organelle-level ground truth'}, inplace=True)
    del datamgr
    for name, data in datasets.items():
        # covert from nyxc to ncyx
        data = data.transpose((0, 3, 1, 2))
        data = zscore_patch(data).astype(np.float32)
        output_fname = os.path.join(output_dir, 'cell_patches_{}.zarr'.format(name))
        data_zar = zarr.open(output_fname, mode='w', shape=data.shape, chunks=data[0].shape, dtype=np.float32)
        data_zar[:] = data
    for name, label in labels.items():
        meta_df = pd.DataFrame(label, columns=['ensemble ID', 'gene', 'primary location', 'secondary location', 'tertiary location', 'cell line ID', 'plate ID', 'path'])
        # split multiple labels in each column

        meta_df = pd.merge(meta_df,
                           gt_uniorg[['gene', 'organelle-level ground truth']],
                           how='left', on='gene', validate='m:1')
        meta_df = pd.merge(meta_df,
                           gt_corum,
                           how='left', on='gene', validate='m:1')
        meta_df[['protein-complex-level ground truth', 'organelle-level ground truth']] = \
            meta_df[['protein-complex-level ground truth', 'organelle-level ground truth']].fillna(value='other')
        for col in ['primary location', 'secondary location', 'tertiary location']:
            split_df  = meta_df[col].str.split(';', expand=True)
            meta_df[[col + ' ' + str(ind) for ind in split_df.columns]] = split_df
        meta_df.to_csv(os.path.join(output_dir, 'patch_meta_{}.csv'.format(name)), sep=',')
        # cytoself data is max projected so each cell has its unique label
        # with open(os.path.join(output_dir, 'patch_labels_{}.npy'.format(name)), 'wb') as f:
        #     np.save(f, np.arange(len(meta_df)))

if __name__ == '__main__':
    input_dir = '/gpfs/gpfsML/ML_group/opencell-microscopy/clustering-results/2021-7-15_good-fovs/cropping/by_batch'
    output_dir = '/gpfs/CompMicro/projects/dynacontrast/opencell/2021-7-15_good-fovs'
    basepath_gt = '/gpfs/gpfsML/ML_group/opencell-microscopy/'
    gt_uniorg = pd.read_csv(join(basepath_gt, 'clustering_gt/unique_organelles_converted.csv'))
    gt_corum = pd.read_csv(join(basepath_gt, 'clustering_gt/unique_CORUM_purified_table.csv'))
    gt_uniorg_dp = gt_uniorg[gt_uniorg.duplicated(subset=['gene'], keep=False)]
    gt_corum_dp = gt_corum[gt_corum.duplicated(subset=['gene'], keep=False)]
    gt_uniorg.drop_duplicates(subset=['gene'], inplace=True)
    gt_corum.drop_duplicates(subset=['gene'], inplace=True)
    main(input_dir, output_dir, gt_uniorg, gt_corum)
