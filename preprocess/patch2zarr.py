import os
import cv2
import glob
from dask import array as da
from utils.patch_utils import cv2_fn_wrapper
import logging
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import pandas as pd
import importlib
import inspect
import time
from utils.config_reader import YamlReader
import zarr
from preprocess.track import process_well_generate_trajectory_relations
from utils.train_utils import zscore_patch, split_data

NETWORK_MODULE = 'run_training'
log = logging.getLogger('dynacontrast.log')


def pool_positions(raw_folder: str,
                   supp_folder: str,
                   sites: list,
                   config: YamlReader,
                   ):
    """ Wrapper method for prepare dataset for VAE encoding

    This function loads data from multiple sites, adjusts intensities to correct
    for batch effect, and assembles into dataset for model prediction

    Resulting dataset will be saved in the summary folder, including:
        "*_file_paths.pkl": list of cell identification strings
        "*_static_patches.pt": all static patches from a given well
        "*_adjusted_static_patches.pt": all static patches from a given well 
            after adjusting phase/retardance intensities (avoid batch effect)

    Args:
        raw_folder (str): folder for raw data, segmentation and
            summarized results
        supp_folder (str): folder for supplementary data
        sites (list of str): list of site names
        config (YamlReader): config file supplied at CLI

    """
    splits = ('all',)
    split_ratio = None
    split_cols = None
    patch_shape = (128, 128)
    if hasattr(config.preprocess, 'splits'):
        splits = config.preprocess.splits
    if hasattr(config.preprocess, 'split_ratio'):
        split_ratio = config.preprocess.split_ratio
    if hasattr(config.preprocess, 'split_cols'):
        split_cols = config.preprocess.split_cols
    if hasattr(config.preprocess, 'size'):
        patch_shape = (config.preprocess.patch_size, config.preprocess.patch_size)
    if type(splits) is str:
        splits = [splits]
    if type(splits) is list:
        splits = tuple(splits)
    # sites should be from a single condition (C5, C4, B-wells, etc..)
    assert len(set(site[:2] for site in sites)) == 1, \
        "Sites should be from a single well/condition"
    well = sites[0][:2]

    df_meta = pd.DataFrame()
    traj_id_offsets = {'time trajectory ID': 0, 'slice trajectory ID': 0}
    for site in sites:
        # print(site)
        supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        meta_path = os.path.join(supp_files_folder, 'patch_meta.csv')
        if not os.path.isfile(meta_path): # no patch metadata is saved if no cell is detected
            continue
        df_meta_site = pd.read_csv(meta_path, index_col=0, converters={
            'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
        if len(df_meta_site) == 0: # if no cell patch was cropped, skip the position
            continue
        # offset trajectory ids to make it unique
        for col in df_meta_site.columns:
            if col in traj_id_offsets:
                df_meta_site[col] += traj_id_offsets[col]
                traj_id_offsets[col] = df_meta_site[col].max() + 1
        df_meta = df_meta.append(df_meta_site, ignore_index=True)
    df_meta.reset_index(drop=True, inplace=True)
    meta_path = os.path.join(supp_folder, '%s-supps' % well, 'patch_meta.csv')
    df_meta.to_csv(meta_path, sep=',')
    dataset = combine_patches(df_meta, supp_folder, input_shape=patch_shape)
    assert len(dataset) == len(df_meta), 'Number of patches {} and rows in metadata {} do not match.'.format(len(dataset), len(df_meta))
    # dataset = zscore(dataset, channel_mean=None, channel_std=None).astype(np.float32)
    # output_fname = os.path.join(raw_folder, 'cell_patches_datasetnorm.zarr')
    # print('saving {}...'.format(output_fname))
    # zarr.save(output_fname, dataset)
    dataset = zscore_patch(dataset).astype(np.float32)
    print('len(dataset):', len(dataset))
    datasets, df_metas = split_data(dataset, df_meta, splits=splits, split_cols=split_cols,
                       val_split_ratio=split_ratio, seed=0)
    for split in splits:
        patch_fname = os.path.join(raw_folder, 'cell_patches_{}.zarr'.format(split))
        meta_fname = os.path.join(raw_folder, 'patch_meta_{}.csv'.format(split))
        print('saving {}...'.format(patch_fname))
        data_zar = zarr.open(patch_fname, mode='w', shape=datasets[split].shape, chunks=datasets[split][0:1].shape, dtype=np.float32)
        data_zar[:] = datasets[split]
        df_metas[split].to_csv(meta_fname, sep=',')
    return


def pool_datasets(config):
    """ Combine multiple datasets

    Args:
        input_dataset_names (list): list of input datasets
            named as $DIRECTORY/$DATASETNAME, this method reads files below:
                $DIRECTORY/$DATASETNAME_file_paths.pkl
                $DIRECTORY/$DATASETNAME_static_patches.pkl
                $DIRECTORY/$DATASETNAME_static_patches_relations.pkl
                $DIRECTORY/$DATASETNAME_static_patches_mask.pkl (if `save_mask`)
        output_dataset_name (str): path to the output save
            the combined dataset will be saved to the specified path with
            corresponding suffix
        save_mask (bool, optional): if to read & save dataset mask

    """
    splits = ('all',)
    val_split_ratio = None
    split_cols = None
    if hasattr(config.data_pooling, 'splits'):
        splits = config.data_pooling.splits
    if hasattr(config.data_pooling, 'split_ratio'):
        val_split_ratio = config.data_pooling.split_ratio
    if hasattr(config.data_pooling, 'split_cols'):
        split_cols = config.data_pooling.split_cols
    if type(splits) is str:
        splits = [splits]
    if type(splits) is list:
        splits = tuple(splits)
    raw_dirs = config.data_pooling.raw_dirs
    dst_dir = config.data_pooling.dst_dir
    datasets = []
    df_meta_all = []
    for raw_dir in raw_dirs:
        patch_fname = os.path.join(raw_dir, 'cell_patches_all.zarr')
        meta_fname = os.path.join(raw_dir, 'patch_meta_all.csv')
        df_meta = pd.read_csv(meta_fname, index_col=0, converters={
            'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
        df_meta['data_dir'] = os.path.dirname(raw_dir)
        df_meta_all.append(df_meta)
        t0 = time.time()
        dataset = da.from_zarr(patch_fname)
        datasets.append(dataset)
        t1 = time.time()
        print('loading dataset takes:', t1 - t0)
        print('dataset.shape:', dataset.shape)
    dataset = da.concatenate(datasets, axis=0)
    df_meta_all = pd.concat(df_meta_all, axis=0)
    df_meta_all.reset_index(drop=True, inplace=True)
    print('len(dataset):', len(dataset))
    # dataset = da.from_zarr(os.path.join(dst_dir, patch_fname))
    t0 = time.time()
    datasets, df_metas = \
        split_data(dataset, df_meta_all, splits=splits, split_cols=split_cols,
                   val_split_ratio=val_split_ratio, seed=0)
    t1 = time.time()
    print('splitting dataset takes:', t1 - t0)

    for split in splits:
        t0 = time.time()
        datasets[split] = datasets[split].rechunk(datasets[split][0:1].shape)
        da.to_zarr(datasets[split], os.path.join(dst_dir, 'cell_patches_{}.zarr'.format(split)), overwrite=True, compressor='default')
        t1 = time.time()
        print('writing dataset takes:', t1 - t0)
        df_metas[split].to_csv(os.path.join(dst_dir, 'patch_meta_{}.csv'.format(split)), sep=',')
    return

def import_object(module_name, obj_name, obj_type='class'):
    """Imports a class or function dynamically

    :param str module_name: modules such as input, utils, train etc
    :param str obj_name: Object to find
    :param str obj_type: Object type (class or function)
    """

    # full_module_name = ".".join(('dynacontrast', module_name))
    full_module_name = module_name
    try:
        module = importlib.import_module(full_module_name)
        obj = getattr(module, obj_name)
        if obj_type == 'class':
            assert inspect.isclass(obj),\
                "Expected {} to be class".format(obj_name)
        elif obj_type == 'function':
            assert inspect.isfunction(obj),\
                "Expected {} to be function".format(obj_name)
        return obj
    except ImportError:
        raise


def concat_relations(labels, offsets):
    """combine relation dictionaries from multiple datasets

    Args:
        labels (list): list of label array to combine
        offsets (list): offset to add to the indices

    Returns: new_labels (array): dictionary of combined labels

    """
    new_labels = []
    for label, offset in zip(labels, offsets):
        new_label = label + offset
        # make a new dict with updated keys
        new_labels.append(new_label)
    new_labels = da.concatenate(new_labels, axis=0)
    return new_labels


def combine_patches(df_meta,
                    supp_folder,
                    input_shape=(128, 128),
                    ):
    """ Prepare input dataset for VAE

    This function reads assembled pickle files (dict)

    Args:
        stack_paths (list of str): list of pickle file paths
        channels (list of int, optional): channels in the input
        input_shape (tuple, optional): input shape (height and width only)
        key (str): 'mat' or 'masked_mat'

    Returns:
        dataset (np array): array of cell patches with dimension (n, c, y, x)

    """
    dataset = []
    # print(df_meta[:10])
    df_ptz = df_meta[['position', 'time', 'slice']].drop_duplicates()
    # pos_ids = df_meta['position'].unique()
    # t_points = df_meta['time'].unique()
    # slices = df_meta['slice'].unique()
    for pos_idx, t, z in df_ptz.to_numpy():
        supp_files_folder = os.path.join(supp_folder, 'im-supps', 'img_p{:03}'.format(pos_idx))
        # for t in t_points:
        #     for z in slices:
        stack_path_site = os.path.join(supp_files_folder, 'patches_t{}_z{}.npy'.format(t, z))
        # print(f"\tloading data {stack_path_site}")
        with open(stack_path_site, 'rb') as f:
            cell_patches = np.load(f)
        resized_dat = cv2_fn_wrapper(cv2.resize, cell_patches, input_shape)
        dataset.append(resized_dat)
    dataset = np.concatenate(dataset, axis=0)
    return dataset