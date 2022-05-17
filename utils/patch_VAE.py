import os

import cv2
from dask import array as da

from SingleCellPatch.patch_utils import cv2_fn_wrapper
import logging
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
import numpy as np
import pandas as pd
import importlib
import inspect
import time
from utils.config_reader import YamlReader
from torch.utils.data import DataLoader
from tqdm import tqdm
import zarr
from SingleCellPatch.extract_patches import process_site_extract_patches
from SingleCellPatch.generate_trajectories import process_site_build_trajectory, process_well_generate_trajectory_relations
from utils.train_utils import zscore, zscore_patch, split_data
from dataset.dataset import ImageDataset
import HiddenStateExtractor.resnet as resnet

NETWORK_MODULE = 'run_training'
log = logging.getLogger('dynacontrast.log')

def extract_patches(raw_folder: str,
                    supp_folder: str,
                    # channels: list,
                    sites: list,
                    config: YamlReader,
                    **kwargs):
    """ Helper function for patch extraction

    Wrapper method `process_site_extract_patches` will be called, which
    extracts individual cells from static frames for each site.

    Results will be saved in the supplementary data folder, including:
        "stacks_*.pkl": single cell patches for each time slice

    Args:
        raw_folder (str): folder for raw data, segmentation and
            summarized results
        supp_folder (str): folder for supplementary data
        sites (list of str): list of site names
        config (YamlReader): config file supplied at CLI
    """
    channels = config.patch.channels

    assert len(channels) > 0, "At least one channel must be specified"

    window_size = config.patch.window_size
    save_fig = config.patch.save_fig
    reload = config.patch.reload
    skip_boundary = config.patch.skip_boundary

    for site in sites:
        site_path = os.path.join(raw_folder + '/' + site + '.npy')
        site_segmentation_path = os.path.join(raw_folder, '%s_NNProbabilities.npy' % site)
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path):
            print("Site data not found %s" % site_path, flush=True)
        if not os.path.exists(site_segmentation_path):
            print("Site data not found %s" % site_segmentation_path, flush=True)
        if not os.path.exists(site_supp_files_folder):
            print("Site supp folder not found %s" % site_supp_files_folder, flush=True)
        # else:
            # print("Building patches %s" % site_path, flush=True)
        try:
            process_site_extract_patches(site_path, 
                                         site_segmentation_path, 
                                         site_supp_files_folder,
                                         window_size=window_size,
                                         channels=channels,
                                         save_fig=save_fig,
                                         reload=reload,
                                         skip_boundary=skip_boundary,
                                         **kwargs)
        except Exception as e:
            log.error('Extracting patches failed for position {}. '.format(site))
            log.exception('')
            print('Extracting patches failed for position {}. '.format(site))
            raise e
    return


def build_trajectories(summary_folder: str,
                       supp_folder: str,
                       # channels: list,
                       sites: list,
                       config: YamlReader,
                       **kwargs):
    """ Helper function for trajectory building

    Wrapper method `process_site_build_trajectory` will be called, which 
    connects and generates trajectories from individual cell identifications.

    Results will be saved in the supplementary data folder, including:
        "cell_traj.pkl": list of trajectories and list of trajectory positions
            trajectories are dict of t_point: cell ID
            trajectory positions are dict of t_point: cell center position

    Args:
        summary_folder (str): folder for raw data, segmentation and
            summarized results
        supp_folder (str): folder for supplementary data
        channels (list of int): indices of channels used for segmentation
            (not used)
        model_path (str, optional): path to model weight (not used)
        sites (list of str): list of site names

    """

    for site in sites:
        site_path = os.path.join(summary_folder + '/' + site + '.npy')
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path) or not os.path.exists(site_supp_files_folder):
            print("Site data not found %s" % site_path, flush=True)
        else:
            print("Building trajectories %s" % site_path, flush=True)
            process_site_build_trajectory(site_supp_files_folder,
                                          min_len=config.patch.min_length,
                                          track_dim=config.patch.track_dim)
    return


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
    label_col = None
    patch_shape = (128, 128)
    if hasattr(config.patch, 'splits'):
        splits = config.patch.splits
    if hasattr(config.patch, 'split_ratio'):
        split_ratio = config.patch.split_ratio
    if hasattr(config.patch, 'split_cols'):
        split_cols = config.patch.split_cols
    if hasattr(config.patch, 'label_col'):
        label_col = config.patch.label_col
    if hasattr(config.patch, 'size'):
        patch_shape = (config.patch.size, config.patch.size)
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
    if label_col is None:
        label = np.arange(len(dataset))
    else:
        _, label = process_well_generate_trajectory_relations(df_meta, track_dim='slice')
    print('len(labels):', len(label))
    print('len(dataset):', len(dataset))
    assert len(dataset) == len(label), 'Number of patches and labels are not consistent.'
    datasets, labels, df_metas = split_data(dataset, label, df_meta, splits=splits, split_cols=split_cols,
                       val_split_ratio=split_ratio, seed=0)
    for split in splits:
        patch_fname = os.path.join(raw_folder, 'cell_patches_{}.zarr'.format(split))
        label_fname = os.path.join(raw_folder, 'patch_labels_{}.npy'.format(split))
        meta_fname = os.path.join(raw_folder, 'patch_meta_{}.csv'.format(split))
        print('saving {}...'.format(patch_fname))
        data_zar = zarr.open(patch_fname, mode='w', shape=datasets[split].shape, chunks=datasets[split][0].shape, dtype=np.float32)
        data_zar[:] = datasets[split]
        print('saving {}...'.format(label_fname))
        with open(label_fname, 'wb') as f:
            np.save(f, labels[split])
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
    raw_dirs = config.data_pooling.raw_dirs
    supp_dirs = config.data_pooling.supp_dirs
    dst_dir = config.data_pooling.dst_dir
    val_split_ratio = config.training.val_split_ratio
    fname_suffix = ['', '_datasetnorm']
    labels = []
    id_offsets = [0]
    df_meta_all = []
    for raw_dir, supp_dir in zip(raw_dirs, supp_dirs):
        label = da.from_zarr(os.path.join(raw_dir, 'patch_labels.zarr')).astype(np.int64)
        labels.append(label)
        meta_path = os.path.join(supp_dir, 'im-supps', 'patch_meta.csv')
        df_meta = pd.read_csv(meta_path, index_col=0, converters={
            'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
        df_meta['data_dir'] = os.path.dirname(raw_dir)
        df_meta_all.append(df_meta)
        id_offsets.append(len(label))
    df_meta_all = pd.concat(df_meta_all, axis=0)
    df_meta_all.reset_index(drop=True, inplace=True)
    id_offsets = id_offsets[:-1]
    labels = concat_relations(labels, offsets=id_offsets)
    print('len(labels):', len(labels))
    for suffix in fname_suffix:
        # datasets = None
        datasets = []
        for raw_dir, supp_dir in zip(raw_dirs, supp_dirs):
            os.makedirs(dst_dir, exist_ok=True)
            data_path = os.path.join(raw_dir, 'cell_patches' + suffix + '.zarr')
            t0 = time.time()
            dataset = da.from_zarr(data_path)
            datasets.append(dataset)
            t1 = time.time()
            print('loading dataset takes:', t1 - t0)
            print('dataset.shape:', dataset.shape)
        dataset = da.concatenate(datasets, axis=0)
        print('len(dataset):', len(dataset))
        # dataset = da.from_zarr(os.path.join(dst_dir, patch_fname))
        t0 = time.time()
        train_set, train_labels, val_set, val_labels, df_meta_all_split = \
            split_data(dataset, labels, df_meta_all, split_cols=['data_dir', 'FOV'],
                       val_split_ratio=val_split_ratio, seed=0)
        t1 = time.time()
        print('splitting dataset takes:', t1 - t0)
        t0 = time.time()
        da.to_zarr(train_set, os.path.join(dst_dir, 'cell_patches' + suffix + '_train.zarr'), overwrite=True, compressor='default')
        da.to_zarr(val_set, os.path.join(dst_dir, 'cell_patches' + suffix + '_val.zarr'), overwrite=True, compressor='default')
        da.to_zarr(train_labels, os.path.join(dst_dir, 'patch_labels' + suffix + '_train.zarr'), overwrite=True, compressor='default')
        da.to_zarr(val_labels, os.path.join(dst_dir, 'patch_labels' + suffix + '_val.zarr'), overwrite=True, compressor='default')
        train_labels = np.asarray(train_labels)
        val_labels = np.asarray(val_labels)
        with open(os.path.join(dst_dir, 'patch_labels' + suffix + '_train.npy'), 'wb') as f:
            np.save(f, train_labels)
        with open(os.path.join(dst_dir, 'patch_labels' + suffix + '_val.npy'), 'wb') as f:
            np.save(f, val_labels)
        t1 = time.time()
        print('writing dataset takes:', t1 - t0)
        df_meta_all_split.to_csv(os.path.join(dst_dir, 'patch_meta' + suffix + '.csv'), sep=',')
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

def encode_patches(raw_dir: str,
                   supp_folder: str,
                   sites: list,
                   config_: YamlReader,
                   gpu: int=0,
                   **kwargs):
    """ Wrapper method for patch encoding

    This function loads prepared dataset and applies trained VAE to encode 
    static patches for each well. 

    Resulting latent vectors will be saved in raw_dir:
        train_embeddings.npy
        val_embeddings.npy

    Args:
        raw_folder (str): folder for raw data, segmentation and
            summarized results
        supp_folder (str): folder for supplementary data
        sites (list): list of FOVs to process
        config_ (YamlReader): Reads fields from the "INFERENCE" category

    """
    model_dir = config_.inference.weights
    channels = config_.inference.channels
    network = config_.inference.network
    network_width = config_.inference.network_width
    batch_size = config_.inference.batch_size
    num_workers = config_.inference.num_workers
    normalization = config_.inference.normalization
    projection = config_.inference.projection
    splits = config_.inference.splits

    assert len(channels) > 0, "At least one channel must be specified"

    model_name = os.path.basename(model_dir)
    if projection:
        output_dir = os.path.join(raw_dir, model_name + '_proj')
        encode_layer = 'z'
    else:
        output_dir = os.path.join(raw_dir, model_name)
        encode_layer = 'h'
    os.makedirs(output_dir, exist_ok=True)
    datasets = {}
    for split in splits:
        if normalization == 'dataset':
            datasets[split] = zarr.open(os.path.join(raw_dir, 'cell_patches_datasetnorm_{}.zarr'.format(split)))
        elif normalization == 'patch':
            # train_set_sync = zarr.ProcessSynchronizer(os.path.join(raw_dir, 'cell_patches_train.sync'))
            datasets[split] = zarr.open(os.path.join(raw_dir, 'cell_patches_{}.zarr'.format(split)))
        else:
            raise ValueError('Parameter "normalization" must be "dataset" or "patch"')
        datasets[split] = ImageDataset(datasets[split])
    device = torch.device('cuda:%d' % gpu)
    print('Encoding images using gpu {}...'.format(gpu))
    # Only ResNet is available now
    if 'ResNet' not in network:
        raise ValueError('Network {} is not available'.format(network))

    for data_name, dataset in datasets.items():
        network_cls = getattr(resnet, 'EncodeProject')
        model = network_cls(arch=network, num_inputs=len(channels), width=network_width)
        model = model.to(device)
        # print(model)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), map_location=device))
        model.eval()
        data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  )
        h_s = []
        with tqdm(data_loader, desc='inference batch') as batch_pbar:
            for batch in batch_pbar:
                print(batch.shape)
                batch = batch.to(device)
                code = model.encode(batch, out=encode_layer).cpu().data.numpy().squeeze()
                print(code.shape)
                h_s.append(code)
        dats = np.concatenate(h_s, axis=0)
        output_fname = '{}_embeddings.npy'.format(data_name)
        print(f"\tsaving {os.path.join(output_dir, output_fname)}")
        with open(os.path.join(output_dir, output_fname), 'wb') as f:
            np.save(f, dats)



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
    pos_ids = df_meta['position'].unique()
    t_points = df_meta['time'].unique()
    slices = df_meta['slice'].unique()
    for pos_idx in pos_ids:
        supp_files_folder = os.path.join(supp_folder, 'im-supps', 'img_p{:03}'.format(pos_idx))
        for t in t_points:
            for z in slices:
                stack_path_site = os.path.join(supp_files_folder, 'patches_t{}_z{}.npy'.format(t, z))
                print(f"\tloading data {stack_path_site}")
                with open(stack_path_site, 'rb') as f:
                    cell_patches = np.load(f)
                resized_dat = cv2_fn_wrapper(cv2.resize, cell_patches, input_shape)
                dataset.append(resized_dat)
    dataset = np.concatenate(dataset, axis=0)
    return dataset