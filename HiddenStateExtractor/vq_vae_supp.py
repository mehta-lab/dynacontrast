# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 22:38:24 2021

@author: Zhenqin Wu
"""
import os
import h5py
import cv2
import numpy as np
import scipy
import queue
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import csr_matrix

CHANNEL_RANGE = [(0.3, 0.8), (0., 0.6)] 
CHANNEL_VAR = np.array([0.0475, 0.0394]) # After normalized to CHANNEL_RANGE
CHANNEL_MAX = 65535.
eps = 1e-9

def cv2_fn_wrapper(cv2_fn, mat, *args, **kwargs):
    """" A wrapper for cv2 functions
    
    Data in channel first format are adjusted to channel last format for 
    cv2 functions
    """
    
    mat_shape = mat.shape
    x_size = mat_shape[-2]
    y_size = mat_shape[-1]
    _mat = mat.reshape((-1, x_size, y_size)).transpose((1, 2, 0))
    _output = cv2_fn(_mat, *args, **kwargs)
    _x_size = _output.shape[0]
    _y_size = _output.shape[1]
    output_shape = tuple(list(mat_shape[:-2]) + [_x_size, _y_size])
    output = _output.transpose((2, 0, 1)).reshape(output_shape)
    return output

def assemble_patches(df_meta,
                     supp_folder,
                     channels=None,
                     input_shape=(128, 128),
                     key='masked_mat'):
    """ Prepare input dataset for VAE

    This function reads assembled pickle files (dict)

    Args:
        stack_paths (list of str): list of pickle file paths
        channels (list of int, optional): channels in the input
        input_shape (tuple, optional): input shape (height and width only)
        key (str): 'mat' or 'masked_mat'

    Returns:
        np array: dataset of training inputs
        list of str: identifiers of single cell image patches

    """
    dataset = []
    sites = df_meta['FOV'].unique()
    t_points = df_meta['time'].unique()
    slices = df_meta['slice'].unique()
    for site in sites:
        supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        for t in t_points:
            for z in slices:
                cell_ids = df_meta.loc[(df_meta['FOV'] == site) &
                                       (df_meta['time'] == t) &
                                       (df_meta['slice'] == z), 'cell ID'].to_list()
                stack_path_site = os.path.join(supp_files_folder, 'stacks_t{}_z{}.pkl'.format(t, z))
                # print(f"\tloading data {stack_path_site}")
                with open(stack_path_site, 'rb') as f:
                    stack_dict = pickle.load(f)
                for cell_id in cell_ids:
                    patch_key = os.path.join(supp_files_folder, 't{}_z{}_cell{}'.format(t, z, cell_id))
                    # patch_key = 't{}_z{}_cell{}'.format(t, z, cell_id)
                    dat = stack_dict[patch_key][key]
                    if channels is None:
                        channels = np.arange(dat.shape[0])
                    dat = np.array(dat)[np.array(channels)].astype(float)
                    resized_dat = cv2_fn_wrapper(cv2.resize, dat, input_shape)
                    dataset.append(resized_dat)
    dataset = np.stack(dataset)
    return dataset

def reorder_with_trajectories(dataset, relations, seed=None, w_a=1.1, w_t=0.1):
    """ Reorder `dataset` to facilitate training with matching loss

    Args:
        dataset (TensorDataset): dataset of training inputs
        relations (dict): dict of pairwise relationship (adjacent frames, same 
            trajectory)
        seed (int or None, optional): if given, random seed
        w_a (float): weight for adjacent frames
        w_t (float): weight for non-adjecent frames in the same trajectory

    Returns:
        TensorDataset: dataset of training inputs (after reordering)
        scipy csr matrix: sparse matrix of pairwise relations
        list of int: index of samples used for reordering

    """
    if not seed is None:
        np.random.seed(seed)
    inds_pool = set(range(len(dataset)))
    inds_in_order = []
    relation_dict = {}
    for pair in relations:
        if relations[pair] == 2: # Adjacent pairs
            if pair[0] not in relation_dict:
                relation_dict[pair[0]] = []
            relation_dict[pair[0]].append(pair[1])
    while len(inds_pool) > 0:
        rand_ind = np.random.choice(list(inds_pool))
        if not rand_ind in relation_dict:
            inds_in_order.append(rand_ind)
            inds_pool.remove(rand_ind)
        else:
            traj = [rand_ind]
            q = queue.Queue()
            q.put(rand_ind)
            while True:
                try:
                    elem = q.get_nowait()
                except queue.Empty:
                    break
                new_elems = relation_dict[elem]
                for e in new_elems:
                    if not e in traj:
                        traj.append(e)
                        q.put(e)
            inds_in_order.extend(traj)
            for e in traj:
                inds_pool.remove(e)
    new_tensor = dataset[np.array(inds_in_order)]
    
    values = []
    new_relations = []
    for k, v in relations.items():
        # 2 - adjacent, 1 - same trajectory
        if v == 1:
            values.append(w_t)
        elif v == 2:
            values.append(w_a)
        new_relations.append(k)
    new_relations = np.array(new_relations)
    relation_mat = csr_matrix((np.array(values), (new_relations[:, 0], new_relations[:, 1])),
                              shape=(len(dataset), len(dataset)))
    relation_mat = relation_mat[np.array(inds_in_order)][:, np.array(inds_in_order)]
    return new_tensor, relation_mat, inds_in_order

def train(model, 
          dataset, 
          output_dir, 
          use_channels=[],
          relation_mat=None, 
          mask=None, 
          n_epochs=10, 
          lr=0.001, 
          batch_size=16, 
          device='cuda:0',
          shuffle_data=False,
          transform=True,
          seed=None):
    """ Train function for VQ-VAE, VAE, IWAE, etc.

    Args:
        model (nn.Module): autoencoder model
        dataset (TensorDataset): dataset of training inputs
        output_dir (str): path for writing model saves and loss curves
        use_channels (list, optional): list of channel indices used for model
            training, by default all channels will be used
        relation_mat (scipy csr matrix or None, optional): if given, sparse 
            matrix of pairwise relations
        mask (TensorDataset or None, optional): if given, dataset of training 
            sample weight masks
        n_epochs (int, optional): number of epochs
        lr (float, optional): learning rate
        batch_size (int, optional): batch size
        device (str, optional): device (cuda or cpu) where models are running
        shuffle_data (bool, optional): shuffle data at the end of the epoch to
            add randomness to mini-batch; Set False when using matching loss
        transform (bool, optional): data augmentation
        seed (int, optional): random seed
    
    Returns:
        nn.Module: trained model

    """
    if not seed is None:
        np.random.seed(seed)
        t.manual_seed(seed)
    total_channels, n_z, x_size, y_size = dataset[0][0].shape[-4:]
    if len(use_channels) == 0:
        use_channels = list(range(total_channels))
    n_channels = len(use_channels)
    assert n_channels == model.num_inputs
    
    model = model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999))
    model.zero_grad()

    n_samples = len(dataset)
    n_batches = int(np.ceil(n_samples/batch_size))
    # Declare sample indices and do an initial shuffle
    sample_ids = np.arange(n_samples)
    if shuffle_data:
        np.random.shuffle(sample_ids)
    writer = SummaryWriter(output_dir)

    for epoch in range(n_epochs):
        mean_loss = {'recon_loss': [],  
                     'commitment_loss': [], 
                     'time_matching_loss': [],  
                     'total_loss': [],  
                     'perplexity': []}
        print('start epoch %d' % epoch) 
        for i in range(n_batches):
            # Deal with last batch might < batch size
            sample_ids_batch = sample_ids[i * batch_size:min((i + 1) * batch_size, n_samples)]
            batch = dataset[sample_ids_batch][0]
            assert len(batch.shape) == 5, "Input should be formatted as (batch, c, z, x, y)"
            batch = batch[:, np.array(use_channels)].permute(0, 2, 1, 3, 4).reshape((-1, n_channels, x_size, y_size))
            batch = batch.to(device)

            # Data augmentation
            if transform:
                for idx_in_batch in range(len(sample_ids_batch)):
                    img = batch[idx_in_batch]
                    flip_idx = np.random.choice([0, 1, 2])
                    if flip_idx != 0:
                        img = t.flip(img, dims=(flip_idx,))
                    rot_idx = int(np.random.choice([0, 1, 2, 3]))
                    batch[idx_in_batch] = t.rot90(img, k=rot_idx, dims=[1, 2])

            # Relation (adjacent frame, same trajectory)
            if not relation_mat is None:
                batch_relation_mat = relation_mat[sample_ids_batch][:, sample_ids_batch]
                batch_relation_mat = batch_relation_mat.todense()
                batch_relation_mat = t.from_numpy(batch_relation_mat).float().to(device)
            else:
                batch_relation_mat = None
            
            # Reconstruction mask
            if not mask is None:
                batch_mask = mask[sample_ids_batch][0][:, 1:2] # Hardcoded second slice (large mask)
                batch_mask = (batch_mask + 1.)/2. # Add a baseline weight
                batch_mask = batch_mask.permute(0, 2, 1, 3, 4).reshape((-1, 1, x_size, y_size))
                batch_mask = batch_mask.to(device)
            else:
                batch_mask = None
              
            _, loss_dict = model(batch, 
                                 time_matching_mat=batch_relation_mat, 
                                 batch_mask=batch_mask)
            loss_dict['total_loss'].backward()
            optimizer.step()
            model.zero_grad()

            for key, loss in loss_dict.items():
                if not key in mean_loss:
                    mean_loss[key] = []
                mean_loss[key].append(loss)
        # shuffle samples ids at the end of the epoch
        if shuffle_data:
            np.random.shuffle(sample_ids)
        for key, loss in mean_loss.items():
            mean_loss[key] = sum(loss)/len(loss) if len(loss) > 0 else -1.
            writer.add_scalar('Loss/' + key, mean_loss[key], epoch)
        writer.flush()
        print('epoch %d' % epoch)
        print(''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in mean_loss.items()]))
        t.save(model.state_dict(), os.path.join(output_dir, 'model_epoch%d.pt' % epoch))
    writer.close()
    return model
