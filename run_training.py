import os
import numpy as np
import argparse
import torch as t
import torch.nn as nn
import pickle
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
from scipy.sparse import csr_matrix
from utils.train_utils import EarlyStopping, zscore, zscore_patch, train_val_split_by_col
from dataset.dataset import TripletDataset
from dataset.augmentation import augment_img
from HiddenStateExtractor.losses import AllTripletMiner, NTXent
from HiddenStateExtractor.resnet import EncodeProject
import HiddenStateExtractor.vae as vae

from configs.config_reader import YamlReader
import queue


def reorder_with_trajectories(dataset, relations, seed=None):
    """ Reorder `dataset` to facilitate training with matching loss

    Args:
        dataset (TensorDataset): dataset of training inputs
        relations (dict): dict of pairwise relationship (adjacent frames, same 
            trajectory)
        seed (int or None, optional): if given, random seed

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
    new_tensor = dataset.tensors[0][np.array(inds_in_order)]
    
    values = []
    new_relations = []
    for k, v in relations.items():
        # 2 - adjacent, 1 - same trajectory
        if v == 1:
            values.append(1)
        elif v == 2:
            values.append(2)
        new_relations.append(k)
    new_relations = np.array(new_relations)
    relation_mat = csr_matrix((np.array(values), (new_relations[:, 0], new_relations[:, 1])),
                              shape=(len(dataset), len(dataset)))
    relation_mat = relation_mat[np.array(inds_in_order)][:, np.array(inds_in_order)]
    return TensorDataset(new_tensor), relation_mat, inds_in_order


def unzscore(im_norm, mean, std):
    """
    Revert z-score normalization applied during preprocessing. Necessary
    before computing SSIM

    :param input_image: input image for un-zscore
    :return: image at its original scale
    """

    im = im_norm * (std + np.finfo(float).eps) + mean

    return im


def concat_relations(relations, labels, offsets):
    """combine relation dictionaries from multiple datasets

    Args:
        relations (list): list of relation dict to combine
        labels (list): list of label array to combine
        offsets (list): offset to add to the indices

    Returns: new_relations (dict): dictionary of combined relations

    """
    new_relations = {}
    new_labels = []
    for relation, label, offset in zip(relations, labels, offsets):
        old_keys = relation.keys()
        new_keys = [(id1 + offset, id2 + offset) for id1, id2 in old_keys]
        new_label = label + offset
        # make a new dict with updated keys
        relation = dict(zip(new_keys, relation.values()))
        new_relations.update(relation)
        new_labels.append(new_label)
    new_labels = np.concatenate(new_labels, axis=0)
    return new_relations, new_labels


def get_relation_tensor(relation_mat, sample_ids, device='cuda:0'):
    """
    Slice relation matrix according to sample_ids; convert to torch tensor
    Args:
        relation_mat (scipy sparse array): symmetric matrix describing the relation between samples
        sample_ids (list): row & column ids to select
        device (str): device to run the model on

    Returns:
        batch_relation_mat (torch tensor or None): sliced relation matrix

    """
    if relation_mat is None:
        return None
    batch_relation_mat = relation_mat[sample_ids, :]
    batch_relation_mat = batch_relation_mat[:, sample_ids]
    batch_relation_mat = batch_relation_mat.todense()
    batch_relation_mat = t.from_numpy(batch_relation_mat).float()
    if device:
        batch_relation_mat = batch_relation_mat.to(device)
    return batch_relation_mat


def get_mask(mask, sample_ids, device='cuda:0'):
    """
    Slice cell masks according to sample_ids; convert to torch tensor
    Args:
        mask (numpy array): cell masks for dataset
        sample_ids (list): mask ids to select
        device (str): device to run the model on

    Returns:
        batch_mask (torch tensor or None): sliced relation matrix
    """
    if mask is None:
        return None
    batch_mask = mask[sample_ids][0][:, 1:2, :, :]  # Hardcoded second slice (large mask)
    batch_mask = (batch_mask + 1.) / 2.
    batch_mask = batch_mask.to(device)
    return batch_mask


def run_one_batch(model, batch, train_loss, model_kwargs = None, optimizer=None,
                transform=False, training=True):
    """ Train on a single batch of data
    Args:
        model (nn.Module): pytorch model object
        batch (TensorDataset): batch of training or validation inputs
        train_loss (dict): batch-wise training or validation loss
        optimizer: pytorch optimizer
        batch_relation_mat (np array or None): matrix of pairwise relations
        batch_mask (TensorDataset or None): if given, dataset of training
            sample weight masks
        transform (bool): data augmentation if true
        training (bool): Set True for training and False for validation (no weights update)

    Returns:
        model (nn.Module): updated model object
        train_loss (dict): updated batch-wise training or validation loss

    """
    if transform:
        for idx_in_batch in range(len(batch)):
            img = batch[idx_in_batch]
            flip_idx = np.random.choice([0, 1, 2])
            if flip_idx != 0:
                img = t.flip(img, dims=(flip_idx,))
            rot_idx = int(np.random.choice([0, 1, 2, 3]))
            batch[idx_in_batch] = t.rot90(img, k=rot_idx, dims=[1, 2])
    _, train_loss_dict = model(batch, **model_kwargs)
    if training:
        train_loss_dict['total_loss'].backward()
        optimizer.step()
        model.zero_grad()
    for key, loss in train_loss_dict.items():
        if key not in train_loss:
            train_loss[key] = []
        # if isinstance(loss, t.Tensor):
        loss = float(loss)  # float here magically removes the history attached to tensors
        train_loss[key].append(loss)
    # print(train_loss_dict)
    del batch, train_loss_dict
    return model, train_loss


def train(model, dataset, output_dir, relation_mat=None, mask=None,
          n_epochs=10, lr=0.001, batch_size=16, device='cuda:0', shuffle_data=False,
          transform=None, val_split_ratio=0.15, patience=20):
    """ Legacy train function for VAE models.

    Args:
        model (nn.Module): autoencoder model
        dataset (TensorDataset): dataset of training inputs
        relation_mat (scipy csr matrix or None, optional): if given, sparse
            matrix of pairwise relations
        mask (TensorDataset or None, optional): if given, dataset of training
            sample weight masks
        n_epochs (int, optional): number of epochs
        lr (float, optional): learning rate
        batch_size (int, optional): batch size
        device (str): device to run the model on
        shuffle_data (bool): shuffle data at the end of the epoch to add randomness to mini-batch.
            Set False when using matching loss
        transform (bool): data augmentation if true
        val_split_ratio (float or None): fraction of the dataset used for validation
        patience (int or None): Number of epochs to wait before stopping training if validation loss does not improve.

    Returns:
        nn.Module: trained model

    """
    assert val_split_ratio is None or 0 < val_split_ratio < 1
    # early stopping requires validation set
    if patience is not None:
        assert val_split_ratio is not None
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999))
    model.zero_grad()
    n_samples = len(dataset)
    # Declare sample indices and do an initial shuffle
    sample_ids = list(range(n_samples))
    split = int(np.floor(val_split_ratio * n_samples))
    # randomly choose the split start
    split_start = np.random.randint(0, n_samples - split)
    if shuffle_data:
        np.random.shuffle(sample_ids)
    val_ids = sample_ids[split_start: split_start + split]
    train_ids = sample_ids[:split_start] + sample_ids[split_start + split:]
    n_train = len(train_ids)
    n_val = len(val_ids)
    n_batches = int(np.ceil(n_train / batch_size))
    n_val_batches = int(np.ceil(n_val / batch_size))
    writer = SummaryWriter(output_dir)
    model_path = os.path.join(output_dir, 'model.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)
    for epoch in range(n_epochs):
        train_loss = {}
        val_loss = {}
        print('start epoch %d' % epoch)
        # loop through training batches
        for i in range(n_batches):
            # deal with last batch might < batch size
            train_ids_batch = train_ids[i * batch_size:min((i + 1) * batch_size, n_train)]
            batch = dataset[train_ids_batch][0].to(device)
            # Relation (adjacent frame, same trajectory)
            batch_relation_mat = get_relation_tensor(relation_mat, train_ids_batch, device=device)
            # Reconstruction mask
            batch_mask = get_mask(mask, train_ids_batch, device=device)
            model, train_loss = \
                run_one_batch(model, batch, train_loss, optimizer=optimizer,
                              model_kwargs={'time_matching_mat': batch_relation_mat,
                              'batch_mask': batch_mask}, transform=transform, training=True)
        # loop through validation batches
        for i in range(n_val_batches):
            val_ids_batch = val_ids[i * batch_size:min((i + 1) * batch_size, n_val)]
            batch = dataset[val_ids_batch][0].to(device)
            # Relation (adjacent frame, same trajectory)
            batch_relation_mat = get_relation_tensor(relation_mat, val_ids_batch, device=device)
            # Reconstruction mask
            batch_mask = get_mask(mask, val_ids_batch, device)
            model, val_loss = \
                run_one_batch(model, batch, val_loss, optimizer=optimizer,
                              model_kwargs={'time_matching_mat': batch_relation_mat,
                                            'batch_mask': batch_mask}, transform=transform, training=False)
        # shuffle train ids at the end of the epoch
        if shuffle_data:
            np.random.shuffle(train_ids)
        for key, loss in train_loss.items():
            train_loss[key] = sum(loss) / len(loss)
            writer.add_scalar('Loss/' + key, train_loss[key], epoch)
        for key, loss in val_loss.items():
            val_loss[key] = sum(loss) / len(loss)
            writer.add_scalar('Val loss/' + key, val_loss[key], epoch)
        early_stopping(val_loss['total_loss'], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        writer.flush()
        print('epoch %d' % epoch)
        print('train: ', ''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in train_loss.items()]))
        print('validation: ', ''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in val_loss.items()]))
    writer.close()
    return model


def train_with_loader(model, train_loader, val_loader, output_dir,
          n_epochs=10, lr=0.001, device='cuda:0',
        patience=20, earlystop_metric='total_loss',
          retrain=False, log_step_offset=0):
    """ Train function using dataloders.

    Args:
        model (nn.Module): model
        train_loader (data loader): dataset of training inputs
        n_epochs (int, optional): number of epochs
        lr (float, optional): learning rate
        device (str): device to run the model on
        earlystop_metric (str): metric to monitor for early stopping
        patience (int or None): Number of epochs to wait before stopping training if validation loss does not improve.
        retrain (bool): Retrain the model from scratch if True. Load existing model and continue training otherwise

    Returns:
        nn.Module: trained model

    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pt')
    if os.path.exists(model_path) and not retrain:
        print('Found previously saved model state {}. Continue training...'.format(model_path))
        model.load_state_dict(t.load(model_path))

    # early stopping requires validation set
    if patience is not None:
        assert val_loader is not None
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999))
    model.zero_grad()
    writer = SummaryWriter(output_dir)
    model_path = os.path.join(output_dir, 'model.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)
    for epoch in tqdm(range(log_step_offset, n_epochs), desc='Epoch'):
        train_loss = {}
        val_loss = {}
        # loop through training batches
        model.train()
        with tqdm(train_loader, desc='train batch') as batch_pbar:
            for b_idx, batch in enumerate(batch_pbar):
                labels, data = batch
                labels = t.cat([label for label in labels], axis=0).to(device)
                batch = t.cat([datum for datum in data], axis=0).to(device)
                model, train_loss = \
                    run_one_batch(model, batch, train_loss, model_kwargs={'labels': labels}, optimizer=optimizer,
                                  transform=False, training=True)
        # loop through validation batches
        model.eval()
        with t.no_grad():
            with tqdm(val_loader, desc='val batch') as batch_pbar:
                for b_idx, batch in enumerate(batch_pbar):
                    labels, data = batch
                    labels = t.cat([label for label in labels], axis=0).to(device)
                    data = t.cat([datum for datum in data], axis=0).to(device)
                    model, val_loss = \
                        run_one_batch(model, data, val_loss, model_kwargs={'labels': labels}, optimizer=optimizer,
                                     transform=False, training=False)
        for key, loss in train_loss.items():
            train_loss[key] = sum(loss) / len(loss)
            writer.add_scalar('Loss/' + key, train_loss[key], epoch)
        for key, loss in val_loss.items():
            val_loss[key] = sum(loss) / len(loss)
            writer.add_scalar('Val loss/' + key, val_loss[key], epoch)
        writer.flush()
        print('epoch %d' % epoch)
        print('train: ', ''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in train_loss.items()]))
        print('val:   ', ''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in val_loss.items()]))
        early_stopping(val_loss[earlystop_metric], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    writer.close()
    return model

def main(config_):
    """
    Args:
        config_ (object): config file object

    Returns:

    """
    config = YamlReader()
    config.read_config(config_)

    # Settings
    # estimate mean and std from the data
    channel_mean = config.training.channel_mean
    channel_std = config.training.channel_std

    raw_dirs = config.training.raw_dirs
    train_dirs = config.training.weights_dirs
    supp_dirs = config.training.supp_dirs
    for train_dir in train_dirs:
        os.makedirs(train_dir, exist_ok=True)

    ### Settings ###
    network = config.training.network
    network_width = config.training.network_width
    num_inputs = config.training.num_inputs
    num_hiddens = config.training.num_hiddens
    num_residual_hiddens = config.training.num_residual_hiddens
    num_residual_layers = config.training.num_residual_layers
    num_embeddings = config.training.num_embeddings
    commitment_cost = config.training.commitment_cost
    weight_matching = config.training.weight_matching
    w_a = config.training.w_a
    w_t = config.training.w_t
    w_n = config.training.w_n
    margin = config.training.margin
    val_split_ratio = config.training.val_split_ratio
    learn_rate = config.training.learn_rate
    patience = config.training.patience
    n_pos_samples = config.training.n_pos_samples
    batch_size = config.training.batch_size
    # adjusted batch size for dataloaders
    batch_size_adj = int(np.floor(batch_size/n_pos_samples))
    num_workers = config.training.num_workers
    n_epochs = config.training.n_epochs
    gpu_id = config.training.gpu_id
    # earlystop_metric = 'total_loss'
    retrain = config.training.retrain
    earlystop_metric = 'positive_triplet'
    model_name = config.training.model_name
    start_model_path = config.training.start_model_path
    start_epoch = config.training.start_epoch
    use_mask = config.training.use_mask
    channels = config.training.channels
    normalization = config.training.normalization
    loss = config.training.loss
    temperature = config.training.temperature
    intensity_jitter = config.training.augmentations.intensity_jitter
    device = t.device('cuda:%d' % gpu_id)
    # use data loader for training ResNet
    use_loader = False
    if 'ResNet' in network:
        use_loader = True

    dir_sets = list(zip(supp_dirs, train_dirs, raw_dirs))
    datasets = []
    masks = []
    relations = []
    labels = []
    id_offsets = [0]
    ### Load Data ###
    df_meta_all = []
    for supp_dir, train_dir, raw_dir in dir_sets:
        os.makedirs(train_dir, exist_ok=True)
        print(f"\tloading static patches {os.path.join(raw_dir, 'im_static_patches.pkl')}")
        dataset = pickle.load(open(os.path.join(raw_dir, 'im_static_patches.pkl'), 'rb'))
        dataset = dataset[:, channels, ...]
        print('dataset.shape:', dataset.shape)
        label = pickle.load(open(os.path.join(raw_dir, "im_static_patches_labels.pkl"), 'rb'))
        # Note that `relations` is depending on the order of fs (should not sort)
        # `relations` is generated by script "generate_trajectory_relations.py"
        relation = pickle.load(open(os.path.join(raw_dir, 'im_static_patches_relations.pkl'), 'rb'))
        # dataset_mask = TensorDataset(dataset_mask.tensors[0][np.array(inds_in_order)])
        # print('relations:', relations)
        print('len(label):', len(label))
        print('len(dataset):', len(dataset))
        relations.append(relation)
        meta_path = os.path.join(supp_dir, 'im-supps', 'patch_meta.csv')
        df_meta = pd.read_csv(meta_path, index_col=0, converters={
            'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
        df_meta['data_dir'] = os.path.dirname(raw_dir)
        df_meta_all.append(df_meta)

        # TODO: handle non-singular z-dimension case earlier in the pipeline
        if normalization == 'dataset':
            dataset = zscore(np.squeeze(dataset), channel_mean=channel_mean, channel_std=channel_std).astype(np.float32)
        elif normalization == 'patch':
            dataset = zscore_patch(np.squeeze(dataset)).astype(np.float32)
        else:
            raise ValueError('Parameter "normalization" must be "dataset" or "patch"')
        datasets.append(dataset)
        labels.append(label)
        id_offsets.append(len(dataset))
        if use_mask:
            mask = pickle.load(open(os.path.join(raw_dir, 'im_static_patches_mask.pkl'), 'rb'))
            masks.append(mask)
    df_meta_all = pd.concat(df_meta_all, axis=0)
    df_meta_all.reset_index(drop=True, inplace=True)
    id_offsets = id_offsets[:-1]
    dataset = np.concatenate(datasets, axis=0)
    if use_mask:
        masks = np.concatenate(masks, axis=0)
    else:
        masks = None
    # dataset = zscore(dataset, channel_mean=channel_mean, channel_std=channel_std).astype(np.float32)
    relations, labels = concat_relations(relations, labels, offsets=id_offsets)
    print('len(labels):', len(labels))
    print('len(dataset):', len(dataset))
    # treat every patch as different
    # labels = np.arange(len(labels))
    # Save the model in the train directory of the last dataset
    model_dir = os.path.join(train_dir, model_name)
    #TODO: write dataset class for VAE models
    if not use_loader:
        dataset = TensorDataset(t.from_numpy(dataset).float())
        dataset, relation_mat, inds_in_order = reorder_with_trajectories(dataset, relations, seed=123)
        labels = labels[inds_in_order]
        network_cls = getattr(vae, network)
        model = network_cls(num_inputs=num_inputs,
                           num_hiddens=num_hiddens,
                           num_residual_hiddens=num_residual_hiddens,
                           num_residual_layers=num_residual_layers,
                           num_embeddings=num_embeddings,
                           commitment_cost=commitment_cost,
                           weight_matching=weight_matching,
                           w_a=w_a,
                           w_t=w_t,
                           w_n=w_n,
                           margin=margin,
                           device=device).to(device)
        model = train(model,
                      dataset,
                      output_dir=model_dir,
                      relation_mat=relation_mat,
                      mask=masks,
                      n_epochs=n_epochs,
                      lr=learn_rate,
                      batch_size=batch_size,
                      device=device,
                      transform=True,
                      val_split_ratio=val_split_ratio,
                      patience=patience,
                      )
    else:
        train_set, train_labels, val_set, val_labels, df_meta_all = \
            train_val_split_by_col(dataset, labels, df_meta_all, split_cols=['data_dir', 'FOV'], val_split_ratio=val_split_ratio, seed=0)
        os.makedirs(model_dir, exist_ok=True)
        meta_path = os.path.join(model_dir, 'patch_meta.csv')
        df_meta_all.to_csv(meta_path, sep=',')
        # SimCLR uses n_pos_samples=2
        tri_train_set = TripletDataset(train_labels, lambda index: augment_img(train_set[index], intensity_jitter=intensity_jitter), n_pos_samples)
        tri_val_set = TripletDataset(val_labels, lambda index: augment_img(val_set[index], intensity_jitter=intensity_jitter), n_pos_samples)
        # Data Loader
        train_loader = DataLoader(tri_train_set,
                                    batch_size=batch_size_adj,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    )
        val_loader = DataLoader(tri_val_set,
                                  batch_size=batch_size_adj,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  )
        if loss == 'triplet':
            loss_fn = AllTripletMiner(margin=margin).to(device)
        elif loss == 'ntxent':
            loss_fn = NTXent(tau=temperature).to(device)
        else:
            raise ValueError('Loss name {} is not defined.'.format(loss))

        # tri_loss = HardNegativeTripletMiner(margin=margin).to(device)
        ## Initialize Model ###

        model = EncodeProject(arch=network, loss=loss_fn, num_inputs=num_inputs, width=network_width).to(device)

        if start_model_path:
            print('Initialize the model with state {} ...'.format(start_model_path))
            model.load_state_dict(t.load(start_model_path))
        model = train_with_loader(model,
                              train_loader=train_loader,
                              val_loader=val_loader,
                              output_dir=model_dir,
                              n_epochs=n_epochs,
                              lr=learn_rate,
                              device=device,
                              patience=patience,
                              earlystop_metric=earlystop_metric,
                              retrain=retrain,
                              log_step_offset=start_epoch)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='path to yaml configuration file'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)

