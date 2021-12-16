import os
import numpy as np
import argparse
import dask
import dask.array as da
import torch as t
import torch.nn as nn
import pandas as pd
import time
import zarr
from numcodecs import blosc
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
from scipy.sparse import csr_matrix

from utils.patch_VAE import concat_relations
from utils.train_utils import EarlyStopping, DataLoader
from dataset.dataset import TripletDataset, TripletIterDataset, worker_init_fn
from dataset.augmentation import augment_img
from HiddenStateExtractor.losses import AllTripletMiner, NTXent
from HiddenStateExtractor.resnet import EncodeProject

from configs.config_reader import YamlReader
import queue
dask.config.set(scheduler='synchronous')
blosc.use_threads = True

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

    raw_dir = config.training.raw_dir
    train_dir = config.training.weights_dir
    # supp_dirs = config.training.supp_dirs
    os.makedirs(train_dir, exist_ok=True)

    ### Settings ###
    network = config.training.network
    network_width = config.training.network_width
    num_inputs = config.training.num_inputs
    margin = config.training.margin
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
    os.makedirs(train_dir, exist_ok=True)
    print('loading data {}'.format(raw_dir))
    t0 = time.time()
    if normalization == 'dataset':
        train_set = zarr.open(os.path.join(raw_dir, 'cell_patches_datasetnorm_train.zarr'))
        val_set = zarr.open(os.path.join(raw_dir, 'cell_patches_datasetnorm_val.zarr'))
        train_labels = np.load(os.path.join(raw_dir, 'patch_labels_datasetnorm_train.npy'))
        val_labels = np.load(os.path.join(raw_dir, 'patch_labels_datasetnorm_val.npy'))
        # df_meta_all = pd.read_csv(os.path.join(raw_dir, 'patch_meta_datasetnorm.csv'), index_col=0, converters={
        #     'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
    elif normalization == 'patch':
        train_set_sync = zarr.ProcessSynchronizer(os.path.join(raw_dir, 'cell_patches_train.sync'))
        train_set = zarr.open(os.path.join(raw_dir, 'cell_patches_train.zarr'), synchronizer=train_set_sync)
        val_set = zarr.open(os.path.join(raw_dir, 'cell_patches_val.zarr'))
        train_labels = np.load(os.path.join(raw_dir, 'patch_labels_train.npy'))
        val_labels = np.load(os.path.join(raw_dir, 'patch_labels_val.npy'))
        # df_meta_all = pd.read_csv(os.path.join(raw_dir, 'patch_meta.csv'), index_col=0, converters={
        #     'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
    else:
        raise ValueError('Parameter "normalization" must be "dataset" or "patch"')
    t1 = time.time()
    print('loading dataset takes:', t1 - t0)
    print('train dataset.shape:', train_set.shape)
    print('val dataset.shape:', val_set.shape)
    # PyTorch 1.2 can't detect length of IterableDataset
    n_batch_train = np.ceil(len(train_labels) / batch_size_adj)
    n_batch_val = np.ceil(len(val_labels) / batch_size_adj)
    # treat every patch as different
    # labels = np.arange(len(labels))
    # Save the model in the train directory of the last dataset
    model_dir = os.path.join(train_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    # SimCLR uses n_pos_samples=2
    tri_train_set = TripletIterDataset(labels=train_labels,
                                        data=train_set,
                                        data_fn=lambda img: augment_img(img, intensity_jitter=intensity_jitter),
                                        n_sample=n_pos_samples,
                                        shuffle=True,
                                        )
    tri_val_set = TripletIterDataset(labels=val_labels,
                                   data=val_set,
                                   data_fn=lambda img: augment_img(img, intensity_jitter=intensity_jitter),
                                   n_sample=n_pos_samples,
                                     shuffle=True,

                                     )
    # Data Loader
    train_loader = DataLoader(tri_train_set,
                                batch_size=batch_size_adj,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn
                                )
    val_loader = DataLoader(tri_val_set,
                              batch_size=batch_size_adj,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    print('loader length:', len(train_loader))
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

