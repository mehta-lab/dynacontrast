import os
import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import ImageDataset
from train import resnet as resnet
from preprocess.patch2zarr import pool_datasets
from utils.patch_utils import get_im_sites
import argparse
from utils.config_reader import YamlReader

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
                # print(batch.shape)
                batch = batch.to(device)
                code = model.encode(batch, out=encode_layer).cpu().data.numpy().squeeze()
                # print(code.shape)
                h_s.append(code)
        dats = np.concatenate(h_s, axis=0)
        output_fname = '{}_embeddings.npy'.format(data_name)
        print(f"\tsaving {os.path.join(output_dir, output_fname)}")
        with open(os.path.join(output_dir, output_fname), 'wb') as f:
            np.save(f, dats)

def main(method_, raw_dir_, supp_dir_, config_):
    method = method_
    inputs = raw_dir_
    outputs = supp_dir_
    gpu_ids = config_.inference.gpu_ids
    if method == 'pool_datasets':
        pool_datasets(config_)
    if config_.inference.fov:
        sites = config_.inference.fov
    else:
        # get all "XX-SITE_#" identifiers in raw data directory
        sites = get_im_sites(inputs)
    if method == 'encode':
        encode_patches(inputs, outputs, sites, config_, gpu=gpu_ids)

def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--method',
        type=str,
        required=False,
        choices=['encode', 'pool_datasets'],
        default='encode',
        help="Method: one of 'encode' or 'pool_datasets'",
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='path to yaml configuration file'
    )
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    config = YamlReader()
    config.read_config(arguments.config)
    if type(config.inference.weights) is not list:
        weights = [config.inference.weights]
    else:
        weights = config.inference.weights
    # batch run
    for (raw_dir, supp_dir) in zip(config.inference.raw_dirs, config.inference.supp_dirs):
        for weight in weights:
            config.inference.weights = weight
            main(arguments.method, raw_dir, supp_dir, config)


