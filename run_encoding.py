import os
import numpy as np
import logging
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
from utils.logger import make_logger

def encode_patches(raw_dir: str,
                   config_: YamlReader,
                   gpu: int=0,
                   ):
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
    log = logging.getLogger('dynacontrast.log')
    model_dir = config_.inference.weights
    n_chan = config_.inference.n_channels
    network = config_.inference.network
    network_width = config_.inference.network_width
    batch_size = config_.inference.batch_size
    num_workers = config_.inference.num_workers
    normalization = config_.inference.normalization
    projection = config_.inference.projection
    splits = config_.inference.splits

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
            zarr_path = os.path.join(raw_dir, 'cell_patches_datasetnorm_{}.zarr'.format(split))
        elif normalization == 'patch':
            zarr_path = os.path.join(raw_dir, 'cell_patches_{}.zarr'.format(split))
        else:
            raise ValueError('Parameter "normalization" must be "dataset" or "patch"')
        if not os.path.isdir(zarr_path):
            msg = '{} is not found.'.format(zarr_path)
            log.error(msg)
            raise FileNotFoundError(msg)
        datasets[split] = zarr.open(zarr_path)
        datasets[split] = ImageDataset(datasets[split])
    device = torch.device('cuda:%d' % gpu)
    print('Encoding images using gpu {}...'.format(gpu))
    # Only ResNet is available now
    if 'ResNet' not in network:
        raise ValueError('Network {} is not available'.format(network))

    for data_name, dataset in datasets.items():
        network_cls = getattr(resnet, 'EncodeProject')
        model = network_cls(arch=network, num_inputs=n_chan, width=network_width)
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

def main(raw_dir, config_):
    logger = make_logger(
        log_dir=raw_dir,
        log_level=20,
    )
    gpu_id = config_.inference.gpu_id
    encode_patches(raw_dir, config_, gpu=gpu_id)

def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

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
    for raw_dir in config.inference.raw_dirs:
        for weight in weights:
            config.inference.weights = weight
            main(raw_dir, config)


