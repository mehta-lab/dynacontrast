from utils.patch_VAE import encode_patches, pool_datasets
from SingleCellPatch.patch_utils import get_im_sites
import argparse
from utils.config_reader import YamlReader

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
        required=True,
        choices=['encode', 'pool_datasets'],
        default='assemble',
        help="Method: one of 'assemble', 'encode' or 'trajectory_matching'",
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
