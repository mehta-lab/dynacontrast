from utils.patch_VAE import assemble_VAE, encode_patches, trajectory_matching, pool_datasets
from SingleCellPatch.patch_utils import get_im_sites
from torch.multiprocessing import Process
import torch.multiprocessing as mp
import argparse
from utils.config_reader import YamlReader


class Worker(Process):
    def __init__(self, inputs, gpuid=0, method='assemble'):
        super().__init__()
        self.gpuid = gpuid
        self.inputs = inputs
        self.method = method

    def run(self):
        if self.method == 'assemble':
            #TODO: make "patch_type" part of the config
            assemble_VAE(*self.inputs, patch_type='mat')
        elif self.method == 'process':
            encode_patches(*self.inputs, gpu=self.gpuid)
        elif self.method == 'trajectory_matching':
            trajectory_matching(*self.inputs)


def main(method_, raw_dir_, supp_dir_, config_):
    method = method_

    inputs = raw_dir_
    outputs = supp_dir_
    weights = config_.inference.weights
    # channels = config_.inference.channels
    # network = config_.inference.model
    gpu_ids = config_.inference.gpu_ids

    # assert len(channels) > 0, "At least one channel must be specified"

    # todo file path checks can be done earlier
    # assemble needs raw (write file_paths/static_patches/adjusted_patches), and supp (read site-supps)
    if method == 'assemble':
        if not inputs:
            raise AttributeError("raw directory must be specified when method = assemble")
        if not outputs:
            raise AttributeError("supplementary directory must be specified when method = assemble")

    # process needs raw (load _file_paths), and target (torch weights)
    elif method == 'process':
        if not inputs:
            raise AttributeError("raw directory must be specified when method = process")
        if not weights:
            raise AttributeError("pytorch VQ-VAE weights path must be specified when method = process")

    # trajectory matching needs raw (load file_paths, write trajectories), supp (load cell_traj)
    elif method == 'trajectory_matching':
        if not inputs:
            raise AttributeError("raw directory must be specified when method = trajectory_matching")
        if not outputs:
            raise AttributeError("supplementary directory must be specified when method = trajectory_matching")
    elif method == 'pool_datasets':
        pool_datasets(config_)
    if config_.inference.fov:
        sites = config_.inference.fov
    else:
        # get all "XX-SITE_#" identifiers in raw data directory
        sites = get_im_sites(inputs)

    wells = set(s[:2] for s in sites)
    mp.set_start_method('spawn', force=True)
    for i, well in enumerate(wells):
        well_sites = [s for s in sites if s[:2] == well]
        args = (inputs, outputs, well_sites, config_)
        p = Worker(args, gpuid=gpu_ids[0], method=method)
        p.start()
        p.join()

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
        choices=['assemble', 'process', 'trajectory_matching', 'pool_datasets'],
        default='assemble',
        help="Method: one of 'assemble', 'process' or 'trajectory_matching'",
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