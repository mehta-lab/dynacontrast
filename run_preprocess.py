from preprocess.find_cells import find_cells_mp
from preprocess.patch2zarr import pool_positions
from preprocess.track import build_trajectories
from preprocess.extract_patches import get_patches_mp
from utils.patch_utils import get_im_sites
from utils.logger import make_logger
from multiprocessing import Process
import os
import numpy as np
import argparse
from utils.config_reader import YamlReader


class Worker(Process):
    def __init__(self, inputs, cpu_id=0, method='get_patches_mp'):
        super().__init__()
        self.cpu_id = cpu_id
        self.inputs = inputs
        self.method = method

    def run(self):
        if self.method == 'find_cells':
            find_cells_mp(*self.inputs, rerun=True)
        elif self.method == 'get_patches':
            get_patches_mp(*self.inputs)
        elif self.method == 'build_trajectories':
            build_trajectories(*self.inputs)

def multiproc(raw, supp, config, segment_sites, method, n_workers):
    processes = []
    sep = np.linspace(0, len(segment_sites), n_workers + 1).astype(int)
    for i in range(n_workers):
        _sites = segment_sites[sep[i]:sep[i + 1]]
        args = (raw, supp, _sites, config)
        p = Worker(args, cpu_id=i, method=method)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def main(raw_dir, supp_dir, config):
    fov = config.preprocess.fov
    n_workers = config.preprocess.num_workers
    os.makedirs(supp_dir, exist_ok=True)
    logger = make_logger(
        log_dir=supp_dir,
        log_level=20,
    )

    assert len(config.preprocess.channels) > 0, "At least one channel must be specified"

    if fov:
        sites = fov
    else:
        # get all "XX-SITE_#" identifiers in raw data directory
        sites = get_im_sites(raw_dir)

    # if probabilities and formatted stack exist
    segment_sites = [site for site in sites if os.path.exists(os.path.join(raw_dir, "%s.npy" % site)) and \
                     os.path.exists(os.path.join(raw_dir, "%s_NNProbabilities.npy" % site))]
    if len(segment_sites) == 0:
        raise AttributeError("no sites found in raw directory with preprocessed data and matching NNProbabilities")
    logger.info('Finding cells in {}'.format(raw_dir))
    multiproc(raw_dir, supp_dir, config, segment_sites, method='find_cells', n_workers=n_workers)
    logger.info('Extract patches in {}'.format(raw_dir))
    multiproc(raw_dir, supp_dir, config, segment_sites, method='get_patches', n_workers=n_workers)
    if hasattr(config.preprocess, 'track_dim'):
        multiproc(raw_dir, supp_dir, config, segment_sites, method='build_trajectories', n_workers=n_workers)
    # assemble patches into a single zarr file
    try:
        pool_positions(raw_dir, supp_dir, sites, config)
    except Exception as e:
        logger.error('pool_positions failed for {}. '.format(supp_dir))
        logger.exception('')
    return

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

    # batch run
    for (raw_dir, supp_dir) in list(zip(config.preprocess.raw_dirs, config.preprocess.supp_dirs)):
        main(raw_dir, supp_dir, config)
