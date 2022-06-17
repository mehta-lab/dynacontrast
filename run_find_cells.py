from preprocess.find_cells import find_cells_mp
from utils.patch_utils import get_im_sites
from multiprocessing import Process
import os
import numpy as np
from utils.logger import make_logger

import argparse
from utils.config_reader import YamlReader


class Worker(Process):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs

    def run(self):
        find_cells_mp(*self.inputs, rerun=True)

def main(raw_dir_, supp_dir_, config_):
    inputs = raw_dir_
    outputs = supp_dir_
    n_workers = config.segmentation.num_workers
    os.makedirs(outputs, exist_ok=True)
    logger = make_logger(
        log_dir=outputs,
        log_level=20,
    )
    logger.info('Finding cells in {}'.format(inputs))
    assert len(config_.segmentation.channels) > 0, "At least one channel must be specified"

    if config_.segmentation.fov:
        sites = config.segmentation.fov
    else:
        # get all "XX-SITE_#" identifiers in raw data directory
        sites = get_im_sites(inputs)

    segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site))]
    # print(segment_sites)
    sep = np.linspace(0, len(segment_sites), n_workers + 1).astype(int)

    processes = []
    for i in range(n_workers):
        _sites = segment_sites[sep[i]:sep[i + 1]]
        args = (inputs, outputs, _sites)
        process = Worker(args)
        process.start()
        processes.append(process)
    for p in processes:
        p.join()


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
        help='path to yaml configuration file.  Run_segmentation takes arguments from "inference" category'
    )
    
    return parser.parse_args()


if __name__ == '__main__':

    arguments = parse_args()
    config = YamlReader()
    config.read_config(arguments.config)

    # batch run
    for (raw_dir, supp_dir) in list(zip(config.segmentation.raw_dirs, config.segmentation.supp_dirs)):
        main(raw_dir, supp_dir, config)
