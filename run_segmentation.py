# bchhun, {2020-02-21}

from utils.segmentation import segmentation, instance_segmentation
from SingleCellPatch.patch_utils import get_im_sites
from utils.segmentation_validation import segmentation_validation_michael
from multiprocessing import Process
import os
import numpy as np
import logging
from utils.logger import make_logger
log = logging.getLogger(__name__)

import argparse
from utils.config_reader import YamlReader


class Worker(Process):
    def __init__(self, inputs, method='segmentation'):
        super().__init__()
        self.inputs = inputs
        self.method = method

    def run(self):
        log.info(f"running instance segmentation")
        instance_segmentation(*self.inputs, rerun=True)

def main(method_, raw_dir_, supp_dir_, config_):
    method = method_
    inputs = raw_dir_
    outputs = supp_dir_
    n_workers = config.segmentation.num_workers
    logger = make_logger(
        log_dir=outputs,
        log_level=20,
    )
    assert len(config_.segmentation.channels) > 0, "At least one channel must be specified"


    # instance segmentation requires raw (stack, NNprob), supp (to write outputs) to be defined
    if method != 'instance_segmentation':
        raise NotImplementedError(f"Method flag {method} not implemented. Use Cellpose to generate cell segmentation")

    # all methods all require
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
        process = Worker(args, method=method)
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
        '-m', '--method',
        type=str,
        required=False,
        choices=['segmentation', 'instance_segmentation'],
        default='instance_segmentation',
        help="Method: 'segmentation' or 'instance_segmentation'",
    )

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
        main(arguments.method, raw_dir, supp_dir, config)
