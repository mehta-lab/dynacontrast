# bchhun, {2020-02-21}

import os
import numpy as np
import pandas as pd
# from NNsegmentation.models import Segment
from NNsegmentation.data import predict_whole_map
# from keras import backend as K
from SingleCellPatch.instance_clustering import process_site_instance_segmentation
from utils.config_reader import YamlReader
import logging
log = logging.getLogger(__name__)


def segmentation(raw_folder_: str,
                 supp_folder_: str,
                 val_folder: str,
                 sites: list,
                 config_: YamlReader,
                 **kwargs):
    """ Wrapper method for semantic segmentation

    This method performs predicion on all specified sites included in the
    input paths.

    Model weight path should be provided, if not a default path will be used:
        UNet: "NNsegmentation/temp_save_unsaturated/final.h5"

    Resulting segmentation results and sample segentation image will be saved
    in the summary folder as "*_NNProbabilities.npy"


    Args:
        summary_folder (str): folder for raw data, segmentation and
            summarized results
        supp_folder:
        val_folder:
        sites (list of str): list of site names

        n_classes (int, optional): number of prediction classes
        window_size (int, optional): winsow size for segmentation model
            prediction
        batch_size (int, optional): batch size
        num_pred_rnd (int, optional): number of extra prediction rounds
            each round of supplementary prediction will be initiated with
            different offset

    """

    weights = config_.inference.weights
    n_classes = config_.inference.num_classes
    channels = config_.inference.channels
    window_size = config_.inference.window_size
    batch_size = config_.inference.batch_size
    n_supp = config_.inference.num_pred_rnd

    if config_.inference.model == 'UNet':
        model = Segment(input_shape=(len(channels),
                                     window_size,
                                     window_size), n_classes=n_classes)
    else:
        raise NotImplementedError(f"segmentation model {config_.inference.model} not implemented")

    try:
        if weights:
            model.load(weights)
        else:
            model.load('NNsegmentation/temp_save_unsaturated/final.h5')
    except Exception as ex:
        log.error(ex)
        raise ValueError("Error in loading UNet weights")

    for site in sites:
        site_path = os.path.join(raw_folder_, '%s.npy' % site)
        if not os.path.exists(site_path):
            log.info("Site not found %s" % site_path, flush=True)
        else:
            log.info("Predicting %s" % site_path, flush=True)
            try:
                # Generate semantic segmentation
                predict_whole_map(site_path,
                                  model,
                                  use_channels=np.array(channels).astype(int),
                                  batch_size=batch_size,
                                  n_supp=n_supp,
                                  **kwargs)
            except Exception as ex:
                log.error(ex)
                log.error("Error in predicting site %s" % site, flush=True)
    return


def instance_segmentation(raw_folder: str,
                          supp_folder: str,
                          sites: list,
                          rerun=False,
                          **kwargs):
    """ Helper function for instance segmentation

    Wrapper method `process_site_instance_segmentation` will be called, which
    loads "*_NNProbabilities.npy" files and performs instance segmentation.

    Results will be saved in the supplementary data folder, including:
        "cell_positions.pkl": dict of cells in each frame (IDs and positions);
        "cell_pixel_assignments.pkl": dict of pixel compositions of cells
            in each frame;
        "segmentation_*.png": image of instance segmentation results.

    Args:
        raw_folder (str): folder for raw data, segmentation and summarized results
        supp_folder (str): folder for supplementary data
        sites (list of str): list of site names
        config (YamlReader):

    """

    # meta_list = []
    for site in sites:
        site_path = os.path.join(raw_folder, '%s.npy' % site)
        site_segmentation_path = os.path.join(raw_folder,
                                              '%s_NNProbabilities_cp_masks.npy' % site)
        if not os.path.exists(site_path) or not os.path.exists(site_segmentation_path):
            log.info("Site not found %s" % site_path)
            continue

        log.info("Clustering %s" % site_path)
        site_supp_files_folder = os.path.join(supp_folder,
                                              '%s-supps' % site[:2],
                                              '%s' % site)

        if os.path.exists(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl')) and not rerun:
            log.info('Found previously saved instance clustering output in {}. Skip processing...'
                  .format(site_supp_files_folder))
            continue
        elif not os.path.exists(site_supp_files_folder):
            os.makedirs(site_supp_files_folder, exist_ok=True)

        meta_list_site = process_site_instance_segmentation(site,
                                                           site_path,
                                                           site_segmentation_path,
                                                           site_supp_files_folder,
                                                           **kwargs)
        df_meta = pd.DataFrame.from_dict(meta_list_site)
        if os.path.isfile(os.path.join(raw_folder, 'metadata.csv')): # merge existing metadata if it exists
            df_meta_exp = pd.read_csv(os.path.join(raw_folder, 'metadata.csv'), index_col=0)
            df_meta = pd.merge(df_meta,
                               df_meta_exp,
                               how='left', on='position', validate='m:1')

        df_meta.to_csv(os.path.join(site_supp_files_folder, 'patch_meta.csv'), sep=',')
    return
