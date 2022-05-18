# bchhun, {2020-02-21}

import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import color

from utils.config_reader import YamlReader
import logging

from utils.patch_utils import within_range, check_segmentation_dim, im_adjust

log = logging.getLogger(__name__)


def find_cells_mp(raw_folder: str,
                  supp_folder: str,
                  sites: list,
                  rerun=True,
                  **kwargs):
    """ Helper function for instance segmentation

    Wrapper method `find_cells` will be called, which
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
        site_supp_files_folder = os.path.join(supp_folder,
                                              '%s-supps' % site[:2],
                                              '%s' % site)
        if not os.path.exists(site_path) or not os.path.exists(site_segmentation_path):
            log.warning("Site not found %s" % site_path)
            continue

        if os.path.exists(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl')) and not rerun:
            log.warning('Found previously saved instance clustering output in {}. Skip processing...'
                  .format(site_supp_files_folder))
            continue

        # log.info("Process %s" % site_path)
        os.makedirs(site_supp_files_folder, exist_ok=True)
        try:
            meta_list_site = find_cells(site,
                                        site_path,
                                        site_segmentation_path,
                                        site_supp_files_folder,
                                        **kwargs)
        except Exception as e:
            log.error('Single cell detection failed for position {}. '.format(site))
            log.exception('')
            continue

        if len(meta_list_site) == 0:
            log.warning('No cell is detected for position {}'.format(site))
            print('No cell is detected for position {}'.format(site))
            continue
        df_meta = pd.DataFrame.from_dict(meta_list_site)
        if os.path.isfile(os.path.join(raw_folder, 'metadata.csv')): # merge existing metadata if it exists
            df_meta_exp = pd.read_csv(os.path.join(raw_folder, 'metadata.csv'), index_col=0)
            df_meta = pd.merge(df_meta,
                               df_meta_exp,
                               how='left', on='position', validate='m:1')

        df_meta.to_csv(os.path.join(site_supp_files_folder, 'patch_meta.csv'), sep=',')
    return


def find_cells_2D(inst_segmentation,
                  ):
    """ Perform instance clustering on a static frame

    Args:
        cell_segmentation (np.array): segmentation mask for the frame,
            size (n_classes(3), z(1), x, y)
        ct_thr (tuple, optional): lower and upper threshold for cell size
            (number of pixels in segmentation mask)
        map_path (str or None, optional): path to the image (if `save_fig`
            is True)

    Returns:
        (list * 3): 3 lists (MG, Non-MG, intermediate) of cell identifiers
            each entry in the list is a tuple of cell ID and cell center position
        np.array: array of x, y coordinates of foreground pixels
        np.array: array of cell IDs of foreground pixels

    """
    pixel_ids = np.nonzero(inst_segmentation) # ([row ID1, row ID2, ...], [col ID1, col ID2, ...]]
    positions_labels = inst_segmentation[pixel_ids]
    pixel_ids = np.transpose(pixel_ids)
    cell_ids, cell_sizes = np.unique(positions_labels, return_counts=True)
    # neglect unclustered pixels
    cell_sizes = cell_sizes[cell_ids >= 0]
    cell_ids = cell_ids[cell_ids >= 0]
    cell_positions = []
    cell_ids = list(cell_ids)
    cell_ids_new = []
    cell_sizes_new = []
    for cell_id, cell_size in zip(cell_ids, cell_sizes):
        # if cell_size <= ct_thr[0] or cell_size >= ct_thr[1]:
        #     # neglect cells that are too small/big
        #     continue
        points = pixel_ids[np.nonzero(positions_labels == cell_id)[0]]
        # calculate cell center
        mean_pos = np.mean(points, 0).astype(int)
        # define window
        window = [(mean_pos[0]-128, mean_pos[0]+128), (mean_pos[1]-128, mean_pos[1]+128)]
        # skip if cell has too many outlying points
        outliers = [p for p in points if not within_range(window, p)]
        if len(outliers) > len(points) * 0.05:
            continue
        cell_positions.append(mean_pos)
        cell_ids_new.append(cell_id)
        cell_sizes_new.append(cell_size)
    return cell_ids_new, cell_positions, cell_sizes_new, pixel_ids, positions_labels


def find_cells(site,
               raw_data,
               raw_data_segmented,
               site_supp_files_folder,
               save_fig=False,
               ):
    """
    Wrapper method for instance segmentation

    Results will be saved to the supplementary data folder as:
        "cell_positions.pkl": list of cells in each frame (IDs and positions);
        "cell_pixel_assignments.pkl": pixel compositions of cells;
        "segmentation_*.png": image of instance segmentation results.

    :param raw_data: (str) path to image stack (.npy)
    :param raw_data_segmented: (str) path to semantic segmentation stack (.npy)
    :param site_supp_files_folder: (str) path to the folder where supplementary files will be saved
    :param save_fig (bool, optional): if to save instance segmentation as an
            image
    :return:
    """

    # TODO: Size is hardcoded here
    # Should be of size (n_frame, n_channels, z, x, y), uint16
    # print(f"\tLoading {raw_data}")
    image_stack = np.load(raw_data)
    # Should be of size (n_frame, n_classes, z, x, y), float
    # print(f"\tLoading {raw_data_segmented}")
    segmentation_stack = np.load(raw_data_segmented)
    segmentation_stack = check_segmentation_dim(segmentation_stack)
    # print(segmentation_stack.shape)
    meta_list = []
    cell_positions = {}
    cell_pixel_assignments = {}
    for t_point in range(image_stack.shape[0]):
        cell_positions[t_point] = {}
        cell_pixel_assignments[t_point] = {}
        for z in range(image_stack.shape[2]):
            # print("\tClustering time {} z {}".format(t_point, z))
            cell_segmentation = segmentation_stack[t_point, 0, z, ...] # assume the first channel is nuclei
            instance_map_path = os.path.join(site_supp_files_folder, 'segmentation_t{}_z{}.png'.format(t_point, z))
            #TODO: expose instance clustering parameters in config
            cell_ids, positions, cell_sizes, pixel_ids, positions_labels = \
                find_cells_2D(cell_segmentation)
            cell_positions[t_point][z] = list(zip(cell_ids, positions)) # List of cell: (cell_id, mean_pos)
            cell_pixel_assignments[t_point][z] = (pixel_ids, positions_labels)

            # Save instance segmentation results as image
            if save_fig is not None:
                img = image_stack[t_point, 1, z, ...]  # get phase channel
                overlay = color.label2rgb(cell_segmentation, im_adjust(img), bg_label=0, alpha=0.3)
                plt.clf()
                plt.imshow(overlay)
                font = {'color': 'white', 'size': 4}
                for cell_id, mean_pos in zip(cell_ids, positions):
                    plt.text(mean_pos[1], mean_pos[0], str(cell_id), fontdict=font)
                plt.axis('off')
                plt.savefig(instance_map_path, dpi=300)
            # new metadata format
            for cell_id, cell_pos, cell_size in zip(cell_ids, positions, cell_sizes):
                meta_row = {'position': int(site[-3:]),
                            'time': t_point,
                            'slice': z,
                            'cell ID': cell_id,
                            'cell position': cell_pos,
                            'cell size': cell_size}
                meta_list.append(meta_row)
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'wb') as f:
        pickle.dump(cell_positions, f)
    with open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'wb') as f:
        pickle.dump(cell_pixel_assignments, f)
    return meta_list