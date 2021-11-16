# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:27:26 2021

@author: Zhenqin Wu
"""

import numpy as np
import os
import matplotlib

from SingleCellPatch.patch_utils import within_range, im_adjust

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pickle
from skimage import color
import logging

log = logging.getLogger(__name__)

""" Functions for clustering single cells from semantic segmentation """


def instance_clustering(inst_segmentation,
                        ct_thr=(0, np.inf),
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


def process_site_instance_segmentation(site,
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
    # Should be of size (n_frame, n_channels, z(1), x(2048), y(2048)), uint16
    print(f"\tLoading {raw_data}")
    image_stack = np.load(raw_data)
    # Should be of size (n_frame, n_classes, z(1), x(2048), y(2048)), float
    print(f"\tLoading {raw_data_segmented}")
    segmentation_stack = np.load(raw_data_segmented)
    meta_list = []
    cell_positions = {}
    cell_pixel_assignments = {}
    for t_point in range(image_stack.shape[0]):
        cell_positions[t_point] = {}
        cell_pixel_assignments[t_point] = {}
        for z in range(image_stack.shape[2]):
            print("\tClustering time {} z {}".format(t_point, z))
            img = image_stack[t_point, 0, z, ...] # get phase channel
            cell_segmentation = segmentation_stack[t_point, 0, z, ...] # assume the first channel is nuclei
            instance_map_path = os.path.join(site_supp_files_folder, 'segmentation_t{}_z{}.png'.format(t_point, z))
            #TODO: expose instance clustering parameters in config
            cell_ids, positions, cell_sizes, pixel_ids, positions_labels = \
                instance_clustering(cell_segmentation, save_fig=True, map_path=instance_map_path)
            cell_positions[t_point][z] = list(zip(cell_ids, positions)) # List of cell: (cell_id, mean_pos)
            cell_pixel_assignments[t_point][z] = (pixel_ids, positions_labels)

            # Save instance segmentation results as image
            if save_fig is not None:
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
                meta_row = {'FOV': site,
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
