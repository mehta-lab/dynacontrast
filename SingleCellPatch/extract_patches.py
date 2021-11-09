#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:43:41 2019

@author: zqwu
"""

import cv2
import numpy as np
import os
import pickle
import matplotlib
import pandas as pd
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from .patch_utils import within_range, check_segmentation_dim, cv2_fn_wrapper, select_window, im_adjust, \
    get_cell_rect_angle

""" Functions for extracting single cells from static frames """

# filter 1 is for the masking of surrounding cells
size1 = 11
filter1 = np.zeros((size1, size1), dtype=int)
for i in range(size1):
    for j in range(size1):
        if np.sqrt((i-size1//2)**2 + (j-size1//2)**2) <= size1//2:
            filter1[i, j] = 1

# filter 2 is for (un-)masking of center cell
size2 = 21 # any pixel within a distance of 21 to center cell will not be masked
filter2 = np.zeros((size2, size2), dtype=int)
for i in range(size2):
    for j in range(size2):
        if np.sqrt((i-size2//2)**2 + (j-size2//2)**2) < size2//2:
            filter2[i, j] = 1

def generate_mask(positions, positions_labels, cell_id, window, window_segmentation):
    """ Generate mask matrix for surrounding cells

    Args:
        positions (np.array): array of x, y coordinates, size (N, 2)
        positions_labels (np.array): cell IDs of each pixel, size (N,)
        cell_id (int): target cell ID
        window (tuple): window centered around the target cell, 
            ((int, int), (int, int)) in the form of 
            ((x_low, x_up), (y_low, y_up))
        window_segmentation (np.array): pixel-level semantic segmentation in 
            the window area, size (n_classes, z(1), window_x, window_y)

    Returns:
        np.array: remove mask (1s will be filled by median)
        np.array: target mask
        np.array: enlarged target mask

    """
    x_size = window[0][1] - window[0][0]
    y_size = window[1][1] - window[1][0]
    remove_mask = np.zeros((x_size, y_size), dtype=int)
    target_mask = np.zeros((x_size, y_size), dtype=int)

    for i, p in enumerate(positions):
        if not within_range(window, p):
            continue
        if positions_labels[i] != cell_id and positions_labels[i] >= 0:
            # pixels belonging to other cells
            remove_mask[p[0] - window[0][0], p[1] - window[1][0]] = 1
        if positions_labels[i] == cell_id:
            # pixels belonging to target cell
            target_mask[p[0] - window[0][0], p[1] - window[1][0]] = 1

    remove_mask = np.sign(convolve2d(remove_mask, filter1, mode='same'))
    target_mask2 = np.sign(convolve2d(target_mask, filter2, mode='same'))
    # Target mask override remove mask
    remove_mask = ((remove_mask - target_mask2) > 0) * 1
    # Add negative boundary paddings
    remove_mask[np.where(window_segmentation[0, 0] == -1)] = 1
    return remove_mask.reshape((x_size, y_size)), \
        target_mask.reshape((x_size, y_size)), \
        target_mask2.reshape((x_size, y_size))


def process_site_extract_patches(site_path,
                                 site_segmentation_path, 
                                 site_supp_files_folder,
                                 window_size=256,
                                 channels=None,
                                 save_fig=False,
                                 reload=True,
                                 skip_boundary=False,
                                 **kwargs):
    """ Wrapper method for patch extraction

    Supplementary files generated by `process_site_instance_segmentation` will
    be loaded for each site, then individual cells from static frames will be
    extracted and saved.

    Results will be saved in supplementary data folder, including:
        "stacks_*.pkl": single cell patches for each time slice

    Args:
        site_path (str): path to image stack (.npy)
        site_segmentation_path (str): path to semantic segmentation stack (.npy)
        site_supp_files_folder (str): path to the folder where supplementary 
            files will be saved
        window_size (int, optional): default=256, x, y size of the patch
        channels (list, optional): channels to extract patches. Default is all the channels
        save_fig (bool, optional): if to save extracted patches (with
            segmentation mask)
        reload (bool, optional): if to load existing stack dat files
        skip_boundary (bool, optional): if to skip patches whose edges exceed
            the image size (do not pad)

    """

    # Load data
    image_stack = np.load(site_path)
    segmentation_stack = np.load(site_segmentation_path)
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'rb') as f:
        cell_positions = pickle.load(f)
    with open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'rb') as f:
        cell_pixel_assignments = pickle.load(f)
    meta_path = os.path.join(site_supp_files_folder, 'patch_meta.csv')
    df_meta = pd.read_csv(meta_path, index_col=0)
    n_z = 1
    if image_stack.ndim == 5:
        n_frames, n_channels, n_z, x_full_size, y_full_size = image_stack.shape
    elif image_stack.ndim == 4:
        n_frames, n_channels, x_full_size, y_full_size = image_stack.shape
        image_stack = np.expand_dims(image_stack, axis=2)
    else:
        raise ValueError('Input image must be 4 or 5D, not {}'.format(image_stack.ndim))
    if channels is None:
        channels = list(range(n_channels))
    image_stack = image_stack[:, channels, ...]
    for t_point in range(n_frames):
        for z in range(n_z):
            print("processing timepoint {} z {}".format(t_point, z))
            stack_dat_path = os.path.join(site_supp_files_folder, 'stacks_t{}_z{}.pkl'.format(t_point, z))
            if reload and os.path.exists(stack_dat_path):
                try:
                    site_data = pickle.load(open(stack_dat_path, 'rb'))
                    continue
                except Exception as e:
                    print(e)
                    site_data = {}
            else:
                site_data = {}
            print('Writing timepoint {} z {}'.format(t_point, z))
            raw_image = image_stack[t_point, :, z, ...]
            cell_segmentation = segmentation_stack[t_point, :, z, ...]
            # cell_segmentation = check_segmentation_dim(cell_segmentation)
            positions, positions_labels = cell_pixel_assignments[t_point][z]
            all_cells = cell_positions[t_point][z]

            # Define fillings for the masked pixels in this slice
            cells_to_keep = []
            background_positions = np.where(cell_segmentation[0] > 0.9)
            background_pool = np.array([np.median(raw_image[i][background_positions]) for i in range(n_channels)])
            background_filling = np.ones((n_channels, window_size, window_size)) * background_pool.reshape((n_channels, 1, 1))

            # Save all cells in this step, filtering will be performed during analysis
            cells_to_keep = []
            for cell_id, cell_position in all_cells:
                # print(f"cell_id : {cell_id}")
                cell_name = os.path.join(site_supp_files_folder, 't{}_z{}_cell{}'.format(t_point, z, cell_id))
                if cell_name in site_data:
                    continue
                # Define window based on cell center and extract mask
                window = [(cell_position[0]-window_size//2, cell_position[0]+window_size//2),
                          (cell_position[1]-window_size//2, cell_position[1]+window_size//2)]
                window_segmentation = select_window(cell_segmentation,
                                                    window,
                                                    padding=-1,
                                                    skip_boundary=skip_boundary)
                if window_segmentation is None:
                    # drop cell that did not get patched
                    df_meta.drop(df_meta[(df_meta['cell ID'] == cell_id) &
                                         (df_meta['time'] == t_point) &
                                         (df_meta['slice'] == z)].index, inplace=True)
                    continue
                # only keep the cells that has patches
                cells_to_keep.append(cell_id)
                remove_mask, tm, tm2 = generate_mask(positions,
                                                     positions_labels,
                                                     cell_id,
                                                     window,
                                                     window_segmentation)

                # Reshape (x, y) to (c, x, y)
                remove_mask = np.expand_dims(remove_mask, 0)
                tm = np.expand_dims(tm, 0)
                tm2 = np.expand_dims(tm2, 0)

                # Select submatrix from the whole slice
                output_mat = select_window(raw_image, window, padding=0, skip_boundary=skip_boundary)
                assert not output_mat is None
                masked_output_mat = output_mat * (1 - remove_mask) + background_filling * remove_mask
                site_data[cell_name] = {
                    "mat": np.concatenate([output_mat, tm, tm2], 0).astype('float64'),
                    "masked_mat": np.concatenate([masked_output_mat, tm, tm2], 0).astype('float64')
                    }

                if save_fig:
                    im_path = os.path.join(site_supp_files_folder, 'patch_t{}_z{}_cell{}'.format(t_point, z, cell_id))
                    save_single_cell_im(output_mat, masked_output_mat, tm, tm2, im_path)

            with open(stack_dat_path, 'wb') as f:
                print(f"WRITING STACKS TO {stack_dat_path}")
                pickle.dump(site_data, f)

            # remove cells that don't have patches, update cell_positions
            updated_cell_positions_tz = [cell for cell in all_cells if cell[0] in cells_to_keep]
            cell_positions[t_point][z] = updated_cell_positions_tz
    df_meta.reset_index(drop=True, inplace=True)
    df_meta.to_csv(meta_path, sep=',')
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'wb') as f:
        pickle.dump(cell_positions, f)



def save_single_cell_im(output_mat,
                        masked_output_mat,
                        tm,
                        tm2,
                        im_path):
    """ Plot single cell patch (unmasked, masked, segmentation mask)
    """
    # tm = tm[0, 0]
    # tm2 = tm2[0, 0]
    im_phase = output_mat[0]
    im_phase_masked = masked_output_mat[0]
    # replace zero-padding with min for display
    im_phase[im_phase == 0] = np.nanmin(im_phase[im_phase != 0])
    im_phase_masked[im_phase_masked == 0] = np.nanmin(im_phase_masked[im_phase_masked != 0])
    im_phase = im_adjust(im_phase)
    im_phase_masked = im_adjust(im_phase_masked)
    n_rows = 2
    n_cols = 2
    fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
    ax = ax.flatten()
    fig.set_size_inches((15, 5 * n_rows))
    axis_count = 0
    for im, name in zip([im_phase, im_phase_masked, tm, tm2],
                        ['output_mat', 'masked_output_mat', 'tm', 'tm2']):
        ax[axis_count].imshow(np.squeeze(im), cmap='gray')
        ax[axis_count].axis('off')
        ax[axis_count].set_title(name, fontsize=12)
        axis_count += 1
    fig.savefig(im_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_site_extract_patches_align_axis(site_path,
                                            site_segmentation_path, 
                                            site_supp_files_folder,
                                            window_size=256,
                                            save_fig=False,
                                            skip_boundary=False,
                                            **kwargs):
    """ Wrapper method for long-axis-aligned patch extraction

    Supplementary files generated by `process_site_instance_segmentation` will
    be loaded for each site, then individual cells from static frames will be
    extracted and saved. An extra step of aligning cell long axis to x-axis is
    performed.

    Results will be saved in supplementary data folder, including:
        "stacks_rotated_*.pkl": long-axis-aligned single cell patches for 
            each time slice

    Args:
        site_path (str): path to image stack (.npy)
        site_segmentation_path (str): path to semantic segmentation stack (.npy)
        site_supp_files_folder (str): path to the folder where supplementary 
            files will be saved
        window_size (int, optional): default=256, x, y size of the patch
        save_fig (bool, optional): if to save extracted patches (with
            segmentation mask)
        skip_boundary (bool, optional): if to skip patches whose edges exceed
            the image size (do not pad)

    """

    # Use a larger window (for rotation)
    output_window_size = window_size
    window_size = int(np.ceil(window_size * np.sqrt(2)) + 1)
    # Load data
    image_stack = np.load(site_path)
    segmentation_stack = np.load(site_segmentation_path)
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'rb') as f:
        cell_positions = pickle.load(f)
    with open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'rb') as f:
        cell_pixel_assignments = pickle.load(f)

    n_frames, n_channels, n_z, x_full_size, y_full_size = image_stack.shape
    for t_point in range(n_frames):
        site_data = {}
        print("\tWriting time %d" % t_point)
        raw_image = image_stack[t_point]
        cell_segmentation = segmentation_stack[t_point]
        positions, positions_labels = cell_pixel_assignments[t_point]
        all_cells = cell_positions[t_point]

        # Define fillings for the masked pixels in this slice
        background_positions = np.where(cell_segmentation[0] > 0.9)
        background_pool = np.array([np.median(raw_image[i][background_positions]) for i in range(n_channels)])
        background_filling = np.ones((n_channels, n_z, window_size, window_size)) * background_pool.reshape((n_channels, 1, 1, 1))

        # Save all cells in this step, filtering will be performed during analysis
        for cell_id, cell_position in all_cells:
            cell_name = os.path.join(site_supp_files_folder, '%d_%d.h5' % (t_point, cell_id))
            # Define window based on cell center and extract mask
            window = [(cell_position[0]-window_size//2, cell_position[0]+window_size//2),
                      (cell_position[1]-window_size//2, cell_position[1]+window_size//2)]
            window_segmentation = select_window(cell_segmentation,
                                                window,
                                                padding=-1,
                                                skip_boundary=skip_boundary)
            if window_segmentation is None:
                continue
            remove_mask, tm, tm2 = generate_mask(positions, 
                                                 positions_labels, 
                                                 cell_id, 
                                                 window, 
                                                 window_segmentation)

            # Select submatrix from the whole slice
            remove_mask = np.expand_dims(np.stack([remove_mask] * n_z, 0), 0)
            output_mat = select_window(raw_image, window, padding=0)
            assert not output_mat is None
            masked_output_mat = output_mat * (1 - remove_mask) + background_filling * remove_mask

            ang = get_cell_rect_angle(tm)
            M = cv2.getRotationMatrix2D((window_size/2, window_size/2), ang, 1)
            _tm = cv2.warpAffine(tm.astype('uint8'), M, (window_size, window_size)).reshape((window_size, window_size))
            _tm2 = cv2.warpAffine(tm2.astype('uint8'), M, (window_size, window_size)).reshape((window_size, window_size))
            _output_mat = cv2_fn_wrapper(cv2.warpAffine,
                                         output_mat.astype('uint16'),
                                         M,
                                         (window_size, window_size))
            _masked_output_mat = cv2_fn_wrapper(cv2.warpAffine,
                                                masked_output_mat.astype('uint16'),
                                                M,
                                                (window_size, window_size))
            # Reshape (x, y) to (c, z, x, y)
            _tm = np.expand_dims(np.stack([_tm] * n_z, 0), 0)
            _tm2 = np.expand_dims(np.stack([_tm2] * n_z, 0), 0)

            tm = _tm[...,
                (window_size//2 - output_window_size//2):(window_size//2 + output_window_size//2),
                (window_size//2 - output_window_size//2):(window_size//2 + output_window_size//2)]
            tm2 = _tm2[...,
                (window_size//2 - output_window_size//2):(window_size//2 + output_window_size//2),
                (window_size//2 - output_window_size//2):(window_size//2 + output_window_size//2)]
            output_mat = _output_mat[...,
                (window_size//2 - output_window_size//2):(window_size//2 + output_window_size//2),
                (window_size//2 - output_window_size//2):(window_size//2 + output_window_size//2)]
            masked_output_mat = _masked_output_mat[...,
                (window_size//2 - output_window_size//2):(window_size//2 + output_window_size//2),
                (window_size//2 - output_window_size//2):(window_size//2 + output_window_size//2)]

            site_data[cell_name] = {
                "mat": np.concatenate([output_mat, tm, tm2], 0).astype('float64'),
                "masked_mat": np.concatenate([masked_output_mat, tm, tm2], 0).astype('float64')
                }

            if save_fig:
                im_path = os.path.join(site_supp_files_folder, 'patch_rotated_t%d_id%d.jpg' % (t_point, cell_id))
                save_single_cell_im(output_mat, masked_output_mat, tm, tm2, im_path)

        with open(os.path.join(site_supp_files_folder, 'stacks_rotated_%d.pkl' % t_point), 'wb') as f:
            pickle.dump(site_data, f)
