import numpy as np
import os
import logging
import matplotlib
import pandas as pd

from plot.plotting import save_single_cell_im
from utils.config_reader import YamlReader

matplotlib.use('AGG')
from utils.patch_utils import check_segmentation_dim, select_window

log = logging.getLogger('dynacontrast.log')

def get_patches_mp(raw_folder: str,
                   supp_folder: str,
                   sites: list,
                   config: YamlReader,
                   **kwargs):
    """ Helper function for patch extraction

    Wrapper method `get_patches` will be called, which
    extracts individual cells from static frames for each site.

    Results will be saved in the supplementary data folder, including:
        "stacks_*.pkl": single cell patches for each time slice

    Args:
        raw_folder (str): folder for raw data, segmentation and
            summarized results
        supp_folder (str): folder for supplementary data
        sites (list of str): list of site names
        config (YamlReader): config file supplied at CLI
    """
    channels = config.preprocess.channels

    assert len(channels) > 0, "At least one channel must be specified"

    crop_size = config.preprocess.crop_size
    save_fig = config.preprocess.save_fig
    skip_boundary = config.preprocess.skip_boundary

    for site in sites:
        site_path = os.path.join(raw_folder + '/' + site + '.npy')
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path):
            print("Site data not found %s" % site_path, flush=True)
        if not os.path.exists(site_supp_files_folder):
            print("Site supp folder not found %s" % site_supp_files_folder, flush=True)
        meta_path = os.path.join(site_supp_files_folder, 'patch_meta.csv')
        if not os.path.isfile(meta_path): # skip position with no cells
            log.warning('No patch_meta.csv is found in {}. Skipping...'.format(site_supp_files_folder))
            # print('No cell is detected for position {}'.format(site))
            continue

        try:
            get_patches(site_path,
                        site_supp_files_folder,
                        crop_size=crop_size,
                        channels=channels,
                        save_fig=save_fig,
                        skip_boundary=skip_boundary,
                        **kwargs)
        except Exception as e:
            log.error('Extracting patches failed for {}. '.format(site_supp_files_folder))
            log.exception('')
            # print('Extracting patches failed for position {}. '.format(site))
            # raise e
    return

def get_patches(site_path,
                site_supp_files_folder,
                crop_size=256,
                channels=None,
                save_fig=False,
                skip_boundary=False,
                ):
    """ Wrapper method for patch extraction

    Supplementary files generated by `find_cells` will
    be loaded for each site, then individual cells from static frames will be
    extracted and saved.

    Results will be saved in supplementary data folder, including:
        "stacks_*.pkl": single cell patches for each time slice

    Args:
        site_path (str): path to image stack (.npy)
        site_segmentation_path (str): path to semantic segmentation stack (.npy)
        site_supp_files_folder (str): path to the folder where supplementary 
            files will be saved
        crop_size (int, optional): default=256, x, y size of the patch
        channels (list, optional): channels to extract patches. Default is all the channels
        save_fig (bool, optional): if to save extracted patches (with
            segmentation mask)
        reload (bool, optional): if to load existing stack dat files
        skip_boundary (bool, optional): if to skip patches whose edges exceed
            the image size (do not pad)

    """

    # Load data
    image_stack = np.load(site_path)
    meta_path = os.path.join(site_supp_files_folder, 'patch_meta.csv')
    df_meta = pd.read_csv(meta_path, index_col=0, converters={
        'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
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
            # print("processing timepoint {} z {}".format(t_point, z))
            stack_dat_path = os.path.join(site_supp_files_folder, 'patches_t{}_z{}.npy'.format(t_point, z))
            # print('Writing timepoint {} z {}'.format(t_point, z))
            raw_image = image_stack[t_point, :, z, ...]
            df_meta_tz = df_meta.loc[(df_meta['time'] == t_point) & (df_meta['slice'] == z), :]
            all_cells = df_meta_tz.loc[:, ['cell ID', 'cell position']].to_numpy()
            # Save all cells in this step, filtering will be performed during analysis
            cell_patches = []
            for cell_id, cell_position in all_cells:
                # print("cell_id : {}, cell position: {}".format(cell_id, cell_position))
                # Define window based on cell center and extract mask
                window = [(cell_position[0]- crop_size//2, cell_position[0]+ crop_size//2),
                          (cell_position[1]- crop_size//2, cell_position[1]+ crop_size//2)]
                cell_patch = select_window(raw_image, window, padding=0, skip_boundary=skip_boundary)
                if cell_patch is None:
                    # drop cell that did not get patched
                    df_meta.drop(df_meta[(df_meta['cell ID'] == cell_id) &
                                         (df_meta['time'] == t_point) &
                                         (df_meta['slice'] == z)].index, inplace=True)
                else:
                    cell_patches.append(cell_patch)
                    if save_fig:
                        im_path = os.path.join(site_supp_files_folder, 'patch_t{}_z{}_cell{}'.format(t_point, z, cell_id))
                        save_single_cell_im(cell_patch, im_path)
            if cell_patches:
                cell_patches = np.stack(cell_patches)
                with open(stack_dat_path, 'wb') as f:
                    # print(f"save patches to {stack_dat_path}")
                    np.save(f, cell_patches)
    df_meta.reset_index(drop=True, inplace=True)
    df_meta.to_csv(meta_path, sep=',')


