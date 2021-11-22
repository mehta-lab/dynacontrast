import os
import numpy as np
import pickle
import matplotlib
import torch as t
from matplotlib import pyplot as plt

from SingleCellPatch.patch_utils import im_adjust

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import itertools

def zoom_axis(x, y, ax, zoom_cutoff=1):
    xlim = [np.percentile(x, zoom_cutoff), np.percentile(x, 100 - zoom_cutoff)]
    ylim = [np.percentile(y, zoom_cutoff), np.percentile(y, 100 - zoom_cutoff)]
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])

def mp_sample_im_pixels(fn_args, workers):
    """Read and computes statistics of images with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from get_im_stats
    """

    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(sample_im_pixels, *zip(*fn_args))
    return list(res)

def grid_sample_pixel_values(im, grid_spacing):
    """Sample pixel values in the input image at the grid. Any incomplete
    grids (remainders of modulus operation) will be ignored.

    :param np.array im: 2D image
    :param int grid_spacing: spacing of the grid
    :return int row_ids: row indices of the grids
    :return int col_ids: column indices of the grids
    :return np.array sample_values: sampled pixel values
    """

    im_shape = im.shape
    assert grid_spacing < im_shape[0], "grid spacing larger than image height"
    assert grid_spacing < im_shape[1], "grid spacing larger than image width"
    # leave out the grid points on the edges
    sample_coords = np.array(list(itertools.product(
        np.arange(grid_spacing, im_shape[0], grid_spacing),
        np.arange(grid_spacing, im_shape[1], grid_spacing))))
    row_ids = sample_coords[:, 0]
    col_ids = sample_coords[:, 1]
    sample_values = im[row_ids, col_ids, :]
    return row_ids, col_ids, sample_values

def sample_im_pixels(im, grid_spacing, meta_row):
    """Read and computes statistics of images

    """

    # im = image_utils.read_image(im_path)
    row_ids, col_ids, sample_values = \
        grid_sample_pixel_values(im, grid_spacing)

    meta_rows = \
        [{**meta_row,
          'row_idx': row_idx,
          'col_idx': col_idx,
          'intensity': sample_value}
          for row_idx, col_idx, sample_value
          in zip(row_ids, col_ids, sample_values)]
    return meta_rows

def ints_meta_generator(
        input_dir,
        order='cztp',
        num_workers=4,
        block_size=256,
        ):
    """
    Generate pixel intensity metadata for estimating image normalization
    parameters during preprocessing step. Pixels are sub-sampled from the image
    following a grid pattern defined by block_size to for efficient estimation of
    median and interquatile range. Grid sampling is preferred over random sampling
    in the case due to the spatial correlation in images.
    Will write found data in ints_meta.csv in input directory.
    Assumed default file naming convention is:
    dir_name
    |
    |- im_c***_z***_t***_p***.png
    |- im_c***_z***_t***_p***.png

    c is channel
    z is slice in stack (z)
    t is time
    p is position (FOV)

    Other naming convention is:
    img_channelname_t***_p***_z***.tif for parse_sms_name

    :param list args:    parsed args containing
        str input_dir:   path to input directory containing images
        str name_parser: Function in aux_utils for parsing indices from file name
        int num_workers: number of workers for multiprocessing
        int block_size: block size for the grid sampling pattern. Default value works
        well for 2048 X 2048 images.
    """
    mp_block_args = []

    # Fill dataframe with rows from image names
    for i in range(len(im_names)):
        kwargs = {"im_name": im_names[i]}
        if name_parser == 'parse_idx_from_name':
            kwargs["order"] = order
        elif name_parser == 'parse_sms_name':
            kwargs["channel_names"] = channel_names
        meta_row = parse_func(**kwargs)
        meta_row['dir_name'] = input_dir
        im_path = os.path.join(input_dir, im_names[i])
        mp_fn_args.append(im_path)
        mp_block_args.append((im_path, block_size, meta_row))

    im_ints_list = mp_sample_im_pixels(mp_block_args, num_workers)
    im_ints_list = list(itertools.chain.from_iterable(im_ints_list))
    ints_meta = pd.DataFrame.from_dict(im_ints_list)

    ints_meta_filename = os.path.join(input_dir, 'ints_meta.csv')
    ints_meta.to_csv(ints_meta_filename, sep=",")
    return ints_meta

def distribution_plot(frames_metadata,
                      y_col,
                      output_path,
                      output_fname):
    my_palette = {'F-actin': 'g', 'nuclei': 'm'}

    fig = plt.figure()
    fig.set_size_inches((18, 9))
    ax = sns.violinplot(x='channel_name', y=y_col,
                        hue='dir_name',
                        bw=0.01,
                        data=frames_metadata, scale='area',
                        linewidth=1, inner='quartile',
                        split=False)
    # ax.set_xticklabels(labels=['retardance',
    #                            'BF',
    #                            'retardance+slow axis+BF'])
    plt.xticks(rotation=25)
    # plt.title(''.join([metric_name, '_']))
    # ax.set_ylim(bottom=0.5, top=1)
    # ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    ax.legend(loc="upper left", borderaxespad=0.1)
    # ax.get_legend().remove()
    ax.set_ylabel('Mean intensity')
    plt.savefig(os.path.join(output_path, ''.join([output_fname, '.png'])),
                dpi=300, bbox_inches='tight')


def save_recon_images(val_dataloader, model, model_dir):
    # %% display recon images
    os.makedirs(model_dir, exist_ok=True)
    batch = next(iter(val_dataloader))
    labels, data = batch
    labels = t.cat([label for label in labels], axis=0)
    data = t.cat([datum for datum in data], axis=0)
    output = model(data.to(device), labels.to(device))[0]
    for i in range(10):
        im_phase = im_adjust(data[i, 0].data.numpy())
        im_phase_recon = im_adjust(output[i, 0].cpu().data.numpy())
        im_retard = im_adjust(data[i, 1].data.numpy())
        im_retard_recon = im_adjust(output[i, 1].cpu().data.numpy())
        n_rows = 2
        n_cols = 2
        fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
        ax = ax.flatten()
        fig.set_size_inches((15, 5 * n_rows))
        axis_count = 0
        for im, name in zip([im_phase, im_phase_recon, im_retard, im_retard_recon],
                            ['phase', 'phase_recon', 'im_retard', 'retard_recon']):
            ax[axis_count].imshow(np.squeeze(im), cmap='gray')
            ax[axis_count].axis('off')
            ax[axis_count].set_title(name, fontsize=12)
            axis_count += 1
        fig.savefig(os.path.join(model_dir, 'recon_%d.jpg' % i),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)