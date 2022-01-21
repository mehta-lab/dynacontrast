import argparse
import os
import numpy as np
import pandas as pd
import dask.array as da
import imageio
from utils.config_reader import YamlReader
from SingleCellPatch.patch_utils import im_adjust

def save_single_cell_movie(config, raw_dir, supp_dir):
    z_ids = config.plotting.slice_ids
    min_length = config.plotting.min_length
    meta_path = os.path.join(supp_dir, 'im-supps', 'patch_meta.csv')
    df_meta = pd.read_csv(meta_path, index_col=0, converters={
                'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
    data_path = os.path.join(raw_dir, 'cell_patches.zarr')
    dataset = da.from_zarr(data_path)
    output_dir = os.path.join(supp_dir, 'movies')
    os.makedirs(output_dir, exist_ok=True)
    for z_idx in z_ids:
        df_meta_z = df_meta[df_meta['slice'] == z_idx]
        traj_ids = df_meta_z['time trajectory ID'].drop_duplicates()
        for traj_idx in traj_ids:
            df_meta_z_trj = df_meta_z[df_meta_z['time trajectory ID'] == traj_idx]
            if len(df_meta_z_trj) >= min_length:
                traj_tstack = dataset[df_meta_z_trj.index.tolist(), 0, ...]
                traj_tstack = im_adjust(traj_tstack)
                output_path = os.path.join(output_dir, 'cell_traj_{}_movie.gif'.format(traj_idx))
                imageio.mimsave(output_path, traj_tstack)

def main():
    arguments = parse_args()
    config = YamlReader()
    config.read_config(arguments.config)
    for (raw_dir, supp_dir) in zip(config.plotting.raw_dirs, config.plotting.supp_dirs):
        save_single_cell_movie(config, raw_dir, supp_dir)


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
    main()





