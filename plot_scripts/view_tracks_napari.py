import argparse
import os
import numpy as np
import pandas as pd
from utils.config_reader import YamlReader
import napari
from napari_animation import Animation

def tracks_2d(df_meta_fov):
    """ convert dynacontrast metadata to napari track format """
    tracks = pd.DataFrame()
    df_meta_fov.sort_values(by=['time trajectory ID', 'time'], ascending=[True, True], inplace=True)
    track_ids = df_meta_fov['time trajectory ID'].dropna().unique()
    for track_id in track_ids:
        track = df_meta_fov.loc[df_meta_fov['time trajectory ID'] == track_id,
                                ['time trajectory ID', 'time', 'slice']]
        track[['y', 'x']] = pd.DataFrame(df_meta_fov.loc[df_meta_fov['time trajectory ID'] == track_id,
                                'cell position'].tolist(), index=track.index)
        # # calculate the speed as a property
        track['vz'] = 0
        track['vy'] = np.gradient(track['y'].to_numpy())
        track['vx'] = np.gradient(track['x'].to_numpy())
        #
        track['speed'] = np.sqrt(track['vx'] ** 2 + track['vy'] ** 2 + track['vx'] ** 2)
        track['distance'] = np.sqrt(track['y'] ** 2 + track['x'] ** 2)
        tracks = tracks.append(track)
    # tracks = np.concatenate(tracks, axis=0)
    tracks = tracks.to_numpy()
    data = tracks[:, :5]  # just the coordinate data
    properties = {
        'time': tracks[:, 1],
        'gradient_z': tracks[:, 5],
        'gradient_y': tracks[:, 6],
        'gradient_x': tracks[:, 7],
        'speed': tracks[:, 8],
        'distance': tracks[:, 9],
    }

    graph = {}
    return data, properties, graph

def view_tracks(config, raw_dir, supp_dir, fovs, slice_ids):
    # load tracks for each FOV into each napari instance
    meta_path = os.path.join(supp_dir, 'im-supps', 'patch_meta.csv')
    df_meta = pd.read_csv(meta_path, index_col=0, converters={
        'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
    output_dir = os.path.join(supp_dir, 'movies', 'tracks')
    os.makedirs(output_dir, exist_ok=True)
    tol = 1
    for fov in fovs:
        df_meta_fov = df_meta[(df_meta['FOV'] == fov) & (df_meta['slice'].isin(slice_ids))]
        tracks, properties, graph = tracks_2d(df_meta_fov)
        tracks[:, 2] = 0 # set z=0
        vertices = tracks[:, 1:]
        viewer = napari.view_points(vertices, size=1, name='points', opacity=0.3)
        img = np.load(os.path.join(raw_dir, fov + '.npy'))
        viewer.add_tracks(tracks, properties=properties, name='tracks', colormap='turbo', tail_length=len(img))
        limit = [-0.06, 0.09]
        viewer.add_image(img[:, 0:1, 2:3,...], channel_axis=1, colormap='gray', gamma=2, contrast_limits=limit)
        seg_map = np.load(os.path.join(raw_dir, fov + '_NNProbabilities.npy'))
        limit = [0, np.max(seg_map)]
        viewer.add_image(seg_map[:, 0:1, 2:3,...], channel_axis=1, colormap='gist_earth', contrast_limits=limit)
        # animation_widget = AnimationWidget(viewer)
        # viewer.window.add_dock_widget(animation_widget, area='right')
        animation = Animation(viewer)
        viewer.dims.set_current_step(0, 0)
        animation.capture_keyframe()
        viewer.dims.set_current_step(0, len(img) - 1)
        animation.capture_keyframe(steps=len(img) - 1)
        animation.animate(os.path.join(output_dir, 'track_{}.mov'.format(fov)),
                          canvas_only=True,
                          fps=4,
                          quality=9,
                          scale_factor=2)


        napari.run()

def main():
    arguments = parse_args()
    config = YamlReader()
    config.read_config(arguments.config)
    for (raw_dir, supp_dir) in zip(config.plotting.raw_dirs, config.plotting.supp_dirs):
        view_tracks(config, raw_dir, supp_dir, config.plotting.fovs, config.plotting.slice_ids)


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