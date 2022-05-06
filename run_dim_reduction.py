import os
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import argparse
import matplotlib

from plot_scripts.plotting import plot_umap, zoom_axis

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import umap


def fit_PCA(train_data, weights_dir, labels, conditions):
    """ Fit a PCA model accounting for top 50% variance to the train_data,
    and outout

    Args:
        train_data (np.array): 2D array of training data (samples, features),
            should be directly extracted from VAE latent space
        weights_dir (str): output directory for fit pca model
        labels (np array): 1D array of sample class indices.

    Returns:
        pca (sklearn PCA model): trained PCA instance

    """
    model_path = os.path.join(weights_dir, 'pca_model.pkl')
    pca = PCA(0.5, svd_solver='auto')
    print('Fitting PCA model {} ...'.format(model_path))
    pcas = pca.fit_transform(train_data)
    print('Saving PCA model {}...'.format(model_path))
    with open(model_path, 'wb') as f:
        pickle.dump(pca, f, protocol=4)

    plt.clf()
    zoom_cutoff = 1
    # conditions = ['mock', 'infected']
    fig, ax = plt.subplots()
    scatter = ax.scatter(pcas[:, 0], pcas[:, 1], s=7, c=labels, cmap='Paired', alpha=0.1)
    scatter.set_facecolor("none")
    zoom_axis(pcas[:, 0], pcas[:, 1], ax, zoom_cutoff=zoom_cutoff)
    legend1 = ax.legend(handles=scatter.legend_elements()[0],
                        loc="upper right", title="condition", labels=conditions)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.savefig(os.path.join(weights_dir, 'PCA.png'), dpi=300)
    return pca

def process_PCA(input_dir, output_dir, weights_dir, prefix, suffix='_after'):
    """
    This function loads latent vectors generated by VQ-VAE and applies trained
    PCA to extract top PCs as morphology descriptors.

    Resulting morphology descriptors will be saved in the summary folder,
    including:
        "*_latent_space_PCAed.pkl": array of top PCs of latent vectors (before
            quantization)
        "*_latent_space_after_PCAed.pkl": array of top PCs of latent vectors
            (after quantization)

    Args:
        input_dir (str): folder for latent vectors
        output_dir (str): folder to save output files
        weights_dir (str): folder for PCA model
        prefix (str): prefix for input file name
        suffix (str): suffix for input file name

    """
    model_path = os.path.join(weights_dir, 'pca_model.pkl')
    try:
        pca = pickle.load(open(model_path, 'rb'))
    except Exception as ex:
        print(ex)
        raise ValueError("Error in loading pre-saved PCA weights")

    input_fname = '{}_latent_space{}.pkl'.format(prefix, suffix)
    output_fname = '{}_latent_space{}_PCAed.pkl'.format(prefix, suffix)
    dats = pickle.load(open(os.path.join(input_dir, input_fname), 'rb'))
    dats_ = pca.transform(dats)
    output_file = os.path.join(output_dir, output_fname)
    print(f"\tsaving {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(dats_, f, protocol=4)

def umap_transform(input_dir, output_dir, weights_dir, prefix, suffix='_after'):
    """Apply trained UMAP model to latent vectors and save the reduced vectors:
        "*_latent_space_{umap model name}.pkl": reduced latent vectors (before
            quantization)
        "*_latent_space_after_{umap model name}.pkl": reduced latent vectors
            (after quantization)

    Args:
        input_dir (str): folder for latent vectors
        output_dir (str): folder to save output files
        weights_dir (str): folder for UMAP model
        prefix (str): prefix for input file name
        suffix (str): suffix for input file name
    """
    model_fnames = [file for file in os.listdir(weights_dir) if file.startswith('umap') & file.endswith('.pkl')]
    model_names = [os.path.splitext(name)[0] for name in model_fnames]
    for model_name, model_fname in zip(model_names, model_fnames):
        print('Transforming using model {}'.format(model_name))
        model_path = os.path.join(weights_dir, model_fname)
        try:
            umap = pickle.load(open(model_path, 'rb'))
        except Exception as ex:
            print(ex)
            raise ValueError("Error in loading pre-saved PCA weights")

        input_fname = '{}_latent_space{}.pkl'.format(prefix, suffix)
        output_fname = '{}_latent_space{}_{}.pkl'.format(prefix, suffix, model_name)

        dats = pickle.load(open(os.path.join(input_dir, input_fname), 'rb'))
        dats_ = umap.transform(dats)
        output_file = os.path.join(output_dir, output_fname)
        print(f"\tsaving {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(dats_, f, protocol=4)


def fit_umap(train_data, embed_dir, labels, label_col, fraction=0.1, seed=0,
             n_nbrs=(15,), a_s=(1.58,), b_s=(0.9,), dist_metric='euclidean', n_runs=1):
    """Fit UMAP model to latent vectors and save the reduced vectors (embeddings), output UMAP plot
    Args:
        train_data (np.array): 2D array of training data (samples, features),
            should be directly extracted from VAE latent space
        embed_dir (str): output directory for the fit umap model
        labels (np array): 1D array of sample class indices.
        n_nbrs (float) (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.
        a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
        b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.

    """
    #TODO: Find a way to save umap models gernerated with version >= 0.5


    n_plots = len(n_nbrs) * len(a_s) * len(b_s) * n_runs
    # n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_cols = min(n_plots, 3)
    n_rows = np.ceil(n_plots / n_cols).astype(np.int32)
    fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
    ax = ax.flatten()
    fig.set_size_inches((6.5 * n_cols, 5 * n_rows))


    # colors = cmap[labels]
    if fraction != 1:
        np.random.seed(seed)
        sample_ids = np.random.choice(len(train_data), int(fraction * len(train_data)), replace=False)
    else:
        sample_ids = np.arange(len(train_data))
    axis_count = 0
    for n_nbr in n_nbrs:
        for a, b in zip(a_s, b_s):
            for run in range(n_runs):
                print('Fitting UMAP model {} with N(neighbors)={}, a={}, b={}, run {} ...'.format(embed_dir, n_nbr, a, b, run))
                reducer = umap.UMAP(a=a, b=b, n_neighbors=n_nbr, metric=dist_metric)
                embedding = reducer.fit_transform(train_data)
                print('Saving UMAP model {}...'.format(embed_dir))
                with open(os.path.join(embed_dir, 'umap_nbr{}_{}_run{}.npy'.format(n_nbr, dist_metric, run)), 'wb') as f:
                    np.save(f, embedding)
                embedding_sub = embedding[sample_ids, :]
                labels_sub = labels[sample_ids]
                print(labels_sub.shape)
                title = 'n_neighbors={}'.format(n_nbr)
                if axis_count == (len(ax) - 1):
                    plot_umap(ax[axis_count], embedding_sub, labels_sub, title=title, leg_title=label_col, zoom_cutoff=0, plot_other=False)
                else:
                    plot_umap(ax[axis_count], embedding_sub, labels_sub, title=title, zoom_cutoff=0, plot_other=False)
                axis_count += 1
                fig.savefig(os.path.join(embed_dir, 'UMAP_{}_frac{}_{}_{}runs.png'.format('_'.join(label_col).replace(' ', '_'), fraction, dist_metric, n_runs)),
                            dpi=300, bbox_inches='tight')
    plt.close(fig)

def dim_reduction(input_dirs,
                  output_dirs,
                  weights_dir,
                  method,
                  fit_model,
                  label_col=None,
                  split='test',
                  fraction=0.1):
    """
    Wrapper fucntion for dimensionality reduction, save the reduced vectors (embeddings),
    output 2D embedding plot.
    supports PCA and UMAP for fitting new models and save reduced vectors.
    Transform from saved model is only suppored for PCA
    Args:
        input_dir (str): folder for latent vectors
        output_dir (str): folder to save output files
        weights_dir (str): folder for loading or saving model
        method (str): reduction method ('pca', 'umap')
        fit_model (bool): Fit new model if Ture, load previous trained model otherwise
        prefix (str): prefix for input file name

    """

    if type(input_dirs) is not list:
        input_dirs = [input_dirs]
    if not output_dirs:
        output_dirs = input_dirs
    assert len(output_dirs) == len(input_dirs), 'Numbers of input and output directories must have match.'
    fname = '{}_embeddings.npy'.format(split)
    # fname = 'static_patches'
    if method == 'pca':
        fit_func = fit_PCA
        transform_func = process_PCA
    elif method == 'umap':
        fit_func = fit_umap
        transform_func = umap_transform
        if not fit_model:
            raise NotImplemented('Inference mode is only supported for PCA at the moment')
    else:
        raise ValueError('Dimensionality reduction method has to be "pca" or "umap"')
    # if conditions is None:
    #     conditions = [os.path.basename(input_dir) for input_dir in input_dirs]
    # elif type(conditions) is not list:
    #     conditions = [conditions]
    # assert len(conditions) == len(input_dirs), '# of conditions has to be equal to # of input directories'
    if fit_model:
        vector_list = []
        df_meta_all = []
        for input_dir in input_dirs:
            # only update label if current condition is different from previous
            df_meta = pd.read_csv(os.path.join(os.path.dirname(input_dir), 'patch_meta_{}.csv'.format(split)), index_col=0, converters={
                'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
            df_meta_all.append(df_meta)
            df_meta_all = pd.concat(df_meta_all, axis=0)
            vec = np.load(os.path.join(input_dir, fname))
            vector_list.append(vec.reshape(vec.shape[0], -1))

        vectors = np.concatenate(vector_list, axis=0)
        if len(label_col) == 1:
            labels = df_meta_all[label_col[0]].to_numpy()
        else:
            labels = df_meta_all[label_col].apply(lambda row: '_'.join(row.values.astype(str)), axis=1).to_numpy()
        print(labels.shape)
        _ = fit_func(vectors, input_dir, labels=labels, label_col=label_col, fraction=fraction)
        # UMAP model from umap 0.5.0 can't be pickled with protocol=4.
        # Transform from saved models is currently not supported
        if method == 'umap':
            return
    # run inference for PCA
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        print('Transforming latent vectors for {}'.format(input_dir))
        transform_func(input_dir=input_dir,
                    output_dir=output_dir,
                    weights_dir=weights_dir,
                    )


def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input',
        nargs='+',
        type=str,
        required=True,
        help='Input directory/directories containing latent vectors to reduce dimension. '
             'If multiple directories are supplied, all inputs will be concatenated for model fitting',
    )
    parser.add_argument(
        '-o', '--output',
        nargs='+',
        default=[],
        type=str,
        required=False,
        help='Output directory to save the reduced vectors. Same as input if not specified',
    )
    parser.add_argument(
        '-m', '--method',
        type=str,
        required=False,
        choices=['pca', 'umap'],
        default='umap',
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        '-w', '--weights',
        type=str,
        required=True,
        help="Directory to load/save the PCA weights.",
    )
    parser.add_argument(
        '-f', '--fit',
        dest='fit_model',
        action='store_true',
        help="Fit pca or umap model to the data and save the weights to 'weights'. "
             "If left out the script will run with inference model and load saved model in 'weights'"
             "to transform the vectors. Inference mode is only supported for PCA at the moment"
    )
    parser.set_defaults(fit_model=False)
    parser.add_argument(
        '-p', '--prefix',
        type=str,
        required=False,
        default=None,
        help='prefix of the latent vector filename "{}_latent_space"',
    )

    parser.add_argument(
        '-c', '--condition',
        nargs='+',
        type=str,
        required=False,
        default=None,
        help='Condition for each input directory',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dim_reduction(input_dirs=args.input,
                  output_dirs=args.output,
                  weights_dir=args.weights,
                  method=args.method,
                  fit_model=args.fit_model,
                  prefix=args.prefix,
                  conditions=args.condition,
                  )


