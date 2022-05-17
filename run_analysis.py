import os
import argparse
import anndata as ad
import glob
import numpy as np
# import torch
# import torch.nn as nn
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from matplotlib import pyplot as plt
from plot_scripts.plotting import plot_umap
from plot_scripts.clustering_workflows import ClusteringWorkflow
from SingleCellPatch.patch_utils import im_adjust

# from torch.utils.data import DataLoader
# from dataset.dataset import TripletDataset
# from run_training import train_with_loader
# from HiddenStateExtractor.resnet import LogisticRegression as LogisticRegressionNN
def load_meta(dataset_dirs, splits=('train', 'val', 'test')):
    df_meta_all = {}
    for split in splits:
        df_meta_all[split] = []
        for dataset_dir in dataset_dirs:
            df_meta = pd.read_csv(os.path.join(dataset_dir, 'patch_meta_{}.csv'.format(split)), index_col=0,
                                  converters={
                                      'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
            df_meta_all[split].append(df_meta)
        df_meta_all[split] = pd.concat(df_meta_all[split], axis=0)
    return df_meta_all

def model_fit(model, train_set, train_labels, val_set, val_labels):
    """Vanilla model training without cross-validiation
    :param object model:
    :param dataframe dtrain: training data with rows being samples and columns being features
    :param list features: column names of features
    :param str train_labels: column name of the target
    :return object model: fitted classifier object.
    :return float score: train_set AUC score.
    """
    model.fit(train_set, train_labels)
    train_pred = model.predict(train_set)
    train_score = metrics.accuracy_score(train_labels, train_pred)
    val_pred = model.predict(val_set)
    val_score = metrics.accuracy_score(val_labels, val_pred)
    return model, train_score, val_score, train_pred, val_pred

def calc_cluster_centrosize(data, cluster_arr, exclude='others'):
    """
    Generate cluster matrix convenient for cluster score computation.
    :param data: coordinate data
    :param cluster_arr: cluster array with the same order as data and labels; size = [:, 1]
    :param exclude: exclude labels from output matrix
    :return: matrix of cluster_name, centroid, cluster_size on each column
    """
    cluster_uniq = np.unique(cluster_arr)
    centroid_list = []
    clustersize_list = []
    for cl in cluster_uniq:
        ind = cluster_arr == cl
        data0 = data[ind.flatten()]
        centroid = np.median(data0, axis=0)
        distsq = (centroid - data0) ** 2  # square distance between each datapoint and centroid
        cluster_size = np.median(np.sqrt(distsq[:, 0] + distsq[:, 1]))  # median of sqrt of distsq as cluster size
        centroid_list.append(centroid)
        clustersize_list.append(cluster_size)
    output = np.vstack([cluster_uniq, np.vstack(centroid_list).T, clustersize_list]).T
    if isinstance(exclude, str):
        exclude = [exclude]
    ind = np.isin(output[:, 0], exclude)
    output = output[~ind]
    return output

def cluster_score(umap, labels):
    cluster_matrix = calc_cluster_centrosize(umap, labels)
    intra_med = np.median(cluster_matrix[:, -1])
    # intra_max = np.max(cluster_matrix[:, -1])
    inter = np.std(cluster_matrix[:, 1:3].astype(float), axis=0).mean()
    # cluster_cen = cluster_matrix[:, 1:3].astype(float)
    # centroid = np.median(cluster_cen, axis=0)
    # distsq = (centroid - cluster_cen) ** 2  # square distance between each datapoint and centroid
    # inter = np.median(np.sqrt(distsq[:, 0] + distsq[:, 1]))  # median of sqrt of distsq as cluster size
    # score_max = inter / intra_max
    score_med = inter / intra_med
    return score_med

def resplit_data(vectors, labels, df_meta_all, splits = ('train', 'test'), split_cols=None, val_split_ratio=0.15, seed=0):
    vectors_temp = [vectors[split] for split in vectors]
    vectors_temp = np.concatenate(vectors_temp, axis=0)
    labels_temp = [labels[split] for split in labels]
    labels_temp = np.concatenate(labels_temp, axis=0)
    df_meta_temp = [df_meta_all[split] for split in df_meta_all]
    df_meta_temp = pd.concat(df_meta_temp, axis=0)
    # print("df_meta_temp:", len(df_meta_temp))

    vectors_temp = vectors_temp[labels_temp != 'other']
    df_meta_temp = df_meta_temp[labels_temp != 'other']
    labels_temp = labels_temp[labels_temp != 'other']
    print("n(labels):", len(np.unique(labels_temp)))
    # print("df_meta_temp:", len(df_meta_temp))
    # print("labels_temp:", len(labels_temp))
    if split_cols is None:
        split_cols = ['data_dir', 'FOV']
    elif type(split_cols) is str:
        split_cols = [split_cols]
    split_key = df_meta_temp[split_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    sgs = StratifiedGroupKFold(n_splits=2)
    split_ids, _ = sgs.split(df_meta_temp, y=labels_temp, groups=split_key)
    print(max(split_ids[0]), max(split_ids[1]))
    vectors = {split: vectors_temp[ids] for split, ids in zip(splits, split_ids)}
    labels = {split: labels_temp[ids] for split, ids in zip(splits, split_ids)}
    df_meta_all = {split: df_meta_temp.iloc[ids] for split, ids in zip(splits, split_ids)}
    # print(set(df_meta_all['train'].loc[:, split_cols[0]].unique()).intersection(set(df_meta_all['test'].loc[:, split_cols[0]].unique())))
    # print(set(labels['train'])==set(labels['test']))
    return vectors, labels, df_meta_all

def plot_linear_eval(weights_dirs, nn=False):
    log_all_df = pd.DataFrame()
    train_dir = os.path.dirname(weights_dirs[0])
    print('plotting evaluation results...')
    for weights_dir in weights_dirs:
        model_name = os.path.basename(weights_dir)
        if nn:
            from plot_scripts import tflogs2df
            learn_rate = 0.1
            tflog_dir = os.path.join(weights_dir, 'evaluation_lr{}'.format(learn_rate))
            log_df = tflogs2df.main(
                logdir_or_logfile=tflog_dir,
                out_dir=tflog_dir,
                write_pkl=False,
                write_csv=False,
                return_df=True)
            log_df = log_df[log_df['metric'] == 'Val_loss/acc']
            log_df = log_df.iloc[-20:, :]
            log_df['model'] = model_name

            log_df['label'] = name_mapping[model_name]
            log_df['Top-1 accuracy'] = log_df['value']
            fig_name = 'model_comparision_zarr.png'
        else:
            log_df = pd.read_csv(os.path.join(weights_dir, 'linear_eval_{}.csv'.format(label_col.replace(' ', '_'))))
            log_df['model'] = model_name
            log_df['label'] = name_mapping[model_name]
            log_df['Top-1 accuracy'] = log_df['train_acc']
            log_df['split'] = 'train'

        if 'datasetnorm' in model_name:
            log_df['normalization'] = 'dataset'
        else:
            log_df['normalization'] = 'patch'
        if 'ntxent' in model_name:
            log_df['loss'] = 'ntxent'
        else:
            log_df['loss'] = 'triplet'
        if 'proj' in model_name:
            log_df['projection'] = 'after'
        else:
            log_df['projection'] = 'before'
        log_all_df = log_all_df.append(log_df)
        log_df['Top-1 accuracy'] = log_df['val_acc']
        log_df['split'] = 'test'
        log_all_df = log_all_df.append(log_df)
    log_all_df.to_csv(os.path.join(train_dir, 'evaluation_resplit_{}.csv'.format(label_col.replace(' ', '_'))), index=None)
    ax = sns.barplot(y='label', x='Top-1 accuracy', data=log_all_df, hue='split', errwidth=1, capsize=.4)
    # ax = sns.barplot(y='model', x='Top-1 accuracy', data=log_all_df, errwidth=1, capsize=.4)
    # g = sns.catplot(x='normalization', y='Top-1 accuracy',
    #                 hue='loss', col='projection',
    #                 data=log_all_df, kind="bar",
    #                 height=4, aspect=.7)
    # g.set(ylim=(0.90, 1))
    # ax.set(xlim=(0.5, 1))
    ax.set(xlim=(0.8, 1))
    # ax.set(xlim=(0, 1))
    fig_name = 'linear_eval_resplit_{}.png'.format(label_col.replace(' ', '_'))
    plt.savefig(os.path.join(train_dir, fig_name), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_mat(weights_dirs):
    for weights_dir in weights_dirs:
        print('plotting confusion matrices...')
        pred_df = pd.read_csv(
            os.path.join(weights_dir, 'val_prediction_{}.csv'.format(label_col.replace(' ', '_'))))
        fig, ax = plt.subplots(figsize=(10, 10))
        metrics.ConfusionMatrixDisplay.from_predictions(pred_df['y true'], pred_df['y pred'], cmap='YlOrRd',
                                                        xticks_rotation=90, normalize='true', ax=ax,
                                                        include_values=False,
                                                        colorbar=True)
        fig_name = 'confusion_mat_resplit_{}.png'.format(label_col.replace(' ', '_'))
        plt.savefig(os.path.join(weights_dir, fig_name), dpi=300, bbox_inches='tight')
        plt.close()

def plot_cluster_scores(weights_dirs):
    train_dir = os.path.dirname(weights_dirs[0])
    ##calculating clustering scores
    print('calculating clustering scores... ')
    split = 'test'
    n_nbr = 15
    dist_metric = 'cosine'
    # dist_metric = 'euclidean'
    log_df = {'model': [], 'label': [], 'clustering score': []}
    for weights_dir in weights_dirs:
        model_name = os.path.basename(weights_dir)
        umap_paths = glob.glob(os.path.join(weights_dir, 'umap_{}_nbr{}_{}_*.npy'.format(label_col.replace(' ', '_'), n_nbr, dist_metric)))
        # print(umap_paths)
        umaps = [np.load(path) for path in umap_paths]
        for umap in umaps:
            score_med = cluster_score(umap, labels=df_meta_all[split][label_col].to_numpy())
            log_df['model'].append(model_name)
            log_df['label'].append(name_mapping[model_name])
            log_df['clustering score'].append(score_med)
    log_df = pd.DataFrame(log_df)
    log_df.to_csv(os.path.join(train_dir, 'clustering_score_{}_{}.csv'.format(label_col.replace(' ', '_'), dist_metric)), index=None)
    ax = sns.barplot(y='label', x='clustering score', data=log_df, errwidth=1, capsize=.4)
    # ax.set(xlim=(0.9, 1))
    # ax.set(xlim=(0, 1))
    fig_name = 'clustering_score_{}_{}.png'.format(label_col.replace(' ', '_'), dist_metric)
    plt.savefig(os.path.join(train_dir, fig_name), dpi=300, bbox_inches='tight')
    plt.close()

def plot_complex_umap(partial_names, weights_dirs, dist_metric='cosine', n_nbr=15):
    regexp = '|'.join(partial_names)
    ###plot selected clusters
    print('plot clusters... ')
    split = 'test'

    df_meta_all = load_meta(dataset_dirs, splits=('train', 'val', 'test'))

    df_complex = df_meta_all['test'].loc[
                 df_meta_all['test']['protein-complex-level ground truth'].str.contains(regexp), :]
    org_names = df_complex['organelle-level ground truth'].unique()
    complex_names = df_complex['protein-complex-level ground truth'].unique()
    print(org_names)
    print(complex_names)
    org_names = [name for name in org_names if name != 'other']
    org_ids = df_meta_all['test']['organelle-level ground truth'].isin(org_names)
    complex_ids = df_meta_all['test']['protein-complex-level ground truth'].isin(complex_names)
    sub_ids = org_ids | complex_ids
    labels_sub = df_meta_all['test'].loc[:, 'gene']
    # labels_sub = df_meta_all['test'].loc[sub_ids, 'gene']
    labels_sub.loc[~complex_ids] = 'other'
    labels_sub = labels_sub[sub_ids]

    for weights_dir in weights_dirs:
        model_name = os.path.basename(weights_dir)
        umap_paths = glob.glob(os.path.join(weights_dir, 'umap_{}_nbr{}_{}_*.npy'.format(label_col.replace(' ', '_'), n_nbr, dist_metric)))
        # print(umap_paths)
        umaps = [np.load(path) for path in umap_paths]
        umaps = umaps[0:1]
        n_plots = len(umaps)
        n_cols = min(n_plots, 3)
        n_rows = np.ceil(n_plots / n_cols).astype(np.int32)
        fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
        ax = ax.flatten()
        fig.set_size_inches((6.5 * n_cols, 5 * n_rows))
        axis_count = 0
        for embeddings in umaps:
            if axis_count == (len(ax) - 1):
                plot_umap(ax[axis_count], embeddings[sub_ids], labels_sub, title=','.join(org_names), leg_title='protein', alpha=0.4, zoom_cutoff=0, plot_other=True)
            else:
                plot_umap(ax[axis_count], embeddings[sub_ids], labels_sub, alpha=0.6, zoom_cutoff=0, plot_other=True)
            axis_count += 1
        fig.savefig(os.path.join(weights_dir,
                                 'UMAP_{}_{}_{}runs.png'.format(complex_names[0].replace(' ', '_'),
                                                                dist_metric, n_plots)),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_organelle_umap(names, weights_dirs, label_col, dist_metric = 'cosine', n_nbr=15, split='test'):
    df_meta_all = load_meta(dataset_dirs, splits=tuple([split]))
    names = np.array(names)
    sub_ids = df_meta_all[split][label_col].isin(names)
    labels_sub = df_meta_all[split].loc[:, label_col]
    labels_sub.loc[~sub_ids] = 'other'

    for weights_dir in weights_dirs:
        model_name = os.path.basename(weights_dir)
        umap_paths = glob.glob(
            os.path.join(weights_dir, 'umap*nbr{}_{}_*.npy'.format(n_nbr, dist_metric)))
        # print(umap_paths)
        umaps = [np.load(path) for path in umap_paths]
        umaps = umaps[0:1]
        n_plots = len(umaps)
        n_cols = min(n_plots, 3)
        n_rows = np.ceil(n_plots / n_cols).astype(np.int32)
        fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
        ax = ax.flatten()
        fig.set_size_inches((6.5 * n_cols, 5 * n_rows))
        axis_count = 0
        for embeddings in umaps:
            if axis_count == (len(ax) - 1):
                plot_umap(ax[axis_count], embeddings, labels_sub, label_order=names, title=','.join(names),
                          leg_title='organelle', alpha=0.1, zoom_cutoff=0, plot_other=True)
            else:
                plot_umap(ax[axis_count], embeddings, labels_sub, alpha=0.4, zoom_cutoff=0, plot_other=True)
            axis_count += 1
        fig.savefig(os.path.join(weights_dir,
                                 'UMAP_{}_{}_{}runs.png'.format('_'.join(names),
                                                                dist_metric, n_plots)),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_tic_umap(input_batch, plot_key, col_key, label_key, dist_metric ='cosine', n_nbr=15, split='test'):
    df_meta_all = load_meta(dataset_dirs, splits=tuple([split]))
    df_meta = df_meta_all[split]
    for embed_dirs in input_batch:
        for embed_dir in embed_dirs:
            model_name = os.path.basename(embed_dir)
            fname = os.path.join(embed_dir, 'umap_nbr{}_{}_*.npy'.format(n_nbr, dist_metric))
            print(fname)
            umap_paths = glob.glob(fname)
            print(umap_paths)
            umaps = [np.load(path) for path in umap_paths]
            embeddings = umaps[0]
            df_meta[['umap x', 'umpa y']] = embeddings
            for plot_val in df_meta[plot_key].unique():
                # df_meta_sub = df_meta.loc[df_meta[plot_key] ==  plot_val, :]
                sub_ids = df_meta[plot_key] ==  plot_val
                n_plots = len(df_meta.loc[sub_ids, col_key].unique())
                n_cols = np.ceil(1.6 * np.sqrt(n_plots / 1.6)).astype(np.int32)
                if n_cols < n_plots:
                    n_cols = max(3, n_cols)
                n_rows = np.ceil(n_plots / n_cols).astype(np.int32)
                fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
                ax = ax.flatten()
                fig.set_size_inches((4 * n_cols / np.sqrt(n_rows), 4 * np.sqrt(n_rows)))
                axis_count = 0
                for col_val in df_meta.loc[sub_ids ,col_key].unique():
                    df_meta_copy = df_meta.copy()
                    df_meta_copy.loc[~((df_meta[col_key] == col_val) & sub_ids), label_key] = 'other'
                    if axis_count == (n_cols - 1):
                        plot_umap(ax[axis_count], df_meta_copy[['umap x', 'umpa y']].to_numpy(), df_meta_copy[label_key], title=col_val,
                                  leg_title=label_key, alpha=0.1, zoom_cutoff=0, plot_other=True)
                    else:
                        plot_umap(ax[axis_count], df_meta_copy[['umap x', 'umpa y']].to_numpy(), df_meta_copy[label_key],
                                  title=col_val,
                                  alpha=0.1, zoom_cutoff=0, plot_other=True)
                    axis_count += 1
                fig.tight_layout()
                fig.savefig(os.path.join(embed_dir,
                                         'UMAP_{}_{}_{}runs.png'.format(plot_key, plot_val,
                                                                        dist_metric)),
                            dpi=300, bbox_inches='tight')
                plt.close(fig)

def display_raw_imgs(dataset_dir, plot_key, plot_vals, col_key, col_vals, split='all'):
    df_meta_all = load_meta([dataset_dir], splits=tuple([split]))
    df_meta = df_meta_all[split].loc[:, [plot_key, col_key, 'position']].drop_duplicates()
    for plot_val in plot_vals:
        df_meta_plot = df_meta.loc[df_meta[plot_key] == plot_val, :]
        n_plots = len(col_vals)
        n_cols = np.ceil(1.6 * np.sqrt(n_plots / 1.6)).astype(np.int32)
        if n_cols < n_plots:
            n_cols = max(3, n_cols)
        n_rows = np.ceil(n_plots / n_cols).astype(np.int32)
        fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
        ax = ax.flatten()
        fig.set_size_inches((4 * n_cols / np.sqrt(n_rows), 4 * np.sqrt(n_rows)))
        axis_count = 0
        for col_val in col_vals:
            df_meta_col = df_meta_plot.loc[df_meta_plot[col_key] == col_val, :]
            imgs = []
            n_imgs = len(df_meta_col)
            for pos_idx in df_meta_col['position']:
                img = np.squeeze(np.load(os.path.join(dataset_dir, 'img_p{:03d}.npy'.format(pos_idx))))
                for c_idx in range(len(img)):
                    img[c_idx] = im_adjust(img[c_idx])
                # print(np.max(img), np.min(img))
                img = np.concatenate([np.zeros_like(img[0:1]).astype(np.uint8), img], axis=0) # add an empty red channel
                imgs.append(img)
            n_chan, ny, nx = img.shape
            stitch_dim = (n_chan,
                          int(np.ceil(np.sqrt(n_imgs)) * ny),
                          int(np.ceil(np.sqrt(n_imgs)) * nx))
            img_stitch = np.zeros(stitch_dim, dtype=np.uint8)
            count = 0
            for i in range(0, stitch_dim[1], ny):
                if count >= len(imgs):
                    break
                for j in range(0, stitch_dim[2], nx):
                    if count >= len(imgs):
                        break
                    img_stitch[:, i : i + ny, j : j + nx] = imgs[count]
                    count += 1

            ax[axis_count].axis('off')
            ax[axis_count].imshow(np.transpose(img_stitch, [1, 2, 0]))
            ax[axis_count].set_title(col_val, fontsize=10)
            axis_count += 1
        fig.suptitle(plot_val, fontsize=12)
        dst_dir = os.path.join(dataset_dir, 'figures')
        os.makedirs(dst_dir, exist_ok=True)
        fig.savefig(os.path.join(dst_dir, '{}.jpg'.format(plot_val)), dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_ard_leiden(dataset_dirs,
                    weights_dirs,
                    df_meta_all,
                    split='test'):
    # the expected number of OpenCell targets in the dataset
    num_targets = 1294
    # number of random seeds for the Leiden clustering over which to average the ARI
    n_random_states = 20
    df_meta = df_meta_all[split]
    ari_corum_all = []
    ari_ocgt_all = []
    train_dir = os.path.dirname(weights_dirs[0])
    for weights_dir in weights_dirs:
        model_name = os.path.basename(weights_dir)
        embed_dirs = [os.path.join(dataset_dir, model_name) for dataset_dir in dataset_dirs]
        vectors = []
        # print("df_meta_all:", [len(df_meta_all[split]) for split in df_meta_all])
        for embed_dir in embed_dirs:
            vec = np.load(os.path.join(embed_dir, '{}_embeddings.npy'.format(split)))
            vectors.append(vec.reshape(vec.shape[0], -1))
        vectors = np.concatenate(vectors, axis=0)
        # labels = {split: df_meta_all[split].loc[:, label_col].to_numpy() for split in splits}
        # labels_temp = labels[split]
        # vectors = vectors[labels_temp != 'other']
        # df_meta = df_meta[labels_temp != 'other']
        # labels = labels[labels_temp != 'other']
            # remove 'other' class
        print('data dimension :', vectors.shape)
        # get aggregated vectors over protein ids and its metadata
        df_meta_mean = df_meta.loc[:, ['gene', 'protein-complex-level ground truth', 'organelle-level ground truth']]
        df_meta_mean.drop_duplicates(subset=['gene'], inplace=True)
        mean_vectors = []
        for protein_id in df_meta_mean['gene']:
            mask = df_meta['gene'] == protein_id
            mean_vectors.append(np.mean(vectors[mask, :], axis=0))
        mean_vectors = np.stack(mean_vectors)
        print('data dimension :', mean_vectors.shape)
        print('Constructing Ann data object ...')
        adata = ad.AnnData(
            sp.sparse.csc_matrix(vectors),
            obs=df_meta,
            var=pd.DataFrame(index=np.arange(vectors.shape[1])),
            dtype='float'
        )
        cwv = ClusteringWorkflow(adata=adata)

        print('preprocess the VQ2 features and calculate the principal components ...')
        # cwv.preprocess(do_log1p=False, do_scaling=False, n_top_genes=None, n_pcs=200)
        cwv.preprocess(do_log1p=False, do_scaling=False, n_top_genes=None, n_pcs=None)

        print('calculate the kNN matrix')
        # cwv.calculate_neighbors(n_neighbors=10, n_pcs=200, metric='euclidean')
        cwv.calculate_neighbors(n_neighbors=10, n_pcs=0, metric='euclidean', use_rep='X')
        # assert cwv.adata.X.shape[0] == num_targets

        print('calculate ARI ...')
        ari_corum = cwv.calculate_ari(
            ground_truth_label='protein-complex-level ground truth',
            n_random_states=n_random_states,
            model_name=name_mapping[model_name]
        )
        # ari_kegg_pathway = cwv.calculate_ari(
        #     ground_truth_label='pathway_id', n_random_states=n_random_states
        # )
        ari_corum_all.append(ari_corum)

        # using our opencell ground-truth (single grade-3 annotations)
        ari_ocgt = cwv.calculate_ari(
            ground_truth_label='organelle-level ground truth',
            n_random_states=n_random_states,
            model_name=name_mapping[model_name])
        ari_ocgt_all.append(ari_ocgt)

    ari_corum_all = pd.concat(ari_corum_all)
    ari_ocgt_all = pd.concat(ari_ocgt_all)
    ari_ocgt_all.reset_index(drop=True, inplace=True)
    ari_corum_all.reset_index(drop=True, inplace=True)
    ari_corum_all.to_csv(os.path.join(train_dir, 'leiden_ari_corum.csv'))
    ari_ocgt_all.to_csv(os.path.join(train_dir, 'leiden_ari_organelle.csv'))
    blue, orange, green, red, *_ = sns.color_palette('tab10')
    x, y = 'resolution', 'ari'

    plt.figure(figsize=(8, 6))
    plt.gca().set_xlabel('Leiden resolution')
    plt.gca().set_ylabel('Adjusted rand index')
    sns.lineplot(data=ari_ocgt_all, x=x, y=y, hue='model')
    plt.gca().set(xscale='log')
    plt.title('OpenCell annotations')
    plt.savefig(os.path.join(train_dir, 'leiden_ari_organelle_raw_no_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # sns.lineplot(data=ari_kegg_pathway, x=x, y=y, label='Kegg pathways', color=green)


    plt.figure(figsize=(8, 6))
    plt.gca().set_xlabel('Leiden resolution')
    plt.gca().set_ylabel('Adjusted rand index')
    sns.lineplot(data=ari_corum_all, x=x, y=y, hue='model')
    plt.gca().set(xscale='log')
    plt.title('CORUM clusters')
    plt.savefig(os.path.join(train_dir, 'leiden_ari_corum_raw_no_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # sns.lineplot(data=ari_corum_wo_largest, x=x, y=y, label='CORUM clusters (w/o largest)', color=orange)

    # plot the median cluster size on the right-hand y-axis
    # if True:
    #     ax2 = plt.gca().twinx()
    #     sns.lineplot(data=ari_ocgt, x=x, y='median_cluster_size', ax=ax2, color='gray')
    #     ax2.set(xscale='log')
    #     ax2.set(yscale='log')
    #     ax2.set_ylabel('Number of clusters')





def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--inference',
        required=False,
        action='store_true',
        help='run inference',
    )
    parser.add_argument(
        '-r', '--reduction',
        required=False,
        action='store_true',
        help='run dimensionality reduction',
    )
    parser.add_argument(
        '-e', '--evaluation',
        required=False,
        action='store_true',
        help='evaluate embedding quality',
    )
    parser.add_argument(
        '-p', '--plot',
        required=False,
        action='store_true',
        help='plot evaluation results',
    )
    parser.add_argument(
        '-g', '--gpu',
        type=int,
        required=False,
        default=0,
        help="ID of the GPU to use",
    )
    parser.add_argument(
        '-n', '--nn',
        required=False,
        action='store_true',
        help="Use neural network for linear evaluation",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # dataset_dirs = ['/CompMicro/projects/dynacontrast/4_cell_types']
    # dataset_dirs = ['/gpfs/CompMicro/projects/dynacontrast/opencell/2021-7-15_good-fovs']
    dataset_dirs = ['/gpfs/CompMicro/projects/dynacontrast/tic/TICM0001-1']
    # label_col = 'data_dir'
    label_col = 'organelle-level ground truth'
    # label_col = 'protein-complex-level ground truth'
    # label_keys = ["/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS",
    #         "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS",
    #         '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice',
    #         '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/MOCK_IFNA_48',
    #         '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNA_24',
    #         '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNA_48',
    #         '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24',]
    # label_values = [0, 1, 2, 3, 3, 3, 3]
    # label_mapping =  dict(zip(label_keys, label_values))

    input_batch = []
    weights_dirs = \
        [
        # "/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_train_tstack/mock_z32_nh16_nrh16_ne512_cc0.25",
        # "/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_train_tstack/mock_z32_nh32_nrh32_ne128_cc0.25",
        # "/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_train_tstack/mock_z32_nh32_nrh32_ne256_cc0.25",
        # "/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_train_tstack/mock_z32_nh32_nrh32_ne512_cc0.25",
        # "/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_train_tstack/mock_z32_nh64_nrh64_ne512_cc0.25",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh32_nrh32_ne128_cc0.25",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne128_cc0.25",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh32_nrh32_ne128_alpha0.05_wa1_wt0.1",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh32_nrh32_ne128_alpha0.01_wa1_wt0.5",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne128_alpha0.01_wa1_wt0.5",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha0.01_wa1_wt0.5",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha0.002_wa1_wt0.5_aug",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha0.01_wa1_wt0.5_aug",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne128_alpha0.01_wa1_wt0.5_aug",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha0.01_wa1_wt0.5_wn-0.5_mrg0.5_aug",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha0.05_wa1_wt0.5_wn-0.5_mrg1_aug_shuff",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha1_mrg1_aug_hardtriloss",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha1_mrg1_aug_alltriloss",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha0.1_mrg1_aug_alltriloss",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha100_mrg1_aug_alltriloss",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha10_mrg1_aug_hardtriloss",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha10_mrg0.1_aug_hardtriloss",
        # "/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack/mock+low_moi_z32_nh64_nrh64_ne512_alpha10_mrg10_aug_hardtriloss",
        # "/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_z32_nh64_nrh64_ne512_alpha100_mrg1_aug_alltriloss",
        # "/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_z32_nh64_nrh64_ne512_alpha100_mrg1_npos8_aug_alltriloss",
        # "/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet50_mrg1_npos8_alltriloss",
        # "/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet50_mrg1_npos16_alltriloss",
        # "/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet50_mrg1_npos32_alltriloss",
        # "/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet50_mrg1_npos16_hardtriloss",
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet50_mrg1_npos8_noeasytriloss',
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet152_mrg1_npos4_bh384_noeasytriloss',
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet101_mrg1_npos4_bh512_noeasytriloss',
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet50_mrg1_npos4_bh768_noeasytriloss',
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet152_mrg1_npos4_bh384_alltriloss',
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet101_mrg1_npos8_bh512_alltriloss',
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet50_mrg1_npos4_bh768_alltriloss',
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet101_mrg1_npos4_bh512_alltriloss',
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet152_mrg1_npos4_bh384_alltriloss_nostop',
        # '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train/CM+kidney_ResNet50_mrg1_npos8_bh768_alltriloss_nostop',
        # '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right/dnm_train/CM+kidney+A549_ResNet101_mrg1_npos4_bh512_noeasytriloss',
        # '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right/dnm_train/CM+kidney+A549_ResNet50_mrg1_npos4_bh768_alltriloss',
        # '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right/dnm_train/CM+kidney+A549_ResNet101_mrg1_npos4_bh512_alltriloss',
        # '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right/dnm_train/A549_ResNet101_mrg1_npos4_bh512_alltriloss_tr',
        # '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right/dnm_train/CM+kidney+A549_ResNet101_mrg1_npos4_bh512_alltriloss_tr',
        # '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right/dnm_train/CM+kidney+A549_ResNet101_mrg1_npos4_bh512_noeasytriloss_datasetnorm',
        # '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right/dnm_train/CM+kidney+A549_ResNet50_mrg1_npos4_bh768_alltriloss_datasetnorm',
        # '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right/dnm_train/CM+kidney+A549_ResNet50_mrg1_npos4_bh768_noeasytriloss_datasetnorm',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_notrj_patchnorm_rot',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_rot',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_notrj_patchnorm_fullrot_jit_crop',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_fullrot_jit_crop',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_fullrot_crop',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_fullrot_jit',
        # # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_fullrot_jit_crop_no_projhd',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_no_projhd',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_2X_moretrj_patchnorm_fullrot_jit_crop_no_projhd',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet101_moretrj_patchnorm_fullrot_jit_crop_no_projhd',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_0.1_npos_2_no_projhd',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_1_npos_2_no_projhd',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_0.5_npos_8_no_projhd',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_0.5_npos_4_no_projhd',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_0.5_npos_2_no_projhd',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_0.1_npos_2',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_1_npos_2',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_0.5_npos_8',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_0.5_npos_4',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop_ntxent_0.5_npos_2',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_datasetnorm_rot_crop_split_npos_2_zarr_shuffle_random',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_patchnorm_rot_crop_split_npos_2_zarr_shuffle_random',
        # # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_patchnorm_rot_crop_split_ntxent_0.5_npos_2_zarr_shuffle_random',
        # # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_datasetnorm_rot_crop_split_ntxent_0.5_npos_2_zarr_shuffle_random',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_datasetnorm_rot_crop_split_ntxent_0.5_npos_2_zarr_random_shuffle_val',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_patchnorm_rot_crop_split_ntxent_0.5_npos_2_zarr_random_shuffle_val',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_datasetnorm_rot_crop_split_npos_2_zarr_shuffle_random_proj',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_patchnorm_rot_crop_split_npos_2_zarr_shuffle_random_proj',
        # # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_patchnorm_rot_crop_split_ntxent_0.5_npos_2_zarr_shuffle_random_proj',
        # # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_datasetnorm_rot_crop_split_ntxent_0.5_npos_2_zarr_shuffle_random_proj',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_datasetnorm_rot_crop_split_ntxent_0.5_npos_2_zarr_random_shuffle_val_proj',
        # '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_train/CM+kidney+A549_ResNet50_patchnorm_rot_crop_split_ntxent_0.5_npos_2_zarr_random_shuffle_val_proj',
        '/gpfs/CompMicro/projects/dynacontrast/opencell/2021-7-15_good-fovs/models/rot_crop_jit_ntxent_0.5',
        '/gpfs/CompMicro/projects/dynacontrast/opencell/2021-7-15_good-fovs/models/rot_crop_jit_ntxent_0.1',
        '/gpfs/CompMicro/projects/dynacontrast/opencell/2021-7-15_good-fovs/models/rot_crop_jit_ntxent_2',
        '/gpfs/CompMicro/projects/dynacontrast/opencell/2021-7-15_good-fovs/models/rot_crop_jit_ntxent_2_label_protein',
        '/gpfs/CompMicro/projects/dynacontrast/opencell/2021-7-15_good-fovs/models/rot_crop_jit_triplet',
        # '/gpfs/CompMicro/projects/dynacontrast/opencell/2021-7-15_good-fovs/models/cytoself2',

        # '/gpfs/CompMicro/projects/dynacontrast/opencell/2021-7-15_good-fovs/models/cytoself1',


        ]

    name_mapping = \
        { 'rot_crop_jit_ntxent_0.5': 'dynacontrast, ntxent T=0.5',
        'rot_crop_jit_ntxent_0.1': 'dynacontrast, ntxent T=0.1',
        'rot_crop_jit_ntxent_2':  'dynacontrast, ntxent T=2',
        'rot_crop_jit_triplet': 'dynacontrast, triplet loss',
        'cytoself2': 'cytoself',
        'rot_crop_jit_ntxent_2_label_protein': 'dynacontrast, label=protein, ntxent T=2'}
    # {'CM+kidney+A549_QLIPP_ResNet50_notrj_patchnorm_rot': 'no augmentation',
    # 'CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_rot': 'defocus',
    # 'CM+kidney+A549_QLIPP_ResNet50_notrj_patchnorm_fullrot_jit_crop': 'rotation + intensity jitter + random crop',
    # 'CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_fullrot_jit_crop': '4 augmentations',
    # 'CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_fullrot_crop': 'defocus + rotation + random crop',
    # 'CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_fullrot_jit': 'defocus + rotation + intensity jitter',
    # 'CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm': 'defocus-1',
    # 'CM+kidney+A549_QLIPP_ResNet50_moretrj_patchnorm_fullrot_jit_crop_no_projhd' : '4 augmentations + no projection',
    # 'CM+kidney+A549_QLIPP_ResNet50_moretrj_datasetnorm_fullrot_crop': 'dataset norm + defocus + rotation + random crop',
    # 'CM+kidney+A549_QLIPP_ResNet50_2X_moretrj_patchnorm_fullrot_jit_crop': '2X model width + 4 augmentations',
    # 'CM+kidney+A549_QLIPP_ResNet101_moretrj_patchnorm_fullrot_jit_crop': 'ResNet101 + 4 augmentations'}


    # splits = ('train', 'val', 'test')
    # splits = ['test']
    splits = ('all',)

    df_meta_all = load_meta(dataset_dirs, splits=splits)
    for weights_dir in weights_dirs:
        model_name = os.path.basename(weights_dir)
        embed_dirs = [os.path.join(dataset_dir, model_name) for dataset_dir in dataset_dirs]
        input_batch.append(embed_dirs)

    if args.evaluation:
        # assert len(df_meta_all[label_col].unique()) == len(label_mapping), 'dataset_dirs and dataset_labels must have equal length'
        # train_labels = df_meta_all.loc[df_meta_all['split'] == 'train', label_col].map(label_mapping)
        # train_labels = train_labels.to_numpy().astype(np.int64)
        # val_labels = df_meta_all.loc[df_meta_all['split'] == 'val', label_col].map(label_mapping).to_numpy().astype(np.int64)

        vectors = {split : [] for split in splits}
        batch_size = 16*1024
        patience = 20
        learn_rate = 0.1
        n_epochs = 5000
        earlystop_metric = 'total_loss'
        retrain = True
        start_epoch=0
        rand_seed=0
        # device = torch.device('cuda:%d' % args.gpu)
        log_df = []
        for embed_dirs, weights_dir in zip(input_batch, weights_dirs):
            model_name = os.path.basename(weights_dir)
            output_dir = os.path.join(weights_dir, 'evaluation_lr{}'.format(learn_rate))
            vectors = {split: [] for split in splits}
            labels = {split: df_meta_all[split].loc[:, label_col].to_numpy() for split in splits}
            # print("df_meta_all:", [len(df_meta_all[split]) for split in df_meta_all])
            for split in splits:
                for embed_dir in embed_dirs:
                    vec = np.load(os.path.join(embed_dir, '{}_embeddings.npy'.format(split)))
                    vectors[split].append(vec.reshape(vec.shape[0], -1))
                vectors[split] = np.concatenate(vectors[split], axis=0)
                # remove 'other' class

            hid_dim = vectors[split].shape[1]
            vectors, labels, df_meta_sub = resplit_data(vectors, labels, df_meta_all, split_cols='gene')
            if args.nn:
                model = LogisticRegressionNN(input_dim=hid_dim, n_class=max(label_values)+1).to(device)
                tri_train_set = TripletDataset(train_labels, lambda index: train_set[index], 1)
                tri_val_set = TripletDataset(val_labels, lambda index: val_set[index], 1)
                # Data Loader
                train_loader = DataLoader(tri_train_set,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          pin_memory=False,
                                          )
                val_loader = DataLoader(tri_val_set,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=False,
                                        )

                model = train_with_loader(model,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  output_dir=output_dir,
                                  n_epochs=n_epochs,
                                  lr=learn_rate,
                                  device=device,
                                  patience=patience,
                                  earlystop_metric=earlystop_metric,
                                  retrain=retrain,
                                  log_step_offset=start_epoch)
            else:
                # clf = LogisticRegressionCV(
                #     Cs=5,
                #     intercept_scaling=1,
                #     max_iter=5000,
                #     random_state=rand_seed,
                #     solver='saga',
                #     dual=False,
                #     fit_intercept=True,
                #     penalty='l2',
                #     tol=0.0001,
                #     cv=5,
                #     verbose=1)
                clf = LogisticRegression(
                    C=10**-3,
                    intercept_scaling=1,
                    max_iter=1000,
                    random_state=rand_seed,
                    solver='lbfgs',
                    dual=False,
                    fit_intercept=True,
                    penalty='l2',
                    tol=0.001,
                    multi_class='multinomial',
                    verbose=1,
                    n_jobs=None)
                if 'test' in splits: # use test set for evaluation if available, otherwise validation set
                    vector_val = vectors['test']
                    label_val =  labels['test']
                else:
                    vector_val = vectors['val']
                    label_val = labels['val']
                clf, train_score, val_score, train_pred, val_pred = model_fit(clf, vectors['train'], labels['train'], vector_val,
                                                                              label_val)
                log_df = {'model': [model_name], 'train_acc': [train_score], 'val_acc': [val_score]}
                pred_df = pd.DataFrame(np.stack([val_pred, label_val], axis=1), columns=['y pred', 'y true'])
                log_df = pd.DataFrame.from_dict(log_df)
                print(log_df)
                log_df.to_csv(os.path.join(weights_dir, 'linear_eval_{}.csv'.format(label_col.replace(' ', '_'))), index=None)
                pred_df.to_csv(os.path.join(weights_dir, 'val_prediction_{}.csv'.format(label_col.replace(' ', '_'))),
                              index=None)


    if args.plot:
        # plot_linear_eval(embed_dirs, nn=args.nn)
        # plot_confusion_mat(embed_dirs)
        # plot_cluster_scores(embed_dirs)
        # plot_complex_umap(['CCDC93', 'SNX'])
        # plot_organelle_umap(['vesicles', 'mitochondria'], embed_dirs)
        # plot_ard_leiden(dataset_dirs,
        #                 weights_dirs,
        #                 df_meta_all,
        #                 split='test')
        # plot_tic_umap(input_batch, plot_key='rating', col_key='gene', label_key='condition',
        #               n_nbr=15, split='all')
        display_raw_imgs(dataset_dirs[0], plot_key='gene', plot_vals=['RAB24', 'RAB28', 'ATL2', 'C8orf33_1'],
                         col_key='condition', col_vals=['Mock', 'Infected'], split='all')
















