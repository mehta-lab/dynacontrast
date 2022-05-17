import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

def plot_umap(ax, embedding_sub, labels_sub, label_order=None, title=None, leg_title=None, zoom_cutoff=1, alpha=0.1, plot_other=True):
    # top and bottom % of data to cut off
    # print(labels_sub[:10])
    if label_order is None:
        label_unique = np.unique(labels_sub)
    else:
        label_unique = label_order
    print(label_unique)
    label_unique = label_unique[label_unique != 'other']
    if plot_other:
        label_unique = np.concatenate([np.array(['other']), label_unique], axis=0)
    cmap = plt.cm.Paired(np.linspace(0, 1, sum(label_unique != 'other')))
    idx = 0
    for label in label_unique:
        if label == 'other':
            color = [0.95, 0.95, 0.95]
        else:
            color = cmap[idx]
            idx += 1
        scatter = ax.scatter(embedding_sub[labels_sub == label, 0], embedding_sub[labels_sub == label, 1],
                                         s=7,
                                         color=color, facecolors='none', alpha=alpha)
        scatter.set_facecolor("none")
    if title is not None:
        ax.set_title(title, fontsize=12)
    zoom_axis(embedding_sub[:, 0], embedding_sub[:, 1], ax, zoom_cutoff=zoom_cutoff)
    if leg_title is not None:
        leg = ax.legend(
            title=leg_title, labels=label_unique,
            loc='center left', bbox_to_anchor=(1, 0.5),
            fontsize='small')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    return ax


def zoom_axis(x, y, ax, zoom_cutoff=1):
    """
    Auto zoom axes of pyplot axes object
    Args:
        x (array): x data
        y (array): y data
        ax (object): pyplot axes object
        zoom_cutoff (float): percentage of outliers to cut off [0, 100]
    """
    xlim = [np.percentile(x, zoom_cutoff), np.percentile(x, 100 - zoom_cutoff)]
    ylim = [np.percentile(y, zoom_cutoff), np.percentile(y, 100 - zoom_cutoff)]
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])