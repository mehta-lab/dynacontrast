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
    # print(label_unique)
    label_unique = label_unique[label_unique != 'other']
    if plot_other:
        label_unique = np.concatenate([np.array(['other']), label_unique], axis=0)
    cmap = plt.cm.Paired(np.linspace(0, 1, sum(label_unique != 'other')))
    idx = 0
    for label in label_unique:
        if label == 'other':
            color = [0.85, 0.85, 0.85]
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


def save_single_cell_im(output_mat,
                        im_path):
    """ Plot single cell patch (unmasked, masked, segmentation mask)
    """
    n_rows = 1
    n_cols = len(output_mat)
    fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
    ax = ax.flatten()
    fig.set_size_inches((15, 5 * n_rows))
    axis_count = 0
    for im in output_mat:
        ax[axis_count].imshow(np.squeeze(im_adjust(im)), cmap='gray')
        ax[axis_count].axis('off')
        axis_count += 1
    fig.savefig(im_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def im_bit_convert(im, bit=16, norm=False, limit=[]):
    im = im.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    if norm:
        if not limit:
            limit = [np.nanmin(im[:]), np.nanmax(im[:])] # scale each image individually based on its min and max
        im = (im-limit[0])/(limit[1]-limit[0])*(2**bit-1)
    im = np.clip(im, 0, 2**bit-1) # clip the values to avoid wrap-around by np.astype
    if bit==8:
        im = im.astype(np.uint8, copy=False) # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False) # convert to 16 bit
    return im


def im_adjust(img, tol=1, bit=8):
    """
    Adjust contrast of the image
    """
    limit = np.percentile(img, [tol, 100 - tol])
    im_adjusted = im_bit_convert(img, bit=bit, norm=True, limit=limit.tolist())
    return im_adjusted