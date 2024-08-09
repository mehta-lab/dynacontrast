import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience. Adapted from
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class DataLoader(DataLoader):
    """Override Pytorch 1.2 Dataloader's behavior of no __len__ for IterableDataset"""

    def __init__(self, dataset, **kwargs):
        super(DataLoader, self).__init__(dataset, **kwargs)

    def __len__(self):
        return np.ceil(len(self.dataset) / self.batch_size).astype(np.int64)


def zscore(input_image, channel_mean=None, channel_std=None):
    """
    Performs z-score normalization. Adds epsilon in denominator for robustness

    :param input_image: input image for intensity normalization
    :return: z score normalized image
    """
    if not channel_mean:
        channel_mean = np.mean(input_image, axis=(0, 2, 3))
    if not channel_std:
        channel_std = np.std(input_image, axis=(0, 2, 3))
    channel_slices = []
    for c in range(len(channel_mean)):
        mean = channel_mean[c]
        std = channel_std[c]
        channel_slice = (input_image[:, c, ...] - mean) / (std + np.finfo(float).eps)
        # channel_slice = t.clamp(channel_slice, -1, 1)
        channel_slices.append(channel_slice)
    norm_img = np.stack(channel_slices, 1)
    print("channel_mean:", channel_mean)
    print("channel_std:", channel_std)
    return norm_img


def zscore_patch(imgs):
    """
    Performs z-score normalization. Adds epsilon in denominator for robustness

    :param input_image: input image for intensity normalization
    :return: z score normalized image
    """
    means = np.mean(imgs, axis=(2, 3))
    stds = np.std(imgs, axis=(2, 3))
    imgs_norm = []
    for img_chan, channel_mean, channel_std in zip(imgs, means, stds):
        channel_slices = []
        for img, mean, std in zip(img_chan, channel_mean, channel_std):
            channel_slice = (img - mean) / (std + np.finfo(float).eps)
            # channel_slice = t.clamp(channel_slice, -1, 1)
            channel_slices.append(channel_slice)
        channel_slices = np.stack(channel_slices)
        imgs_norm.append(channel_slices)
    imgs_norm = np.stack(imgs_norm)
    # print('channel_mean:', channel_mean)
    # print('channel_std:', channel_std)
    return imgs_norm


def train_val_split(dataset, labels, val_split_ratio=0.15, seed=0):
    """Split the dataset into train and validation sets

    Args:
        dataset (TensorDataset): dataset of training inputs
        labels (list or np array): labels corresponding to inputs
        val_split_ratio (float): fraction of the dataset used for validation
        seed (int): seed controlling random split of the dataset

    Returns:
        train_set (TensorDataset): train set
        train_labels (list or np array): train labels corresponding to inputs in train set
        val_set (TensorDataset): validation set
        val_labels (list or np array): validation labels corresponding to inputs in train set

    """
    assert 0 < val_split_ratio < 1
    n_samples = len(dataset)
    # Declare sample indices and do an initial shuffle
    sample_ids = list(range(n_samples))
    np.random.seed(seed)
    np.random.shuffle(sample_ids)
    split = int(np.floor(val_split_ratio * n_samples))
    # randomly choose the split start
    np.random.seed(seed)
    split_start = np.random.randint(0, n_samples - split)
    val_ids = sample_ids[split_start : split_start + split]
    train_ids = sample_ids[:split_start] + sample_ids[split_start + split :]
    train_set = dataset[train_ids]
    train_labels = labels[train_ids]
    val_set = dataset[val_ids]
    val_labels = labels[val_ids]
    return train_set, train_labels, val_set, val_labels


def split_data(
    dataset,
    df_meta,
    split_cols=None,
    splits=("train", "val"),
    val_split_ratio=0.15,
    seed=0,
):
    """Split the dataset into train and validation sets

    Args:
        dataset (TensorDataset): dataset of training inputs
        val_split_ratio (float): fraction of the dataset used for validation
        seed (int): seed controlling random split of the dataset

    Returns:
        train_set (TensorDataset): train set
        train_labels (list or np array): train labels corresponding to inputs in train set
        val_set (TensorDataset): validation set
        val_labels (list or np array): validation labels corresponding to inputs in train set

    """
    if splits == ("all",):
        split_ids = [np.arange(len(dataset))]
    elif splits == ("train", "val"):
        assert 0 < val_split_ratio < 1
        if split_cols is None:
            split_cols = ["data_dir", "FOV"]
        elif type(split_cols) is str:
            split_cols = [split_cols]
        split_key = df_meta[split_cols].apply(
            lambda row: "_".join(row.values.astype(str)), axis=1
        )
        gss = GroupShuffleSplit(
            test_size=val_split_ratio, n_splits=2, random_state=seed
        )
        split_ids, _ = gss.split(df_meta, groups=split_key)
    else:
        raise NotImplementedError("Unsupported split type {}".format(splits))
    datasets = {split: dataset[ids] for split, ids in zip(splits, split_ids)}
    df_metas = {split: df_meta.iloc[ids] for split, ids in zip(splits, split_ids)}

    # train_set = train_set.rechunk((1, 2, 128, 128))
    # val_set = val_set.rechunk((1, 2, 128, 128))

    return datasets, df_metas
