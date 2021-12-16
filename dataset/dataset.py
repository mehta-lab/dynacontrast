from typing import Callable, Tuple, Iterator
import dask.array as da
from dask.array.slicing import shuffle_slice
import math
import numpy as np
import torch
import random
import zarr
from torch.utils.data import Dataset, IterableDataset

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    # Pytorch seeds each worker in DataLoader with seed + worker_id.
    # Here we want to ensure each worker has the same random seed but change after every epoch
    # So each worker reads different part of the dataset in each epoch but the data are randomized after every epoch
    worker_seed = (torch.initial_seed() - worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = 0
    overall_end = len(dataset)
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))

    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

class TripletDataset(Dataset):
    """Old version of Triplet Dataset for sampling positive pairs from numpy arrays in memory.
    For sampling from large dataset, convert to zarr array and using TripletIterDataset.
    Adapted from https://github.com/TowardHumanizedInteraction/TripletTorch
    The TripletDataset extends the standard Dataset provided by the pytorch
    utils. It provides simple access to data with the possibility of returning
    more than one sample per index based on the label.
    Attributes
    ----------
    labels  : np.ndarray
              Array containing all the labels respectively to each data sample.
              Labels needs to provide a way to access a sample label by index.
    data_fn : Callable
              The data_fn provides access to sample data given its index in the
              dataset. Providding a function instead of array has been chosen
              for preprocessing and other reasons.
    size    : int
              Size gives the dataset size, number of samples.
    n_sample: int
              The value represents the number of sample per index. The other
              samples will be chosen to be the same label as the selected one. This
              allows to augment the number of possible valid triplet when used
              with a tripelt mining strategy.
    """

    def __init__(
            self: 'TripletDataset',
            labels: np.ndarray,
            data_fn: Callable,
            n_sample: int,
    ) -> None:
        """Init
        Parameters
        ----------
        labels  : np.ndarray
                  Array containing all the labels respectively to each data
                  sample. Labels needs to provide a way to access a sample label
                  by index.
        data_fn : Callable
                  The data_fn provides access to sample data given its index in
                  the dataset. Providding a function instead of array has been
                  chosen for preprocessing and other reasons.
        size    : int
                  Size gives the dataset size, number of samples.
        n_sample: int
                  The value represents the number of sample per index. The other
                  samples will be chosen to be the same as the selected one.
                  This allows to augment the number of possible valid triplet
                  when used with a tripelt mining strategy.
        """
        super(Dataset, self).__init__()
        self.labels = labels
        self.data_fn = data_fn
        self.size = len(labels)
        self.n_sample = n_sample

    def __len__(self: 'TripletDataset') -> int:
        """Len
        Returns
        -------
        size: int
              Returns the size of the dataset, number of samples.
        """
        return self.size

    def __getitem__(self: 'TripletDataset', index: int) -> Tuple[np.ndarray]:
        """GetItem
        Parameters
        ----------
        index: int
               Index of the sample to draw. The value should be less than the
               dataset size and positive.
        Returns
        -------
        labels: torch.Tensor
                Returns the labels respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        data  : torch.Tensor
                Returns the data respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        Raises
        ------
        IndexError: If index is negative or greater than the dataset size.
        """
        if not (index >= 0 and index < len(self)):
            raise IndexError(f'Index {index} is out of range [ 0, {len(self)} ]')

        label = np.array([self.labels[index]])
        datum = np.array([self.data_fn(index)])

        if self.n_sample == 1:
            return label, datum

        mask = self.labels == label
        # mask[ index ] = False
        mask = mask.astype(np.float32)

        indexes = mask.nonzero()[0]
        indexes = np.random.choice(indexes, self.n_sample - 1, replace=True)
        data = np.array([self.data_fn(i) for i in indexes])

        labels = np.repeat(label, self.n_sample)
        data = np.concatenate((datum, data), axis=0)

        labels = torch.from_numpy(labels)
        data = torch.from_numpy(data)

        return labels, data


class TripletIterDataset(IterableDataset):
    """Same as TripletDataset but optimized for data in zarr format. Use n_workers > 10
    and worker_init_fn with DataLoader for better loading speed and avoid repetitive sampling
    Attributes
    ----------
    labels  : np.ndarray
              Array containing all the labels respectively to each data sample.
              Labels needs to provide a way to access a sample label by index.
    data_fn : Callable
              The data_fn provides access to sample data given its index in the
              dataset. Providding a function instead of array has been chosen
              for preprocessing and other reasons.
    size    : int
              Size gives the dataset size, number of samples.
    n_sample: int
              The value represents the number of sample per index. The other
              samples will be chosen to be the same label as the selected one. This
              allows to augment the number of possible valid triplet when used
              with a tripelt mining strategy.
    """

    def __init__(
            self: 'TripletIterDataset',
            labels: np.ndarray,
            data: zarr.array,
            data_fn: Callable,
            n_sample: int,
            shuffle: bool = False,
    ) -> None:
        """Init
        Parameters
        ----------
        labels  : np.ndarray
                  Array containing all the labels respectively to each data
                  sample. Labels needs to provide a way to access a sample label
                  by index.
        data_fn : Callable
                  The data_fn provides access to sample data given its index in
                  the dataset. Providding a function instead of array has been
                  chosen for preprocessing and other reasons.
        size    : int
                  Size gives the dataset size, number of samples.
        n_sample: int
                  The value represents the number of sample per index. The other
                  samples will be chosen to be the same as the selected one.
                  This allows to augment the number of possible valid triplet
                  when used with a tripelt mining strategy.
        """
        super(IterableDataset, self).__init__()
        self.labels = labels
        self.data = data
        self.data_fn = data_fn
        self.size = len(labels)
        # self.size = 50
        self.n_sample = n_sample
        self.shuffle = shuffle
        self.start = 0
        self.end = len(labels)
        # self.end = 50

    def __len__(self: 'TripletIterDataset') -> int:
        """Len
        Returns
        -------
        size: int
              Returns the size of the dataset, number of samples.
        """
        return self.size

    def __iter__(self: 'TripletIterDataset') -> Iterator:
        """GetItem
        Parameters
        ----------
        index: int
               Index of the sample to draw. The value should be less than the
               dataset size and positive.
        Returns
        -------
        labels: torch.Tensor
                Returns the labels respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        data  : torch.Tensor
                Returns the data respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        Raises
        ------
        IndexError: If index is negative or greater than the dataset size.
        """

        shuffle_ind = np.array(list(range(self.size)))
        for index in range(self.start, self.end):
            # shuffle after every epoch for randomness
            if index ==  self.start and self.shuffle:
                np.random.shuffle(shuffle_ind)
            label = np.array([self.labels[shuffle_ind[index]]])
            datum = np.array([self.data_fn(self.data[shuffle_ind[index]])])
            if self.n_sample == 1:
                return label, datum

            mask = self.labels == label
            mask = mask.astype(np.float32)
            indexes = mask.nonzero()[0]
            indexes = np.random.choice(indexes, self.n_sample - 1, replace=True)
            data = np.array([self.data_fn(self.data[i]) for i in indexes])
            labels = np.repeat(label, self.n_sample)
            data = np.concatenate((datum, data), axis=0)

            labels = torch.from_numpy(labels)
            data = torch.from_numpy(data)

            yield labels, data


class ImageDataset(Dataset):
    """Basic dataset class without labels for inference
        Attributes
        ----------
        data : np.ndarray
                  The data_fn provides access to sample data given its index in the
                  dataset. Providding a function instead of array has been chosen
                  for preprocessing and other reasons.
        """

    def __init__(
            self: 'ImageDataset',
            data: np.ndarray,
             ) -> None:

        super(Dataset, self).__init__()
        self.data = data
        self.size = len(data)

    def __len__(self: 'ImageDataset') -> int:
        """Len
        Returns
        -------
        size: int
              Returns the size of the dataset, number of samples.
        """
        return self.size

    def __getitem__(self: 'ImageDataset', index: int) -> np.ndarray:
        """GetItem
        Parameters
        ----------
        index: int
               Index of the sample to draw. The value should be less than the
               dataset size and positive.
        Returns
        -------
        labels: torch.Tensor
                Returns the labels respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        datum  : torch.Tensor
                sample drawn at the selected index,
        Raises
        ------
        IndexError: If index is negative or greater than the dataset size.
        """
        if not (index >= 0 and index < len(self)):
            raise IndexError(f'Index {index} is out of range [ 0, {len(self)} ]')
        datum = self.data[index]
        return datum