"""
Generic structure for loading and holding datasets that can be used model running.
"""
from definitions import *
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class TimeSeriesDataset(Dataset):
    """Class for loading time-series data to be put into the experiment run functions.

    This class currently accepts as arguments a dataset name and a folder. The folder is just for if the dataset is
    sub-categorised (for example being of the uea data). It then assumes that inside folder/ds_name there are the
    following files:
        data.pkl - A pickle saved torch Tensor data of shape [N, L, C].
        labels.pkl - Corresponding labels of shape [N,].
        original_idxs.pkl - The indexes of the desired train/test split. Currently this must be specified in this data
            save file, done this way out of a desire to test on the original UEA/UCR splits.
    """
    def __init__(self, ds_name):
        """
        Args:
            ds_name (str): Name of the dataset to be loaded.
        """
        self.ds_name = ds_name

        # Get the data and labels
        self.data, self.labels, self.original_idxs, self.n_classes = get_dataset(ds_name)
        self.data = self.data.float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def size(self, *args):
        """ Copy of torch size. """
        return self.data.size(*args)

    def get_train_test_split(self, method='original', test_frac=0.33, seed=1):
        """ Returns the original train/test split as TimeSeriesDatasets. """
        # Make copies
        train_ds, test_ds = deepcopy(self), deepcopy(self)

        if method == 'original':
            train_idxs, test_idxs = self.original_idxs[0], self.original_idxs[1]
        elif method == 'alternative':
            train_idxs, test_idxs = train_test_split(np.arange(len(train_ds.data)), test_size=test_frac, random_state=seed)
        else:
            raise ValueError("method param must be one of ['original', 'alternative']")

        train_ds.data, train_ds.labels = self.data[train_idxs], self.labels[train_idxs]
        test_ds.data, test_ds.labels = self.data[test_idxs], self.labels[test_idxs]

        return train_ds, test_ds

    def to_ml(self):
        """ Returns (data, labels) format, ready for use in machine-learning. """
        return self.data, self.labels


def get_dataset(ds_name):
    """Gets a dataset with a given name.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, info): Data, labels, original split idxs, and dataset information.
    """
    # Get save_dir
    save_dir = DATA_DIR + '/processed/{}'.format(ds_name)

    # Load
    data = load_pickle(save_dir + '/data.pkl')
    labels = load_pickle(save_dir + '/labels.pkl').view(-1)
    n_classes = len(np.unique(labels))
    if n_classes > 2:
        labels = labels.long()

    # Get original train/test indexes
    original_idxs = load_pickle(save_dir + '/original_idxs.pkl')

    # Reshape labels if n_classes = 2
    if n_classes == 2:
        labels = labels.reshape(-1, 1)

    # Forward fill data if nans
    data = torch_ffill(data)

    return data, labels, original_idxs, n_classes


def torch_ffill(data):
    """ Forward fills in the length dim if data is shape [N, L, C]. """
    def fill2d(x):
        """ Forward fills in the L dimension if L is of shape [L, N]. """
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = x[np.arange(idx.shape[0])[:, None], idx]
        return out

    if isinstance(data, list):
        data_out = [torch.Tensor(fill2d(d.numpy().T).T) for d in data]
    elif data.dim() == 3:
        # Reshape to apply 2d ffill
        data_shaped = data.transpose(1, 2).reshape(-1, data.size(1)).numpy()
        data_fill = fill2d(data_shaped).reshape(-1, data.size(2), data.size(1))
        data_out = torch.Tensor(data_fill).transpose(1, 2)
    else:
        raise NotImplementedError('Needs implementing for different dimensions.')

    return data_out

