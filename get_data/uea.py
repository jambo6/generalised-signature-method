"""
uea.py
============================
Downloads the UEA archive and converts to torch tensors. The data is saved in /processed/uea.

This can also download and convert the UCR data, if wanted, by changing the dataset variable to 'ucr'.
"""
from definitions import *
import numpy as np
import torch
from tqdm import tqdm
from sktime.utils.load_data import load_from_arff_to_dataframe
from sklearn.preprocessing import LabelEncoder
from get_data.helpers import save_zip, unzip, mkdir_if_not_exists


def download(dataset='uea'):
    """ Downloads the uea data to '/raw/uea'. """
    raw_dir = DATA_DIR + '/raw'
    mkdir_if_not_exists(raw_dir)

    if dataset == 'uea':
        url = 'http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip'
        save_dir = DATA_DIR + '/raw/uea'
        zipname = save_dir + '/uea.zip'
    elif dataset == 'ucr':
        url = 'http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip'
        save_dir = DATA_DIR + '/raw/ucr'
        zipname = save_dir + '/ucr.zip'
    else:
        raise ValueError('Can only download uea or ucr. Was asked for {}.'.format(dataset))

    if os.path.exists(save_dir):
        print('Path already exists at {}. If you wish to re-download you must delete this folder.'.format(save_dir))
        return

    mkdir_if_not_exists(save_dir)

    if len(os.listdir(save_dir)) == 0:
        if not os.path.exists(zipname):
            save_zip(url, zipname)
        unzip(zipname, save_dir)


def create_torch_data(train_file, test_file):
    """Creates torch tensors for test and training from the UCR arff format.

    Args:
        train_file (str): The location of the training data arff file.
        test_file (str): The location of the testing data arff file.

    Returns:
        data_train, data_test, labels_train, labels_test: All as torch tensors.
    """
    # Get arff format
    train_data, train_labels = load_from_arff_to_dataframe(train_file)
    test_data, test_labels = load_from_arff_to_dataframe(test_file)

    def convert_data(data):
        # Expand the series to numpy
        data_expand = data.applymap(lambda x: x.values).values
        # Single array, then to tensor
        data_numpy = np.stack([np.vstack(x).T for x in data_expand])
        tensor_data = torch.Tensor(data_numpy)
        return tensor_data

    train_data, test_data = convert_data(train_data), convert_data(test_data)

    # Encode labels as often given as strings
    encoder = LabelEncoder().fit(train_labels)
    train_labels, test_labels = encoder.transform(train_labels), encoder.transform(test_labels)
    train_labels, test_labels = torch.Tensor(train_labels), torch.Tensor(test_labels)

    return train_data, test_data, train_labels, test_labels


def convert_all_files(dataset='uea'):
    """ Convert all files from a given /raw/{subfolder} into torch data to be stored in /interim. """
    assert dataset in ['uea', 'ucr']
    if dataset == 'uea':
        arff_folder = DATA_DIR + '/raw/uea/Multivariate_arff'
    elif dataset == 'ucr':
        arff_folder = DATA_DIR + '/raw/ucr/Univariate_arff'

    # Time for a big for loop
    for ds_name in tqdm([x for x in os.listdir(arff_folder) if os.path.isdir(arff_folder + '/' + x)]):
        # File locations
        train_file = arff_folder + '/{}/{}_TRAIN.arff'.format(ds_name, ds_name)
        test_file = arff_folder + '/{}/{}_TEST.arff'.format(ds_name, ds_name)

        # Ready save dir
        save_dir = DATA_DIR + '/processed/{}'.format(ds_name)

        # If files don't exist, skip.
        if any([x.split('/')[-1] not in os.listdir(arff_folder + '/{}'.format(ds_name)) for x in (train_file, test_file)]):
            if ds_name not in ['Images', 'Descriptions']:
                print('No files found for folder: {}'.format(ds_name))
            continue
        elif os.path.isdir(save_dir):
            print('Files already exist for: {}'.format(ds_name))
            continue
        else:
            train_data, test_data, train_labels, test_labels = create_torch_data(train_file, test_file)

            # Compile train and test data together
            data = torch.cat([train_data, test_data])
            labels = torch.cat([train_labels, test_labels])

            # Save original train test indexes in case we wish to use original splits
            original_idxs = (np.arange(0, train_data.size(0)), np.arange(train_data.size(0), data.size(0)))

            # Save data
            save_pickle(data, save_dir + '/data.pkl')
            save_pickle(labels, save_dir + '/labels.pkl')
            save_pickle(original_idxs, save_dir + '/original_idxs.pkl')


if __name__ == '__main__':
    # Download
    dataset = 'uea'
    download(dataset)

    # Convert
    convert_all_files(dataset)

