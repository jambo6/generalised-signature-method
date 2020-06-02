import os
import dill, pickle
from multiprocessing import Pool, cpu_count
import json
import pandas as pd


def save_pickle(obj, filename, use_dill=False, protocol=4, create_folder=True):
    """ Basic pickle/dill dumping.

    Given a python object and a filename, the method will save the object under that filename.

    Args:
        obj (python object): The object to be saved.
        filename (str): Location to save the file.
        use_dill (bool): Set True to save using dill.
        protocol (int): Pickling protocol (see pickle docs).
        create_folder (bool): Set True to create the folder if it does not already exist.

    Returns:
        None
    """
    if create_folder:
        _create_folder_if_not_exist(filename)

    # Save
    with open(filename, 'wb') as file:
        if not use_dill:
            pickle.dump(obj, file, protocol=protocol)
        else:
            dill.dump(obj, file)


def load_pickle(filename, use_dill=False):
    """ Basic dill/pickle load function.

    Args:
        filename (str): Location of the object.
        use_dill (bool): Set True to load with dill.

    Returns:
        python object: The loaded object.
    """
    with open(filename, 'rb') as file:
        if not use_dill:
            obj = pickle.load(file)
        else:
            obj = dill.load(file)
    return obj


def _create_folder_if_not_exist(filename):
    """ Makes a folder if the folder component of the filename does not already exist. """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def save_json(obj, filename, create_folder=True):
    """ Save file with json. """
    if create_folder:
        _create_folder_if_not_exist(filename)

    # Save
    with open(filename, 'wb') as file:
        json.dump(obj, file)


def load_json(filename):
    """ Load file with json. """
    with open(filename) as file:
        obj = json.load(file)
    return obj


def groupby_apply_parallel(grouped_df, func, *args):
    """ Performs a pandas groupby.apply operation in parallel.

    Args:
        grouped_df (grouped dataframe): A dataframe that has been grouped by some key.
        func (python function): A python function that can act on the grouped dataframe.
        *args (list): List of arguments to be supplied to the function.

    Returns:
        dataframe: The dataframe after application of the function.
    """
    with Pool(cpu_count()) as p:
        return_list = p.starmap(func, [(group, *args) for name, group in grouped_df])
    return pd.concat(return_list)


