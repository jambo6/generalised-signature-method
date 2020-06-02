"""
prepare_data.py
===================================
Some pre-model data preparation functions.
"""
from definitions import *
import signatory
import numpy as np
from src.features.signatures.functions import get_augmentation_pipeline
from src.features.scaling import TrickScaler
from src.data.make_dataset import TimeSeriesDataset
import src.features.window as window_module
from experiments.dicts.grids import ADDITIONAL_PARAM_GRIDS
from src.features.signatures.transforms import Disintegrator


def get_data(ds_name, train_test_split):
    """ Load in the train and test datasets. """
    dataset = TimeSeriesDataset(ds_name)
    ds_train, ds_test = dataset.get_train_test_split(method=train_test_split, seed=1)
    return ds_train, ds_test


def preprocess(dataset, scaling, tfms, scaling_first=True):
    """Sets up a pre-processing pipeline for transformations to occur before model training.

    This function handles scaling and path augmentations. This is in its own script primarily because it makes computing
    the classifier input channels easier once we already know the in channels after the preprocessing steps. Really this
    is a bit lazy and we could move it.

    Args:
        dataset (TimeSeriesDataset): The dataset being tested.
        scaling (str): Method of data scaling. Can choose from ('stdsc', 'maxabs', 'minmax')
        tfms (list): List of path transforms that can be handled by `get_augmentation_pipeline(tfms)'.
        scaling_first (bool): True for scaling to be applied first (to the paths) False for after (to the signatures).

    Returns:
        list: A list of steps ready to be put into an sklearn pipeline
    """
    # Setup scaling
    scaler = TrickScaler(scaling)

    # Apply tfms here so they are not computed multiple times
    if isinstance(tfms, list):
        aug_pipe = get_augmentation_pipeline(tfms)
    else:
        aug_pipe = NullTransformer()

    # Now make a pipeline ready list of trasnforms
    pipe_list = [
        ('scaling', scaler) if scaling_first is True else None,
        ('path_tfms', aug_pipe),
        ('scaling', scaler) if scaling_first is False else None
    ]
    pipe_list = [x for x in pipe_list if x is not None]

    # Run on a single entry to give size information
    in_channels = aug_pipe.fit_transform(dataset.data[[0], :, :]).size(2)

    return pipe_list, in_channels


def compute_input_size(in_channels,
                       ds_length,
                       window_name,
                       window_kwargs,
                       clf,
                       disintegrations,
                       augment_out,
                       num_augments,
                       num_projections,
                       projection_channels,
                       sig_tfm,
                       depth,
                       sanity_dim=1e5,
                       ):
    """Compute the size of the input to the classifier after sig tfms so can be used to setup a network.

    There are a huge amount of signature options that can be applied that will change the number of features that are
    fed into the model. This function computes, given some set of options, how many input features there will be to the
    classifier. This output is a
    """
    # Pre-logic for correct values
    if any([augment_out is None, num_augments is None]):
        num_augments, augment_out = 1, 0

    # Projection channels can specify a net, in which case we care only about the final value.
    augment_out_is_tuple = False
    if isinstance(augment_out, tuple):
        augment_out_is_tuple = True
        augment_out = augment_out[-1]

    # Bool if random_projections
    rp_bool, rp_vars = False, [projection_channels, num_projections]
    if all([isinstance(x, int) for x in rp_vars]):
        if all([x > 0 for x in rp_vars]):
            rp_bool = True

    # Bool if learnt_augment
    augment_bool, augment_vars = False, [num_augments, augment_out]
    if all([isinstance(x, int) for x in augment_vars]):
        if all([x > 0 for x in augment_vars]):
            augment_bool = True

    # Cannot have both
    assert any([augment_bool is False, rp_bool is False])

    # Signature vs logsignature.
    compute_channels = signatory.signature_channels if sig_tfm == 'signature' else signatory.logsignature_channels

    # Number of windows
    window = prepare_window(ds_length, window_name, window_kwargs)
    num_windows = window.num_windows(ds_length)

    # Disintegration sizes
    num_disint, len_disint = Disintegrator(size=disintegrations).num_channels(in_channels)

    # In channels to the signature, differs if random projection of learnt augment
    if rp_bool:
        signature_channels = compute_channels(projection_channels, depth)
        in_channels_clf = signature_channels * num_windows * num_projections * num_disint
    elif augment_bool:
        if not augment_out_is_tuple:
            len_disint = 0
        signature_channels = compute_channels(len_disint + augment_out, depth)
        in_channels_clf = signature_channels * num_windows * num_augments * num_disint
    else:
        signature_channels = compute_channels(len_disint, depth)
        in_channels_clf = signature_channels * num_windows * num_disint

    # Make errors
    class InChannelsError(Exception):
        pass

    if in_channels_clf > sanity_dim:
        raise InChannelsError('In channels = {}. Too high!'.format(in_channels_clf))

    # Fix for CNNRes which sweeps over the stream dim.
    if all([clf in ('cnnres', 'gru'), window_name in ['Sliding', 'Expanding', 'Dyadic']]):
        in_channels_clf = int(in_channels_clf / num_windows)

    return in_channels_clf, signature_channels


def prepare_window(ds_length, window_name, window_kwargs):
    """Window needs special preparation as the parameters can be dependent on dataset length.

    Args:
        ds_length (int): Length of the dataset.
        window_name (str): Name of the window module.
        window_kwargs (dit): Key word arguments from the grid run.

    Returns:
        window module
    """
    # Variable parameters for ('Expanding'/'Sliding') dependent on the size of the dataset
    if window_name in ['Sliding', 'Expanding']:
        num_windows = ADDITIONAL_PARAM_GRIDS['window'][window_name][window_kwargs['size']]['num_windows']
        length = int(np.floor(ds_length / num_windows))
        window_kwargs = {'length': length, 'step': length}

    window = window_module.window_getter(window_name, **window_kwargs)

    return window
