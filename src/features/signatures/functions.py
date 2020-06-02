"""
functions.py
===========================
Helper functions for all things signatures. This contains augmentation application helper functions and functions to
deal with 'path tricking'.
"""
import collections as co
from sklearn.pipeline import Pipeline
from src.features.signatures.transforms import *

# Helps keep track of path 'tricking'
_TrickInfo = co.namedtuple('TrickInfo', ('tricked', 'batch_size', 'group_size'))

# Dictionary of augmentations
AUGMENTATIONS = {
    'leadlag': LeadLag(),
    'penoff': PenOff(),
    'addtime': AddTime(),
    'cumsum': CumulativeSum(),
    'basepoint': AppendZero()
}


def get_augmentation_pipeline(aug_list):
    """ Converts a list of augmentations into an sklearn pipeline of specified augmentations. """
    pipeline = Pipeline([
        (tfm_str, AUGMENTATIONS[tfm_str]) for tfm_str in aug_list
    ])
    return pipeline


def apply_augmentation_list(data, aug_list):
    """Applies augmentations to the data if specified in list format with keys corresponding to AUGMENTATIONS.keys().

    This will build a sklearn pipeline from the augmentation list, as such, each augmentation must operate a fit and
    a transform method.

    Example:
        >>> out_data = apply_augmentation_list(data, ['addtime', 'leadlag'])
        # Is equivalent to
        >>> out_data = LeadLag().transform(AddTime().transform(data))

    Args:
        data (torch.Tensor): [N, L, C] shaped data.
        aug_list (list): A list of augmentation strings that correspond to an element of AUGMENTATIONS.

    Returns:
        torch.Tensor: Data with augmentations applied in order.
    """
    pipeline = get_augmentation_pipeline(aug_list)

    # Transform
    data_tfmd = pipeline.fit_transform(data)

    return data_tfmd


def push_batch_trick(x):
    # If given a 4D tensor, will push the first and second dimensions together to 'hide' the second dimension inside the
    # first (batch) dimension.
    if len(x.shape) == 3:
        return _TrickInfo(False, None, None), x
    elif len(x.shape) == 4:
        return _TrickInfo(True, x.size(0), x.size(1)), x.view(x.size(0) * x.size(1), x.size(2), x.size(3))
    else:
        raise RuntimeError("x has {} dimensions, rather than the expected 3 or 4".format(len(x.shape)))


def unpush_batch_trick(trick_info, x):
    # Once a batch-trick'd tensor has been through the signature then it has lost its stream dimension, so all that's
    # left is a (batch * group, signature_channel)-shaped tensor. This now pushes the trick back the other way, so that
    # the group dimensions becomes part of the channel dimension. (Where it really belongs.)
    if len(x.shape) != 2:
        raise RuntimeError("x has {} dimensions, rather than the expected 2.".format(len(x.shape)))
    if trick_info.tricked:
        return x.view(trick_info.batch_size, trick_info.group_size * x.size(1))
    else:
        return x


def combine_tricks(trick_1, trick_2):
    """ Combine two trick info's. """
    if all([not trick_1.tricked, not trick_2.tricked]):
        return _TrickInfo(False, None, None)
    else:
        batch_size = max(filter(None, [trick_1.batch_size, trick_2.batch_size]))
        group_size = sum(filter(None, [trick_1.group_size, trick_2.group_size]))
        return _TrickInfo(True, batch_size, group_size)
