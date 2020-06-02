"""
prepare_grids.py
===============================
Functions for setting the parameter grids for different models.
"""
import inspect
import numpy as np
from copy import deepcopy
from experiments.dicts.grids import *
from src.models.nets import DyadicModelIndividual


def set_learnt_param_grid(model_args, is_meta=False, is_sklearn=False, grid_search=False):
    """ Set the parameter grid for a learnt model."""
    space = SML_CLASSIFIER_PARAM_GRIDS[model_args['clf']]
    names = list(space.keys())

    # Loop over and put into the correct form
    param_grids = []
    for name, grid in space.items():
        # Start with the fixed args to be given to the module
        param_grid = {
            'classifier__module__{}'.format(k): v for k, v in model_args.items()
        }
        # Now get module args from the space
        param_grid.update(
            {'classifier__module__' + k: v for k, v in grid.items() if k.split('__')[0] in ['aug', 'arch']}
        )
        # Finally add the optimizer arguments
        param_grid.update(
            {'classifier__' + k.split('__')[1]: v for k, v in grid.items() if k.split('__')[0] == 'optim'}
        )
        param_grids.append(param_grid)

    if is_meta:
        param_grids = set_meta_grid(param_grids, names)

    if grid_search:
        param_grids, names = set_gs_grid(param_grids, names)

    return param_grids, names


def set_non_learnt_param_grid(model_args, is_sklearn=False, is_meta=False, grid_search=False):
    """ Set the parameter grid for a non-learnt model. """
    space = SML_CLASSIFIER_PARAM_GRIDS[model_args['clf']]
    names = list(space.keys())

    # Loop over and put into the correct form
    param_grids = []
    for name, grid in space.items():
        # Now get module args from the space
        param_grid = {
            'classifier__' + k: v for k, v in grid.items() if k.split('__')[0] == 'module'
        }
        # Finally add the optimizer arguments
        param_grid.update(
            {'classifier__' + k.split('__')[1]: v for k, v in grid.items() if k.split('__')[0] == 'optim'}
        )

        if not is_sklearn:
            param_grid['classifier__module__in_channels'] = model_args['in_channels_clf']
            param_grid['classifier__module__out_channels'] = model_args['out_channels']
        else:
            param_grid = {k.replace('__module', ''): v for k, v in param_grid.items()}

        param_grids.append(param_grid)

    if is_meta:
        param_grids = set_meta_grid(param_grids, names)

    if grid_search:
        param_grids, names = set_gs_grid(param_grids, names)

    return param_grids, names


def set_meta_grid(param_grids, names):
    """ If a meta model, the model is in classifier.module.model so we must edit the grid to reflect this. """
    # Get the grids for the meta params
    meta_grids = ADDITIONAL_PARAM_GRIDS['dyadic_meta']

    # Hold as a list or the names get overwritten and confused
    meta_params = [{} for n in names]

    for i, (name, grid) in enumerate(zip(names, param_grids)):
        # Now params for the meta classifier
        meta_grid_ = meta_grids[name]
        for k, v in meta_grid_.items():
            meta_params[i]['classifier__module__' + k] = v

        # Upate the params that need to go into the basic classifier
        for key, value in deepcopy(list(grid.items())):
            if 'classifier__module__' in key:
                # Rename classifier params
                p = key.split('classifier__module__')[1]
                grid['classifier__module__model__' + p] = grid.pop(key)

    for grid, meta in zip(param_grids, meta_params):
        grid.update(meta)

    return param_grids


def set_gs_grid(param_grids, names):
    """ Gridsearch needs just a single grid rather than s/m/l, so we just reduce to the first one. """
    return [param_grids[0]], ['grid_search']


def set_meta_params(model, grid, model_args):
    """Cannot set params normally in meta models due to the additional class wrapping.

    This is confusing and hacky, but setting params for all the different models is difficult and there seems to be no
    simple way around this.
    """
    # Params for skorch
    dunder = {
        key: value for key, value in grid.items() if key.count('__') == 1
    }

    # Params for the inner classifier
    triple_dunder = {
        key.split('__')[-1]: value for key, value in grid.items() if key.count('__') == 3
    }

    # Length is is model attr
    if hasattr(model.named_steps['classifier'].module.model, 'length'):
        triple_dunder['length'] = 1

    # Initialise
    model.set_params(**dunder)

    # Set the inner classifier
    classifier_ = CLASSIFIERS[model_args['clf']](**triple_dunder)

    # Sort meta model with params
    model.named_steps['classifier'].module = DyadicModelIndividual(
        out_channels=model_args['out_channels'], dyadic_depth=model_args['window_kwargs']['depth'],
        hidden_channels=grid['classifier__module__hidden_channels'], model=classifier_
    )

    return model


def set_batch_size(grid, X, is_sklearn):
    """ Sets a variable batch size depending on the number of training samples. """
    if not is_sklearn:
        num_samples = X.size(0)
        poss_batches = [2 ** i for i in range(1, 8)]
        idx = np.argmin(np.abs([(num_samples / x) - 40 for x in poss_batches]))
        grid['classifier__batch_size'] = poss_batches[idx]


def set_window_length(grid, model_args, model, is_learnt, is_sklearn):
    """ Sorts out an annoying issue where the window length has to be input manually. """
    if all([not is_learnt, not is_sklearn]):
        clf_args = inspect.getfullargspec(CLASSIFIERS[model_args['clf']]).args
        if 'length' in clf_args:
            grid['classifier__module__length'] = model.named_steps['signatures'].module.num_windows(
                model_args['ds_length'])
