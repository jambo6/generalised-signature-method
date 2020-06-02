"""
train.py
===================
Function for training experiment models.
"""
import warnings
import time
from copy import deepcopy
from sklearn.model_selection import RandomizedSearchCV
from experiments.ingredients.prepare.prepare_model import prepare_learnt_model, prepare_non_learnt_model
from experiments.ingredients.prepare.prepare_grids import set_learnt_param_grid, set_meta_params, set_non_learnt_param_grid, \
    set_window_length, set_batch_size
from experiments.dicts.grids import GRIDSEARCH_PARAM_GRIDS

warnings.simplefilter('ignore', FutureWarning)


def train_models(_run, model_args, path_tfms, dataset, is_learnt, is_sklearn, is_meta, grid_search, verbose):
    """Main function for building and training a model over an experiment configuration.

    Args:
        _run (sacred._run): The sacred run object.
        model_args (dict): A dictionary of arguments to be used in model construction.
        path_tfms (list): A list of transformations to be applied to the path before model training.
        dataset (TimeSeriesDataset): The raw data in a TimeSeriesDataset class.
        is_learnt (bool): True if requires learnt transforms.
        is_sklearn (bool): True is using a sklearn model.
        is_meta (bool): True if a dyadic meta model.
        grid_search (bool): True if performing a gridsearch (only works for sklearn).
        verbose (int): Verbosity level.

    Returns:
        dict: Dictionary of trained models and corresponding information.
    """
    # Get data
    X, y = dataset.to_ml()

    # Prepare functions
    if is_learnt:
        prepare_grid, prepare_model = set_learnt_param_grid, prepare_learnt_model
    else:
        prepare_grid, prepare_model = set_non_learnt_param_grid, prepare_non_learnt_model

    # Param grid
    param_grids, names = prepare_grid(model_args, is_sklearn=is_sklearn, is_meta=is_meta, grid_search=grid_search)

    # For storing the models
    trained_models = {k: {} for k in names}

    # Train both models using a hold out validation set
    for grid, name in zip(param_grids, names):
        # Get model
        model = prepare_model(model_args, path_tfms, is_meta, verbose)

        # Some extra setting
        set_batch_size(grid, X, is_sklearn)
        set_window_length(grid, model_args, model, is_learnt, is_sklearn)

        # Set the params
        if not is_meta:
            model.set_params(**grid)
        else:
            model = set_meta_params(model, grid, model_args)

        # Train
        start = time.time()
        if grid_search:
            grid_search = RandomizedSearchCV(
                model, GRIDSEARCH_PARAM_GRIDS[model_args['clf']], verbose=1, n_jobs=-1, n_iter=20
            )
            grid_search.fit(X, y.reshape(-1))
            model = grid_search.best_estimator_
            grid = grid_search.best_params_
        else:
            model.fit(X, y)
        elapsed = time.time() - start

        # Roll back to best validation loss
        if not is_sklearn:
            callbacks = model.named_steps['classifier'].callbacks
            state_dict = list(filter(lambda x: x[0] == 'checkpoint', callbacks))[0][-1].state_dict
            model.named_steps['classifier'].module.load_state_dict(state_dict)

        # Sklearn bug
        if is_sklearn:
            model = deepcopy(model)

        # Save
        trained_models[name]['model'] = model
        trained_models[name]['training_time'] = elapsed
        trained_models[name]['param_grid'] = grid

    return trained_models

