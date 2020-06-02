"""
evaluate.py
=====================
Given a trained model, performs the evaluation steps on the training data along with saving information to the run
directory.
"""
from definitions import *
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from experiments.utils import handle_error


def ready_output_tfm(is_sklearn, n_classes):
    """ If a torch model we transform the output through sigmoid/softmax (dependent on what it was trained with). """
    output_tfm = lambda x: x
    if ~is_sklearn:
        if n_classes > 2:
            output_tfm = lambda x: torch.softmax(x, dim=1)
        else:
            output_tfm = lambda x: torch.sigmoid(x)
    return output_tfm


def get_probas_and_preds(model, X, output_tfm):
    """ Return the probabilities and integer predictions of the model. """
    # Get probas
    probas = output_tfm(torch.Tensor(model.predict_proba(X)))

    # Make preds
    if probas.shape[1] > 1:
        preds = torch.argmax(probas, axis=1)
    else:
        preds = torch.round(probas)

    return probas, preds


def get_num_params(model):
    """ Get number of params if the model has a 'module' (a nn.Module) component. """
    # Number of params in model
    if hasattr(model.named_steps['classifier'], 'module'):
        model_parameters = filter(lambda p: p.requires_grad, model.named_steps['classifier'].module.parameters())
        params = np.sum([np.prod(p.size()) for p in model_parameters])
    else:
        params = np.nan
    return params


def evaluate_models(_run, trained_models, ds_train, ds_test, is_sklearn, n_classes, save_best_model):
    """Model evaluation function.

    Args:
        _run (sacred._run): A sacred experiment run object.
        trained_models:
        ds_train (TimeSeriesDataset): Training data.
        ds_test (TimeSeriesDataset): Test data.
        is_sklearn (bool): Set True if the model is an sklearn model, else pytorch assumed.
        n_classes (int): Total number of classes.
        save_best_model (bool): Set True to save the best performing model.

    Returns:
        None. (Instead saves results to the _run object.)

    """
    # Get ML-form of the data
    X_train, y_train = ds_train.to_ml()
    X_test, y_test = ds_test.to_ml()

    # Output transform is done outside of the model for stability.
    output_tfm = ready_output_tfm(is_sklearn, n_classes)

    # Save all the info
    info = {k: {} for k in trained_models.keys()}
    best_run, best_score = None, 0
    for name, model_dict in trained_models.items():
        # Get model
        model = model_dict['model']

        for tt_name, X, y in (('train', X_train, y_train), ('test', X_test, y_test)):
            probas, preds = get_probas_and_preds(model, X, output_tfm)

            # Return and save the scores
            acc = accuracy_score(y, preds)
            f1_macro = f1_score(y, preds, average='macro')
            f1_micro = f1_score(y, preds, average='micro')

            # oob only exists for RF
            oob_score = model['classifier'].oob_score_ if hasattr(model['classifier'], 'oob_score_') else None

            # Update info
            new_info = {
                'acc': acc,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'oob_score': oob_score
            }
            new_info = {k + '.{}'.format(tt_name): v for k, v in new_info.items()}
            info[name].update(new_info)

            if tt_name == 'test':
                if acc > best_score:
                    best_score = acc
                    best_run = name

        # Number of params in model
        params = get_num_params(model)

        # Useful info to save
        info[name].update({
            'training_time': model_dict['training_time'],
            'param_grid': model_dict['param_grid'],
            'n_trainable_params': params
        })

    # Save info from all hyperparams
    save_pickle(info, _run.save_dir + '/classifier_scores.pkl')

    # Log the best test score run info
    for name, item in info[best_run].items():
        _run.log_scalar(name, item)

    # Save best model
    if save_best_model:
        try:
            save_pickle(trained_models[best_run]['model'], use_dill=True)
        except Exception as e:
            handle_error(_run, e, err_name='model_save_error')




