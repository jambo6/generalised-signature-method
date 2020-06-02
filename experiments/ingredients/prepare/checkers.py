"""
checkers.py
=======================================
Checks if a model is of a given type, to help with determining which route of preparation to go down.
"""
from sklearn.base import BaseEstimator
from experiments.dicts.grids import CLASSIFIERS


def check_sklearn(clf_string):
    """ Given a classifier, returns a bool to mark if that classifier is an sklearn classifier. """
    is_sklearn = True if isinstance(CLASSIFIERS[clf_string], BaseEstimator) else False
    return is_sklearn


def check_learnt(num_augments, augment_out, normalisation):
    """ Determine if there is a learnt component to the model. """
    is_learnt = False
    if any([all([x is not None for x in (num_augments, augment_out)]), isinstance(normalisation, dict)]):
        is_learnt = True
    return is_learnt


def check_meta(window_name, clf):
    """ Check if is a dyadic meta model. """
    is_meta = False
    if (window_name == 'Dyadic') and (clf in ['gru', 'cnnres']):
        is_meta = True
    return is_meta


