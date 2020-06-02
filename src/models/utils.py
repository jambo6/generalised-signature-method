"""
utils.py
==================================
Some helper functions for building sklearn models/using sklearn functions.
"""
from sklearn.pipeline import Pipeline
from src.features.signatures.functions import push_batch_trick, unpush_batch_trick


class Sklearnify:
    """ Sklearnifies a torch nn module. """
    def __init__(self, module, trick=False):
        self.module = module

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.module(X)


class TrickSignature:
    """ Tricks a path before signature computation, then untricks each element in the output. """
    def __init__(self, window):
        self.module = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        trick_info, tricked_path = push_batch_trick(X)
        signatures = self.module(tricked_path)
        output_list = []
        for l in signatures:
            inner_list = []
            for x in l:
                inner_list.append(unpush_batch_trick(trick_info, x))
            output_list.append(inner_list)
        return output_list


def sklearn_signature_model(sig_model, classifier):
    """Creates a sklearn pipeline with a SignatureModel and a sklearn classifier component.
    Args:
        sig_model (SignatureModel): A SignatureModel class.
        classifier (sklearn classifier): An sklearn classifier.
    Returns:
        sklearn.pipeline.Pipeline: Sklearn pipeline object consisting of the SignatureModel followed by classifier.
    """
    pipeline = Pipeline([
        ('signatures', Sklearnify(sig_model)),
        ('arch', classifier)
    ])
    return pipeline