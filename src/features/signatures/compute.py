"""
compute.py
======================================
Methods for computing signatures over windows.
"""
import torch
from torch import nn
import signatory
from sklearn.base import BaseEstimator, TransformerMixin
import src.models.rescaling as rescaling_module
import src.features.window as window_module
from src.features.signatures.functions import push_batch_trick, unpush_batch_trick
from experiments.ingredients.prepare.prepare_data import prepare_window


class ComputeWindowSignature(nn.Module):
    """ Generic class for computing signatures over windows. """
    def __init__(self,
                 window_name=None,
                 window_kwargs=None,
                 ds_length=None,
                 sig_tfm=None,
                 depth=None,
                 rescaling=None,
                 normalisation=None):
        """
        Args:
            window_name (str): The name of the window transformation to use (must be the name of a class in the window
                module)
            window_kwargs (dict): Dictionary of kwargs to pass to the window class.
            sig_tfm (str): The signature transformation to use 'signature' or 'logsignature'.
            depth (int): The depth to compute the signature up to.
            rescaling (str): Scaling before signature computation or after: 'pre' or 'post'.
            normalisation (
        """
        super(ComputeWindowSignature, self).__init__()
        self.ds_length = ds_length
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.ds_length = ds_length
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.normalisation = normalisation

        self.window = prepare_window(ds_length, window_name, window_kwargs)

        # Setup rescaling options
        self.pre_rescaling = lambda path, depth: path
        self.post_rescaling = lambda signature, channels, depth: signature
        if rescaling == 'pre':
            self.pre_rescaling = rescaling_module.rescale_path
        elif rescaling == 'post':
            self.post_rescaling = rescaling_module.rescale_signature

    def _check_inputs(self, window):
        assert isinstance(window, window_module.Window)

    def num_windows(self, length):
        """ Gets the window classes num_windows function. """
        return self.window.num_windows(length)

    def forward(self, path, channels=1, trick_info=False):
        # Rescale
        path = self.pre_rescaling(path, self.depth)

        # Prepare for signature computation
        path_obj = signatory.Path(path, self.depth)
        transform = getattr(path_obj, self.sig_tfm)
        length = path_obj.size(1)

        # Compute signatures in each window returning the grouped list structure
        signatures = []
        for window_group in self.window(length):
            signature_group = []
            for window in window_group:
                signature = transform(window.start, window.end)
                rescaled_signature = self.post_rescaling(signature, path.size(2), self.depth)
                untricked_path = rescaled_signature

                if self.normalisation is not None:
                    untricked_path = self.normalisation(untricked_path)

                if trick_info is not False:
                    untricked_path = unpush_batch_trick(trick_info, rescaled_signature)

                signature_group.append(untricked_path)
            signatures.append(signature_group)

        return signatures


class SklearnComputeWindowSignature(BaseEstimator, TransformerMixin):
    """ A sklearnification of the ComputeWindowSignature class.

    The window class needs to be accessed differently for sklearn vs torch models. If the model has learnt behaviour,
    then the signatures are computed in the forward call of the model. If however we use a sklearn classifier, then the
    signatures are computed prior to using the model and so we use this function to convert the class into an sklearn
    transformer.
    """
    def __init__(self,
                 window_name=None,
                 window_kwargs=None,
                 ds_length=None,
                 sig_tfm=None,
                 depth=None,
                 rescaling=None):
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.ds_length = ds_length
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.rescaling = rescaling

    def fit(self, X, y=None):
        self.computer = ComputeWindowSignature(
            self.window_name, self.window_kwargs, self.ds_length, self.sig_tfm, self.depth, self.rescaling
        )
        return self

    def transform(self, X):
        trick_info, tricked_path = push_batch_trick(X)
        signatures_tricked = torch.cat([x for l in self.computer(tricked_path) for x in l], axis=1)
        signatures = unpush_batch_trick(trick_info, signatures_tricked)
        return signatures


