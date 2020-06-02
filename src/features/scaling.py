"""
scaling.py
=================================
Scaling functionality for 3D tensors.
"""
import torch
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from src.omni.base import NullTransformer


class TrickScaler:
    """Tricks an sklearn scaler so that it uses the correct dimensions.

    This class was created out of a desire to use sklearn scaling functionality on 3D tensors. Sklearn operates on a
    tensor of shape [N, C] and normalises along the channel dimensions. To make this functionality work on tensors of
    shape [N, L, C] we simply first stack the first two dimensions to get shape [N * L, C], apply a scaling function
    and finally stack back to shape [N, L, C].
    """
    def __init__(self, scaling):
        """
        Args:
            scaling (str): Scaling method, one of ['stdsc', 'maxabs', 'minmax']. If anything else will ignore.
        """
        self.scaling = scaling
        if scaling == 'stdsc':
            self.scaler = StandardScaler()
        elif scaling == 'maxabs':
            self.scaler = MaxAbsScaler()
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = NullTransformer()

    def _trick(self, X):
        return X.reshape(-1, X.shape[2])

    def _untrick(self, X, shape):
        return X.reshape(shape)

    def fit(self, X, y=None):
        self.scaler.fit(self._trick(X), y)
        return self

    def transform(self, X):
        X_tfm = self.scaler.transform(self._trick(X))
        return torch.Tensor(self._untrick(X_tfm, X.shape))