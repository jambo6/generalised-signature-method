"""
transforms.py
================================
Contains sklearn transformers for path augmentations to be applied before computation of signatures. This includes both
the non-learnt augmentations and the learnt transforms.
"""
import itertools
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin


class AddTime(BaseEstimator, TransformerMixin):
    """Add time component to each path.

    For a path of shape [B, L, C] this adds a time channel to be placed at the first index. The time channel will be of
    length L and scaled to exist in [0, 1].
    """
    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        # Batch and length dim
        B, L = data.shape[0], data.shape[1]

        # Time scaled to 0, 1
        time_scaled = torch.linspace(0, 1, L).repeat(B, 1).view(B, L, 1)

        return torch.cat((time_scaled, data), 2)


class PenOff(BaseEstimator, TransformerMixin):
    """ Adds a 'penoff' dimension to each path. """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Batch, length, channels
        B, L, C = X.shape[0], X.shape[1], X.shape[2]

        # Add in a dimension of ones
        X_pendim = torch.cat((torch.ones(B, L, 1), X), 2)

        # Add pen down to 0
        pen_down = X_pendim[:, [-1], :]
        pen_down[:, :, 0] = 0
        X_pendown = torch.cat((X_pendim, pen_down), 1)

        # Add home
        home = torch.zeros(B, 1, C + 1)
        X_penoff = torch.cat((X_pendown, home), 1)

        return X_penoff


class LeadLag(BaseEstimator, TransformerMixin):
    """ Applies the leadlag transformation to each path.

    Example:
        This is a string man
            [1, 2, 3] -> [[1, 1], [2, 1], [2, 2], [3, 2], [3, 3]]
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Interleave
        X_repeat = X.repeat_interleave(2, dim=1)

        # Split out lead and lag
        lead = X_repeat[:, 1:, :]
        lag = X_repeat[:, :-1, :]

        # Combine
        X_leadlag = torch.cat((lead, lag), 2)

        return X_leadlag


class ShiftToZero():
    """ Performs a translation so all paths begin at zero. """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X - X[:, [0], :]


class CumulativeSum():
    """ Cumulative sum transform. """
    def __init__(self, append_zero=False):
        self.append_zero = append_zero

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.append_zero:
            X = AppendZero().transform(X)
        return torch.cumsum(X, 1)


class AppendZero():
    """ This will append a zero starting vector to every path. """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        zero_vec = torch.zeros(size=(X.size(0), 1, X.size(2)))
        return torch.cat((zero_vec, X), dim=1)


class Disintegrator(BaseEstimator, TransformerMixin):
    """Class for performing path disintegrations.

    Here a path of shape [N, L, C] with disintegration size D < C is transformed onto a path of shape
    [N, C-choose-D, L, D] with the second dimension representing all possible D-choosings of the C channels.
    """
    def __init__(self, size):
        """
        Args:
            size (int): Disintegration size.
        """
        self.size = size

    def fit(self, X, y=None):
        return self

    def num_channels(self, channels):
        """ The size of the disintegrated shape and the number of shapes. """
        if self.size is None:
            num_disintegrations = 1
            num_disintegration_channels = channels
        else:
            idxs = self.get_idxs(channels)
            num_disintegrations, num_disintegration_channels = len(idxs), len(idxs[0])
        return num_disintegrations, num_disintegration_channels

    def get_idxs(self, num_channels):
        if self.size >= num_channels:
            return [tuple(range(num_channels))]
        else:
            return list(itertools.combinations(list(range(num_channels)), self.size))

    def transform(self, path):
        # Get the indexes (including first co-ordinate time idx) that give disintegration idxs
        num_channels = path.size(2)
        idxs = self.get_idxs(num_channels)

        # Form a new path [batch_size * channels-choose-size, length, size]
        path = torch.stack([path[:, :, idx] for idx in idxs]).transpose(0, 1).contiguous()

        return path


class LearntAugment(nn.Module):
    """As LearntProjections, except that the maps aren't just simple linear projections. Instead, this corresponds to
    constructing a feedforward network taking in_channels-many inputs, and sweeping that over the whole path.

    In fact, this allows for constructing several independent feedforward networks, and sweeping all of them over the
    path.

    Note that each network operates pointwise, in the sense that it only consumes as input the values of the path at a
    particular time; it doesn't look back in time, for example. (That's a bit messier to do with irregularly sampled
    data.)
    """
    def __init__(self, in_channels, num_augments, augment_sizes, keep_original):
        """
        Arguments:
            in_channels: int; specifying how many channels are in the input.
            num_augments: int; how many feedforward networks to use.
            augment_sizes: tuple of ints; how large each layer in the feedforward networks should be. They will be
                separated by ReLU activations. For example, taking augment_sizes=(30,) corresponds to just a linear
                projection, as in LearntProjections.
            keep_original: bool; whether to include the original path as well.
        """
        super(LearntAugment, self).__init__()
        self.in_channels = in_channels
        self.num_augments = num_augments
        self.augment_sizes = augment_sizes
        self.keep_original = keep_original

        self.networks = nn.ModuleList()
        for _ in range(num_augments):
            network = nn.ModuleList()
            self.networks.append(network)
            prev_layer_size = in_channels
            for layer_size in augment_sizes:
                network.append(nn.Conv1d(in_channels=prev_layer_size,
                                         out_channels=layer_size,
                                         kernel_size=1))
                prev_layer_size = layer_size

    def __call__(self, path):
        # Convolution requires shape [N, C, L]
        path_conv = path.transpose(1, 2)

        # Each network corresponds to a different learnt augmentation
        transformed_paths = []
        for network in self.networks:
            transformed_path = network[0](path_conv)
            for conv in network[1:]:
                transformed_path = torch.relu(transformed_path)
                transformed_path = conv(transformed_path)
            if self.keep_original:
                transformed_path = torch.cat([path_conv, transformed_path], dim=1)
            transformed_paths.append(transformed_path.transpose(1, 2))

        return torch.stack(transformed_paths, dim=1)  # returns a 4D tensor (batch, num_projections, stream, channel)


class LearntProjections(LearntAugment):
    """Augments the stream of data with learnt affine projections down to lower dimensional spaces."""
    def __init__(self, in_channels, num_projections, projection_channels):
        """
        Arguments:
            in_channels: int; specifying how many channels are in the input.
            num_projections: int; how many learnt projections to take.
            projection_channels: int; how many channels should be in the output of each projection
        """
        super(LearntProjections, self).__init__(in_channels=in_channels,
                                                num_augments=num_projections,
                                                augment_sizes=(projection_channels,),
                                                keep_original=False)


class RandomProjections(LearntProjections):
    """As LearntProjections, except that the projections are randomized when they are created and not learnt during
    backpropagation.
    """
    def __init__(self, in_channels, num_projections, projection_channels):
        super(RandomProjections, self).__init__(in_channels=in_channels,
                                                num_projections=num_projections,
                                                projection_channels=projection_channels)
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class NullAugmenter(nn.Module):
    """ Basic null augmenter than can sub for a transformer but performs no operation. """
    def __call__(self, data):
        return data


