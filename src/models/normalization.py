"""
normalization.py
=========================
Contains the GradedNormalization class that contains numerous methods of signature normalisation.
"""
import torch
import warnings


def _check_shape_and_dims(shape, dims, track_running_stats):
    shape = tuple(shape)
    for s in shape:
        if not (s is None or isinstance(s, int)):
            raise ValueError("shape must be an iterable of just Nones and integers.")

    dims = tuple(dims)
    for d in dims:
        if not isinstance(d, int):
            raise ValueError("dims must be an iterable of integers.")

    if track_running_stats:
        for i, s in enumerate(shape):
            if not (i in dims or isinstance(s, int)):
                raise ValueError("The {}-th dimension is not a dimension to compute statistics over, so an integer "
                                 "value for its size must be provided.".format(i))


def _check_input_compat(input_shape, expected_shape):
    if len(input_shape) != len(expected_shape):
        raise ValueError("Expected shape compatible with {}, instead given {}.".format(expected_shape, input_shape))
    for given_dim, expected_dim in zip(input_shape, expected_shape):
        if isinstance(expected_dim, int) and given_dim != expected_dim:
            raise ValueError("Expected shape compatible with {}, instead given {}.".format(expected_shape, input_shape))


class _Normalization(torch.nn.Module):
    def __init__(self, shape, dims, eps=1e-5, momentum=0.1, track_running_stats=True):
        super(_Normalization, self).__init__()

        _check_shape_and_dims(shape, dims, track_running_stats)

        self.shape = shape
        self.dims = dims
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            statistics_shape = [1 if i in dims else s for i, s in enumerate(shape)]
            self.register_buffer('running_mean', torch.zeros(statistics_shape))
            self.register_buffer('running_var', torch.ones(statistics_shape))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def extra_repr(self):
        return 'shape={shape}, dims={dims}, eps={eps}, momentum={momentum}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def forward(self, input):
        _check_input_compat(input.shape, self.shape)

        # Calculate batch statistics if appropriate
        if self.training or not self.track_running_stats:
            batch_mean = input.mean(dim=self.dims, keepdim=True)
            batch_var = input.var(dim=self.dims, keepdim=True)

        # Update running statistics if appropriate
        if self.training and self.track_running_stats:
            # Sort out the momentum
            if self.momentum is None:  # cumulative moving average
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    momentum = 1.0 / float(self.num_batches_tracked)
                else:
                    momentum = 0.0
            else:  # exponential moving average
                momentum = self.momentum

            # Detach to not backpropagate through history. Just want to backpropagate through the most recent batch.
            self.running_mean.detach_()
            self.running_mean *= (1 - momentum)
            self.running_mean += momentum * batch_mean
            self.running_var.detach_()
            self.running_var *= (1 - momentum)
            self.running_var += momentum * batch_var

        # Perform the normalization
        if self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = batch_mean
            var = batch_var
        normed = input - mean
        normed *= (var + self.eps).rsqrt()
        return normed


class _Affine(torch.nn.Module):
    def __init__(self, shape, dims):
        super(_Affine, self).__init__()

        _check_shape_and_dims(shape, dims, track_running_stats=False)

        self.shape = shape
        self.dims = dims
        linear_shape = [s if i in dims else 1 for i, s in enumerate(shape)]
        self.weight = torch.nn.Parameter(torch.ones(linear_shape))
        self.bias = torch.nn.Parameter(torch.zeros(linear_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        _check_input_compat(input.shape, self.shape)
        out = input * self.weight
        out += self.bias
        return out


class UniversalNormalization(torch.nn.Module):
    """There's been approximately a million normalization schemes proposed: batch, layer, instance, group, positional...

    They all basically operate on a tensor in the same way:
    - Pick some dimensions to compute the mean/variance statistics over (and batch over the other dimensions).
    - Pick some dimensions to apply an affine transformation over (and batch over the other dimensions).
    - Pick whether to:
        - Compute moving average statistics during training, and use them during both training and inference.
        - Compute single-batch statistics during both training and inference.

    For example:
    - Given an input of shape (batch, channel, stream), then BatchNorm1d compute statistics over (batch, stream) and
        applies an affine transformation over (channel,), and uses moving averages statistics.
    - Given an input of shape (batch, channel, height, width), then BatchNorm2d computes statistics over
        (batch, height, width), and applies an affine transformation over (channel,), and uses moving average
        statistics.
    - Given an input of shape (batch, channel, stream), then InstanceNorm1d computes statistics over (stream,) and
        applies an affine transformation over (channel,), and doesn't use moving average statistics.
    - Given an input of shape (batch, channel, stream), then GroupNorm first splits the channel axis into two
        dimensions, to obtain a tensor of shape (batch, group, grouped_channels, stream), and then computes statistics
        over (grouped_channels, stream) and applies an affine transformation over (group, grouped_channels), and doesn't
        use moving average statistics.
    - Given an input of shape (batch, channel, stream), then LayerNorm computes statistics over (channel, stream) and
        also applies an affine transformation over (channel, stream), and doesn't use moving average statistics.

    This class offers a flexible way to generically pick all of these options.

    For example:
    - Given a 3D input, BatchNorm1d is equivalent to norm_dims=(0, 2) and affine_dims=(1,).
    - Given a 4D input, BatchNorm2d is equivalent to norm_dims=(0, 2, 3) and affine_dims=(1,).
    - Given a 3D input, InstanceNorm1d is equivalent to norm_dims=(2,), and affine_dims=(1,)
    - Given a 3D input, GroupNorm is equivalent to viewing as (batch, group, grouped_channels, stream), and then using
        norm_dims=(2, 3) and affine_dims=(1, 2), and then reshaping back again.
    - Given a 3D input, LayerNorm is equivalent to norm_dims=(1, 2) and affine_dims=(1, 2).

    Note that you should be a little bit careful with batch dimensions.
    - The batch dimension should never be included in affine_dims. (Otherwise location within the batch dimension will
        affect the affine transformation.)
    - If track_running_stats is True, then the batch dimension should be included in norm_dims. (Otherwise a separate
        running statistic will be computed for each index in the batch dimension.)
    - If track_running_stats is False, then the batch dimension should not be included in norm_dims. (Because taking
        track_running_stats to be False suggests that we don't want a dependency on the rest of our samples.)
    No attempt is made to enforce this, as this module doesn't know which is the batch dimension.
    """

    def __init__(self, shape, norm_dims, affine_dims, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        """
        Arguments:
            shape: tuple of ints or Nones, specifying the expected shape of inputs. None is allowed for dimensions that
                we don't need to know the dimension of. We need to know the dimension for all dimensions passed in
                affine_dims, and if track_running_stats is True, then we also need to know every dimension that _isn't_
                passed in norm_dims.
            norm_dims: tuple of ints, specifying the dimensions to compute statistics over.
            affine_dims: tuple of ints, specifying the dimensions on which a linear transformation may depend. Note that
                an empty tuple correspons to a single scalar transformation applied to every element, not to no affine
                transformation. (That can be toggled with the 'affine' argument.)
            eps: As other normalization schemes.
            momentum: As other normalization schemes.
            affine: As other normalization schemes.
            track_running_stats: As other normalization schemes.
        """
        super(UniversalNormalization, self).__init__()

        self._norm = _Normalization(shape=shape, dims=norm_dims, eps=eps, momentum=momentum,
                                    track_running_stats=track_running_stats)

        # Note that this is necessary even if affine_dims is trivial, as that still corresponds to taking a single
        # (the same) affine transformation over every element.
        self.affine = affine
        if affine:
            self._affine = _Affine(shape=shape, dims=affine_dims)

    def reset_running_stats(self):
        self._norm.reset_running_stats()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self._affine.reset_parameters()

    def extra_repr(self):
        return 'affine={affine}'.format(**self.__dict__)

    def forward(self, input):
        x = self._norm(input)
        if self.affine:
            x = self._affine(x)
        return x


class GradedNormalization(torch.nn.Module):
    """Graded normalization of a signature.

    A signature is often thought of as a two or three dimensional tensor, corresponding to either
    (batch, stream, channel) or (batch, channel). However we know that the channels come with a graded structure to
    them, corresponding to the different depths, so we can split these up in a three or four dimensional ragged tensor,
    corresponding to either (batch, stream, graded, graded_channels) or (batch, graded, graded_channels). It is a
    ragged tensor as the length of the graded_channels dimension depends on the location in the graded dimension.

    It now makes sense to apply universal normalization to this ragged tensor - with a few restrictions.

    First of all, the usual caveats on the batch dimension apply. (See UniversalNormalization.__doc__.)

    Next, we explicitly disallow affine transformations from depending on the stream dimension, as stream dimensions can
    potentially get very large, and it's unlikely that this is really what we want anyway. (Just like how we use
    convolutional/recurrent/etc. networks over stream dimensions, not feedforward networks.)

    In terms of the graded structure: if computing statistics over the graded dimension then statistics must also be
    computed over the graded_channels dimension, as the latter dimension depends upon the former dimension; they're not
    independent. Conversely if allowing an affine transformation to depend on the graded_channels dimension, then it
    must also depend on the graded dimension, as a location within the former implies a location within the latter.

    This class provides for all of these possible normalizations.
    """
    def __init__(self, shape, channels, depth,
                 norm_batch=True, norm_stream=False, norm_graded=False, norm_channel=True,
                 affine_graded=True, affine_channel=False,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        # channels should be the number of channels of the underlying path
        # shape should either be of shape (batch, signature_channels(channels, depth)) or
        #     (batch, stream, signature_channels(channels, depth))
        super(GradedNormalization, self).__init__()

        if all(not x for x in (norm_batch, norm_stream, norm_graded, norm_channel)):
            raise ValueError("Not normmalizing over anything.")
        if norm_graded and not norm_channel:
            raise ValueError("Computing statistics over the graded 'dimension' implies that the channel dimension must "
                             "also have statistics computed over it. Thus norm_graded=True and norm_channel=False are "
                             "incompataible with each other.")
        if affine_channel and not affine_graded:
            raise ValueError("Having an affine transformation depend on the channel dimension implies that the affine "
                             "transformation must also depend on the graded 'dimension'. Thus affine_channel=True and "
                             "affine_graded=False are incompatible with each other.")
        if len(shape) not in (2, 3):
            raise ValueError("shape must have either two or three dimensions, corresponding to either "
                             "(batch, signature_channel) or (batch, stream, signature_channel).")
        if track_running_stats and not norm_batch:
            warnings.warn("Probably don't want to track running stats and not conglomerate statistics over the batch "
                          "dimension, as that means that a separate running statistic will be computed for each index "
                          "in the batch dimension, which is probably undesired.")
        if not track_running_stats and norm_batch:
            warnings.warn("Probably do want to track running stats when conglomerating statistics over the batch "
                          "dimension, as that means that the statistics are more accurate.")

        self.shape = shape
        self.channels = channels
        self.depth = depth
        self.norm_batch = norm_batch
        self.norm_stream = norm_stream
        self.norm_graded = norm_graded
        self.norm_channel = norm_channel
        self.affine_graded = affine_graded
        self.affine_channel = affine_channel
        self.affine = affine

        stream = len(shape) == 3

        if norm_graded == norm_channel:
            # Graded 'dimension' and channel dimension are treated the same, so we don't need to split them apart.
            dims = []
            if norm_batch:
                dims.append(0)
            if stream:
                if norm_stream:
                    dims.append(1)
                if norm_graded:  # == norm_channel
                    dims.append(2)
            else:
                if norm_graded:  # == norm_channel
                    dims.append(1)
            self._norm_layer = _Normalization(shape=shape, dims=dims, eps=eps, momentum=momentum,
                                              track_running_stats=track_running_stats)
        else:
            # Implies that norm_graded=False and norm_channel=True, as the reverse is disallowed.
            self._norm_layers = torch.nn.ModuleList()
            term_channels = 1
            dims = []
            if norm_batch:
                dims.append(0)
            if stream:
                if norm_stream:
                    dims.append(1)
                dims.append(2)
            else:
                dims.append(1)
            for _ in range(depth):
                term_channels *= channels
                if len(shape) == 3:
                    term_shape = shape[0], shape[1], term_channels
                else:
                    term_shape = shape[0], term_channels
                self._norm_layers.append(_Normalization(shape=term_shape, dims=dims, eps=eps, momentum=momentum,
                                                        track_running_stats=track_running_stats))

        if self.affine:
            if affine_graded == affine_channel:
                dims = []
                if stream:
                    if affine_graded:  # == affine_channel
                        dims.append(2)
                else:
                    if affine_graded:  # == affine_channel
                        dims.append(1)
                self._affine_layer = _Affine(shape=shape, dims=dims)
            else:
                # Implies that affine_graded=True and affine_channel=False, as the reverse is disallowed.
                self._affine_layers = torch.nn.ModuleList()
                for _ in range(depth):
                    # shape doesn't matter because this is just a scalar affine transformation over the whole lot
                    self._affine_layers.append(_Affine(shape=tuple(None for _ in shape), dims=()))

    def forward(self, input):
        if self.norm_graded == self.norm_channel:
            out = self._norm_layer(input)
        else:
            end = 0
            term_length = 1
            outs = []
            for norm_layer in self._norm_layers:
                start = end
                term_length *= self.channels
                end = start + term_length
                outs.append(norm_layer(input[:, start:end]))
            out = torch.cat(outs, dim=-1)

        if self.affine:
            if self.affine_graded == self.affine_channel:
                out = self._affine_layer(out)
            else:
                end = 0
                term_length = 1
                outs = []
                for affine_layer in self._affine_layers:
                    start = end
                    term_length *= self.channels
                    end = start + term_length
                    outs.append(affine_layer(out[:, start:end]))
                # TODO: not super efficient if we've just cat'd them together above, to split them apart and cat them
                #       again now.
                out = torch.cat(outs, dim=-1)

        return out


if __name__ == '__main__':
    import torch
    import signatory
    a = torch.randn(2, 100, 10)
    signature = [[signatory.signature(a, 2)]]
    normalisation = GradedNormalization(shape=signature[0][0].shape, channels=10, depth=2)
    normalisation.forward(signature)
