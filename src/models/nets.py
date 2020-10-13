"""
nets.py
========================
torch.nn modules that can be used as the classification method.
"""
import itertools as it
import torch
import torch.nn as nn
from src.features import window as window_module


class Model(nn.Module):
    """Abstract base class for all the post-signature models we consider. Each subclass should specify the window whose
    output data format they are compatible with.
    """
    _compatible_windows = ()

    @classmethod
    def assert_window_compatibility(cls, window):
        assert window in cls._compatible_windows


class NullConcatModel(Model):
    """ Returns concatenated signatures with no model application. Used to prepare for sklearn model. """
    _compatible_windows = (window_module.Global, window_module.Sliding, window_module.Expanding, window_module.Dyadic)

    def forward(self, signatures):
        return torch.cat([signature for signature_group in signatures for signature in signature_group], dim=1)


class Linear(Model):
    _compatible_windows = (window_module.Global, window_module.Sliding, window_module.Expanding, window_module.Dyadic)

    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, signatures):
        flattened_signatures = []
        for signature_group in signatures:
            for signature in signature_group:
                flattened_signatures.append(signature)
        x = torch.cat(flattened_signatures, dim=1)
        return self.linear(x)


class MLP(Model):
    """A feedforward neural network using ReLU nonlinearities."""
    _compatible_windows = (window_module.Global, window_module.Sliding, window_module.Expanding, window_module.Dyadic)

    def __init__(self, in_channels, out_channels, hidden_channels=(10, 10), normalization=True):
        """
        Arguments:
            in_channels: int. The number of input channels to this network.
            out_channels: int. The size of the output.
            hidden_channels: tuple of int. The size of each hidden layer.
            normalization: bool. Whether to perform normalization between each layer.
        """
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.normalization = normalization

        self.norm_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList((nn.Linear(in_channels, hidden_channels[0]),))
        prev_layer_size = hidden_channels[0]
        for layer_size in it.chain(hidden_channels[1:], [out_channels]):
            self.norm_layers.append(nn.BatchNorm1d(prev_layer_size))
            self.linear_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size

    def forward(self, signatures):
        flattened_signatures = []
        for signature_group in signatures:
            for signature in signature_group:
                flattened_signatures.append(signature)
        x = torch.cat(flattened_signatures, dim=1)
        x = self.linear_layers[0](x)
        for linear_layer, norm_layer in zip(self.linear_layers[1:], self.norm_layers):
            if self.normalization:
                x = norm_layer(x)
            x = torch.relu(x)
            x = linear_layer(x)
        return x


class CNN(Model):
    """Standard convolutional model. Convolutional base followed by feedforward network."""
    _compatible_windows = (window_module.Sliding, window_module.Expanding)

    def __init__(self, length, in_channels, out_channels, hidden_channels, kernel_sizes, mlp_hidden_channels,
                 normalization=True):
        """
        Arguments:
            length: int. Length of input sequences.
            in_channels: int. Number of input channels in input.
            hidden_channels: tuple of int. Number of channels in each hidden convolutional layer.
            kernel_sizes: tuple of int. Kernel size in each hidden convolutional layer.
            mlp_hidden_channels: tuple of int. Hidden layer sizes for final MLP.
            normalization: bool. Whether to use batch normalization.
        """
        super(CNN, self).__init__()

        assert len(hidden_channels) == len(kernel_sizes)

        self.length = length
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.mlp_hidden_channels = mlp_hidden_channels
        self.normalization = normalization

        self._mlp_in_channels = (length - sum(kernel_sizes) + len(kernel_sizes)) * hidden_channels[-1]

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        prev_channels = in_channels
        for hidden_channels_, kernel_size in zip(hidden_channels, kernel_sizes):
            self.conv_layers.append(nn.Conv1d(in_channels=prev_channels,
                                              out_channels=hidden_channels_,
                                              kernel_size=kernel_size))
            self.norm_layers.append(nn.BatchNorm1d(prev_channels))
            prev_channels = hidden_channels_
        self.mlp = MLP(in_channels=self._mlp_in_channels,
                       out_channels=out_channels,
                       hidden_channels=mlp_hidden_channels,
                       normalization=normalization)

    def forward(self, signatures):
        assert len(signatures) == 1
        signatures = signatures[0]
        # note that the convention here is (batch, channel, stream), rather than the (batch, stream, channel) used
        # elsewhere
        x = torch.stack(signatures, dim=2)
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            x = conv_layer(x)
            x = norm_layer(x)
            x = torch.relu(x)
        x = x.view(x.size(0), self._mlp_in_channels)
        return self.mlp(x)


class MLPResNet(Model):
    """A feedforward resnet using ReLU nonlinearities."""
    _compatible_windows = (window_module.Global, window_module.Sliding, window_module.Expanding, window_module.Dyadic)

    def __init__(self, in_channels, out_channels, residual_channels, block_channels, num_blocks):
        """
        Arguments:
            in_channels: int. The number of input channels to this network.
            out_channels: int. The number of output channels to make a linear transformation, right at the end.
            residual_channels: int. The number of channels to make a linear transformation to, right at the start.
            block_channels: The number of channels in each residual block.
            num_blocks: How many residual blocks. Each block has two affine transformations of width block_channels.

        Thus the architecture is:

        [Linear transform in_channels -> residual_channels]
                                |
                                |
                                +---------------------\
                                |                     |
                                |                [Batch norm]
                                |                     |
                                |                  [ReLU]
                                |                     |
                                |     [Linear transform residual_channels -> block_channels]
                                |                     |
                                |                [Batch norm]
                                |                     |
                                |                  [ReLU]
                                |                     |
                                |     [Linear transform block_channels -> residual_channels]
                                |                     |
                           [Addition]-----------------/
                                |
                                |

                                .
                                .  repeat for num_blocks blocks
                                .

                                |
        [Linear transform residual_channels -> out_channels]
        """

        super(MLPResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.block_channels = block_channels
        self.num_blocks = num_blocks

        self.first_affine = nn.Linear(in_channels, residual_channels)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(nn.BatchNorm1d(residual_channels),
                                  nn.ReLU(),
                                  nn.Linear(residual_channels, block_channels),
                                  nn.BatchNorm1d(block_channels),
                                  nn.ReLU(),
                                  nn.Linear(block_channels, residual_channels))
            self.blocks.append(block)
        self.last_affine = nn.Linear(residual_channels, out_channels)

    def forward(self, signatures):
        flattened_signatures = []
        for signature_group in signatures:
            for signature in signature_group:
                flattened_signatures.append(signature)
        x = torch.cat(flattened_signatures, dim=1)

        x = self.first_affine(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.last_affine(x)
        return x


class CNNResNet(Model):
    """A convolutional resnet using ReLU nonlinearities."""

    _compatible_windows = (window_module.Global, window_module.Sliding, window_module.Expanding, window_module.Dyadic)

    def __init__(self, length, in_channels, out_channels, residual_channels, block_channels, kernel_size, num_blocks,
                 feedforward_channels):
        """
        Arguments:
            length: int. The length of input sequences.
            in_channels: int. The number of input channels to this network.
            out_channels: int. The number of output channels to make a linear transformation, right at the end.
            residual_channels: int. The number of channels to make a Convolutional transformation to, at the start.
            block_channels: The number of channels in each residual block.
            kernel_size: The size of the kernel in each convolutional layer.
            num_blocks: How many residual blocks. Each block has two affine transformations of width block_channels.
            feedforward_channels: Size of hidden layer in final feedforward network.

        Thus the architecture is:

        [Convolutional transform in_channels -> residual_channels]
                                |
                                |
                                +---------------------\
                                |                     |
                                |                [Batch norm]
                                |                     |
                                |                  [ReLU]
                                |                     |
                                |     [Convolutional transform residual_channels -> block_channels]
                                |                     |
                                |                [Batch norm]
                                |                     |
                                |                  [ReLU]
                                |                     |
                                |     [Convolutional transform block_channels -> residual_channels]
                                |                     |
                           [Addition]-----------------/
                                |
                                |

                                .
                                .  repeat for num_blocks blocks
                                .

                                |
        [Linear transform residual_channels -> out_channels]
        """

        super(CNNResNet, self).__init__()

        self.length = length
        self.in_channels = in_channels
        self.residual_channels = residual_channels
        self.block_channels = block_channels
        self.num_blocks = num_blocks
        self.feedforward_channels = feedforward_channels
        self.out_channels = out_channels

        self.first_padding = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.first_conv = nn.Conv1d(in_channels=in_channels,
                                    out_channels=residual_channels,
                                    kernel_size=kernel_size)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(nn.BatchNorm1d(residual_channels),
                                  nn.ReLU(),
                                  nn.ConstantPad1d((kernel_size - 1, 0), 0),
                                  nn.Conv1d(in_channels=residual_channels,
                                            out_channels=block_channels,
                                            kernel_size=kernel_size),
                                  nn.BatchNorm1d(block_channels),
                                  nn.ReLU(),
                                  nn.ConstantPad1d((kernel_size - 1, 0), 0),
                                  nn.Conv1d(in_channels=block_channels,
                                            out_channels=residual_channels,
                                            kernel_size=kernel_size))
            self.blocks.append(block)

        self.final_affine_one = nn.Linear(length * residual_channels, feedforward_channels)
        self.final_affine_two = nn.Linear(feedforward_channels, out_channels)

    def forward(self, signatures):
        assert len(signatures) == 1
        signatures = signatures[0]
        # note that the convention here is (batch, channel, stream), rather than the (batch, stream, channel) used
        # elsewhere (because for some reason that's what the convolutions and paddings expect.)
        x = torch.stack(signatures, dim=2)
        x = self.first_padding(x)
        x = self.first_conv(x)
        for block in self.blocks:
            x = x + block(x)
        x = x.view(x.size(0), self.length * self.residual_channels)
        x = torch.relu(x)
        x = self.final_affine_one(x)
        x = torch.relu(x)
        x = self.final_affine_two(x)
        return x


class TCN(Model):
    """Standard temporal convolutional model. Convolutional base followed by feedforward network; convolutional layers
    use expanding dilations to get an exponential lookback."""
    _compatible_windows = (window_module.Sliding, window_module.Expanding)

    def __init__(self, length, in_channels, out_channels, residual_channels, block_channels, num_blocks,
                 feedforward_channels):
        """
        Arguments:
            length: int. Length of input sequences.
            in_channels: int. Number of input channels in input.
            out_channels: int. Number of output channels.
            residual_channels: int. Number of channels to transform to at the start.
            block_channels: int. Number of channels in dilated convolutions.
            num_blocks: int. Number of dilated convlutions, with skip connections, to take.
            feedforward_channels: int. Size of hidden layer in final feedforward network.

        Note that the WaveNet paper doesn't use normalization; the TCN paper uses only a funny kind of normalization for
        ReLUs but also states that the choice of doing so didn't affect performance. So we're not using normalization
        either.
        """
        super(TCN, self).__init__()

        self.length = length
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.block_channels = block_channels
        self.num_blocks = num_blocks
        self.feedforward_channels = feedforward_channels

        self.first_padding = nn.ConstantPad1d((3, 0), 0)
        self.first_conv = nn.Conv1d(in_channels=in_channels,
                                    out_channels=residual_channels,
                                    kernel_size=4)  # arbitrarily

        self.blocks = nn.ModuleList()
        dilation = 1
        for _ in range(num_blocks):
            block = nn.Sequential(nn.ConstantPad1d((dilation, 0), 0),
                                  nn.Conv1d(in_channels=residual_channels,
                                            out_channels=block_channels,
                                            kernel_size=2,
                                            dilation=dilation),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=block_channels,
                                            out_channels=residual_channels,
                                            kernel_size=1))
            self.blocks.append(block)
            dilation *= 2

        self.final_affine_one = nn.Linear(length * residual_channels, feedforward_channels)
        self.final_affine_two = nn.Linear(feedforward_channels, out_channels)

    def forward(self, signatures):
        assert len(signatures) == 1
        signatures = signatures[0]
        # note that the convention here is (batch, channel, stream), rather than the (batch, stream, channel) used
        # elsewhere (because for some reason that's what the convolutions and paddings expect.)
        x = torch.stack(signatures, dim=2)
        x = self.first_padding(x)
        x = self.first_conv(x)
        for block in self.blocks:
            x = x + block(x)
        x = x.view(x.size(0), self.length * self.residual_channels)
        x = torch.relu(x)
        x = self.final_affine_one(x)
        x = torch.relu(x)
        x = self.final_affine_two(x)
        return x


class RNN(Model):
    """An RNN model. Use as torch.nn.RNN."""
    _compatible_windows = (window_module.Global, window_module.Sliding, window_module.Expanding, window_module.Dyadic)

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, nonlinearity='tanh', bias=True,
                 dropout=0):
        super(RNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout

        self.rnn = nn.RNN(input_size=in_channels,
                          hidden_size=hidden_channels,
                          num_layers=num_layers,
                          nonlinearity=nonlinearity,
                          bias=bias,
                          dropout=dropout,
                          batch_first=True)
        self.total_hidden_size = num_layers * hidden_channels
        self.linear = nn.Linear(self.total_hidden_size, out_channels)

    def forward(self, signatures):
        assert len(signatures) == 1
        signatures = signatures[0]
        x = torch.stack(signatures, dim=1)
        hidden = self.rnn(x)[1]
        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(hidden.size(0), self.total_hidden_size)
        return self.linear(hidden)


class GRU(Model):
    """A GRU model. Use as torch.nn.GRU."""
    _compatible_windows = (window_module.Global, window_module.Sliding, window_module.Expanding, window_module.Dyadic)

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, bias=True, dropout=0):
        super(GRU, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout

        self.gru = nn.GRU(input_size=in_channels,
                          hidden_size=hidden_channels,
                          num_layers=num_layers,
                          bias=bias,
                          dropout=dropout,
                          batch_first=True)
        self.total_hidden_size = num_layers * hidden_channels
        self.linear = nn.Linear(self.total_hidden_size, out_channels)

    def forward(self, signatures):
        assert len(signatures) == 1
        signatures = signatures[0]
        x = torch.stack(signatures, dim=1)
        hidden = self.gru(x)[1]
        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(hidden.size(0), self.total_hidden_size)
        return self.linear(hidden)


class LSTM(Model):
    """An LSTM model. Use as torch.nn.LSTM."""
    _compatible_windows = (window_module.Global, window_module.Sliding, window_module.Expanding, window_module.Dyadic)

    def __init__(self, in_channels, out_channels, hidden_channels=10, num_layers=1, bias=True, dropout=0):
        super(LSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout

        self.gru = nn.LSTM(input_size=in_channels,
                           hidden_size=hidden_channels,
                           num_layers=num_layers,
                           bias=bias,
                           dropout=dropout,
                           batch_first=True)
        self.total_hidden_size = 2 * num_layers * hidden_channels
        self.linear = nn.Linear(self.total_hidden_size, out_channels)

    def forward(self, signatures):
        assert len(signatures) == 1
        signatures = signatures[0]
        x = torch.stack(signatures, dim=1)
        hidden = self.gru(x)[1]
        hidden = torch.cat(hidden, dim=2)
        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(hidden.size(0), self.total_hidden_size)
        return self.linear(hidden)


class Attention(Model):
    """A TransformerEncoder model."""
    _compatible_windows = (window_module.Sliding, window_module.Expanding)

    def __init__(self, in_channels, out_channels, residual_channels, num_heads, hidden_channels, num_layers, dropout=0,
                 activation='relu'):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(d_model=residual_channels,
                                                   nhead=num_heads,
                                                   dim_feedforward=hidden_channels,
                                                   dropout=dropout,
                                                   activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(residual_channels, out_channels)

    def forward(self, signatures):
        assert len(signatures) == 1
        signatures = signatures[0]
        x = torch.stack(signatures, dim=1)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x


class DyadicModelIndividual(Model):
    """A meta-model, which takes a single model, runs it on every individual dyadic interval, and then puts a small
    MLP on the top of the result.
    """
    _compatible_windows = (window_module.Dyadic,)

    def __init__(self, out_channels, dyadic_depth, hidden_channels, model):
        """
        Arguments:
            out_channels: int. How many channels in the output.
            dyadic_depth: int. What depth was used in the dyadic window.
            hidden_channels: int. How many hidden channels to have in the final single-hidden-layer network.
            model: One of the other models defined in this file. It should not have its out_channels argument set to the
                actual number of output channels; this should instead be treated as a hidden layer size.
        """
        super(DyadicModelIndividual, self).__init__()
        assert isinstance(model, Model)
        model.assert_window_compatibility(window_module.Global)

        self.out_channels = out_channels
        self.dyadic_depth = dyadic_depth
        self.hidden_channels = hidden_channels

        self.model = model
        num_dyadic_pieces = sum(2 ** i for i in range(dyadic_depth + 1))
        self.linear_one = nn.Linear(num_dyadic_pieces * self.model.out_channels, hidden_channels)
        self.linear_two = nn.Linear(hidden_channels, out_channels)

    def forward(self, signatures):
        xs = []
        for signature_group in signatures:
            for signature in signature_group:
                x = self.model([[signature]])
                xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.linear_one(x)
        x = torch.relu(x)
        x = self.linear_two(x)
        return x


class DyadicModelSequence(Model):
    """A meta-model, which takes a single model, runs it on every sequence of dyadic partitions, and then puts a small
    MLP on the top of the result.
    """
    _compatible_windows = (window_module.Dyadic,)

    def __init__(self, out_channels, dyadic_depth, hidden_channels, models):
        """
        Arguments:
            out_channels: int. How many channels in the output.
            dyadic_depth: int. What depth was used in the dyadic window.
            hidden_channels: int. How many hidden channels to have in the final single-hidden-layer network.
            models: list of the other models defined in this file. Its length should be the same as dyadic_depth. They
                should not have its out_channels argument set to the actual number of output channels; this should
                instead be treated as a hidden layer size.
        """
        super(DyadicModelSequence, self).__init__()
        for model in models:
            assert isinstance(model, Model)
            model.assert_window_compatibility(window_module.Sliding)

        self.out_channels = out_channels
        self.dyadic_depth = dyadic_depth
        self.hidden_channels = hidden_channels

        self.models = nn.ModuleList(models)
        self.linear_one = nn.Linear(dyadic_depth * self.model.out_channels, hidden_channels)
        self.linear_two = nn.Linear(hidden_channels, out_channels)

    def forward(self, signatures):
        xs = []
        for model, signature_group in zip(self.models, signatures):
            x = model([signature_group])
            torch.append(x)
        x = torch.cat(xs, dim=1)
        x = self.linear_one(x)
        x = torch.relu(x)
        x = self.linear_two(x)
        return x
