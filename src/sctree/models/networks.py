"""
Encoder, decoder, transformation, router, and dense layer architectures.
"""

import collections
from typing import Iterable, Literal, Optional

import scvi.nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.nn import FCLayers
from scvi.module._utils import one_hot


def actvn(x):
    return F.leaky_relu(x, negative_slope=0.3)


class BatchDecoder(nn.Module):
    def __init__(self, n_cat_list: Iterable[int], n_out: int):
        super().__init__()
        self.n_cat_list = n_cat_list
        self.n_out = n_out

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)

        self.decoder = nn.Linear(cat_dim, n_out)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def thaw(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.decoder(x)


class TreeFCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        batch_decoder: Optional[BatchDecoder] = None,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        self.batch_decoder = batch_decoder
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        if isinstance(self.batch_decoder, int):
            cat_dim = self.batch_decoder
        elif self.batch_decoder is not None:
            cat_dim = 0
        else:
            cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            (
                                nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                                if use_batch_norm
                                else None
                            ),
                            (
                                nn.LayerNorm(n_out, elementwise_affine=False)
                                if use_layer_norm
                                else None
                            ),
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        return layer_num == 0

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if self.batch_decoder is None:
                                if x.dim() == 3:
                                    one_hot_cat_list_layer = [
                                        o.unsqueeze(0).expand(
                                            (x.size(0), o.size(0), o.size(1))
                                        )
                                        for o in one_hot_cat_list
                                    ]
                                else:
                                    one_hot_cat_list_layer = one_hot_cat_list
                                x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                                x = layer(x)

                            else:
                                if (one_hot_cat_list == []) or isinstance(
                                    self.batch_decoder, int
                                ):  # TODO: Make sure this fix is proper and repeat for other if's
                                    batch_offset = 0
                                else:
                                    batch_offset = self.batch_decoder(
                                        torch.cat(one_hot_cat_list, dim=-1)
                                    )
                                x = layer(x) + batch_offset

        return x


class DecoderTree(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        px_r,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        batch_decoder: Optional[BatchDecoder] = None,
        **kwargs,
    ):

        super().__init__()
        self.px_decoder = TreeFCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            batch_decoder=batch_decoder,
            **kwargs,
        )

        # mean gamma
        px_scale_activation = nn.Softmax(dim=-1)
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        if px_r is None:
            self.px_r = torch.nn.Parameter(torch.randn(n_output))
        else:
            self.px_r = px_r
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
    ):

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r
        return px_scale, px_r, px_rate, px_dropout


class LinearDecoderTree(nn.Module):
    """Linear decoder for scVI."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        px_r,
        n_cat_list: Iterable[int] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        bias: bool = False,
        batch_decoder: Optional[BatchDecoder] = None,
        **kwargs,
    ):
        super().__init__()
        # mean gamma
        self.factor_regressor = TreeFCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            batch_decoder=batch_decoder,
            **kwargs,
        )

        # dropout
        self.px_dropout_decoder = TreeFCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            batch_decoder=batch_decoder,
            **kwargs,
        )

        if px_r is None:
            self.px_r = torch.nn.Parameter(torch.randn(n_output))
        else:
            self.px_r = px_r

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        """Forward pass."""
        # The decoder returns values for the parameters of the ZINB distribution
        raw_px_scale = self.factor_regressor(z, *cat_list)
        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout_decoder(z, *cat_list)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r
        return px_scale, px_r, px_rate, px_dropout


class ScviEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        n_cat_list,
        n_layers=1,
        n_hidden=128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = FCLayers(
            n_in=input_shape,
            n_out=output_shape,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

    def forward(self, inputs, *cat_list):
        return self.encoder(inputs, *cat_list), None, None


class EncoderSmall(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EncoderSmall, self).__init__()

        self.dense1 = nn.Linear(
            in_features=input_shape, out_features=output_shape, bias=False
        )
        self.bn1 = nn.BatchNorm1d(output_shape)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = actvn(x)
        return x, None, None


class EncoderSmall2(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EncoderSmall2, self).__init__()

        self.dense1 = nn.Linear(
            in_features=input_shape, out_features=output_shape, bias=False
        )
        self.bn1 = nn.BatchNorm1d(output_shape)
        self.dense2 = nn.Linear(
            in_features=output_shape, out_features=output_shape, bias=False
        )
        self.bn2 = nn.BatchNorm1d(output_shape)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = actvn(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = actvn(x)
        return x, None, None


class EncoderSmall3(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EncoderSmall3, self).__init__()

        self.dense1 = nn.Linear(
            in_features=input_shape, out_features=output_shape, bias=False
        )
        self.bn1 = nn.BatchNorm1d(output_shape)
        self.dense2 = nn.Linear(
            in_features=output_shape, out_features=output_shape, bias=False
        )
        self.bn2 = nn.BatchNorm1d(output_shape)
        self.dense3 = nn.Linear(
            in_features=output_shape, out_features=output_shape, bias=False
        )
        self.bn3 = nn.BatchNorm1d(output_shape)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = actvn(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = actvn(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = actvn(x)
        return x, None, None


# Small branch transformation
class MLP(nn.Module):
    def __init__(self, input_size, encoded_size, hidden_unit, n_layers=1):
        super(MLP, self).__init__()
        hidden_unit = hidden_unit if n_layers > 0 else input_size
        self.mu = nn.Linear(hidden_unit, encoded_size)
        self.sigma = nn.Linear(hidden_unit, encoded_size)
        layer_dims = [input_size] + (n_layers) * [hidden_unit]
        if n_layers == 0:
            self.network = nn.Sequential(
                nn.BatchNorm1d(hidden_unit), nn.LeakyReLU(negative_slope=0.3)
            )
        else:
            self.network = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(input_dim, output_dim, bias=False),
                        nn.BatchNorm1d(hidden_unit),
                        nn.LeakyReLU(negative_slope=0.3),
                    )
                    for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:])
                ]
            )

    def forward(self, inputs):
        x = self.network(inputs)
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        return x, mu, sigma


class Dense(nn.Module):
    def __init__(self, input_size, encoded_size):
        super(Dense, self).__init__()
        self.mu = nn.Linear(input_size, encoded_size)
        self.sigma = nn.Linear(input_size, encoded_size)

    def forward(self, inputs):
        x = inputs
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        return mu, sigma


class Router(nn.Module):
    def __init__(
        self, input_size, hidden_units=128, n_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        layer_dims = [input_size] + (n_layers) * [hidden_units]
        hidden_units = hidden_units if n_layers > 0 else input_size
        if n_layers == 0:
            self.network = nn.Sequential(
                nn.BatchNorm1d(hidden_units),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Dropout(p=dropout),
            )
        else:
            self.network = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(input_dim, output_dim, bias=False),
                        nn.BatchNorm1d(hidden_units),
                        nn.LeakyReLU(negative_slope=0.3),
                        nn.Dropout(p=dropout),
                    )
                    for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:])
                ]
            )

        self.dense3 = nn.Linear(hidden_units, 1)

    def forward(self, inputs, return_last_layer=False):
        x = self.network(inputs)
        d = F.sigmoid(self.dense3(x))
        if return_last_layer:
            return d, x
        else:
            return d


def get_encoder(architecture, encoded_size, x_shape, n_cat_list=None):
    if architecture == "mlp":
        encoder = EncoderSmall(input_shape=x_shape, output_shape=encoded_size)
    elif architecture == "mlp2":
        encoder = EncoderSmall2(input_shape=x_shape, output_shape=encoded_size)
    elif architecture == "mlp3":
        encoder = EncoderSmall3(input_shape=x_shape, output_shape=encoded_size)
    elif architecture == "scvi":
        encoder = ScviEncoder(x_shape, encoded_size, n_cat_list, n_layers=1)
    elif architecture == "scvi2":
        encoder = ScviEncoder(x_shape, encoded_size, n_cat_list, n_layers=2)
    elif architecture == "scvi3":
        encoder = ScviEncoder(x_shape, encoded_size, n_cat_list, n_layers=3)
    else:
        raise ValueError(f"The encoder architecture {architecture} is mispecified.")
    return encoder
