from typing import Mapping

import flax.linen as nn
import jax.numpy as jnp
from flax import nnx
from jaxkan.models.KAN import KAN as nnxKAN


class KAN(nn.Module):
    """Linen-compatible wrapper around jaxKAN's NNX KAN model.

    Parameters
    ----------
    hidden_dim : int
        Width of each hidden layer.
    num_layers : int
        Number of hidden layers.
    output_heads : Mapping[str, int]
        Named output heads.
    input_dim : int
        Dimension of the input layer.
    constrained_heads: list[str]
        The heads to ensure are smooth and zero on the spatial boundary.
    grid_size : int
        Grid size for spline-based KAN networks
    degree : int
        Degree of Chebyshev polynomials in Chebyshev-KAN
    model_type : str
        Specifies the type of model: "efficient" | "cheby" | "chebyshev" | "original" | "base" | "spline".
    seed : int
        The random seed used for KAN initialisation.
    """

    hidden_dim: int
    num_layers: int
    output_heads: Mapping[str, int]
    input_dim: int
    constrained_heads: list[str]
    grid_size: int = 5  # Used in "original", "base", and "spline"
    degree: int = 3  # Degree of chebyshev polynomials.
    model_type: str = (
        "efficient"  # aliases: "efficient" | "cheby" | "chebyshev" | "original" | "base" | "spline"
    )
    seed: int = 42

    def _layer_dims(self) -> list[int]:
        total_out_dim = sum(dim for _, dim in sorted(self.output_heads.items()))
        return [self.input_dim] + [self.hidden_dim] * self.num_layers + [total_out_dim]

    def setup(self):
        self.validate()
        layer_type, required_parameters = self._kan_hparams()
        self.kan = nnx.bridge.to_linen(
            nnxKAN,
            self._layer_dims(),
            layer_type=layer_type,
            required_parameters=required_parameters,
            seed=self.seed,
            skip_rng=True,
            name="kan_backbone",
        )

    @nn.compact
    def __call__(self, x) -> dict[str, jnp.ndarray]:
        self.validate()

        was_unbatched = x.ndim == 1
        x_in = x[None, :] if was_unbatched else x

        y = self.kan(x_in)

        outputs = self._split_output_heads(y)

        # If specified, then multiply with a smooth function that is zero on the spatial boundary.
        for head in outputs.keys():
            if head in self.constrained_heads:
                p = 2.0
                eps = 1e-12
                spatial_coords = x[..., 1:]

                a_left = jnp.clip(spatial_coords, eps, 1.0)
                a_right = jnp.clip(1.0 - spatial_coords, eps, 1.0)

                boundary_func = jnp.sum(
                    a_left ** (-p) + a_right ** (-p), axis=-1, keepdims=True
                ) ** (-1.0 / p)

                outputs[head] = boundary_func * outputs[head]

        if was_unbatched:
            return {name: value[0] for name, value in outputs.items()}

        return outputs

    def _split_output_heads(self, y: jnp.ndarray) -> dict[str, jnp.ndarray]:
        outputs: dict[str, jnp.ndarray] = {}
        start = 0
        for name, dim in sorted(self.output_heads.items()):
            outputs[name] = y[..., start : start + dim]
            start += dim
        return outputs

    def _kan_hparams(self) -> tuple[str, dict[str, object]]:
        model_type = self.model_type.lower()

        if model_type == "original":
            return "base", {"k": self.degree, "G": self.grid_size}

        if model_type in {"cheby", "chebychev"}:
            return "chebyshev", {"D": self.degree, "flavor": "default"}

        if model_type == "efficient":
            return "chebyshev", {"D": self.degree, "flavor": "exact"}

        if model_type in {"base", "spline"}:
            return model_type, {"k": self.degree, "G": self.grid_size}

        raise ValueError(
            "Unknown model_type. Supported values: "
            "efficient, cheby, original, base, spline, chebyshev"
        )
