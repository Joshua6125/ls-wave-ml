from typing import Mapping

import flax.linen as nn
import jax.numpy as jnp
from flax import nnx
from jaxkan.models.KAN import KAN as nnxKAN


class KAN(nn.Module):
    """
    Flax-compatible wrapper around a jaxKAN NNX Kolmogorov-Arnold Network.

    This module serves three purposes:

    1. Backbone model
       A configurable KAN architecture constructed via the jaxKAN library,
       supporting multiple spline/chebyshev variants.

    2. Multi-head output structure
       The final KAN output is partitioned into named prediction heads.

    3. Boundary-constrained outputs
       Selected heads are multiplied by a smooth barrier function that
       enforces vanishing values on the spatial boundary of [0, 1]^d.

    Notes
    -----
    The model assumes the first input coordinate corresponds to time,
    and spatial coordinates are x[..., 1:].
    """

    hidden_dim: int
    num_layers: int
    output_heads: Mapping[str, int]
    input_dim: int

    # Heads to enforce Dirichlet-style boundary conditions.
    constrained_heads: list[str]

    # KAN hyperparameters (used depending on model_type).
    grid_size: int = 5
    degree: int = 3

    # Backend variant selector for jaxKAN.
    model_type: str = "efficient"

    seed: int = 42

    def _layer_dims(self) -> list[int]:
        """Construct MLP-style layer widths used by the KAN backbone."""
        total_out_dim = sum(dim for _, dim in sorted(self.output_heads.items()))
        return [self.input_dim] + [self.hidden_dim] * self.num_layers + [total_out_dim]

    def setup(self):
        """
        Initialize the underlying jaxKAN model via the Flax-NNX bridge.
        """
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
        """
        Forward pass returning a dictionary of named output heads.
        """

        # Allow single-sample inputs (shape [dim]) in addition to batches.
        was_unbatched = x.ndim == 1
        x_in = x[None, :] if was_unbatched else x

        y = self.kan(x_in)
        outputs = self._split_output_heads(y)

        # Enforce boundary vanishing condition on selected heads.
        for head in self.constrained_heads:
            if head not in outputs:
                continue

            spatial_coords = x[..., 1:]

            eps = 1e-12
            p = 2.0

            # Smooth barrier: approaches zero when any coordinate approaches 0 or 1.
            a_left = jnp.clip(spatial_coords, eps, 1.0)
            a_right = jnp.clip(1.0 - spatial_coords, eps, 1.0)

            boundary_func = jnp.sum(
                a_left ** (-p) + a_right ** (-p),
                axis=-1,
                keepdims=True,
            ) ** (-1.0 / p)

            outputs[head] = boundary_func * outputs[head]

        # Restore original input shape if needed.
        if was_unbatched:
            return {name: value[0] for name, value in outputs.items()}

        return outputs

    def _split_output_heads(self, y: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """
        Split concatenated KAN output into named heads.
        """
        outputs: dict[str, jnp.ndarray] = {}
        start = 0

        for name, dim in sorted(self.output_heads.items()):
            outputs[name] = y[..., start : start + dim]
            start += dim

        return outputs

    def _kan_hparams(self) -> tuple[str, dict[str, object]]:
        """
        Map high-level model_type to jaxKAN backend configuration.
        """
        model_type = self.model_type.lower()

        if model_type == "original":
            return "base", {"k": self.degree, "G": self.grid_size}

        if model_type in {"cheby", "chebyshev"}:
            return "chebyshev", {"D": self.degree, "flavor": "default"}

        if model_type == "efficient":
            return "chebyshev", {"D": self.degree, "flavor": "exact"}

        if model_type in {"base", "spline"}:
            return model_type, {"k": self.degree, "G": self.grid_size}

        raise ValueError(
            "Unknown model_type. Supported values: "
            "efficient, cheby, original, base, spline, chebyshev"
        )
