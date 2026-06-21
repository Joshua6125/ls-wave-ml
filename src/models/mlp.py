'''
Multi-Layer Perceptron model.
'''
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from typing import Mapping

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """
    Multi-layer perceptron with optional boundary-constrained output heads.

    The network produces multiple named outputs ("heads"). Certain heads
    can be forced to vanish smoothly at the spatial boundary by multiplying
    them with a positive barrier function.

    Architecture
    ------------
    The core network is a standard fully-connected MLP with `num_layers`
    hidden layers of width `hidden_dim` and tanh activations.

    Boundary constraint
    -------------------
    For selected output heads, the output is multiplied by a smooth function
    that approaches zero as any spatial coordinate approaches the boundary
    of [0, 1]^d. This enforces Dirichlet-type boundary behaviour.
    """

    hidden_dim: int
    num_layers: int
    output_heads: Mapping[str, int]
    constrained_heads: list[str]

    @nn.compact
    def __call__(self, x) -> dict[str, jnp.ndarray]:
        h = x

        # Standard fully-connected trunk
        for _ in range(self.num_layers):
            h = jnp.tanh(nn.Dense(self.hidden_dim)(h))

        # Independent linear heads for each output field
        output = {
            name: nn.Dense(dim, name=name)(h)
            for name, dim in sorted(self.output_heads.items())
        }

        # Apply boundary vanishing constraint to selected heads
        for head in self.constrained_heads:
            if head not in output:
                continue

            # Spatial coordinates exclude time (assumed x[..., 0])
            spatial_coords = x[..., 1:]

            # Barrier function: small near 0 or 1 in any coordinate.
            # Constructed via smooth p-norm of inverse distances.
            eps = 1e-12
            p = 2.0

            a_left = jnp.clip(spatial_coords, eps, 1.0)
            a_right = jnp.clip(1.0 - spatial_coords, eps, 1.0)

            # Smooth approximation of min-distance-to-boundary.
            boundary_func = jnp.sum(
                a_left ** (-p) + a_right ** (-p), axis=-1, keepdims=True
            ) ** (-1.0 / p)

            # Enforce vanishing boundary condition
            output[head] = boundary_func * output[head]

        return output
