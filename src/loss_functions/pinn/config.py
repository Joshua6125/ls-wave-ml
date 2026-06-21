'''
Config for PINN loss.
'''
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax.numpy as jnp

from ...models import AnyModelConfig, MLPConfig
from ..base import AlgorithmConfig


@dataclass(frozen=True)
class PINNConfig(AlgorithmConfig):
    """Configuration for Physics-Informed Neural Network algorithm.

    Combines model architecture and PDE parameters into a single configuration.
    """

    kind: Literal["pinn"] = "pinn"

    # Neural network model used to represent unknown fields.
    model: AnyModelConfig = field(default_factory=MLPConfig)

    # Source term in the PDE system.
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0

    # Initial displacement
    u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0

    # Initial velocity
    ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0

    # Weight of the initial condition loss term.
    ic_weight: float = 1.0

    # Weight of the boundary condition loss term.
    bc_weight: float = 1.0
