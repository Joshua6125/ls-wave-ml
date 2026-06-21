'''
Config for gradient PINN loss.
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
class gPINNConfig(AlgorithmConfig):
    """
    Configuration for gradient-enhanced PINN (gPINN).

    The method takes regular PINN and adds weights for penalising
    the gradient of the PDE residual and the solution gradient.
    """

    kind: Literal["gpinn"] = "gpinn"

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

    # Weight of the gradient condition loss term.
    residual_grad_weight: float = 0.0
