'''
Config for variational PINN loss.
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
class vPINNConfig(AlgorithmConfig):
    """
    Configuration for Variational PINN algorithm.

    The method tests the neural model against a set of test
    functions.
    """

    kind: Literal["vpinn"] = "vpinn"

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

    # Amount of test functions to use
    n_test_functions: int = 400

    # Boundary minima/maxima
    domain_min: jnp.ndarray | None = None
    domain_max: jnp.ndarray | None = None
