'''
Config for First-Order Systems Least-Squares loss.
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
class FOSLSConfig(AlgorithmConfig):
    """
    Configuration for a First-Order System Least Squares (FOSLS) method.

    The method rewrites a PDE as a first-order system and constructs a
    least-squares objective over residuals of the system variables.
    """

    kind: Literal["fosls"] = "fosls"

    # Neural network model used to represent unknown fields.
    model: AnyModelConfig = field(default_factory=MLPConfig)

    # Source term in the PDE system.
    f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0

    # Boundary/forcing terms for auxiliary variables in the first-order system.
    g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0

    # Initial condition for velocity variable.
    v0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0

    # Initial condition for flux/gradient variable.
    sigma0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0

    # Weight of the initial condition loss term.
    ic_weight: float = 1.0
