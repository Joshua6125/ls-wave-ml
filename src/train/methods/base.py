from abc import ABC, abstractmethod
from typing import Callable, Any

import jax
import jax.numpy as jnp


class TrainingMethod(ABC):
    """Base interface for all method-specific training plugins."""

    @abstractmethod
    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        """Initialize model parameters for this method."""
        ...

    @abstractmethod
    def loss_functions(
            self,
            params: Any
        ) -> tuple[
            Callable[[jnp.ndarray], jnp.ndarray],
            Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ]:
        """Return integrands for interior and boundary losses."""
        ...
