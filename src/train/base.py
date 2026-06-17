from abc import ABC, abstractmethod
from typing import Any, Callable

import jax
import jax.numpy as jnp


class TrainingMethod(ABC):
    """
    Abstract interface for PDE training algorithms.

    A training method defines the full optimisation pipeline:

    1. Parameter initialisation for a given model and input shape
    2. Construction of loss function closures for fixed parameters
    3. Optional aggregation of structured loss outputs into a scalar objective

    Concrete implementations include PINNs, FOSLS, and related variational methods.
    """

    @abstractmethod
    def init_params(
        self,
        rng_key: jax.Array,
        sample_input: jnp.ndarray,
    ) -> Any:
        """
        Initialize model parameters and validate model compatibility.

        The implementation may enforce architecture-specific output contracts.

        Parameters
        ----------
        rng_key
            Random key for parameter initialisation.
        sample_input
            Example input used to infer model dimensions.

        Returns
        -------
        Any
            Model parameters (PyTree or backend-specific structure).
        """
        ...

    @abstractmethod
    def loss_functions(
        self,
        params: Any,
    ) -> tuple[
        Callable[[jnp.ndarray], jnp.ndarray],
        Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ]:
        """
        Construct loss function closures for fixed parameters.

        Returns
        -------
        interior_loss_fn
            Maps interior points -> pointwise or integrated loss contribution.

        boundary_loss_fn
            Maps boundary points and normals -> loss contribution.
        """
        ...

    def aggregate_loss(self, interior: Any, boundary: Any) -> jnp.ndarray:
        """
        Reduce structured interior and boundary losses into a scalar objective.

        Default behaviour:
        - Flattens PyTrees
        - Sums all scalar-valued contributions

        This can be overridden for methods requiring weighted or coupled losses.
        """

        def tree_sum(tree: Any) -> jnp.ndarray:
            leaves = jax.tree_util.tree_leaves(tree)

            if not leaves:
                return jnp.array(0.0)

            return sum(jnp.sum(leaf) for leaf in leaves)  # type: ignore

        return tree_sum(interior) + tree_sum(boundary)
