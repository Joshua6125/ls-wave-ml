from typing import Any
import jax
import jax.numpy as jnp

from ...models import BuiltModelProtocol
from ...train import TrainingMethod
from .loss import vPINNLoss
from .config import vPINNConfig


class vPINN(TrainingMethod):
    """
    Training wrapper for Variational PINN (vPINN)

    Responsibilities
    -----------------
    1. Initialize model parameters via the backend (Flax/NX).
    2. Validate that the model output satisfies the required PDE field structure.
    3. Construct a closure-based loss function tied to current parameters.
    4. Define the aggregate loss function.
    """

    def __init__(self, model: BuiltModelProtocol, config: vPINNConfig):
        self.model = model
        self.config = config

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        """
        Initialize model parameters and enforce output contract.

        The model must return a dictionary containing:
            - u : displacement
        """
        params = self.model.init(rng_key, sample_input)

        outputs = self.model.apply(params, sample_input)

        if not isinstance(outputs, dict) or "u" not in outputs:
            raise ValueError("vPINN model must return dict with 'u' key (scalar)")

        # u must be scalar-valued per sample.
        if jnp.asarray(outputs["u"]).reshape(-1).shape[0] != 1:
            raise ValueError("vPINN model 'u' output must be scalar")

        return params

    def loss_functions(self, params):
        """
        Construct vPINN loss functions for fixed model parameters.

        Returns a callable loss object that evaluates:
            - interior PDE residual
            - boundary / initial condition residual
        """

        def u_apply(x: jnp.ndarray) -> jnp.ndarray:
            """
            Flattened model interface used by the PDE loss.

            Converts structured dict output into a single value [u]
            """
            return self.model.apply(params, x)["u"]

        loss = vPINNLoss(
            u_model=u_apply,
            f=self.config.f,
            u0=self.config.u0,
            ut0=self.config.ut0,
            ic_weight=self.config.ic_weight,
            bc_weight=self.config.bc_weight,
            n_test_functions=self.config.n_test_functions,
            domain_min=self.config.domain_min,
            domain_max=self.config.domain_max,
        )
        return loss.loss_functions()

    def aggregate_loss(self, interior: Any, boundary: Any) -> jnp.ndarray:
        """For vPINN, the interior loss is a vector of integral evaluations.
        We square and sum them here to form the final variance objective."""

        # interior is expected to be a PyTree containing the evaluated integrals.
        # boundary is expected to evaluate to scalars directly (squared residuals).

        def square_and_sum(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x**2)

        interior_loss_total = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_map(square_and_sum, interior), 0.0
        )

        def tree_sum(tree: Any) -> jnp.ndarray:
            leaves = jax.tree_util.tree_leaves(tree)
            if not leaves:
                return jnp.array(0.0)
            return sum(jnp.sum(leaf) for leaf in leaves)  # type: ignore

        boundary_loss_total = tree_sum(boundary)

        return interior_loss_total + boundary_loss_total
