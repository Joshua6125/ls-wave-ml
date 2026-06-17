import jax
import jax.numpy as jnp

from ...models import BuiltModelProtocol
from ...train import TrainingMethod
from .loss import PINNLoss
from .config import PINNConfig


class PINN(TrainingMethod):
    """
    Training wrapper for Physics-Informed Neural Network (PINN)

    Responsibilities
    -----------------
    1. Initialize model parameters via the backend (Flax/NX).
    2. Validate that the model output satisfies the required PDE field structure.
    3. Construct a closure-based loss function tied to current parameters.
    """

    def __init__(self, model: BuiltModelProtocol, config: PINNConfig):
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
            raise ValueError("PINN model must return dict with 'u' key (scalar)")

        # u must be scalar-valued per sample.
        if jnp.asarray(outputs["u"]).reshape(-1).shape[0] != 1:
            raise ValueError("PINN model 'u' output must be scalar")

        return params

    def loss_functions(self, params):
        """
        Construct PINN loss functions for fixed model parameters.

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

        loss = PINNLoss(
            u_model=u_apply,
            f=self.config.f,
            u0=self.config.u0,
            ut0=self.config.ut0,
            ic_weight=self.config.ic_weight,
            bc_weight=self.config.bc_weight,
        )
        return loss.loss_functions()
