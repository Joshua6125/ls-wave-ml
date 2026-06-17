import jax
import jax.numpy as jnp

from ...models import BuiltModelProtocol
from ...train import TrainingMethod
from .loss import FOSLSLoss
from .config import FOSLSConfig


class FOSLS(TrainingMethod):
    """
    Training wrapper for First-Order System Least Squares (FOSLS).

    Responsibilities
    -----------------
    1. Initialize model parameters via the backend (Flax/NX).
    2. Validate that the model output satisfies the required PDE field structure.
    3. Construct a closure-based loss function tied to current parameters.
    """

    def __init__(
        self,
        model: BuiltModelProtocol,
        config: FOSLSConfig,
    ):
        self.model = model
        self.config = config

    def init_params(
        self,
        rng_key: jax.Array,
        sample_input: jnp.ndarray,
    ):
        """
        Initialize model parameters and enforce output contract.

        The model must return a dictionary containing:
            - v : scalar field
            - sigma : vector field
        """
        params = self.model.init(rng_key, sample_input)

        outputs = self.model.apply(params, sample_input)

        if not isinstance(outputs, dict):
            raise ValueError("Model must return a dict with keys 'v' and 'sigma'")

        if "v" not in outputs or "sigma" not in outputs:
            raise ValueError("Missing required output heads: 'v', 'sigma'")

        v_sample = outputs["v"]
        sigma_sample = outputs["sigma"]

        # v must be scalar-valued per sample.
        if jnp.asarray(v_sample).reshape(-1).shape[0] != 1:
            raise ValueError("Output 'v' must be scalar per input point")

        # Vector field dimension must match spatial dimension.
        expected_sigma_dim = max(sample_input.shape[-1] - 1, 1)

        if jnp.asarray(sigma_sample).reshape(-1).shape[0] != expected_sigma_dim:
            raise ValueError(
                "Output 'sigma' has incorrect dimension: "
                f"expected {expected_sigma_dim}"
            )

        return params

    def loss_functions(self, params):
        """
        Construct FOSLS loss functions for fixed model parameters.

        Returns a callable loss object that evaluates:
            - interior PDE residual
            - boundary / initial condition residual
        """

        def fosls_apply(x: jnp.ndarray) -> jnp.ndarray:
            """
            Flattened model interface used by the PDE loss.

            Converts structured dict output into a single vector:
                [v, sigma]
            """
            out = self.model.apply(params, x)

            return jnp.concatenate(
                [
                    jnp.atleast_1d(out["v"]),
                    jnp.atleast_1d(out["sigma"]),
                ],
                axis=-1,
            )

        loss = FOSLSLoss(
            model=fosls_apply,
            f=self.config.f,
            g=self.config.g,
            v0=self.config.v0,
            sigma0=self.config.sigma0,
            ic_weight=self.config.ic_weight,
        )

        return loss.loss_functions()
