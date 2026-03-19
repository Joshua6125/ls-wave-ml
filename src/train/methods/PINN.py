from typing import Any

import jax
import jax.numpy as jnp

from ...loss_functions import LossPINN, PINNLossConfig
from .base import TrainingMethod

class PINNMethod(TrainingMethod):
    """Method plugin for PINN training."""

    def __init__(
            self,
            u_model: Any,
            loss_cfg: PINNLossConfig
        ):
        self.u_model = u_model
        self.loss_cfg = loss_cfg

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        """Initialize PINN model parameters."""
        return self.u_model.init(rng_key, sample_input)

    def loss_functions(self, params: Any):
        """Create interior and boundary integrands for the current parameters."""

        def u_apply(x: jnp.ndarray) -> jnp.ndarray:
            return self.u_model.apply(params, x)

        loss = LossPINN(
            u_model=u_apply,
            c=self.loss_cfg.c,
            f=self.loss_cfg.f,
            u0=self.loss_cfg.u0,
            ut0=self.loss_cfg.ut0,
            ic_weight=self.loss_cfg.ic_weight,
            bc_weight=self.loss_cfg.bc_weight,
        )

        return loss.loss_functions()
