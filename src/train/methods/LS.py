from typing import Any

import jax
import jax.numpy as jnp

from ...loss_functions import LSLossConfig, LossLS
from .base import TrainingMethod


class LSMethod(TrainingMethod):
    """Method plugin for least-squares wave-system training."""

    def __init__(
            self,
            v_model: Any,
            sigma_model: Any,
            loss_cfg: LSLossConfig
        ):
        self.v_model = v_model
        self.sigma_model = sigma_model
        self.loss_cfg = loss_cfg

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        """Initialize parameter trees for both LS models."""
        k1, k2 = jax.random.split(rng_key)
        params_v = self.v_model.init(k1, sample_input)
        params_sigma = self.sigma_model.init(k2, sample_input)

        return {
            "v": params_v,
            "sigma": params_sigma,
        }

    def loss_functions(self, params: Any):
        """Create interior and boundary integrands for the current parameters."""
        if "v" not in params or "sigma" not in params:
            raise ValueError("LS params must contain 'v' and 'sigma' entries.")

        def v_apply(x: jnp.ndarray) -> jnp.ndarray:
            return self.v_model.apply(params["v"], x)

        def sigma_apply(x: jnp.ndarray) -> jnp.ndarray:
            return self.sigma_model.apply(params["sigma"], x)

        loss = LossLS(
            v_model=v_apply,
            sigma_model=sigma_apply,
            f=self.loss_cfg.f,
            g=self.loss_cfg.g,
            v0=self.loss_cfg.v0,
            sigma0=self.loss_cfg.sigma0,
        )

        return loss.loss_functions()
