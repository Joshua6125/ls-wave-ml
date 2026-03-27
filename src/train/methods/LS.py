from typing import Any

import jax
import jax.numpy as jnp

from ...loss_functions import LSLossConfig, LossLS
from .base import TrainingMethod


class LSMethod(TrainingMethod):
    """Method plugin for least-squares wave-system training."""

    def __init__(
            self,
            ls_model: Any,
            loss_cfg: LSLossConfig
        ):
        self.ls_model = ls_model
        self.loss_cfg = loss_cfg

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray):
        """Initialize parameters for the shared multi-head LS model."""
        params = self.ls_model.init(rng_key, sample_input)

        # Fail early if model output contract does not match LS expectations.
        outputs = self.ls_model.apply(params, sample_input)
        if not isinstance(outputs, dict):
            raise ValueError("LS model must return named heads as a dict with keys 'v' and 'sigma'.")
        if "v" not in outputs or "sigma" not in outputs:
            raise ValueError("LS model outputs must include heads named 'v' and 'sigma'.")

        v_sample = outputs["v"]
        sigma_sample = outputs["sigma"]

        if jnp.asarray(v_sample).reshape(-1).shape[0] != 1:
            raise ValueError("LS model v head must output a scalar (shape [1]).")

        expected_sigma_dim = max(sample_input.shape[0] - 1, 1)
        if jnp.asarray(sigma_sample).reshape(-1).shape[0] != expected_sigma_dim:
            raise ValueError(
                f"LS model sigma head must output {expected_sigma_dim} values for input shape "
                f"{tuple(sample_input.shape)}."
            )

        return params

    def loss_functions(self, params: Any):
        """Create interior and boundary integrands for the current parameters."""

        def ls_apply(x: jnp.ndarray) -> dict[str, jnp.ndarray]:
            return self.ls_model.apply(params, x)

        def v_apply(x: jnp.ndarray) -> jnp.ndarray:
            return ls_apply(x)["v"]

        def sigma_apply(x: jnp.ndarray) -> jnp.ndarray:
            return ls_apply(x)["sigma"]

        loss = LossLS(
            v_model=v_apply,
            sigma_model=sigma_apply,
            f=self.loss_cfg.f,
            g=self.loss_cfg.g,
            v0=self.loss_cfg.v0,
            sigma0=self.loss_cfg.sigma0,
        )

        return loss.loss_functions()
