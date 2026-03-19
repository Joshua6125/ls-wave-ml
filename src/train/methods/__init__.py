from typing import Any

from ...loss_functions import LSLossConfig, PINNLossConfig
from .LS import LSMethod
from .PINN import PINNMethod
from .base import TrainingMethod


def get_training_method(
        loss_cfg: LSLossConfig | PINNLossConfig,
        model_bundle: dict[str, Any]
    ) -> TrainingMethod:
    """Factory function for choosing training method plugin."""

    if isinstance(loss_cfg, PINNLossConfig):
        if "u_model" not in model_bundle:
            raise ValueError("PINN method requires 'u_model' in model_bundle.")
        return PINNMethod(u_model=model_bundle["u_model"], loss_cfg=loss_cfg)

    if isinstance(loss_cfg, LSLossConfig):
        if "v_model" not in model_bundle or "sigma_model" not in model_bundle:
            raise ValueError("LS method requires 'v_model' and 'sigma_model' in model_bundle.")
        return LSMethod(
            v_model=model_bundle["v_model"],
            sigma_model=model_bundle["sigma_model"],
            loss_cfg=loss_cfg,
        )

    raise ValueError("Unknown loss configuration type.")


__all__ = ["TrainingMethod", "PINNMethod", "LSMethod", "get_training_method"]
