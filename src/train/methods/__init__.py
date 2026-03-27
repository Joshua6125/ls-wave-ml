from ...loss_functions import LSLossConfig, PINNLossConfig
from ...models import AnyBuiltModel, LSModelConfig, PINNModelConfig
from .LS import LSMethod
from .PINN import PINNMethod
from .base import TrainingMethod


def get_training_method(
        loss_cfg: LSLossConfig | PINNLossConfig,
        model_cfg: PINNModelConfig | LSModelConfig,
        model: AnyBuiltModel,
    ) -> TrainingMethod:
    """Factory function for choosing training method plugin."""

    if isinstance(loss_cfg, PINNLossConfig):
        if not isinstance(model_cfg, PINNModelConfig):
            raise ValueError("PINN loss config requires PINN model config.")
        return PINNMethod(u_model=model, loss_cfg=loss_cfg)

    if isinstance(loss_cfg, LSLossConfig):
        if not isinstance(model_cfg, LSModelConfig):
            raise ValueError("LS loss config requires LS model config.")
        return LSMethod(
            ls_model=model,
            loss_cfg=loss_cfg,
        )

    raise ValueError("Unknown loss configuration type.")


__all__ = ["TrainingMethod", "PINNMethod", "LSMethod", "get_training_method"]
