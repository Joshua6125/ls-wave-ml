from typing import Callable

import jax.numpy as jnp

from .loss_functions import AnyLossConfig
from .integration import IntegrationConfig, get_integrator
from .models import AnyModelConfig, build_model
from .train.methods import get_training_method
from .train import (
    TrainConfig,
    TrainState,
    TrainStepMetrics,
    Trainer,
    get_optimizer,
)


def run_training(
        integration_cfg: IntegrationConfig,
        loss_cfg: AnyLossConfig,
        train_cfg: TrainConfig,
        model_cfg: AnyModelConfig,
        sample_input: jnp.ndarray | None = None,
        state: TrainState | None = None,
        callback: Callable[[TrainStepMetrics], None] | None = None,
    ) -> tuple[TrainState, list[TrainStepMetrics]]:
    """Execute training

    """
    integrator = get_integrator(integration_cfg)
    model = build_model(model_cfg)
    method = get_training_method(loss_cfg=loss_cfg, model_cfg=model_cfg, model=model)
    optimizer = get_optimizer(train_cfg)

    trainer = Trainer(
        method=method,
        integrator=integrator,
        optimizer=optimizer,
        train_cfg=train_cfg,
    )

    return trainer.fit(sample_input=sample_input, state=state, callback=callback)
