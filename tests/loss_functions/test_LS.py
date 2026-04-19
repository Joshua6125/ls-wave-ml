"""Tests for Least-Squares algorithm."""

import jax.numpy as jnp
import pytest

from src.loss_functions import LSConfig
from src.cli import run_training
from src.integration.config import QuadratureConfig
from src.models import AnyModelConfig, NeuralNetModelConfig
from src.train import TrainConfig


@pytest.mark.LS
def test_ls_training_smoke_runs_two_steps():
    integration_cfg = QuadratureConfig(dim=2, degree=3)

    model = NeuralNetModelConfig(
        hidden_dim=8,
        num_layers=2,
        output_heads={"v": 1, "sigma": 1},
    )
    algorithm_cfg = LSConfig(model=model)
    train_cfg = TrainConfig(epochs=2, log_every=1, use_jit=False)

    state, history = run_training(
        algorithm_cfg=algorithm_cfg,
        integration_cfg=integration_cfg,
        model_cfg=model,
        train_cfg=train_cfg,
        sample_input=jnp.zeros((2,)),
    )

    assert state.step == 2
    assert len(history) == 2
