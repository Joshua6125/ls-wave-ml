import pytest

import jax
import jax.numpy as jnp

from src.models import MLP, MLPConfig, KAN, KANConfig


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def sample_batch():
    # (batch, input_dim)
    return jnp.array(
        [
            [0.2, 0.3],
            [0.4, 0.7],
            [0.6, 0.5],
        ]
    )


@pytest.fixture
def sample_point():
    return jnp.array([0.5, 0.5])


@pytest.fixture
def mlp_model():
    return MLP(
        hidden_dim=16,
        num_layers=2,
        output_heads={
            "u": 1,
            "v": 2,
        },
        constrained_heads=["u"],
    )


@pytest.fixture
def kan_model():
    return KAN(
        hidden_dim=16,
        num_layers=2,
        input_dim=2,
        output_heads={
            "u": 1,
            "v": 2,
        },
        constrained_heads=["u"],
        model_type="efficient",
    )

@pytest.fixture
def mlp_config():
    return MLPConfig(
        hidden_dim=16,
        num_layers=2,
        output_heads={
            "u": 1,
            "v": 2,
        },
        constrained_heads=["u"],
    )


@pytest.fixture
def kan_config():
    return KANConfig(
        hidden_dim=16,
        num_layers=2,
        input_dim=2,
        output_heads={
            "u": 1,
            "v": 2,
        },
        constrained_heads=["u"],
        model_type="efficient",
    )