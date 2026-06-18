"""Shared fixtures for training tests."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest

from src.integration import NDCubeIntegration
from src.loss_functions import FOSLS, FOSLSConfig, PINN, PINNConfig
from src.models import MLPConfig, build_model
from src.train import TrainConfig, TrainingMethod


class MockTrainingMethod(TrainingMethod):
    """Minimal differentiable method used for trainer unit tests."""

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        del rng_key, sample_input
        return {"w": jnp.array(0.0)}

    def loss_functions(self, params: Any):
        w = params["w"]

        def interior_loss(x: jnp.ndarray) -> jnp.ndarray:
            del x
            return jnp.ones((4,)) * (w - 1.0) ** 2

        def boundary_loss(x: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
            del x, normal
            return jnp.ones((4,)) * 0.5 * (w + 1.0) ** 2

        return interior_loss, boundary_loss


class MockConstantMethod(TrainingMethod):
    """Method with constant loss values for convergence and timeout tests."""

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        del rng_key, sample_input
        return {"w": jnp.array(0.0)}

    def loss_functions(self, params: Any):
        del params

        def interior_loss(x: jnp.ndarray) -> jnp.ndarray:
            del x
            return jnp.ones((4,))

        def boundary_loss(x: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
            del x, normal
            return jnp.ones((4,))

        return interior_loss, boundary_loss


class MockMalformedMethod(TrainingMethod):
    """Method with malformed loss_functions output for error-path tests."""

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        del rng_key, sample_input
        return {"w": jnp.array(0.0)}

    def loss_functions(self, params: Any):
        del params
        return (lambda x: x,)


class DeterministicTestIntegrator(NDCubeIntegration):
    """Simple deterministic integrator for trainer unit tests."""

    def __init__(self) -> None:
        self._interior_points = jnp.array(
            [[0.2, 0.1], [0.4, 0.3], [0.6, 0.5], [0.8, 0.7]]
        )
        self._boundary_points = jnp.array(
            [[0.0, 0.2], [0.0, 0.8], [0.5, 0.0], [0.5, 1.0]]
        )
        self._boundary_normals = jnp.array(
            [[-1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )

    def integrate_interior(self, func):
        values = func(self._interior_points)
        return jnp.mean(values)

    def integrate_boundary(self, func):
        values = func(self._boundary_points, self._boundary_normals)
        return jnp.mean(values)


@pytest.fixture
def sample_input_vector_1d():
    return jnp.array([0.5, 0.25])


@pytest.fixture
def sample_input_vector_2d():
    return jnp.array([0.5, 0.25, 0.75])


@pytest.fixture
def sample_input_vector_3d():
    return jnp.array([0.5, 0.25, 0.75, 0.1])


@pytest.fixture
def train_cfg_default():
    return TrainConfig(
        epochs=5,
        learning_rate=optax.constant_schedule(1e-2),
        optimiser="adam",
        seed=7,
        log_every=1,
        use_jit=False,
    )


@pytest.fixture
def train_cfg_short_jit():
    return TrainConfig(
        epochs=2,
        learning_rate=optax.constant_schedule(1e-2),
        optimiser="adamw",
        seed=3,
        log_every=1,
        use_jit=True,
    )


@pytest.fixture
def mock_training_method():
    return MockTrainingMethod()


@pytest.fixture
def mock_constant_method():
    return MockConstantMethod()


@pytest.fixture
def mock_malformed_method():
    return MockMalformedMethod()


@pytest.fixture
def deterministic_integrator():
    return DeterministicTestIntegrator()


@pytest.fixture
def callback_recorder():
    recorded = []

    def _callback(metrics, previous_state):
        recorded.append((metrics, previous_state))

    return recorded, _callback


@pytest.fixture
def real_pinn_method():
    cfg = PINNConfig(
        model=MLPConfig(hidden_dim=8, num_layers=2, output_heads={"u": 1}),
        f=0.0,
        u0=0.0,
        ut0=0.0,
        ic_weight=1.0,
        bc_weight=1.0,
    )
    model = build_model(cfg.model)
    return PINN(model=model, config=cfg)


@pytest.fixture
def real_fosls_method():
    cfg = FOSLSConfig(
        model=MLPConfig(hidden_dim=8, num_layers=2, output_heads={"v": 1, "sigma": 1}),
        f=0.0,
        g=0.0,
        v0=0.0,
        sigma0=0.0,
        ic_weight=1.0,
    )
    model = build_model(cfg.model)
    return FOSLS(model=model, config=cfg)
