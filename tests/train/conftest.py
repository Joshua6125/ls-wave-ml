"""Shared fixtures for training module tests."""
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from typing import Any, Callable
import jax
import jax.numpy as jnp
import pytest
import optax

from src.train import TrainConfig, TrainState
from src.train import TrainingMethod


class DummyMethod(TrainingMethod):
    """A minimal mock implementation of TrainingMethod for JAX transformations."""

    def init_params(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        return {"w": jnp.array([1.0, 2.0])}

    def loss_functions(self, params: Any) -> tuple[Callable, Callable]:
        def interior_loss_fn(x):
            return jnp.sum(x * params["w"])

        def boundary_loss_fn(x, normal):
            return jnp.sum(x)

        return interior_loss_fn, boundary_loss_fn


class DummyIntegrator:
    """A minimal mock implementation of NDCubeIntegration."""

    def integrate(
        self, interior_fn: Callable, boundary_fn: Callable, integration_key: jax.Array
    ):
        # Simulate evaluation points
        x_int = jnp.array([[0.5, 0.5]])
        x_bnd = jnp.array([[0.0, 0.5]])
        normal = jnp.array([[-1.0, 0.0]])

        return interior_fn(x_int), boundary_fn(x_bnd, normal)


@pytest.fixture
def dummy_method():
    return DummyMethod()


@pytest.fixture
def dummy_integrator():
    return DummyIntegrator()


@pytest.fixture
def default_train_config():
    return TrainConfig(epochs=5, log_every=1, use_jit=False)


@pytest.fixture
def dummy_train_state():
    return TrainState(
        step=0,
        params={"w": jnp.array([1.0, 2.0])},
        opt_state=optax.adam(1e-3).init({"w": jnp.array([1.0, 2.0])}),
        integration_key=jax.random.PRNGKey(42),
    )
