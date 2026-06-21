"""Shared fixtures for vPINN tests."""
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import jax.numpy as jnp
import pytest

from src.loss_functions.vpinn import vPINNConfig


# AI-Generated
@pytest.fixture
def vpinn_config_1d():
    return vPINNConfig(
        n_test_functions=10,
        domain_min=jnp.array([0.0]),
        domain_max=jnp.array([1.0]),
        ic_weight=1.0,
        bc_weight=1.0,
    )


# AI-Generated
@pytest.fixture
def vpinn_config_2d():
    return vPINNConfig(
        n_test_functions=16,
        domain_min=jnp.array([0.0, 0.0]),
        domain_max=jnp.array([1.0, 1.0]),
        ic_weight=2.0,
        bc_weight=3.0,
    )


# AI-Generated
@pytest.fixture
def mock_vpinn_model_valid():
    class ValidModel:
        def init(self, rng_key, x):
            return {"params": {}}

        def apply(self, params, x):
            # Must return a dictionary with a scalar 'u' per batch item
            return {"u": jnp.sum(x**2, axis=-1, keepdims=True)}

    return ValidModel()


# AI-Generated
@pytest.fixture
def mock_vpinn_model_invalid_type():
    class InvalidTypeModel:
        def init(self, rng_key, x):
            return {"params": {}}

        def apply(self, params, x):
            return jnp.sum(x)  # Returns array instead of dict

    return InvalidTypeModel()


# AI-Generated
@pytest.fixture
def mock_vpinn_model_missing_u():
    class MissingUModel:
        def init(self, rng_key, x):
            return {"params": {}}

        def apply(self, params, x):
            return {"v": jnp.sum(x)}  # Missing 'u' key

    return MissingUModel()


# AI-Generated
@pytest.fixture
def mock_vpinn_model_non_scalar():
    class NonScalarModel:
        def init(self, rng_key, x):
            return {"params": {}}

        def apply(self, params, x):
            return {"u": x}  # Returns vector per point instead of scalar

    return NonScalarModel()