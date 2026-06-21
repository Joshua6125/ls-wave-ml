"""Shared fixtures for integration tests."""
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import jax.numpy as jnp
import pytest

from src.integration.config import MonteCarloConfig, QuadratureConfig


# AI-Generated
@pytest.fixture
def config_quadrature_1d():
    return QuadratureConfig(
        spatial_dim=1,
        t_min=0.0,
        t_max=1.0,
        x_min=0.0,
        x_max=1.0,
        degree=3,
        grid_size=4,
    )


# AI-Generated
@pytest.fixture
def config_quadrature_2d():
    return QuadratureConfig(
        spatial_dim=2,
        t_min=0.0,
        t_max=1.0,
        x_min=0.0,
        x_max=1.0,
        degree=3,
        grid_size=2,
    )


# AI-Generated
@pytest.fixture
def config_quadrature_3d():
    return QuadratureConfig(
        spatial_dim=3,
        t_min=0.0,
        t_max=1.0,
        x_min=0.0,
        x_max=1.0,
        degree=2,
        grid_size=1,
    )


# AI-Generated
@pytest.fixture
def config_monte_carlo_1d():
    return MonteCarloConfig(
        spatial_dim=1,
        t_min=0.0,
        t_max=1.0,
        x_min=0.0,
        x_max=1.0,
        interior_samples=256,
        boundary_samples=64,
    )


# AI-Generated
@pytest.fixture
def config_monte_carlo_2d():
    return MonteCarloConfig(
        spatial_dim=2,
        t_min=0.0,
        t_max=1.0,
        x_min=0.0,
        x_max=1.0,
        interior_samples=512,
        boundary_samples=64,
    )


# AI-Generated
@pytest.fixture
def config_monte_carlo_3d():
    return MonteCarloConfig(
        spatial_dim=3,
        t_min=0.0,
        t_max=1.0,
        x_min=0.0,
        x_max=1.0,
        interior_samples=512,
        boundary_samples=32,
    )


# AI-Generated
@pytest.fixture
def test_functions_1d():
    return {
        "constant": {
            "func": lambda x: jnp.ones(x.shape[0]),
            "integral": 1.0,
        },
        "linear": {
            "func": lambda x: x[:, 1],
            "integral": 0.5,
        },
        "quadratic": {
            "func": lambda x: x[:, 1] ** 2,
            "integral": 1.0 / 3.0,
        },
        "sine": {
            "func": lambda x: jnp.sin(jnp.pi * x[:, 1]),
            "integral": 2.0 / jnp.pi,
        },
        "exponential": {
            "func": lambda x: jnp.exp(x[:, 1]),
            "integral": jnp.e - 1.0,
        },
    }


# AI-Generated
@pytest.fixture
def test_functions_2d():
    return {
        "constant": {
            "func": lambda x: jnp.ones(x.shape[0]),
            "integral": 1.0,
        },
        "separable": {
            "func": lambda x: x[:, 1] * x[:, 2],
            "integral": 0.25,
        },
        "product_sine": {
            "func": lambda x: jnp.sin(jnp.pi * x[:, 1]) * jnp.sin(jnp.pi * x[:, 2]),
            "integral": (2.0 / jnp.pi) ** 2,
        },
    }


# AI-Generated
@pytest.fixture
def test_functions_3d():
    return {
        "constant": {
            "func": lambda x: jnp.ones(x.shape[0]),
            "integral": 1.0,
        },
        "separable": {
            "func": lambda x: x[:, 1] * x[:, 2] * x[:, 3],
            "integral": 0.125,
        },
    }
