"""Shared fixtures for PINN tests."""
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from dataclasses import replace

import jax.numpy as jnp
import pytest

from src.loss_functions import PINNConfig
from src.models import MLPConfig


# AI-Generated
@pytest.fixture
def pinn_config_default():
    return PINNConfig(
        model=MLPConfig(),
        f=0.0,
        u0=0.0,
        ut0=0.0,
        ic_weight=1.0,
        bc_weight=1.0,
    )


# AI-Generated
@pytest.fixture
def pinn_config_weighted():
    return PINNConfig(
        model=MLPConfig(),
        f=0.0,
        u0=0.0,
        ut0=0.0,
        ic_weight=5.0,
        bc_weight=7.0,
    )


# AI-Generated
@pytest.fixture
def constant_solution():
    return lambda x: jnp.array([1.0])


# AI-Generated
@pytest.fixture
def zero_solution():
    return lambda x: jnp.array([0.0])


# AI-Generated
@pytest.fixture
def wave_solution():
    """
    u(t,x)=t²+x²

    u_tt = 2
    Δu = 2   (1D space)
    residual = 0
    """
    return lambda x: jnp.array([x[0] ** 2 + x[1] ** 2])


# AI-Generated
@pytest.fixture
def interior_points():
    return jnp.array(
        [
            [0.25, 0.25],
            [0.50, 0.50],
            [0.75, 0.75],
        ]
    )


# AI-Generated
@pytest.fixture
def initial_boundary_points():
    return jnp.array(
        [
            [0.0, 0.25],
            [0.0, 0.50],
            [0.0, 0.75],
        ]
    )


# AI-Generated
@pytest.fixture
def initial_boundary_normals():
    return jnp.array(
        [
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ]
    )


# AI-Generated
@pytest.fixture
def spatial_boundary_points():
    return jnp.array(
        [
            [0.25, 0.0],
            [0.50, 1.0],
        ]
    )


# AI-Generated
@pytest.fixture
def spatial_boundary_normals():
    return jnp.array(
        [
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )


# AI-Generated
@pytest.fixture
def dummy_params():
    return {"params": {}}


# AI-Generated
@pytest.fixture
def sample_input():
    return jnp.array([0.5, 0.5])
