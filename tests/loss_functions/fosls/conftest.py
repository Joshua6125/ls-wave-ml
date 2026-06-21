"""Shared fixtures for FOSLS tests."""
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import jax.numpy as jnp
import pytest

from src.loss_functions import FOSLSConfig
from src.models import MLPConfig


pytestmark = pytest.mark.fosls


# AI-Generated
@pytest.fixture
def fosls_config_default():
    return FOSLSConfig(
        model=MLPConfig(),
        f=0.0,
        g=0.0,
        v0=0.0,
        sigma0=0.0,
        ic_weight=1.0,
    )


# AI-Generated
@pytest.fixture
def fosls_config_weighted():
    return FOSLSConfig(
        model=MLPConfig(),
        f=0.0,
        g=0.0,
        v0=0.0,
        sigma0=0.0,
        ic_weight=5.0,
    )


# AI-Generated
@pytest.fixture
def exact_fosls_solution():
    """
    State vector [v, sigma].

    v(t,x) = t
    sigma(t,x) = x

    Then

        dt(v) = 1
        div(sigma) = 1

        dt(v) - div(sigma) = 0

        dt(sigma) = 0
        grad(v) = 0

        dt(sigma) - grad(v) = 0
    """
    return lambda x: jnp.array([x[0], x[1]])


# AI-Generated
@pytest.fixture
def zero_fosls_solution():
    return lambda x: jnp.array([0.0, 0.0])


# AI-Generated
@pytest.fixture
def constant_fosls_solution():
    return lambda x: jnp.array([1.0, 1.0])


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
def outgoing_time_normals():
    return jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
