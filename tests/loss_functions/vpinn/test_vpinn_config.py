# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import jax.numpy as jnp
import pytest

from src.loss_functions.vpinn import vPINNConfig

pytestmark = pytest.mark.vpinn


# AI-Generated
def test_vpinn_config_defaults():
    config = vPINNConfig()

    assert config.kind == "vpinn"
    assert config.ic_weight == 1.0
    assert config.bc_weight == 1.0
    assert config.n_test_functions == 400
    assert config.domain_min is None
    assert config.domain_max is None


# AI-Generated
def test_vpinn_config_custom_values(vpinn_config_2d):
    assert vpinn_config_2d.n_test_functions == 16
    assert vpinn_config_2d.ic_weight == 2.0
    assert vpinn_config_2d.bc_weight == 3.0
    assert jnp.array_equal(vpinn_config_2d.domain_min, jnp.array([0.0, 0.0]))
    assert jnp.array_equal(vpinn_config_2d.domain_max, jnp.array([1.0, 1.0]))
