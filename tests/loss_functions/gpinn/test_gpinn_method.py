# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import jax
import jax.numpy as jnp
import pytest

from src.loss_functions.gpinn import gPINN
from src.models import BuiltModelProtocol


pytestmark = pytest.mark.gpinn


# AI-Generated
class ValidModel(BuiltModelProtocol):
    def init(self, rng_key, sample_input):
        return {"params": {}}

    def apply(self, params, x):
        return {"u": jnp.array([x[0] + x[1]])}


# AI-Generated
class MissingUModel(BuiltModelProtocol):
    def init(self, rng_key, sample_input):
        return {}

    def apply(self, params, x):
        return {"v": jnp.array([1.0])}


# AI-Generated
class VectorUModel(BuiltModelProtocol):
    def init(self, rng_key, sample_input):
        return {}

    def apply(self, params, x):
        return {"u": jnp.array([1.0, 2.0])}


# AI-Generated
def test_init_params_accepts_valid_model(
    gpinn_config_default,
    sample_input,
):
    method = gPINN(
        ValidModel(),
        gpinn_config_default,
    )

    params = method.init_params(
        jax.random.PRNGKey(0),
        sample_input,
    )

    assert params == {"params": {}}


# AI-Generated
def test_init_params_rejects_missing_u_output(
    gpinn_config_default,
    sample_input,
):
    method = gPINN(
        MissingUModel(),
        gpinn_config_default,
    )

    with pytest.raises(
        ValueError,
        match="PINN model must return dict with 'u' key",
    ):
        method.init_params(
            jax.random.PRNGKey(0),
            sample_input,
        )


# AI-Generated
def test_init_params_rejects_vector_output(
    gpinn_config_default,
    sample_input,
):
    method = gPINN(
        VectorUModel(),
        gpinn_config_default,
    )

    with pytest.raises(
        ValueError,
        match="PINN model 'u' output must be scalar",
    ):
        method.init_params(
            jax.random.PRNGKey(0),
            sample_input,
        )


# AI-Generated
def test_loss_functions_returns_callable_loss_object(
    gpinn_config_default,
):
    method = gPINN(
        ValidModel(),
        gpinn_config_default,
    )

    loss = method.loss_functions({})

    assert isinstance(loss, tuple)
    assert len(loss) == 2
