import jax
import jax.numpy as jnp
import pytest

from src.loss_functions.fosls import FOSLS
from src.models import BuiltModelProtocol


pytestmark = pytest.mark.fosls


# AI-Generated
class ValidModel(BuiltModelProtocol):
    def init(self, rng_key, sample_input):
        return {"params": {}}

    def apply(self, params, x):
        return {
            "v": jnp.array([1.0]),
            "sigma": jnp.array([2.0]),
        }


# AI-Generated
class NotDictModel(BuiltModelProtocol):
    def init(self, rng_key, sample_input):
        return {}

    def apply(self, params, x): # type: ignore
        return jnp.array([1.0])


# AI-Generated
class MissingSigmaModel(BuiltModelProtocol):
    def init(self, rng_key, sample_input):
        return {}

    def apply(self, params, x):
        return {
            "v": jnp.array([1.0]),
        }


# AI-Generated
class VectorVModel(BuiltModelProtocol):
    def init(self, rng_key, sample_input):
        return {}

    def apply(self, params, x):
        return {
            "v": jnp.array([1.0, 2.0]),
            "sigma": jnp.array([1.0]),
        }


# AI-Generated
class WrongSigmaDimensionModel(BuiltModelProtocol):
    def init(self, rng_key, sample_input):
        return {}

    def apply(self, params, x):
        return {
            "v": jnp.array([1.0]),
            "sigma": jnp.array([1.0, 2.0]),
        }


# AI-Generated
def test_init_params_accepts_valid_model(
    fosls_config_default,
):
    method = FOSLS(
        ValidModel(),
        fosls_config_default,
    )

    params = method.init_params(
        jax.random.PRNGKey(0),
        jnp.array([0.5, 0.5]),
    )

    assert params == {"params": {}}


# AI-Generated
def test_init_params_requires_dict_output(
    fosls_config_default,
):
    method = FOSLS(
        NotDictModel(),
        fosls_config_default,
    )

    with pytest.raises(
        ValueError,
        match="Model must return a dict",
    ):
        method.init_params(
            jax.random.PRNGKey(0),
            jnp.array([0.5, 0.5]),
        )


# AI-Generated
def test_init_params_requires_sigma_head(
    fosls_config_default,
):
    method = FOSLS(
        MissingSigmaModel(),
        fosls_config_default,
    )

    with pytest.raises(
        ValueError,
        match="Missing required output heads",
    ):
        method.init_params(
            jax.random.PRNGKey(0),
            jnp.array([0.5, 0.5]),
        )


# AI-Generated
def test_init_params_requires_scalar_v(
    fosls_config_default,
):
    method = FOSLS(
        VectorVModel(),
        fosls_config_default,
    )

    with pytest.raises(
        ValueError,
        match="Output 'v' must be scalar",
    ):
        method.init_params(
            jax.random.PRNGKey(0),
            jnp.array([0.5, 0.5]),
        )


# AI-Generated
def test_init_params_requires_correct_sigma_dimension(
    fosls_config_default,
):
    method = FOSLS(
        WrongSigmaDimensionModel(),
        fosls_config_default,
    )

    with pytest.raises(
        ValueError,
        match="Output 'sigma' has incorrect dimension",
    ):
        method.init_params(
            jax.random.PRNGKey(0),
            jnp.array([0.5, 0.5]),
        )
