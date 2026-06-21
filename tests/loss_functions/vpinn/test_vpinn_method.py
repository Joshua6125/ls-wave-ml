# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import jax
import jax.numpy as jnp
import pytest

from src.loss_functions.vpinn import vPINN

pytestmark = pytest.mark.vpinn


# AI-Generated
def test_vpinn_init_params_valid_model(mock_vpinn_model_valid, vpinn_config_2d):
    vpinn = vPINN(model=mock_vpinn_model_valid, config=vpinn_config_2d)

    rng = jax.random.PRNGKey(0)
    sample_input = jnp.ones((1, 2))

    # Should not raise any exceptions
    params = vpinn.init_params(rng, sample_input)
    assert params is not None


# AI-Generated
@pytest.mark.parametrize(
    ("model_fixture", "expected_error"),
    [
        ("mock_vpinn_model_invalid_type", "must return dict with 'u' key"),
        ("mock_vpinn_model_missing_u", "must return dict with 'u' key"),
        ("mock_vpinn_model_non_scalar", "must be scalar"),
    ],
)
def test_vpinn_init_params_rejects_invalid_models(
    request, model_fixture, expected_error, vpinn_config_2d
):
    model = request.getfixturevalue(model_fixture)
    vpinn = vPINN(model=model, config=vpinn_config_2d)

    rng = jax.random.PRNGKey(0)
    sample_input = jnp.ones((2, 2))  # Batch size 2

    with pytest.raises(ValueError, match=expected_error):
        vpinn.init_params(rng, sample_input)


# AI-Generated
def test_vpinn_aggregate_loss(mock_vpinn_model_valid, vpinn_config_2d):
    vpinn = vPINN(model=mock_vpinn_model_valid, config=vpinn_config_2d)

    # Mocking interior evaluations (tensor product of points and basis functions)
    # 2 arrays in the PyTree, representing evaluated integrals
    interior_mock = {
        "res1": jnp.array([1.0, 2.0]),  # square and sum -> 1 + 4 = 5
        "res2": jnp.array([-1.0]),  # square and sum -> 1
    }  # Total interior = 6.0

    # Mocking boundary evaluations (already squared residuals)
    boundary_mock = {
        "ic": jnp.array([2.0, 3.0]),  # sum -> 5
        "bc": jnp.array([1.5]),  # sum -> 1.5
    }  # Total boundary = 6.5

    total_loss = vpinn.aggregate_loss(interior_mock, boundary_mock)

    assert jnp.allclose(total_loss, 12.5, atol=1e-6)
