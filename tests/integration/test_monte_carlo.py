# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import pytest

from src.integration import MonteCarloIntegration

pytestmark = pytest.mark.monte_carlo


# AI-Generated
@pytest.mark.parametrize(
    ("fixture_name", "expected_integral"),
    [
        ("config_monte_carlo_1d", 1.0),
        ("config_monte_carlo_2d", 1.0),
        ("config_monte_carlo_3d", 1.0),
    ],
)
def test_constant_interior_integral_matches_volume(
    request,
    fixture_name,
    expected_integral,
):
    config = request.getfixturevalue(fixture_name)
    integrator = MonteCarloIntegration(config)

    integrator.key = jr.PRNGKey(0)
    result = integrator.integrate_interior(lambda points: jnp.ones(points.shape[0]))

    assert jnp.allclose(result, expected_integral)


# AI-Generated
def test_interior_tree_outputs_are_aggregated(config_monte_carlo_1d):
    integrator = MonteCarloIntegration(config_monte_carlo_1d)
    integrator.key = jr.PRNGKey(0)

    result = integrator.integrate_interior(
        lambda points: {
            "left": jnp.ones(points.shape[0]),
            "right": 2.0 * jnp.ones(points.shape[0]),
        }
    )

    assert result["left"] == 1.0
    assert result["right"] == 2.0


# AI-Generated
@pytest.mark.parametrize(
    ("fixture_name", "expected_boundary_measure"),
    [
        ("config_monte_carlo_1d", 4.0),
        ("config_monte_carlo_2d", 6.0),
        ("config_monte_carlo_3d", 8.0),
    ],
)
def test_constant_boundary_integral_matches_boundary_measure(
    request,
    fixture_name,
    expected_boundary_measure,
):
    config = request.getfixturevalue(fixture_name)
    integrator = MonteCarloIntegration(config)

    integrator.key = jr.PRNGKey(0)
    result = integrator.integrate_boundary(
        lambda points, normals: jnp.ones(points.shape[0])
    )

    assert jnp.allclose(result, expected_boundary_measure)


# AI-Generated
def test_integrate_rejects_missing_rng_key(config_monte_carlo_1d):
    integrator = MonteCarloIntegration(config_monte_carlo_1d)

    with pytest.raises(ValueError, match="rng_key may not be None"):
        integrator.integrate(
            lambda points: points[:, 0], lambda pts, normals: pts[:, 0], rng_key=None
        )


# AI-Generated
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("spatial_dim", 0, "dim must be strictly positive"),
        ("x_min", 2.0, "x_min must be < x_max"),
        ("t_max", -1.0, "t_min must be < t_max"),
        ("interior_samples", 0, "interior_samples must be strictly positive"),
        ("boundary_samples", 0, "boundary_samples must be strictly positive"),
    ],
)
def test_monte_carlo_config_validation_rejects_invalid_inputs(
    config_monte_carlo_1d,
    field,
    value,
    message,
):
    bad_config = replace(config_monte_carlo_1d, **{field: value})

    with pytest.raises(ValueError, match=message):
        MonteCarloIntegration(bad_config)


# AI-Generated
def test_monte_carlo_respects_custom_bounds(config_monte_carlo_1d):
    config = replace(config_monte_carlo_1d, x_min=-1.0, x_max=1.0)
    integrator = MonteCarloIntegration(config)

    integrator.key = jr.PRNGKey(0)
    result = integrator.integrate_interior(lambda points: jnp.ones(points.shape[0]))

    assert jnp.allclose(result, 2.0)


# AI-Generated
def test_boundary_sampling_shapes_are_consistent(config_monte_carlo_2d):
    integrator = MonteCarloIntegration(config_monte_carlo_2d)
    integrator.key = jr.PRNGKey(0)
    boundary_data = integrator._setup_boundary_samples()

    assert boundary_data["points"].shape[1] == integrator.dim
    assert boundary_data["normals"].shape == boundary_data["points"].shape
    assert boundary_data["weights"].shape[0] == boundary_data["points"].shape[0]
    assert jnp.all(boundary_data["weights"] > 0)


def test_monte_carlo_integrate_combined_1d(config_monte_carlo_1d, test_functions_1d):
    integrator = MonteCarloIntegration(config_monte_carlo_1d)
    const_func = test_functions_1d["constant"]

    # Add key, as this is usually passed via integrate()
    integrator.key = jr.PRNGKey(0)

    # Compute using combined method
    interior_func = const_func["func"]
    boundary_func = lambda pts, normals: jnp.ones(pts.shape[0])

    interior_loss, boundary_loss = integrator.integrate(interior_func, boundary_func)

    # Verify interior matches direct call
    assert jnp.allclose(interior_loss, const_func["integral"], atol=1e-2)

    # Verify boundary is 4 (all 4 sides of space-time cylinder)
    assert jnp.allclose(boundary_loss, 4.0, atol=1e-2)


def test_monte_carlo_boundary_2d(config_monte_carlo_2d):
    integrator = MonteCarloIntegration(config_monte_carlo_2d)

    # Add key, as this is usually passed via integrate()
    integrator.key = jr.PRNGKey(0)

    # Boundary function always returns 1
    boundary_func = lambda pts, normals: jnp.ones(pts.shape[0])

    result = integrator.integrate_boundary(boundary_func)

    # 3D space-time cube has 6 boundaries.
    assert jnp.allclose(result, 6.0, atol=1e-2)


def test_monte_carlo_boundary_normals_2d(config_monte_carlo_2d):
    integrator = MonteCarloIntegration(config_monte_carlo_2d)

    # Add key, as this is usually passed via integrate()
    integrator.key = jr.PRNGKey(0)

    # Function returns x-component of outward normal
    boundary_func = lambda pts, normals: normals[:, 0]

    result = integrator.integrate_boundary(boundary_func)

    # Total should be 0 due to cancellation (up to numerical precision)
    assert jnp.allclose(result, 0.0, atol=1e-2)


def test_monte_carlo_resamples(config_monte_carlo_1d):
    def interior_func(x):
        return x[:, 0]

    def boundary_func(pts, normals):
        return jnp.zeros(pts.shape[0])

    key0 = jr.PRNGKey(0)

    integrator_a = MonteCarloIntegration(config_monte_carlo_1d)
    interior_1a, _ = integrator_a.integrate(interior_func, boundary_func, key0)
    key1a, _ = jr.split(key0)
    interior_2a, _ = integrator_a.integrate(interior_func, boundary_func, key1a)

    # New key should produce a new sample set and therefore a different estimate.
    assert not jnp.allclose(interior_1a, interior_2a)

    # Replaying the same key sequence reproduces the same estimates exactly.
    integrator_b = MonteCarloIntegration(config_monte_carlo_1d)
    interior_1b, _ = integrator_b.integrate(interior_func, boundary_func, key0)
    key1b, _ = jr.split(key0)
    interior_2b, _ = integrator_b.integrate(interior_func, boundary_func, key1b)

    assert jnp.allclose(interior_1a, interior_1b)
    assert jnp.allclose(interior_2a, interior_2b)
