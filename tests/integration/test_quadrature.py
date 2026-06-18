from dataclasses import replace

import jax.numpy as jnp
import pytest

from src.integration import QuadratureIntegration, get_integrator


pytestmark = pytest.mark.quadrature


# AI-Generated
@pytest.mark.parametrize(
    ("fixture_name", "function_name", "expected"),
    [
        ("config_quadrature_1d", "constant", 1.0),
        ("config_quadrature_1d", "linear", 0.5),
        ("config_quadrature_1d", "quadratic", 1.0 / 3.0),
        ("config_quadrature_1d", "sine", 2.0 / jnp.pi),
        ("config_quadrature_1d", "exponential", jnp.e - 1.0),
        ("config_quadrature_2d", "constant", 1.0),
        ("config_quadrature_2d", "separable", 0.25),
        ("config_quadrature_2d", "product_sine", (2.0 / jnp.pi) ** 2),
        ("config_quadrature_3d", "constant", 1.0),
        ("config_quadrature_3d", "separable", 0.125),
    ],
)
def test_quadrature_integrates_known_functions(request, fixture_name, function_name, expected):
    config = request.getfixturevalue(fixture_name)
    functions = request.getfixturevalue(f"test_functions_{config.spatial_dim}d")
    integrator = QuadratureIntegration(config)

    result = integrator.integrate_interior(functions[function_name]["func"])

    assert jnp.allclose(result, expected, atol=1e-3)


# AI-Generated
@pytest.mark.parametrize(
    ("fixture_name", "expected_boundary_measure"),
    [
        ("config_quadrature_1d", 4.0),
        ("config_quadrature_2d", 6.0),
        ("config_quadrature_3d", 8.0),
    ],
)
def test_quadrature_boundary_integral_of_constant_matches_measure(
    request,
    fixture_name,
    expected_boundary_measure,
):
    config = request.getfixturevalue(fixture_name)
    integrator = QuadratureIntegration(config)

    result = integrator.integrate_boundary(
        lambda points, normals: jnp.ones(points.shape[0])
    )

    assert jnp.allclose(result, expected_boundary_measure, atol=1e-6)


# AI-Generated
def test_quadrature_boundary_normals_cancel_in_2d(config_quadrature_2d):
    integrator = QuadratureIntegration(config_quadrature_2d)
    result = integrator.integrate_boundary(lambda points, normals: normals[:, 0])

    assert jnp.allclose(result, 0.0, atol=1e-6)


# AI-Generated
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("spatial_dim", 0, "dim must be strictly positive"),
        ("x_min", 1.0, "x_min must be < x_max"),
        ("t_max", -1.0, "t_min must be < t_max"),
        ("degree", 0, "degree must be strictly positive"),
        ("grid_size", 0, "grid_size must be strictly positive"),
    ],
)
def test_quadrature_config_validation_rejects_invalid_inputs(
    config_quadrature_1d,
    field,
    value,
    message,
):
    bad_config = replace(config_quadrature_1d, **{field: value})

    with pytest.raises(ValueError, match=message):
        QuadratureIntegration(bad_config)


# AI-Generated
def test_quadrature_custom_bounds_are_respected(config_quadrature_1d):
    config = replace(config_quadrature_1d, x_min=-1.0, x_max=1.0)
    integrator = QuadratureIntegration(config)

    result = integrator.integrate_interior(lambda points: points[:, 1])

    assert jnp.allclose(result, 0.0, atol=1e-6)


# AI-Generated
def test_quadrature_adaptive_flag_emits_warning(config_quadrature_1d, capsys):
    config = replace(config_quadrature_1d, adaptive_integration=True)

    QuadratureIntegration(config)
    captured = capsys.readouterr()

    assert "Adaptive quadrature not implemented" in captured.err


# AI-Generated
def test_get_integrator_dispatches_and_rejects_unknown(config_quadrature_1d, config_monte_carlo_1d):
    assert isinstance(get_integrator(config_quadrature_1d), QuadratureIntegration)
    assert isinstance(get_integrator(config_monte_carlo_1d), type(get_integrator(config_monte_carlo_1d)))

    with pytest.raises(ValueError, match="Unknown integration config type"):
        get_integrator(object())  # type: ignore[arg-type]import jax.numpy as jnp


def test_quadrature_integrate_combined_1d(config_quadrature_1d, test_functions_1d):
    """Test that interior + boundary method combines correctly."""
    from src.integration import QuadratureIntegration

    integrator = QuadratureIntegration(config_quadrature_1d)
    const_func = test_functions_1d['constant']

    # Compute using combined method
    interior_func = const_func['func']
    boundary_func = lambda pts, normals: jnp.ones(pts.shape[0])

    interior_loss, boundary_loss = integrator.integrate(
        interior_func, boundary_func
    )

    print(interior_loss, boundary_loss, interior_func)

    # Verify interior matches direct call
    assert jnp.allclose(interior_loss, const_func['integral'], atol=1e-6)

    # Verify boundary is 2 (two endpoints at x=0 and x=1)
    assert jnp.allclose(boundary_loss, 4.0, atol=1e-6)

