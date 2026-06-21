# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import jax.numpy as jnp
import pytest

from src.loss_functions.fosls import FOSLSLoss


pytestmark = pytest.mark.fosls


# AI-Generated
def test_interior_residual_zero_for_exact_solution(
    exact_fosls_solution,
    interior_points,
):
    loss = FOSLSLoss(exact_fosls_solution)

    result = loss.loss_interior(interior_points)

    assert jnp.allclose(result, 0.0, atol=1e-6)


# AI-Generated
def test_scalar_f_forcing_is_applied(
    exact_fosls_solution,
    interior_points,
):
    loss = FOSLSLoss(
        exact_fosls_solution,
        f=1.0,
    )

    result = loss.loss_interior(interior_points)

    assert jnp.allclose(result, 1.0, atol=1e-6)


# AI-Generated
def test_scalar_g_forcing_is_applied(
    exact_fosls_solution,
    interior_points,
):
    loss = FOSLSLoss(
        exact_fosls_solution,
        g=2.0,
    )

    result = loss.loss_interior(interior_points)

    assert jnp.allclose(result, 4.0, atol=1e-6)

# AI-Generated
def test_initial_condition_residual_vanishes():
    loss = FOSLSLoss(
        lambda x: jnp.array([2.0, 3.0]),
        v0=2.0,
        sigma0=3.0,
    )

    points = jnp.array([[0.0, 0.5]])
    normals = jnp.array([[-1.0, 0.0]])

    result = loss.loss_boundary(points, normals)

    assert jnp.allclose(result, 0.0, atol=1e-6)


# AI-Generated
def test_initial_condition_weight_is_applied():
    loss = FOSLSLoss(
        lambda x: jnp.array([2.0, 3.0]),
        v0=0.0,
        sigma0=0.0,
        ic_weight=5.0,
    )

    points = jnp.array([[0.0, 0.5]])
    normals = jnp.array([[-1.0, 0.0]])

    result = loss.loss_boundary(points, normals)

    expected = 5.0 * (2.0**2 + 3.0**2)

    assert jnp.allclose(result, expected, atol=1e-6)


# AI-Generated
def test_non_initial_boundary_returns_zero(
    constant_fosls_solution,
    initial_boundary_points,
    outgoing_time_normals,
):
    loss = FOSLSLoss(constant_fosls_solution)

    result = loss.loss_boundary(
        initial_boundary_points,
        outgoing_time_normals,
    )

    assert jnp.allclose(result, 0.0, atol=1e-6)


# AI-Generated
@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("f", lambda x: jnp.array([1.0, 2.0])),
        ("v0", lambda x: jnp.array([1.0, 2.0])),
    ],
)
def test_scalar_quantities_must_be_scalar(name, bad_value):
    kwargs = {name: bad_value}

    loss = FOSLSLoss(
        lambda x: jnp.array([0.0, 0.0]),
        **kwargs,
    )

    point = jnp.array([[0.0, 0.5]])
    normal = jnp.array([[-1.0, 0.0]])

    with pytest.raises(ValueError):
        if name == "f":
            loss.loss_interior(point)
        else:
            loss.loss_boundary(point, normal)


# AI-Generated
def test_g_shape_must_match_spatial_dimension():
    loss = FOSLSLoss(
        lambda x: jnp.array([0.0, 0.0]),
        g=lambda x: jnp.array([1.0, 2.0]),
    )

    with pytest.raises(
        ValueError,
        match="g must match shape of spatial gradient",
    ):
        loss.loss_interior(jnp.array([[0.5, 0.5]]))


# AI-Generated
def test_sigma0_shape_must_match_sigma_dimension():
    loss = FOSLSLoss(
        lambda x: jnp.array([0.0, 0.0]),
        sigma0=lambda x: jnp.array([1.0, 2.0]),
    )

    with pytest.raises(
        ValueError,
        match="sigma0 must match sigma shape",
    ):
        loss.loss_boundary(
            jnp.array([[0.0, 0.5]]),
            jnp.array([[-1.0, 0.0]]),
        )