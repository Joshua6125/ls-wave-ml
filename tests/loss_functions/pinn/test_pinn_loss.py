import jax.numpy as jnp
import pytest

from src.loss_functions.pinn import PINNLoss

pytestmark = pytest.mark.pinn


# AI-Generated
def test_pde_residual_zero_for_exact_solution(
    wave_solution,
    interior_points,
):
    loss = PINNLoss(wave_solution)

    result = loss.loss_interior(interior_points)

    assert jnp.allclose(result, 0.0, atol=1e-6)


# AI-Generated
def test_constant_source_term_is_applied(
    wave_solution,
    interior_points,
):
    loss = PINNLoss(
        wave_solution,
        f=1.0,
    )

    result = loss.loss_interior(interior_points)

    assert jnp.allclose(result, 1.0, atol=1e-6)


# AI-Generated
def test_initial_condition_residual_vanishes():
    u = lambda x: jnp.array([2.0])

    loss = PINNLoss(
        u,
        u0=2.0,
        ut0=0.0,
    )

    points = jnp.array([[0.0, 0.5]])
    normals = jnp.array([[-1.0, 0.0]])

    result = loss.loss_boundary(points, normals)

    assert jnp.allclose(result, 0.0, atol=1e-6)


# AI-Generated
def test_initial_condition_weight_is_applied():
    u = lambda x: jnp.array([2.0])

    loss = PINNLoss(
        u,
        u0=0.0,
        ut0=0.0,
        ic_weight=5.0,
    )

    points = jnp.array([[0.0, 0.5]])
    normals = jnp.array([[-1.0, 0.0]])

    result = loss.loss_boundary(points, normals)

    assert jnp.allclose(result, 20.0, atol=1e-6)


# AI-Generated
def test_spatial_boundary_residual_vanishes_for_zero_solution(
    zero_solution,
    spatial_boundary_points,
    spatial_boundary_normals,
):
    loss = PINNLoss(zero_solution)

    result = loss.loss_boundary(
        spatial_boundary_points,
        spatial_boundary_normals,
    )

    assert jnp.allclose(result, 0.0, atol=1e-6)


# AI-Generated
def test_boundary_weight_is_applied(
    constant_solution,
    spatial_boundary_points,
    spatial_boundary_normals,
):
    loss = PINNLoss(
        constant_solution,
        bc_weight=7.0,
    )

    result = loss.loss_boundary(
        spatial_boundary_points,
        spatial_boundary_normals,
    )

    assert jnp.allclose(result, 7.0, atol=1e-6)


# AI-Generated
def test_outgoing_time_boundary_returns_zero(
    constant_solution,
):
    loss = PINNLoss(constant_solution)

    points = jnp.array([[1.0, 0.5]])
    normals = jnp.array([[1.0, 0.0]])

    result = loss.loss_boundary(points, normals)

    assert jnp.allclose(result, 0.0, atol=1e-6)


# AI-Generated
@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("f", lambda x: jnp.array([1.0, 2.0])),
        ("u0", lambda x: jnp.array([1.0, 2.0])),
        ("ut0", lambda x: jnp.array([1.0, 2.0])),
    ],
)
def test_non_scalar_functions_raise(name, bad_value):
    kwargs = {name: bad_value}

    loss = PINNLoss(
        lambda x: jnp.array([0.0]),
        **kwargs,
    )

    point = jnp.array([[0.0, 0.5]])
    normal = jnp.array([[-1.0, 0.0]])

    with pytest.raises(ValueError):
        if name == "f":
            loss.loss_interior(point)
        else:
            loss.loss_boundary(point, normal)
