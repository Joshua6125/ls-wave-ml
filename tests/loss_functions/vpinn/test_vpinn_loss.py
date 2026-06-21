# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import jax.numpy as jnp
import pytest

from src.loss_functions.vpinn import vPINNLoss

pytestmark = pytest.mark.vpinn


# AI-Generated
def test_vpinn_loss_initializes_basis_correctly():
    def dummy_u(x):
        return jnp.sum(x)

    # 2D domain, 16 test functions requested -> J = 16**(1/2) = 4
    # Expected basis functions: 4 * 4 = 16
    loss_2d = vPINNLoss(
        u_model=dummy_u,
        n_test_functions=16,
        domain_min=jnp.array([0.0, 0.0]),
        domain_max=jnp.array([1.0, 1.0]),
    )

    assert loss_2d._k_vecs.shape == (16, 2)

    # 3D domain, 27 test functions requested -> J = 27**(1/3) = 3
    # Expected basis functions: 3 * 3 * 3 = 27
    loss_3d = vPINNLoss(
        u_model=dummy_u,
        n_test_functions=27,
        domain_min=jnp.array([0.0, 0.0, 0.0]),
        domain_max=jnp.array([1.0, 1.0, 1.0]),
    )

    assert loss_3d._k_vecs.shape == (27, 3)


# AI-Generated
def test_vpinn_loss_interior_shape():
    def dummy_u(x):
        return jnp.sum(x**2)  # Quadratic ensures non-zero second derivatives

    loss = vPINNLoss(
        u_model=dummy_u,
        n_test_functions=16,
        domain_min=jnp.array([0.0, 0.0]),
        domain_max=jnp.array([1.0, 1.0]),
    )

    batch_size = 5
    x_interior = jnp.ones((batch_size, 2)) * 0.5

    residual = loss.loss_interior(x_interior)

    # Residual should be evaluated for every point and every test function
    assert residual.shape == (batch_size, 16)


# AI-Generated
def test_vpinn_loss_boundary_routing():
    def dummy_u(x):
        return jnp.sum(x**2)

    loss = vPINNLoss(
        u_model=dummy_u,
        ic_weight=2.0,
        bc_weight=3.0,
        domain_min=jnp.array([0.0, 0.0]),
        domain_max=jnp.array([1.0, 1.0]),
    )

    # 3 points:
    # [0] IC (normal[0] < 0)
    # [1] Spatial BC (normal[0] == 0)
    # [2] Other (normal[0] > 0)
    x_boundary = jnp.array([[0.0, 0.5], [0.5, 0.0], [1.0, 0.5]])
    normals = jnp.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])

    boundary_loss = loss.loss_boundary(x_boundary, normals)

    assert boundary_loss.shape == (3,)
    # IC is penalized by 2.0 (ic_weight)
    assert boundary_loss[0] > 0
    # BC is penalized by 3.0 (bc_weight)
    assert boundary_loss[1] > 0
    # Unhandled boundary condition returns 0.0
    assert boundary_loss[2] == 0.0
