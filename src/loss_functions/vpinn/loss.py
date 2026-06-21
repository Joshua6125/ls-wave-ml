'''
Variational PINN loss.
'''
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import itertools
from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class vPINNLoss(Loss):
    """
    Variational Physics-Informed Neural Network (vPINN) loss.

    Instead of enforcing the PDE pointwise, the residual is projected onto a
    finite set of test functions (Galerkin-style formulation).

    This implementation uses a tensor-product Fourier sine basis:
        v_k(x) = \\prod_d sin(k_d * pi * x_d)

    Key idea:
        Instead of minimising R(x)^2, enforce:
            <R, v_k> \\approx 0 for multiple test functions v_k

    This improves stability and can reduce sensitivity to collocation sampling.
    """

    def __init__(
        self,
        u_model: Callable[[jnp.ndarray], jnp.ndarray],
        c: float | Callable[[jnp.ndarray], jnp.ndarray] = 1.0,
        f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ic_weight: float = 1.0,
        bc_weight: float = 1.0,
        n_test_functions: int = 10,
        domain_min: jnp.ndarray | None = None,
        domain_max: jnp.ndarray | None = None,
    ):
        self.u_model = u_model
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.n_test_functions = n_test_functions

        # Physical parameters / source terms can be constant or spatially varying.
        self._c_fn = c if callable(c) else self._constant_function(c)
        self._f_fn = f if callable(f) else self._constant_function(f)
        self._u0_fn = u0 if callable(u0) else self._constant_function(u0)
        self._ut0_fn = ut0 if callable(ut0) else self._constant_function(ut0)

        # Domain definition (used to map coordinates into [0, 1]^d for basis functions)
        self.domain_min = (
            domain_min if domain_min is not None else jnp.array([0.0, 0.0])
        )
        self.domain_max = (
            domain_max if domain_max is not None else jnp.array([1.0, 1.0])
        )
        self.dim = self.domain_min.shape[0]

        # Construct tensor-product Fourier sine basis indices.
        # We distribute n_test_functions approximately evenly across dimensions.
        J = int(self.n_test_functions ** (1.0 / self.dim))
        if J < 1:
            J = 1

        # Multi-index set for basis frequencies k = (k1, ..., kd)
        indices = list(itertools.product(range(1, J + 1), repeat=self.dim))
        self._k_vecs = jnp.array(indices)

        # Vectorised residual evaluation over batches of points.
        self._vmapped_pde_residual = jax.vmap(self._pde_residual)
        self._vmapped_ic_residual = jax.vmap(self._ic_residual)
        self._vmapped_spatial_bc_residual = jax.vmap(self._spatial_bc_residual)

    def _u(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Scalar neural network solution u(t, x).
        """
        return self.u_model(x).squeeze()

    def _pde_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Weak-form integrand of the PDE residual against test functions.

        Instead of returning (R(x))^2, this returns:
            R(x) * v_k(x) for all test functions v_k

        where:
            R(x) = u_tt - \\Delta u - f(x)
        """
        H = jax.hessian(self._u)(x)

        # Second derivative in time
        u_tt = H[0, 0]

        # Spatial Laplacian
        laplacian_u = jnp.trace(H[1:, 1:])

        c = self._c_fn(x)
        f = self._f_fn(x)

        # Strong form residual
        residual = u_tt - c**2 * laplacian_u - f

        # Map physical coordinates to unit cube [0, 1]^d for basis construction
        x_unit = (x - self.domain_min) / (self.domain_max - self.domain_min)

        # Tensor-product Fourier sine basis:
        # v_k(x) = \prod_d sin(k_d * \pi * x_d)
        #
        # self._k_vecs shape: (n_test_functions, dim)
        # x_unit shape: (dim,)
        test_vals = jnp.prod(
            jnp.sin(self._k_vecs * jnp.pi * x_unit),
            axis=-1,
        )

        # Return weak-form integrand (no squaring)
        return residual * test_vals

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """
        Batch evaluation of weak-form residual.

        Output shape:
            (N_points, n_test_functions)
        """
        return self._vmapped_pde_residual(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Initial condition penalty at t = 0:

            u(0, x) \\approx u0(x)
            u_t(0, x) \\approx ut0(x)
        """
        u_val = self._u(x)
        ut_val = jax.grad(self._u)(x)[0]

        return self.ic_weight * (
            (u_val - self._u0_fn(x)) ** 2 + (ut_val - self._ut0_fn(x)) ** 2
        )

    def _spatial_bc_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Homogeneous Dirichlet boundary condition on spatial boundary:

            u = 0
        """
        return self.bc_weight * self._u(x) ** 2

    def loss_boundary(
        self,
        x_boundary: jnp.ndarray,
        normal_vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Boundary loss combining:
            - initial time boundary conditions
            - spatial Dirichlet boundary conditions

        Classification rule:
            normal[0] < 0 -> initial condition boundary
            normal[0] == 0 -> spatial boundary
        """
        is_ic = normal_vector[:, 0] < 0
        is_spatial_bc = normal_vector[:, 0] == 0

        return jnp.where(
            is_ic,
            self._vmapped_ic_residual(x_boundary),
            jnp.where(
                is_spatial_bc,
                self._vmapped_spatial_bc_residual(x_boundary),
                0.0,
            ),
        )
