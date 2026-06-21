'''
Gradient PINN loss.
'''
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class gPINNLoss(Loss):
    """
    Gradient-enhanced Physics-Informed Neural Network (gPINN) loss for the wave equation.

    The formulation augments the standard PDE residual with gradient penalties:
        - standard residual: R(x) = u_tt - \\Delta u - f
        - residual gradient penalty: || \\nabla R(x)||^2 (stabilises training)
        - optional solution smoothness is implicitly encouraged through gradients

    The interior objective per collocation point is:
        R(x)^2 + residual_grad_weight * || \\nabla R(x)||^2

    Boundary handling distinguishes:
        - initial conditions (time boundary)
        - spatial boundary conditions (Dirichlet-type u = 0)
    """

    def __init__(
        self,
        u_model: Callable[[jnp.ndarray], jnp.ndarray],
        f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ic_weight: float = 1.0,
        bc_weight: float = 10.0,
        residual_grad_weight: float = 1e-2,
    ):
        self.u_model = u_model
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.residual_grad_weight = residual_grad_weight

        # Allow coefficients to be either constants or functions of space-time.
        self._f_fn = f if callable(f) else self._constant_function(f)
        self._u0_fn = u0 if callable(u0) else self._constant_function(u0)
        self._ut0_fn = ut0 if callable(ut0) else self._constant_function(ut0)

        # Vectorised evaluation over batches of collocation points.
        self._vmapped_pde_residual = jax.vmap(self._pde_residual)
        self._vmapped_ic_residual = jax.vmap(self._ic_residual)
        self._vmapped_spatial_bc_residual = jax.vmap(self._spatial_bc_residual)

    def _u(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Scalar field evaluation of the neural network solution.
        """
        return self.u_model(x).squeeze()

    def _residual_scalar(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Pointwise PDE residual:
            R(x) = u_tt - \\Delta u - f(x)

        Computed via full Hessian of u.
        """
        H = jax.hessian(self._u)(x)

        # Second time derivative
        u_tt = H[0, 0]

        # Spatial Laplacian (trace of spatial Hessian block)
        laplacian_u = jnp.trace(H[1:, 1:])

        f = self._f_fn(x)
        if jnp.ndim(f) != 0:
            raise ValueError("f should be scalar-valued.")

        return u_tt - laplacian_u - f

    def _pde_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Interior loss contribution for a single collocation point.

        Combines:
            - squared PDE residual
            - squared spatial gradient of residual (stabilisation term)
        """
        R = self._residual_scalar(x)

        # Standard PINN residual term
        res_sq = R**2

        # Gradient of residual with respect to full input (t, x)
        grad_R = jax.grad(self._residual_scalar)(x)

        # Only spatial derivatives are used for regularisation
        grad_R_spatial = grad_R[1:]
        grad_R_norm_sq = jnp.sum(grad_R_spatial**2)

        return res_sq + self.residual_grad_weight * grad_R_norm_sq

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """
        Batch interior loss over collocation points.
        """
        return self._vmapped_pde_residual(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Initial condition penalty at t = t_min.

        Enforces:
            u(x, 0) \\approx u0(x)
            u_t(x, 0) \\approx ut0(x)
        """
        u_val = self._u(x)

        # Time derivative of u at the initial time
        ut_val = jax.grad(self._u)(x)[0]

        u0_val = self._u0_fn(x)
        if jnp.ndim(u0_val) != 0:
            raise ValueError("u0 should be scalar-valued.")

        ut0_val = self._ut0_fn(x)
        if jnp.ndim(ut0_val) != 0:
            raise ValueError("ut0 should be scalar-valued.")

        return self.ic_weight * ((u_val - u0_val) ** 2 + (ut_val - ut0_val) ** 2)

    def _spatial_bc_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Spatial Dirichlet boundary condition penalty.

        Enforces that u(x) is approximately 0 on spatial boundaries.
        """
        return self.bc_weight * self._u(x) ** 2

    def loss_boundary(
        self,
        x_boundary: jnp.ndarray,
        normal_vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Boundary loss combining:
            - initial condition boundary (time-like boundary)
            - spatial Dirichlet boundary

        Classification is based on normal direction:
            normal[0] < 0 -> initial time boundary
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
