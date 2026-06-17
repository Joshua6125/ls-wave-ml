"""PINN loss function."""

from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class PINNLoss(Loss):
    """
    Physics-Informed Neural Network (PINN) loss for the wave equation:

        u_tt - \\Delta u = f

    The network approximates a scalar field u(t, x), where:
        - x[0] is time t
        - x[1:] are spatial coordinates

    The loss enforces:
        - PDE residual in the interior
        - Initial conditions at t = 0
        - Homogeneous Dirichlet boundary conditions on spatial boundaries
    """

    def __init__(
        self,
        u_model: Callable[[jnp.ndarray], jnp.ndarray],
        f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        u0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ut0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ic_weight: float = 1.0,
        bc_weight: float = 1.0,
    ):
        self.u_model = u_model
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight

        # Allow forcing and initial conditions to be constant or spatially varying functions.
        self._f_fn = f if callable(f) else self._constant_function(f)
        self._u0_fn = u0 if callable(u0) else self._constant_function(u0)
        self._ut0_fn = ut0 if callable(ut0) else self._constant_function(ut0)

        # Vectorised residual evaluations over batches of points.
        self._vmapped_pde_residual = jax.vmap(self._pde_residual)
        self._vmapped_ic_residual = jax.vmap(self._ic_residual)
        self._vmapped_spatial_bc_residual = jax.vmap(self._spatial_bc_residual)

    def _u(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Scalar network output u(t, x).
        """
        return self.u_model(x).squeeze()

    def _pde_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Pointwise PDE residual:

            (u_tt - \\Delta u - f)^2

        Computed using the Hessian of u.
        """
        H = jax.hessian(self._u)(x)

        # Second derivative in time direction
        u_tt = H[0, 0]

        # Spatial Laplacian from spatial block of Hessian
        laplacian_u = jnp.trace(H[1:, 1:])

        f = self._f_fn(x)
        if jnp.ndim(f) != 0:
            raise ValueError("f should be scalar-valued.")

        return (u_tt - laplacian_u - f) ** 2

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """
        Batch evaluation of PDE residual over interior collocation points.
        """
        return self._vmapped_pde_residual(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Initial condition penalty at t = 0.

        Enforces:
            u(0, x) \\approx u0(x)
            u_t(0, x) \\approx ut0(x)
        """
        u_val = self._u(x)

        # Time derivative at initial time
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
        Homogeneous Dirichlet boundary condition:

            u = 0 on spatial boundary
        """
        return self.bc_weight * self._u(x) ** 2

    def loss_boundary(
        self,
        x_boundary: jnp.ndarray,
        normal_vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Boundary loss combining:
            - initial condition (time boundary)
            - spatial Dirichlet boundary condition

        Classification:
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
