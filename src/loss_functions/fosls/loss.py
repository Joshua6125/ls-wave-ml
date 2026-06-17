from typing import Callable

import jax
import jax.numpy as jnp

from ..base import Loss


class FOSLSLoss(Loss):
    """
    First-Order System Least Squares (FOSLS) loss for the acoustic wave system.

    The method rewrites the wave equation as a first-order system
    and minimises the L2 norm of the residuals over space-time.

    Initial conditions can be enforced via a separate weighted penalty.
    """

    def __init__(
        self,
        model: Callable[[jnp.ndarray], jnp.ndarray],
        f: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        g: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        v0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        sigma0: float | Callable[[jnp.ndarray], jnp.ndarray] = 0.0,
        ic_weight: float = 1.0,
    ):
        self.model = model
        self.ic_weight = ic_weight

        # Allow constant or spatially varying coefficients.
        self._f_fn = f if callable(f) else self._constant_function(f)
        self._g_fn = g if callable(g) else self._constant_function(g)
        self._v0_fn = v0 if callable(v0) else self._constant_function(v0)
        self._sigma0_fn = (
            sigma0 if callable(sigma0) else self._constant_function(sigma0)
        )

        # Vectorized residual evaluation over batches of points.
        self._vmapped_interior_residual = jax.vmap(self._interior_residual)
        self._vmapped_ic_residual = jax.vmap(self._ic_residual)

    def _interior_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Pointwise squared residual of the first-order PDE system.
        """
        jac = jax.jacobian(self.model)(x)

        # Time derivative of scalar field v.
        dt_v = jac[0, 0]

        # Spatial gradient of v.
        grad_v = jac[0, 1:]

        # Time derivative of vector field σ.
        dt_sigma = jac[1:, 0]

        # Divergence of σ via trace of spatial Jacobian.
        div_sigma = jnp.trace(jac[1:, 1:])

        f = self._f_fn(x)
        if jnp.ndim(f) != 0:
            raise ValueError("f must be scalar-valued.")

        g = self._g_fn(x)

        # Allow isotropic forcing if scalar is provided.
        if jnp.ndim(g) == 0:
            g = g * jnp.ones_like(grad_v)

        if jnp.shape(g) != jnp.shape(grad_v):
            raise ValueError("g must match shape of spatial gradient.")

        res_v = dt_v - div_sigma - f
        res_sigma = dt_sigma - grad_v - g

        return res_v**2 + jnp.sum(res_sigma**2)

    def loss_interior(self, x_interior: jnp.ndarray) -> jnp.ndarray:
        """Batch interior residual evaluation."""
        return self._vmapped_interior_residual(x_interior)

    def _ic_residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Pointwise initial condition penalty at t = t_min.
        """
        out = self.model(x)
        v_val = out[0]
        sigma_val = out[1:]

        v0_val = self._v0_fn(x)
        if jnp.ndim(v0_val) != 0:
            raise ValueError("v0 must be scalar-valued.")

        sigma0_val = self._sigma0_fn(x)

        # Allow scalar IC to be broadcast to vector field.
        if jnp.ndim(sigma0_val) == 0:
            sigma0_val = sigma0_val * jnp.ones_like(sigma_val)

        if jnp.shape(sigma0_val) != jnp.shape(sigma_val):
            raise ValueError("sigma0 must match sigma shape.")

        return self.ic_weight * (
            (v_val - v0_val) ** 2 + jnp.sum((sigma_val - sigma0_val) ** 2)
        )

    def loss_boundary(
        self,
        x_boundary: jnp.ndarray,
        normal_vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Boundary contribution to the loss.

        Only the initial time boundary is penalised in this formulation.
        """
        is_ic = normal_vector[:, 0] < 0

        return jnp.where(
            is_ic,
            self._vmapped_ic_residual(x_boundary),
            0.0,
        )
