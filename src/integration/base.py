'''
Base class for N-D integration domains for hypercubes.
'''
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from abc import ABC, abstractmethod
from typing import Callable, Any

import jax
import jax.numpy as jnp


class NDCubeIntegration(ABC):
    """
    Abstract interface for integration over a space-time hyperrectangle.

    Implementations provide numerical approximations of the integral of a
    functions over the domain interior and an integral over the domain
    boundary, with the outward unit normal.

    Integrators may use different numerical schemes (e.g. Monte Carlo,
    tensor-product quadrature), but should expose a common interface.
    """

    @abstractmethod
    def integrate_interior(
        self,
        func: Callable[[jnp.ndarray], Any],
    ) -> Any:
        """
        Integrate a function over the interior of the domain.

        Parameters
        ----------
        func
            Callable accepting an array of evaluation points with shape
            (N, dim) and returning values to be integrated.

        Returns
        -------
        Any
            Integral estimate with the same pytree structure as the
            function output.
        """
        pass

    @abstractmethod
    def integrate_boundary(
        self,
        func: Callable[[jnp.ndarray, jnp.ndarray], Any],
    ) -> Any:
        """
        Integrate a function over the domain boundary.

        Parameters
        ----------
        func
            Callable accepting boundary points and their associated
            outward unit normals.

        Returns
        -------
        Any
            Boundary integral estimate with the same pytree structure as
            the function output.
        """
        pass

    def integrate(
        self,
        interior_func: Callable[[jnp.ndarray], Any],
        boundary_func: Callable[[jnp.ndarray, jnp.ndarray], Any],
        rng_key: jax.Array | None = None,
    ) -> tuple[Any, Any]:
        """
        Compute both interior and boundary integrals.

        Parameters
        ----------
        interior_func
            Function to integrate over the interior domain.

        boundary_func
            Function to integrate over the boundary.

        rng_key
            Optional random number generator key. Deterministic
            integration methods may ignore this argument.

        Returns
        -------
        tuple[Any, Any]
            Pair consisting of the interior and boundary integral
            estimates.

        Notes
        -----
        Combination of the returned quantities into a single loss value
        is intentionally left to higher-level training logic.
        """
        loss_interior = self.integrate_interior(interior_func)
        loss_boundary = self.integrate_boundary(boundary_func)

        return loss_interior, loss_boundary
