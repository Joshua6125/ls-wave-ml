from typing import Callable
from .base import NDCubeIntegration
from ..config import Config

import jax
import jax.numpy as jnp
import jax.random as jr


class MonteCarloIntegration(NDCubeIntegration):
    '''
    Monte Carlo integration for n-D cube integration.

    Complexity: O(samples) evaluations.
    Convergence: O(1/sqrt(samples)) error.
    Advantage: Dimension-independent scaling, simple implementation.
    Disadvantage: Slower convergence than quadrature for smooth functions.

    Boundary: Assumes Dirichlet BC (u=0 on boundary).
    '''

    def __init__(self, config: Config):
        assert config.dim > 0, "dim must be positive"
        assert config.monte_carlo_interior_samples > 0, "interior_samples must be positive"
        assert config.monte_carlo_boundary_samples > 0, "boundary_samples must be positive"
        assert config.x_min < config.x_max, "x_min must be < x_max"

        self.dim = config.dim
        self.interior_samples = config.monte_carlo_interior_samples
        self.boundary_samples = config.monte_carlo_boundary_samples
        self.x_min = config.x_min
        self.x_max = config.x_max
        self.seed = config.monte_carlo_seed

        self.key = jr.PRNGKey(self.seed)

        self.volume = (self.x_max - self.x_min) ** self.dim
        self.face_area = (self.x_max - self.x_min) ** (self.dim - 1)

        # Set up sampling for interior and boundary
        self.points_interior, self.key = self._sample_interior()
        self.boundary_data = self._setup_boundary_samples()

    def _sample_interior(self) -> tuple[jnp.ndarray, jax.Array]:
        """Generate random samples uniformly in the domain interior."""
        self.key, subkey = jr.split(self.key)

        # Sample uniform in [0, 1)^dim
        samples = jr.uniform(subkey, shape=(self.interior_samples, self.dim))

        # Transform to [x_min, x_max]^dim
        points = self.x_min + samples * (self.x_max - self.x_min)
        return points, self.key

    def _setup_boundary_samples(self) -> dict:
        """Generate random samples on all boundary faces."""
        face_points = []
        face_normals = []

        for axis in range(self.dim):
            for boundary_value in [self.x_min, self.x_max]:
                self.key, subkey = jr.split(self.key)

                # Sample random points on the (dim-1)-dimensional face
                # We need dim-1 free dimensions
                samples = jr.uniform(subkey, shape=(self.boundary_samples, self.dim - 1))
                samples = self.x_min + samples * (self.x_max - self.x_min)

                # Insert the fixed boundary coordinate at the correct axis
                pts = jnp.insert(samples, axis, boundary_value, axis=1)

                # Compute outward-pointing normal for this face
                normal = jnp.zeros(self.dim)
                normal = normal.at[axis].set(1.0 if boundary_value == self.x_max else -1.0)
                normals = jnp.tile(normal, (self.boundary_samples, 1))

                face_points.append(pts)
                face_normals.append(normals)

        return {
            "points": jnp.concatenate(face_points),
            "normals": jnp.concatenate(face_normals),
        }

    @staticmethod
    @jax.jit
    def _integrate_interior(func, points, volume, n_samples):
        """Compute interior integral using Monte Carlo."""
        func_values = func(points)
        # Monte Carlo: volume * (1/n) * sum(f)
        integral = (volume / n_samples) * jnp.sum(func_values)
        return integral

    def integrate_interior(
        self, func: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        """Integrate over interior using Monte Carlo sampling."""
        return self._integrate_interior(
            func, self.points_interior, self.volume, self.interior_samples
        )

    @staticmethod
    @jax.jit
    def _integrate_boundary(func, points, normals, face_area, n_faces, n_samples_per_face):
        """Compute boundary integral using Monte Carlo."""
        func_values = func(points, normals)
        # Monte Carlo on boundary: sum over faces, each with area * (1/n) * sum(f)
        integral = (face_area / n_samples_per_face) * jnp.sum(func_values)
        return integral

    def integrate_boundary(
        self, func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        """Integrate over boundary using Monte Carlo sampling."""
        return self._integrate_boundary(
            func,
            self.boundary_data["points"],
            self.boundary_data["normals"],
            self.face_area,
            2 * self.dim,
            self.boundary_samples,
        )
