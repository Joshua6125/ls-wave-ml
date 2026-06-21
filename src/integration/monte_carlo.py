"""
Monte-Carlo integration.
"""
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from typing import Callable, Any
from .base import NDCubeIntegration
from .config import MonteCarloConfig

import jax
import jax.numpy as jnp
import jax.random as jr


class MonteCarloIntegration(NDCubeIntegration):
    """
    Monte Carlo integration on a space-time hyperrectangle

        [t_min, t_max] x [x_min, x_max]^d.

    Interior integrals are estimated using uniform random sampling
    over the full domain. Boundary integrals are estimated by sampling
    uniformly on each boundary face and weighting by the corresponding
    face measure.

    Boundary condition assumption:
        u = 0 on the boundary of the space (Dirichlet).
    """

    def __init__(self, config: MonteCarloConfig):
        config.validate()

        self.spatial_dim = config.spatial_dim
        self.dim = self.spatial_dim + 1

        self.interior_samples = config.interior_samples
        self.boundary_samples = config.boundary_samples

        self.t_min = config.t_min
        self.t_max = config.t_max
        self.x_min = config.x_min
        self.x_max = config.x_max

        # Avoid sampling exactly at zero when used in other
        # operations that may contain singularities.
        self.eps = 1e-8

        self.domain_min = jnp.array(
            [config.t_min] + [config.x_min] * config.spatial_dim
        )
        self.domain_max = jnp.array(
            [config.t_max] + [config.x_max] * config.spatial_dim
        )

        # Measure of the full integration domain.
        spatial_volume = (self.x_max - self.x_min) ** self.spatial_dim
        self.volume = (self.t_max - self.t_min) * spatial_volume

        # Face measures used for boundary integration weights.
        self.time_face_area = spatial_volume
        self.spatial_face_area = (self.t_max - self.t_min) * (
            (self.x_max - self.x_min) ** (self.spatial_dim - 1)
        )

    def _sample_interior(self) -> jnp.ndarray:
        """
        Draw uniform samples from the interior of the space-time domain.

        Returns
        -------
        points : (N, dim) array
            Random points distributed uniformly over the domain.
        """
        self.key, subkey = jr.split(self.key)

        # Uniform samples in the unit hypercube [eps, 1)^dim.
        samples = jr.uniform(
            subkey,
            shape=(self.interior_samples, self.dim),
            minval=self.eps,
        )

        # Affine map from the unit cube to the physical domain.
        points = self.domain_min + samples * (self.domain_max - self.domain_min)

        return points

    def _setup_boundary_samples(self) -> dict:
        """
        Sample uniformly from every boundary face.

        For each coordinate axis, samples are generated on both the
        lower and upper face. Each sample is paired with its outward
        unit normal and Monte Carlo quadrature weight.
        """
        face_points = []
        face_normals = []
        face_weights = []

        for axis in range(self.dim):
            bound_min = self.domain_min[axis]
            bound_max = self.domain_max[axis]

            # Measure of a face orthogonal to this axis.
            area = self.time_face_area if axis == 0 else self.spatial_face_area

            # Monte Carlo weight for a single sample on this face.
            weight_per_sample = area / self.boundary_samples

            for is_max, boundary_value in [
                (False, bound_min),
                (True, bound_max),
            ]:
                self.key, subkey = jr.split(self.key)

                # Sample free coordinates on the (dim - 1)-dimensional face.
                samples = jr.uniform(
                    subkey,
                    shape=(self.boundary_samples, self.dim - 1),
                )

                free_min = jnp.concatenate(
                    [self.domain_min[:axis], self.domain_min[axis + 1 :]]
                )
                free_max = jnp.concatenate(
                    [self.domain_max[:axis], self.domain_max[axis + 1 :]]
                )

                # Map unit-cube samples to the face coordinates.
                samples = free_min + samples * (free_max - free_min)

                # Reinsert the fixed boundary coordinate.
                pts = jnp.insert(
                    samples,
                    axis,
                    boundary_value,
                    axis=1,
                )

                # Outward unit normal of the current face.
                normal = jnp.zeros(self.dim)
                normal = normal.at[axis].set(1.0 if is_max else -1.0)

                normals = jnp.tile(
                    normal,
                    (self.boundary_samples, 1),
                )

                face_points.append(pts)
                face_normals.append(normals)
                face_weights.append(
                    jnp.full(
                        self.boundary_samples,
                        weight_per_sample,
                    )
                )

        return {
            "points": jnp.concatenate(face_points),
            "normals": jnp.concatenate(face_normals),
            "weights": jnp.concatenate(face_weights),
        }

    def integrate_interior(
        self,
        func: Callable[[jnp.ndarray], Any],
    ) -> Any:
        """
        Estimate the integral of func over the space using
        the standard Monte Carlo estimator.
        """
        points_interior = self._sample_interior()

        func_values = func(points_interior)

        factor = self.volume / self.interior_samples

        return jax.tree_util.tree_map(
            lambda x: factor * jnp.sum(x, axis=0),
            func_values,
        )

    def integrate_boundary(
        self,
        func: Callable[[jnp.ndarray, jnp.ndarray], Any],
    ) -> Any:
        """
        Estimate the boundary integral of func by sampling uniformly
        on every boundary face and summing the corresponding weighted
        contributions.
        """
        boundary_data = self._setup_boundary_samples()

        func_values = func(
            boundary_data["points"],
            boundary_data["normals"],
        )

        weights = boundary_data["weights"]

        return jax.tree_util.tree_map(
            lambda x: jnp.tensordot(
                weights,
                x,
                axes=([0], [0]),
            ),
            func_values,
        )

    def integrate(
        self,
        interior_func: Callable[[jnp.ndarray], Any],
        boundary_func: Callable[[jnp.ndarray, jnp.ndarray], Any],
        rng_key: jax.Array | None = jax.random.PRNGKey(42),
    ) -> tuple[Any, Any]:
        """
        Compute interior and boundary integrals using a supplied RNG key.

        Explicit key threading ensures reproducible Monte Carlo samples.
        """
        if rng_key is None:
            raise ValueError("rng_key may not be None in Monte Carlo Integration.")

        self.key = rng_key

        interior_loss = self.integrate_interior(interior_func)
        boundary_loss = self.integrate_boundary(boundary_func)

        return interior_loss, boundary_loss
