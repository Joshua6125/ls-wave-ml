'''
Gauss-Legendre quadrature integration.
'''
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from typing import Callable, Any
from .base import NDCubeIntegration
from .config import QuadratureConfig

from numpy.polynomial.legendre import leggauss

import sys
import jax
import jax.numpy as jnp


class QuadratureIntegration(NDCubeIntegration):
    """
    Tensor-product Gauss-Legendre quadrature on a space-time hyperrectangle.

    Each coordinate axis is partitioned into `grid_size` sub-intervals and a
    degree-`degree` Gauss-Legendre rule is applied on every segment. The
    resulting multidimensional rule is formed via tensor products.

    Notes
    -----
    The number of quadrature points grows exponentially with dimension:

        (degree * grid_size)^dim

    making this method practical only for low-dimensional problems.

    Boundary condition assumption:
        u = 0 on ∂Ω (Dirichlet).
    """

    def __init__(self, config: QuadratureConfig):
        config.validate()

        self.degree = config.degree
        self.grid_size = config.grid_size

        self.adaptive = config.adaptive_integration
        if self.adaptive:
            print(
                "Warning: Adaptive quadrature not implemented. Ignoring adaptive flag.",
                file=sys.stderr,
            )

        # Space-time dimension: one temporal dimension plus
        # `spatial_dim` spatial dimensions.
        self.spatial_dim = config.spatial_dim
        self.dim = self.spatial_dim + 1

        # Warn about the exponential growth of tensor-product rules.
        if self.dim > 3:
            print(
                f"Warning: {self.dim}-dimensional quadrature with degree {self.degree} "
                f"creates {self.degree**self.dim} points.",
                file=sys.stderr,
            )

        # Lower and upper bounds of the space-time domain.
        self.t_min = config.t_min
        self.t_max = config.t_max
        self.x_min = config.x_min
        self.x_max = config.x_max

        # Time and space domains explicitly separated
        self.domain_min = jnp.array(
            [config.t_min] + [config.x_min] * config.spatial_dim
        )
        self.domain_max = jnp.array(
            [config.t_max] + [config.x_max] * config.spatial_dim
        )

        # Precompute interior and boundary quadrature rules since they
        # remain fixed for all integrations.
        self.points_interior, self.weights_interior = self._setup_quadrature_grids()
        self.boundary_faces = self._setup_boundary_grids()

    def _segmented_1d_rule(
        self,
        a: float,
        b: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Construct a piecewise Gauss-Legendre quadrature rule on [a, b].

        The interval is divided into `grid_size` subintervals and an
        independent degree-`degree` Gauss rule is applied on each segment.

        Returns
        -------
        points :
            Quadrature nodes on [a, b].

        weights :
            Corresponding quadrature weights.
        """

        if self.grid_size < 1:
            raise ValueError("grid_size must be >= 1")

        # Reference Gauss-Legendre nodes and weights on [-1, 1].
        p, w = leggauss(self.degree)
        p = jnp.asarray(p)
        w = jnp.asarray(w)

        # Segment boundaries.
        edges = jnp.linspace(a, b, self.grid_size + 1)

        all_points = []
        all_weights = []

        for left, right in zip(edges[:-1], edges[1:]):
            # Affine map from the reference interval [-1, 1]
            # to the current segment [left, right].
            center = (left + right) / 2.0
            half_width = (right - left) / 2.0

            # Transform nodes and weights to the physical segment.
            seg_points = half_width * p + center
            seg_weights = half_width * w

            all_points.append(seg_points)
            all_weights.append(seg_weights)

        return jnp.concatenate(all_points), jnp.concatenate(all_weights)

    def _setup_quadrature_grids(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Construct the tensor-product quadrature rule on the full domain.

        A segmented 1D Gauss rule is generated for each coordinate axis,
        then combined using tensor products to obtain multidimensional
        quadrature points and weights.
        """

        axis_points = []
        axis_weights = []

        # Build independent 1D quadrature rules for each axis.
        for d in range(self.dim):
            pts_d, wts_d = self._segmented_1d_rule(
                float(self.domain_min[d]),
                float(self.domain_max[d]),
            )
            axis_points.append(pts_d)
            axis_weights.append(wts_d)

        # Tensor-product quadrature points.
        points_mesh = jnp.meshgrid(*axis_points, indexing="ij")
        points = jnp.stack(points_mesh, axis=-1).reshape(-1, self.dim)

        # Tensor-product quadrature weights obtained as the
        # product of 1D weights.
        weight_mesh = jnp.meshgrid(*axis_weights, indexing="ij")
        weights = jnp.prod(jnp.stack(weight_mesh, axis=-1), axis=-1).reshape(-1)

        return points, weights

    def integrate_interior(self, func: Callable[[jnp.ndarray], Any]) -> Any:
        """Integrate over interior using precomputed quadrature rule."""
        # Evaluate function at quadrature points
        func_values = func(self.points_interior)

        # Compute weighted sum
        integral = jax.tree_util.tree_map(
            lambda x: jnp.tensordot(self.weights_interior, x, axes=([0], [0])),
            func_values,
        )
        return integral

    def _setup_boundary_grids(self) -> dict[str, jnp.ndarray]:
        """
        Construct quadrature rules on every boundary face.

        For a face orthogonal to coordinate axis `axis`, quadrature is
        performed over the remaining free coordinates using a lower-
        dimensional tensor-product Gauss rule.

        Each quadrature point is paired with the corresponding outward
        unit normal vector.
        """

        # Special case: the boundary of a 1D interval consists
        # of its two endpoints.
        if self.dim == 1:
            # Boundary of an interval is two points.
            points = jnp.array([[self.domain_min[0]], [self.domain_max[0]]])
            normals = jnp.array([[-1.0], [1.0]])
            weights = jnp.ones(2)
            return {
                "points": points,
                "normals": normals,
                "weights": weights,
            }

        face_points = []
        face_normals = []
        face_weights = []

        for axis in range(self.dim):
            # Coordinates that remain free on the current face.
            free_axes = [d for d in range(self.dim) if d != axis]

            free_points_axes = []
            free_weights_axes = []

            # Build the lower-dimensional tensor-product rule
            # on the free coordinates.
            for d in free_axes:
                pts_d, wts_d = self._segmented_1d_rule(
                    float(self.domain_min[d]),
                    float(self.domain_max[d]),
                )
                free_points_axes.append(pts_d)
                free_weights_axes.append(wts_d)

            # Tensor-product quadrature points on the face.
            free_mesh = jnp.meshgrid(*free_points_axes, indexing="ij")
            free_points = jnp.stack(free_mesh, axis=-1).reshape(-1, self.dim - 1)

            # Tensor-product quadrature weights on the face.
            weight_mesh = jnp.meshgrid(*free_weights_axes, indexing="ij")
            w = jnp.prod(jnp.stack(weight_mesh, axis=-1), axis=-1).reshape(-1)

            for is_max, boundary_value in [
                (False, self.domain_min[axis]),
                (True, self.domain_max[axis]),
            ]:
                # Insert the fixed boundary coordinate.
                pts = jnp.insert(free_points, axis, boundary_value, axis=1)

                # Outward unit normal associated with this face.
                normal = jnp.zeros(self.dim)
                normal = normal.at[axis].set(1.0 if is_max else -1.0)
                normals = jnp.tile(normal, (pts.shape[0], 1))

                face_points.append(pts)
                face_normals.append(normals)
                face_weights.append(w)

        return {
            "points": jnp.concatenate(face_points),
            "normals": jnp.concatenate(face_normals),
            "weights": jnp.concatenate(face_weights),
        }

    def integrate_boundary(
        self, func: Callable[[jnp.ndarray, jnp.ndarray], Any]
    ) -> Any:
        """Integrate function over the cube boundary."""
        func_values = func(
            self.boundary_faces["points"], self.boundary_faces["normals"]
        )

        integral = jax.tree_util.tree_map(
            lambda x: jnp.tensordot(self.boundary_faces["weights"], x, axes=([0], [0])),
            func_values,
        )
        return integral
