import jax
import jax.numpy as jnp
import sys

from numpy.polynomial.legendre import leggauss


def gauss_legendre(
    dim: int,
    deg: int=32,
    interval_start: float=0,
    interval_end: float=1
) -> tuple[jax.Array, jax.Array]:
    '''
    Generates D-dimensional sampling grids in the shape of a cube
    for the gauss-legendre quadrature

    :param dim: Dimension of mesh
    :type dim: int
    :param deg: Sampling degree
    :type deg: int
    :param interval_start: Beginning of mesh in each dimension
    :type interval_start: float
    :param interval_end: End of mesh in each dimension
    :type interval_end: float
    :return: Returns tuple of (points, weights)
    :rtype: tuple[Array, Array]
    '''
    assert interval_start < interval_end
    assert deg > 0
    assert dim > 0

    if dim > 3:
        print(f"Warning: {dim}-dimensional quadrature with degree {deg} "
              f"creates {deg**dim} points.", file=sys.stderr)

    # Generate gauss-legendre sample points on [-1, 1]
    lg_samp = leggauss(deg)
    p = jnp.array(lg_samp[0])
    w = jnp.array(lg_samp[1])

    # Transform sample interval
    center = (interval_end + interval_start) / 2.0
    half_width = (interval_end - interval_start) / 2.0
    p_transformed = half_width * p + center

    # Transform weight accordingly as well
    w_transformed = w * half_width

    # Create sample mesh
    points_mesh_axis = [p_transformed] * dim
    points_mesh = jnp.meshgrid(*points_mesh_axis)
    points = jnp.stack(points_mesh, axis=-1).reshape(-1, dim)

    # Create weight mesh
    weight_mesh_axis = [w_transformed] * dim
    weight_mesh = jnp.meshgrid(*weight_mesh_axis)
    weights = jnp.stack(weight_mesh, axis=-1)
    weights = jnp.prod(weights, axis=-1).reshape(-1)

    return points, weights


def uniform(
    dim: int,
    deg: int=32,
    interval_start: float=0,
    interval_end: float=1
) -> jax.Array:
    '''
    Generates a uniform sampling grid in D-dimensional space

    :param dim: Dimension of mesh
    :type dim: int
    :param deg: Sampling degree
    :type deg: int
    :param interval_start: Beginning of mesh in each dimension
    :type interval_start: float
    :param interval_end: End of mesh in each dimension
    :type interval_end: float
    :return: Sample point mesh
    :rtype: Array
    '''
    assert interval_start < interval_end
    assert deg > 0
    assert dim > 0

    if dim > 3:
        print(f"Warning: {dim}-dimensional quadrature with degree {deg} "
              f"creates {deg**dim} points.", file=sys.stderr)

    # Generate samples in one axis
    p = jnp.linspace(interval_start, interval_end, deg)

    # Create sample mesh
    points_mesh_axis = [p] * dim
    points_mesh = jnp.meshgrid(*points_mesh_axis)
    points = jnp.stack(points_mesh, axis=-1).reshape(-1, dim)

    return points