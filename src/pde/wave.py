import jax.numpy as jnp
import jax

from config import DIM


def domain() -> dict[str, tuple[float, float] | int]:
    return {
        "t": (0.0, 1.0),
        "x": (0.0, 1.0),
        "dim": DIM
    }


def initial_condition(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    '''
    Initial conditions for the acoustic wave equation.
     - v_0 is 0.
     - sigma is separable product of sines.

    :param x: spatial cooridinate slice
    '''
    v0 = jnp.zeros_like(x)
    sigma0 = jnp.prod(jnp.sin(jnp.pi * x), axis=-1)
    return v0, sigma0


def boundary_condition(x: jax.Array) -> jax.Array:
    # Not sure if this should just return
    # - 0
    # - 0 for each time-step
    # - or 0 for each boundary value for each time-step.
    raise NotImplementedError


def operatorA(v_t: jax.Array, sigma_t: jax.Array, div_sigma: jax.Array, div_v: jax.Array) -> tuple[jax.Array, jax.Array]:
    '''
    implement first-order acoustic operator A_0:

    d_t v - div s

    d_t s - div v

    :param v_t: time derivative of v
    :param sigma_t: time derivative of sigma
    :param div_v: divergence of v
    :param div_sigma: divergence of sigma
    '''
    r1 = v_t - div_sigma
    r2 = sigma_t - div_v
    return r1, r2


def analytical_sigma(t: jax.Array, x: jax.Array) -> jax.Array:
    '''
    Calculates the analytical solution for sigma for the wave equation in 1 dimension

    :param t: time
    :param x: place
    '''

    # Dimension should be 1 for analytical solution
    assert DIM == 1

    return jnp.cos(jnp.pi * t) * jnp.sin(jnp.pi * x)


def analytical_v(t: jax.Array, x: jax.Array) -> jax.Array:
    '''
    Calculates the analytical solution for v for the wave equation in 1 dimension

    :param t: time
    :param x: place
    '''

    # Dimension should be 1 for analytical solution
    assert DIM == 1

    return -jnp.sin(jnp.pi * t) * jnp.cos(jnp.pi * x)
