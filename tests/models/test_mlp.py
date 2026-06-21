# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from src.models import MLP

import jax.numpy as jnp


# AI-Generated
def test_mlp_output_heads_have_expected_shapes(
    mlp_model,
    rng,
    sample_batch,
):
    variables = mlp_model.init(rng, sample_batch)
    outputs = mlp_model.apply(variables, sample_batch)

    assert outputs["u"].shape == (3, 1)
    assert outputs["v"].shape == (3, 2)


# AI-Generated
def test_mlp_returns_all_requested_heads(
    mlp_model,
    rng,
    sample_batch,
):
    variables = mlp_model.init(rng, sample_batch)
    outputs = mlp_model.apply(variables, sample_batch)

    assert set(outputs.keys()) == {"u", "v"}


# AI-Generated
def test_mlp_boundary_constraint_reduces_output_near_boundary(rng):
    model = MLP(
        hidden_dim=8,
        num_layers=1,
        output_heads={"u": 1},
        constrained_heads=["u"],
    )

    interior = jnp.array([[0.5, 0.5]])
    boundary = jnp.array([[0.5, 0.0]])

    variables = model.init(rng, interior)

    interior_value = model.apply(variables, interior)
    boundary_value = model.apply(variables, boundary)

    assert type(interior_value) == dict
    assert type(boundary_value) == dict

    assert jnp.abs(boundary_value["u"]).max() < jnp.abs(interior_value["u"]).max()


# AI-Generated
def test_mlp_unconstrained_head_not_forced_to_vanish(rng):
    model = MLP(
        hidden_dim=8,
        num_layers=1,
        output_heads={
            "u": 1,
            "v": 1,
        },
        constrained_heads=["u"],
    )

    point = jnp.array([[0.5, 0.0]])

    variables = model.init(rng, point)
    outputs = model.apply(variables, point)

    assert type(outputs) == dict

    assert outputs["u"].shape == (1, 1)
    assert outputs["v"].shape == (1, 1)


# AI-Generated
def test_mlp_constraint_ignored_for_missing_head(rng):
    model = MLP(
        hidden_dim=8,
        num_layers=1,
        output_heads={"u": 1},
        constrained_heads=["missing_head"],
    )

    x = jnp.array([[0.5, 0.5]])

    variables = model.init(rng, x)
    outputs = model.apply(variables, x)

    assert type(outputs) == dict

    assert set(outputs.keys()) == {"u"}


# AI-Generated
def test_mlp_handles_multi_dimensional_heads(
    rng,
):
    model = MLP(
        hidden_dim=8,
        num_layers=1,
        output_heads={
            "scalar": 1,
            "vector": 3,
        },
        constrained_heads=[],
    )

    x = jnp.ones((4, 2))

    variables = model.init(rng, x)
    outputs = model.apply(variables, x)

    assert type(outputs) == dict

    assert outputs["scalar"].shape == (4, 1)
    assert outputs["vector"].shape == (4, 3)

# AI-Generated
def test_mlp_handles_single_sample_input(
    rng,
):
    model = MLP(
        hidden_dim=8,
        num_layers=1,
        output_heads={"u": 1},
        constrained_heads=[],
    )

    x = jnp.array([0.5, 0.5])

    variables = model.init(rng, x)
    outputs = model.apply(variables, x)

    assert type(outputs) == dict

    assert outputs["u"].shape == (1,)
