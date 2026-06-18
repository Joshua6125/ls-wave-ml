from src.models import KAN

import jax.numpy as jnp
import pytest


# AI-Generated
def test_layer_dims_constructed_correctly():
    model = KAN(
        hidden_dim=32,
        num_layers=3,
        input_dim=4,
        output_heads={
            "u": 1,
            "v": 2,
        },
        constrained_heads=[],
    )

    assert model._layer_dims() == [4, 32, 32, 32, 3]


# AI-Generated
@pytest.mark.parametrize(
    ("model_type", "layer_type", "required_parameters"),
    [
        ("original", "base", {"k": 3, "G": 5}),
        ("base", "base", {"k": 3, "G": 5}),
        ("spline", "spline", {"k": 3, "G": 5}),
        ("cheby", "chebyshev", {"D": 3, "flavor": "default"}),
        ("chebyshev", "chebyshev", {"D": 3, "flavor": "default"}),
        ("efficient", "chebyshev", {"D": 3, "flavor": "exact"}),
    ],
)
def test_kan_hparams_mapping(
    model_type,
    layer_type,
    required_parameters,
):
    model = KAN(
        hidden_dim=8,
        num_layers=1,
        input_dim=2,
        output_heads={"u": 1},
        constrained_heads=[],
        model_type=model_type,
    )

    result_layer_type, result_params = model._kan_hparams()

    assert result_layer_type == layer_type
    assert result_params == required_parameters


# AI-Generated
def test_kan_hparams_rejects_unknown_model_type():
    model = KAN(
        hidden_dim=8,
        num_layers=1,
        input_dim=2,
        output_heads={"u": 1},
        constrained_heads=[],
        model_type="invalid",
    )

    with pytest.raises(ValueError, match="Unknown model_type"):
        model._kan_hparams()


# AI-Generated
def test_split_output_heads():
    model = KAN(
        hidden_dim=8,
        num_layers=1,
        input_dim=2,
        output_heads={
            "a": 1,
            "b": 2,
            "c": 1,
        },
        constrained_heads=[],
    )

    y = jnp.arange(16).reshape(4, 4)

    outputs = model._split_output_heads(y)

    assert outputs["a"].shape == (4, 1)
    assert outputs["b"].shape == (4, 2)
    assert outputs["c"].shape == (4, 1)

    assert jnp.array_equal(outputs["a"], y[:, 0:1])
    assert jnp.array_equal(outputs["b"], y[:, 1:3])
    assert jnp.array_equal(outputs["c"], y[:, 3:4])


# AI-Generated
def test_kan_forward_output_shapes(
    kan_model,
    rng,
    sample_batch,
):
    variables = kan_model.init(rng, sample_batch)
    outputs = kan_model.apply(variables, sample_batch)

    assert outputs["u"].shape == (3, 1)
    assert outputs["v"].shape == (3, 2)


# AI-Generated
def test_kan_returns_all_requested_heads(
    kan_model,
    rng,
    sample_batch,
):
    variables = kan_model.init(rng, sample_batch)
    outputs = kan_model.apply(variables, sample_batch)

    assert set(outputs.keys()) == {"u", "v"}


# AI-Generated
def test_kan_accepts_unbatched_input(
    kan_model,
    rng,
    sample_point,
):
    variables = kan_model.init(rng, sample_point)
    outputs = kan_model.apply(variables, sample_point)

    assert outputs["u"].shape == (1,)
    assert outputs["v"].shape == (2,)


# AI-Generated
def test_kan_boundary_constraint_reduces_output_near_boundary(rng):
    model = KAN(
        hidden_dim=8,
        num_layers=1,
        input_dim=2,
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
def test_kan_constraint_ignored_for_missing_head(rng):
    model = KAN(
        hidden_dim=8,
        num_layers=1,
        input_dim=2,
        output_heads={"u": 1},
        constrained_heads=["missing"],
    )

    x = jnp.array([[0.5, 0.5]])

    variables = model.init(rng, x)
    outputs = model.apply(variables, x)

    assert type(outputs) == dict

    assert set(outputs.keys()) == {"u"}


# AI-Generated
def test_layer_dims_respects_sorted_head_order():
    model = KAN(
        hidden_dim=8,
        num_layers=2,
        input_dim=3,
        output_heads={
            "z": 2,
            "a": 1,
        },
        constrained_heads=[],
    )

    assert model._layer_dims() == [3, 8, 8, 3]


# AI-Generated
def test_split_output_heads_uses_sorted_order():
    model = KAN(
        hidden_dim=8,
        num_layers=1,
        input_dim=2,
        output_heads={
            "z": 2,
            "a": 1,
        },
        constrained_heads=[],
    )

    y = jnp.array([[1.0, 2.0, 3.0]])

    outputs = model._split_output_heads(y)

    assert jnp.array_equal(outputs["a"], jnp.array([[1.0]]))
    assert jnp.array_equal(outputs["z"], jnp.array([[2.0, 3.0]]))
