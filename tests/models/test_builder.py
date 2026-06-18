from dataclasses import replace

import pytest

from src.models import (
    BuiltModelAdapter,
    MLPConfig,
    KANConfig,
    build_model,
)


# AI-Generated
def test_build_model_returns_mlp_adapter(
    mlp_config,
):
    model = build_model(mlp_config)

    assert isinstance(model, BuiltModelAdapter)


# AI-Generated
def test_build_model_returns_kan_adapter(
    kan_config,
):
    model = build_model(kan_config)

    assert isinstance(model, BuiltModelAdapter)


# AI-Generated
def test_built_mlp_produces_named_outputs(
    mlp_config,
    rng,
    sample_batch,
):
    model = build_model(mlp_config)

    params = model.init(rng, sample_batch)
    outputs = model.apply(params, sample_batch)

    assert set(outputs.keys()) == {"u", "v"}


# AI-Generated
def test_built_kan_produces_named_outputs(
    kan_config,
    rng,
    sample_batch,
):
    model = build_model(kan_config)

    params = model.init(rng, sample_batch)
    outputs = model.apply(params, sample_batch)

    assert set(outputs.keys()) == {"u", "v"}


# AI-Generated
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_dim", 0, "hidden_dim must be strictly positive"),
        ("num_layers", 0, "num_layers must be strictly positive"),
    ],
)
def test_base_model_config_validation_rejects_invalid_values(
    mlp_config,
    field,
    value,
    message,
):
    bad_cfg = replace(mlp_config, **{field: value})

    with pytest.raises(ValueError, match=message):
        build_model(bad_cfg)


# AI-Generated
def test_validation_rejects_empty_output_heads(
    mlp_config,
):
    bad_cfg = replace(
        mlp_config,
        output_heads={},
    )

    with pytest.raises(ValueError, match="output_heads must be non-empty"):
        build_model(bad_cfg)


# AI-Generated
def test_validation_rejects_empty_output_head_name(
    mlp_config,
):
    bad_cfg = replace(
        mlp_config,
        output_heads={"": 1},
    )

    with pytest.raises(ValueError, match="output head names must be non-empty"):
        build_model(bad_cfg)


# AI-Generated
def test_validation_rejects_non_positive_output_dimension(
    mlp_config,
):
    bad_cfg = replace(
        mlp_config,
        output_heads={"u": 0},
    )

    with pytest.raises(
        ValueError,
        match="each output head dimension must be strictly positive",
    ):
        build_model(bad_cfg)


# AI-Generated
def test_kan_validation_rejects_non_positive_input_dim(
    kan_config,
):
    bad_cfg = replace(
        kan_config,
        input_dim=0,
    )

    with pytest.raises(
        ValueError,
        match="input_dim must be strictly positive",
    ):
        build_model(bad_cfg)


# AI-Generated
@pytest.mark.parametrize(
    "model_type",
    [
        "efficient",
        "cheby",
        "chebyshev",
    ],
)
def test_chebyshev_models_reject_negative_degree(
    kan_config,
    model_type,
):
    bad_cfg = replace(
        kan_config,
        model_type=model_type,
        degree=-1,
    )

    with pytest.raises(
        ValueError,
        match="degree of Chebyshev polynomials must be non-negative",
    ):
        build_model(bad_cfg)


# AI-Generated
@pytest.mark.parametrize(
    "model_type",
    [
        "original",
        "base",
        "spline",
    ],
)
def test_spline_models_reject_non_positive_grid_size(
    kan_config,
    model_type,
):
    bad_cfg = replace(
        kan_config,
        model_type=model_type,
        grid_size=0,
    )

    with pytest.raises(
        ValueError,
        match="grid_size must be strictly positive",
    ):
        build_model(bad_cfg)
