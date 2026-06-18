import pytest
import optax
import jax.numpy as jnp

from src.train import TrainConfig, get_optimiser

pytestmark = pytest.mark.training


# AI-Generated
@pytest.mark.parametrize(
    ("field", "value", "match_message"),
    [
        ("epochs", 0, "epochs must be strictly positive"),
        ("epochs", -10, "epochs must be strictly positive"),
        ("max_training_time", 0.0, "max_training_time must be strictly positive"),
        ("max_training_time", -1.5, "max_training_time must be strictly positive"),
        ("log_every", -1, "log_every must be non-negative"),
        ("convergence_window_size", 0, "convergence_window_size must be strictly positive"),
        ("convergence_rel_tol", 0.0, "convergence_rel_tol must be strictly positive"),
    ],
)
def test_train_config_validation_rejects_invalid_inputs(field, value, match_message):
    kwargs = {field: value}
    # Force convergence options to trigger validation checks when needed
    if "convergence" in field:
        kwargs["convergence_check"] = True

    config = TrainConfig(**kwargs)
    with pytest.raises(ValueError, match=match_message):
        config.validate()


# AI-Generated
@pytest.mark.parametrize("optimiser_name", ["adam", "adamw", "sgd"])
def test_get_optimiser_resolves_valid_strings(optimiser_name):
    config = TrainConfig(optimiser=optimiser_name)
    optimiser = get_optimiser(config)
    assert isinstance(optimiser, optax.GradientTransformation)


# AI-Generated
def test_get_optimiser_raises_on_unknown_string():
    config = TrainConfig(optimiser="invalid_opt")
    with pytest.raises(ValueError, match="Unknown optimiser: 'invalid_opt'"):
        get_optimiser(config)


# AI-Generated
def test_train_state_apply_gradients(dummy_train_state):
    optimiser = optax.sgd(1e-1)
    grads = {"w": jnp.array([-1.0, -1.0])}

    next_state = dummy_train_state.apply_gradients(grads, optimiser)

    assert next_state.step == dummy_train_state.step + 1
    # Expected: 1.0 - (0.1 * -1.0) = 1.1
    assert pytest.approx(float(next_state.params["w"][0])) == 1.1