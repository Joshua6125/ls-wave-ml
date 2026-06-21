# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from collections import deque
import jax.numpy as jnp
import pytest
import optax

from src.train import Trainer, TrainConfig

pytestmark = pytest.mark.training


# AI-Generated
@pytest.mark.parametrize("use_jit", [True, False])
def test_trainer_initialization_jit_dispatch(dummy_method, dummy_integrator, use_jit):
    config = TrainConfig(use_jit=use_jit)
    optimiser = optax.adam(1e-3)

    trainer = Trainer(dummy_method, dummy_integrator, optimiser, config)

    # Verify that the internal step function is wrapped correctly
    if use_jit:
        assert str(type(trainer._train_step_fn)).count(
            "CompiledFunction"
        ) > 0 or hasattr(trainer._train_step_fn, "lower")
    else:
        assert trainer._train_step_fn == trainer._train_step_impl


# AI-Generated
def test_trainer_init_state(dummy_method, dummy_integrator, default_train_config):
    optimiser = optax.adam(1e-3)
    trainer = Trainer(dummy_method, dummy_integrator, optimiser, default_train_config)

    sample_input = jnp.zeros((1, 2))
    state = trainer._init_state(sample_input)

    assert state.step == 0
    assert "w" in state.params
    assert state.total_training_time == 0.0


# AI-Generated
def test_trainer_loss_with_aux(
    dummy_method, dummy_integrator, default_train_config, dummy_train_state
):
    optimiser = optax.adam(1e-3)
    trainer = Trainer(dummy_method, dummy_integrator, optimiser, default_train_config)

    total, (interior, boundary, next_key) = trainer._loss_with_aux(
        dummy_train_state.params, dummy_train_state.integration_key
    )

    assert isinstance(total, jnp.ndarray)
    assert next_key is not None


# AI-Generated
@pytest.mark.parametrize(
    ("window_values", "expected_convergence"),
    [
        ([1.0, 1.0001, 0.9999, 1.0], True),
        ([1.0, 2.0, 1.5, 3.0], False),
        ([1.0, float("nan"), 1.0], False),
        ([1.0, float("inf"), 1.0], False),
    ],
)
def test_has_converged_logic(
    dummy_method, dummy_integrator, window_values, expected_convergence
):
    config = TrainConfig(
        convergence_check=True, convergence_window_size=4, convergence_rel_tol=1e-2
    )
    trainer = Trainer(dummy_method, dummy_integrator, optax.adam(1e-3), config)

    loss_window = deque(window_values, maxlen=config.convergence_window_size)
    assert trainer._has_converged(loss_window) == expected_convergence


# AI-Generated
def test_fit_executes_epochs_and_logs(dummy_method, dummy_integrator):
    config = TrainConfig(epochs=3, log_every=1, use_jit=False)
    trainer = Trainer(dummy_method, dummy_integrator, optax.adam(1e-3), config)

    called_metrics = []

    def spy_callback(metrics):
        called_metrics.append(metrics)

    sample_input = jnp.zeros((1, 2))
    final_state, history = trainer.fit(sample_input=sample_input, callback=spy_callback)

    assert final_state.step == 3
    assert len(history) == 3
    assert len(called_metrics) == 3
    assert history[0].step == 1
    assert history[2].step == 3


# AI-Generated
def test_fit_terminates_on_max_training_time(dummy_method, dummy_integrator):
    # Enforce zero runtime allowance to trigger immediate timeout extraction
    config = TrainConfig(epochs=100, log_every=1, max_training_time=0.0, use_jit=False)
    trainer = Trainer(dummy_method, dummy_integrator, optax.adam(1e-3), config)

    sample_input = jnp.zeros((1, 2))
    final_state, history = trainer.fit(sample_input=sample_input)

    # Execution must break early within the loop lifecycle
    assert final_state.step < 100


# AI-Generated
def test_fit_terminates_on_early_convergence(dummy_method, dummy_integrator):
    config = TrainConfig(
        epochs=10,
        log_every=1,
        convergence_check=True,
        convergence_window_size=2,
        convergence_rel_tol=1e-1,
        use_jit=False,
    )
    trainer = Trainer(
        dummy_method, dummy_integrator, optax.adam(0.0), config
    )  # 0.0 learning rate means zero change

    sample_input = jnp.zeros((1, 2))
    final_state, history = trainer.fit(sample_input=sample_input)

    # Must exit early once the window criteria are matched
    assert final_state.step < 10
