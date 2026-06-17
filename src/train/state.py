from dataclasses import dataclass
from typing import Any

import jax
import optax
from optax import Schedule


@dataclass(frozen=True)
class TrainConfig:
    """
    Configuration for training loop execution.

    This class defines optimisation hyperparameters, runtime behaviour,
    and optional convergence control logic.
    """

    epochs: int = 1000
    max_training_time: float = 60

    learning_rate: Schedule = optax.exponential_decay(
        init_value=1e-4,
        transition_steps=1000,
        decay_rate=0.95,
        staircase=True,
    )

    optimiser: str = "adamw"
    seed: int = 0

    log_every: int = 0
    use_jit: bool = True

    convergence_check: bool = False
    convergence_window_size: int = 100
    convergence_rel_tol: float = 1e-3

    def validate(self) -> None:
        """Validate consistency of training hyperparameters."""
        if self.epochs <= 0:
            raise ValueError("epochs must be strictly positive")

        if self.max_training_time <= 0:
            raise ValueError("max_training_time must be strictly positive")

        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")

        if self.convergence_check:
            if self.convergence_window_size <= 0:
                raise ValueError("convergence_window_size must be strictly positive")
            if self.convergence_rel_tol <= 0:
                raise ValueError("convergence_rel_tol must be strictly positive")


@dataclass(frozen=True)
class TrainState:
    """
    Immutable training state used during optimisation.

    Encapsulates all mutable quantities of the training loop:
    model parameters, optimiser state, and stochastic keys.
    """

    step: int
    params: Any
    opt_state: optax.OptState
    integration_key: jax.Array
    total_training_time: float = 0.0

    def apply_gradients(
        self,
        grads: Any,
        optimiser: optax.GradientTransformation,
    ) -> "TrainState":
        """
        Apply one optimisation step using Optax.

        Returns a new TrainState with updated parameters and optimiser state.
        """
        updates, opt_state = optimiser.update(grads, self.opt_state, self.params)

        params = optax.apply_updates(self.params, updates)

        return TrainState(
            step=self.step + 1,
            params=params,
            opt_state=opt_state,
            integration_key=self.integration_key,
        )


def get_optimiser(config: TrainConfig) -> optax.GradientTransformation:
    """
    Construct an Optax optimiser from configuration.
    """
    config.validate()

    if config.optimiser == "adam":
        return optax.adam(config.learning_rate)

    if config.optimiser == "adamw":
        return optax.adamw(config.learning_rate)

    if config.optimiser == "sgd":
        return optax.sgd(config.learning_rate)

    raise ValueError(f"Unknown optimiser: '{config.optimiser}'")
