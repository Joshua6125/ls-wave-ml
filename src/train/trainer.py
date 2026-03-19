from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ..integration import NDCubeIntegration
from .methods import TrainingMethod
from .state import TrainConfig, TrainState


@dataclass(frozen=True)
class TrainStepMetrics:
    """Metrics tracked at each optimization step."""

    step: int
    total_loss: float
    interior_loss: float
    boundary_loss: float


class Trainer:
    """Generic training loop independent of method and integration choices."""

    def __init__(
            self,
            method: TrainingMethod,
            integrator: NDCubeIntegration,
            optimizer: optax.GradientTransformation,
            train_cfg: TrainConfig,
        ):
        self.method = method
        self.integrator = integrator
        self.optimizer = optimizer
        self.train_cfg = train_cfg
        self.train_cfg.validate()

        if self.train_cfg.use_jit:
            self._train_step_impl = jax.jit(self._train_step_impl)

    def init_state(self, sample_input: jnp.ndarray) -> TrainState:
        """Initialize parameters, optimizer state, and RNG."""
        rng = jr.PRNGKey(self.train_cfg.seed)
        rng, init_key = jr.split(rng)
        params = self.method.init_params(init_key, sample_input)
        opt_state = self.optimizer.init(params)
        return TrainState(step=0, params=params, opt_state=opt_state, rng_key=rng)

    def _loss_with_aux(self, params: Any) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        interior_fn, boundary_fn = self.method.loss_functions(params)
        total, interior, boundary = self.integrator.integrate(interior_fn, boundary_fn)
        return total, (interior, boundary)

    def _train_step_impl(
            self,
            params: Any,
            opt_state: optax.OptState,
        ) -> tuple[Any, optax.OptState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Core train step that can be JIT-compiled."""
        (total_loss, (interior_loss, boundary_loss)), grads = jax.value_and_grad(
            self._loss_with_aux,
            has_aux=True,
        )(params)

        updates, next_opt_state = self.optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, total_loss, interior_loss, boundary_loss

    def train_step(self, state: TrainState) -> tuple[TrainState, TrainStepMetrics]:
        """Run one optimization step and return updated state and metrics."""
        params, opt_state, total_loss, interior_loss, boundary_loss = self._train_step_impl(
            state.params,
            state.opt_state,
        )

        state = TrainState(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            rng_key=state.rng_key,
        )

        metrics = TrainStepMetrics(
            step=state.step,
            total_loss=float(total_loss),
            interior_loss=float(interior_loss),
            boundary_loss=float(boundary_loss),
        )
        return state, metrics

    def fit(
            self,
            sample_input: jnp.ndarray | None = None,
            state: TrainState | None = None,
            callback: Callable[[TrainStepMetrics], None] | None = None,
        ) -> tuple[TrainState, list[TrainStepMetrics]]:
        """Run training for the configured number of epochs.

        Parameters
        ----------
        sample_input : jnp.ndarray | None
            Required if ``state`` is not provided. Used for model initialization.
        state : TrainState | None
            Existing state for continuing training.
        callback : Callable[[TrainStepMetrics], None] | None
            Optional callback invoked each epoch.
        """
        if state is None and sample_input is None:
            raise ValueError("Must provide either an initial state or sample_input.")

        if state is None:
            assert sample_input is not None
            state = self.init_state(sample_input)

        history: list[TrainStepMetrics] = []
        for _ in range(self.train_cfg.epochs):
            state, metrics = self.train_step(state)
            history.append(metrics)

            if callback is not None:
                callback(metrics)

        return state, history
