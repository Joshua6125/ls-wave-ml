'''
Main training loop.
'''
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

import inspect
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ..integration import NDCubeIntegration
from .base import TrainingMethod
from .state import TrainConfig, TrainState


@dataclass(frozen=True)
class TrainStepMetrics:
    """
    Structured diagnostics recorded during training.

    Contains scalar summaries of loss decomposition and timing
    information for a single optimisation step.
    """

    step: int
    total_loss: float
    interior_loss: float
    boundary_loss: float
    training_time: float


class Trainer:
    """
    Generic training engine for PDE-based learning systems.

    This class orchestrates the full optimisation pipeline:

    1. Parameter initialisation (via TrainingMethod)
    2. Construction of loss functionals
    3. Numerical integration over domain and boundary
    4. Differentiation and optimiser update
    5. Optional convergence monitoring and logging

    The implementation is independent of:
    - PDE formulation (PINN, FOSLS, etc.)
    - Neural architecture
    - Integration scheme
    """

    def __init__(
        self,
        method: TrainingMethod,
        integrator: NDCubeIntegration,
        optimiser: optax.GradientTransformation,
        train_cfg: TrainConfig,
    ):
        self.method = method
        self.integrator = integrator
        self.optimiser = optimiser
        self.train_cfg = train_cfg

        self.train_cfg.validate()

        self._train_step_fn = (
            jax.jit(self._train_step_impl)
            if self.train_cfg.use_jit
            else self._train_step_impl
        )

    def _init_state(self, sample_input: jnp.ndarray) -> TrainState:
        """
        Initialize model parameters, optimiser state, and RNG stream.
        """
        root_key = jr.PRNGKey(self.train_cfg.seed)
        root_key, init_key, integration_key = jr.split(root_key, 3)

        params = self.method.init_params(init_key, sample_input)
        opt_state = self.optimiser.init(params)

        return TrainState(
            step=0,
            params=params,
            opt_state=opt_state,
            integration_key=integration_key,
        )

    def _loss_with_aux(
        self,
        params: Any,
        integration_key: jax.Array,
    ):
        """
        Evaluate loss functional and return auxiliary decomposition.

        Returns:
            total loss
            (interior contribution, boundary contribution, next RNG key)
        """
        interior_fn, boundary_fn = self.method.loss_functions(params)

        interior, boundary = self.integrator.integrate(
            interior_fn,
            boundary_fn,
            integration_key,
        )

        total = self.method.aggregate_loss(interior, boundary)
        next_key, _ = jr.split(integration_key)

        return total, (interior, boundary, next_key)

    def _train_step_impl(
        self,
        params: Any,
        opt_state: optax.OptState,
        integration_key: jax.Array,
    ):
        """
        Single optimisation step (JIT-compatible core kernel).
        """
        fun = lambda p: self._loss_with_aux(p, integration_key)
        (total_loss, (interior_loss, boundary_loss, next_key)), grads = (
            jax.value_and_grad(fun, has_aux=True)(params)
        )

        updates, next_opt_state = self.optimiser.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)

        return (
            next_params,
            next_opt_state,
            total_loss,
            interior_loss,
            boundary_loss,
            next_key,
        )

    @staticmethod
    def _tree_sum(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(float(jnp.sum(leaf)) for leaf in leaves) if leaves else 0.0

    @staticmethod
    def _invoke_callback(callback, metrics, previous_state):
        """
        Flexible callback dispatcher supporting multiple signatures.
        """
        signature = inspect.signature(callback)

        positional_params = [
            p
            for p in signature.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in signature.parameters.values()
        ):
            callback(metrics, previous_state)
        elif len(positional_params) >= 2:
            callback(metrics, previous_state)
        elif len(positional_params) == 1:
            callback(metrics)
        else:
            callback()

    def _has_converged(self, loss_window: deque[float]) -> bool:
        """
        Determine convergence via relative stability of recent loss values.
        """
        if not loss_window:
            return False

        losses = tuple(loss_window)

        if not all(math.isfinite(l) for l in losses):
            return False

        mean_loss = sum(losses) / len(losses)
        mean_abs = sum(abs(l) for l in losses) / len(losses)

        tolerance = self.train_cfg.convergence_rel_tol * mean_abs
        max_dev = max(abs(l - mean_loss) for l in losses)

        return max_dev <= tolerance

    def fit(
        self,
        sample_input: jnp.ndarray | None = None,
        state: TrainState | None = None,
        callback: Callable[..., None] | None = None,
    ):
        """
        Execute full training loop.

        Supports either fresh initialization or continuation from existing state.
        """
        if state is None and sample_input is None:
            raise ValueError("Either state or sample_input must be provided.")

        if state is None:
            state = self._init_state(sample_input) # type: ignore

        loss_window = deque(maxlen=self.train_cfg.convergence_window_size)
        start_time = time.time()
        history = []

        for epoch in range(1, self.train_cfg.epochs + 1):
            prev_state = state

            (
                params,
                opt_state,
                total_loss,
                interior_loss,
                boundary_loss,
                integration_key,
            ) = self._train_step_fn(
                state.params,
                state.opt_state,
                state.integration_key,
            )

            state = TrainState(
                step=state.step + 1,
                params=params,
                opt_state=opt_state,
                integration_key=integration_key,
                total_training_time=time.time() - start_time,
            )

            if self.train_cfg.log_every > 0 and epoch % self.train_cfg.log_every == 0:
                metrics = TrainStepMetrics(
                    step=epoch,
                    total_loss=float(total_loss),
                    interior_loss=self._tree_sum(interior_loss),
                    boundary_loss=self._tree_sum(boundary_loss),
                    training_time=state.total_training_time,
                )

                print(f"Epoch {epoch}/{self.train_cfg.epochs}, Time {state.total_training_time:.2f}/{self.train_cfg.max_training_time}")

                history.append(metrics)

                if callback:
                    self._invoke_callback(callback, metrics, prev_state)

            if self.train_cfg.convergence_check:
                loss_window.append(float(total_loss))

                if len(loss_window) == loss_window.maxlen and self._has_converged(
                    loss_window
                ):
                    return state, history

            if state.total_training_time > self.train_cfg.max_training_time:
                return state, history

        return state, history
