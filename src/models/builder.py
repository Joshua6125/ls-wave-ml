from dataclasses import dataclass, field
from typing import TypeAlias, Mapping, Protocol, Any, cast
from typing_extensions import runtime_checkable

import jax
import jax.numpy as jnp
import flax.linen as nn

from .mlp import MLP
from .kan import KAN


@runtime_checkable
class BuiltModelProtocol(Protocol):
    """
    Minimal interface expected from built neural models.
    """

    def init(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any: ...

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]: ...


class BuiltModelAdapter:
    """
    Adapter that enforces a uniform interface over Flax modules.

    Ensures that model outputs are dictionaries of named heads,
    which is required by the downstream training pipeline.
    """

    def __init__(self, module: nn.Module):
        self._module = module

    def init(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        return self._module.init(rng_key, sample_input)

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        out = self._module.apply(params, x)

        if not isinstance(out, dict):
            raise TypeError("Model.apply must return a dict[str, ndarray].")

        return cast(dict[str, jnp.ndarray], out)


@dataclass(frozen=True)
class BaseModelConfig:
    """
    Declarative configuration for neural network models.

    Specifies shared architecture parameters used by all model types.
    """

    kind: str

    hidden_dim: int = 64
    num_layers: int = 4

    output_heads: Mapping[str, int] = field(default_factory=lambda: {"output": 1})

    constrained_heads: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate shared architectural constraints."""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be strictly positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be strictly positive")

        if not self.output_heads:
            raise ValueError("output_heads must be non-empty")

        for name, dim in self.output_heads.items():
            if not name:
                raise ValueError("output head names must be non-empty")
            if dim <= 0:
                raise ValueError("each output head dimension must be strictly positive")


@dataclass(frozen=True)
class MLPConfig(BaseModelConfig):
    """
    Configuration for a fully-connected MLP model.
    """

    kind: str = "mlp"

    def validate(self) -> None:
        super().validate()


@dataclass(frozen=True)
class KANConfig(BaseModelConfig):
    """
    Configuration for a Kolmogorov-Arnold Network (KAN) model.
    """

    kind: str = "kan"

    input_dim: int = 1
    grid_size: int = 5
    degree: int = 3
    model_type: str = "efficient"
    seed: int = 42

    def validate(self) -> None:
        super().validate()

        if self.input_dim <= 0:
            raise ValueError("input_dim must be strictly positive")

        if self.model_type in {"efficient", "cheby", "chebyshev"}:
            if self.degree < 0:
                raise ValueError("degree of Chebyshev polynomials must be non-negative")

        if self.model_type in {"original", "base", "spline"}:
            if self.grid_size <= 0:
                raise ValueError("grid_size must be strictly positive")


AnyModelConfig: TypeAlias = MLPConfig | KANConfig


def build_model(cfg: AnyModelConfig) -> BuiltModelAdapter:
    """
    Construct a model from a declarative configuration.

    The returned object is a thin adapter over a Flax module,
    exposing a uniform init/apply interface expected by training code.
    """

    cfg.validate()

    if isinstance(cfg, MLPConfig):
        return BuiltModelAdapter(
            MLP(
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                output_heads=cfg.output_heads,
                constrained_heads=cfg.constrained_heads,
            )
        )

    if isinstance(cfg, KANConfig):
        return BuiltModelAdapter(
            KAN(
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                output_heads=cfg.output_heads,
                input_dim=cfg.input_dim,
                constrained_heads=cfg.constrained_heads,
                grid_size=cfg.grid_size,
                degree=cfg.degree,
                model_type=cfg.model_type,
                seed=cfg.seed,
            )
        )

    raise ValueError("Unknown model config type.")
