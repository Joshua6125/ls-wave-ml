from dataclasses import dataclass, field
from typing import TypeAlias, Mapping, Protocol, Any, cast
from typing_extensions import runtime_checkable

from .mlp import MLP
from .kan import KAN

import jax
import jax.numpy as jnp
import flax.linen as nn


@runtime_checkable
class BuiltModelProtocol(Protocol):
    def init(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any: ...
    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]: ...


class BuiltModelAdapter:
    def __init__(self, module: nn.Module):
        self._module = module

    def init(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        return self._module.init(rng_key, sample_input)

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        out = self._module.apply(params, x)
        if not isinstance(out, dict):
            raise TypeError(
                "Model.apply must return a dict[str, ndarray] for training."
            )
        return cast(dict[str, jnp.ndarray], out)


@dataclass(frozen=True)
class BaseModelConfig:
    """
    Base configuration of neural models.

    Parameters
    ----------
    kind: str
        The name of the model instance.
    hidden_dim : int
        Width of each hidden layer.
    num_layers : int
        Number of hidden layers.
    output_heads : Mapping[str, int]
        Named output heads.
    constrained_heads: list[str]
        The heads to ensure are smooth and zero on the spatial boundary.
    """

    kind: str
    hidden_dim: int = 64
    num_layers: int = 4
    constrained_heads: list[str] = []
    output_heads: Mapping[str, int] = field(default_factory=lambda: {"output": 1})

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be strictly positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be strictly positive")
        if len(self.output_heads) == 0:
            raise ValueError("output_heads must be non-empty")
        for name, dim in self.output_heads.items():
            if not name:
                raise ValueError("output head names must be non-empty")
            if dim <= 0:
                raise ValueError("each output head dimension must be strictly positive")


@dataclass(frozen=True)
class MLPConfig(BaseModelConfig):
    """
    Configuration for the built-in fully connected model.

    Parameters
    ----------
    kind: str
        The name of the model instance.
    hidden_dim : int
        Width of each hidden layer.
    num_layers : int
        Number of hidden layers.
    output_heads : Mapping[str, int]
        Named output heads.
    constrained_heads: list[str]
        The heads to ensure are smooth and zero on the spatial boundary.
    """

    kind: str = "mlp"

    def validate(self) -> None:
        super().validate()


@dataclass(frozen=True)
class KANConfig(BaseModelConfig):
    """
    Configuration for KAN model.

    Parameters
    ----------
    kind: str
        The name of the model instance.
    hidden_dim : int
        Width of each hidden layer.
    num_layers : int
        Number of hidden layers.
    output_heads : Mapping[str, int]
        Named output heads.
    input_dim : int
        Dimension of the input layer.
    constrained_heads: list[str]
        The heads to ensure are smooth and zero on the spatial boundary.
    grid_size : int
        Grid size for spline-based KAN networks
    degree : int
        Degree of Chebyshev polynomials in Chebyshev-KAN
    model_type : str
        Specifies the type of model: "efficient" | "cheby" | "chebyshev" | "original" | "base" | "spline".
    seed : int
        The random seed used for KAN initialisation.
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
        if self.model_type in ["efficient", "cheby", "chebyshev"]:
            if self.degree < 0:
                raise ValueError("Degree of Chebyshev polynomials must be non-negative")
        if self.model_type in ["original", "base", "spline"]:
            if self.grid_size <= 0:
                raise ValueError("Grid size of spline-based KAN must be positive")


AnyModelConfig: TypeAlias = MLPConfig | KANConfig


def build_model(cfg: AnyModelConfig) -> BuiltModelAdapter:
    """
    Build model from declarative model config.

    Parameters
    ----------
    cfg : AnyModelConfig
        Either an MLPConfig or KANConfig instance.
    """
    if isinstance(cfg, MLPConfig):
        cfg.validate()
        return BuiltModelAdapter(
            MLP(
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                output_heads=cfg.output_heads,
                constrained_heads=cfg.constrained_heads,
            )
        )

    if isinstance(cfg, KANConfig):
        cfg.validate()
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
