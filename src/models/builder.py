from dataclasses import dataclass, field
from typing import Literal, TypeAlias, Mapping

from .neuralnet import NeuralNet


@dataclass(frozen=True)
class NeuralNetModelConfig:
    """Configuration for the built-in fully connected model."""

    kind: Literal["neuralnet"] = "neuralnet"
    hidden_dim: int = 64
    num_layers: int = 4
    output_heads: Mapping[str, int] = field(default_factory=lambda:{"output": 1})

    def validate(self) -> None:
        assert self.hidden_dim > 0, "hidden_dim must be strictly positive"
        assert self.num_layers > 0, "num_layers must be strictly positive"
        assert self.output_heads,   "output_heads must not be non-empty"
        assert len(self.output_heads) > 0, "output_heads must be non-empty when provided"
        for name, dim in self.output_heads.items():
            assert name, "output head names must be non-empty"
            assert dim > 0, "each output head dimension must be strictly positive"


@dataclass(frozen=True)
class PINNModelConfig:
    """Model specification for PINN training."""

    kind: Literal["pinn"] = "pinn"
    u_model: NeuralNetModelConfig = field(default_factory=NeuralNetModelConfig)


@dataclass(frozen=True)
class LSModelConfig:
    """Model specification for LS training with one shared multi-head model."""

    kind: Literal["ls"] = "ls"
    ls_model: NeuralNetModelConfig = field(
        default_factory=lambda: NeuralNetModelConfig(
            output_heads={"v": 1, "sigma": 1}
        )
    )

    def validate(self) -> None:
        self.ls_model.validate()
        if "v" not in self.ls_model.output_heads or "sigma" not in self.ls_model.output_heads:
            raise ValueError("LS model config output_heads must include 'v' and 'sigma'.")
        if self.ls_model.output_heads["v"] != 1:
            raise ValueError("LS model config requires 'v' head to have output dimension 1.")


AnyModelConfig: TypeAlias = PINNModelConfig | LSModelConfig
AnyBuiltModel: TypeAlias = NeuralNet


def _build_neuralnet(cfg: NeuralNetModelConfig) -> NeuralNet:
    cfg.validate()
    return NeuralNet(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_heads=cfg.output_heads,
    )


def build_model(model_cfg: AnyModelConfig) -> AnyBuiltModel:
    """Build model from declarative model config."""
    if isinstance(model_cfg, PINNModelConfig):
        return _build_neuralnet(model_cfg.u_model)

    if isinstance(model_cfg, LSModelConfig):
        model_cfg.validate()
        return _build_neuralnet(model_cfg.ls_model)

    raise ValueError("Unknown model config type.")