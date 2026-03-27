import flax.linen as nn
import jax.numpy as jnp
from typing import Mapping


class NeuralNet(nn.Module):
    """Simple fully-connected network used across methods.

    Parameters
    ----------
    hidden_dim : int
        Width of each hidden layer.
    num_layers : int
        Number of hidden layers.
    output_dim : int
        Output dimension of the final layer.
    output_heads : Mapping[str, int] | None
        Optional named output heads. If provided, model returns a dict mapping
        head names to tensors.
    """

    hidden_dim: int
    num_layers: int
    output_dim: int = 1
    output_heads: Mapping[str, int] | None = None

    @nn.compact
    def __call__(self, x):
        assert self.hidden_dim > 0, "hidden_dim must be strictly positive"
        assert self.num_layers > 0, "num_layers must be strictly positive"
        assert self.output_dim > 0, "output_dim must be strictly positive"
        if self.output_heads is not None:
            assert len(self.output_heads) > 0, "output_heads must be non-empty when provided"
            for name, dim in self.output_heads.items():
                assert name, "output head names must be non-empty"
                assert dim > 0, "each output head dimension must be strictly positive"

        h = x
        for _ in range(self.num_layers):
            h = jnp.tanh(nn.Dense(self.hidden_dim)(h))

        if self.output_heads is not None:
            return {
                name: nn.Dense(dim, name=f"{name}_head")(h)
                for name, dim in self.output_heads.items()
            }

        return nn.Dense(self.output_dim)(h)
