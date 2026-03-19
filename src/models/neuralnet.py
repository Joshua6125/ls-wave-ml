import flax.linen as nn
import jax.numpy as jnp


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
    """

    hidden_dim: int
    num_layers: int
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        assert self.hidden_dim > 0, "hidden_dim must be strictly positive"
        assert self.num_layers > 0, "num_layers must be strictly positive"
        assert self.output_dim > 0, "output_dim must be strictly positive"

        h = x
        for _ in range(self.num_layers):
            h = jnp.tanh(nn.Dense(self.hidden_dim)(h))
        return nn.Dense(self.output_dim)(h)
