from typing import List

from flax import nnx


class FlexMLP(nnx.Module):
    def __init__(self, template: List[int], rngs: nnx.Rngs):
        init = nnx.initializers.normal()
        self.template = template
        self.layers = [
            nnx.Linear(n_in, n_out, rngs=rngs, kernel_init=init, bias_init=init)
            for n_in, n_out in zip(template[:-1], template[1:])
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))
        x = self.layers[-1](x)
        return x


def test_flex_mlp():
    """Basic test of FlexMLP. Shouldn't raise errors."""
    import jax.numpy as jnp

    mlp = FlexMLP(template=[2, 8, 4], rngs=nnx.Rngs(0))
    x = jnp.ones((100, 2))
    y = mlp(x)

    assert y.shape == (100, 4)
