import jax
import jax.numpy as jnp

from scm.safe_math import safe_root


def test_sqrt():
    def _test():
        # Default sqrt has infinit gradient
        y, dy = jax.value_and_grad(jnp.sqrt)(0.0)
        assert jnp.isinf(dy)
        assert y == 0.0

        # Safe root clips in regular forward pass AND gradient computation
        eps = 1e-6
        y = jax.jit(safe_root)(0.0, 1 / 2, eps=eps)
        assert y == eps ** (1 / 2)

        # Assert finite gradient
        y, dy = jax.value_and_grad(jax.jit(safe_root))(0.0, 1 / 2, eps=eps)
        assert jnp.isfinite(dy)
        assert y == eps ** (1 / 2)  # zero gets clipped to eps
        assert dy == 1 / (2 * eps ** (1 / 2))  # manual evaluation of gradient at eps value

    # test without jit
    with jax.disable_jit():
        _test()

    # test with jit
    _test()
