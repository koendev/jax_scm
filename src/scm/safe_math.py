"""Safe math functions with custom VJPs to prevent NaN gradients."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from scm.consts import smooth_eps

GRAD_MIN, GRAD_MAX = -1e6, 1e6


@jax.custom_vjp
def safe_root(x: jnp.ndarray | float, p: float, eps: float = smooth_eps) -> jnp.ndarray:
    """Compute ``x**p`` with argument and gradient clipping to prevent NaNs.

    The argument is floored at ``eps`` before the power is taken, so the
    function is safe for ``x ≤ 0``.  Gradients are additionally clipped to
    ``[GRAD_MIN, GRAD_MAX]`` via a custom VJP (see ``safe_root_fwd`` /
    ``safe_root_bwd``).

    Parameters
    ----------
    x : array-like or float
        Input value(s); may be zero or negative.
    p : float
        Exponent (non-differentiable; must be a Python scalar).
    eps : float, optional
        Minimum value used to floor ``x`` before exponentiation.

    Returns
    -------
    jnp.ndarray
        ``max(x, eps) ** p``.
    """
    # Default forward: gets called outside gradient computation
    x_safe = jnp.maximum(x, eps)
    return jnp.pow(x_safe, p)


def safe_root_fwd(x: jnp.ndarray | float, p: float, eps: float):
    """Custom VJP forward pass: floors ``x`` and saves residuals for the backward pass."""
    x_safe = jnp.maximum(x, eps)
    return safe_root(x_safe, p, eps), (x_safe, p)  # pass x_safe and p as residuals for backward


def safe_root_bwd(res, g):
    """Custom VJP backward pass: computes the power-rule gradient and clips it to ``[GRAD_MIN, GRAD_MAX]``."""
    x_safe, p = res
    raw = g * p * jnp.power(x_safe, p - 1)
    return (jnp.clip(raw, GRAD_MIN, GRAD_MAX), None, None)  # p and eps are non diffable, so return None.


safe_root.defvjp(safe_root_fwd, safe_root_bwd)
