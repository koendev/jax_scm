"""Ensemble helpers for running parallel sets of simulations.

The ensemble flow is:

1. Build N ``Simulation`` objects that share pytree structure (same forcing
   functions, same array shapes — only leaf values differ).
2. ``stack(sims)`` combines them into one ``Simulation`` whose array leaves
   carry a leading member axis of size N.
3. ``jax.vmap(run)(stacked_sim)`` (or :func:`pmap_run`) executes all members
   in parallel.
4. ``unstack(out)`` splits the batched output back into a list of per-member
   :class:`scm.interfaces.Output` objects for analysis or serialization.

For ensemble members to share pytree structure, *forcing* functions must be
written with :class:`jax.tree_util.Partial` rather than closures over
per-member arrays. See the :class:`scm.interfaces.Forcing` docstring.
"""

from __future__ import annotations

from typing import Callable, List, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

T = TypeVar("T")


def stack(members: List[T]) -> T:
    """Stack a list of pytrees into a single pytree with a leading member axis.

    All members must share the same pytree structure. A :class:`ValueError`
    is raised otherwise (typically caused by forcings using closures that
    capture differing arrays — use :class:`jax.tree_util.Partial` instead).
    """
    if len(members) == 0:
        raise ValueError("Cannot stack an empty list of members.")

    ref_treedef = jax.tree_util.tree_structure(members[0])
    for i, m in enumerate(members[1:], start=1):
        td = jax.tree_util.tree_structure(m)
        if td != ref_treedef:
            raise ValueError(
                f"Member {i} has a different pytree structure than member 0. "
                f"Ensure forcings use jax.tree_util.Partial(fn, *captured_arrays) "
                f"with the same `fn` reference across members.\n"
                f"  member 0: {ref_treedef}\n"
                f"  member {i}: {td}"
            )

    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *members)


def unstack(batched: T) -> List[T]:
    """Inverse of :func:`stack`: split a batched pytree along the leading axis."""
    leaves, treedef = jax.tree_util.tree_flatten(batched)
    n = leaves[0].shape[0]
    return [treedef.unflatten([leaf[i] for leaf in leaves]) for i in range(n)]


def vmap_run(run_fn: Callable[[T], T]) -> Callable[[T], T]:
    """Vectorize a single-member run function over a stacked pytree.

    Equivalent to ``jax.vmap(run_fn)``; provided for symmetry with
    :func:`pmap_run` and to give the ensemble entry point a consistent name.
    """
    return jax.vmap(run_fn)


def pmap_run(run_fn: Callable[[T], T]) -> Callable[[T], T]:
    """Parallelize a single-member run function across devices.

    On a single-CPU host this is effectively serial; set
    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` to expose multiple
    virtual devices, or prefer :func:`vmap_run` for portable parallelism.
    """
    return jax.pmap(run_fn)
