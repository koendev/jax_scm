"""Explicit time-stepping factories: first-order Euler and Adams-Bashforth 2."""

from __future__ import annotations

from typing import Callable

import jax

from scm.interfaces import ModelFn
from scm.time_stepping.utils import StepCarry, clip_state


def get_euler_step_fn(model: ModelFn) -> Callable:
    """Return a first-order Euler step function used for AB2 warm-up.

    Parameters
    ----------
    model : ModelFn
        Right-hand-side function ``(t_s, state, params) -> (tends, diag, mo)``.

    Returns
    -------
    Callable
        Function with signature ``(t_s, dt_s, y0, params) -> StepCarry`` that
        advances ``y0`` by one Euler step and populates both ``prev_*`` history
        fields with the current-step values.
    """

    def _euler(t_s, dt_s, y0, params) -> StepCarry:
        """Advance state by one Euler step."""
        dydt0, diag0, mo0 = model(t_s, y0, params)
        y1 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, y0, dydt0)
        y1 = clip_state(y1)
        return StepCarry(y=y1, prev_tends=dydt0, prev_mo=mo0, diag=diag0, mo=mo0)

    return _euler


def get_ab2_step_fn(model: ModelFn) -> Callable:
    """Return a second-order Adams-Bashforth step function.

    Parameters
    ----------
    model : ModelFn
        Right-hand-side function ``(t_s, state, params) -> (tends, diag, mo)``.

    Returns
    -------
    Callable
        Function with signature ``(carry, t_s, dt_s, params) -> StepCarry``
        that advances the state using the AB2 extrapolation
        ``dy/dt ≈ 1.5 * f(t) - 0.5 * f(t-1)``.
        When ``cfg.adaptive_timestep`` is set, ``dt_s`` is computed externally
        from the CFL condition ``dt = cfl_max * dz² / K_max`` and passed in at
        each sub-step.
    """

    def _ab2(carry: StepCarry, t_s, dt_s, params) -> StepCarry:
        """Advance carry by one AB2 step."""
        dydt1, diag1, mo1 = model(t_s, carry.y, params)
        dydt_ab = jax.tree_util.tree_map(lambda d1, d0: (3 / 2) * d1 - (1 / 2) * d0, dydt1, carry.prev_tends)
        y2 = jax.tree_util.tree_map(lambda y, dy: y + dt_s * dy, carry.y, dydt_ab)
        y2 = clip_state(y2)
        return StepCarry(y=y2, prev_tends=dydt1, prev_mo=mo1, diag=diag1, mo=mo1)

    return _ab2
