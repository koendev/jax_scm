"""Utility functions for dynamics"""

from __future__ import annotations

from jax import numpy as jnp

from scm import consts


def get_fc(*, lat_deg: float) -> float:
    """Compute the Coriolis parameter at a given latitude.

    Parameters
    ----------
    lat_deg : float
        Latitude in degrees.

    Returns
    -------
    float
        Coriolis parameter f = 2 * Omega * sin(lat) (rad/s).
    """
    omega = 7.2921e-5  # rad/s, Earth's angular velocity
    return float(2 * omega * jnp.sin(jnp.deg2rad(lat_deg)))


def w_eff(*, omega, rho):
    """Convert pressure vertical velocity (omega) to geometric vertical velocity (w).

    Parameters
    ----------
    omega : jnp.ndarray
        Pressure vertical velocity (Pa/s).
    rho : jnp.ndarray
        Air density (kg/m^3).

    Returns
    -------
    jnp.ndarray
        Vertical velocity w (m/s); positive upward.
    """
    return -omega / (rho * consts.g)
