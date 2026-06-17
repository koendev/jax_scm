"""Thermodynamic utility functions and conversions."""

from __future__ import annotations

from typing import Literal, TypeVar

import jax
import pandas as pd
import xarray as xr
from jax import numpy as jnp

from scm import consts

# Type variable for functions that perform pure computations without library specific functions
T = TypeVar("T", jax.Array, xr.DataArray, pd.Series)


def tv_to_t(*, tv: T, qv: T) -> T:
    """Convert virtual temperature to dry temperature.

    Parameters
    ----------
    tv : jnp.ndarray
        Virtual (or virtual potential) temperature (K).
    qv : jnp.ndarray
        Specific humidity (kg/kg).

    Returns
    -------
    jnp.ndarray
        Dry (or dry potential) temperature (K).
    """
    return tv / (1 + 0.61 * qv)


def t_to_tv(*, t: T, qv: T) -> T:
    """Convert dry temperature to virtual temperature.

    Parameters
    ----------
    t : jnp.ndarray
        Dry (or dry potential) temperature (K).
    qv : jnp.ndarray
        Specific humidity (kg/kg).

    Returns
    -------
    jnp.ndarray
        Virtual (or virtual potential) temperature (K).
    """
    return t * (1 + 0.61 * qv)


def tk_to_th(*, tk: T, p_hPa: T) -> T:
    """Convert temperature (K) to potential temperature (K).

    Parameters
    ----------
    tk : xr.DataArray
        Temperature in Kelvin.
    p_hPa : xr.DataArray
        Pressure in hPa.

    Returns
    -------
    xr.DataArray
        Potential temperature in Kelvin.
    """
    p0_hPa = 1000.0  # Reference pressure in hPa
    exp = (consts.gamma - 1) / consts.gamma
    th_k = tk * (p0_hPa / p_hPa) ** exp
    return th_k


def w_th_to_w_thv(*, th: T, w_th: T, w_qv: T) -> T:
    """Convert sensible heat flux (w'theta') to buoyancy flux (w'theta_v').

    Parameters
    ----------
    th : jnp.ndarray
        Dry potential temperature (K).
    w_th : jnp.ndarray
        Sensible heat flux, w'theta' (K m/s).
    w_qv : jnp.ndarray
        Moisture flux, w'qv' ((kg/kg) m/s).

    Returns
    -------
    jnp.ndarray
        Buoyancy flux w'theta_v' (K m/s).
    """
    return w_th + 0.61 * th * w_qv


def w_thv_to_w_th(*, th: T, w_thv: T, w_qv: T) -> T:
    """Convert buoyancy flux (w'theta_v') to sensible heat flux (w'theta').

    Parameters
    ----------
    th : jnp.ndarray
        Dry potential temperature (K).
    w_thv : jnp.ndarray
        Buoyancy flux, w'theta_v' (K m/s).
    w_qv : jnp.ndarray
        Moisture flux, w'qv' ((kg/kg) m/s).

    Returns
    -------
    jnp.ndarray
        Sensible heat flux w'theta' (K m/s).
    """
    return w_thv - 0.61 * th * w_qv


def get_p_rho_fn(mode: Literal["th", "tk"]):
    """Build a function that integrates pressure and density profiles upward from the surface.

    Uses the hypsometric equation with the ideal gas law.

    Parameters
    ----------
    mode : {"th", "tk"}
        Temperature input convention: ``"th"`` for potential temperature,
        ``"tk"`` for absolute temperature.

    Returns
    -------
    Callable
        Function with signature ``(t, qv, z, p_s) -> (p_profile, rho_profile)``.
        See inner ``get_p_rho`` for argument details.
    """

    def get_p_rho(t: jnp.ndarray, qv: jnp.ndarray, z: jnp.ndarray, p_s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes density and pressure profiles using the hypsometric equation.

        Parameters
        ----------
        t : jnp.ndarray
            Air temperature profile (K) or potential temperature profile (K), depending on mode.
        qv : jnp.ndarray
            Specific humidity profile (kg/kg).
        z : jnp.ndarray
            Geopotential height profile (m).
        p_s : float
            Surface pressure (Pa).

        Returns
        -------
        p_profile : jnp.ndarray
            Pressure profile (Pa).
        rho_profile : jnp.ndarray
            Density profile (kg/m^3).
        """
        p0 = 100000.0

        def scan_fn(p_i, carry_vars):
            """Compute pressure and density from previous level using hypsometric equation."""
            if mode == "th":
                # Estimate air temp from pot temp
                th_j, qv_j, dz = carry_vars
                tk_j = th_j * (p_i / p0) ** (consts.Rd / consts.cp)
            elif mode == "tk":
                # Directly use air temp
                tk_j, qv_j, dz = carry_vars
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'th' or 'tk'.")

            # Get virtual air temp
            tkv_j = t_to_tv(t=tk_j, qv=qv_j)

            # Use hypsometric equation to compute pressure at this level
            # p_j = p_prev * exp(-g * dz / (Rd * Tv))
            p_j = p_i * jnp.exp(-consts.g * dz / (consts.Rd * tkv_j))

            # Density from ideal gas law
            rho_j = p_j / (consts.Rd * tkv_j)

            return p_j, (p_j, rho_j)

        dz = jnp.diff(z)
        dz = jnp.concat([jnp.array([z[0]]), dz])

        # Integrate from surface to top of the column
        _, (p_profile, rho_profile) = jax.lax.scan(scan_fn, p_s, (t, qv, dz))

        return p_profile, rho_profile

    return get_p_rho


# Pre-built instances of get_p_rho_fn for the two supported temperature conventions.
# p_rho_from_th expects potential temperature; p_rho_from_tk expects absolute temperature.
p_rho_from_th = get_p_rho_fn(mode="th")
p_rho_from_tk = get_p_rho_fn(mode="tk")
