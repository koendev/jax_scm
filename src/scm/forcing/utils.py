"""Utility helpers for evaluating and inspecting :class:`~scm.interfaces.Forcing` objects."""

from __future__ import annotations

import warnings
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from scm.interfaces import Forcing
from scm.physics.utils.dynamics import get_fc


def sample_forcing(f: Forcing, t_s: jnp.ndarray) -> Dict[str, jnp.ndarray | None]:
    """Evaluate all forcing functions over an array of simulation times.

    Parameters
    ----------
    f : Forcing
        Forcing object whose callable fields are evaluated.
    t_s : jnp.ndarray
        1D array of simulation times in seconds at which to sample the forcing.

    Returns
    -------
    dict[str, jnp.ndarray | None]
        Dictionary mapping forcing field names to sampled arrays.  Fields that
        are ``None`` on ``f`` are returned as ``None``.  Large-scale tendencies
        (``ls_tends``) are not sampled and a warning is emitted if present.
    """

    def _sample(fn, ndim_expected: int) -> jnp.ndarray | None:
        """Expand scalar or 1D output to expected dimensions.
        If, e.g., simulation forced with constant geostrophic wind, forcing may not broadcast to (time, z) shape.
        """
        # Handle missing forcing
        if fn is None:
            return None

        return jax.vmap(fn)(t_s)

        # Sample forcing
        res = fn(t_s)
        if res.ndim == 0 and ndim_expected == 1:
            res = jnp.full_like(t_s, res)
        elif res.ndim == 1 and ndim_expected == 2:
            res = jnp.tile(res, (t_s.size, 1))
        return res

    if f.ls_tends is not None:
        warnings.warn("Sampling of large-scale tendencies not implemented yet. Ignoring ls_tends!")

    return {
        # Expect 2D (time, z) for geostrophic wind
        "u_geo": _sample(f.u_geo, 2),
        "v_geo": _sample(f.v_geo, 2),
        # Expect 1D (time,) for surface forcing
        "w_th_s": _sample(f.w_th_s, 1),
        "th_s": _sample(f.th_s, 1),
        "w_qv_s": _sample(f.w_qv_s, 1),
        # Constants
        "f_c": f.f_c,
        "dth_dz_top": f.dth_dz_top,
    }


def uv_geo_from_z(
    lat_deg: xr.DataArray,
    lon_deg: xr.DataArray,
    z: xr.DataArray,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.Dataset:
    """Compute geostrophic wind components (ug, vg) from geopotential (z) using finite differences.

    Parameters
    ----------
    lat_deg : xr.DataArray
        Latitude in degrees.
    lon_deg : xr.DataArray
        Longitude in degrees.
    z : xr.DataArray
        Geopotential in m^2/s^2. Attention, not geopotential height!
    lat_dim : str
        Name of latitude dimension in the DataArray.
    lon_dim : str
        Name of longitude dimension in the DataArray.

    Returns
    -------
    xr.Dataset
        Dataset containing geostrophic wind components 'ug' and 'vg' in m/s.
    """
    # Constants
    R = 6371e3  # Earth's mean radius (m)

    # Convert lat/lon to radians
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    # 1. Calculate Coriolis parameter (f = 2 * Omega * sin(lat))
    f = get_fc(lat_deg=lat_deg)

    # 2. Calculate grid spacing in meters
    # dx = R * cos(lat) * dlon, dy = R * dlat
    dlon = lon_rad.diff(lon_dim)  # noqa: lon_rad remains xr.DataArray
    dlat = lat_rad.diff(lat_dim)  # noqa: lon_rad remains xr.DataArray

    dx = R * np.cos(lat_rad) * dlon.mean()
    dy = R * dlat.mean()

    # 3. Compute gradients of geopotential (z)
    # ERA5 'z' is geopotential (m^2/s^2). If you have geopotential height, multiply by 9.80665
    dz_dy = z.differentiate(lat_dim) / dy
    dz_dx = z.differentiate(lon_dim) / dx

    # 4. Compute components
    ug = -dz_dy / f
    vg = dz_dx / f

    return xr.Dataset({"ug": ug, "vg": vg})
