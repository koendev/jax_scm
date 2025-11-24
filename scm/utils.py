from __future__ import annotations

import dataclasses

import xarray as xr
from jax import numpy as jnp

from scm.grid import StaggeredGrid
from scm.interfaces import ProgVars, DiagVars


def make_dataset(
    state_hist: ProgVars,
    diag_hist: DiagVars,
    time: jnp.ndarray,
    grid: StaggeredGrid,
) -> xr.Dataset:
    """Convert history to xarray Dataset."""
    state_dict = dataclasses.asdict(state_hist)
    state_dict = {v: (("time", "z"), v_data) for v, v_data in state_dict.items()}
    state_ds = xr.Dataset(state_dict, coords={"time": time, "z": grid.z})

    diag_dict = dataclasses.asdict(diag_hist)
    diag_dict = {
        v: (
            ("time", "zh") if diag_dict[v].ndim == 2 else ("time",),
            v_data,
        )
        for v, v_data in diag_dict.items()
    }
    diag_ds = xr.Dataset(diag_dict, coords={"time": time, "zh": grid.zh})

    return xr.merge([state_ds, diag_ds])
