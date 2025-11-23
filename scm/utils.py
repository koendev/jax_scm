from __future__ import annotations

import dataclasses
from typing import List

import xarray as xr
from jax import numpy as jnp

from scm.interfaces import ProgVars, DiagVars
from scm.grid import StaggeredGrid


def make_dataset(
    state_hist: List[ProgVars],
    diag_hist: List[DiagVars],
    time: jnp.ndarray,
    grid: StaggeredGrid,
) -> xr.Dataset:
    """Convert history to xarray Dataset."""
    state_dict = dataclasses.asdict(state_hist[0])
    diag_dict = dataclasses.asdict(diag_hist[0])

    ds = xr.Dataset(
        {k: (("time", "z"), jnp.array([getattr(s, k) for s in state_hist])) for k in state_dict},
        coords={
            "time": time,
            "z": grid.z,
        },
    )

    ds_diag = xr.Dataset(
        {k: (("time", "zh"), jnp.array([getattr(d, k) for d in diag_hist])) for k in diag_dict},
        coords={
            "time": jnp.arange(len(diag_hist)),
            "zh": grid.zh,
        },
    )

    return xr.merge([ds, ds_diag])
