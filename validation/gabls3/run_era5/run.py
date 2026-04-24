from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from scm.config import load_namelist
from scm.forcing.era5 import get_era5_sim
from scm.forcing.interp import interp_dtindex
from scm.grid import StaggeredGrid
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate


def run(use_lf: bool):
    # Setup simulation for GABLS3 case from ERA5
    sim = get_era5_sim(
        name="GABLS3_ERA5",
        lat_deg=52.0,
        lon_deg=5.0,
        time_slice=("2006-07-01T11:00", "2006-07-02T12:00"),
        grid=StaggeredGrid(Nz=100, H=3000.0),
        source="cds",
        cache_dir="era5",
        th_ref=273.15 + 20.0,
    )
    if not use_lf:
        sim.forcing = dataclasses.replace(sim.forcing, ls_tends=None)  # disable large-scale tendencies

    # Load config and run simulation
    cfg = load_namelist("namelist_cn.yaml")
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Save output
    out_file = "out_lf.nc" if use_lf else "out_no_lf.nc"
    out_file = pathlib.Path(out_file)
    ds = out_to_ds(
        out=out,
        sim=sim,
        time=interp_dtindex(t_s=np.array(out.t_s), idx=sim.t_index).round("1min"),
    )
    ds.to_netcdf(out_file)
    print("Written to disk.")
    return ds


if __name__ == "__main__":
    run(use_lf=True)
    run(use_lf=False)
