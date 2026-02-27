from __future__ import annotations

from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
from scm.forcing.era5 import get_era5_sim
import xarray as xr
from scm.grid import StaggeredGrid
from scm.forcing.interp import interp_dtindex

import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from scm.config import load_namelist
from scm.examples.gabls1 import get_gabls1
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.reporter import BaseReport
from scm.time_stepping import simulate

plot_kwargs = {
    "color": "C1",
    "linewidth": 2,
    "marker": "o",
    "markevery": 10,
    "label": "jax-scm",
}


def get_ref_ax(
    img_path: str,
    x_lims: Tuple[float, float],
    y_lims: Tuple[float, float],
    trim: Tuple[int, int, int, int] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    # Load image and trim if needed
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)

    if trim is not None:
        w, h = img.size
        left, bottom, right, top = trim
        img = img.crop(
            (
                0 + left,
                0 + top,
                w - right,
                h - bottom,
            )
        )  # (left, top) to  (right, bottom)

    fig, ax = plt.subplots()

    # format: [xmin, xmax, ymin, ymax]
    ax.imshow(img, extent=(*x_lims, *y_lims), aspect="auto", cmap="Greys_r")

    return fig, ax


def run():
    # Setup simulation for GABLS3 case from ERA5
    sim = get_era5_sim(
        name="ERA5 Test Simulation",
        lat_deg=52.0,
        lon_deg=5.0,
        time_slice=("2006-07-01T12:00", "2006-07-02T12:00"),
        grid=StaggeredGrid(Nz=150, H=3000.0),
        source="cds",
        cache_dir="./era5",
    )

    # Load config and run simulation
    cfg = load_namelist("namelist_cn.yaml")
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Save output
    out_file = pathlib.Path(f"out_{sim.grid.Nz}.nc")
    ds = out_to_ds(
        out=out,
        sim=sim,
        time=interp_dtindex(t_s=np.array(out.t_s), idx=sim.t_index),  # todo: this is weird
    )
    ds.to_netcdf(out_file)
    print("Written to disk.")
    return ds


if __name__ == "__main__":
    # ds = run()
    ds = xr.open_dataset("out_150.nc")

    ds["m"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    ds["d"] = np.rad2deg(np.arctan2(-ds["u"], -ds["v"]))
    z = ds["z"].values

    with BaseReport(title="GABLS3 Validation", path=f"val_gabls3.html") as r:
        r.add_text(
            "This report compares the jax-scm model against GABLS3 reference results from Bosveld et al. (2014). "
            "Instead of using the prescribed initial and boundary conditions from the paper, we use ERA5 data."
        )

        r.add_heading("Profiles after init (12:10 UTC)", level=2)
        ds_1210UTC = ds.isel(time=2)

        # th
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig1.png",
            (296, 308),
            (0, 3000),
            trim=(98, 636, 535, 17),
        )
        ax.plot(ds_1210UTC["th"], z, **plot_kwargs)
        ax.set_xlabel("Potential temperature, K")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Potential temperature profile at 12:10 UTC")

        # qv
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig1.png",
            (0, 0.01),
            (0, 3000),
            trim=(604, 636, 28, 17),
        )
        ax.plot(ds_1210UTC["qv"], z, **plot_kwargs)
        ax.set_xlabel("Specific humidity, kg/kg")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Specific humidity profile at 12:10 UTC")

        # m
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig1.png",
            (1, 6),
            (0, 3000),
            trim=(98, 132, 527, 511),
        )
        ax.plot(ds_1210UTC["m"], z, **plot_kwargs)
        ax.set_xlabel("Wind speed, K")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind profile at 12:10 UTC")

        # d
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig1.png",
            (80, 140),
            (0, 3000),
            trim=(604, 132, 20, 511),
        )
        ax.plot(ds_1210UTC["d"], z, **plot_kwargs)
        ax.set_xlabel("Wind direction, deg")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind direction at 12:10 UTC")

        r.add_heading("Profiles at 00:00 UTC", level=2)
        ds_000UTC = ds.isel(time=144).sel(z=slice(0, 500))

        # th
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig2.png",
            (286, 300),
            (0, 500),
            trim=(92, 616, 530, 13),  # left, bottom, right, top
        )
        ax.plot(ds_000UTC["th"], ds_000UTC.z, **plot_kwargs)
        ax.set_xlabel("Potential temperature, K")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Potential temperature profile at 00:00 UTC")

        # qv
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig2.png",
            (0.007, 0.012),
            (0, 500),
            trim=(602, 616, 24, 15),  # left, bottom, right, top
        )
        ax.plot(ds_000UTC["qv"], ds_000UTC.z, **plot_kwargs)
        ax.set_xlabel("Specific humidity, kg/kg")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Specific humidity profile at 00:00 UTC")

        # m
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig2.png",
            (0, 14),
            (0, 500),
            trim=(93, 106, 525, 519),  # left, bottom, right, top
        )
        ax.plot(ds_000UTC["m"], ds_000UTC.z, **plot_kwargs)
        ax.set_xlabel("Wind speed, K")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind profile at 00:00 UTC")

        # d
        fig, ax = get_ref_ax(
            "ref_bosveld14/fig2.png",
            (70, 140),
            (0, 500),
            trim=(602, 106, 20, 522),  # left, bottom, right, top
        )
        ax.plot(ds_000UTC["d"], ds_000UTC.z, **plot_kwargs)
        ax.set_xlabel("Wind direction, deg")
        ax.set_ylabel("Height, z")
        ax.legend()
        r.add_mpl_fig(fig, caption="Wind direction at 00:00 UTC")
