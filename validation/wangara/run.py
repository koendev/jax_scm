import jax.numpy as jnp
import numpy as np
import pandas as pd

from scm import convert
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation, Forcing
from scm.mo import MOSettings
from scm.mynn.interfaces import ProgVarsMYNN
from scm.mynn.model import init_model
from scm.config import load_namelist
from scm.time_stepping import simulate
from scm.io.local import out_to_ds


def get_wangara_33(Nz: int = 50) -> Simulation:
    ## Grid
    grid = StaggeredGrid(H=2000, Nz=Nz)

    ## Initial conditions
    df = pd.read_csv("ref/day33_0900.csv")
    tk = df["tc"] + 273.15  # Convert to K
    p_hPa = df["p"]
    th = convert.tk_to_th(tk=tk, p_hPa=p_hPa)
    th = np.interp(grid.z, df["z"], th)  # Interpolate to model grid

    u = np.interp(grid.z, df["z"], df["u"])
    v = np.interp(grid.z, df["z"], df["v"])

    init = ProgVarsMYNN(
        u=jnp.array(u),
        v=jnp.array(v),
        th=jnp.array(th),
        qke=0.01 * jnp.ones_like(th),  # small initial turbulence
        qv=jnp.zeros_like(th),  # No initial moisture
    )

    ## Forcing
    # t_s = 0 corresponds to 00 local time
    w_thl_fn = lambda t_s: 2.16e-1 * jnp.cos(((t_s / 3600) - 13) / 11 * jnp.pi)  # K m/s
    w_qw_fn = lambda t_s: 2.29e-5 * jnp.cos(((t_s / 3600) - 13) / 11 * jnp.pi)  #  m/s
    dthl_dz_top = 0.0075  # K/m

    u_g = jnp.where(
        grid.z < 1000,
        -5.5 + 2.9e-3 * grid.z,  # linear decrease from -5.5m/s to -2.6m/s at 1000m
        -2.6 + 1.4e-3 * (grid.z - 1000),  # linear decrease from -2.6m/s to -1.2m/s at 2000m
    )
    v_g = jnp.zeros(grid.Nz)

    forcing = Forcing(
        u_geo=lambda t_s: u_g,
        v_geo=lambda t_s: v_g,
        f_c=convert.get_fc(lat_deg=np.abs(-34.5)),  # 34.5°S latitude
        w_th_s=w_thl_fn,
        w_qv_s=w_qw_fn,
        dth_dz_top=dthl_dz_top,
    )

    ## Simulation
    sim = Simulation(
        name="Wangara_Day33",
        grid=grid,
        init=init,
        forcing=forcing,
        th_ref=277.0,  # Pot. temp. close to surface from soundings
        mo_settings=MOSettings(z0m=0.1, z0h=0.1),  # todo: check if agrees with paper
        t_start_s=9 * 3600,
        t_end_s=16 * 3600,
    )
    return sim


if __name__ == "__main__":
    sim = get_wangara_33()
    cfg = load_namelist("namelist_cn.yaml")
    model = init_model(sim, cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(
        out,
        sim,
        time=pd.date_range(
            "1967-08-16T09:00",
            freq=f"{cfg.dt_s_out:.0f}s",
            periods=out.n_steps,
        ),
    )
    ds.to_netcdf("wangara_day33.nc")
