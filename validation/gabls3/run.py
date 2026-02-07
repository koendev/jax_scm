import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import pandas as pd

from scm import consts
from scm import convert
from scm.forcing.interp import get_ts_interp_fn
from scm.interfaces import Simulation, TransientForcing
from scm.mynn.interfaces import ProgVarsMYNN
from scm.mynn.model import init_model
from scm.grid import StaggeredGrid
from scm.time_stepping import simulate_adaptive_dt
from scm.mo import MOSettings, BusingerDyerAltSimFuncs

from scm.io.local import make_dataset


def get_gabls3(Nz: int = 64, plot: bool = False, random_seed: int = 0) -> Simulation[ProgVarsMYNN]:
    """Get a GABLS3 simulation setup.

    References
    ----------
    Bosveld, Fred C., et al. “The Third GABLS Intercomparison Case for Evaluation Studies of Boundary-Layer Models.
    Part A: Case Selection and Set-Up.” Boundary-Layer Meteorology, vol. 152, no. 2, Aug. 2014, pp. 133–56.
    Springer Link, https://doi.org/10.1007/s10546-014-9917-3.

    """

    # MO settings; no other soil or vegetation parameters implemented
    mo_settings = MOSettings(
        z0m=0.15,
        z0h=0.0015,
        sim_funcs=BusingerDyerAltSimFuncs(),
    )

    ## Grid
    grid = StaggeredGrid(H=4000, Nz=Nz)  # expecting BLH of 2000m

    ## Forcing
    # Geostrophic wind
    df_uv_geo_0 = pd.read_csv("ic_bc/bc_uvgeo_sfc.csv")
    time_h = jnp.array(df_uv_geo_0["t"])  # time in hours
    ug_0 = jnp.array(df_uv_geo_0["ug_0"])  # surface geostrophic wind in m/s
    vg_0 = jnp.array(df_uv_geo_0["vg_0"])  # surface geostrophic wind in m/s

    ug = ug_0[:, None] + (-2.0 - ug_0[:, None]) * (grid.z / 2000)  # m/s, linear decrease to -2 m/s at 2000m
    ug = jnp.where(grid.z > 2000, -2.0, ug)  # m/s, constant above 2000m
    vg = vg_0[:, None] + (2.0 - vg_0[:, None]) * (grid.z / 2000)  # m/s, linear decrease to 2 m/s at 2000m
    vg = jnp.where(grid.z > 2000, 2.0, vg)  # m/s, constant above 2000m

    ug_fn = get_ts_interp_fn(time_s=time_h * 3600, data=ug)
    vg_fn = get_ts_interp_fn(time_s=time_h * 3600, data=vg)

    # Sensible heat flux
    df_shfx = pd.read_csv("ic_bc/bc_shf.csv")
    time_h = jnp.array(df_shfx["t"])  # time in hours
    w_th_s = jnp.array(df_shfx["shf"]) / (consts.cp * consts.rho_0)  # W/m^2 to K m / s
    w_th_s_fn = get_ts_interp_fn(time_s=time_h * 3600, data=w_th_s)

    # Latent heat flux
    df_lhfx = pd.read_csv("ic_bc/bc_lhf.csv")
    time_h = jnp.array(df_lhfx["t"])  # time in hours
    w_qv_s = jnp.array(df_lhfx["lhf"]) / (consts.L_v * consts.rho_0)  # W/m^2 to (kg/kg) m / s
    w_qv_s_fn = get_ts_interp_fn(time_s=time_h * 3600, data=w_qv_s)

    # Gather all forcings
    forcing = TransientForcing(
        u_geo=ug_fn,
        v_geo=vg_fn,
        f_c=convert.get_fc(lat_deg=51.9711),  # Cabauw
        w_qv_s=w_qv_s_fn,
        w_th_s=w_th_s_fn,
    )

    # Plot forcings
    if plot:
        t = jnp.linspace(0, 24 * 3600, 50)  # 0 to 24 hours

        fig, (ax_ug, ax_vg, ax_shfx) = plt.subplots(
            ncols=3,
            figsize=(12, 4),
            layout="constrained",
            width_ratios=[1, 1, 2],
        )
        ax_lhfx = ax_shfx.twinx()

        i = ax_ug.imshow(
            jax.vmap(ug_fn)(t).T,  # (time, z)
            extent=(t[0], t[-1], grid.z[0], grid.z[-1]),
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(i, ax=ax_ug)

        i = ax_vg.imshow(
            jax.vmap(vg_fn)(t).T,  # (time, z)
            extent=(t[0], t[-1], grid.z[0], grid.z[-1]),
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(i, ax=ax_vg)

        ax_shfx.plot(t, w_th_s_fn(t), label="w_th_s (K m/s)", color="C0")
        ax_lhfx.plot(t, w_qv_s_fn(t), label="w_qv_s (kg/kg m/s)", color="C1")
        ax_shfx.set_xlabel("Time (s)")

        fig.show()

    ## Initial conditions
    # Initial wind profile
    df_uv_init = pd.read_csv("ic_bc/ic_u_v.csv")
    u = jnp.copy(jnp.interp(grid.z, jnp.array(df_uv_init["z"]), jnp.array(df_uv_init["U"])))
    v = jnp.copy(jnp.interp(grid.z, jnp.array(df_uv_init["z"]), jnp.array(df_uv_init["V"])))

    # Initial air temperature
    df_tc_qv_init = pd.read_csv("ic_bc/ic_tc_q.csv")
    z = jnp.array(df_tc_qv_init["z"])
    tk = jnp.array(df_tc_qv_init["TC"] + 273.15)  # convert to K
    tk = jnp.interp(grid.z, z, tk)

    # Initial specific humidity
    qv = jnp.array(df_tc_qv_init["q"] / 1000)  # kg/kg
    qv = jnp.interp(grid.z, z, qv)

    tkv = convert.th_to_thv(th=tk, qv=qv)  # virtual temperature
    p_hPa = convert.get_p_hypsometric(z=grid.z, tkv=tkv, p_0_hPa=1024.4)
    th = convert.tk_to_th(tk=tk, p_hPa=p_hPa)  # potential temperature

    # Initial TKE
    tke = jnp.zeros(grid.Nz)
    tke = jnp.where(grid.z < 1000, 0.4 * (1 - grid.z / 1000) ** 3, tke)  # m^2 s^-2

    init = ProgVarsMYNN(u=u, v=v, th=th, qke=2 * tke, qv=qv)

    if plot:
        # Initial conditions
        fig, axarr = plt.subplots(ncols=3, figsize=(8, 2), sharey="row", layout="constrained")
        axarr[0].plot(u, grid.z, label="u")
        axarr[0].plot(v, grid.z, label="v")
        axarr[0].set_xlabel("Wind (m/s)")
        axarr[0].set_ylabel("Height (m)")
        axarr[0].legend()

        axarr[1].plot(th, grid.z)
        axarr[1].set_xlabel("Potential Temperature (K)")

        ax2 = axarr[1].twiny()
        ax2.plot(qv, grid.z, color="C1")
        ax2.set_xlabel("Specific Humidity (kg/kg)", color="C1")
        ax2.tick_params(axis="x", colors="C1")

        axarr[2].plot(tke, grid.z)
        axarr[2].set_xlabel("TKE (m$^2$/s$^2$)")
        fig.show()

    return Simulation(
        name="GABLS3",
        grid=grid,
        init=init,
        forcing=forcing,
        mo_settings=mo_settings,
        t_start_s=0,
        t_end_s=24 * 60 * 60,
    )


if __name__ == "__main__":
    sim = get_gabls3(100, plot=True)
    model = init_model(sim)
    state_hist, diag_hist, mo_hist, t = simulate_adaptive_dt(
        model=model,
        sim=sim,
        dt_s_init=0.001,
        dt_s_max=1,
        cfl_max=0.05,
        dt_s_out=60 * 5,
    )

    # Save output
    ds = make_dataset(state_hist, diag_hist, mo_hist, time=t / 60 / 60, grid=sim.grid)
    ds.to_netcdf(f"out_{sim.grid.Nz}.nc")
    print("Written to disk.")
